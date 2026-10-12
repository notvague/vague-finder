"""Review changes: real factory wiring, source preservation and run/export gates."""

import ast
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from src.retrieval.context_search_settings import (
    ContextSearchSettings, ENV_DEFAULTS, ROUTER_FIELDS,
    assert_fixed_context_settings, load_context_settings,
)
from experiments.namuwiki import install_context_review_cleanup as installer
from experiments.namuwiki import context_review_cleanup_runner as runner


FACTORY = '''from pathlib import Path

@singleton
def get_search_router():
    # Keep the merged main feature and shutdown/cache behavior.
    return SearchRouter(
        search_service=get_search_service(),
        context_search=ContextQdrantSearch(get_vector_client(), text_embedder=get_text_embedder()),
        context_weight=0.5, # retain this comment
        context_fact_k=100,
        context_sparse_k=100,
        context_named_media_multiplier=2.0,
        default_candidate_k=30,
        path_k=321,
    )

def unrelated():
    return "do not change this merged code"
'''
PROMPT = '''_PROMPT_TEMPLATE = """Other modality rules remain intact. Reference={reference_year}.
Context examples (other fields follow the output schema above):
- "old example from a disclosed query" -> context_clues=[]

Now analyze:
Query: "{query}"
"""
'''
EVALUATOR = '''from typing import Any

def _configure(router: Any) -> None:
    router._context_weight = SETTING.weight
    router._context_named_media_multiplier = SETTING.named_media_multiplier
    router._context_fact_k = SETTING.fact_k
    router._context_sparse_k = SETTING.sparse_k

def registration():
    return dict(analysis_policy="new cache in this experiment only; same analysis in OFF/ON",
                reference_year=2026)

NOTE = "기존 dev/test는 이미 공개된 회귀 평가이며, 새로운 60문항만 새 독립 평가다."
'''
ACTIVATION = '''def rewrite_factory(data):
    content = data
    factories = []
    call = None
    lines = content.splitlines(keepends=True)
    return lines
'''


def make_project(tmp_path):
    project = tmp_path / "project"
    files = {
        "src/backend/api/dependencies.py": FACTORY,
        "src/retrieval/query_analyzer.py": PROMPT,
        "experiments/namuwiki/evaluate_context_fixed.py": EVALUATOR,
        "experiments/namuwiki/apply_validated_context_settings.py": ACTIVATION,
        ".env.example": "GEMINI_API_KEY=\nCONTEXT_WEIGHT=1.0\nCONTEXT_NAMED_MEDIA_MULTIPLIER=1.0\n",
        ".env": "# preserve credentials exactly\nGEMINI_API_KEY=test-private-key\nMONGO_URI=test-private-uri\nCONTEXT_WEIGHT=1.0\n",
        "docs/data_policy.md": "# Existing team policy\n\n| 나무위키 사실 | 검색 연결 전 프로토타입 |\nOther policy stays.\n",
        "src/retrieval/search_router.py": "# merged main router must stay byte-identical\n",
        "src/vector_db/qdrant_backend.py": "# merged main backend must stay byte-identical\n",
        "data/context/labels.csv": "private labels stay byte-identical\n",
        "artifacts/context/songs/fake.json": "{}",
        "docker-compose.yml": "services: {}\n",
        "compose.context-ranking.yml": "services: {}\n",
    }
    for name, content in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return project


def factory_namespace(data: bytes):
    tree = ast.parse(data.decode("utf-8-sig"))
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_search_router")
    factory.decorator_list = []
    # Execute the real patched factory body, with only heavy dependencies replaced.
    space = dict(load_context_settings=load_context_settings,
                 SearchRouter=lambda **kwargs: kwargs,
                 get_search_service=lambda: "service",
                 get_vector_client=lambda: "same-client",
                 get_text_embedder=lambda: "same-embedder",
                 ContextQdrantSearch=lambda client, **kw: (client, kw["text_embedder"]))
    exec(compile(ast.Module(body=[factory], type_ignores=[]), "actual_factory.py", "exec"), space)
    return space


def test_defaults_are_the_previously_measured_preset():
    assert load_context_settings({}) == ContextSearchSettings()
    assert load_context_settings(ENV_DEFAULTS).router_arguments() == {
        "context_weight": .5, "context_fact_k": 100, "context_sparse_k": 100,
        "context_named_media_multiplier": 2., "default_candidate_k": 30,
    }


def test_all_five_environment_variables_reach_the_real_patched_factory(monkeypatch):
    values = dict(CONTEXT_WEIGHT="0.73", CONTEXT_NAMED_MEDIA_MULTIPLIER="1.4",
                  CONTEXT_FACT_K="143", CONTEXT_SPARSE_K="87", SEARCH_DEFAULT_CANDIDATE_K="51")
    for key in ENV_DEFAULTS:
        monkeypatch.setenv(key, values[key])
    actual = factory_namespace(installer.edit_factory(FACTORY.encode()))["get_search_router"]()
    assert {key: actual[key] for key in ROUTER_FIELDS} == dict(
        context_weight=.73, context_named_media_multiplier=1.4,
        context_fact_k=143, context_sparse_k=87, default_candidate_k=51)
    assert actual["path_k"] == 321
    assert actual["context_search"] == ("same-client", "same-embedder")


def test_environment_is_read_each_time_a_new_router_is_created(monkeypatch):
    for key in ENV_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    factory = factory_namespace(installer.edit_factory(FACTORY.encode()))["get_search_router"]
    assert factory()["context_weight"] == .5
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.8")
    assert factory()["context_weight"] == .8


@pytest.mark.parametrize("key,bad", [
    (key, bad) for key in ENV_DEFAULTS for bad in ("", "garbage", "nan", "inf", "-1")
] + [
    ("CONTEXT_NAMED_MEDIA_MULTIPLIER", ".9"),
    ("CONTEXT_WEIGHT", "-inf"), ("CONTEXT_WEIGHT", "1e999"),
    ("CONTEXT_FACT_K", "0"), ("CONTEXT_SPARSE_K", "0"), ("SEARCH_DEFAULT_CANDIDATE_K", "0"),
    ("CONTEXT_FACT_K", "1.5"), ("CONTEXT_SPARSE_K", "100.0"),
    ("SEARCH_DEFAULT_CANDIDATE_K", "101"), ("SEARCH_DEFAULT_CANDIDATE_K", "true"),
])
def test_invalid_explicit_environment_fails_with_the_key_name(key, bad):
    with pytest.raises(ValueError, match=key):
        load_context_settings({key: bad})


def test_zero_weight_and_candidate_bounds_are_supported():
    assert load_context_settings({"CONTEXT_WEIGHT": " 0 ", "SEARCH_DEFAULT_CANDIDATE_K": "1"}).weight == 0
    assert load_context_settings({"SEARCH_DEFAULT_CANDIDATE_K": "100"}).candidate_k == 100


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_factory_preserves_merged_features_comments_encoding_and_other_functions(bom, newline):
    source = bom + ("# 한글 주석\n" + FACTORY).replace("\n", newline).encode()
    patched = installer.edit_factory(source)
    assert patched.startswith(bom + ("# 한글 주석" + newline).encode())
    assert b"path_k=321" in patched and b"# retain this comment" in patched
    assert 'return "do not change this merged code"' in patched.decode("utf-8-sig")
    assert installer.edit_factory(patched) == patched
    if newline == "\r\n":
        assert b"\n" not in patched.replace(b"\r\n", b"")


@pytest.mark.parametrize("source", [
    FACTORY.replace("context_search=ContextQdrantSearch(get_vector_client(), text_embedder=get_text_embedder())", "context_search=None"),
    FACTORY.replace("path_k=321,", "**settings,"),
    FACTORY.replace("context_fact_k=100,", ""),
    FACTORY.replace("return SearchRouter(", "return Wrapper(SearchRouter("),
    FACTORY + FACTORY,
])
def test_unknown_factory_shapes_are_refused(source):
    with pytest.raises((ValueError, SyntaxError)):
        installer.edit_factory(source.encode())


def test_prompt_uses_only_fictional_examples_and_formats_user_query_verbatim():
    patched = installer.edit_prompt(PROMPT.encode())
    tree = ast.parse(patched.decode())
    template = ast.literal_eval(tree.body[0].value)
    examples = template.split(installer.PROMPT_START)[1].split("Now analyze:")[0]
    assert "fictional" in examples and "가상 작품 A" in examples
    for forbidden in ("q203", "nw001", "짱구", "나미리"):
        assert forbidden not in examples
    query = '알 수 없는 작품의 장면에 나온 노래 {literal}'
    rendered = template.format(query=query, reference_year=2026)
    assert f'Query: "{query}"' in rendered
    assert "Other modality rules remain intact" in rendered
    assert installer.edit_prompt(patched) == patched


def test_real_project_prompt_example_does_not_name_disclosed_targets():
    path = Path("src/retrieval/query_analyzer.py")
    if not path.is_file():
        pytest.skip("Run from project root to check actual prompt")
    patched = installer.edit_prompt(path.read_bytes())
    tree = ast.parse(patched.decode("utf-8-sig"))
    literal = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_PROMPT_TEMPLATE" for t in n.targets))
    examples = literal.split(installer.PROMPT_START)[1].split("Now analyze:")[0]
    assert "짱구" not in examples and "나미리" not in examples
    assert "가상 작품 A" in literal.format(query="새 입력", reference_year=2026,
                                        reference_year_minus_one=2025, recent_start_year=2018)


def test_evaluator_checks_settings_and_never_mutates_router():
    patched = installer.edit_evaluator(EVALUATOR.encode())
    tree = ast.parse(patched.decode())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_configure")
    space = dict(Any=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "real_configure.py", "exec"), space)
    router = SimpleNamespace(**{"_" + k: v for k, v in ContextSearchSettings().router_arguments().items()})
    assert space["_configure"](router) is None
    router._context_weight = 1.0
    before = dict(vars(router))
    with pytest.raises(ValueError, match="Effective Context settings"):
        space["_configure"](router)
    assert vars(router) == before
    assert "not a new blind sample" in patched.decode()
    assert "새로운 60문항만 새 독립 평가다" not in patched.decode()
    assert installer.edit_evaluator(patched) == patched


@pytest.mark.parametrize("arg", list(ROUTER_FIELDS))
def test_all_effective_factory_values_are_compared(arg):
    values = {"_" + k: v for k, v in ContextSearchSettings().router_arguments().items()}
    values["_" + arg] += 1
    with pytest.raises(ValueError):
        assert_fixed_context_settings(SimpleNamespace(**values))


def test_invalid_environment_fails_before_any_client_or_model_is_created(monkeypatch):
    monkeypatch.setenv("CONTEXT_WEIGHT", "invalid")
    space = factory_namespace(installer.edit_factory(FACTORY.encode()))
    def unexpected():
        raise AssertionError("No client/model should be created before settings validation")
    space["get_search_service"] = unexpected
    space["get_vector_client"] = unexpected
    with pytest.raises(ValueError, match="CONTEXT_WEIGHT"):
        space["get_search_router"]()


def test_legacy_activation_does_not_replace_environment_wiring_with_literals():
    from experiments.namuwiki.apply_validated_context_settings import rewrite_factory
    wired = installer.edit_factory(FACTORY.encode())
    assert rewrite_factory(wired) == wired


def test_pipeline_fixtures_gain_effective_defaults_without_bypassing_assertions():
    source = '''from types import SimpleNamespace

def test_fake(monkeypatch):
    untouched = SimpleNamespace()
    monkeypatch.setattr(fixed.old, "_corpus", lambda *_: ({"same": True}, SimpleNamespace(), frozenset({"a"})))
'''
    patched = installer.edit_pipeline_test_stubs(source.encode())
    assert b"untouched = SimpleNamespace()" in patched
    assert b"_review_fixed_router_stub(), frozenset" in patched
    assert b"assert_fixed_context_settings" not in patched  # No patch/mocking of the guard.
    space = {}
    exec(compile(patched.decode(), "updated_pipeline_fixture.py", "exec"), space)
    assert assert_fixed_context_settings(space["_review_fixed_router_stub"]()) == ContextSearchSettings().router_arguments()
    assert installer.edit_pipeline_test_stubs(patched) == patched


def test_example_comments_describe_effective_defaults_and_restart():
    data = b"# Other policy\n# Step 10 Context ranking. Keep old defaults.\n# old description\nCONTEXT_WEIGHT=1.0\n"
    patched = installer.edit_env_example(data)
    assert b"CONTEXT_WEIGHT=0.5" in patched and b"Restart" in patched
    assert b"Keep old defaults" not in patched and b"# Other policy" in patched
    assert installer.edit_env_example(patched) == patched


def test_environment_update_preserves_shared_credentials_and_deduplicates_only_context_keys():
    source = b"\xef\xbb\xbf# private settings\r\nGEMINI_API_KEY=keep-key\r\nMONGO_URI=keep-uri\r\nCONTEXT_WEIGHT=1\r\nCONTEXT_WEIGHT=2\r\nOTHER=keep\r\n"
    patched = installer.edit_env(source)
    assert patched.count(b"CONTEXT_WEIGHT=") == 1
    assert b"GEMINI_API_KEY=keep-key\r\nMONGO_URI=keep-uri\r\n" in patched
    assert b"OTHER=keep\r\n" in patched and patched.startswith(b"\xef\xbb\xbf")
    assert installer.edit_env(patched) == patched


def test_atomic_install_is_idempotent_and_preserves_database_labels_and_merge(tmp_path):
    project = make_project(tmp_path)
    output = project / "artifacts/new_review"
    names = ("src/retrieval/search_router.py", "src/vector_db/qdrant_backend.py",
             "data/context/labels.csv", "artifacts/context/songs/fake.json")
    before = {name: (project / name).read_bytes() for name in names}
    result = installer.install(project, output, apply_env=True)
    assert result["status"] == "applied" and ".env" in result["changed_files"]
    assert result["source_hashes"].get(".env") is None
    assert all((project / name).read_bytes() == data for name, data in before.items())
    assert "API 검색 후보 결합에 연결됨" in (project / "docs/data_policy.md").read_text()
    assert "Other policy stays." in (project / "docs/data_policy.md").read_text()
    assert "test-private-key" in (project / ".env").read_text()
    assert installer.install(project, output, apply_env=True)["changed_files"] == []


def test_preflight_failure_writes_no_partial_source_edits(tmp_path):
    project = make_project(tmp_path)
    (project / "experiments/namuwiki/evaluate_context_fixed.py").write_text("# unknown implementation\n")
    before = {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        installer.install(project, project / "artifacts/new_review", apply_env=True)
    assert all(path.read_bytes() == data for path, data in before.items())


def test_mid_install_failure_rolls_back_already_written_sources(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    before = (project / "src/backend/api/dependencies.py").read_bytes()
    real_write = installer.atomic_write
    calls = 0

    def failing_write(path, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("simulated file lock")
        real_write(path, data)

    monkeypatch.setattr(installer, "atomic_write", failing_write)
    with pytest.raises(PermissionError):
        installer.install(project, project / "artifacts/new_review", apply_env=True)
    assert (project / "src/backend/api/dependencies.py").read_bytes() == before


def test_feedback_is_unique_verified_and_never_sweeps_private_backups(tmp_path):
    root = tmp_path / "artifacts/run"
    (root / "installation/backups").mkdir(parents=True)
    (root / "installation/backups/.env").write_text("SECRET=must-not-export")
    (root / ".env").write_text("SECRET=must-not-export")
    (root / "runner_status.json").write_text('{"status":"stopped"}')
    (root / "console_run.log").write_text("actual console output")
    one, two = runner.pack_feedback(root), runner.pack_feedback(root)
    assert one != two and one.is_file() and two.is_file()
    with ZipFile(one) as archive:
        assert set(archive.namelist()) == {"runner_status.json", "console_run.log", "export_manifest.json"}
        assert all(b"must-not-export" not in archive.read(name) for name in archive.namelist())


def test_normal_compose_keeps_existing_profile_and_appends_ranking_last(tmp_path):
    project = make_project(tmp_path)
    (project / "compose.context-test.yml").write_text("existing shared profile")
    command = runner.compose_base(project)
    assert command == ["docker", "compose", "-f", "docker-compose.yml", "-f", "compose.context-test.yml", "-f", "compose.context-ranking.yml"]
    root = project / "artifacts/run"
    overlay = runner.make_overlay(project, root)
    text = overlay.read_text()
    assert "RERANKER_ENABLED" in text and "SEARCH_TIMING_LOG" in text
    assert all(key not in text for key in ENV_DEFAULTS)


@pytest.mark.parametrize("failure", ["none", "tests", "fresh_off_on", "evidence", "api", "restore"])
def test_runner_stops_on_failures_restores_backend_and_preserves_reports(tmp_path, monkeypatch, failure):
    project = make_project(tmp_path)
    root = project / "artifacts/new_run"
    for name in runner.INPUTS + ("experiments/namuwiki/verify_context_api_ready.py",):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic required input")
    monkeypatch.setattr(runner.shutil, "which", lambda _: "available")
    commands = []

    def captured(command, log, *, cwd):
        commands.append(command)
        with log.open("a") as stream:
            stream.write("simulated native command\n")
        if "pytest" in command and failure == "tests":
            return 1
        if "experiments.namuwiki.evaluate_context_fixed" in command:
            (root / "fresh").mkdir(exist_ok=True)
            (root / "fresh/validation_report.json").write_text('{"status":"' +
                ("evidence_review_required" if failure == "evidence" else "passed_observed_tests") + '"}')
            return 2 if failure == "fresh_off_on" else 0
        if "experiments.namuwiki.verify_context_api_ready" in command:
            (root / "activation").mkdir(exist_ok=True)
            (root / "activation/api_smoke_report.json").write_text('{"status":"passed_api_smoke"}')
            return 2 if failure == "api" else 0
        if failure == "restore" and command[-5:] == ["up", "-d", "--no-deps", "--force-recreate", "backend"] and "exec" in [c for cmd in commands[:-1] for c in cmd]:
            return 1
        return 0

    monkeypatch.setattr(runner, "capture", captured)
    result = runner.run(project, root)
    assert (result == 0) == (failure == "none")
    assert commands[-1][-5:] == ["up", "-d", "--no-deps", "--force-recreate", "backend"]
    assert "compose.evaluation.json" not in " ".join(commands[-1])
    assert list(root.parent.glob("new_run_feedback_*.zip"))
    assert (root / "runner_status.json").is_file()
    if failure == "evidence":
        assert not any("experiments.namuwiki.apply_validated_context_settings" in cmd for cmd in commands)


@pytest.mark.parametrize("path", ["../escape", "/tmp/artifacts/a", "artifacts/../escape", "data/run"])
def test_output_path_must_stay_in_artifacts(tmp_path, path):
    with pytest.raises(ValueError):
        runner.output_path(tmp_path, path)


def test_existing_output_requires_explicit_resume(tmp_path):
    project = make_project(tmp_path)
    root = project / "artifacts/old_run"
    root.mkdir()
    (root / "registration.json").write_text("old result preserved")
    with pytest.raises(ValueError, match="already has files"):
        runner.run(project, root)
    assert (root / "registration.json").read_text() == "old result preserved"


def test_export_failure_cannot_leave_a_passing_runner_status(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    root = project / "artifacts/test_only"
    monkeypatch.setattr(runner.shutil, "which", lambda _: "available")
    monkeypatch.setattr(runner, "capture", lambda *a, **kw: 0)
    def locked(_):
        raise PermissionError("simulated export lock")
    monkeypatch.setattr(runner, "pack_feedback", locked)
    assert runner.run(project, root, mode="Tests") == 2
    status = runner.read_json(root / "runner_status.json")
    assert status["status"] == "feedback_export_failed" and status["exit_code"] == 2
    assert status["backend_restore_exit"] == 0
    assert not status["performance_verified"]


def test_specificity_runner_installs_before_build_and_exports_its_report(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    for name in ("context_query.py", "context_clue_specificity.py"):
        path = project / "src/retrieval" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((Path("src/retrieval") / name).read_bytes())
    root = project / "artifacts/media_run"
    commands = []
    monkeypatch.setattr(runner.shutil, "which", lambda _: "available")
    def capture(cmd, *args, **kwargs):
        commands.append(cmd)
        if "build" in cmd:
            assert (root / "specificity/install_report.json").is_file()
        return 0
    monkeypatch.setattr(runner, "capture", capture)
    assert runner.run(project, root, mode="Tests", media_specificity_fix=True) == 0
    state = runner.read_json(root / "runner_status.json")
    assert state["media_specificity_fix"] and state["status"] == "passed_code_tests_only"
    with ZipFile(next(root.parent.glob("media_run_feedback_*.zip"))) as archive:
        assert "specificity/install_report.json" in archive.namelist()
    assert "tests/context" in next(cmd for cmd in commands if "pytest" in cmd)


def test_unknown_specificity_source_stops_before_other_installer_writes(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    path = project / "src/retrieval/context_query.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def changed_parser():\n    return []\n")
    env = (project / ".env").read_bytes()
    prompt = (project / "src/retrieval/query_analyzer.py").read_bytes()
    monkeypatch.setattr(runner.shutil, "which", lambda _: "available")
    monkeypatch.setattr(runner, "capture", lambda *a, **kw: 0)
    root = project / "artifacts/invalid_source"
    assert runner.run(project, root, mode="Tests", media_specificity_fix=True) == 2
    state = runner.read_json(root / "runner_status.json")
    assert state["stage"] == "specificity_preflight" and state["backend_restore_exit"] == 0
    assert not state["performance_verified"] and not state["api_verified"]
    assert (project / ".env").read_bytes() == env
    assert (project / "src/retrieval/query_analyzer.py").read_bytes() == prompt
