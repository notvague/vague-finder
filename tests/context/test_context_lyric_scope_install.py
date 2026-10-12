"""Source-checked lyric fix: preservation, rollback, and runner integration."""

import json
from zipfile import ZipFile

import pytest

from experiments.namuwiki import install_context_lyric_scope as scope
from experiments.namuwiki import context_review_cleanup_runner as runner
from tests.context.test_context_review_cleanup import make_project


def fixture_project(tmp_path, *, windows=False):
    project = tmp_path / "project"
    source = project / scope.TARGET
    source.parent.mkdir(parents=True, exist_ok=True)
    before = b"# unrelated comment\nVALUE = 1\n"
    if windows:
        before = b"\xef\xbb\xbf" + before.replace(b"\n", b"\r\n")
    source.write_bytes(before)
    helper = project / "experiments/namuwiki/helper.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_bytes(b"VALUE = 'existing helper'\n")
    manifest = dict(schema="context_lyric_scope_payload_v1", target=scope.TARGET,
                    before=scope.digest(before), after=scope.digest(b"# unrelated comment\nVALUE = 2\n"),
                    required_files={helper.relative_to(project).as_posix(): scope.digest(helper.read_bytes())})
    payload = project / scope.PAYLOAD
    payload.mkdir(parents=True)
    (payload / "modality_queries.py.payload").write_bytes(b"# unrelated comment\nVALUE = 2\n")
    write_manifest(project, manifest)
    return project, manifest, before


def write_manifest(project, manifest):
    (project / scope.PAYLOAD / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("windows", (False, True))
def test_install_preserves_encoding_backup_and_repeat_is_idempotent(tmp_path, windows):
    project, manifest, before = fixture_project(tmp_path, windows=windows)
    output = project / "artifacts/run"
    result = scope.install(project, output)
    source = project / scope.TARGET
    assert result["changed_files"] == [scope.TARGET]
    assert (output / "lyric_scope/backups" / scope.TARGET).read_bytes() == before
    assert scope.digest(source.read_bytes()) == manifest["after"]
    assert source.read_bytes().startswith(b"\xef\xbb\xbf") is windows
    assert (b"\r\n" in source.read_bytes()) is windows
    assert scope.install(project, output)["changed_files"] == []
    assert (output / "lyric_scope/backups" / scope.TARGET).read_bytes() == before


@pytest.mark.parametrize("problem", (
    "unknown_source", "tampered_payload", "missing_helper", "changed_helper", "bad_schema", "invalid_python",
))
def test_preflight_failure_does_not_write_source_or_backup(tmp_path, problem):
    project, manifest, _ = fixture_project(tmp_path)
    source = project / scope.TARGET
    helper = project / next(iter(manifest["required_files"]))
    payload = project / scope.PAYLOAD / "modality_queries.py.payload"
    if problem == "unknown_source":
        source.write_bytes(b"# newer user changes\n")
    elif problem == "tampered_payload":
        payload.write_bytes(b"VALUE = 99\n")
    elif problem == "missing_helper":
        helper.unlink()
    elif problem == "changed_helper":
        helper.write_bytes(b"VALUE = 'newer'\n")
    elif problem == "bad_schema":
        manifest["schema"] = "different"
        write_manifest(project, manifest)
    else:
        payload.write_bytes(b"def invalid(:\n")
        manifest["after"] = scope.digest(payload.read_bytes())
        write_manifest(project, manifest)
    snapshot = source.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError, SyntaxError)):
        scope.install(project, project / "artifacts/run")
    assert source.read_bytes() == snapshot
    assert not (project / "artifacts/run/lyric_scope/backups").exists()


@pytest.mark.parametrize("name", ("../file.py", "/file.py", "C:/file.py", "a\\b.py", "src/retrieval/search_router.py"))
def test_manifest_cannot_use_outside_or_unrelated_required_files(tmp_path, name):
    project, manifest, before = fixture_project(tmp_path)
    manifest["required_files"] = {name: "not-allowed"}
    write_manifest(project, manifest)
    with pytest.raises(ValueError, match="Invalid lyric scope required file"):
        scope.preflight(project)
    assert (project / scope.TARGET).read_bytes() == before


@pytest.mark.parametrize("failure", ("source", "report"))
def test_write_failure_rolls_back_source(tmp_path, monkeypatch, failure):
    project, _, before = fixture_project(tmp_path)
    real = scope.atomic_write
    def failing_write(path, data):
        if ((failure == "source" and path == project / scope.TARGET and data != before)
                or (failure == "report" and path.name == "install_report.json")):
            raise OSError("injected failure")
        return real(path, data)
    monkeypatch.setattr(scope, "atomic_write", failing_write)
    with pytest.raises(OSError, match="injected failure"):
        scope.install(project, project / "artifacts/run")
    assert (project / scope.TARGET).read_bytes() == before


def test_output_cannot_be_outside_artifacts(tmp_path):
    project, _, before = fixture_project(tmp_path)
    for output in (project, tmp_path / "outside", project / "data/report"):
        with pytest.raises(ValueError, match="inside artifacts"):
            scope.install(project, output)
    assert (project / scope.TARGET).read_bytes() == before


@pytest.mark.parametrize("unknown", (False, True))
def test_runner_checks_source_before_edits_installs_before_build_and_restores_backend(tmp_path, monkeypatch, unknown):
    project = make_project(tmp_path)
    source = project / scope.TARGET
    original = b"VALUE = 1\n"
    source.write_bytes(b"# user update\n" if unknown else original)
    payload = project / scope.PAYLOAD
    payload.mkdir(parents=True)
    (payload / "modality_queries.py.payload").write_bytes(b"VALUE = 2\n")
    write_manifest(project, dict(schema="context_lyric_scope_payload_v1", target=scope.TARGET,
                               before=scope.digest(original), after=scope.digest(b"VALUE = 2\n"), required_files={}))
    preserved = {name: (project / name).read_bytes() for name in (
        ".env", "src/retrieval/search_router.py", "src/vector_db/qdrant_backend.py", "data/context/labels.csv")}
    prompt_before = (project / "src/retrieval/query_analyzer.py").read_bytes()
    root = project / "artifacts/lyric_run"
    commands = []
    monkeypatch.setattr(runner.shutil, "which", lambda _: "available")
    def captured(cmd, *args, **kwargs):
        commands.append(cmd)
        if "build" in cmd:
            assert source.read_bytes() == b"VALUE = 2\n"
            assert (root / "lyric_scope/install_report.json").is_file()
        return 0
    monkeypatch.setattr(runner, "capture", captured)
    assert runner.run(project, root, mode="Tests", update_env=False, lyric_scope_fix=True) == (2 if unknown else 0)
    state = runner.read_json(root / "runner_status.json")
    assert state["lyric_scope_fix"] and state["backend_restore_exit"] == 0
    assert all((project / name).read_bytes() == data for name, data in preserved.items())
    assert not state["performance_verified"] and not state["api_verified"]
    if unknown:
        assert state["stage"] == "lyric_scope_preflight"
        assert (project / "src/retrieval/query_analyzer.py").read_bytes() == prompt_before
        assert not any("build" in command for command in commands)
    else:
        with ZipFile(next(root.parent.glob("lyric_run_feedback_*.zip"))) as archive:
            assert "lyric_scope/install_report.json" in archive.namelist()
            assert not any("backup" in name or name.endswith(".env") for name in archive.namelist())
