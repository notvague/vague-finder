"""The handoff preserves changed merged work and rolls back incomplete installs."""

import json

import pytest

from experiments.namuwiki import install_context_media_scope as scope
from experiments.namuwiki import context_review_cleanup_runner as runner
from tests.context.test_context_review_cleanup import make_project


def fixture_project(tmp_path, *, windows=False):
    project = tmp_path / "project"
    files = ("src/retrieval/context_query.py", "src/retrieval/context_evidence.py")
    entries, originals = [], {}
    for i, name in enumerate(files):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        before = f"# unrelated local comment\nVALUE = {i}\n".encode()
        if windows:
            before = b"\xef\xbb\xbf" + before.replace(b"\n", b"\r\n")
        path.write_bytes(before)
        originals[name] = before
        after = f"# unrelated local comment\nVALUE = {i + 10}\n".encode()
        payload = project / scope.PAYLOAD / f"{i}.py"
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(after)
        entries.append(dict(target=name, payload=payload.name, before=scope.sha(before), after=scope.sha(after)))
    required = "src/retrieval/context_media_match.py"
    (project / required).write_text("NAME = 'new helper'\n", encoding="utf-8")
    manifest = dict(schema=scope.SCHEMA, files=entries,
                    required_new_files={required: scope.sha((project / required).read_bytes())})
    write_manifest(project, manifest)
    return project, manifest, originals


def write_manifest(project, manifest):
    (project / scope.PAYLOAD / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("windows", (False, True))
def test_apply_backup_idempotence_and_windows_encoding(tmp_path, windows):
    project, manifest, originals = fixture_project(tmp_path, windows=windows)
    output = project / "artifacts/check"
    assert len(scope.preflight(project)) == 2
    result = scope.install(project, output)
    assert result["changed_files"] == list(originals)
    assert (output / "scope/install_report.json").is_file()
    for entry in manifest["files"]:
        name = entry["target"]
        assert (output / "scope/backups" / name).read_bytes() == originals[name]
        data = (project / name).read_bytes()
        assert scope.sha(data) == entry["after"]
        assert data.startswith(b"\xef\xbb\xbf") is windows
        assert (b"\r\n" in data) is windows
    assert scope.preflight(project) == []
    assert scope.install(project, output)["changed_files"] == []
    assert (output / "scope/backups" / next(iter(originals))).read_bytes() == next(iter(originals.values()))


@pytest.mark.parametrize("problem", ("unknown_work", "tampered_payload", "missing_helper", "invalid_python"))
def test_all_targets_checked_before_any_write(tmp_path, problem):
    project, manifest, originals = fixture_project(tmp_path)
    second = manifest["files"][1]
    if problem == "unknown_work":
        (project / second["target"]).write_bytes(b"# newer merged work\n")
    elif problem == "tampered_payload":
        (project / scope.PAYLOAD / second["payload"]).write_bytes(b"VALUE = 900\n")
    elif problem == "missing_helper":
        (project / next(iter(manifest["required_new_files"]))).unlink()
    else:
        data = b"def invalid(:\n"
        (project / scope.PAYLOAD / second["payload"]).write_bytes(data)
        second["after"] = scope.sha(data)
        write_manifest(project, manifest)
    snapshot = {name: (project / name).read_bytes() for name in originals}
    with pytest.raises((ValueError, FileNotFoundError, SyntaxError)):
        scope.install(project, project / "artifacts/check")
    assert all((project / name).read_bytes() == data for name, data in snapshot.items())
    assert not (project / "artifacts/check/scope/backups").exists()


@pytest.mark.parametrize("failure", ("second_source", "report"))
def test_atomic_write_failure_rolls_back_completed_source_writes(tmp_path, monkeypatch, failure):
    project, _, originals = fixture_project(tmp_path)
    real = scope.atomic_write
    def fail(path, data):
        if ((failure == "report" and path.name == "install_report.json")
                or (failure == "second_source" and path.name == "context_evidence.py")):
            raise OSError("injected write failure")
        real(path, data)
    monkeypatch.setattr(scope, "atomic_write", fail)
    with pytest.raises(OSError, match="injected"):
        scope.install(project, project / "artifacts/check")
    assert all((project / name).read_bytes() == data for name, data in originals.items())


def test_concurrent_change_is_preserved_and_earlier_write_rolled_back(tmp_path, monkeypatch):
    project, _, originals = fixture_project(tmp_path)
    names = list(originals)
    real = scope.atomic_write
    def update_other(path, data):
        real(path, data)
        if path == project / names[0] and data != originals[names[0]]:
            (project / names[1]).write_bytes(b"# concurrent user work\n")
    monkeypatch.setattr(scope, "atomic_write", update_other)
    with pytest.raises(ValueError, match="changed during installation"):
        scope.install(project, project / "artifacts/check")
    assert (project / names[0]).read_bytes() == originals[names[0]]
    assert (project / names[1]).read_bytes() == b"# concurrent user work\n"


@pytest.mark.parametrize("name", ("../source.py", "/source.py", "C:/source.py", "a\\b.py", "a/../b.py", "a//b.py"))
def test_scope_paths_cannot_escape_the_registered_payload(tmp_path, name):
    project, manifest, _ = fixture_project(tmp_path)
    manifest["files"][0]["payload"] = name
    write_manifest(project, manifest)
    with pytest.raises(ValueError, match="relative scope path"):
        scope.preflight(project)


def test_merged_router_and_backend_are_not_allowed_payload_targets(tmp_path):
    project, manifest, _ = fixture_project(tmp_path)
    manifest["files"][0]["target"] = "src/retrieval/search_router.py"
    write_manifest(project, manifest)
    with pytest.raises(ValueError, match="scope target"):
        scope.preflight(project)


def test_backups_and_reports_must_be_inside_artifacts(tmp_path):
    project, _, _ = fixture_project(tmp_path)
    for output in (tmp_path / "outside", project, project / "data/backups"):
        with pytest.raises(ValueError, match="inside artifacts"):
            scope.install(project, output)


@pytest.mark.parametrize("unknown", (False, True))
def test_runner_checks_scope_before_other_edits_and_installs_before_build(tmp_path, monkeypatch, unknown):
    project = make_project(tmp_path)
    name = "src/retrieval/context_evidence.py"
    before, after = b"VALUE = 1\n", b"VALUE = 2\n"
    source = project / name
    source.write_bytes(b"# newer user edit\n" if unknown else before)
    payload = project / scope.PAYLOAD / "replacement.py"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(after)
    write_manifest(project, dict(schema=scope.SCHEMA,
        files=[dict(target=name, payload=payload.name, before=scope.sha(before), after=scope.sha(after))],
        required_new_files={}))
    preserved = {n: (project / n).read_bytes() for n in (
        ".env", "src/retrieval/search_router.py", "src/vector_db/qdrant_backend.py")}
    prompt_before = (project / "src/retrieval/query_analyzer.py").read_bytes()
    commands = []
    root = project / "artifacts/scope_run"
    monkeypatch.setattr(runner.shutil, "which", lambda _: "available")
    def captured(cmd, *args, **kwargs):
        commands.append(cmd)
        if "build" in cmd:
            assert (root / "scope/install_report.json").is_file()
            assert source.read_bytes() == after
        return 0
    monkeypatch.setattr(runner, "capture", captured)
    exit_code = runner.run(project, root, mode="Tests", update_env=False, media_scope_fix=True)
    state = runner.read_json(root / "runner_status.json")
    assert exit_code == (2 if unknown else 0)
    assert state["media_scope_fix"] and state["backend_restore_exit"] == 0
    assert not state["performance_verified"] and not state["api_verified"]
    assert all((project / n).read_bytes() == data for n, data in preserved.items())
    if unknown:
        assert state["stage"] == "scope_preflight"
        assert (project / "src/retrieval/query_analyzer.py").read_bytes() == prompt_before
        assert not any("build" in cmd for cmd in commands)
    else:
        from zipfile import ZipFile
        with ZipFile(next(root.parent.glob("scope_run_feedback_*.zip"))) as archive:
            assert "scope/install_report.json" in archive.namelist()
