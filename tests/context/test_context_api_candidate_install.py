"""An API-only patch must not silently overwrite a different project revision."""

from pathlib import Path
import json

import pytest

from experiments.namuwiki import install_context_api_candidate as installer


def prepared_project(tmp_path):
    project = tmp_path / "project"
    targets = {}
    for name in installer.TARGETS:
        file = project / name
        file.parent.mkdir(parents=True, exist_ok=True)
        before, after = b"# old synthetic verifier\nVALUE=1\n", b"# new synthetic verifier\nVALUE=2\n"
        file.write_bytes(before)
        payload = project / installer.PAYLOAD / (Path(name).name + ".payload")
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(after)
        targets[name] = dict(payload=payload.name, before=installer.digest(before), after=installer.digest(after))
    companion = project / "experiments/namuwiki/synthetic_companion.py"
    companion.write_bytes(b"# synthetic companion\n")
    manifest = dict(schema="context_api_candidate_payload_v1", targets=targets,
                    required_files={str(companion.relative_to(project).as_posix()): installer.digest(companion.read_bytes())})
    (project / installer.PAYLOAD / "manifest.json").write_text(json.dumps(manifest))
    root = project / "artifacts/completed"
    root.mkdir(parents=True)
    return project, root, manifest


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_known_base_and_repeated_install_preserve_encoding_and_back_up_only_experiment_sources(tmp_path, monkeypatch, bom, newline):
    from experiments.namuwiki import resume_context_api_check
    monkeypatch.setattr(resume_context_api_check, "passed_binding", lambda *a: ({}, {}))
    project, root, _ = prepared_project(tmp_path)
    before = {}
    for name in installer.TARGETS:
        file = project / name
        file.write_bytes(bom + file.read_bytes().replace(b"\n", newline))
        before[name] = file.read_bytes()
    installed = installer.install(project, root)
    assert installed["changed_files"] == list(installer.TARGETS)
    for name in installer.TARGETS:
        assert (root / "api_candidate_install/backups" / name).read_bytes() == before[name]
        data = (project / name).read_bytes()
        assert data.startswith(bom) and b"VALUE=2" in data
        if newline == b"\r\n":
            assert b"\n" not in data.replace(b"\r\n", b"")
    assert installer.install(project, root)["changed_files"] == []


@pytest.mark.parametrize("fault", ["unknown_first", "unknown_second", "corrupt_payload", "missing_companion", "changed_companion", "bad_payload_path"])
def test_all_targets_are_checked_before_any_source_is_written(tmp_path, monkeypatch, fault):
    from experiments.namuwiki import resume_context_api_check
    monkeypatch.setattr(resume_context_api_check, "passed_binding", lambda *a: ({}, {}))
    project, root, manifest = prepared_project(tmp_path)
    if fault in {"unknown_first", "unknown_second"}:
        (project / installer.TARGETS[0 if fault == "unknown_first" else 1]).write_bytes(b"# teammate's different revision\n")
    elif fault == "corrupt_payload":
        (project / installer.PAYLOAD / manifest["targets"][installer.TARGETS[1]]["payload"]).write_bytes(b"# corrupt\n")
    elif fault in {"missing_companion", "changed_companion"}:
        file = project / "experiments/namuwiki/synthetic_companion.py"
        if fault == "missing_companion":
            file.unlink()
        else:
            file.write_bytes(b"# changed\n")
    else:
        manifest["targets"][installer.TARGETS[0]]["payload"] = "../outside.py"
        (project / installer.PAYLOAD / "manifest.json").write_text(json.dumps(manifest))
    before = {n: (project / n).read_bytes() for n in installer.TARGETS}
    with pytest.raises((ValueError, FileNotFoundError)):
        installer.install(project, root)
    assert {n: (project / n).read_bytes() for n in installer.TARGETS} == before
    assert not (root / "api_candidate_install/backups").exists()


def test_incomplete_evaluation_cannot_install_or_claim_resume(tmp_path, monkeypatch):
    from experiments.namuwiki import resume_context_api_check
    project, root, _ = prepared_project(tmp_path)
    def blocked(*args):
        raise ValueError("Frozen evaluation incomplete")
    monkeypatch.setattr(resume_context_api_check, "passed_binding", blocked)
    before = {n: (project / n).read_bytes() for n in installer.TARGETS}
    with pytest.raises(ValueError, match="incomplete"):
        installer.install(project, root)
    assert {n: (project / n).read_bytes() for n in installer.TARGETS} == before
