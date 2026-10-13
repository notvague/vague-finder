"""Resume controls must preserve complete measurement evidence and normal services."""

from copy import deepcopy
import json
from zipfile import ZipFile

import pytest

from experiments.namuwiki import resume_context_api_check as runner
from experiments.namuwiki.apply_validated_context_settings import DEPENDENCIES, SETTING, sha


def completed_project(tmp_path):
    project = tmp_path / "project"
    output = project / "artifacts/completed"
    values = {
        DEPENDENCIES: b"# fixed production source\n",
        "data/context/synthetic_labels.csv": b"id,label\n1,Synthetic\n",
        ".env": b"SYNTHETIC_PRIVATE_TOKEN=unchanged\n",
        "docker-compose.yml": b"services: {}\n",
        "compose.context-ranking.yml": b"services: {}\n",
        "artifacts/context/manifest.json": b"{}",
        "artifacts/vector_db/context_qdrant/dev/manifest.json": b"{}",
        "artifacts/bm25_params.json": b"{}",
        "artifacts/qdrant/collection/synthetic-text/storage.sqlite": b"synthetic stored vector data",
    }
    for name, data in values.items():
        file = project / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    registration = dict(schema="context_fixed_registered_v1", setting=SETTING, rerank_enabled=False,
                        reference_year=2026, source_hashes={"data/context/synthetic_labels.csv": sha(values["data/context/synthetic_labels.csv"])},
                        code_sha256={DEPENDENCIES: sha(values[DEPENDENCIES])},
                        environment=dict(QDRANT_NAMESPACE="dev", QDRANT_PATH="artifacts/qdrant"))
    corpus = dict(context_build_id="synthetic-build", context_dense_collection="synthetic-dense",
                  context_sparse_collection="synthetic-sparse", source_scope_complete=True,
                  context_manifest_sha256=sha(b"{}"), state_manifest_sha256=sha(b"{}"),
                  text_bm25_path="artifacts/bm25_params.json", text_bm25_sha256=sha(b"{}"),
                  local_collection_sha256={"synthetic-text": sha(values["artifacts/qdrant/collection/synthetic-text/storage.sqlite"])})
    reports = {phase: dict(status="passed", evaluated=n, setting=SETTING, rerank_enabled=False,
                          violations=[], evidence=dict(unreviewed_count=0), corpus=deepcopy(corpus))
               for phase, n in dict(dev=74, test=32, independent=60).items()}
    report = dict(schema="context_fixed_validation_v1", status="passed_observed_tests", registration=registration,
                  completed_phases=list(reports), reports=reports,
                  analyzer_repeat_check=dict(question_count=10, violations=[], evidence=dict(unreviewed_count=0)))
    (output / "fresh").mkdir(parents=True)
    (output / "activation").mkdir()
    (output / "fresh/validation_report.json").write_text(json.dumps(report))
    (output / "fresh/registration.json").write_text(json.dumps(registration))
    (output / "fresh/dev_detail.csv").write_text("synthetic original detail")
    (output / "fresh/independent_analysis_cache.json").write_text("synthetic original analysis cache")
    (output / "runner_status.json").write_text('{"status":"stopped","stage":"actual_api_check"}')
    activation = dict(status="settings_applied", setting=SETTING,
                      validation_report_sha256=sha((output / "fresh/validation_report.json").read_bytes()),
                      dependencies_after_sha256=sha(values[DEPENDENCIES]), api_verified=False)
    (output / "activation/activation_report.json").write_text(json.dumps(activation))
    return project, output, values, corpus


def install_capture(monkeypatch, project, output, corpus, *, fault=""):
    commands = []

    def capture(command, log, *, cwd):
        commands.append(command)
        assert cwd == project
        log.write_text("synthetic captured output")
        if command[-2:] == ["stop", "backend"] and fault == "stop":
            return 2
        if command[-2:] == ["build", "backend"] and fault == "build":
            return 2
        if "pytest" in command and fault == "tests":
            return 2
        if "--runtime-only" in command and fault == "runtime":
            return 2
        if "experiments.namuwiki.check_context_review_runtime" in command:
            value = dict(status="passed_effective_settings", context_build_id=corpus["context_build_id"],
                         dense_collection=corpus["context_dense_collection"], sparse_collection=corpus["context_sparse_collection"])
            if fault == "generation":
                value["context_build_id"] = "another build"
            (output / "api_resume").mkdir(exist_ok=True)
            (output / "api_resume/runtime_settings.json").write_text(json.dumps(value))
        if "experiments.namuwiki.verify_context_api_ready" in command:
            if fault == "api":
                return 2
            (output / "activation/api_smoke_report.json").write_text(json.dumps(dict(
                schema="context_api_ready_smoke_v3", status="passed_api_smoke")))
            activation = json.loads((output / "activation/activation_report.json").read_text())
            activation.update(api_verified=True, api_smoke_report_sha256=sha((output / "activation/api_smoke_report.json").read_bytes()))
            if fault == "binding":
                activation["api_verified"] = False
            (output / "activation/activation_report.json").write_text(json.dumps(activation))
        if fault == "history" and "pytest" in command:
            (output / "fresh/dev_detail.csv").write_text("unauthorized change")
        if "up" in command and "-f" in command and command[-1] == "backend":
            if fault == "restore" and str(output / "setup/compose.evaluation.json") not in command:
                # The normal command has no evaluation overlay; use last step.
                if commands[-2:] and any("experiments.namuwiki.verify_context_api_ready" in c for c in commands):
                    return 2
        return 0

    monkeypatch.setattr(runner, "capture", capture)
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/synthetic/docker")
    return commands


def test_api_only_resume_preserves_completed_files_settings_sources_and_uses_no_full_evaluation(tmp_path, monkeypatch):
    project, output, values, corpus = completed_project(tmp_path)
    before = runner.frozen_files(output)
    commands = install_capture(monkeypatch, project, output, corpus)
    assert runner.run(project, output) == 0
    state = json.loads((output / "api_resume_status.json").read_text())
    assert state["status"] == "passed_api_resume" and state["api_verified"] and state["performance_verified"]
    assert state["completed_evaluation_preserved"] and state["backend_restore_exit"] == 0
    assert runner.frozen_files(output) == before
    assert all((project / name).read_bytes() == data for name, data in values.items())
    assert commands[-1][-5:] == ["up", "-d", "--no-deps", "--force-recreate", "backend"]
    assert all("mongodb" not in c and "fresh_off_on" not in c and "experiments.namuwiki.evaluate_context_fixed" not in c
               and "experiments.namuwiki.apply_validated_context_settings" not in c for c in commands)
    assert sum("experiments.namuwiki.verify_context_api_ready" in c for c in commands) == 1
    assert all("--no-deps" in c for c in commands if "run" in c or "up" in c)


@pytest.mark.parametrize("fault", ["stop", "build", "tests", "runtime", "generation", "api", "binding", "restore"])
def test_every_failed_stage_restores_normal_backend_and_reports_no_api_pass(tmp_path, monkeypatch, fault):
    project, output, _, corpus = completed_project(tmp_path)
    before = runner.frozen_files(output)
    commands = install_capture(monkeypatch, project, output, corpus, fault=fault)
    assert runner.run(project, output) == 2
    state = json.loads((output / "api_resume_status.json").read_text())
    assert state["status"] != "passed_api_resume"
    assert commands[-1][-1] == "backend" and "up" in commands[-1] and "--no-deps" in commands[-1]
    assert runner.frozen_files(output) == before
    assert state["completed_evaluation_preserved"]


@pytest.mark.parametrize("fault", ["unfinished", "registration", "source", "labels", "corpus", "activation", "failure_report"])
def test_invalid_history_or_changed_sources_are_rejected_before_backend_stop(tmp_path, monkeypatch, fault):
    project, output, _, corpus = completed_project(tmp_path)
    commands = install_capture(monkeypatch, project, output, corpus)
    if fault == "unfinished":
        report = json.loads((output / "fresh/validation_report.json").read_text())
        report["status"] = "blocked"
        (output / "fresh/validation_report.json").write_text(json.dumps(report))
    elif fault == "registration":
        reg = json.loads((output / "fresh/registration.json").read_text())
        reg["reference_year"] = 2025
        (output / "fresh/registration.json").write_text(json.dumps(reg))
    elif fault in {"source", "labels", "corpus"}:
        name = {"source": DEPENDENCIES, "labels": "data/context/synthetic_labels.csv",
                "corpus": "artifacts/qdrant/collection/synthetic-text/storage.sqlite"}[fault]
        (project / name).write_bytes(b"changed data")
    elif fault == "activation":
        (output / "activation/activation_report.json").write_text('{}')
    else:
        (output / "fresh/failure_report.json").write_text('{}')
    assert runner.run(project, output) == 2
    assert commands == []


def test_completed_checkpoint_or_detail_mutation_is_detected_even_if_api_passes(tmp_path, monkeypatch):
    project, output, _, corpus = completed_project(tmp_path)
    install_capture(monkeypatch, project, output, corpus, fault="history")
    assert runner.run(project, output) == 2
    state = json.loads((output / "api_resume_status.json").read_text())
    assert state["status"] == "completed_evaluation_changed"
    assert state["completed_evaluation_preserved"] is False and state["performance_verified"] is False


def test_export_uses_unique_files_preserves_original_log_and_excludes_credentials_and_backups(tmp_path):
    project, output, _, _ = completed_project(tmp_path)
    (output / "api_resume_console.log").write_text("Synthetic API console")
    (output / "api_resume_status.json").write_text('{"status":"passed_api_resume"}')
    (output / ".env").write_text("PRIVATE_TEST_TOKEN=not for export")
    (output / "activation/backups").mkdir()
    (output / "activation/backups/private.py").write_text("private source backup")
    one, two = runner.pack_api_feedback(output), runner.pack_api_feedback(output)
    assert one != two and one.is_file() and two.is_file()
    with ZipFile(two) as archive:
        assert archive.testzip() is None
        assert "api_resume_status.json" in archive.namelist()
        assert "fresh/validation_report.json" in archive.namelist()
        assert not any(".env" in name or "backups" in name for name in archive.namelist())
        manifest = json.loads(archive.read("export_manifest.json"))
        for name, item in manifest["files"].items():
            assert item["sha256"] == sha(archive.read(name))


def test_existing_api_resume_status_is_archived_on_a_second_run(tmp_path, monkeypatch):
    project, output, _, corpus = completed_project(tmp_path)
    install_capture(monkeypatch, project, output, corpus)
    assert runner.run(project, output) == 0
    previous = (output / "api_resume_status.json").read_bytes()
    assert runner.run(project, output) == 0
    assert (output / "activation/history" / ("api_resume_" + sha(previous) + ".json")).read_bytes() == previous
