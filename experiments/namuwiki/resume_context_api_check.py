"""Resume only API verification after the complete frozen evaluation passed.

No reanalysis/evaluation, setting application, .env edit or index load is run.
Original evaluation files and runner_status.json remain historical evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.namuwiki.apply_validated_context_settings import (
    SETTING, assert_registered, read_json, sha, validate_pass,
)
from experiments.namuwiki.context_review_cleanup_runner import (
    EXTRA_FILES, REPORT_FILES, capture, compose_base, make_overlay, output_path,
)


def passed_binding(project: Path, root: Path) -> tuple[dict, dict]:
    data = (root / "fresh/validation_report.json").read_bytes()
    report = json.loads(data.decode("utf-8-sig"))
    registration = read_json(root / "fresh/registration.json")
    validate_pass(report, registration)
    if (root / "fresh/failure_report.json").exists():
        raise ValueError("An unresolved fresh evaluation failure report remains")
    activation = read_json(root / "activation/activation_report.json")
    if (activation.get("status") != "settings_applied" or activation.get("setting") != SETTING
            or activation.get("validation_report_sha256") != sha(data)):
        raise ValueError("Settings have not been bound to this completed evaluation")
    assert_registered(project, registration, activation, sha(data))
    return report, registration


def frozen_files(root: Path) -> dict[str, str]:
    # Include cached analyses/checkpoints/details and the original runner status.
    return {str(path.relative_to(root).as_posix()): sha(path.read_bytes())
            for path in [*(root / "fresh").rglob("*"), root / "runner_status.json"]
            if path.is_file()}


def assert_frozen_files(root: Path, before: dict[str, str]) -> None:
    if frozen_files(root) != before:
        raise ValueError("Completed evaluation/history changed during API-only resume")


def check_corpus_files(project: Path, corpus: dict, registration: dict) -> dict:
    namespace = registration["environment"]["QDRANT_NAMESPACE"]
    if not namespace or "/" in namespace or "\\" in namespace or namespace in {".", ".."}:
        raise ValueError("Registered Qdrant namespace is invalid")
    files = {
        "artifacts/context/manifest.json": corpus["context_manifest_sha256"],
        f"artifacts/vector_db/context_qdrant/{namespace}/manifest.json": corpus["state_manifest_sha256"],
        corpus["text_bm25_path"]: corpus["text_bm25_sha256"],
    }
    qdrant = Path(registration["environment"]["QDRANT_PATH"])
    for collection, expected in corpus.get("local_collection_sha256", {}).items():
        if "/" in collection or "\\" in collection or collection in {".", ".."}:
            raise ValueError("Registered Qdrant collection is invalid")
        files[str((qdrant / "collection" / collection / "storage.sqlite").as_posix())] = expected
    checked = {}
    for name, expected in files.items():
        file = project / name
        if file.is_symlink() or project.resolve() not in file.resolve().parents:
            raise ValueError("Registered corpus file must stay inside project without links")
        if not file.is_file() or sha(file.read_bytes()) != expected:
            raise ValueError("Corpus changed after completed evaluation: " + name)
        checked[name] = expected
    return checked


def pack_api_feedback(root: Path) -> Path:
    # Reuse the established whitelist; never include .env, corpus or backups.
    names = tuple("fresh/" + name for name in REPORT_FILES) + EXTRA_FILES + (
        "api_resume_status.json", "api_resume_console.log", "api_resume/runtime_settings.json",
        "api_candidate_install/install_report.json",
    )
    sources = []
    for name in names:
        path = root / name
        if path.is_file():
            if path.is_symlink() or root.resolve() not in path.resolve().parents:
                raise ValueError("Feedback file must stay inside output without links")
            sources.append((name, path.read_bytes()))
    manifest = dict(schema="context_api_resume_feedback_v1",
                    evaluation_policy="Original completed evaluation preserved; API-only follow-up, not a new blind pass.",
                    files={name: dict(bytes=len(data), sha256=sha(data)) for name, data in sources})
    fd, name = tempfile.mkstemp(prefix=root.name + "_api_feedback_", suffix=".zip", dir=root.parent)
    destination = Path(name)
    try:
        with os.fdopen(fd, "w+b") as stream:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
                for name, data in sources:
                    archive.writestr(name, data)
                archive.writestr("export_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        with ZipFile(destination) as archive:
            if archive.testzip() is not None or len(archive.namelist()) != len(sources) + 1:
                raise ValueError("API feedback ZIP verification failed")
            if any(archive.read(name) != data for name, data in sources):
                raise ValueError("API feedback ZIP content differs")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def runtime_identity(project: Path, root: Path) -> int:
    # Run inside the newly built backend image while the server is stopped.
    from dotenv import load_dotenv
    load_dotenv(override=False)
    from experiments.namuwiki.evaluate_context_fixed import _environment, _runtime
    _, registration = passed_binding(project, root)
    if (_environment() != registration["environment"] or _runtime() != registration["runtime"]):
        raise ValueError("API environment/runtime changed since the completed evaluation")
    print("[PASS] Container environment and package versions match completed evaluation", flush=True)
    return 0


def run(project: Path, root: Path) -> int:
    state = dict(schema="context_api_only_resume_v1", status="running", stage="preflight",
                 recorded_at=datetime.now(timezone.utc).isoformat(), steps=[],
                 performance_verified=False, api_verified=False,
                 policy="No full evaluation, setting application, .env change or index load; preserve completed evidence.")
    log = root / "api_resume_console.log"
    base, stopped, exit_code = None, False, 2
    snapshot = None
    try:
        report, registration = passed_binding(project, root)
        snapshot = frozen_files(root)
        state.update(performance_verified=True, validation_report_sha256=sha((root / "fresh/validation_report.json").read_bytes()),
                     frozen_file_sha256=snapshot)
        state["corpus_file_sha256"] = check_corpus_files(project, report["reports"]["dev"]["corpus"], registration)
        if shutil.which("docker") is None:
            raise FileNotFoundError("Docker CLI is unavailable")
        base = compose_base(project)

        def step(stage: str, command: list[str]) -> None:
            state["stage"] = stage
            code = capture(command, log, cwd=project)
            state["steps"].append(dict(stage=stage, exit_code=code))
            if code:
                raise RuntimeError(f"{stage} failed (exit={code}); see API resume console log")

        overlay = make_overlay(project, root)
        evaluation = base + ["-f", str(overlay.relative_to(project))]
        relative = root.relative_to(project).as_posix()
        stopped = True
        step("stop_backend", base + ["stop", "backend"])
        step("build", evaluation + ["build", "backend"])
        prefix = evaluation + ["run", "--rm", "--no-deps", "-T", "backend", "python", "-u", "-m"]
        step("tests", prefix + ["pytest", "-q", "tests/context", "tests/vector_db/test_qdrant_backend.py",
                                "tests/clarify/test_clarify_state.py", "tests/search/test_search_explain.py", "tests/search/test_lyric_evidence.py"])
        passed_binding(project, root)
        step("runtime_identity", prefix + ["experiments.namuwiki.resume_context_api_check", "--runtime-only", "--output-dir", relative])
        step("effective_settings", prefix + ["experiments.namuwiki.check_context_review_runtime", "--output-dir", relative + "/api_resume"])
        runtime = read_json(root / "api_resume/runtime_settings.json")
        corpus = report["reports"]["dev"]["corpus"]
        if (runtime.get("status") != "passed_effective_settings" or runtime.get("context_build_id") != corpus["context_build_id"]
                or runtime.get("dense_collection") != corpus["context_dense_collection"]
                or runtime.get("sparse_collection") != corpus["context_sparse_collection"]):
            raise ValueError("Active Context generation differs from completed evaluation")
        check_corpus_files(project, corpus, registration)
        step("start_api_for_check", evaluation + ["up", "-d", "--no-deps", "--force-recreate", "backend"])
        step("actual_api_check", evaluation + ["exec", "-T", "backend", "python", "-u", "-m",
             "experiments.namuwiki.verify_context_api_ready", "--output-dir", relative, "--wait-ready-seconds", "600"])
        api = read_json(root / "activation/api_smoke_report.json")
        if api.get("schema") != "context_api_ready_smoke_v3" or api.get("status") != "passed_api_smoke":
            raise ValueError("New Candidate@30 HTTP verification did not pass")
        passed_binding(project, root)
        activation = read_json(root / "activation/activation_report.json")
        if (activation.get("api_verified") is not True
                or activation.get("api_smoke_report_sha256") != sha((root / "activation/api_smoke_report.json").read_bytes())):
            raise ValueError("Passed API report is not bound to the active settings")
        state.update(status="passed_api_resume", api_verified=True,
                     api_smoke_report_sha256=sha((root / "activation/api_smoke_report.json").read_bytes()))
        exit_code = 0
    except Exception as exc:
        state.update(status="stopped", error_type=type(exc).__name__, error=str(exc))
        print(f"API-only resume stopped at {state['stage']}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if stopped and base is not None:
            try:
                restored = capture(base + ["up", "-d", "--no-deps", "--force-recreate", "backend"], log, cwd=project)
            except Exception as exc:
                restored = 2
                state["restore_error"] = f"{type(exc).__name__}: {exc}"
            state["backend_restore_exit"] = restored
            if restored:
                state["status"], exit_code = "backend_restore_failed", 2
        if snapshot is not None:
            try:
                assert_frozen_files(root, snapshot)
                state["completed_evaluation_preserved"] = True
            except Exception as exc:
                state.update(status="completed_evaluation_changed", completed_evaluation_preserved=False,
                             performance_verified=False, api_verified=False,
                             preservation_error=str(exc))
                exit_code = 2
        state["exit_code"] = exit_code
        destination = root / "api_resume_status.json"
        if destination.exists():
            history = root / "activation/history"
            history.mkdir(parents=True, exist_ok=True)
            previous = destination.read_bytes()
            (history / ("api_resume_" + sha(previous) + ".json")).write_bytes(previous)
        destination.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            feedback = pack_api_feedback(root)
            print(f"Send this file: {feedback}", flush=True)
        except Exception as exc:
            print(f"API feedback export failed: {type(exc).__name__}: {exc}; send {log}", flush=True)
            exit_code = 2
            state.update(status="feedback_export_failed", exit_code=2, export_error=str(exc))
            destination.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Result: {state['status']}; API report: {root / 'activation/api_smoke_report.json'}", flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    try:
        root = output_path(project, args.output_dir)
        if not root.is_dir():
            raise FileNotFoundError("The completed evaluation output directory does not exist")
        return runtime_identity(project, root) if args.runtime_only else run(project, root)
    except Exception as exc:
        print(f"API resume preflight failed: {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
