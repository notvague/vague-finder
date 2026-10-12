"""Install, test, reanalyze known queries, and verify the API in one host run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


INPUTS = (
    "experiments/namuwiki/eval_queries_context_v01.csv",
    "experiments/namuwiki/queries_context_v01.json",
    "experiments/reranking/eval_queries_v05.csv",
    "data/context/eval_blind_step10.csv",
    "data/context/eval_independent_fixed_20261005.csv",
    "data/context/eval_independent_fixed_20261005_audit.json",
)
REPORT_FILES = (
    "registration.json", "corpus_identity.json", "registered_new_labels.csv", "registered_new_review.json",
    "dev_report.json", "dev_detail.csv", "test_report.json", "test_detail.csv",
    "independent_report.json", "independent_detail.csv", "repeat_report.json",
    "validation_report.json", "validation_report.md", "failure_report.json",
    "dev_checkpoint.json", "test_checkpoint.json", "independent_checkpoint.json", "repeat_checkpoint.json",
    "dev_analysis_cache.json", "test_analysis_cache.json", "independent_analysis_cache.json", "repeat_analysis_cache.json",
)
EXTRA_FILES = (
    "installation/install_report.json", "runtime_settings.json", "runner_status.json", "console_run.log",
    "specificity/install_report.json",
    "scope/install_report.json",
    "lyric_scope/install_report.json",
    "activation/activation_report.json", "activation/api_smoke_report.json",
)


def output_path(project: Path, name: str) -> Path:
    name = name.replace("\\", "/").rstrip("/")
    if not re.fullmatch(r"artifacts/[A-Za-z0-9_./-]+", name) or ".." in name.split("/"):
        raise ValueError("OutputDir must be a relative path inside artifacts")
    path = project / name
    if path.is_symlink() or project.resolve() not in path.resolve().parents:
        raise ValueError("OutputDir must stay inside the project without links")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pack_feedback(root: Path) -> Path:
    """Use a new ZIP name every time, excluding .env and all source backups."""
    names = tuple("fresh/" + name for name in REPORT_FILES) + EXTRA_FILES
    sources = []
    for name in names:
        source = root / name
        if source.is_file():
            if source.is_symlink() or root.resolve() not in source.resolve().parents:
                raise ValueError("Feedback files must stay inside output without links")
            sources.append((name, source.read_bytes()))
    manifest = dict(schema="context_review_cleanup_feedback_v1",
                    evaluation_policy="New analyses of disclosed questions; not a new blind evaluation.",
                    files={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in sources})
    fd, destination = tempfile.mkstemp(prefix=root.name + "_feedback_", suffix=".zip", dir=root.parent)
    destination = Path(destination)
    try:
        with os.fdopen(fd, "w+b") as stream:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
                for name, data in sources:
                    archive.writestr(name, data)
                archive.writestr("export_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        with ZipFile(destination) as archive:
            if archive.testzip() is not None or len(archive.namelist()) != len(sources) + 1:
                raise ValueError("Feedback archive verification failed")
            for name, data in sources:
                if archive.read(name) != data:
                    raise ValueError("Feedback archive content mismatch")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def compose_base(project: Path) -> list[str]:
    files = ["docker-compose.yml"]
    if (project / "compose.context-test.yml").is_file():
        files.append("compose.context-test.yml")
    files.append("compose.context-ranking.yml")
    for name in files:
        if not (project / name).is_file():
            raise FileNotFoundError(f"Compose file missing: {name}")
    return ["docker", "compose", *[item for file in files for item in ("-f", file)]]


def make_overlay(project: Path, root: Path) -> Path:
    # Override only evaluation conditions. Context values still come from the
    # normal factory/.env/profile and are checked, never forced by this overlay.
    timing = (root.relative_to(project) / "timing/api.jsonl").as_posix()
    overlay = root / "setup/compose.evaluation.json"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(json.dumps({"services": {"backend": {"environment": {
        "RERANKER_ENABLED": "false", "SEARCH_REFERENCE_YEAR": "2026",
        "SEARCH_TIMING_LOG": timing, "PYTHONUTF8": "1",
    }}}}, indent=2) + "\n", encoding="utf-8")
    return overlay


def capture(command: list[str], log_path: Path, *, cwd: Path) -> int:
    """Capture both native streams without PowerShell's stderr exception behavior."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write("\nCommand: " + json.dumps(command, ensure_ascii=True) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        log.write(f"Child process exit: {code}\n")
    return code


def run(project: Path, root: Path, *, mode: str = "All", update_env: bool = True,
        resume: bool = False, media_specificity_fix: bool = False, media_scope_fix: bool = False,
        lyric_scope_fix: bool = False) -> int:
    if root.exists() and any(root.iterdir()) and not resume:
        raise ValueError("OutputDir already has files. Use a new name, or --resume with unchanged code/settings/inputs.")
    root.mkdir(parents=True, exist_ok=True)
    log = root / "console_run.log"
    state = dict(schema="context_review_cleanup_runner_v1", status="running", mode=mode,
                 recorded_at=datetime.now(timezone.utc).isoformat(), stage="preflight", steps=[],
                 performance_verified=False, api_verified=False,
                 question_policy="166 previously observed questions plus 10 repeat analyses; not a new blind sample")
    state["media_specificity_fix"] = media_specificity_fix
    state["media_scope_fix"] = media_scope_fix
    state["lyric_scope_fix"] = lyric_scope_fix
    base = None
    stop_attempted = False
    code = 2
    restore_code = None
    try:
        base = compose_base(project)
        if shutil.which("docker") is None:
            raise FileNotFoundError("Docker CLI is not available")
        if update_env and not (project / ".env").is_file():
            raise FileNotFoundError("Keep your existing .env with shared Gemini/Mongo settings before running")
        if mode == "All":
            for name in INPUTS + ("experiments/namuwiki/verify_context_api_ready.py",):
                if not (project / name).is_file():
                    raise FileNotFoundError(f"Existing private input or verification code missing: {name}")

        def step(stage: str, command: list[str]) -> None:
            state["stage"] = stage
            status = capture(command, log, cwd=project)
            state["steps"].append(dict(stage=stage, exit_code=status))
            if status:
                raise RuntimeError(f"{stage} failed (exit={status}); see captured console output")

        stop_attempted = True
        step("stop_backend", base + ["stop", "backend"])
        from experiments.namuwiki.install_context_review_cleanup import install
        if media_specificity_fix:
            state["stage"] = "specificity_preflight"
            from experiments.namuwiki.install_context_media_specificity import edit_context_media_safeguards
            # Confirm the source can be patched before the cleanup installer
            # writes any of its separate, targeted changes.
            edit_context_media_safeguards((project / "src/retrieval/context_query.py").read_bytes())
            if not (project / "src/retrieval/context_clue_specificity.py").is_file():
                raise FileNotFoundError("Extract the complete Context specificity fix ZIP first")
        if media_scope_fix:
            state["stage"] = "scope_preflight"
            from experiments.namuwiki.install_context_media_scope import preflight
            preflight(project)
        if lyric_scope_fix:
            state["stage"] = "lyric_scope_preflight"
            from experiments.namuwiki.install_context_lyric_scope import preflight as preflight_lyrics
            preflight_lyrics(project)
        state["stage"] = "install"
        installed = install(project, root, apply_env=update_env)
        print("[PASS] Targeted install: " + ", ".join(installed["changed_files"]), flush=True)
        if media_specificity_fix:
            from experiments.namuwiki.install_context_media_specificity import install as install_specificity
            state["stage"] = "install_media_specificity"
            specificity = install_specificity(project, root)
            print("[PASS] Context media specificity: " + ", ".join(specificity["changed_files"]), flush=True)
        if media_scope_fix:
            from experiments.namuwiki.install_context_media_scope import install as install_scope
            state["stage"] = "install_media_scope"
            scope = install_scope(project, root)
            print("[PASS] Context media scope: " + ", ".join(scope["changed_files"]), flush=True)
        if lyric_scope_fix:
            from experiments.namuwiki.install_context_lyric_scope import install as install_lyrics
            state["stage"] = "install_lyric_scope"
            lyrics = install_lyrics(project, root)
            print("[PASS] Literal lyric modality boundary: " + ", ".join(lyrics["changed_files"]), flush=True)
        overlay = make_overlay(project, root)
        evaluation = base + ["-f", str(overlay.relative_to(project))]
        step("build", evaluation + ["build", "backend"])
        prefix = evaluation + ["run", "--rm", "--no-deps", "-T", "backend", "python", "-u", "-m"]
        step("tests", prefix + ["pytest", "-q", "tests/context", "tests/test_qdrant_backend.py",
                                "tests/test_clarify_state.py", "tests/test_search_explain.py", "tests/test_lyric_evidence.py"])
        if lyric_scope_fix:
            step("lyric_boundary_smoke", prefix + ["experiments.namuwiki.inspect_context_lyric_scope"])
        if shutil.which("git") and (project / ".git").exists():
            step("git_diff_check", ["git", "--no-pager", "diff", "--check"])
        if mode == "Tests":
            state["status"] = "passed_code_tests_only"
        else:
            relative = root.relative_to(project).as_posix()
            step("effective_settings", prefix + ["experiments.namuwiki.check_context_review_runtime", "--output-dir", relative])
            step("fresh_off_on", prefix + ["experiments.namuwiki.evaluate_context_fixed", "--phase", "all",
                                          "--output-dir", relative + "/fresh"])
            report = read_json(root / "fresh/validation_report.json")
            if report.get("status") != "passed_observed_tests":
                raise ValueError("Comparison gate did not pass: " + str(report.get("status")) +
                                 "; keep settings/labels fixed and review the report")
            state["performance_verified"] = True
            step("bind_activation", prefix + ["experiments.namuwiki.apply_validated_context_settings", "--output-dir", relative])
            step("start_api_for_check", evaluation + ["up", "-d", "--no-deps", "--force-recreate", "backend"])
            step("actual_api_check", evaluation + ["exec", "-T", "backend", "python", "-u", "-m",
                 "experiments.namuwiki.verify_context_api_ready", "--output-dir", relative, "--wait-ready-seconds", "600"])
            if read_json(root / "activation/api_smoke_report.json").get("status") != "passed_api_smoke":
                raise ValueError("Actual HTTP API gate did not pass")
            state.update(status="passed_review_cleanup_validation", api_verified=True)
        code = 0
    except Exception as exc:
        state.update(status="stopped", error_type=type(exc).__name__, error=str(exc))
        print(f"Stopped at {state['stage']}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        # Return to the normal reranker/reference/timing settings. Do not start
        # the local Mongo service: this project uses the shared Mongo instance.
        if stop_attempted and base is not None:
            try:
                restore_code = capture(base + ["up", "-d", "--no-deps", "--force-recreate", "backend"], log, cwd=project)
            except Exception as exc:
                restore_code = 2
                state["restore_error"] = f"{type(exc).__name__}: {exc}"
            if restore_code:
                state["status"] = "backend_restore_failed"
                code = 2
        state["backend_restore_exit"] = restore_code
        state["exit_code"] = code
        (root / "runner_status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            feedback = pack_feedback(root)
            print(f"Send this file: {feedback}", flush=True)
        except Exception as exc:
            print(f"Feedback export failed: {type(exc).__name__}: {exc}; send {log}", flush=True)
            code = 2
            state.update(status="feedback_export_failed", exit_code=2,
                         export_error=f"{type(exc).__name__}: {exc}")
            (root / "runner_status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Result: {state['status']}; reports: {root}", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("All", "Tests"), default="All")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-update-env", action="store_true")
    parser.add_argument("--media-specificity-fix", action="store_true")
    parser.add_argument("--media-scope-fix", action="store_true")
    parser.add_argument("--lyric-scope-fix", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    prefix = ("artifacts/context_lyric_scope_" if args.lyric_scope_fix else
              "artifacts/context_media_scope_" if args.media_scope_fix else
              "artifacts/context_media_specificity_" if args.media_specificity_fix else "artifacts/context_review_cleanup_")
    name = args.output_dir or (prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    try:
        return run(project, output_path(project, name), mode=args.mode,
                   update_env=not args.no_update_env, resume=args.resume,
                   media_specificity_fix=args.media_specificity_fix, media_scope_fix=args.media_scope_fix,
                   lyric_scope_fix=args.lyric_scope_fix)
    except Exception as exc:
        print(f"Runner preflight stopped: {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
