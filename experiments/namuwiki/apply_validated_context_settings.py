"""Apply only the measured router settings after every frozen phase passes.

Runs on the host with the Python standard library. It does not open Qdrant,
modify .env, change evaluation labels, or import the backend factories.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


SETTING = dict(weight=0.5, fact_k=100, sparse_k=100, candidate_k=30,
               top_k=10, named_media_multiplier=2.0)
ROUTER_VALUES = dict(context_weight=0.5, context_fact_k=100, context_sparse_k=100,
                     context_named_media_multiplier=2.0, default_candidate_k=30)
DEPENDENCIES = "src/backend/api/dependencies.py"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_pass(report: dict, registration: dict) -> None:
    if (report.get("schema") != "context_fixed_validation_v1"
            or report.get("status") != "passed_observed_tests"
            or report.get("registration") != registration
            or registration.get("schema") != "context_fixed_registered_v1"
            or registration.get("setting") != SETTING
            or registration.get("rerank_enabled") is not False
            or registration.get("reference_year") != 2026):
        raise ValueError("Complete frozen evaluation with the measured settings has not passed")
    counts = dict(dev=74, test=32, independent=60)
    if (set(report.get("completed_phases", ())) != set(counts)
            or set(report.get("reports", {})) != set(counts)):
        raise ValueError("Dev, test and independent phases must all be present")
    corpora = []
    for phase, count in counts.items():
        result = report["reports"][phase]
        if (result.get("status") != "passed" or result.get("evaluated") != count
                or result.get("setting") != SETTING or result.get("rerank_enabled") is not False
                or result.get("violations") != []
                or result.get("evidence", {}).get("unreviewed_count") != 0):
            raise ValueError(f"{phase}: incomplete, failed, or unreviewed evaluation")
        corpus = result.get("corpus")
        if not isinstance(corpus, dict) or corpus.get("source_scope_complete") is not True:
            raise ValueError(f"{phase}: complete corpus identity is missing")
        corpora.append(corpus)
    if any(corpus != corpora[0] for corpus in corpora[1:]):
        raise ValueError("Evaluation phases used different corpora")
    repeat = report.get("analyzer_repeat_check") or {}
    if (repeat.get("question_count") != 10 or repeat.get("violations") != []
            or repeat.get("evidence", {}).get("unreviewed_count") != 0):
        raise ValueError("Fresh reanalysis checks are incomplete, failed, or unreviewed")


def rewrite_factory(data: bytes) -> bytes:
    """Edit five argument values in the one router factory, preserving other code."""
    bom = b"\xef\xbb\xbf" if data.startswith(b"\xef\xbb\xbf") else b""
    content = data[len(bom):]
    source = content.decode("utf-8")
    tree = ast.parse(source)
    factories = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "get_search_router"]
    if len(factories) != 1:
        raise ValueError("Expected one get_search_router factory; source was not changed")
    calls = [node for node in ast.walk(factories[0]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "SearchRouter"]
    if len(calls) != 1 or any(kw.arg is None for kw in calls[0].keywords):
        raise ValueError("Expected one explicit SearchRouter call without **kwargs; source was not changed")
    call = calls[0]
    keywords = {kw.arg: kw for kw in call.keywords}
    if len(keywords) != len(call.keywords) or "context_search" not in keywords:
        raise ValueError("Context factory wiring is missing or duplicate; source was not changed")
    if isinstance(keywords["context_search"].value, ast.Constant) and keywords["context_search"].value.value is None:
        raise ValueError("Context search is explicitly disabled; source was not changed")
    # Preserve validated environment wiring. Do not turn it back into literals.
    from src.retrieval.context_search_settings import factory_uses_context_settings
    if factory_uses_context_settings(factories[0], call):
        return data
    lines = content.splitlines(keepends=True)
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    def position(node, *, end=False):
        return starts[(node.end_lineno if end else node.lineno) - 1] + (node.end_col_offset if end else node.col_offset)

    edits = []
    missing = []
    for name, value in ROUTER_VALUES.items():
        if name in keywords:
            node = keywords[name].value
            edits.append((position(node), position(node, end=True), repr(value).encode("ascii")))
        else:
            missing.append((name, value))
    if missing:
        close = position(call, end=True) - 1
        if content[close:close + 1] != b")":
            raise ValueError("Router call closing delimiter could not be located")
        last = call.keywords[-1].value if call.keywords else call.args[-1] if call.args else None
        trailing_comma = last is not None and content[position(last, end=True):close].lstrip().startswith(b",")
        start = starts[call.end_lineno - 1]
        prefix = content[start:close]
        if not prefix.strip():
            if last is not None and not trailing_comma:
                edits.append((position(last, end=True), position(last, end=True), b","))
            newline = b"\r\n" if b"\r\n" in content else b"\n"
            insertion = b"".join(prefix + b"    " + f"{name}={value!r},".encode("ascii") + newline
                                  for name, value in missing)
            edits.append((start, start, insertion))
        else:
            separator = b", " if last is not None and not trailing_comma else b" "
            insertion = separator + b", ".join(f"{name}={value!r}".encode("ascii") for name, value in missing)
            edits.append((close, close, insertion))
    for start, end, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        content = content[:start] + replacement + content[end:]
    ast.parse(content.decode("utf-8"))
    return bom + content


def assert_registered(project: Path, registration: dict, prior: dict | None, report_sha: str) -> None:
    hashes = {**registration["source_hashes"], **registration["code_sha256"]}
    if (prior and prior.get("status") == "settings_applied"
            and prior.get("validation_report_sha256") == report_sha):
        hashes[DEPENDENCIES] = prior["dependencies_after_sha256"]
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Registered file path must stay inside the project")
        target = project / path
        if target.is_symlink() or project.resolve() not in target.resolve().parents:
            raise ValueError("Registered files must stay inside the project without links")
        if not target.is_file() or sha(target.read_bytes()) != expected:
            raise ValueError(f"Registered code/input changed after evaluation: {name}")


def apply(project: Path, output: Path) -> dict:
    output = output.resolve()
    project = project.resolve()
    if output == project or project not in output.parents or output.relative_to(project).parts[0] != "artifacts":
        raise ValueError("Output directory must be inside the project's artifacts")
    report_path, registration_path = output / "fresh/validation_report.json", output / "fresh/registration.json"
    report_data = report_path.read_bytes()
    report, registration = json.loads(report_data.decode("utf-8-sig")), read_json(registration_path)
    validate_pass(report, registration)
    if (output / "fresh/failure_report.json").exists():
        raise ValueError("An unresolved evaluation failure report remains")
    destination = output / "activation/activation_report.json"
    prior = read_json(destination) if destination.exists() else None
    assert_registered(project, registration, prior, sha(report_data))
    source = project / DEPENDENCIES
    before = source.read_bytes()
    after = rewrite_factory(before)
    backup_dir = destination.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / ("dependencies_" + sha(before)[:16] + ".py")
    if not backup.exists():
        backup.write_bytes(before)
    if before != after:
        handle, temporary = tempfile.mkstemp(prefix=".context_settings_", suffix=".py", dir=source.parent)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(after)
                stream.flush()
                os.fsync(stream.fileno())
            # Check again so a simultaneous edit cannot be overwritten silently.
            if source.read_bytes() != before:
                raise ValueError("Router factory changed during settings application")
            os.replace(temporary, source)
        finally:
            Path(temporary).unlink(missing_ok=True)
    result = dict(schema="context_validated_activation_v1", status="settings_applied",
                  recorded_at=datetime.now(timezone.utc).isoformat(), setting=SETTING,
                  router_arguments=ROUTER_VALUES, validation_report_sha256=sha(report_data),
                  dependencies_before_sha256=sha(before), dependencies_after_sha256=sha(after),
                  source_changed=before != after, backup=str(backup.relative_to(project)),
                  api_verified=False,
                  note="Measured settings wired into the factory; API smoke verification still required. Reranking is OFF in the verified requests.")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[PASS] Complete frozen evaluation gate; measured Context settings applied")
    print("Router: weight=0.5, fact_k=100, sparse_k=100, named_media_multiplier=2, candidate_k=30")
    print(f"Private activation report: {destination}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    try:
        apply(project, project / args.output_dir)
    except Exception as exc:
        print(f"Settings application stopped: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
