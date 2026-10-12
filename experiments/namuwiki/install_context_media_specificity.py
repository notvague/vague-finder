"""Patch only the accepted Context rewrite in the merged analyzer safeguards."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.namuwiki.install_context_review_cleanup import (
    atomic_write, decode, digest, positions, safe_file,
)


HELPER_MODULE = "src.retrieval.context_clue_specificity"
HELPER_NAME = "preserve_context_media_search"


def _is_search_target(node: ast.AST) -> bool:
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "rule" and isinstance(node.slice, ast.Constant)
            and node.slice.value == "search_query")


def edit_context_media_safeguards(data: bytes) -> bytes:
    """Fail closed for an unfamiliar body; preserve everything around it."""
    bom, source, newline = decode(data)
    content = source.encode("utf-8")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "apply_context_query_safeguards"]
    if len(functions) != 1:
        raise ValueError("Expected one apply_context_query_safeguards function")
    function = functions[0]
    assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and _is_search_target(n.targets[0])]
    if len(assignments) != 1:
        raise ValueError("Expected one accepted model Context search rewrite; source preserved")
    assignment = assignments[0]
    imports = [n for n in ast.walk(function) if isinstance(n, ast.ImportFrom)
               and n.module == HELPER_MODULE and any(a.name == HELPER_NAME for a in n.names)]
    expected = ast.parse(
        'preserve_context_media_search(search, source_query=rule["search_query"], '
        'relation=rule["relation"], target=rule["target"])', mode="eval",
    ).body
    if ast.dump(assignment.value) == ast.dump(expected):
        if len(imports) != 1:
            raise ValueError("Specificity call exists without the expected helper import")
        return data
    if imports or not isinstance(assignment.value, ast.Name) or assignment.value.id != "search":
        raise ValueError("Unrecognized Context search rewrite; source preserved")
    # Require the preceding grounding check, not an arbitrary same-named field
    # elsewhere in a new implementation. Do not patch a weakened parser.
    enclosing = [n for n in ast.walk(function) if isinstance(n, ast.If)
                 and assignment in n.body and isinstance(n.test, ast.BoolOp)]
    if len(enclosing) != 1 or not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_grounded"
        for n in ast.walk(enclosing[0].test)
    ):
        raise ValueError("Cannot verify the existing source-grounding check")
    _, position = positions(content)
    start, end = position(assignment.value), position(assignment.value, end=True)
    replacement = (
        'preserve_context_media_search(search, source_query=rule["search_query"], '
        'relation=rule["relation"], target=rule["target"])'
    ).encode("utf-8")
    body = function.body
    anchor = body[0]
    if isinstance(anchor, ast.Expr) and isinstance(anchor.value, ast.Constant) and isinstance(anchor.value.value, str):
        anchor = body[1]
    import_at = content.rfind(b"\n", 0, position(anchor)) + 1
    indent = content[import_at:position(anchor)]
    if indent.strip():
        raise ValueError("Expected a standalone parser body statement")
    addition = indent + f"from {HELPER_MODULE} import {HELPER_NAME}".encode() + newline.encode()
    changes = [(start, end, replacement), (import_at, import_at, addition)]
    for start, end, replacement in sorted(changes, reverse=True):
        content = content[:start] + replacement + content[end:]
    compile(content.decode("utf-8"), "context_query.py", "exec")
    return bom + content


def install(project: Path, output: Path) -> dict:
    project, output = project.resolve(), output.resolve()
    if project not in output.parents or output.relative_to(project).parts[0] != "artifacts":
        raise ValueError("Specificity backups/reports must stay inside artifacts")
    path = safe_file(project, "src/retrieval/context_query.py")
    before = path.read_bytes()
    after = edit_context_media_safeguards(before)
    if not safe_file(project, "src/retrieval/context_clue_specificity.py").is_file():
        raise FileNotFoundError("Context specificity helper missing; extract the complete fix ZIP first")
    report = output / "specificity/install_report.json"
    previous = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
    if before != after:
        backup = output / "specificity/backups/src/retrieval/context_query.py"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(before)
        if path.read_bytes() != before:
            raise ValueError("Context source changed during installation")
        atomic_write(path, after)
    result = dict(
        schema="context_media_specificity_install_v1", status="applied",
        recorded_at=datetime.now(timezone.utc).isoformat(),
        changed_files=["src/retrieval/context_query.py"] if before != after else [],
        source_sha256={"before": digest(before), "after": digest(after)},
        scope="One grounded Context search rewrite; no weights, labels or merged router/backend changes.",
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    if previous is not None:
        result["first_install_source_sha256"] = previous.get(
            "first_install_source_sha256", previous.get("source_sha256"),
        )
    else:
        result["first_install_source_sha256"] = dict(result["source_sha256"])
    try:
        atomic_write(report, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    except Exception:
        if before != after and path.read_bytes() == after:
            atomic_write(path, before)
        raise
    return result
