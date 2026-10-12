"""Apply reviewed changes to the supplied merged project, preserving unknown work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from experiments.namuwiki.install_context_review_cleanup import atomic_write, safe_file


PAYLOAD = Path("experiments/namuwiki/context_media_scope_payload")
SCHEMA = "context_media_scope_payload_v1"
TARGETS = frozenset({
    "src/retrieval/context_query.py",
    "src/retrieval/context_evidence.py",
    "src/retrieval/context_qdrant_search.py",
    "experiments/namuwiki/evaluate_context_fixed.py",
    "tests/context/test_context_fixed_pipeline.py",
    "tests/context/test_context_fixed_followup_retrieval.py",
    "tests/context/test_context_qdrant_load.py",
})


def relative_name(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("Invalid relative scope path")
    return value


def normalized(data: bytes) -> bytes:
    return data.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(normalized(data)).hexdigest()


def preflight(project: Path) -> list[tuple[str, bytes, bytes]]:
    project = project.resolve()
    manifest = json.loads(safe_file(project, str(PAYLOAD / "manifest.json")).read_text(encoding="utf-8-sig"))
    if (manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), list)
            or not manifest["files"] or not isinstance(manifest.get("required_new_files"), dict)):
        raise ValueError("Invalid scope manifest schema")
    changes = []
    seen = set()
    for entry in manifest["files"]:
        name = relative_name(entry["target"])
        if name in seen or name not in TARGETS:
            raise ValueError("Invalid or duplicate scope target")
        if any(not re.fullmatch(r"[0-9a-f]{64}", entry[key]) for key in ("before", "after")):
            raise ValueError("Invalid scope source digest")
        seen.add(name)
        source = safe_file(project, name)
        before = source.read_bytes()
        replacement = safe_file(project, str(PAYLOAD / relative_name(entry["payload"]))).read_bytes()
        if sha(replacement) != entry["after"]:
            raise ValueError(f"Scope payload integrity mismatch: {name}")
        text = normalized(replacement).decode("utf-8")
        compile(text, name, "exec")
        if sha(before) == entry["after"]:
            continue
        if sha(before) != entry["before"]:
            raise ValueError(f"Source differs from the supplied project: {name}. Preserved; do not force overwrite.")
        if b"\r\n" in before:
            text = text.replace("\n", "\r\n")
        after = (b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b"") + text.encode("utf-8")
        changes.append((name, before, after))
    for name, expected in manifest["required_new_files"].items():
        name = relative_name(name)
        if not name.startswith(("src/retrieval/", "tests/context/", "experiments/namuwiki/")):
            raise ValueError("Invalid required scope file")
        if sha(safe_file(project, name).read_bytes()) != expected:
            raise ValueError(f"Complete scope fix extraction required: {name}")
    return changes


def install(project: Path, output: Path) -> dict:
    project, output = project.resolve(), output.resolve()
    if project not in output.parents or output.relative_to(project).parts[0] != "artifacts":
        raise ValueError("Scope reports/backups must stay inside artifacts")
    changes = preflight(project)
    changed = []
    try:
        for name, before, after in changes:
            path = safe_file(project, name)
            if path.read_bytes() != before:
                raise ValueError(f"Source changed during installation: {name}")
            backup = output / "scope/backups" / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(before)
            atomic_write(path, after)
            changed.append((name, before, after))
        result = dict(schema="context_media_scope_install_v1", status="applied",
                      changed_files=[name for name, _, _ in changed],
                      fixed_inputs="No settings, labels, corpus or merged search router/backend changes.",
                      source_sha256={name: dict(before=sha(before), after=sha(after)) for name, before, after in changed})
        report = output / "scope/install_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(report, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    except Exception:
        for name, before, after in reversed(changed):
            path = safe_file(project, name)
            if path.read_bytes() == after:
                atomic_write(path, before)
        raise
    return result
