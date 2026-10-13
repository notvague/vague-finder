"""Install one reviewed modality change against the supplied source revision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.namuwiki.install_context_review_cleanup import atomic_write, safe_file


TARGET = "src/retrieval/modality_queries.py"
PAYLOAD = "experiments/namuwiki/context_lyric_scope_payload"


def normalized(data: bytes) -> bytes:
    return data.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(normalized(data)).hexdigest()


def preflight(project: Path) -> tuple[bytes, bytes]:
    manifest = json.loads(safe_file(project, PAYLOAD + "/manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != "context_lyric_scope_payload_v1" or manifest.get("target") != TARGET:
        raise ValueError("Invalid lyric scope manifest")
    data = safe_file(project, PAYLOAD + "/modality_queries.py.payload").read_bytes()
    if digest(data) != manifest["after"]:
        raise ValueError("Lyric scope payload integrity mismatch")
    for name, expected in manifest["required_files"].items():
        if (not isinstance(name, str) or "\\" in name or ":" in name or ".." in name.split("/")
                or not name.startswith(("experiments/namuwiki/", "tests/context/"))):
            raise ValueError("Invalid lyric scope required file")
        if digest(safe_file(project, name).read_bytes()) != expected:
            raise ValueError(f"Extract the complete lyric scope fix ZIP: {name}")
    compile(normalized(data), TARGET, "exec")
    before = safe_file(project, TARGET).read_bytes()
    if digest(before) == manifest["after"]:
        return before, before
    if digest(before) != manifest["before"]:
        raise ValueError("Modality source differs from the supplied project. Preserved; do not force overwrite.")
    text = normalized(data).decode("utf-8")
    if b"\r\n" in before:
        text = text.replace("\n", "\r\n")
    after = (b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b"") + text.encode()
    return before, after


def install(project: Path, output: Path) -> dict:
    project, output = project.resolve(), output.resolve()
    if project not in output.parents or output.relative_to(project).parts[0] != "artifacts":
        raise ValueError("Lyric scope backups/reports must stay inside artifacts")
    before, after = preflight(project)
    source = safe_file(project, TARGET)
    changed = before != after
    if source.read_bytes() != before:
        raise ValueError("Modality source changed during installation")
    if changed:
        backup = output / "lyric_scope/backups" / TARGET
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(before)
    try:
        if changed:
            atomic_write(source, after)
        result = dict(schema="context_lyric_scope_install_v1", status="applied",
                      changed_files=[TARGET] if changed else [],
                      source_sha256=dict(before=digest(before), after=digest(after)),
                      fixed_settings_labels_and_corpus=True)
        atomic_write(output / "lyric_scope/install_report.json",
                     (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    except Exception:
        if changed and source.read_bytes() == after:
            atomic_write(source, before)
        raise
    return result
