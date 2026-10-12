"""Resolve a fully reviewed needs_review batch as not_found, with safeguards.

This command does not search the web. The operator must have individually
confirmed that every song in the current unresolved queue has no matching
Namuwiki song page. The default mode is read-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.crawler.context.schemas import NAMUWIKI_META_VERSION, NamuwikiMeta


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, value: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _song_id(path: Path, meta: dict) -> str:
    match = re.search(r"_(\d+)$", path.parent.name)
    if not match:
        raise ValueError(f"raw folder has no numeric ID suffix: {path.parent}")
    from_folder = match.group(1)
    from_meta = str(meta.get("song_id") or meta.get("id") or from_folder)
    if from_meta != from_folder:
        raise ValueError(f"folder and meta song IDs differ: {path}")
    return from_folder


def preflight(raw_dir: Path, unresolved_path: Path, expected_count: int) -> tuple[list[dict], dict]:
    if expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if not raw_dir.is_dir():
        raise ValueError(f"raw directory missing: {raw_dir}")
    unresolved = json.loads(unresolved_path.read_text(encoding="utf-8-sig"))
    ids = unresolved.get("needs_review_song_ids")
    if not isinstance(ids, list) or len(ids) != expected_count:
        raise ValueError("needs_review count changed; inspect the queue again")
    wanted = set(str(value) for value in ids)
    if len(wanted) != expected_count or not all(value.isdigit() for value in wanted):
        raise ValueError("needs_review IDs are duplicated or invalid")
    if (unresolved.get("unresolved_count") != expected_count
            or set(unresolved.get("unresolved_song_ids", [])) != wanted
            or any(unresolved.get(key) for key in (
                "collection_error_song_ids", "artifact_pending_song_ids",
                "pending_collection_song_ids",
            ))):
        raise ValueError("unresolved queue contains other work; resolve it separately")
    rows = unresolved.get("songs", [])
    if (not isinstance(rows, list) or len(rows) != expected_count
            or {str(row.get("song_id")) for row in rows} != wanted
            or any(row.get("namuwiki_status") != "needs_review" for row in rows)):
        raise ValueError("unresolved rows do not match the needs_review ID list")

    selected: dict[str, dict] = {}
    for path in raw_dir.glob("*/meta.json"):
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError(f"raw metadata symlink is not allowed: {path}")
        original = path.read_bytes()
        meta = json.loads(original.decode("utf-8-sig"))
        if not isinstance(meta, dict):
            raise ValueError(f"meta must be an object: {path}")
        song_id = _song_id(path, meta)
        if song_id not in wanted:
            continue
        if song_id in selected:
            raise ValueError(f"duplicate raw song ID: {song_id}")
        current = meta.get("namuwiki")
        if not isinstance(current, dict) or current.get("schema_version") != NAMUWIKI_META_VERSION:
            raise ValueError(f"current Namuwiki schema differs: {song_id}")
        validated = NamuwikiMeta.model_validate(current)
        if validated.status != "needs_review" or validated.facts:
            raise ValueError(f"manual decision cannot be applied to this status: {song_id}")

        updated = copy.deepcopy(meta)
        # A reviewed absence has no page and no fact. Keep the original crawl
        # timestamp; the separate private audit log records the review time.
        updated["namuwiki"] = NamuwikiMeta.model_validate({
            **current, "status": "not_found", "source_url": None,
            "facts": [], "error_code": None,
        }).model_dump(exclude_none=True)
        encoded = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        selected[song_id] = {
            "song_id": song_id, "path": path, "original": original,
            "updated": encoded, "old_error_code": current.get("error_code"),
        }
    if set(selected) != wanted:
        raise ValueError(f"raw metadata missing for {len(wanted - set(selected))} reviewed IDs")
    return [selected[song_id] for song_id in sorted(wanted, key=int)], unresolved


def resolve(raw_dir: Path, unresolved_path: Path, backup_root: Path,
            expected_count: int, *, apply: bool, confirm: bool) -> dict:
    entries, unresolved = preflight(raw_dir, unresolved_path, expected_count)
    reasons = dict(Counter(str(entry["old_error_code"]) for entry in entries))
    result = {"status": "dry_run", "reviewed_count": len(entries),
              "old_error_code_counts": reasons,
              "queue_updated_at": unresolved.get("updated_at")}
    if not apply:
        return result
    if not confirm:
        raise ValueError("--apply requires --confirm-all-reviewed-no-page")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / f"needs_review_to_not_found_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    audit = {
        "decision": "all reviewed IDs have no matching Namuwiki song page",
        "review_applied_at": datetime.now(timezone.utc).isoformat(),
        "expected_count": expected_count,
        "queue_updated_at": unresolved.get("updated_at"),
        "items": [],
    }
    for entry in entries:
        backup = backup_dir / f"{entry['song_id']}.meta.json"
        backup.write_bytes(entry["original"])
        audit["items"].append({
            "song_id": entry["song_id"], "meta_path": str(entry["path"]),
            "old_error_code": entry["old_error_code"],
            "old_sha256": _sha256(entry["original"]),
            "new_sha256": _sha256(entry["updated"]),
        })
    (backup_dir / "review_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    changed: list[dict] = []
    try:
        for entry in entries:
            if entry["path"].read_bytes() != entry["original"]:
                raise ValueError("raw metadata changed after preflight; aborting")
            _atomic_write(entry["path"], entry["updated"])
            changed.append(entry)
    except Exception:
        for entry in reversed(changed):
            _atomic_write(entry["path"], entry["original"])
        raise
    return {**result, "status": "applied", "backup_dir": str(backup_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(os.getenv("VAGUEFINDER_DATA_DIR") or "data/raw"))
    parser.add_argument("--unresolved", type=Path, default=Path("artifacts/context/unresolved.json"))
    parser.add_argument("--backup-root", type=Path, default=Path("data/context/manual_review_backups"))
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-all-reviewed-no-page", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = resolve(args.raw_dir, args.unresolved, args.backup_root,
                         args.expected_count, apply=args.apply,
                         confirm=args.confirm_all_reviewed_no_page)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
