"""Resolve individually reviewed NamuWiki misses without inventing search facts.

The private CSV is a decision ledger, not a list of automatically inferred
404s. Every decision is tied to the exact raw metadata bytes seen at export.
The ordinary backfill owns artifact generation after these status changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

from src.crawler.context.schemas import NAMUWIKI_META_VERSION, NamuwikiMeta, now
from src.crawler.context.store import Store
from src.crawler.scripts_py.backfill_namuwiki_context import (
    ROOT,
    identify,
    merge_namuwiki,
    read_meta,
    write_meta,
)
from src.embedding.fixtures.meta_validation import validate_meta_document

DECISION_CODE = "manual_review_no_matching_song_page"
FIELDS = (
    "song_id", "title", "artists", "previous_error_code", "attempted_url",
    "meta_sha256", "decision", "review_note",
)


def _catalogue(data_dir: Path) -> tuple[dict[str, tuple[Path, bytes, dict]], int]:
    if not data_dir.is_dir():
        raise ValueError(f"raw directory not found: {data_dir}")
    songs: dict[str, tuple[Path, bytes, dict]] = {}
    skipped = 0
    for path in sorted(data_dir.glob("*/meta.json")):
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError(f"refusing symlinked raw metadata: {path}")
        raw, meta = read_meta(path)
        song_id, _, _ = identify(meta, path.parent)
        if song_id in songs:
            raise ValueError(f"duplicate raw song_id: {song_id}")
        songs[song_id] = (path, raw, meta)
    eligible = {}
    for song_id, item in songs.items():
        path, _, meta = item
        if validate_meta_document(meta, song_dir=path.parent, require_media_files=True):
            skipped += 1
            continue
        eligible[song_id] = item
    return eligible, skipped


def _review_queue(songs: dict) -> dict[str, tuple[Path, bytes, dict]]:
    queue = {}
    for song_id, item in songs.items():
        context = item[2].get("namuwiki")
        if not isinstance(context, dict) or context.get("status") != "needs_review":
            continue
        parsed = NamuwikiMeta.model_validate(context)
        if parsed.schema_version != NAMUWIKI_META_VERSION:
            raise ValueError(f"outdated context schema for song_id={song_id}")
        queue[song_id] = item
    return queue


def _check_paths(data_dir: Path, cache_dir: Path, csv_path: Path) -> None:
    data_dir, cache_dir, csv_path = (path.resolve() for path in (data_dir, cache_dir, csv_path))
    if cache_dir == data_dir or cache_dir.is_relative_to(data_dir):
        raise ValueError("cache-dir must be outside raw")
    if csv_path == data_dir or csv_path.is_relative_to(data_dir):
        raise ValueError("review CSV must be outside raw")


def _export(songs: dict, destination: Path, *, overwrite: bool = False) -> dict:
    if destination.is_symlink() or (destination.exists() and not overwrite):
        raise ValueError(f"review CSV already exists; refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for song_id, (path, raw, meta) in sorted(_review_queue(songs).items(), key=lambda item: int(item[0])):
        _, title, artists = identify(meta, path.parent)
        context = meta["namuwiki"]
        writer.writerow({
            "song_id": song_id,
            "title": title,
            "artists": " / ".join(artists),
            "previous_error_code": context.get("error_code") or "",
            "attempted_url": context.get("source_url") or "",
            "meta_sha256": hashlib.sha256(raw).hexdigest(),
            "decision": "",
            "review_note": "",
        })
    handle, temporary = tempfile.mkstemp(prefix=".review-", suffix=".csv", dir=destination.parent)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(b"\xef\xbb\xbf" + stream.getvalue().encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        # No silent loss of somebody else's manual edits between preflight and replace.
        if destination.exists() and not overwrite:
            raise ValueError(f"review CSV appeared during export: {destination}")
        os.replace(temporary, destination)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return {"status": "exported", "review_count": len(_review_queue(songs)), "file": str(destination)}


def _read_decisions(path: Path) -> list[dict[str, str]]:
    if path.is_symlink():
        raise ValueError("refusing symlinked review CSV")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != set(FIELDS):
            raise ValueError("review CSV columns changed; re-export before editing")
        rows = list(reader)
    ids = set()
    for row in rows:
        if None in row or any(row.get(field) is None for field in FIELDS):
            raise ValueError("review CSV row has missing/extra cells")
        song_id = row["song_id"].strip()
        if not re.fullmatch(r"\d+", song_id) or song_id in ids:
            raise ValueError(f"invalid or duplicate review song_id: {song_id}")
        ids.add(song_id)
        if row["decision"].strip() != "not_found":
            raise ValueError(f"song_id={song_id}: explicit decision=not_found required")
        if len(row["review_note"].strip()) < 12:
            raise ValueError(f"song_id={song_id}: describe the manual title/artist verification")
        if not re.fullmatch(r"[0-9a-f]{64}", row["meta_sha256"].strip()):
            raise ValueError(f"song_id={song_id}: invalid metadata SHA-256")
    return rows


def _preflight(songs: dict, rows: list[dict], store: Store) -> tuple[list[tuple], int]:
    row_ids = {row["song_id"].strip() for row in rows}
    missing = set(_review_queue(songs)) - row_ids
    extra = row_ids - set(songs)
    if missing or extra:
        raise ValueError(f"review CSV does not match eligible queue: missing={len(missing)} extra={len(extra)}")
    pending = []
    already = 0
    for row in rows:
        song_id = row["song_id"].strip()
        path, raw, meta = songs[song_id]
        expected_hash = row["meta_sha256"].strip()
        original = hashlib.sha256(raw).hexdigest()
        receipt = store.path(f"manual_resolutions/{song_id}.json")
        if receipt.exists():
            saved = store.read(f"manual_resolutions/{song_id}.json")
            if (saved.get("original_meta_sha256") != expected_hash
                    or saved.get("review_note") != row["review_note"].strip()
                    or saved.get("decision") != "not_found"):
                raise ValueError(f"song_id={song_id}: existing review receipt conflicts with CSV")
        context = meta.get("namuwiki")
        if not isinstance(context, dict):
            raise ValueError(f"song_id={song_id}: context disappeared")
        if context.get("status") == "needs_review" and original == expected_hash:
            NamuwikiMeta.model_validate(context)
            pending.append((song_id, path, raw, meta, row))
            continue
        # Allow exact continuation after a partially completed run. write_meta
        # always saves the original bytes before replacing the raw metadata.
        backup_ref = f"meta_backups/{song_id}/{expected_hash}.json"
        backup = store.path(backup_ref)
        if context.get("status") != "not_found" or context.get("error_code") != DECISION_CODE or not backup.is_file():
            raise ValueError(f"song_id={song_id}: metadata changed since CSV export")
        backup_bytes = backup.read_bytes()
        if hashlib.sha256(backup_bytes).hexdigest() != expected_hash:
            raise ValueError(f"song_id={song_id}: original metadata backup is corrupt")
        original_meta = json.loads(backup_bytes.decode("utf-8-sig"))
        if original_meta.get("namuwiki", {}).get("status") != "needs_review":
            raise ValueError(f"song_id={song_id}: backup is not a pending review")
        NamuwikiMeta.model_validate(context)
        projection = NamuwikiMeta(
            status="not_found", source_url=None, facts=[], error_code=DECISION_CODE,
            collected_at=context["collected_at"],
        ).model_dump(exclude_none=True)
        if merge_namuwiki(original_meta, projection) != meta:
            raise ValueError(f"song_id={song_id}: finalized metadata was modified")
        already += 1
    return pending, already


def execute(*, command: str, data_dir: Path, cache_dir: Path, csv_path: Path, overwrite: bool = False) -> dict:
    _check_paths(data_dir, cache_dir, csv_path)
    songs, skipped = _catalogue(data_dir)
    if command == "export":
        result = _export(songs, csv_path, overwrite=overwrite)
        return {**result, "eligible_songs": len(songs), "core_invalid_skipped": skipped}
    rows = _read_decisions(csv_path)
    store = Store(cache_dir)
    pending, already = _preflight(songs, rows, store)
    if command == "validate":
        return {"status": "validated", "decisions": len(rows), "pending": len(pending),
                "already_applied": already, "eligible_songs": len(songs), "core_invalid_skipped": skipped}
    if command != "apply":
        raise ValueError(f"unknown command: {command}")
    applied = 0
    with store.writer():
        # Refresh under the same lock used by the crawler. Per-song writes
        # also check the original bytes and fail on concurrent raw edits.
        songs, _ = _catalogue(data_dir)
        pending, already = _preflight(songs, rows, store)
        for song_id, path, raw, meta, row in pending:
            value = NamuwikiMeta(
                status="not_found", source_url=None, facts=[],
                error_code=DECISION_CODE, collected_at=now(),
            ).model_dump(exclude_none=True)
            backup = write_meta(path, raw, merge_namuwiki(meta, value), store, song_id)
            applied += 1
            store.write(f"manual_resolutions/{song_id}.json", {
                "song_id": song_id,
                "decision": "not_found",
                "review_note": row["review_note"].strip(),
                "original_meta_sha256": row["meta_sha256"].strip(),
                "original_backup": backup,
                "applied_at": value["collected_at"],
            })
        # If interrupted after write_meta and before its receipt, a retry
        # reconstructs the receipt only after validating the exact backup.
        for row in rows:
            song_id = row["song_id"].strip()
            receipt = store.path(f"manual_resolutions/{song_id}.json")
            if not receipt.exists():
                _, current = read_meta(songs[song_id][0])
                store.write(f"manual_resolutions/{song_id}.json", {
                    "song_id": song_id, "decision": "not_found",
                    "review_note": row["review_note"].strip(),
                    "original_meta_sha256": row["meta_sha256"].strip(),
                    "original_backup": f"meta_backups/{song_id}/{row['meta_sha256'].strip()}.json",
                    "applied_at": current["namuwiki"]["collected_at"],
                })
    return {"status": "applied", "decisions": len(rows), "applied": applied,
            "already_applied": already, "eligible_songs": len(songs), "core_invalid_skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export and apply reviewed NamuWiki absences")
    parser.add_argument("command", choices=("export", "validate", "apply"))
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("VAGUEFINDER_DATA_DIR") or ROOT / "data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/context")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/context/review_decisions_step9.csv")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing review CSV only on export")
    args = parser.parse_args(argv)
    try:
        if args.overwrite and args.command != "export":
            raise ValueError("--overwrite applies only to export")
        result = execute(command=args.command, data_dir=args.data_dir, cache_dir=args.cache_dir,
                         csv_path=args.csv, overwrite=args.overwrite)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
