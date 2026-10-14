"""Bind reviewed needs_review songs to the page a person confirmed, with safeguards.

The counterpart of ``resolve_manual_not_found``: here the operator has opened
each page and confirmed that it IS the song's page (or names the one section
that is). This command does not search the web. It reads the already cached
snapshot, extracts trivia from the confirmed scope with the normal extractor
and writes the same compact ``namuwiki`` projection a verified crawl would.
The default mode is read-only.

Decisions file (private, do not commit) — a JSON list:

    [{"song_id": "123", "url": "<requested URL as cached>", "scope": "page"},
     {"song_id": "456", "url": "...", "scope": "section", "section": ["수록곡", "곡명"]},
     {"song_id": "789", "url": "...", "scope": "none"}]

``page``    the whole document is about this song
``section`` only this heading path (headings without their ordinals)
``none``    the page is right but has no part that belongs to this song alone
            (an artist page listing it) — recorded as a page without trivia
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.crawler.context.document_parser import parse_document, section_path
from src.crawler.context.fetcher import article_url
from src.crawler.context.schemas import (
    NAMUWIKI_META_VERSION,
    NamuwikiMeta,
    ParsedDocument,
    SourceSnapshot,
)
from src.crawler.context.store import Store
from src.crawler.context.trivia import _clean_heading, extract_trivia_facts
from src.crawler.scripts_py.backfill_namuwiki_context import merge_namuwiki, write_meta

SCOPES = {"page", "section", "none"}


def _song_id(path: Path, meta: dict) -> str:
    match = re.search(r"_(\d+)$", path.parent.name)
    if not match:
        raise ValueError(f"raw folder has no numeric ID suffix: {path.parent}")
    from_folder = match.group(1)
    from_meta = str(meta.get("song_id") or meta.get("id") or from_folder)
    if from_meta != from_folder:
        raise ValueError(f"folder and meta song IDs differ: {path}")
    return from_folder


def _scope_section(document: ParsedDocument, wanted: list[str]) -> str:
    """Return the one section whose heading path equals the reviewed path."""
    target = [_clean_heading(part) for part in wanted]
    matches = [
        section.section_id
        for section in document.sections
        if section.heading
        and [_clean_heading(part) for part in section_path(document, section.section_id)] == target
    ]
    if len(matches) != 1:
        raise ValueError(
            f"reviewed section path matches {len(matches)} sections: {' > '.join(wanted)}"
        )
    return matches[0]


def _load_decisions(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("decisions must be a non-empty JSON list")
    decisions: dict[str, dict] = {}
    for row in rows:
        song_id = str(row.get("song_id") or "")
        scope = row.get("scope")
        if not song_id.isdigit() or song_id in decisions:
            raise ValueError(f"decision song IDs are duplicated or invalid: {song_id!r}")
        if scope not in SCOPES or not isinstance(row.get("url"), str) or not row["url"].strip():
            raise ValueError(f"decision needs a url and a scope of {sorted(SCOPES)}: {song_id}")
        section = row.get("section")
        if (scope == "section") != (isinstance(section, list) and bool(section)):
            raise ValueError(f"'section' is required for, and only for, scope 'section': {song_id}")
        decisions[song_id] = row
    return decisions


def preflight(raw_dir: Path, cache_dir: Path, decisions_path: Path) -> list[dict]:
    if not raw_dir.is_dir():
        raise ValueError(f"raw directory missing: {raw_dir}")
    decisions = _load_decisions(decisions_path)
    store = Store(cache_dir)
    entries: dict[str, dict] = {}
    for path in raw_dir.glob("*/meta.json"):
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError(f"raw metadata symlink is not allowed: {path}")
        original = path.read_bytes()
        meta = json.loads(original.decode("utf-8-sig"))
        if not isinstance(meta, dict):
            raise ValueError(f"meta must be an object: {path}")
        song_id = _song_id(path, meta)
        if song_id not in decisions:
            continue
        if song_id in entries:
            raise ValueError(f"duplicate raw song ID: {song_id}")
        current = meta.get("namuwiki")
        if not isinstance(current, dict) or current.get("schema_version") != NAMUWIKI_META_VERSION:
            raise ValueError(f"current Namuwiki schema differs: {song_id}")
        validated = NamuwikiMeta.model_validate(current)
        if validated.status != "needs_review" or validated.facts:
            raise ValueError(f"manual binding cannot be applied to this status: {song_id}")

        decision = decisions[song_id]
        snapshot_ref = store.cached(article_url(decision["url"]))
        if snapshot_ref is None:
            raise ValueError(f"reviewed URL is not in the snapshot cache: {song_id}")
        # Parse in memory: the preview must not write into the snapshot cache.
        snapshot = SourceSnapshot.model_validate(store.read(snapshot_ref))
        document = parse_document(
            store.raw(snapshot),
            snapshot.snapshot_id,
            snapshot.final_url or snapshot.requested_url,
        )
        if document.parse_status == "structure_error" or not document.page_title:
            raise ValueError(f"cached page cannot be parsed: {song_id}")

        scope_id = None
        if decision["scope"] == "section":
            scope_id = _scope_section(document, decision["section"])
        facts: list[dict] = []
        if decision["scope"] != "none":
            facts, _ = extract_trivia_facts(document, scope_section_id=scope_id)
        source_url = document.canonical_url or article_url(decision["url"])
        value = NamuwikiMeta(
            schema_version=NAMUWIKI_META_VERSION,
            status="ok" if facts else "no_trivia",
            source_url=source_url,
            facts=facts,
        ).model_dump(exclude_none=True)
        entries[song_id] = {
            "song_id": song_id, "path": path, "original": original, "meta": meta,
            "value": value, "page_title": document.page_title,
            "scope": decision["scope"], "section": decision.get("section"),
            "old_error_code": current.get("error_code"),
        }
    if set(entries) != set(decisions):
        raise ValueError(f"raw metadata missing for {len(set(decisions) - set(entries))} reviewed IDs")
    return [entries[song_id] for song_id in sorted(entries, key=int)]


def resolve(raw_dir: Path, cache_dir: Path, decisions_path: Path, backup_root: Path,
            *, apply: bool, confirm: bool) -> dict:
    entries = preflight(raw_dir, cache_dir, decisions_path)
    items = [{
        "song_id": entry["song_id"], "page_title": entry["page_title"],
        "scope": entry["scope"], "section": entry["section"],
        "status": entry["value"]["status"], "fact_count": len(entry["value"]["facts"]),
        "source_url": entry["value"]["source_url"], "old_error_code": entry["old_error_code"],
    } for entry in entries]
    result = {
        "status": "dry_run", "reviewed_count": len(entries),
        "status_counts": dict(Counter(item["status"] for item in items)),
        "fact_count": sum(item["fact_count"] for item in items), "items": items,
    }
    if not apply:
        return result
    if not confirm:
        raise ValueError("--apply requires --confirm-all-reviewed-pages")

    store = Store(cache_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = backup_root / f"needs_review_to_bound_{stamp}"
    audit_dir.mkdir(parents=True, exist_ok=False)
    (audit_dir / "review_manifest.json").write_text(json.dumps({
        "decision": "each reviewed ID is bound to the page a person confirmed",
        "review_applied_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for entry in entries:
        (audit_dir / f"{entry['song_id']}.meta.json").write_bytes(entry["original"])

    changed: list[dict] = []
    try:
        for entry in entries:
            updated = merge_namuwiki(entry["meta"], entry["value"])
            write_meta(entry["path"], entry["original"], updated, store, entry["song_id"])
            changed.append(entry)
    except Exception:
        for entry in reversed(changed):
            entry["path"].write_bytes(entry["original"])
        raise
    return {**result, "status": "applied", "backup_dir": str(audit_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path(os.getenv("VAGUEFINDER_DATA_DIR") or "data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/context"))
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, default=Path("data/context/manual_review_backups"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-all-reviewed-pages", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = resolve(args.raw_dir, args.cache_dir, args.decisions, args.backup_root,
                         apply=args.apply, confirm=args.confirm_all_reviewed_pages)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
