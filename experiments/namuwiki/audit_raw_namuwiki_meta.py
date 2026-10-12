"""Read-only aggregate validation of private data/raw/*/meta.json Namuwiki fields.

Optional None for source_url/error_code is valid for some statuses. The report
distinguishes those from invalid or absent required fields. No titles, song
IDs, source URLs or fact text are printed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from pydantic import ValidationError

from src.crawler.context.schemas import NamuwikiMeta


def audit(raw_dir: Path) -> dict:
    if not raw_dir.is_dir():
        raise ValueError("raw song directory is missing")
    status_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    count = 0
    validated = 0
    for folder in sorted(raw_dir.iterdir()):
        if not folder.is_dir():
            continue
        count += 1
        meta_path = folder / "meta.json"
        if folder.is_symlink() or meta_path.is_symlink() or not meta_path.is_file():
            issue_counts["missing_or_unsafe_meta"] += 1
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            issue_counts["unreadable_meta"] += 1
            continue
        context = meta.get("namuwiki") if isinstance(meta, dict) else None
        if not isinstance(context, dict):
            issue_counts["missing_or_nonobject_namuwiki"] += 1
            continue
        status = context.get("status")
        status_counts[str(status) if status is not None else "<missing/null>"] += 1
        for key in ("schema_version", "status"):
            value = context.get(key)
            if not isinstance(value, str) or not value.strip():
                issue_counts[f"required_{key}_missing_or_null"] += 1
            elif value.strip().casefold() == "unknown":
                issue_counts[f"required_{key}_unknown"] += 1
        facts = context.get("facts")
        if isinstance(facts, list):
            for fact in facts:
                if isinstance(fact, dict):
                    for key in ("category", "section", "text"):
                        value = fact.get(key)
                        if not isinstance(value, str) or not value.strip():
                            issue_counts[f"fact_{key}_missing_or_null"] += 1
                        elif value.strip().casefold() == "unknown":
                            issue_counts[f"fact_{key}_unknown"] += 1
        try:
            NamuwikiMeta.model_validate(context)
        except (TypeError, ValueError, ValidationError):
            issue_counts["schema_invalid"] += 1
        else:
            validated += 1
    return {
        "status": "ok" if not issue_counts else "issues_found",
        "raw_song_directories": count,
        "schema_valid_namuwiki_objects": validated,
        "status_counts": dict(sorted(status_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "optional_null_policy": "source_url/error_code may be null when status permits",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(json.dumps(audit(args.raw_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
