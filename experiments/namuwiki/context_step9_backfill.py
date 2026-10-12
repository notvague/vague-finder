"""Compact ASCII JSON status for PowerShell-driven Context artifact batches.

The backfill CLI deliberately prints individual song titles and paths. Windows
PowerShell 5.1 can decode that UTF-8 stream through a legacy console code page,
making the resulting JSON invalid. This adapter never prints song metadata.
"""
from __future__ import annotations

import argparse
import json

from src.crawler.scripts_py.backfill_namuwiki_context import make_parser, run_backfill


def summarize(report: dict) -> dict:
    manifest = report.get("artifact_manifest") or {}
    bad = [str(row.get("song_id", "")) for row in report.get("songs", [])
           if row.get("action") in {"error", "artifact_error"}
           or (row.get("status") == "error" and row.get("action") not in {"deferred", "queued"})]
    return {
        "status": "error" if bad or manifest.get("invalid_artifact_count", 0) else "ok",
        "dry_run": report["dry_run"],
        "catalog_progress": report["catalog_progress"],
        "selected_count": report["selected_count"],
        "artifact_manifest": {
            key: manifest.get(key) for key in (
                "song_count", "record_count", "sparse_profile_count",
                "invalid_artifact_count", "orphan_artifact_count",
            )
        },
        "unresolved": report.get("unresolved") or {},
        "failed_song_ids": bad,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    options = ["--artifacts-only", "--quiet-progress", "--limit", str(args.limit)]
    if args.dry_run:
        options.append("--dry-run")
    try:
        report = summarize(run_backfill(make_parser().parse_args(options)))
        print(json.dumps(report, ensure_ascii=True, separators=(",", ":")))
        return 2 if report["status"] == "error" else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
