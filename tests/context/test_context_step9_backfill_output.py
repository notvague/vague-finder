"""The PowerShell adapter reports coverage without emitting private titles."""
from __future__ import annotations

import json

from experiments.namuwiki import context_step9_backfill as adapter


def test_ascii_summary_omits_song_titles_and_keeps_completion_counts(monkeypatch, capsys):
    def fake_backfill(args):
        assert args.artifacts_only and args.dry_run and args.limit == 100
        assert args.quiet_progress and not args.force
        return {
            "dry_run": True,
            "catalog_progress": {"completed_before": 3016, "completed_after": 3016,
                                 "scope_total": 3016, "pending_before": 0, "pending_after": 0},
            "selected_count": 0,
            "artifact_manifest": {},
            "songs": [{"title": '곡 제목에 "따옴표"', "status": "ok", "action": "skip"}],
        }

    monkeypatch.setattr(adapter, "run_backfill", fake_backfill)
    assert adapter.main(["--dry-run", "--limit", "100"]) == 0
    raw = capsys.readouterr().out
    assert "곡 제목" not in raw and all(ord(char) < 128 for char in raw)
    report = json.loads(raw)
    assert report["catalog_progress"]["pending_before"] == 0
    assert report["selected_count"] == 0
