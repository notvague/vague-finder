import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.crawler.scripts_py.audit_namuwiki_context import run_audit
from src.embedding.context_artifacts import ContextArtifactState


class NamuwikiAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "data/raw"
        self.cache = self.root / "data/context"
        self.artifacts = self.root / "artifacts/context"
        self.output = self.root / "audit"

    def write_song(self, song_id, title, status, *, error_code=None, facts=None, source_url=None):
        folder = self.raw / f"가수_{title}_{song_id}"
        folder.mkdir(parents=True)
        context = {
            "schema_version": "namuwiki_v3",
            "status": status,
            "collected_at": "2026-09-22T00:00:00+00:00",
            "facts": facts or [],
        }
        if error_code:
            context["error_code"] = error_code
        if source_url:
            context["source_url"] = source_url
        meta = {
            "song_id": str(song_id),
            "metadata": {
                "title": title,
                "artist": ["가수"],
                "album": "앨범",
                "release_date": "2020.01.01",
            },
            "community_feedback": {
                "popularity": {"youtube_view_count": song_id},
            },
            "namuwiki": context,
        }
        (folder / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False),
            encoding="utf-8",
        )

    def args(self):
        return Namespace(
            data_dir=self.raw,
            cache_dir=self.cache,
            artifact_dir=self.artifacts,
            output_dir=self.output,
            sample_per_review_code=5,
            not_found_sample=5,
            resolved_sample=5,
            strict=False,
        )

    def test_outputs_full_inventory_grouped_review_and_bounded_manual_queue(self):
        self.write_song(
            1,
            "정상곡",
            "ok",
            source_url="https://namu.wiki/w/정상곡(가수)",
            facts=[{
                "category": "production",
                "section": "여담",
                "text": "가수가 직접 작곡하고 녹음한 곡으로 알려져 있다.",
            }],
        )
        self.write_song(
            2,
            "검토곡",
            "needs_review",
            error_code="title_only_page_artist_not_verified",
        )
        self.write_song(3, "없는곡", "not_found")

        def inspect(_store, meta):
            status = meta["namuwiki"]["status"]
            facts = meta["namuwiki"]["facts"]
            if status in {"ok", "not_found"}:
                return ContextArtifactState(
                    applicable=True,
                    ready=True,
                    reason="artifact_current",
                    record_count=len(facts),
                    source_fact_count=len(facts),
                    status=status,
                )
            return ContextArtifactState(
                applicable=False,
                ready=False,
                reason="not_terminal_or_invalid_source:ValueError",
                status=status,
            )

        with patch(
            "src.crawler.scripts_py.audit_namuwiki_context.ContextArtifactStore.inspect",
            autospec=True,
            side_effect=inspect,
        ):
            summary = run_audit(self.args())

        self.assertEqual(summary["song_count"], 3)
        self.assertEqual(
            summary["status_counts"],
            {"ok": 1, "needs_review": 1, "not_found": 1},
        )
        self.assertEqual(
            summary["needs_review_error_counts"],
            {"title_only_page_artist_not_verified": 1},
        )
        inventory = (self.output / "inventory.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(inventory), 3)
        grouped = json.loads(
            (self.output / "needs_review_groups.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            grouped["groups"]["title_only_page_artist_not_verified"][0]["song_id"],
            "2",
        )
        with (self.output / "manual_review.csv").open(encoding="utf-8-sig", newline="") as stream:
            manual = list(csv.DictReader(stream))
        self.assertEqual({row["song_id"] for row in manual}, {"1", "2", "3"})
        self.assertTrue(all("decision" in row for row in manual))

    def test_legacy_title_only_binding_is_queued_for_current_recheck(self):
        self.write_song(
            10,
            "동명곡",
            "no_trivia",
            source_url="https://namu.wiki/w/동명곡",
        )
        runs = self.cache / "raw_runs"
        runs.mkdir(parents=True)
        (runs / "old.json").write_text(json.dumps({
            "finished_at": "2026-09-21T00:00:00+00:00",
            "audits": [{
                "song_id": "10",
                "attempts": [{
                    "source_url": "https://namu.wiki/w/동명곡",
                    "identity": {"verified": True, "reason": "title_only_match"},
                }],
            }],
        }, ensure_ascii=False), encoding="utf-8")

        with patch(
            "src.crawler.scripts_py.audit_namuwiki_context.ContextArtifactStore.inspect",
            autospec=True,
            return_value=ContextArtifactState(
                applicable=True,
                ready=True,
                reason="artifact_current",
                status="no_trivia",
            ),
        ):
            run_audit(self.args())

        row = json.loads(
            (self.output / "inventory.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertEqual(row["binding_contract"], "legacy")
        self.assertEqual(row["risk_level"], "high")
        self.assertIn("identity_binding_requires_current_recheck", row["issues"])


if __name__ == "__main__":
    unittest.main()
