import copy
import inspect
import json
import unittest
from pathlib import Path

from src.crawler.scripts_py import refine_data
from src.embedding.fixtures.data_songs import _build_song
from src.embedding.fixtures.meta_validation import validate_meta_document
from src.embedding.text.namuwiki_passage import (
    build_namuwiki_song_artifact,
    iter_namuwiki_fact_records,
)
from src.embedding.text.passage_builder import build_dense_passage, build_sparse_passage


class SearchBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.meta = json.loads(
            (Path(__file__).parent / "fixtures/song_meta.json").read_text(encoding="utf-8-sig")
        )
        self.context = {
            "schema_version": "namuwiki_v3",
            "status": "ok",
            "source_url": "https://namu.wiki/w/사랑했나봐(윤도현)",
            "collected_at": "2026-09-15T00:00:00+00:00",
            "facts": [{
                "category": "media_usage",
                "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                "text": "짱구는 못말려 국내판에서 이별 장면의 삽입곡으로 사용되었다.",
            }],
        }

    def test_optional_empty_or_failed_namuwiki_never_rejects_core_song(self):
        for context in (
            {"schema_version": "namuwiki_v1", "status": "no_trivia", "facts": []},
            {"schema_version": "namuwiki_v1", "status": "not_found", "facts": []},
            {"schema_version": "namuwiki_v1", "status": "error", "facts": [], "error_code": "timeout"},
        ):
            meta = copy.deepcopy(self.meta)
            meta["namuwiki"] = context
            self.assertEqual(validate_meta_document(meta, require_media_files=False), [])

    def test_loader_carries_context_without_mixing_song_passages(self):
        meta = copy.deepcopy(self.meta)
        plain = _build_song(meta, Path("/nonexistent/song/meta.json"))
        meta["namuwiki"] = self.context
        enriched = _build_song(meta, Path("/nonexistent/song/meta.json"))
        self.assertEqual(enriched["namuwiki"], self.context)
        self.assertEqual(build_dense_passage(plain), build_dense_passage(enriched))
        self.assertEqual(build_sparse_passage(plain), build_sparse_passage(enriched))

    def test_fact_level_record_contains_q203_clues_and_song_id(self):
        song = {
            "id": "837567",
            "metadata": {"title": "사랑했나봐", "artist": ["윤도현"]},
            "namuwiki": self.context,
        }
        records = list(iter_namuwiki_fact_records(song))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["song_id"], "837567")
        for clue in ("짱구는 못말려", "이별 장면", "삽입곡"):
            self.assertIn(clue, record["dense_text"])
        self.assertTrue(record["record_id"].startswith("nw:837567:"))
        self.assertEqual(records, list(iter_namuwiki_fact_records(song)))
        sparse_terms = build_namuwiki_song_artifact(song)["retrieval"][
            "sparse_profile"
        ]["terms"]
        self.assertIn("짱구는_못말려", sparse_terms)
        self.assertIn("삽입곡", sparse_terms)

    def test_non_ok_or_malformed_context_produces_no_index_records(self):
        for context in ({}, {"status": "no_trivia", "facts": []}, {"status": "ok", "facts": [{}]}):
            song = {"id": "1", "metadata": {}, "namuwiki": context}
            self.assertEqual(list(iter_namuwiki_fact_records(song)), [])

    def test_music_vibe_llm_has_no_namuwiki_input(self):
        self.assertNotIn("namuwiki_data", inspect.signature(refine_data.generate_prompt).parameters)
        self.assertNotIn("namuwiki_data", inspect.signature(refine_data.refine_data).parameters)


if __name__ == "__main__":
    unittest.main()
