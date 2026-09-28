from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.crawler.scripts_py.backfill_namuwiki_context import default_artifact_dir
from src.embedding.context_artifacts import (
    CONTEXT_MANIFEST_VERSION,
    ContextArtifactStore,
    iter_context_artifact_records,
    iter_context_sparse_profiles,
)
from src.embedding.text.context_bm25_tokenizer import tokenize_context_for_bm25
from src.embedding.text.namuwiki_passage import (
    CONTEXT_ARTIFACT_VERSION,
    CONTEXT_RECORD_VERSION,
    ContextSongArtifact,
    build_namuwiki_song_artifact,
)


def song_with(facts: list[dict], *, status: str = "ok") -> dict:
    return {
        "id": "837567",
        "metadata": {
            "title": "사랑했나봐",
            "artist": ["윤도현"],
        },
        "namuwiki": {
            "schema_version": "namuwiki_v3",
            "status": status,
            "source_url": "https://namu.wiki/w/사랑했나봐(윤도현)",
            "collected_at": "2026-09-20T00:00:00+00:00",
            "facts": facts,
        },
    }


class ContextArtifactV6Tests(unittest.TestCase):
    def test_every_source_fact_is_covered_and_embedding_inputs_are_explicit(self):
        facts = [
            {
                "category": "media_usage",
                "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                "text": "정식 발매 전 SBS에서 짱구는 못말려의 배경음악으로 사용됐다.",
            },
            {
                "category": "media_usage",
                "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                "text": "이때 창밖을 보며 눈물을 흘리는 장면과 함께 이 곡이 흐른다.",
            },
            {
                "category": "musical_detail",
                "section": "여담",
                "text": "최고음은 2옥타브 라(A4)이다.",
            },
            {
                "category": "musical_detail",
                "section": "여담",
                "text": "이 음이 곡 내내 반복되어 부르기 까다롭다.",
            },
            {
                "category": "media_usage",
                "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                "text": "정식 발매 전 SBS에서 짱구는 못말려의 배경음악으로 사용됐다.",
            },
        ]

        value = build_namuwiki_song_artifact(song_with(facts))
        artifact = ContextSongArtifact.model_validate(value)
        retrieval = artifact.retrieval

        self.assertEqual(CONTEXT_ARTIFACT_VERSION, "context_artifact_v6")
        self.assertEqual(CONTEXT_RECORD_VERSION, "context_record_v6")
        self.assertEqual(retrieval.source_fact_count, 5)
        self.assertEqual(retrieval.covered_source_fact_count, 5)
        self.assertEqual(
            {index for record in retrieval.records for index in record.source_fact_indices},
            set(range(5)),
        )
        # One exact duplicate and one dependent pitch continuation are merged.
        self.assertEqual(retrieval.record_count, 3)

        media = next(record for record in retrieval.records if record.source_fact_indices == [0, 4])
        self.assertIn("짱구는 못말려", media.dense_text)
        self.assertNotIn("윤도현의 곡 '사랑했나봐'", media.dense_text)
        self.assertIn("짱구는_못말려", media.keywords)
        self.assertNotIn("사랑했나봐", media.keywords)
        self.assertNotIn("윤도현", media.keywords)
        self.assertEqual(len(media.keywords), len(set(media.keywords)))

        profile = retrieval.sparse_profile
        self.assertIsNotNone(profile)
        self.assertEqual(profile.profile_id, "nws:837567")
        self.assertEqual(profile.terms.count("사랑했나봐"), 1)
        self.assertEqual(profile.terms.count("윤도현"), 1)
        self.assertEqual(profile.terms.count("짱구는_못말려"), 1)

        pitch = next(record for record in retrieval.records if record.source_fact_indices == [2, 3])
        self.assertEqual(pitch.quality, "merged_context")
        self.assertIn("2옥타브", pitch.evidence_text)
        self.assertIn("부르기 까다롭다", pitch.evidence_text)

        scene = next(record for record in retrieval.records if record.source_fact_indices == [1])
        self.assertEqual(scene.quality, "section_resolved")
        self.assertTrue(scene.evidence_text.startswith("짱구는 못말려에서"))

    def test_a_curated_fact_is_not_silently_dropped_for_weak_sparse_signal(self):
        facts = [{
            "category": "other",
            "section": "여담",
            "text": "별명과 관련된 짧은 이야기가 전해진다.",
        }]
        artifact = ContextSongArtifact.model_validate(
            build_namuwiki_song_artifact(song_with(facts))
        )
        self.assertEqual(artifact.retrieval.source_fact_count, 1)
        self.assertEqual(artifact.retrieval.covered_source_fact_count, 1)
        self.assertEqual(artifact.retrieval.record_count, 1)
        self.assertIsNotNone(artifact.retrieval.sparse_profile)

    def test_record_keywords_are_bounded_and_low_information_terms_are_removed(self):
        facts = [{
            "category": "media_usage",
            "section": "여담 > 짱구는 못말려 국내판 삽입곡",
            "text": (
                "당시 SBS 담당 PD가 전해성에게 사전에 양해를 구해 "
                "짱구는 못말려 6기 15화의 배경음악으로 사용했고, "
                "이후 투니버스 더빙판에서도 다시 방영되었다."
            ),
        }]
        artifact = ContextSongArtifact.model_validate(
            build_namuwiki_song_artifact(song_with(facts))
        )
        record = artifact.retrieval.records[0]
        self.assertLessEqual(len(record.keywords), 12)
        self.assertNotIn("당시", record.keywords)
        self.assertNotIn("담당", record.keywords)
        self.assertNotIn("사전", record.keywords)
        self.assertIn("짱구는_못말려", record.keywords)
        self.assertIn("15화", record.keywords)
        self.assertIn("투니버스", record.keywords)

    def test_manifest_streams_the_exact_search_inputs(self):
        facts = [{
            "category": "production",
            "section": "여담",
            "text": "앨범 녹음이 끝난 뒤 추가로 녹음해 수록한 곡이다.",
        }]
        song = song_with(facts)
        with TemporaryDirectory() as directory:
            store = ContextArtifactStore(Path(directory))
            with store.writer():
                result = store.sync_song(song)
                manifest = store.publish_manifest(active_song_ids={"837567"})

            self.assertEqual(manifest["schema_version"], CONTEXT_MANIFEST_VERSION)
            self.assertEqual(manifest["retrieval_policy"]["dense_group_by"], "song_id")
            self.assertEqual(manifest["retrieval_policy"]["dense_embedding_unit"], "fact")
            self.assertEqual(manifest["retrieval_policy"]["sparse_embedding_unit"], "song")
            self.assertEqual(result["source_fact_count"], 1)
            self.assertEqual(result["covered_source_fact_count"], 1)

            records = list(iter_context_artifact_records(Path(directory)))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["song_id"], "837567")
            self.assertEqual(records[0]["dense_text"], records[0]["dense_passage"])
            self.assertEqual(records[0]["fact_text"], records[0]["evidence_passage"])

            profiles = list(iter_context_sparse_profiles(Path(directory)))
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0]["profile_id"], "nws:837567")
            self.assertEqual(
                " ".join(profiles[0]["sparse_terms"]),
                profiles[0]["sparse_passage"],
            )

    def test_query_side_context_synonyms_remain_bounded(self):
        terms = tokenize_context_for_bm25("짱구 OST 뮤비 패러디 무대", expand_synonyms=True)
        for expected in ("ost", "삽입곡", "배경음악", "bgm", "뮤직비디오", "mv", "패러디", "무대"):
            self.assertIn(expected, terms)

    def test_default_artifact_path_follows_custom_raw_sandbox(self):
        self.assertEqual(
            default_artifact_dir(Path("/tmp/context-test/raw")),
            Path("/tmp/context-test/artifacts/context"),
        )
        self.assertEqual(
            default_artifact_dir(Path("/app/data/raw")),
            Path("/app/artifacts/context"),
        )


if __name__ == "__main__":
    unittest.main()
