from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.crawler.scripts_py.backfill_namuwiki_context import run_backfill
from src.embedding.context_artifacts import (
    ContextArtifactStore,
    iter_context_artifact_records,
)
from src.embedding.text.namuwiki_passage import (
    ContextSongArtifact,
    build_namuwiki_song_artifact,
    embedding_record_projection,
)


def song(
    song_id="837567",
    *,
    title="사랑했나봐",
    artist="윤도현",
    status="ok",
    facts=None,
):
    return {
        "song_id": song_id,
        "metadata": {"title": title, "artist": [artist]},
        "namuwiki": {
            "schema_version": "namuwiki_v3",
            "status": status,
            "source_url": (
                f"https://namu.wiki/w/{title}({artist})"
                if status in {"ok", "no_trivia"}
                else None
            ),
            "collected_at": "2026-09-16T00:00:00+00:00",
            "facts": facts or [],
        },
    }


def args(root: Path, *, dry_run=False, song_ids=None):
    return argparse.Namespace(
        data_dir=root / "raw",
        cache_dir=root / "cache",
        artifact_dir=root / "artifacts/context",
        url_map=None,
        song_ids=song_ids,
        retry_not_found=False,
        force=False,
        limit=None,
        dry_run=dry_run,
        html_dir=None,
        offline=False,
        refresh=False,
        interval=0.0,
        timeout=1.0,
        no_render=True,
        render_timeout=1.0,
        max_requests=1,
        quiet_progress=True,
    )


def write_meta(root: Path, payload: dict) -> Path:
    song_id = str(payload["song_id"])
    title = payload["metadata"]["title"]
    artist = payload["metadata"]["artist"][0]
    folder = root / "raw" / f"{artist}_{title}_{song_id}"
    folder.mkdir(parents=True)
    path = folder / "meta.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class ContextArtifactV6RecordTests(unittest.TestCase):
    def test_one_compact_record_per_fact_and_stable_identity(self):
        payload = song(
            facts=[
                {
                    "category": "media_usage",
                    "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                    "text": (
                        "짱구는 못말려 6기 15화에서 창밖을 보며 눈물을 흘리는 "
                        "나미리 선생님과 함께 이 곡이 흐른다."
                    ),
                }
            ]
        )
        first = build_namuwiki_song_artifact(payload)
        second = build_namuwiki_song_artifact(payload)
        self.assertEqual(first, second)
        self.assertEqual(first["artifact_schema_version"], "context_artifact_v6")
        self.assertEqual(first["song"]["title"], "사랑했나봐")
        self.assertEqual(first["song"]["artists"], ["윤도현"])
        self.assertEqual(first["source"]["schema_version"], "namuwiki_v3")
        self.assertEqual(first["retrieval"]["record_count"], 1)
        self.assertEqual(first["retrieval"]["dense_embedding_unit"], "fact")
        self.assertEqual(first["retrieval"]["sparse_embedding_unit"], "song")
        self.assertEqual(
            first["retrieval"]["dense_text_format"],
            "artist_title_category_fact_v1",
        )
        self.assertEqual(first["retrieval"]["score_aggregation"], "max_per_song")
        record = first["retrieval"]["records"][0]
        self.assertEqual(record["source_fact_indices"], [0])
        self.assertEqual(record["section"], "여담 > 짱구는 못말려 국내판 삽입곡")
        self.assertTrue(record["dense_text"].startswith(
            "윤도현 - 사랑했나봐 | 매체 사용: "
        ))
        self.assertIn("나미리 선생님과 함께 이 곡이 흐른다", record["dense_text"])
        self.assertEqual(record["evidence_text"], record["dense_text"].split(": ", 1)[1])
        self.assertNotIn("title", record)
        self.assertNotIn("artists", record)
        self.assertNotIn("source_url", record)
        self.assertNotIn("sparse_terms", record)
        sparse_terms = first["retrieval"]["sparse_profile"]["terms"]
        self.assertEqual(len(sparse_terms), len(set(sparse_terms)))
        for term in ("짱구는_못말려", "삽입곡"):
            self.assertIn(term, sparse_terms)
        for unsupported in ("ost", "bgm", "여담"):
            self.assertNotIn(unsupported, sparse_terms)
        self.assertEqual(sparse_terms[:2], ["사랑했나봐", "윤도현"])
        for term in ("6기", "15화"):
            self.assertIn(term, sparse_terms)

        projection = embedding_record_projection(
            ContextSongArtifact.model_validate(first),
            ContextSongArtifact.model_validate(first).retrieval.records[0],
        )
        self.assertTrue(projection["dense_passage"].startswith(
            "윤도현 - 사랑했나봐 | 매체 사용: "
        ))
        self.assertNotIn("sparse_passage", projection)

    def test_identity_aliases_are_exact_and_do_not_cross_fields(self):
        payload = song(
            song_id="1698598",
            title="거짓말",
            artist="BIGBANG (빅뱅)",
            facts=[{
                "category": "record",
                "section": "여담",
                "text": "2016년 설문조사에서 역대 보이그룹 노래 1위를 기록했다.",
            }],
        )
        artifact = build_namuwiki_song_artifact(payload)
        terms = artifact["retrieval"]["sparse_profile"]["terms"]
        self.assertEqual(terms[:3], ["거짓말", "bigbang", "빅뱅"])
        self.assertNotIn("거짓말_bigbang", terms)
        self.assertNotIn("bigbang_(빅뱅", terms)

    def test_context_dependencies_merge_only_when_required(self):
        payload = song(
            facts=[
                {
                    "category": "musical_detail",
                    "section": "여담",
                    "text": "최고음은 2옥타브 라(A4).",
                },
                {
                    "category": "musical_detail",
                    "section": "여담",
                    "text": "이 음이 곡 내내 반복되어 부르기 까다롭다.",
                },
                {
                    "category": "media_usage",
                    "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                    "text": "이때 창밖을 보며 눈물을 흘리는 나미리 선생님과 함께 이 곡이 흐른다.",
                },
                {
                    "category": "meme",
                    "section": "여담 > 밈화",
                    "text": "곰돌이 푸가 강남스타일을 추는 영상에 본 곡을 합성한 밈이 유행했다.",
                },
                {
                    "category": "meme",
                    "section": "여담 > 밈화",
                    "text": "메타톤 잼민이와 비슷한 결의 밈이다.",
                },
                {
                    "category": "meme",
                    "section": "여담 > 밈화",
                    "text": "영상뿐만 아니라 챌린지로도 확장되었다.",
                },
            ]
        )
        records = build_namuwiki_song_artifact(payload)["retrieval"]["records"]
        self.assertEqual(
            [row["source_fact_indices"] for row in records],
            [[0, 1], [2], [3, 5], [4]],
        )
        self.assertIn("최고음은 2옥타브 라(A4)", records[0]["dense_text"])
        self.assertIn("짱구는 못말려에서 창밖을 보며", records[1]["dense_text"])
        self.assertNotIn("이때 창밖", records[1]["dense_text"])
        self.assertIn("메타톤_잼민이", records[3]["keywords"])

    def test_generic_media_usage_does_not_inherit_unsupported_ost_terms(self):
        payload = song(
            song_id="1698598",
            title="거짓말",
            artist="BIGBANG",
            facts=[
                {
                    "category": "media_usage",
                    "section": "여담",
                    "text": "금영노래방 46034, TJ노래방 18553번 곡으로 수록되었다.",
                }
            ],
        )
        retrieval = build_namuwiki_song_artifact(payload)["retrieval"]
        record = retrieval["records"][0]
        self.assertIn("매체 사용", record["dense_text"])
        self.assertNotIn("OST", record["dense_text"])
        self.assertTrue(
            {"ost", "bgm", "삽입곡", "배경음악"}.isdisjoint(
                set(retrieval["sparse_profile"]["terms"])
            )
        )

    def test_sparse_deduplicates_and_removes_identity_and_generic_terms(self):
        payload = song(
            facts=[
                {
                    "category": "influence",
                    "section": "여담",
                    "text": (
                        "당시 윤도현은 이후 사랑했나봐 영상 시청 이후 "
                        "윤도현의 대표곡으로 사랑했나봐를 언급했다."
                    ),
                }
            ]
        )
        retrieval = build_namuwiki_song_artifact(payload)["retrieval"]
        terms = retrieval["sparse_profile"]["terms"]
        self.assertEqual(len(terms), len(set(terms)))
        for noisy in ("당시", "이후", "영상", "시청", "여담"):
            self.assertNotIn(noisy, terms)
        self.assertEqual(terms[:2], ["사랑했나봐", "윤도현"])

    def test_exact_duplicate_fact_is_indexed_once(self):
        fact = {
            "category": "record",
            "section": "여담",
            "text": "2016년 설문조사에서 역대 보이그룹 노래 1위를 기록했다.",
        }
        artifact = build_namuwiki_song_artifact(song(facts=[fact, dict(fact)]))
        self.assertEqual(artifact["retrieval"]["record_count"], 1)

    def test_non_terminal_or_invalid_v3_refuses_artifact(self):
        with self.assertRaises(ValueError):
            build_namuwiki_song_artifact(
                song(status="error", facts=[])
            )
        missing_time = song(status="not_found")
        missing_time["namuwiki"].pop("collected_at")
        with self.assertRaisesRegex(ValueError, "collected_at"):
            build_namuwiki_song_artifact(missing_time)


class ContextArtifactStoreTests(unittest.TestCase):
    def test_path_traversal_and_final_symlink_are_rejected(self):
        payload = song(status="not_found")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ContextArtifactStore(root)
            with self.assertRaises(ValueError):
                store.path("../escape.json")
            (root / "songs").mkdir()
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "songs/837567.json"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "symlink"):
                store.read("songs/837567.json")
            with store.writer():
                with self.assertRaisesRegex(ValueError, "symlink"):
                    store.sync_song(payload)

    def test_atomic_idempotent_sync_manifest_and_stream_loader(self):
        payload = song(
            facts=[
                {
                    "category": "production",
                    "section": "여담",
                    "text": (
                        "처음에는 예정에 없던 노래였지만 추가 녹음으로 "
                        "앨범에 수록되었다."
                    ),
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ContextArtifactStore(root)
            with store.writer():
                created = store.sync_song(payload)
                path = store.path(created["artifact_ref"])
                first_mtime = path.stat().st_mtime_ns
                unchanged = store.sync_song(payload)
                self.assertEqual(unchanged["artifact_action"], "unchanged")
                self.assertEqual(path.stat().st_mtime_ns, first_mtime)
                manifest = store.publish_manifest(active_song_ids={"837567"})
            self.assertEqual(created["artifact_action"], "created")
            self.assertTrue(store.inspect(payload).ready)
            self.assertEqual(manifest["song_count"], 1)
            self.assertEqual(manifest["record_count"], 1)
            records = list(iter_context_artifact_records(root))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["song_id"], "837567")

    def test_stale_source_updates_and_orphan_is_excluded_from_manifest(self):
        original = song(
            facts=[
                {
                    "category": "record",
                    "section": "여담",
                    "text": "한 설문조사에서 역대 노래 1위를 기록했다.",
                }
            ]
        )
        changed = json.loads(json.dumps(original, ensure_ascii=False))
        changed["namuwiki"]["facts"][0]["text"] = (
            "2016년 설문조사에서 역대 보이그룹 노래 1위를 기록했다."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ContextArtifactStore(root)
            with store.writer():
                first = store.sync_song(original)
                self.assertEqual(first["artifact_action"], "created")
                self.assertFalse(store.inspect(changed).ready)
                updated = store.sync_song(changed)
                self.assertEqual(updated["artifact_action"], "updated")
                invalid_orphan = store.path("songs/999.json")
                invalid_orphan.write_text("not json", encoding="utf-8")
                manifest = store.publish_manifest(active_song_ids=set())
            self.assertEqual(manifest["song_count"], 0)
            self.assertEqual(manifest["orphan_artifact_count"], 2)
            self.assertEqual(manifest["invalid_artifact_count"], 0)

    def test_not_found_gets_empty_terminal_artifact(self):
        payload = song(status="not_found")
        with tempfile.TemporaryDirectory() as directory:
            store = ContextArtifactStore(Path(directory))
            with store.writer():
                result = store.sync_song(payload)
            artifact = store.read(result["artifact_ref"])
            self.assertEqual(artifact["status"], "not_found")
            self.assertEqual(artifact["retrieval"]["record_count"], 0)
            self.assertEqual(artifact["retrieval"]["records"], [])

    def test_content_hash_rejects_valid_looking_record_tampering(self):
        payload = song(
            facts=[
                {
                    "category": "record",
                    "section": "여담",
                    "text": "2016년 설문조사에서 역대 보이그룹 노래 1위를 기록했다.",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ContextArtifactStore(root)
            with store.writer():
                result = store.sync_song(payload)
            path = store.path(result["artifact_ref"])
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["retrieval"]["records"][0]["dense_text"] += (
                " 검증되지 않은 변조"
            )
            path.write_text(
                json.dumps(artifact, ensure_ascii=False),
                encoding="utf-8",
            )
            state = store.inspect(payload)
            self.assertFalse(state.ready)
            self.assertIn("artifact_invalid", state.reason)
            with store.writer():
                manifest = store.publish_manifest(active_song_ids={"837567"})
            self.assertEqual(manifest["invalid_artifact_count"], 1)


class BackfillArtifactIntegrationTests(unittest.TestCase):
    @staticmethod
    def fail_fetcher(*_args, **_kwargs):
        raise AssertionError("artifact reconciliation must not create a fetcher")

    def test_existing_v6_dry_run_then_artifact_only_sync_and_skip(self):
        payload = song(
            facts=[
                {
                    "category": "media_usage",
                    "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                    "text": (
                        "SBS에서 짱구는 못말려에 삽입되어 "
                        "선공개 식으로 방영되었다."
                    ),
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta_path = write_meta(root, payload)
            original_meta = meta_path.read_bytes()
            with patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            ):
                dry = run_backfill(
                    args(root, dry_run=True, song_ids=["837567"]),
                    fetcher_factory=self.fail_fetcher,
                )
                self.assertEqual(dry["songs"][0]["action"], "artifact")
                self.assertEqual(dry["selected_count"], 1)
                self.assertFalse((root / "artifacts").exists())

                first = run_backfill(
                    args(root, song_ids=["837567"]),
                    fetcher_factory=self.fail_fetcher,
                )
                self.assertEqual(first["songs"][0]["action"], "artifact_synced")
                self.assertEqual(meta_path.read_bytes(), original_meta)
                self.assertEqual(first["progress"]["completed_after"], 1)
                self.assertEqual(first["progress"]["scope_total"], 1)
                self.assertEqual(first["progress"]["completion_percent_after"], 100.0)
                artifact = root / "artifacts/context/songs/837567.json"
                self.assertTrue(artifact.is_file())
                self.assertTrue((root / "artifacts/context/manifest.json").is_file())
                progress = json.loads(
                    (root / "artifacts/context/progress.json").read_text(encoding="utf-8")
                )
                self.assertEqual(progress["state"], "completed")

                second = run_backfill(
                    args(root, song_ids=["837567"]),
                    fetcher_factory=self.fail_fetcher,
                )
                self.assertEqual(second["selected_count"], 0)
                self.assertEqual(second["songs"][0]["action"], "skip")
                self.assertEqual(second["songs"][0]["reason"], "already_complete")
                self.assertTrue(second["songs"][0]["artifact_ready"])

    def test_existing_v2_artifact_is_rebuilt_to_v6_without_fetching(self):
        payload = song(
            facts=[
                {
                    "category": "media_usage",
                    "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                    "text": "짱구는 못말려 6기 15화에 삽입되어 방영되었다.",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_meta(root, payload)
            old = root / "artifacts/context/songs/837567.json"
            old.parent.mkdir(parents=True)
            old.write_text(
                json.dumps({"artifact_schema_version": "context_artifact_v2"}),
                encoding="utf-8",
            )
            with patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            ):
                result = run_backfill(
                    args(root, song_ids=["837567"]),
                    fetcher_factory=self.fail_fetcher,
                )
            self.assertEqual(result["songs"][0]["action"], "artifact_synced")
            rewritten = json.loads(old.read_text(encoding="utf-8"))
            self.assertEqual(rewritten["artifact_schema_version"], "context_artifact_v6")
            self.assertEqual(rewritten["retrieval"]["record_count"], 1)

    def test_two_song_progress_counts_terminal_empty_artifact_as_complete(self):
        with_facts = song(
            facts=[
                {
                    "category": "media_usage",
                    "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                    "text": (
                        "SBS에서 짱구는 못말려에 삽입되어 "
                        "선공개 식으로 방영되었다."
                    ),
                }
            ]
        )
        absent = song(
            song_id="1698598",
            title="거짓말",
            artist="BIGBANG",
            status="not_found",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_meta(root, with_facts)
            write_meta(root, absent)
            with patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            ):
                result = run_backfill(
                    args(root, song_ids=["837567", "1698598"]),
                    fetcher_factory=self.fail_fetcher,
                )
            self.assertEqual(result["selected_count"], 2)
            self.assertEqual(result["progress"]["completed_after"], 2)
            self.assertEqual(result["progress"]["scope_total"], 2)
            self.assertEqual(result["progress"]["completion_percent_after"], 100.0)
            self.assertEqual(result["artifact_manifest"]["song_count"], 2)
            self.assertEqual(result["artifact_manifest"]["record_count"], 1)

    def test_targeted_noop_keeps_other_core_valid_song_in_manifest(self):
        first_song = song(status="not_found")
        second_song = song(
            song_id="1698598",
            title="거짓말",
            artist="BIGBANG",
            status="not_found",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_meta(root, first_song)
            write_meta(root, second_song)
            store = ContextArtifactStore(root / "artifacts/context")
            with store.writer():
                store.sync_song(first_song)
                store.sync_song(second_song)
            with patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            ):
                result = run_backfill(
                    args(root, song_ids=["837567"]),
                    fetcher_factory=self.fail_fetcher,
                )
            self.assertEqual(result["selected_count"], 0)
            self.assertEqual(result["artifact_manifest"]["song_count"], 2)
            self.assertEqual(result["artifact_manifest"]["orphan_artifact_count"], 0)
            self.assertEqual(result["progress"]["scope_total"], 1)
            self.assertEqual(result["catalog_progress"]["scope_total"], 2)
            self.assertEqual(result["catalog_progress"]["completed_after"], 2)

    def test_v1_migration_commits_v3_meta_and_artifact(self):
        payload = song(
            facts=[
                {
                    "category": "production",
                    "section": "여담",
                    "text": (
                        "이 노래는 원래 앨범에 넣을 예정이 없었지만 녹음이 끝난 뒤 "
                        "소속사 대표의 제안으로 수록되었다."
                    ),
                }
            ]
        )
        payload["namuwiki"]["schema_version"] = "namuwiki_v1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta_path = write_meta(root, payload)
            with patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            ):
                result = run_backfill(
                    args(root, song_ids=["837567"]),
                    fetcher_factory=self.fail_fetcher,
                )
            self.assertEqual(result["songs"][0]["action"], "migrated")
            updated = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["namuwiki"]["schema_version"], "namuwiki_v3")
            artifact = json.loads(
                (root / "artifacts/context/songs/837567.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(artifact["source"]["schema_version"], "namuwiki_v3")
            self.assertEqual(artifact["retrieval"]["record_count"], 1)

    def test_artifact_failure_after_meta_commit_repairs_without_fetching(self):
        payload = song(
            facts=[
                {
                    "category": "production",
                    "section": "여담",
                    "text": (
                        "이 노래는 원래 앨범에 넣을 예정이 없었지만 녹음이 끝난 뒤 "
                        "소속사 대표의 제안으로 수록되었다."
                    ),
                }
            ]
        )
        payload["namuwiki"]["schema_version"] = "namuwiki_v1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta_path = write_meta(root, payload)
            validation = patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            )
            with validation, patch.object(
                ContextArtifactStore,
                "sync_song",
                side_effect=OSError("simulated artifact disk failure"),
            ):
                failed = run_backfill(
                    args(root, song_ids=["837567"]),
                    fetcher_factory=self.fail_fetcher,
                )
            self.assertEqual(failed["songs"][0]["action"], "artifact_error")
            self.assertTrue(failed["songs"][0]["namuwiki_committed"])
            committed = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(committed["namuwiki"]["schema_version"], "namuwiki_v3")
            self.assertFalse((root / "artifacts/context/songs/837567.json").exists())

            with patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            ):
                repaired = run_backfill(
                    args(root, song_ids=["837567"]),
                    fetcher_factory=self.fail_fetcher,
                )
            self.assertEqual(repaired["songs"][0]["action"], "artifact_synced")
            self.assertTrue((root / "artifacts/context/songs/837567.json").is_file())

    def test_needs_review_is_a_manual_queue_not_an_artifact_error(self):
        payload = song(status="not_found")
        payload.pop("namuwiki")

        class PassiveFetcher:
            stopped = False
            requests_made = 0

            def close(self):
                pass

        review = {
            "schema_version": "namuwiki_v3",
            "status": "needs_review",
            "collected_at": "2026-09-22T00:00:00+00:00",
            "facts": [],
            "error_code": "page_title_mismatch",
        }
        audit = {"song_id": "837567", "status": "needs_review", "attempts": []}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta_path = write_meta(root, payload)
            validation = patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            )
            collection = patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.collect_target",
                return_value=(review, audit),
            )
            with validation, collection:
                first = run_backfill(
                    args(root, song_ids=["837567"]),
                    fetcher_factory=lambda *_a, **_kw: PassiveFetcher(),
                )

            row = first["songs"][0]
            self.assertEqual(row["action"], "needs_review")
            self.assertEqual(row["reason"], "page_title_mismatch")
            self.assertTrue(row["namuwiki_committed"])
            self.assertFalse(row["artifact_ready"])
            self.assertEqual(first["progress"]["deferred_this_run"], 1)
            self.assertEqual(first["progress"]["failed_this_run"], 0)
            self.assertFalse((root / "artifacts/context/songs/837567.json").exists())
            self.assertEqual(
                json.loads(meta_path.read_text(encoding="utf-8"))["namuwiki"]["status"],
                "needs_review",
            )

            unresolved = json.loads(
                (root / "artifacts/context/unresolved.json").read_text(encoding="utf-8")
            )
            self.assertEqual(unresolved["needs_review_song_ids"], ["837567"])
            self.assertEqual(unresolved["unresolved_song_ids"], ["837567"])

            with validation:
                second = run_backfill(
                    args(root, song_ids=["837567"]),
                    fetcher_factory=self.fail_fetcher,
                )
            self.assertEqual(second["selected_count"], 0)
            self.assertEqual(second["songs"][0]["reason"], "needs_review_manual_resolution")

    def test_request_budget_stops_batch_and_next_run_resumes(self):
        first_song = song(status="not_found")
        first_song.pop("namuwiki")
        second_song = song(
            song_id="1698598",
            title="거짓말",
            artist="BIGBANG",
            status="not_found",
        )
        second_song.pop("namuwiki")

        class BudgetFetcher:
            stopped = False

            def __init__(self):
                self.requests_made = 0

            def close(self):
                pass

        def collect(_store, fetcher, target, **_kwargs):
            fetcher.requests_made += 1
            return (
                {
                    "schema_version": "namuwiki_v3",
                    "status": "not_found",
                    "collected_at": "2026-09-22T00:00:00+00:00",
                    "facts": [],
                },
                {"song_id": target.song_id, "status": "not_found", "attempts": []},
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_meta(root, first_song)
            write_meta(root, second_song)
            validation = patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            )
            collection = patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.collect_target",
                side_effect=collect,
            )
            with validation, collection:
                first = run_backfill(
                    args(root),
                    fetcher_factory=lambda *_a, **_kw: BudgetFetcher(),
                )
            self.assertEqual(first["selected_count"], 2)
            self.assertEqual(first["progress"]["processed_this_run"], 1)
            self.assertEqual(first["progress"]["queued_this_run"], 1)
            self.assertTrue(first["batch"]["stopped_early"])
            self.assertEqual(first["batch"]["stop_reason"], "request_budget_exhausted")
            self.assertEqual(first["catalog_progress"]["pending_after"], 1)

            with validation, collection:
                second = run_backfill(
                    args(root),
                    fetcher_factory=lambda *_a, **_kw: BudgetFetcher(),
                )
            self.assertEqual(second["progress"]["processed_this_run"], 1)
            self.assertEqual(second["catalog_progress"]["pending_after"], 0)
            unresolved = json.loads(
                (root / "artifacts/context/unresolved.json").read_text(encoding="utf-8")
            )
            self.assertEqual(unresolved["unresolved_count"], 0)

    def test_full_catalog_defaults_to_bounded_batch_without_run_limit_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(41):
                payload = song(
                    song_id=str(100000 + index),
                    title=f"노래 {index}",
                    artist="가수",
                    status="not_found",
                )
                payload.pop("namuwiki")
                write_meta(root, payload)
            with patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            ):
                result = run_backfill(args(root, dry_run=True))
            self.assertEqual(result["batch"]["limit"], 40)
            self.assertEqual(result["selected_count"], 40)
            self.assertEqual(result["batch"]["queued_after_limit"], 1)
            self.assertEqual(len(result["songs"]), 40)
            self.assertFalse(any(row.get("reason") == "run_limit" for row in result["songs"]))


if __name__ == "__main__":
    unittest.main()
