import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src.crawler.context.document_parser import parse_document
from src.crawler.context.schemas import NAMUWIKI_META_VERSION, NamuwikiFact
from src.crawler.context.trivia import (
    MAX_FACT_CHARS,
    classify,
    clean_fact_text,
    is_indexable_fact,
    extract_trivia_facts,
    refine_existing_facts,
    split_fact_text,
)
from src.crawler.context.store import Store
from src.crawler.scripts_py.backfill_namuwiki_context import (
    migrate_existing,
    run_backfill,
    should_process,
)
from src.embedding.text.context_bm25_tokenizer import tokenize_context_for_bm25
from src.embedding.text.namuwiki_passage import iter_namuwiki_fact_records


class TriviaRefinementTests(unittest.TestCase):
    def test_fresh_dom_projection_uses_same_refinement_gate(self):
        raw = """
        <article><h1>사랑했나봐(윤도현)</h1>
        <h2>1. 개요</h2><p>윤도현의 노래다.</p>
        <h2>6. 여담</h2><p>최고음은 2옥타브 라(A4).[1]</p>
        <h3>6.1. 짱구는 못말려 국내판 삽입곡</h3>
        <p>SBS 애니메이션의 이별 장면에서 이 곡이 배경음악으로 사용되었다.[20]</p>
        <p>그걸 본 원장은 담담한 어조로,</p>
        <h3>6.2. 밈화</h3><p>경연 영상을 합성한 챌린지가 유행했다.</p>
        <p>오망 / 소련했나봐 2024.07.17.</p></article>
        """.encode()
        document = parse_document(raw, "snapshot", "https://namu.wiki/w/test")
        facts, found = extract_trivia_facts(document)
        self.assertTrue(found)
        self.assertEqual(
            {row["category"] for row in facts},
            {"musical_detail", "media_usage", "meme"},
        )
        self.assertFalse(any("[20]" in row["text"] or row["text"].endswith(",") for row in facts))
        self.assertFalse(any("2024.07.17" in row["text"] for row in facts))

    def test_citations_removed_but_named_brackets_preserved(self):
        value = "SBS에서 삽입되었다.[20][21] [히틀러]라는 제목은 그대로 둔다. #"
        cleaned = clean_fact_text(value)
        self.assertNotIn("[20]", cleaned)
        self.assertNotIn("[21]", cleaned)
        self.assertIn("[히틀러]", cleaned)
        self.assertFalse(cleaned.endswith("#"))

    def test_sentence_units_are_atomic_and_hard_bounded(self):
        parts = split_fact_text("첫 번째 곡은 방송에 삽입되었다. 두 번째 사실은 차트 1위를 기록했다.")
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(len(part) <= MAX_FACT_CHARS for part in parts))
        self.assertEqual(split_fact_text("가" * (MAX_FACT_CHARS + 1) + "."), [])
        with self.assertRaises(ValueError):
            split_fact_text("문장.", limit=MAX_FACT_CHARS + 1)

    def test_sentence_splitter_does_not_cut_inside_straight_quotes(self):
        value = (
            "처음 데모곡을 들었을 때 윤도현의 반응은 '으에? 이걸요?'라고 했다. "
            "그 뒤 이 노래가 크게 흥행했다."
        )
        parts = split_fact_text(value)
        self.assertEqual(len(parts), 2)
        self.assertIn("'으에? 이걸요?'라고 했다.", parts[0])
        self.assertTrue(is_indexable_fact(parts[0], category="production"))
        self.assertFalse(is_indexable_fact(
            "처음 데모곡을 들었을 때 윤도현의 반응은 '으에?",
            category="production",
        ))
        # The apostrophe in an English contraction is not a quote delimiter.
        self.assertTrue(is_indexable_fact(
            "노래 가사 중 \"I'm ready\"를 합성한 밈이 유행했다.",
            category="meme",
        ))

    def test_leading_video_caption_and_timecode_are_removed(self):
        value = "(재더빙 버전, 16:27~)[30][31] 상당히 잘 선정한 배경음악이라 호평을 받았다."
        self.assertEqual(
            clean_fact_text(value),
            "상당히 잘 선정한 배경음악이라 호평을 받았다.",
        )

    def test_incomplete_quote_and_listing_fragments_are_excluded(self):
        self.assertFalse(is_indexable_fact("그걸 본 원장은 담담한 어조로,", category="media_usage"))
        self.assertFalse(is_indexable_fact("라고 명언을 남기며 공항으로 출발했다.", category="media_usage"))
        self.assertFalse(is_indexable_fact("오망 / 소련했나봐 2024.07.17.", category="meme", section="여담 > 밈화"))
        self.assertFalse(is_indexable_fact('"앞부분만 남은 인용문', category="production"))

    def test_section_priority_and_specific_text_priorities(self):
        self.assertEqual(classify("경연대회 영상도 유명하다.", "여담 > 밈화"), "meme")
        self.assertEqual(classify("나미리가 우는 장면이다.", "여담 > 국내판 삽입곡"), "media_usage")
        self.assertEqual(classify("최고음은 2옥타브 라이고 노래방에서 어렵다.", "여담"), "musical_detail")
        self.assertEqual(classify("띄어쓰기로 생긴 유머가 노래방 책에 적혔다.", "여담"), "meme")
        self.assertEqual(classify("리듬 게임에 수록되었다.", "여담"), "media_usage")

    def test_other_requires_a_complete_concrete_song_relation(self):
        self.assertFalse(is_indexable_fact("그때 이런 일이 있었다.", category="other"))
        self.assertTrue(is_indexable_fact(
            "이 곡으로 얻은 수익은 전부 YB 밴드에 재투자했다고 한다.",
            category="other",
        ))

    def test_love_sample_is_split_cleanly_and_keeps_query_evidence(self):
        facts = [
            {
                "category": "production",
                "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                "text": (
                    "정식 발매 전 SBS에서 짱구는 못말려에 삽입되었다.[20][21] "
                    "내용을 설명하자면 나미리 선생님이 연인과 싸우고 이별하는 장면에서 운다."
                ),
            },
            {
                "category": "other",
                "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                "text": "그걸 본 원장은 담담한 어조로,",
            },
            {
                "category": "other",
                "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                "text": "라고 명언을 남겼다. 이때 창밖을 보며 눈물을 흘리는 나미리와 함께 이 곡이 흐른다.",
            },
            {
                "category": "performance",
                "section": "여담 > 밈화",
                "text": "한 밴드의 경연 영상에 합성한 챌린지가 유행했다.",
            },
            {
                "category": "other",
                "section": "여담 > 밈화",
                "text": "오망 / 소련했나봐 2024.07.17.",
            },
        ]
        refined = refine_existing_facts(facts, page_title="사랑했나봐")
        self.assertTrue(refined)
        self.assertTrue(all(len(row["text"]) <= MAX_FACT_CHARS for row in refined))
        self.assertTrue(all("[20]" not in row["text"] for row in refined))
        self.assertFalse(any(row["text"].startswith("라고") for row in refined))
        self.assertFalse(any(row["text"].endswith(",") for row in refined))
        self.assertFalse(any("2024.07.17" in row["text"] for row in refined))
        self.assertTrue(any("이별하는 장면" in row["text"] for row in refined))
        self.assertTrue(any("이 곡이 흐른다" in row["text"] for row in refined))
        self.assertTrue(any(row["category"] == "meme" for row in refined))

    def test_dependent_sentences_merge_without_unrelated_plot_noise(self):
        facts = [
            {
                "category": "production",
                "section": "여담",
                "text": (
                    "처음 데모곡을 들었을 때 반응은 '으에? 이걸요?'라고 할 정도로 "
                    "마음에 들지 않았던 노래였다. 그런데 이후 노래가 크게 흥행해 "
                    "자신의 안목을 한탄했다고 한다."
                ),
            },
            {
                "category": "meme",
                "section": "여담",
                "text": (
                    "띄어쓰기를 잘못하면 다른 뜻이 된다는 유머가 존재한다. "
                    "심지어 노래방 책에 같은 드립을 낙서한 사례가 있다. "
                    "의외로 방송에서도 이 드립을 사용했다."
                ),
            },
            {
                "category": "media_usage",
                "section": "여담 > 짱구는 못말려 국내판 삽입곡",
                "text": (
                    "재더빙에서도 이 곡을 배경음악으로 사용해 호평을 받았다. "
                    "덕분에 음원 차트 상위권을 다시 기록했다. "
                    "그러나 이현우라는 캐릭터는 나중에 원작에서 사망한다."
                ),
            },
        ]
        refined = refine_existing_facts(facts, page_title="사랑했나봐")
        texts = [row["text"] for row in refined]
        self.assertTrue(any("그런데 이후 노래가 크게 흥행" in text for text in texts))
        self.assertTrue(any("심지어 노래방 책" in text and "의외로 방송" in text for text in texts))
        self.assertTrue(any("덕분에 음원 차트 상위권" in text for text in texts))
        self.assertFalse(any("캐릭터는 나중에 원작에서 사망" in text for text in texts))
        self.assertTrue(all(len(text) <= MAX_FACT_CHARS for text in texts))

    def test_adjacent_legacy_quote_fragments_are_rejoined(self):
        facts = [
            {
                "category": "production",
                "section": "여담",
                "text": "처음 데모곡을 들었을 때 윤도현의 반응은 '으에?",
            },
            {
                "category": "production",
                "section": "여담",
                "text": "이걸요?'라고 할 정도로 마음에 들지 않았던 노래였다고 한다.",
            },
        ]
        refined = refine_existing_facts(facts, page_title="사랑했나봐")
        self.assertEqual(len(refined), 1)
        self.assertIn("'으에? 이걸요?'라고", refined[0]["text"])

    def test_schema_enforces_musical_detail_and_hard_limit(self):
        fact = NamuwikiFact(category="musical_detail", section="여담", text="최고음은 2옥타브 라(A4).")
        self.assertEqual(fact.category, "musical_detail")
        with self.assertRaises(ValidationError):
            NamuwikiFact(category="other", section="여담", text="가" * 401)


class CompactMigrationTests(unittest.TestCase):
    def test_v1_terminal_data_migrates_offline_and_current_version_skips(self):
        meta = {
            "metadata": {"title": "사랑했나봐"},
            "namuwiki": {
                "schema_version": "namuwiki_v1",
                "status": "ok",
                "source_url": "https://namu.wiki/w/test",
                "collected_at": "2026-09-15T00:00:00+00:00",
                "facts": [{
                    "category": "media_usage",
                    "section": "여담 > 삽입곡",
                    "text": "애니메이션의 이별 장면에서 이 곡이 배경음악으로 사용되었다.[20]",
                }],
            },
        }
        self.assertEqual(should_process(meta), ("migrate", "upgrade_compact_namuwiki_schema"))
        upgraded = migrate_existing(meta)
        self.assertEqual(upgraded["schema_version"], NAMUWIKI_META_VERSION)
        self.assertNotIn("[20]", upgraded["facts"][0]["text"])
        meta["namuwiki"] = upgraded
        self.assertEqual(should_process(meta), ("skip", "already_complete"))

    def test_v2_success_reextracts_from_cache_instead_of_recrawling(self):
        meta = {
            "namuwiki": {
                "schema_version": "namuwiki_v2",
                "status": "ok",
                "source_url": "https://namu.wiki/w/test",
                "facts": [{
                    "category": "production",
                    "section": "여담",
                    "text": "이 노래는 원래 예정에 없던 곡이었다.",
                }],
            },
        }
        self.assertEqual(
            should_process(meta),
            ("reextract", "upgrade_refinement_rules_from_cached_source"),
        )

    def test_v2_reextract_uses_cached_snapshot_with_zero_network_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "raw"
            cache_dir = root / "context"
            song_dir = data_dir / "윤도현_사랑했나봐_837567"
            song_dir.mkdir(parents=True)
            url = "https://namu.wiki/w/사랑했나봐(윤도현)"
            meta = {
                "song_id": "837567",
                "metadata": {"title": "사랑했나봐", "artist": ["윤도현"]},
                "namuwiki": {
                    "schema_version": "namuwiki_v2",
                    "status": "ok",
                    "source_url": url,
                    "facts": [{
                        "category": "production",
                        "section": "여담",
                        "text": "이 노래는 원래 예정에 없던 곡이었다.",
                    }],
                },
            }
            (song_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False),
                encoding="utf-8",
            )
            html = (
                "<article><h1>사랑했나봐(윤도현)</h1><h2>1. 개요</h2>"
                "<p>윤도현의 노래다.</p><h2>6. 여담</h2>"
                "<p>처음 반응은 '으에? 이걸요?'라고 했다. "
                "그런데 이후 노래가 크게 흥행했다.</p>"
                "<p>(재더빙 버전, 16:27~) 이 곡은 애니메이션 배경음악으로 사용되었다.</p>"
                "</article>"
            ).encode()
            with Store(cache_dir).writer():
                Store(cache_dir).save_snapshot(
                    html,
                    requested_url=url,
                    acquisition_method="browser_rendered",
                    final_url=url,
                    http_status=200,
                    headers={"Content-Type": "text/html; charset=utf-8"},
                )
            args = argparse.Namespace(
                data_dir=data_dir,
                cache_dir=cache_dir,
                url_map=None,
                song_ids=["837567"],
                retry_not_found=False,
                force=False,
                limit=None,
                dry_run=False,
                html_dir=None,
                offline=False,
                refresh=False,
                interval=0.0,
                timeout=1.0,
                no_render=True,
                render_timeout=1.0,
                max_requests=1,
            )
            with patch(
                "src.crawler.scripts_py.backfill_namuwiki_context.validate_meta_document",
                return_value=[],
            ):
                report = run_backfill(args)
            self.assertEqual(report["songs"][0]["action"], "reprocessed")
            updated = json.loads((song_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["namuwiki"]["schema_version"], NAMUWIKI_META_VERSION)
            texts = [row["text"] for row in updated["namuwiki"]["facts"]]
            self.assertTrue(any("'으에? 이걸요?'라고" in text for text in texts))
            self.assertTrue(any(text.startswith("이 곡은 애니메이션") for text in texts))
            stored = Store(cache_dir).read(report["report_ref"])
            fetch = stored["audits"][0]["attempts"][0]["fetch"]
            self.assertEqual(fetch["requests_made"], 0)
            self.assertTrue(fetch["cache_hit"])

    def test_retry_not_found_still_collects(self):
        meta = {"namuwiki": {"schema_version": "namuwiki_v1", "status": "not_found", "facts": []}}
        self.assertEqual(should_process(meta, retry_not_found=True), ("collect", "retry_not_found"))


class ContextTokenizerTests(unittest.TestCase):
    def test_strict_pos_compound_entity_relations_and_synonyms(self):
        tokens = tokenize_context_for_bm25(
            "짱구는 못말려의 이별 장면에 OST로 삽입되어 배경음악으로 사용되었다."
        )
        for expected in ("짱구는_못말려", "ost", "삽입곡", "배경음악", "bgm", "사용되다"):
            self.assertIn(expected, tokens)
        for noisy in ("그런데", "매우", "이것"):
            self.assertNotIn(noisy, tokens)

    def test_context_only_synonym_families(self):
        tokens = set(tokenize_context_for_bm25("뮤비 패러디 챌린지 라이브 무대 제작 데모"))
        self.assertTrue({"뮤직비디오", "뮤비", "mv"} <= tokens)
        self.assertTrue({"밈", "패러디", "합성", "챌린지"} <= tokens)
        self.assertTrue({"무대", "공연", "경연", "라이브"} <= tokens)
        self.assertTrue({"제작", "작곡", "녹음", "데모"} <= tokens)

    def test_document_tokens_do_not_expand_query_synonyms(self):
        document = set(
            tokenize_context_for_bm25(
                "애니메이션 삽입곡으로 사용되었다.",
                expand_synonyms=False,
            )
        )
        query = set(tokenize_context_for_bm25("애니메이션 삽입곡"))
        self.assertNotIn("ost", document)
        self.assertNotIn("bgm", document)
        self.assertTrue({"ost", "bgm", "배경음악"} <= query)

    def test_musical_notation_is_protected(self):
        tokens = set(
            tokenize_context_for_bm25(
                "원키 코드는 D-E-F#m이고 최고음은 2옥타브 라(A4)다.",
                expand_synonyms=False,
            )
        )
        self.assertTrue({"d-e-f#m", "2옥타브", "a4"} <= tokens)

    def test_surface_recovery_does_not_manufacture_punctuation_tokens(self):
        tokens = set(
            tokenize_context_for_bm25(
                "코드는 D-E-F#m만 반복되고 1~2번 부른다. 영어 제목은 'LIE'지만 'LIES'로도 표기한다.",
                expand_synonyms=False,
            )
        )
        self.assertTrue({"d-e-f#m", "2번", "lie", "lies"} <= tokens)
        self.assertNotIn("d-e-f#m만", tokens)
        self.assertNotIn("12번", tokens)
        self.assertFalse(any("'지" in token for token in tokens))

    def test_unknown_named_phrase_is_preserved_without_particle_cross_join(self):
        tokens = set(
            tokenize_context_for_bm25(
                "메타톤 잼민이 밈이며 김영민과 윤형빈이 출연했다.",
                expand_synonyms=False,
            )
        )
        self.assertIn("메타톤_잼민이", tokens)
        self.assertNotIn("김영민_윤형빈", tokens)

    def test_passage_accepts_only_current_schema_and_preserves_auditable_facts(self):
        song = {
            "id": "837567",
            "metadata": {"title": "사랑했나봐", "artist": ["윤도현"]},
            "namuwiki": {
                "schema_version": NAMUWIKI_META_VERSION,
                "status": "ok",
                "source_url": "https://namu.wiki/w/test",
                "facts": [
                    {"category": "media_usage", "section": "여담 > 삽입곡", "text": "짱구는 못말려의 이별 장면에 삽입되었다."},
                    {"category": "other", "section": "여담", "text": "그때 이런 일이 있었다."},
                ],
            },
        }
        records = list(iter_namuwiki_fact_records(song))
        self.assertEqual(len(records), 2)
        media = records[0]
        self.assertEqual(media["source_fact_indices"], [0])
        self.assertIn("짱구는 못말려", media["dense_text"])
        self.assertIn("삽입되었다", media["dense_text"])
        dependent = records[1]
        self.assertEqual(dependent["source_fact_indices"], [1])
        self.assertEqual(dependent["quality"], "context_dependent")
        self.assertIn("그때 이런 일이 있었다", dependent["dense_text"])
        song["namuwiki"]["schema_version"] = "namuwiki_v2"
        self.assertEqual(list(iter_namuwiki_fact_records(song)), [])


if __name__ == "__main__":
    unittest.main()
