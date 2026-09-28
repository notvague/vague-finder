import unittest

from src.crawler.context.document_parser import parse_document
from src.crawler.context.trivia import (
    classify,
    decide_song_page,
    extract_legacy_trivia_facts,
    extract_trivia_facts,
    is_trivia_path,
    split_fact_text,
    verify_song_page,
)

URL = "https://namu.wiki/w/사랑했나봐(윤도현)"


def page(body: str, *, title="사랑했나봐(윤도현)") -> bytes:
    return (
        f"<!doctype html><html><head><title>{title} - 나무위키</title></head>"
        f"<body><nav>메뉴</nav><article><h1>{title}</h1>{body}</article><footer>푸터</footer></body></html>"
    ).encode()


class TriviaExtractionTests(unittest.TestCase):
    def test_keeps_only_trivia_and_descendants(self):
        raw = page(
            "<h2>1. 개요</h2><p>윤도현의 곡 사랑했나봐다.</p>"
            "<h2>2. 가사</h2><p>검색에 섞이면 안 되는 가사</p>"
            "<h2>3. 여담[편집]</h2><ul>"
            "<li>이 노래는 애니메이션의 이별 장면에 삽입곡으로 사용되었다.</li>"
            "<li>원래는 다른 가수를 위한 곡으로 만들었다.</li></ul>"
            "<h3>3.1. 밈화[편집]</h3><p>인터넷에서 패러디가 화제가 되었다. #</p>"
            "<blockquote>길고 감정적인 등장인물 대사는 검색 사실에서 제외한다.</blockquote>"
            "<blockquote>가수는 원래 이 노래를 솔로곡으로 만들었다고 말했다.</blockquote>"
            "<h2>4. 관련 문서</h2><p>다른 문서</p>"
        )
        document = parse_document(raw, "snap", URL)
        facts, found = extract_trivia_facts(document)
        self.assertTrue(found)
        self.assertEqual(len(facts), 3)
        self.assertEqual(
            [fact["category"] for fact in facts],
            ["media_usage", "production", "meme"],
        )
        self.assertEqual(facts[2]["section"], "여담 > 밈화")
        text = " ".join(fact["text"] for fact in facts)
        self.assertNotIn("가사", text)
        self.assertNotIn("등장인물 대사", text)
        self.assertNotIn("솔로곡으로 만들었다고 말했다", text)
        self.assertNotIn("관련 문서", text)
        self.assertFalse(facts[2]["text"].endswith("#"))

    def test_attribution_footer_inside_trivia_is_removed(self):
        document = parse_document(page(
            "<h2>1. 개요</h2><p>윤도현의 곡이다.</p>"
            "<h2>2. 여담</h2><p>뮤직비디오에서 주인공이 기차를 탄다.</p>"
            "<p>이 문서의 내용 중 전체 또는 일부는</p><p>사랑했나봐</p>"
            "<p>문서의 r238 판에서 가져왔습니다. 이전 역사 보러 가기</p>"
            "<h2>3. 관련 문서</h2><p>끝</p>"
        ), "snap", URL)
        facts, _ = extract_trivia_facts(document)
        self.assertEqual(facts, [{
            "category": "music_video",
            "section": "여담",
            "text": "뮤직비디오에서 주인공이 기차를 탄다.",
        }])

    def test_no_trivia_is_a_valid_empty_result(self):
        document = parse_document(page(
            "<h2>1. 개요</h2><p>윤도현의 노래다.</p>"
            "<h2>2. 관련 문서</h2><p>끝</p>"
        ), "snap", URL)
        self.assertEqual(extract_trivia_facts(document), ([], False))

    def test_single_section_opaque_page_is_bounded_without_body_fallback(self):
        raw = (
            "<html><body><header>사이트 메뉴</header><div><h1>짧은곡(가수)</h1></div>"
            "<div><div><h2>1. 개요</h2><p>가수가 부른 노래이며 여담은 없다.</p></div></div>"
            "<aside>추천 뉴스</aside></body></html>"
        ).encode()
        document = parse_document(raw, "short", "https://namu.wiki/w/짧은곡(가수)")
        self.assertEqual(document.parse_status, "ok")
        self.assertEqual(document.diagnostics["root_selector"], "numbered_h2_common_ancestor")
        self.assertNotIn("추천 뉴스", " ".join(block.source_text for block in document.blocks))

    def test_supported_heading_variants_but_not_guitar_heading(self):
        for heading in ("여담", "기타 사항", "기타사항", "트리비아", "Trivia"):
            self.assertTrue(is_trivia_path([heading]))
        self.assertFalse(is_trivia_path(["기타"]))

    def test_love_song_nested_media_and_meme_format_preserves_q203_fact(self):
        document = parse_document(page(
            "<h2>1. 개요</h2><p>윤도현이 부른 노래이다.</p>"
            "<h2>6. 여담</h2><ul><li>최고음과 기타 연주에 관한 추가 설명이다.</li></ul>"
            "<h3>6.1. 짱구는 못말려 국내판 삽입곡</h3>"
            "<p>2005년 짱구는 못말려에서 이별하는 장면의 삽입곡으로 사용되었다.</p>"
            "<blockquote>봉미선:\"오늘은 참으라고!\" 신짱구:\"보고 싶은 건 보고 싶다구요!\"</blockquote>"
            "<h3>6.2. 밈화</h3><p>이 장면을 이용한 밈 영상이 퍼졌다.</p>"
            "<h2>7. 관련 문서</h2><p>끝</p>"
        ), "love", URL)
        facts, found = extract_trivia_facts(document)
        self.assertTrue(found)
        q203 = next(fact for fact in facts if "짱구는 못말려" in fact["text"])
        self.assertEqual(q203["category"], "media_usage")
        self.assertEqual(q203["section"], "여담 > 짱구는 못말려 국내판 삽입곡")
        self.assertFalse(any("봉미선" in fact["text"] for fact in facts))
        self.assertTrue(any(fact["category"] == "meme" for fact in facts))

    def test_long_fact_is_split_without_truncation(self):
        original = "첫 번째 설명이다. " * 100
        parts = split_fact_text(original.strip(), limit=90)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 90 for part in parts))
        self.assertEqual(" ".join(parts), original.strip())

    def test_category_examples_cover_project_use_cases(self):
        cases = {
            "뮤직비디오의 줄거리는 두 사람이 헤어지는 내용이다.": "music_video",
            "원래는 솔로곡으로 만들었지만 그룹 타이틀곡으로 바뀌었다.": "production",
            "복면가왕 경연 무대에서 불렀다.": "performance",
            "인터넷 패러디가 화제가 되었다.": "meme",
            "드라마의 배경음악으로 사용되었다.": "media_usage",
            "영어 제목은 복수형 버전이다.": "version",
            "뮤직비디오의 영어 제목은 LIE지만 음원은 LIES이다.": "version",
            "설문조사에서 역대 노래 1위를 기록했다.": "record",
            "다른 래퍼가 이 곡을 듣고 랩을 시작했다.": "influence",
            "멤버가 이 곡을 좋아한다고 말했다.": "other",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(classify(text), expected)

    def test_legacy_projection_uses_only_trivia_blocks(self):
        sources = [{
            "url": "https://namu.wiki/w/거짓말(BIGBANG)",
            "page_title": "거짓말(BIGBANG)",
            "blocks": [
                {"section_path": ["개요"], "block_type": "paragraph", "text": "개요 설명"},
                {"section_path": ["여담"], "block_type": "list_item",
                 "text": "리듬 게임에 수록되어 있다."},
                {"section_path": ["여담"], "block_type": "paragraph",
                 "text": "이 문서의 내용 중 전체 또는 일부는"},
            ],
        }]
        facts, found, url = extract_legacy_trivia_facts(sources)
        self.assertTrue(found)
        self.assertEqual(url, sources[0]["url"])
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["category"], "media_usage")


class PageIdentityTests(unittest.TestCase):
    def parse(self, title, body):
        return parse_document(page(body, title=title), "snap", URL)

    def test_title_artist_page_accepts_parenthesized_catalog_artist(self):
        document = self.parse(
            "거짓말(BIGBANG)",
            "<h2>1. 개요</h2><p>대표곡이다.</p><h2>2. 여담</h2><p>추가 정보가 있다.</p>",
        )
        self.assertEqual(verify_song_page("거짓말", ["BIGBANG (빅뱅)"], document)[0], True)

    def test_title_artist_page_accepts_space_normalized_artist(self):
        document = self.parse(
            "아리랑(SG워너비)",
            "<h2>1. 개요</h2><p>SG워너비가 발표한 노래다.</p>"
            "<h2>6. 여담</h2><p>추가 정보가 있다.</p>",
        )
        decision = decide_song_page("아리랑", ["SG 워너비"], document)
        self.assertTrue(decision.verified)
        self.assertEqual(decision.reason, "title_and_artist_match")

    def test_collaboration_page_accepts_an_exact_credited_artist(self):
        document = self.parse(
            "원더우먼(씨야, 다비치, 티아라)",
            "<h2>1. 개요</h2><p>세 그룹이 함께 발표한 프로젝트 싱글 곡이다.</p>"
            "<h2>4. 여담</h2><p>방송 무대에서 함께 불렀다.</p>",
        )
        decision = decide_song_page("원더우먼", ["씨야"], document)
        self.assertTrue(decision.verified)
        self.assertEqual(decision.reason, "title_and_coartist_match")
        self.assertIn("coartist_disambiguator", decision.evidence)

    def test_collaboration_page_rejects_another_same_title_artist(self):
        document = self.parse(
            "원더우먼(씨야, 다비치, 티아라)",
            "<h2>1. 개요</h2><p>세 그룹이 함께 발표한 프로젝트 싱글 곡이다.</p>"
            "<h2>4. 여담</h2><p>다른 가수의 동명곡도 존재한다.</p>",
        )
        decision = decide_song_page("원더우먼", ["걸스데이"], document)
        self.assertFalse(decision.verified)
        self.assertEqual(decision.reason, "page_title_mismatch")

    def test_collaboration_page_does_not_use_partial_artist_substrings(self):
        document = self.parse(
            "원더우먼(씨야, 다비치, 티아라)",
            "<h2>1. 개요</h2><p>세 그룹이 함께 발표한 프로젝트 싱글 곡이다.</p>"
            "<h2>4. 여담</h2><p>추가 정보가 있다.</p>",
        )
        decision = decide_song_page("원더우먼", ["티아"], document)
        self.assertFalse(decision.verified)
        self.assertEqual(decision.reason, "page_title_mismatch")

    def test_collaboration_page_requires_every_catalog_artist(self):
        document = self.parse(
            "원더우먼(씨야, 다비치, 티아라)",
            "<h2>1. 개요</h2><p>세 그룹이 함께 발표한 프로젝트 싱글 곡이다.</p>"
            "<h2>4. 여담</h2><p>추가 정보가 있다.</p>",
        )
        decision = decide_song_page("원더우먼", ["씨야", "다른 가수"], document)
        self.assertFalse(decision.verified)
        self.assertEqual(decision.reason, "page_title_mismatch")

    def test_love_song_exact_page_is_accepted(self):
        document = self.parse(
            "사랑했나봐(윤도현)",
            "<h2>1. 개요</h2><p>노래 설명</p><h2>2. 여담</h2><p>추가 설명</p>",
        )
        self.assertEqual(verify_song_page("사랑했나봐", ["윤도현"], document)[0], True)

    def test_title_only_page_requires_local_artist_identity(self):
        good = self.parse(
            "사랑했나봐",
            "<h2>1. 개요</h2><p>윤도현이 부른 노래이다.</p><h2>2. 여담</h2><p>추가 설명</p>",
        )
        bad = self.parse(
            "사랑했나봐",
            "<h2>1. 개요</h2><p>동명의 작품이다.</p><h2>2. 여담</h2><p>추가 설명</p>",
        )
        self.assertTrue(verify_song_page("사랑했나봐", ["윤도현"], good)[0])
        self.assertFalse(verify_song_page("사랑했나봐", ["윤도현"], bad)[0])

    def test_generic_song_page_requires_local_artist_identity(self):
        good = self.parse(
            "instagram(노래)",
            "<h2>1. 개요</h2><p>대한민국 가수 DEAN이 발표한 싱글 곡이다.</p>"
            "<h2>7. 여담</h2><p>방송에서 라이브 무대를 선보였다.</p>",
        )
        wrong_artist = self.parse(
            "instagram(노래)",
            "<h2>1. 개요</h2><p>다른 가수가 발표한 싱글 곡이다.</p>"
            "<h2>7. 여담</h2><p>DEAN과 관련된 별개의 설명이다.</p>",
        )

        self.assertEqual(
            verify_song_page("instagram", ["DEAN"], good),
            (True, "generic_song_page_with_local_artist_identity"),
        )
        self.assertEqual(
            verify_song_page("instagram", ["DEAN"], wrong_artist),
            (False, "generic_song_page_artist_not_verified"),
        )

    def test_disambiguation_page_selects_unique_artist_song_subtree(self):
        document = self.parse(
            "사랑했나봐",
            "<h2>1. 음악</h2>"
            "<h3>1.1. 윤도현의 노래</h3><p>윤도현의 곡이다.</p>"
            "<h4>1.1.1. 여담</h4><p>드라마의 배경음악으로 사용되었다.</p>"
            "<h3>1.2. 김종국의 노래</h3><p>김종국의 곡이다.</p>"
            "<h4>1.2.1. 여담</h4><p>인터넷 패러디가 화제가 되었다.</p>"
            "<h2>2. 드라마</h2><p>작품</p>",
        )
        decision = decide_song_page("사랑했나봐", ["윤도현"], document)
        self.assertTrue(decision.verified)
        self.assertEqual(decision.reason, "artist_song_subsection_match")
        self.assertIsNotNone(decision.scope_section_id)

        facts, found = extract_trivia_facts(
            document,
            scope_section_id=decision.scope_section_id,
        )
        self.assertTrue(found)
        self.assertEqual(len(facts), 1)
        self.assertIn("배경음악", facts[0]["text"])
        self.assertFalse(any("패러디" in fact["text"] for fact in facts))

    def test_disambiguation_page_keeps_equal_artist_sections_for_review(self):
        document = self.parse(
            "사랑했나봐",
            "<h2>1. 음악</h2>"
            "<h3>1.1. 윤도현의 노래</h3><p>윤도현의 첫 번째 곡이다.</p>"
            "<h3>1.2. 윤도현의 노래</h3><p>윤도현의 두 번째 곡이다.</p>",
        )
        decision = decide_song_page("사랑했나봐", ["윤도현"], document)
        self.assertFalse(decision.verified)
        self.assertEqual(decision.reason, "multiple_equally_strong_song_sections")

    def test_global_artist_text_cannot_approve_multiple_unbound_song_sections(self):
        document = self.parse(
            "동명곡",
            "<h2>1. 개요</h2><p>대상가수도 부른 노래를 모은 문서다.</p>"
            "<h2>2. 다른가수의 노래</h2><h3>2.1. 여담</h3>"
            "<p>방송에 삽입되었다.</p>"
            "<h2>3. 또다른가수의 노래</h2><h3>3.1. 여담</h3>"
            "<p>인터넷 밈으로 유행했다.</p>",
        )
        decision = decide_song_page("동명곡", ["대상가수"], document)
        self.assertFalse(decision.verified)
        self.assertEqual(
            decision.reason,
            "disambiguation_page_multiple_song_sections",
        )

    def test_title_only_page_uses_table_identity_and_album_metadata(self):
        document = self.parse(
            "나의 어깨에 기대어요",
            "<table><tr><th>가수</th><td>10CM</td></tr>"
            "<tr><th>앨범</th><td>호텔 델루나 OST Part.2</td></tr></table>"
            "<h2>1. 개요</h2><p>2019년에 발매된 OST 수록곡이다.</p>"
            "<h2>2. 여담</h2><p>드라마의 이별 장면에 삽입되었다.</p>",
        )
        decision = decide_song_page(
            "나의 어깨에 기대어요",
            ["10CM"],
            document,
            album="호텔 델루나 OST Part.2",
            release_year=2019,
        )
        self.assertTrue(decision.verified)
        self.assertIn("page_artist:10CM", decision.evidence)
        self.assertIn("page_album", decision.evidence)

    def test_album_page_selects_nested_artist_track_and_normalizes_ost_part(self):
        document = self.parse(
            "호텔 델루나/OST",
            "<h2>1. 개요</h2><p>호텔 델루나 OST 음반이다.</p>"
            "<h2>2. 10CM</h2><p>10CM이 참여했다.</p>"
            "<h3>2.1. 나의 어깨에 기대어요</h3><p>2019년 발매 수록곡이다.</p>"
            "<h4>2.1.1. 여담</h4><p>드라마의 이별 장면에 삽입되었다.</p>"
            "<h2>3. 다른 가수</h2><h3>3.1. 다른 곡</h3>"
            "<h4>3.1.1. 여담</h4><p>인터넷 밈으로 유행했다.</p>",
        )
        decision = decide_song_page(
            "나의 어깨에 기대어요",
            ["10CM"],
            document,
            album="호텔 델루나 OST Part.2",
            release_year=2019,
        )
        self.assertTrue(decision.verified)
        self.assertEqual(decision.reason, "title_section_artist_match")
        facts, found = extract_trivia_facts(
            document,
            scope_section_id=decision.scope_section_id,
        )
        self.assertTrue(found)
        self.assertEqual(len(facts), 1)
        self.assertIn("이별 장면", facts[0]["text"])
        self.assertFalse(any("밈" in fact["text"] for fact in facts))

    def test_hash_artist_does_not_match_an_ordinary_greeting(self):
        document = self.parse(
            "옛 생각",
            "<h2>1. 개요</h2><p>안녕하세요. 브레이브걸스의 수록곡이다.</p>"
            "<h2>2. 여담</h2><p>무대에서 라이브로 불렀다.</p>",
        )
        decision = decide_song_page(
            "옛 생각",
            ["#안녕"],
            document,
            album="옛 생각",
        )
        self.assertFalse(decision.verified)
        self.assertEqual(decision.reason, "title_only_page_artist_not_verified")

    def test_mismatched_page_with_same_section_title_but_wrong_artist_is_rejected(self):
        document = self.parse(
            "브레이브걸스의 앨범",
            "<h2>1. 개요</h2><p>브레이브걸스의 음반이다.</p>"
            "<h2>2. 옛 생각</h2><p>브레이브걸스가 부른 수록곡이다.</p>"
            "<h3>2.1. 여담</h3><p>무대에서 라이브로 불렀다.</p>",
        )
        decision = decide_song_page(
            "옛 생각",
            ["#안녕"],
            document,
            album="옛 생각",
            release_year=2026,
        )
        self.assertFalse(decision.verified)
        self.assertEqual(decision.reason, "page_title_mismatch")


if __name__ == "__main__":
    unittest.main()
