"""External events can open Context without turning sensory clues into facts."""

import unittest

from src.backend.schemas.query import ContextClue
from src.retrieval.context_evidence import _related
from src.retrieval.context_qdrant_search import ContextFactHit
from src.retrieval.context_query import apply_context_query_safeguards


def _clues(query: str) -> list[ContextClue]:
    return [ContextClue.model_validate(value) for value in
            apply_context_query_safeguards(query, {"context_clues": []})["context_clues"]]


def _fact(text: str, category: str, *, artists=("가수 도라",),
          title="가상곡", song_id="100") -> ContextFactHit:
    return ContextFactHit(
        song_id=song_id, record_id=f"nw:{song_id}:example", score=0.9,
        fact_text=text, source_url="https://namu.wiki/w/example",
        title=title, artists=artists, category=category, section="여담",
        quality="ok", source_fact_indices=(0,),
    )


class ExternalEventVariantsTest(unittest.TestCase):
    def test_natural_external_events_create_searchable_clues(self):
        examples = (
            ("가수 도라가 가상곡 다음 후속곡으로 밀었으나 다른 노래를 골랐대", "제작·발매 비화"),
            ("팬 의견을 받고 두 곡을 더블 타이틀로 바꾼 음반 수록곡", "제작·발매 비화"),
            ("지난해 멜론 연간 차트 11위였다는 발라드", "기록·영향"),
            ("음원 사이트 아홉 곳에서 올킬했다는 노래", "기록·영향"),
            ("음악 방송 인기가요에서 1위를 받은 노래", "기록·영향"),
            ("SUPER SHOW 5에서 멤버의 파트를 대신 불렀던 노래", "방송·공연 일화"),
            ("위문열차에서 군 복무 중 라이브로 불렀다는 노래", "방송·공연 일화"),
            ("응원 소리가 녹음된 콘서트 버전 음원이 따로 나온 노래", "방송·공연 일화"),
            ("2008년 1박 2일 어느 섬 편의 숭어잡기 장면에 삽입된 노래", "다른 작품에 사용"),
            ("코러스에 김도현 목소리가 들린다고 알려진 발라드", "보컬 참여 일화"),
            ("한 가수만 'hello dear'를 다르게 발음한다는 여담이 있던 노래", "음원·보컬 여담"),
            ("1분 11초에 보컬 목소리로 '퓽퓽퓽' 소리가 난다는 여담", "음원·보컬 여담"),
        )
        for query, relation in examples:
            with self.subTest(query=query):
                clues = _clues(query)
                self.assertTrue(any(clue.relation == relation for clue in clues))
                self.assertTrue(all(clue.search_query in query for clue in clues))

    def test_lyrics_cover_and_sound_analogies_are_not_external_events(self):
        queries = (
            "멜론 차트 1위 같은 짜릿한 EDM 느낌으로 들리는 노래",
            "콘서트처럼 신나는 라이브 사운드가 들리는 곡",
            "후속곡으로 밀었다는 가사가 나오는 곡",
            "앨범 표지에 1박 2일 장면이 그려진 노래",
            "코러스에 남자 목소리가 들리는 발라드",
            "1분 11초에 들리는 특정 소리의 노래",
            "발음이 특이하고 보컬 소리가 들리는 노래",
        )
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(_clues(query), [])

    def test_metadata_name_does_not_have_to_appear_inside_an_event_fact(self):
        clue = _clues("가수 도라의 뮤직비디오에서 배경 로봇을 감독이 디자인했대")[0]
        fact = _fact("뮤직비디오의 배경 로봇을 감독이 직접 디자인했다.", "music_video")
        self.assertTrue(_related(clue, fact, "100"))
        self.assertFalse(_related(clue, _fact(
            "뮤직비디오의 배경 로봇을 설치했다.", "music_video"
        ), "100"))

        numbered_artist = _clues("그룹 2K1 뮤직비디오에서 김도현이 로봇을 디자인했대")[0]
        self.assertTrue(_related(numbered_artist, _fact(
            "뮤직비디오 로봇은 김도현이 디자인했다.", "music_video",
            artists=("2K1",)
        ), "100"))

    def test_concrete_event_is_required_even_for_matching_category(self):
        concert = _clues("가수 도라가 콘서트 버전 음원을 따로 불렀대")[0]
        self.assertTrue(_related(concert, _fact(
            "별도 콘서트 버전 음원으로 불렀다.", "performance"
        ), "100"))
        self.assertFalse(_related(concert, _fact(
            "첫 콘서트에서 다음 음반을 발표했다.", "performance"
        ), "100"))

        production = _clues("팬 요청으로 더블 타이틀로 바꾼 곡")[0]
        self.assertTrue(_related(production, _fact(
            "팬 의견을 받아 더블 타이틀로 변경했다.", "record"
        ), "100"))
        self.assertFalse(_related(production, _fact(
            "팬들에게 사랑받아 차트 상위권을 기록했다.", "record"
        ), "100"))

    def test_broadcast_fact_can_be_classified_as_media_usage(self):
        clue = _clues("방송 인기가요 1위를 받은 곡")[0]
        matching = _fact("방송 인기가요에서 1위를 받았다.", "media_usage")
        unrelated = _fact("방송 무대에 출연했다.", "media_usage")
        self.assertTrue(_related(clue, matching, "100"))
        self.assertFalse(_related(clue, unrelated, "100"))

    def test_reported_recording_detail_still_needs_exact_fact(self):
        clue = _clues("1분 11초에 보컬 목소리로 '퓽퓽퓽' 소리가 난다는 여담")[0]
        self.assertTrue(_related(clue, _fact(
            "음원 1분 11초에 보컬 목소리로 퓽퓽퓽 소리가 들린다.", "other"
        ), "100"))
        self.assertFalse(_related(clue, _fact(
            "음원 1분 12초에 보컬 목소리로 퓽퓽퓽 소리가 들린다.", "other"
        ), "100"))

    def test_wrong_song_source_or_negation_cannot_be_cited(self):
        clue = _clues("코러스에 김도현 목소리가 들린다고 알려진 곡")[0]
        self.assertFalse(_related(clue, _fact(
            "코러스에 김도현 목소리가 들린다.", "musical_detail", song_id="101"
        ), "100"))
        self.assertFalse(_related(clue, _fact(
            "코러스에 김도현 목소리가 들리는 것은 아니다.", "musical_detail"
        ), "100"))


if __name__ == "__main__":
    unittest.main()
