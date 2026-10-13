"""Numbers, specific events and article subjects cannot substitute for facts."""

from __future__ import annotations

import pytest

from src.backend.schemas.query import ContextClue
from src.retrieval.context_evidence import _numbers_supported, _related
from src.retrieval.context_qdrant_search import ContextFactHit


def _clue(query, relation):
    return ContextClue(target="", relation=relation, search_query=query, confidence=0.8)


def _fact(text, category, title="가상곡"):
    return ContextFactHit(
        song_id="100", record_id="nw:100:synthetic", score=0.9,
        fact_text=text, source_url="https://namu.wiki/w/example",
        title=title, artists=("가상 가수",), category=category, section="여담",
        quality="ok", source_fact_indices=(0,),
    )


@pytest.mark.parametrize("query,text,expected", [
    ("1위", "2013년 멜론 연간 차트 26위", False),
    ("9개 사이트", "9월말 차트에 올랐다", False),
    ("11위", "1위를 기록했다", False),
    ("1위", "11위를 기록했다", False),
    ("1위", "1위로 진입했다", True),
    ("9개 사이트", "9개 사이트에서 기록했다", True),
    ("1위 9개 사이트", "멜론 1위와 9개 사이트 기록", True),
    ("1위 9개 사이트", "멜론 1위와 9월 기록", False),
    ("1분 11초", "1분 12초에 소리가 들린다", False),
    ("1분 11초", "1분 11초에 소리가 들린다", True),
    ("2020년", "2020년에 발표했다", True),
    ("2020년", "2021년에 발표했다", False),
    ("3집", "3월에 타이틀을 바꿨다", False),
    ("2억회", "2억 뷰를 기록했다", True),
    ("1,234회", "1234회로 기록됐다", True),
    ("1234회", "1,234회로 기록됐다", True),
    ("그룹 2K1의 노래", "로봇을 디자인했다", True),
])
def test_numeric_value_and_unit_are_both_supported(query, text, expected):
    assert _numbers_supported(query, text) is expected


def test_unrelated_annual_chart_fact_cannot_support_all_kill_event():
    clue = _clue("가상 가수의 첫 싱글이 멜론 1위로 진입하고 새벽에 9개 사이트 올킬",
                 "기록·영향")
    wrong = _fact("9월말에 음원이 나왔고 2013년 멜론 연간 차트 26위에 올랐다.", "record")
    right = _fact("첫 싱글은 멜론 1위로 진입하고 새벽에 9개 음원 사이트 올킬을 기록했다.", "record")
    assert not _related(clue, wrong, "100")
    assert _related(clue, right, "100")


def test_same_numbers_without_the_queried_event_are_insufficient():
    clue = _clue("멜론 1위와 9개 사이트 올킬을 기록한 곡", "기록·영향")
    assert not _related(clue, _fact("멜론 1위와 9개 사이트 차트에 이름을 올렸다.", "record"), "100")


def test_generic_stage_and_shared_names_do_not_support_military_event():
    clue = _clue("두 멤버가 군 복무 중 위문열차 무대에서 라이브로 불렀던 곡", "방송·공연 일화")
    wrong = _fact("무대에서는 두 멤버가 파트를 생략하고 라이브로 노래를 부른다.", "performance")
    right = _fact("두 멤버가 군 복무 중 위문열차 무대에서 라이브로 불렀다.", "performance")
    assert not _related(clue, wrong, "100")
    assert _related(clue, right, "100")


@pytest.mark.parametrize("text", [
    "두 멤버가 군 복무 중 라이브 무대에서 불렀다.",
    "두 멤버가 위문열차 무대에서 라이브로 불렀다.",
    "두 멤버가 군 복무했다. 위문열차에서 라이브로 불렀다.",
])
def test_specific_event_conditions_must_be_in_the_same_clause(text):
    clue = _clue("두 멤버가 군 복무 중 위문열차에서 라이브로 불렀다", "방송·공연 일화")
    assert not _related(clue, _fact(text, "performance"), "100")


def test_music_video_installation_does_not_prove_design_credit():
    clue = _clue("뮤직비디오의 로봇을 감독이 디자인했다", "뮤직비디오·영상 일화")
    assert not _related(clue, _fact("뮤직비디오의 로봇을 감독이 설치했다.", "music_video"), "100")
    assert _related(clue, _fact("뮤직비디오의 로봇을 감독이 디자인했다.", "music_video"), "100")


@pytest.mark.parametrize("brackets", [("<", ">"), ("〈", "〉"), ("《", "》")])
def test_article_about_a_song_cannot_cite_other_songs_event(brackets):
    left, right = brackets
    clue = _clue("인기곡으로 활동하던 때 후속곡으로 결정하려던 곡", "제작·발매 비화")
    wrong = _fact(f"{left}다른곡{right}은 인기곡으로 활동하던 때 후속곡으로 결정하려 했다.",
                  "musical_detail")
    right = _fact(f"{left}가상곡{right}은 인기곡으로 활동하던 때 후속곡으로 결정하려 했다.",
                  "musical_detail")
    assert not _related(clue, wrong, "100")
    assert _related(clue, right, "100")


def test_demonstrative_can_bind_current_song_when_another_work_is_named():
    clue = _clue("이 곡을 후속곡으로 결정하려 했다", "제작·발매 비화")
    fact = _fact("이 곡은 <다른곡> 다음 후속곡으로 결정하려 했다.", "musical_detail")
    assert _related(clue, fact, "100")


def test_named_works_rule_does_not_reject_a_real_media_scene():
    clue = ContextClue(target="가상극", relation="삽입곡·배경음악",
                       search_query="가상극 OST 삽입곡", confidence=0.8)
    fact = _fact("<가상극> 이별 장면에서 이 곡이 OST로 흘렀다.", "media_usage")
    assert _related(clue, fact, "100")
