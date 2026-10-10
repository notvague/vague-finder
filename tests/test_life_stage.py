"""생애 단계 표현("중학교 때·어릴 때")은 사용자 나이에 걸린 시간 단서다 — NEXT_WORK §2-13.

규칙: ① 발매 시기를 추측하지 않는다(release_era 비움) ② korean_tags로 새지 않는다 ③ 임베딩 질의문(search_text)에서 뺀다
④ 가사 내용·곡의 사건으로 쓰인 같은 말은 잡지 않는다 ⑤ 출생 연도를 받으면 창을 만든다.
"""
import pytest

from src.backend.schemas.query import LifeStageClue, QueryAnalysis
from src.retrieval import query_analyzer as qa


@pytest.mark.parametrize("query, stage, text", [
    ("중학교 때 많이 듣던 남자 발라드인데 제목이 기억 안 나", "middle", "중학교 때"),
    ("중딩 시절 유행했던 걸그룹 노래", "middle", "중딩 시절"),
    ("고3 때 야자 끝나고 듣던 노래", "high", "고3 때"),
    ("고등학교 수학여행 버스에서 틀었던 신나는 노래, 남자 그룹", "high", "고등학교 수학여행"),
    ("초등학교 때 엄청 유행했던 댄스곡", "elementary", "초등학교 때"),
    ("대학교 신입생 때 축제에서 다 같이 부르던 걸그룹 댄스곡", "freshman", "대학교 신입생 때"),
    ("대학 다닐 때 자주 듣던 밴드 노래", "college", "대학 다닐 때"),
    ("군대에서 자주 듣던 발라드", "military", "군대에서"),
    ("어릴 때 엄마가 차에서 자주 틀어주던 여자 가수 노래", "childhood", "어릴 때"),
    ("학창 시절에 맨날 듣던 노래", "school", "학창 시절"),
    ("스무 살 때 한창 듣던 노래", "age", "스무 살 때"),
    ("25살 때 자주 들었던 힙합", "age", "25살 때"),
])
def test_life_stage_is_detected_as_a_time_clue(query, stage, text):
    found = qa._extract_life_stage(query)
    assert found is not None and (found["stage"], found["text"]) == (stage, text)
    assert found["age_from"] <= found["age_to"] and 0 < found["confidence"] <= 1


@pytest.mark.parametrize("query", [
    # 가사 내용 — v09 n001·n039
    "어릴 때 집이 어려워서 엄마가 짜장면이 싫다고 했다는 얘기 나오는 노래 뭐였지?",
    "무한도전에서 MC랑 가수가 팀 짜서 진지하게 불렀던 노래. '나 스무 살 적에' 이런 가사로 시작했던 것 같은데",
    # 곡의 사건 — v10 초안
    "군대 가 있는 동안 역주행해서 차트 1위까지 한 남자 솔로 노래",
    "판타지 소설 속 마법 주문 같은 단어가 제목인 여자 초등학생들에게 인기좋은 걸그룹 노래",
    "대학가요제에서 대상 받고 광고에도 나왔던 노래 있잖아. 잔잔한 발라드였던 것 같아, 2005년쯤",
    "몇 년 지나서 대학 축제 영상 때문에 다시 떴어",
    "비 오는 날 듣기 좋은 재즈",
])
def test_lyric_content_and_song_events_are_not_life_stage(query):
    assert qa._extract_life_stage(query) is None


def _model_raw(**over):
    raw = {
        "intent_type": "mixed", "korean_tags": ["발라드", "남성보컬", "추억", "어린시절", "수학여행"],
        "lyric_keywords": [], "lyric_clues": [], "lyric_semantic_query": "", "text_alpha": 0.5,
        "release_era": {"start_year": 2000, "end_year": 2015, "confidence": 0.4},  # 10/10 실제 모델 출력 — 근거 없는 추측
        "artist_type": {"values": [], "confidence": 0.0},
        "performance_clues": {"vocal_count": None, "vocal_roles": [], "sound_ensemble": [], "confidence": 0.0},
    }
    raw.update(over)
    return raw


def test_safeguards_drop_guessed_era_and_leaked_tags_when_life_stage_present():
    out = qa._apply_metadata_safeguards("중학교 때 많이 듣던 남자 발라드", _model_raw())
    assert out["life_stage"]["stage"] == "middle"
    assert out["release_era"] == {"start_year": None, "end_year": None, "confidence": 0.0}
    assert out["korean_tags"] == ["발라드", "남성보컬"]


def test_absolute_era_wins_over_life_stage():
    out = qa._apply_metadata_safeguards("2000년대 중학교 때 듣던 댄스곡", _model_raw())
    assert out["life_stage"]["stage"] == "middle"
    assert (out["release_era"]["start_year"], out["release_era"]["end_year"]) == (2000, 2009)


def test_without_life_stage_tags_and_model_era_are_kept():
    """'어린 시절' 이야기를 담은 가사를 찾는 질의에서는 태그도 시기도 건드리지 않는다."""
    out = qa._apply_metadata_safeguards("어린 시절 이야기를 담은 잔잔한 발라드", _model_raw(release_era={"start_year": 2010, "end_year": 2019, "confidence": 0.5}))
    assert out["life_stage"]["stage"] is None
    assert "어린시절" in out["korean_tags"]
    assert out["release_era"]["start_year"] == 2010


def test_search_text_strips_the_phrase_but_original_query_stays():
    analysis = qa._fallback("중학교 때 많이 듣던 남자 발라드인데 제목이 기억 안 나")
    assert analysis.has_life_stage
    assert analysis.original_query.startswith("중학교 때")
    assert analysis.search_text == "많이 듣던 남자 발라드인데 제목이 기억 안 나"
    plain = qa._fallback("비 오는 날 듣기 좋은 재즈")
    assert not plain.has_life_stage and plain.search_text == plain.original_query


def test_default_life_stage_clue_is_empty():
    a = qa._fallback("비 오는 날 듣기 좋은 재즈")
    assert a.life_stage == LifeStageClue()
    assert not a.has_life_stage and a.search_text == a.original_query
    assert "life_stage" in QueryAnalysis.model_fields  # 캐시·API 응답에 실린다


@pytest.mark.parametrize("life, birth, expected", [
    ({"age_from": 13, "age_to": 15, "confidence": 0.6}, 2001, (2013, 2017, 0.6)),   # 01년생 중학교
    ({"age_from": 13, "age_to": 15, "confidence": 0.6}, 1990, (2002, 2006, 0.6)),   # 90년생 중학교
    ({"age_from": 4, "age_to": 12, "confidence": 0.3}, 2001, (2004, 2014, 0.3)),    # 어릴 때는 넓고 약하게
])
def test_release_era_from_birth_year(life, birth, expected):
    era = qa.release_era_from_birth_year(life, birth)
    assert (era["start_year"], era["end_year"], era["confidence"]) == expected


def test_release_era_from_birth_year_rejects_missing_or_absurd_input():
    assert qa.release_era_from_birth_year({"stage": None}, 2001) is None
    assert qa.release_era_from_birth_year({"age_from": 13, "age_to": 15, "confidence": 0.6}, 1800) is None
