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
    ("중학교 때 좋아하던 노래인데 여자 솔로였어", "middle", "중학교 때"),   # 곡을 목적어로 받는 '좋아하던'은 시간 단서
    ("고등학교 때 반에서 인기 있던 댄스곡", "high", "고등학교 때"),
    # 활용형 — 리뷰 2차
    ("어릴 때 자주 나오던 여자 가수 노래", "childhood", "어릴 때"),
    ("고등학교 때 제일 좋아했던 노래", "high", "고등학교 때"),
    ("초등학교 다닐 때 듣던 발라드", "elementary", "초등학교 다닐 때"),
    ("고등학교 다닐 때 듣던 발라드", "high", "고등학교 다닐 때"),
    ("중학교 다녔을 때 유행하던 걸그룹 노래", "middle", "중학교 다녔을 때"),
    ("중학교 때 유명했던 노래", "middle", "중학교 때"),             # 곡을 목적어로 받으면 시간 단서
    ("어릴 때 엄마가 들려주던 노래", "childhood", "어릴 때"),
    ("고등학교 때 교실에서 자주 들리던 발라드", "high", "고등학교 때"),
    # 동사와 곡 명사 사이의 수식어, 다른 곡 명사 — 리뷰 참고
    ("중학교 때 유명했던 아이돌 노래", "middle", "중학교 때"),
    ("고등학교 때 들리던 그 노래", "high", "고등학교 때"),
    ("어릴 때 엄마가 들려주던 자장가", "childhood", "어릴 때"),
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
    # 부사·'좋아하다'만으로는 시간 단서가 아니다 — 리뷰(#35)
    "어릴 때 엄마가 자주 아팠다는 가사 나오는 노래",
    "중학교 때 좋아하던 사람 얘기하는 가사",
    "고등학교 때 많이 싸웠던 친구한테 사과하는 내용의 노래",
    # 목적어 없는 유명·들려·들리는 가사 내용 — 리뷰 3차
    "중학교 때 유명했던 일진 얘기하는 가사",
    "어릴 때 엄마가 들려주던 옛날 이야기 같은 가사",
    "중학교 때 유명했던 가요제 얘기하는 가사",          # '가요제'는 곡이 아니다
    "고등학교 때 좋아하던 노래방 친구 얘기하는 가사",   # '노래방'도
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
    # '추억'(일반 회상어)·'어린시절'(생애 단계 질의마다 모델이 다는 회상어)은 빠지고, '수학여행'은 잡힌 단계(중학교)의 말이
    # 아니라 남는다 — 장면 단서일 수 있다
    assert out["korean_tags"] == ["발라드", "남성보컬", "수학여행"]


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


def test_tags_naming_song_facts_survive_even_if_they_contain_a_life_stage_word():
    """'대학가요제'는 곡 정보다 — '대학' 부분 일치로 지우면 안 된다(리뷰)."""
    out = qa._apply_metadata_safeguards("중학교 때 듣던 대학가요제 대상 받은 노래",
                                        _model_raw(korean_tags=["대학가요제", "대상", "중학교", "추억", "학창시절", "수학여행송"]))
    assert out["life_stage"]["stage"] == "middle"
    assert out["korean_tags"] == ["대학가요제", "대상", "수학여행송"]


def test_every_life_stage_phrase_outside_quotes_is_removed_from_search_text():
    """대표는 '고3 때'지만 '야자 끝나고'도 시간 표현이라 검색문에서 빠진다(리뷰)."""
    analysis = qa._fallback("고3 때 야자 끝나고 듣던 노래")
    assert analysis.life_stage.text == "고3 때" and len(analysis.life_stage.spans) == 2
    assert analysis.search_text == "듣던 노래"


def test_quoted_lyric_with_the_same_phrase_is_kept():
    """따옴표 안의 '중학교 때'는 가사다 — 위치로 지우므로 남는다(리뷰)."""
    q = '중학교 때 듣던 노래인데 가사에 "중학교 때 널 만났지"가 나와'
    analysis = qa._fallback(q)
    assert analysis.life_stage.stage == "middle" and analysis.life_stage.spans == [[0, 5]]
    assert analysis.search_text == '듣던 노래인데 가사에 "중학교 때 널 만났지"가 나와'
    # 따옴표 안에만 있으면 시간 단서가 아니다
    assert qa._extract_life_stage('가사에 "중학교 때 널 만났지"가 나오는 노래') is None


def test_context_words_next_to_the_phrase_stay_in_search_text():
    """'버스에서 틀었던'은 장면 단서다 — 시간 표현만 빼고 조사가 매달린 채 남지 않게 한다."""
    analysis = qa._fallback("고등학교 수학여행 버스에서 틀었던 신나는 노래, 남자 그룹")
    assert analysis.life_stage.stage == "high"
    assert analysis.search_text == "버스에서 틀었던 신나는 노래, 남자 그룹"


def test_school_name_with_danil_ttae_is_removed_whole():
    a = qa._fallback("초등학교 다닐 때 듣던 발라드")
    assert (a.life_stage.age_from, a.life_stage.age_to) == (7, 12)
    assert a.search_text == "듣던 발라드"


def test_detached_phrase_without_a_listening_verb_is_lyric_content_and_stays():
    """대표('중학교 때 듣던')가 잡혀도 떨어져 있는 '어릴 때 집이 어려웠다는'은 가사 내용 — 검색문·태그 모두 남긴다(리뷰)."""
    q = "중학교 때 듣던 노래인데 어릴 때 집이 어려웠다는 가사가 나와"
    a = qa._fallback(q)
    assert a.life_stage.stage == "middle" and a.life_stage.spans == [[0, 5]]
    assert a.search_text == "듣던 노래인데 어릴 때 집이 어려웠다는 가사가 나와"
    out = qa._apply_metadata_safeguards(q, _model_raw(korean_tags=["발라드", "어린시절", "중학교", "추억", "가난"]))
    assert out["korean_tags"] == ["발라드", "어린시절", "가난"]


@pytest.mark.parametrize("query", [
    '중학교 때 듣던 노래인데 "어릴 때 집이 어려웠다"는 가사가 나와',   # 인용 안의 표현도 내용 단계
    "어릴 때 듣던 노래인데 어린 시절 집이 어려웠다는 가사가 나와",      # 청취 단계와 내용 단계가 같아도 내용 태그는 남긴다
])
def test_content_tags_survive_when_the_stage_is_also_quoted_or_equals_the_listening_stage(query):
    out = qa._apply_metadata_safeguards(query, _model_raw(korean_tags=["발라드", "어린시절", "추억", "가난"]))
    assert out["life_stage"]["stage"] in ("middle", "childhood")
    assert out["korean_tags"] == ["발라드", "어린시절", "가난"]


def test_adjacent_phrase_is_still_removed_with_the_representative():
    a = qa._fallback("고3 때 야자 끝나고 듣던 노래")
    assert len(a.life_stage.spans) == 2 and a.search_text == "듣던 노래"


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
