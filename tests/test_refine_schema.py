"""
tests/test_refine_schema.py

LLM 정제 결과 검증(refine_data.validate_llm_schema)과 실패 신호의 회귀 테스트.

배경 (2026-09-18 점검):
- JSON 객체 여부만 보고 저장했다. 태그가 문자열로 오면 main의 no_space()가 글자 단위로
  분해했다. 성별·유형의 허용값도 보지 않았다.
- Gemini 호출이 세 번 모두 실패하면 '분석실패' 값을 채운 사전을 돌려줬고, 호출부가
  정상 레코드처럼 저장했다. 이제 빈 사전이 실패 신호다.

Gemini는 가짜다.

실행:
    venv/bin/python -m pytest tests/test_refine_schema.py -v
"""
from __future__ import annotations

import json
import types

import pytest

import src.crawler.scripts_py.refine_data as refine

FULL = {
    "album_summary": "앨범 소개 요약", "artist_type": ["솔로"], "vocal_gender": "여성",
    "lyrics_highlight": "가장 기억에 남는 한 줄", "lyrics_summary": "가사 요약",
    "search_style_summary": "상황 묘사", "mood_tags": ["잔잔함"], "time_weather_tags": ["새벽"],
    "place_activity_tags": ["창가"], "emotion_tags": ["그리움"], "vibe_tags": ["몽환적"],
    "relation_context_tags": ["이별"], "color_tags": ["파랑"], "sound_tags": ["피아노"],
    "visual_imagery": ["비 오는 창가"], "sentiment_summary": "댓글 요약", "fans_tags": ["인생곡"],
    "major_emotion": "슬픔", "context_tags": [], "fact_summary": "",
}


def payload(**overrides):
    return {**FULL, **overrides}


# --- 구조 위반은 재요청 --------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        {"mood_tags": "잔잔함, 새벽감성"},            # 배열 자리에 문자열 -> 글자 단위 분해 경로
        {"artist_type": "솔로"},
        {"lyrics_summary": None},
        {"mood_tags": {"a": 1}},
    ],
)
def test_wrong_types_raise(bad) -> None:
    with pytest.raises(ValueError):
        refine.validate_llm_schema(payload(**bad))


def test_missing_field_raises() -> None:
    broken = payload()
    del broken["emotion_tags"]
    with pytest.raises(ValueError, match="emotion_tags"):
        refine.validate_llm_schema(broken)


def test_top_level_must_be_an_object() -> None:
    with pytest.raises(ValueError):
        refine.validate_llm_schema(["not", "a", "dict"])


# --- 값 표기는 고쳐 준다 -------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [("남자", "남성"), ("male", "남성"), ("여성 보컬", "여성"), ("Mixed", "혼성"), ("남녀 듀엣", "혼성"),
     ("외계인", "unknown"), ("", "unknown"), (["여성"], "여성"), (["남성", "여성"], "혼성")],
)
def test_vocal_gender_is_normalized(value, expected) -> None:
    assert refine.normalize_vocal_gender(value) == expected


@pytest.mark.parametrize(
    "value",
    ["female", "Female", "FEMALE", "female vocal", "woman", "women", "girl", "girls", "lady",
     "여성 듀엣", "여성 듀오", "girl group"],
)
def test_female_english_is_not_matched_as_male(value) -> None:
    """부분 문자열로 비교하면 'female'에 'male'이, 'woman'에 'man'이 들어 있어 남성이 된다.

    실제로 여성 보컬이 남성으로 저장됐다. vocal_gender는 재질문이 묻는 두 슬롯 중 하나이고
    회수율이 가장 높은 슬롯이라(v07 개입 9건 중 7건) 조용히 뒤집히면 타격이 크다.
    """
    assert refine.normalize_vocal_gender(value) == "여성"


@pytest.mark.parametrize("value", ["male", "man", "men", "boy", "boys", "남성 듀엣", "남자 듀오"])
def test_male_spellings_stay_male(value) -> None:
    assert refine.normalize_vocal_gender(value) == "남성"


@pytest.mark.parametrize("value", ["남성 듀엣", "여성 듀엣", "남성 듀오", "여성 그룹"])
def test_duet_alone_does_not_make_a_song_mixed_gender(value) -> None:
    """편성과 성별은 별개다. 예전에는 '듀엣'이 혼성 표현이라 남성·여성 듀엣이 모두 혼성이 됐다."""
    assert refine.normalize_vocal_gender(value) in {"남성", "여성"}


def test_duet_without_a_gender_word_is_unknown() -> None:
    assert refine.normalize_vocal_gender("duet") == "unknown"
    assert refine.normalize_vocal_gender("듀엣") == "unknown"


def test_normalized_values_stay_inside_the_controlled_sets() -> None:
    samples = ["남자", "female", "duet", "외계인", "", "혼성 그룹", None, 3, ["남성"], {"a": 1}]
    assert {refine.normalize_vocal_gender(v) for v in samples} <= refine.VOCAL_GENDERS
    for value in samples:
        assert set(refine.normalize_artist_type(value)) <= refine.ARTIST_TYPES, value


@pytest.mark.parametrize(
    "value, expected",
    [(["걸그룹"], ["그룹"]), (["힙합듀오"], ["듀오"]), (["싱어송라이터", "솔로"], ["솔로"]),
     (["남성중창단"], ["그룹"]), (["록밴드", "혼성"], ["밴드", "혼성"]), (["피처링"], ["unknown"]),
     ([], ["unknown"]), (["unknown", "그룹"], ["그룹"]),
     (["Band"], ["밴드"]), (["Duo"], ["듀오"]), (["girl group"], ["그룹"]), (["Rapper"], ["솔로"]),
     (["듀엣"], ["듀오"])],
)
def test_artist_type_is_normalized(value, expected) -> None:
    assert refine.normalize_artist_type(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [(["듀오", "보컬리스트"], ["듀오"]), (["프로듀서", "듀오"], ["듀오"]), (["그룹", "래퍼"], ["그룹"]),
     (["밴드", "싱어송라이터"], ["밴드"]), (["솔로", "콜라보레이션"], ["솔로", "그룹"])],
)
def test_role_words_do_not_add_a_second_ensemble_value(value, expected) -> None:
    """역할(보컬리스트·프로듀서·래퍼)은 편성을 말하지 않는다.

    항목마다 따로 옮기면 ['듀오','보컬리스트']가 ['듀오','솔로']가 되고, 한 곡이 듀오이면서
    솔로라고 주장한다. metadata.type은 재질문(clarify.answer_matches)과 검색 가산점이 읽으므로,
    듀오를 솔로로 답한 사용자에게 이 곡이 맞는 것으로 처리된다.
    마지막 사례는 둘 다 편성 표기라 그대로 둔다(솔로 + 콜라보 = 그룹).
    """
    assert refine.normalize_artist_type(value) == expected


def test_list_items_that_are_not_strings_are_dropped() -> None:
    out = refine.validate_llm_schema(payload(mood_tags=["잔잔함", 3, None, "잔잔함", " "]))
    assert out["mood_tags"] == ["잔잔함"]


def test_highlight_given_as_lines_is_joined() -> None:
    out = refine.validate_llm_schema(payload(lyrics_highlight=["첫 줄", "둘째 줄"]))
    assert out["lyrics_highlight"] == "첫 줄 / 둘째 줄"


def test_optional_fields_tolerate_null() -> None:
    out = refine.validate_llm_schema(payload(major_emotion=None, context_tags=None, fact_summary=None))
    assert (out["major_emotion"], out["context_tags"], out["fact_summary"]) == ("", [], "")


# --- refine_data: 실패는 빈 사전 -------------------------------------------------------

@pytest.fixture
def fake_gemini(monkeypatch):
    monkeypatch.setattr(refine, "GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(refine.time, "sleep", lambda s: None)
    calls = []

    def install(responses):
        queue = list(responses)

        class _Models:
            def generate_content(self, **kwargs):
                calls.append(1)
                item = queue.pop(0)
                if isinstance(item, Exception):
                    raise item
                return types.SimpleNamespace(text=json.dumps(item, ensure_ascii=False))

        class _Client:
            def __init__(self, api_key=None):
                self.models = _Models()

        monkeypatch.setattr(refine.genai, "Client", _Client)
        return calls
    return install


def test_total_failure_returns_empty_dict_not_placeholder_values(fake_gemini) -> None:
    calls = fake_gemini([RuntimeError("quota")] * 3)
    result = refine.refine_data({"title": "곡", "artist": ["가수"]}, "가사", None)
    assert result == {}
    assert len(calls) == 3


def test_schema_violation_is_retried_and_then_fails(fake_gemini) -> None:
    calls = fake_gemini([payload(mood_tags="문자열")] * 3)
    assert refine.refine_data({"title": "곡", "artist": ["가수"]}, "가사", None) == {}
    assert len(calls) == 3


def test_schema_violation_recovers_on_retry(fake_gemini) -> None:
    fake_gemini([payload(mood_tags="문자열"), payload(vocal_gender="남자", artist_type=["걸그룹"])])
    result = refine.refine_data({"title": "곡", "artist": ["가수"]}, "가사", None)
    assert result["vocal_gender"] == "남성"
    assert result["artist_type"] == ["그룹"]
    assert result["mood_tags"] == ["잔잔함"]


def test_missing_api_key_is_also_a_failure(monkeypatch) -> None:
    monkeypatch.setattr(refine, "GEMINI_API_KEY", "")
    assert refine.refine_data({"title": "곡"}, "가사", None) == {}


def test_female_answer_survives_the_whole_llm_path(fake_gemini) -> None:
    """헬퍼만 고쳐도 refine_data가 그 헬퍼를 부르지 않으면 의미가 없다."""
    fake_gemini([payload(vocal_gender="Female vocalist")])
    result = refine.refine_data({"title": "곡", "artist": ["가수"]}, "가사", None)
    assert result["vocal_gender"] == "여성"


def test_zero_comments_blank_the_response_summary(fake_gemini) -> None:
    """댓글 블록이 비면 LLM은 지어내거나 '댓글이 없어 알 수 없다'고 답한다. 둘 다 저장하지 않는다."""
    fake_gemini([payload(sentiment_summary="청자들은 깊은 위로를 받는다", fans_tags=["인생곡"])])
    result = refine.refine_data({"title": "곡", "melon_comments": []}, "가사", {"comments": []})
    assert result["sentiment_summary"] == ""
    assert result["fans_tags"] == []


def test_comments_present_keep_the_response_summary(fake_gemini) -> None:
    fake_gemini([payload(sentiment_summary="청자들은 깊은 위로를 받는다")])
    result = refine.refine_data(
        {"title": "곡", "melon_comments": ["새벽에 듣기 좋은 곡"]}, "가사", {"comments": []}
    )
    assert result["sentiment_summary"] == "청자들은 깊은 위로를 받는다"


def test_crawler_and_retrieval_agree_on_artist_type() -> None:
    """같은 필드를 정규화하는 곳이 둘이다 — 크롤러(저장 시점)와 검색 단계(질의 시점).

    두 어휘가 갈라지면 저장된 값이 재질문 답과 맞지 않는다. 크롤러가 내는 편성 값은
    검색 단계가 같은 표기에서 읽어내는 값과 어긋나면 안 된다.
    """
    from src.retrieval.clarify import canonical_artist_types

    samples = ["솔로", "그룹", "듀오", "듀엣", "밴드", "걸그룹", "혼성그룹", "아이돌그룹",
               "인디밴드", "솔로가수", "힙합듀오"]
    for value in samples:
        crawler_side = set(refine.normalize_artist_type([value])) - {"unknown", "혼성"}
        retrieval_side = canonical_artist_types([value])
        if crawler_side and retrieval_side:
            assert crawler_side == retrieval_side, (value, crawler_side, retrieval_side)
