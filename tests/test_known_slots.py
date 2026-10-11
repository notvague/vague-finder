"""재질문 전용 "이미 말한 슬롯" 판정(`src/retrieval/known_slots.py`).

원문이 근거다. 가창자를 수식하는 성별, 재질문 선택지에 대응하는 장르만 확정이고, 영상 속 인물·인용·이름·부분 문자열·
불확실·부정 표현은 확정이 아니다. 문맥 단위다 — 한 절의 불확실성이 다른 절의 단서를 지우지 않는다.
"""
from __future__ import annotations

import pytest

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import KNOWN_SLOT_ORDER, ClarifyRequest, _normalize_known_slots
from src.retrieval.clarify import ALLOWED_SLOTS
from src.retrieval.known_slots import (
    KNOWN_SLOT_NAMES,
    known_slot_values,
    known_slots_from_query,
    known_slots_from_text,
)
from src.retrieval.query_analyzer import _fallback


def test_the_three_slot_tuples_are_one_value() -> None:
    """묻는 슬롯(clarify) · 이미 말한 슬롯(known_slots) · 요청/응답 스키마가 같은 이름·순서를 써야 한다."""
    assert tuple(ALLOWED_SLOTS) == tuple(KNOWN_SLOT_NAMES) == tuple(KNOWN_SLOT_ORDER) == ("vocal_gender", "genre")


# --- 사용자 예시 ------------------------------------------------------------------

def test_user_example_1_states_gender_only() -> None:
    assert known_slot_values("중학생 때 나온 엄청 유명한 남자 아이돌 노래") == {"vocal_gender": "남성"}
    assert known_slots_from_text("중학생 때 나온 엄청 유명한 남자 아이돌 노래") == ["vocal_gender"]


def test_user_example_2_states_gender_and_genre() -> None:
    q = "남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어"
    assert known_slot_values(q) == {"vocal_gender": "남성", "genre": "인디"}
    assert known_slots_from_text(q) == ["vocal_gender", "genre"]


@pytest.mark.parametrize("query", [
    "중학생 때 나온 엄청 유명한 남자 아이돌 노래",
    "남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어",
])
def test_fallback_analysis_with_empty_fields_still_counts_the_text(query: str) -> None:
    """규칙 폴백은 vocal_gender=None·genre=''다. 그래도 원문이 말했으면 안다 — 분석 필드만 보는 수정으로는 부족하다."""
    analysis = _fallback(query)
    assert analysis.vocal_gender is None and analysis.genre == ""
    assert known_slots_from_query(analysis) == known_slots_from_text(query)
    assert "vocal_gender" in known_slots_from_query(analysis)


def test_analysis_fields_alone_are_not_evidence() -> None:
    """모델이 채운 값은 추측일 수 있다 — 원문에 없으면 모른다."""
    analysis = QueryAnalysis(original_query="비 오는 날 듣기 좋은 노래", intent_type="mood",
                             image_english_query="", audio_english_query="",
                             vocal_gender="여성", genre="발라드")
    assert known_slots_from_query(analysis) == []


# --- 정상 질문 유지 ---------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "비 오는 날 듣기 좋은 노래", "앨범이 파란 잔잔한 노래", "중학교 때 듣던 노래",
    "앨범 커버가 온통 빨간색이었던 노래를 찾고 있어", "케이팝 노래", "가사에 '가시처럼 깊게 박힌 기억은'이 나오는 노래",
])
def test_queries_without_clues_know_nothing(query: str) -> None:
    assert known_slot_values(query) == {}


def test_one_clue_leaves_the_other_slot_open() -> None:
    assert known_slots_from_text("비 오는 날 혼자 듣기 좋은 잔잔한 발라드") == ["genre"]
    assert known_slots_from_text("남자가 부르는 잔잔한 노래") == ["vocal_gender"]


# --- 성별 문맥 --------------------------------------------------------------------

@pytest.mark.parametrize("query, gender", [
    ("남자 아이돌 노래", "남성"), ("여자 보컬의 발라드", "여성"), ("남자 인디 노래", "남성"),
    ("남자가 부르는 노래", "남성"), ("여자가 혼자 노래하는 곡", "여성"), ("남녀 듀엣 발라드", "혼성"),
    ("남자는 랩하고 여자는 노래하는 곡", "혼성"), ("남성 솔로이고 멜로디는 신났던 것 같아", "남성"),
    ("걸그룹 댄스곡", "여성"), ("2000년대 여자 그룹 댄스곡인데 제목이 알파벳 세 글자였던 것 같아", "여성"),
    ("여자 가수 말고 남자 가수", "남성"), ("여자 목소리였어", "여성"),
    # '부르다' 활용형과 사이 어절 — 가장 흔한 문형(리뷰)
    ("남자가 부른 노래", "남성"), ("여자가 부른 발라드", "여성"), ("남자분이 부른 노래", "남성"), ("여자 혼자 부른 노래", "여성"),
    ("남자가 불러서 유명해진 노래", "남성"), ("남자가 엄청 고음으로 절규하듯 부르는 록 발라드", "남성"),
    ("여자애들 여러 명이 부르는 빠른 댄스곡", "여성"), ("여자들이 일부러 콧소리 내면서 귀엽게 불렀어", "여성"),
    ("남자 둘이 듀엣으로 부른 노래", "남성"), ("오디션 방송에서 남자 참가자가 불러 다시 알려진 곡", "남성"),
    ("허스키한 남자가 커버한 곡이야", "남성"), ("남자는 랩을 하고 여자는 노래를", "혼성"),
    # 인원·결합
    ("남자 둘 인디 듀오", "남성"), ("남자랑 여자가 번갈아 주고받는 듀엣곡", "혼성"), ("남자 둘 여자 하나였어", "혼성"),
    ("남성 4인조 그룹 노래", "남성"), ("여자 3명이 부르는 노래", "여성"), ("남녀 섞인 그룹이었어", "혼성"),
    # 서술어
    ("보컬이 남자야", "남성"), ("가수는 여자였어", "여성"), ("부른 사람이 남자", "남성"), ("목소리가 여자", "여성"),
    # 아티스트
    ("남자 아티스트 발라드", "남성"), ("여자 아티스트 노래", "여성"),
    # 영상 낱말이 같은 절에 있어도 곡 수식은 곡 얘기다
    ("뮤비가 인상적인 남자 발라드", "남성"), ("남자 노래 뮤비에 의자가 떠 있어", "남성"), ("EDM 처럼 신나는 남자 노랜데", "남성"),
])
def test_gender_describes_the_singer(query: str, gender: str) -> None:
    assert known_slot_values(query).get("vocal_gender") == gender


@pytest.mark.parametrize("query", [
    "뮤비에 남자가 의자에 앉아 있어",                 # 영상 속 인물
    "뮤비에서 남자가 노래 부르는 장면이 있어",          # 영상 절의 '부르는'도 가창자 증거가 아니다
    "뮤비에 나오는 사람이 남자",
    "표지에 남자 사진이 있는 노래",                   # 표지 속 인물
    "회색빛 커버였는데 남자 네 명이 아주 작게 흩어져 서 있었어",   # 표지 속 인원
    "초록색 분위기였고 여자 여러명이 킥보드 타고 있는 이미지였어",
    "가사에 \"남자니까\"가 나오는 노래",               # 가사 인용
    "제목이 \"한 남자\"인 노래",                      # 제목 인용
    "아티스트명 남자답게",                           # 이름으로 지정된 구절
    "아티스트 이름이 남자답게인 노래",
    "남자답게라는 가수의 노래",
    "남자 이름이 제목인 노래",                        # 가창자가 아니라 제목 얘기
    "남자친구가 노래방에서 불렀던 노래",               # 남자친구는 가창자가 아니다
    "전 남자친구가 자주 부르던 노래",
    "여자친구한테 부르고 싶은 노래",
    "여자를 위해 부른 노래",                          # 가창의 대상이지 주체가 아니다(리뷰 2차)
    "여자한테 고백하면서 부르는 노래",
    "남자에게 불러주는 노래",
    "남자랑 헤어지고 부른 노래",
    "여자 둘이 주인공인 드라마 OST",                   # 곡 밖의 인물 인원
    "남자 둘이 싸우는 내용의 가사",
    "남자 셋이 나오는 뮤비",
    "여자가 좋아할 만한 노래를 찾고 있어",             # 듣는 사람 — '노래를'은 가창 동사가 아니다(PR 리뷰)
    "남자들이 좋아하는 노래만 모아 둔 앨범",
])
def test_gender_is_not_inferred_from_scenes_quotes_or_names(query: str) -> None:
    assert "vocal_gender" not in known_slot_values(query)


def test_comparison_and_songwriter_are_not_the_singer() -> None:
    assert known_slot_values("여자 목소리인데 남자 록커처럼 쇳소리 섞어서 내지르는 하드록")["vocal_gender"] == "여성"
    assert known_slot_values("여자 가수가 낮고 허스키하게 불렀고, 가사는 다른 그룹 남자 아이돌이 써줬다고 들었어")["vocal_gender"] == "여성"


def test_scene_sentence_after_the_clue_keeps_the_clue() -> None:
    q = "남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어. 가사에 '여자'가 나와"
    assert known_slot_values(q)["vocal_gender"] == "남성"


# --- 불확실·부정 ------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "남자인지 모르겠어", "남자인지 여자인지 모르겠어", "남자 가수인지 여자 가수인지 모르겠어",
    "남자일 수도 있어", "남자 보컬일 수도 있어", "여자는 아니야", "여자 가수는 아니야", "남자 가수 아니야",
    "남자 목소리인지 여자 목소리인지 기억이 안 나",
])
def test_uncertain_or_negated_gender_is_not_known(query: str) -> None:
    assert "vocal_gender" not in known_slot_values(query)


@pytest.mark.parametrize("query", [
    "인디는 아니고 장르는 모르겠어", "발라드인지 록인지 모르겠어", "발라드는 아닌 것 같아", "장르는 모르겠어",
    "래퍼로 유명한 사람이 부른 건데 랩 없이 노래만 했어", "랩 빼고 노래만 있는 곡",
])
def test_uncertain_or_negated_genre_is_not_known(query: str) -> None:
    assert "genre" not in known_slot_values(query)


def test_uncertainty_is_scoped_to_its_own_clause() -> None:
    """`남자 보컬이고 장르는 모르겠어` — 장르 불확실성이 성별을 지우지 않는다."""
    assert known_slot_values("남자 보컬이고 장르는 모르겠어") == {"vocal_gender": "남성"}
    # 뒤의 '모르겠어'가 다른 주제(제목)에 붙은 경우
    assert known_slot_values("남자 아이돌 노래인데 제목이 기억 안 나")["vocal_gender"] == "남성"
    assert known_slot_values("남자 아이돌 노래 제목 모르겠어")["vocal_gender"] == "남성"


@pytest.mark.parametrize("query, expected", [
    # 단서에서 떨어진 "모르겠어/기억 안 나"는 곡·제목이 뭔지 모른다는 말이다 — 이 서비스 질의의 흔한 끝맺음(리뷰)
    ("남자 아이돌 노래 뭔지 모르겠어", {"vocal_gender": "남성"}),
    ("남자 아이돌 노래 기억 안 나", {"vocal_gender": "남성"}),
    ("남자 가수가 부른 발라드 뭐였는지 기억이 안 나", {"vocal_gender": "남성", "genre": "발라드"}),
    ("남자 발라드 찾아줘 뭔지 모르겠어", {"vocal_gender": "남성", "genre": "발라드"}),
    ("남자가 부른 노래 같은데 확실하지 않아", {"vocal_gender": "남성"}),
    # 단서에 바로 붙은 것은 여전히 불확실
    ("남자 가수 노래인지 여자 가수 노래인지 모르겠어", {}),
])
def test_far_away_uncertainty_belongs_to_the_song_not_the_clue(query: str, expected: dict) -> None:
    assert known_slot_values(query) == expected


# --- 장르 --------------------------------------------------------------------------

@pytest.mark.parametrize("query, genre", [
    ("잔잔한 발라드", "발라드"), ("인디록 노래", "인디"), ("인디 팝 느낌", "인디"), ("힙합인데 남자 래퍼", "힙합"),
    ("록발라드", "록"), ("락 음악", "록"), ("알앤비 곡", "알앤비"), ("R&B 노래", "알앤비"), ("댄스곡", "댄스"),
    ("트로트 노래", "트로트"), ("재즈 같은 노래", "재즈"), ("포크 느낌의 노래", "포크"), ("일렉트로니카 곡", "일렉트로니카"),
    ("랩 곡인데 비트가 느려", "힙합"), ("댄스곡인데 뮤비가 멋있어", "댄스"),
    # 어미 — "장르는 발라드야"(리뷰 3)
    ("장르는 발라드야", "발라드"), ("댄스야", "댄스"), ("장르는 인디야", "인디"), ("발라드예요", "발라드"), ("발라드입니다", "발라드"),
    ("발라드지", "발라드"), ("발라드네", "발라드"), ("발라드랑 비슷", "발라드"), ("발라드에 가까워", "발라드"), ("발라드처럼", "발라드"),
    ("발라드만 불러", "발라드"), ("발라드임", "발라드"), ("발라드인 건 확실해", "발라드"), ("느린 발라든데 제목 몰라", "발라드"),
    ("남자가 호소하듯 부르는 발라드야", "발라드"),
])
def test_genre_words_count(query: str, genre: str) -> None:
    assert known_slot_values(query).get("genre") == genre


def test_quoted_genre_value_counts() -> None:
    """따옴표를 일괄 삭제하지 않는다 — 장르를 가리키는 따옴표는 값이다."""
    assert known_slot_values("장르는 '인디'야") == {"genre": "인디"}
    assert known_slot_values("장르는 \"발라드\"였어") == {"genre": "발라드"}


@pytest.mark.parametrize("query", [
    "인디아나 존스 느낌의 노래",       # 부분 문자열
    "블락비 노래",                     # 아티스트명 안의 '락'
    "재즈바 노래", "소울메이트", "메탈릭한 소리", "일렉기타",
    "제목이 '발라드'인 노래",           # 제목 인용
    "가사에 \"힙합\"이 나오는 노래",     # 가사 인용
    "제목에 록이 들어가는 노래",         # 따옴표 없는 제목·가사 언급
    "가사에 발라드라는 말이 나오는 노래",
    # 단독 '랩'은 랩 파트 얘기다(평가 질의 17/17)
    "중간에 랩 파트가 있는 노래", "랩이 들어가고 후렴이 좋은 노래", "남자는 랩을 하고 여자는 노래를",
    "랩 하던 여자가 이번엔 노래 위주로 부른 곡",
    # 악기·장소·행위
    "일렉 기타 소리가 강한 노래", "댄스 브레이크가 있는 노래", "뮤비에서 댄스가 멋있던 노래", "재즈 카페에서 들은 노래",
    "락 페스티벌에서 들은 노래",
])
def test_genre_is_not_a_substring_quote_part_or_place(query: str) -> None:
    assert "genre" not in known_slot_values(query)


# --- 정규화 ------------------------------------------------------------------------

def test_request_schema_dedupes_and_orders_known_slots() -> None:
    assert _normalize_known_slots(["genre", "vocal_gender", "genre"]) == list(KNOWN_SLOT_NAMES)
    assert _normalize_known_slots([]) == []
    assert ClarifyRequest(known_slots=["genre", "genre"]).known_slots == ["genre"]
    assert ClarifyRequest(known_slots=["genre", "vocal_gender"]).known_slots == ["vocal_gender", "genre"]
