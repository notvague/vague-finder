"""단계 4 — 무엇을 물을지 고르는 규칙.

여기서 고정하는 것은 대부분 단계 3 측정에서 나온 제약이다.

1. `type`과 `release_era`는 절대 묻지 않는다. 맞게 답해도 해롭거나(type),
   틀리게 답하면 정답이 후보 풀에서 사라진다(release_era).
2. 남은 후보를 가르지 못하는 슬롯은 묻지 않는다. 물어도 아무것도 안 좁혀진다.
3. 물을 게 없으면 None — 그러면 호출부가 Reject-only로 진행한다.
4. 최초 질의에서 이미 말한 성별·장르는 묻지 않는다(2026-10-11). 후보가 반대쪽으로 갈려도 뒤집지 않는다.
   판정은 원문이 근거다 — 분석 필드만으로 확정하지 않고, 분석이 비어 있어도 원문 표현은 처리한다.
   세부 판정 규칙은 tests/query/test_known_slots.py.

fixture의 원문은 분석값과 맞춘다 — 기본 원문에 '발라드'가 있으면 장르 미지정 시험이 새 규칙에 걸린다.
"""
from __future__ import annotations

import pytest

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer, MatchingTrack
from src.retrieval.clarify import (
    ALLOWED_SLOTS,
    analysis_with_answers,
    canonical_artist_types,
    answer_matches,
    apply_answer_bonus,
    build_question,
    discrimination,
    can_ask,
    excluded_slots,
    merge_answer,
    pick_data_question,
    pick_question,
    remaining_data_slots,
    slot_value,
    value_counts,
)
from src.retrieval.known_slots import known_slots_from_query


def _analysis(**kwargs) -> QueryAnalysis:
    base = dict(
        original_query="비 오는 날 듣기 좋은 노래",   # 성별·장르를 말하지 않은 원문
        intent_type="mood",
        image_english_query="",
        audio_english_query="",
    )
    base.update(kwargs)
    return QueryAnalysis(**base)


def _track(sid: str, gender: str | None = None, genre: str | None = None) -> MatchingTrack:
    return MatchingTrack(
        id=sid, score=1.0, title=f"곡{sid}",
        vocal_gender=gender, genre=genre,
    )


def _pool(genders: list, genres: list | None = None) -> list:
    genres = genres or [None] * len(genders)
    return [_track(f"s{i}", g, j) for i, (g, j) in enumerate(zip(genders, genres))]


# ---------------------------------------------------------------------------
# 슬롯 값 읽기
# ---------------------------------------------------------------------------

def test_reads_gender_and_first_genre() -> None:
    track = _track("s1", gender="여성", genre="발라드, 국내드라마")
    assert slot_value(track, "vocal_gender") == "여성"
    # 쉼표로만 나눈다 — '랩/힙합'은 슬래시가 장르명의 일부다.
    assert slot_value(track, "genre") == "발라드"
    assert slot_value(_track("s2", genre="랩/힙합"), "genre") == "랩/힙합"


def test_unknown_gender_is_not_an_option() -> None:
    """선택지로 내보낼 수 없는 값은 없는 것으로 친다."""
    assert slot_value(_track("s1", gender="미상"), "vocal_gender") is None
    assert slot_value(_track("s1"), "vocal_gender") is None


# ---------------------------------------------------------------------------
# 가르는 정도
# ---------------------------------------------------------------------------

def test_uniform_candidates_discriminate_nothing() -> None:
    """전부 남성이면 성별을 물어도 아무것도 좁혀지지 않는다."""
    assert discrimination(value_counts(_pool(["남성"] * 10), "vocal_gender")) == 0.0


def test_even_split_discriminates_most() -> None:
    even = discrimination(value_counts(_pool(["남성"] * 5 + ["여성"] * 5), "vocal_gender"))
    lopsided = discrimination(value_counts(_pool(["남성"] * 9 + ["여성"]), "vocal_gender"))
    assert even == pytest.approx(0.5)
    assert lopsided < even


def test_more_values_beat_fewer_when_both_balanced() -> None:
    """네 갈래로 고르게 갈리면 두 갈래보다 더 많이 좁힌다.

    이 성질 때문에 discrimination 점수로 슬롯을 고르면 안 된다 — 장르는 값이
    4~6종이라 성별보다 항상 높게 나오는데 실제 회수 효과는 반대다(v02에서 5/9).
    선택은 ALLOWED_SLOTS 순서가 한다. 이 함수는 "가를 수 있는가"의 판정에만 쓴다.
    """
    two = discrimination(value_counts(_pool(["남성"] * 4 + ["여성"] * 4), "vocal_gender"))
    four = discrimination(value_counts(
        _pool([None] * 8, ["발라드", "댄스", "록/메탈", "랩/힙합"] * 2), "genre"))
    assert four > two


# ---------------------------------------------------------------------------
# 묻지 않을 조건
# ---------------------------------------------------------------------------

def test_never_asks_the_unsafe_slots() -> None:
    """type은 맞게 답해도 해롭고, release_era는 틀리면 정답을 잃는다 (단계 3)."""
    assert "type" not in ALLOWED_SLOTS
    assert "release_era" not in ALLOWED_SLOTS
    assert set(ALLOWED_SLOTS) == {"vocal_gender", "genre"}


def _mixed_pool() -> list:
    """성별로도 장르로도 갈리는 후보 — 어느 쪽이든 물을 수 있는 상태."""
    genres = ["발라드", "댄스", "록/메탈", "랩/힙합", "R&B/Soul"] * 2
    return [_track(f"s{i}", "남성" if i < 5 else "여성", genres[i]) for i in range(10)]


def test_does_not_ask_what_the_user_already_said_in_the_query() -> None:
    """이미 말한 슬롯은 묻지 않는다 — 후보가 반대쪽으로 갈려 있어도.

    v02까지는 오기억(c701·c705·q101)을 잡으려고 이미 말한 슬롯도 다시 물었다. 실제 화면에서는
    "남자 아이돌 노래"라고 한 사용자에게 성별을 또 묻는 것이 되돌아왔고, 2026-10-11에 정책을 바꿨다.
    후보가 다양하다는 것은 사용자가 틀렸다는 증거가 아니다. 오기억 재확인은 별도 설계다.
    """
    pool = _mixed_pool()
    # 사용자 예시 1 — 성별은 말했고 장르는 모른다 → 성별을 건너뛰고 장르를 묻는다
    question = pick_question(_analysis(original_query="중학생 때 나온 엄청 유명한 남자 아이돌 노래"), pool)
    assert question is not None and question.slot == "genre"
    # 사용자 예시 2 — 둘 다 말했다 → 물을 것이 없다(거절만 진행)
    assert pick_question(_analysis(original_query="남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어"), pool) is None
    assert pick_question(_analysis(original_query="여자 보컬의 발라드"), pool) is None
    # 장르만 말했으면 성별은 묻는다
    assert pick_question(_analysis(original_query="비 오는 날 혼자 듣기 좋은 잔잔한 발라드"), pool).slot == "vocal_gender"


def test_known_slots_come_from_the_original_text_not_the_analysis_fields() -> None:
    """분석의 vocal_gender·genre는 근거가 아니다 — 모델 추측일 수 있고(채워져 있어도 묻는다), 규칙 폴백이면 비어
    있다(비어 있어도 원문이 말했으면 묻지 않는다). 사용자 예시 둘은 폴백 분석에서 두 칸이 비어 있었다."""
    pool = _mixed_pool()
    # 분석만 채워지고 원문에는 없다 → 모델 추측 — 그대로 묻는다
    guessed = _analysis(vocal_gender="여성", genre="발라드")
    assert known_slots_from_query(guessed) == []
    assert pick_question(guessed, pool).slot == "vocal_gender"
    # 원문은 말했는데 분석은 비었다(폴백) → 묻지 않는다
    blank = _analysis(original_query="남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어", vocal_gender=None, genre="")
    assert known_slots_from_query(blank) == ["vocal_gender", "genre"]
    assert pick_question(blank, pool) is None
    # 정상 분석(둘 다 채워짐)도 같은 결과
    filled = _analysis(original_query="남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어", vocal_gender="남성", genre="인디음악")
    assert pick_question(filled, pool) is None


def test_known_slots_do_not_touch_history_or_answers() -> None:
    """known_slots는 질문 선택에서만 빠진다 — asked_slots·answers에 끼워 넣지 않고, 한도에도 세지 않는다."""
    assert excluded_slots(["birth_year"], ["genre"]) == {"birth_year", "genre"}
    assert remaining_data_slots([], []) == list(ALLOWED_SLOTS)
    assert remaining_data_slots(["vocal_gender"], []) == ["genre"]
    assert remaining_data_slots([], ["vocal_gender"]) == ["genre"]
    assert remaining_data_slots(["birth_year"], ["vocal_gender", "genre"]) == []
    # 분석값을 바꾸지 않는다
    analysis = _analysis(original_query="남자 아이돌 노래")
    before = analysis.model_dump()
    pick_question(analysis, _mixed_pool())
    assert analysis.model_dump() == before


def test_does_not_ask_the_same_slot_twice() -> None:
    pool = _pool(["남성"] * 5 + ["여성"] * 5)
    assert pick_question(_analysis(), pool, asked_slots=["vocal_gender"]) is None


def test_does_not_ask_when_metadata_is_mostly_missing() -> None:
    """값을 가진 후보가 적으면 답변을 받아도 대부분을 판별할 수 없다."""
    pool = _pool(["남성", "여성"] + [None] * 8)
    assert not can_ask(pool, "vocal_gender")


def test_does_not_ask_when_the_split_is_too_lopsided() -> None:
    pool = _pool(["남성"] * 19 + ["여성"])
    assert not can_ask(pool, "vocal_gender")


def test_returns_none_for_an_empty_pool() -> None:
    assert pick_question(_analysis(), []) is None
    assert pick_data_question([]) is None


def test_data_question_needs_only_the_candidates() -> None:
    """질문만 조회하는 경로는 분석 없이 후보만 넘긴다. 같은 후보면 `pick_question`과 같은 질문이어야 한다
    — 두 경로가 갈리면 화면(지연 조회)과 재질문 측정(즉시 계산)이 다른 질문을 낸다."""
    pool = _pool(["남성"] * 6 + ["여성"] * 4)
    assert pick_data_question(pool) == pick_question(_analysis(), pool)
    assert pick_data_question(pool).slot == "vocal_gender"
    for asked in (["vocal_gender"], ["birth_year"], ["birth_year", "vocal_gender"], list(ALLOWED_SLOTS)):
        assert pick_data_question(pool, asked) == pick_question(_analysis(), pool, asked_slots=asked)
    # 이미 말한 슬롯 — 지연 조회는 응답의 known_slots를 받고, 즉시 질문은 원문에서 같은 것을 찾는다. 결과가 같아야 한다
    mixed = _mixed_pool()
    for query, known in (("남자 아이돌 노래", ["vocal_gender"]), ("잔잔한 발라드", ["genre"]),
                         ("여자 보컬의 발라드", ["vocal_gender", "genre"]), ("비 오는 날 노래", [])):
        analysis = _analysis(original_query=query)
        assert known_slots_from_query(analysis) == known
        for asked in ([], ["birth_year"], ["vocal_gender"], ["genre"]):
            assert pick_data_question(mixed, asked, known) == pick_question(analysis, mixed, asked_slots=asked), (query, asked)


# ---------------------------------------------------------------------------
# 슬롯 고르기
# ---------------------------------------------------------------------------

def test_always_prefers_gender_when_it_can_split() -> None:
    """실측에서 성별 7/9 > 장르 5/9. 성별이 가를 수 있으면 무조건 성별이다.

    점수 비교가 아니라 순서다 — 장르가 더 잘게 갈라도 성별이 먼저다.
    """
    # 장르를 일부러 더 잘게 쪼개 둔다. 그래도 성별이 뽑혀야 한다.
    genres = ["발라드", "댄스", "록/메탈", "랩/힙합", "R&B/Soul"] * 2
    pool = [
        _track(f"s{i}", "남성" if i < 5 else "여성", genres[i])
        for i in range(10)
    ]
    question = pick_question(_analysis(), pool)
    assert question is not None
    assert question.slot == "vocal_gender"


def test_falls_back_to_genre_when_gender_cannot_split() -> None:
    """q203 상황 — 남은 후보가 전부 남성이면 성별은 물어도 소용이 없다.

    실측에서 q203은 성별을 정확히 답해도 순위가 그대로였고(후보 14위 유지)
    장르를 답했을 때만 5위로 올라왔다.
    """
    pool = [
        _track(f"s{i}", "남성", "록/메탈" if i < 5 else "발라드")
        for i in range(10)
    ]
    question = pick_question(_analysis(), pool)
    assert question is not None
    assert question.slot == "genre"


def test_returns_none_when_no_slot_helps() -> None:
    """물을 게 없으면 None. 호출부는 Reject-only로 진행한다."""
    pool = [_track(f"s{i}", "남성", "발라드") for i in range(10)]
    assert pick_question(_analysis(), pool) is None


# ---------------------------------------------------------------------------
# 질문 만들기
# ---------------------------------------------------------------------------

def test_options_come_from_the_candidates_with_counts() -> None:
    pool = _pool(["남성"] * 6 + ["여성"] * 4)
    question = build_question("vocal_gender", pool)

    assert [o.value for o in question.options] == ["남성", "여성"]
    assert [o.count for o in question.options] == [6, 4]
    assert question.question == "부른 사람 목소리는 어느 쪽이었나요?"


def test_reason_states_the_actual_split() -> None:
    """'왜 이걸 묻는지'를 화면에 보여줄 수 있어야 한다."""
    question = build_question("vocal_gender", _pool(["남성"] * 6 + ["여성"] * 4))
    assert "10곡" in question.reason
    assert "남성 6" in question.reason and "여성 4" in question.reason


def test_skip_option_is_not_a_candidate_value() -> None:
    """'잘 모르겠어요'는 후보에서 나온 값이 아니라 UI가 붙이는 버튼이다."""
    question = build_question("vocal_gender", _pool(["남성"] * 6 + ["여성"] * 4))
    assert all(o.value != "잘 모르겠어요" for o in question.options)


# ---------------------------------------------------------------------------
# 답변 반영 — 재검색이 아니라 후보 재정렬
# ---------------------------------------------------------------------------

BOOST_UNIT = 1.0 / 61  # search_router의 boost_unit과 같은 값


def _scored(pool):
    """점수 내림차순 후보. 앞에 있을수록 순위가 높다."""
    return [(t.id, 1.0 - i * 0.001) for i, t in enumerate(pool)]


def test_matching_candidates_move_up() -> None:
    pool = [_track("a", "남성"), _track("b", "여성")]
    out = apply_answer_bonus(_scored(pool), {t.id: t for t in pool},
                             [ClarifyAnswer(slot="vocal_gender", value="여성")],
                             BOOST_UNIT)
    assert [sid for sid, _ in out] == ["b", "a"]


def test_the_candidate_pool_itself_never_changes() -> None:
    """이게 이 함수의 전부다 — 재검색이 아니라 재정렬이므로 후보가 늘거나 줄지 않는다.

    회귀 이력: 답변을 분석에 병합해 재검색했더니 맞는 답변인데도 정답이
    후보 4위에서 15위로 밀렸다(test split q200). 후보 구성을 건드리지 않으면
    그런 일이 생기지 않는다.
    """
    pool = [_track(f"s{i}", "남성" if i % 2 else "여성", "발라드") for i in range(10)]
    scored = _scored(pool)
    out = apply_answer_bonus(scored, {t.id: t for t in pool},
                             [ClarifyAnswer(slot="vocal_gender", value="여성")],
                             BOOST_UNIT)
    assert {sid for sid, _ in out} == {sid for sid, _ in scored}
    assert len(out) == len(scored)


def test_no_answers_leaves_the_order_alone() -> None:
    pool = [_track("a", "남성"), _track("b", "여성")]
    scored = _scored(pool)
    assert apply_answer_bonus(scored, {t.id: t for t in pool}, [], BOOST_UNIT) == scored


def test_skipped_answer_changes_nothing() -> None:
    """'잘 모르겠어요'는 어떤 후보에도 보너스를 주지 않는다."""
    pool = [_track("a", "남성"), _track("b", "여성")]
    scored = _scored(pool)
    out = apply_answer_bonus(scored, {t.id: t for t in pool},
                             [ClarifyAnswer(slot="vocal_gender", skipped=True)],
                             BOOST_UNIT)
    assert out == scored


def test_mismatching_candidates_are_not_penalised() -> None:
    """반대쪽 후보를 깎지 않는다 — 사람의 기억이 틀렸을 수 있다."""
    pool = [_track("a", "남성"), _track("b", "여성")]
    scored = dict(_scored(pool))
    out = dict(apply_answer_bonus(list(scored.items()), {t.id: t for t in pool},
                                  [ClarifyAnswer(slot="vocal_gender", value="여성")],
                                  BOOST_UNIT))
    assert out["a"] == scored["a"]
    assert out["b"] > scored["b"]


def test_multiplier_scales_the_bonus() -> None:
    """보정용 파라미터. 기본값은 운영 상수이고, 하네스가 배수를 바꿔 가며 잰다."""
    pool = [_track("a", "남성"), _track("b", "여성")]
    tracks = {t.id: t for t in pool}
    answer = [ClarifyAnswer(slot="vocal_gender", value="여성")]

    small = dict(apply_answer_bonus(_scored(pool), tracks, answer, BOOST_UNIT, multiplier=1.0))
    large = dict(apply_answer_bonus(_scored(pool), tracks, answer, BOOST_UNIT, multiplier=5.0))

    assert large["b"] > small["b"]
    # 일치하지 않는 후보는 배수와 무관하게 그대로다.
    assert large["a"] == small["a"]


def test_genre_answer_matches_compound_values() -> None:
    track = _track("a", genre="발라드, 국내드라마")
    assert answer_matches(track, ClarifyAnswer(slot="genre", value="발라드"))
    assert not answer_matches(track, ClarifyAnswer(slot="genre", value="댄스"))


def test_matches_artist_type_through_canonical_forms() -> None:
    """코퍼스 type은 '혼성그룹', '듀엣'처럼 자유 문자열이라 접어서 비교해야 한다.

    회귀 이력: type과 release_era를 answer_matches에서 빠뜨려 하네스의 해당
    정책 네 개가 53/53건에서 reject_only와 완전히 같은 값을 냈다.
    "효과 없음"이 아니라 "반영 안 됨"이었는데 슬롯별 효과로 읽을 뻔했다.
    """
    track = MatchingTrack(id="a", score=1.0, title="A",
                          artist_types=["혼성그룹", "듀엣"])
    assert answer_matches(track, ClarifyAnswer(slot="type", value="그룹"))
    assert answer_matches(track, ClarifyAnswer(slot="type", value="듀오"))
    assert not answer_matches(track, ClarifyAnswer(slot="type", value="밴드"))
    assert canonical_artist_types(["혼성그룹", "듀엣"]) == {"그룹", "듀오"}


def test_matches_release_era_by_decade() -> None:
    track = MatchingTrack(id="a", score=1.0, title="A", release_date="2013.03.04")
    assert answer_matches(track, ClarifyAnswer(slot="release_era", value="2010년대"))
    assert not answer_matches(track, ClarifyAnswer(slot="release_era", value="2000년대"))
    # 연도를 못 읽으면 일치로 치지 않는다.
    assert not answer_matches(MatchingTrack(id="b", score=1.0, title="B"),
                              ClarifyAnswer(slot="release_era", value="2010년대"))


def test_all_four_slots_actually_do_something() -> None:
    """네 슬롯 모두 재정렬에 영향을 줘야 한다. 하나라도 no-op이면 측정이 거짓말을 한다."""
    track = MatchingTrack(id="a", score=1.0, title="A", vocal_gender="여성",
                          genre="발라드", artist_types=["듀오"], release_date="2013.01.01")
    answers = {"vocal_gender": "여성", "genre": "발라드",
               "type": "듀오", "release_era": "2010년대"}
    for slot, value in answers.items():
        assert answer_matches(track, ClarifyAnswer(slot=slot, value=value)), slot


# ---------------------------------------------------------------------------
# 정정된 답변이 리랭크 그룹 정렬에도 반영되는가
# ---------------------------------------------------------------------------

def test_analysis_with_answers_returns_a_corrected_copy() -> None:
    """검색은 원래 분석으로 하지만, 검색 이후 순위 비교에는 정정본이 필요하다.

    회귀 이력: exact 가사 후보를 성별·장르로 다시 묶는 자리에 원래 분석을
    넘겨서, "남성"이라고 잘못 기억했다가 "여성"으로 정정해도 남성 그룹이
    먼저 왔다. 보너스로 후보 1위가 된 여성 곡이 최종 Top-10에서 밀려났다.
    """
    analysis = _analysis(vocal_gender="남성")
    corrected = analysis_with_answers(
        analysis, [ClarifyAnswer(slot="vocal_gender", value="여성")])

    assert corrected.vocal_gender == "여성"
    assert analysis.vocal_gender == "남성", "원본이 바뀌었다"


def test_analysis_with_answers_is_a_noop_without_answers() -> None:
    analysis = _analysis(vocal_gender="남성")
    assert analysis_with_answers(analysis, []) is analysis


def test_corrected_analysis_changes_exact_lyric_grouping() -> None:
    """정정본을 쓰면 가사 exact 그룹 정렬이 실제로 뒤집힌다."""
    from src.retrieval.search_router import exact_lyric_constraint_score

    male = MatchingTrack(id="m", score=1.0, title="M", vocal_gender="남성")
    female = MatchingTrack(id="f", score=1.0, title="F", vocal_gender="여성")
    analysis = _analysis(vocal_gender="남성")

    # 정정 전에는 남성이 앞선다.
    assert exact_lyric_constraint_score(analysis, male) > \
           exact_lyric_constraint_score(analysis, female)

    corrected = analysis_with_answers(
        analysis, [ClarifyAnswer(slot="vocal_gender", value="여성")])
    assert exact_lyric_constraint_score(corrected, female) > \
           exact_lyric_constraint_score(corrected, male)


# ---------------------------------------------------------------------------
# 답변 병합 — 구 방식 (재검색 경로에서는 더 이상 쓰지 않는다)
# ---------------------------------------------------------------------------

def test_merge_answer_moved_but_behaves_the_same() -> None:
    analysis = _analysis()
    merge_answer(analysis, ClarifyAnswer(slot="vocal_gender", value="여성"))
    assert analysis.vocal_gender == "여성"


def test_merge_answer_still_handles_unasked_slots() -> None:
    """pick_question이 묻지 않아도 옛 클라이언트가 보낼 수 있다."""
    analysis = _analysis()
    merge_answer(analysis, ClarifyAnswer(slot="type", value="그룹"))
    assert analysis.artist_type.values == ["그룹"]
