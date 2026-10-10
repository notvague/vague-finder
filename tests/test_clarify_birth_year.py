"""재질문 출생 연도 슬롯 — 생애 단계 질의("중학교 때 듣던")의 기준점. NEXT_WORK §2-13 ②.

1. 생애 단계가 있고 시기를 모르면 데이터 슬롯보다 먼저 birth_year를 묻는다. 시기를 알거나 이미 물었으면 안 묻는다.
2. 답(5년 밴드)은 후보 재정렬 보너스가 아니라 분석의 release_era 창이 된다(가산, 필터 아님).
3. 요청의 birth_year(프로필)가 있으면 묻지 않고 바로 창을 쓴다. 응답의 analysis는 원본이다.
4. 리랭커 프롬프트에는 출생 연도가 아니라 그걸로 만든 시기 창이 간다.
"""
from __future__ import annotations

from typing import List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.dependencies import get_query_analyzer, get_search_router
from src.backend.api.routes.search import router as search_route
from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer, MatchingTrack
from src.retrieval import query_analyzer as qa
from src.retrieval.clarify import (
    BIRTH_YEAR_SLOT,
    ALLOWED_SLOTS,
    answer_matches,
    answers_for_reranker,
    apply_birth_year_answers,
    birth_year_band,
    merge_answer,
    parse_birth_year,
    pick_question,
    with_birth_year,
)
from src.retrieval.evaluate_clarification import choose_answer, noisy_birth_year, oracle_birth_year, oracle_value


def _life_stage_analysis(query: str = "중학교 때 많이 듣던 남자 발라드") -> QueryAnalysis:
    """규칙 폴백으로 만든 분석 — 생애 단계는 규칙이 잡으므로 Gemini가 필요 없다."""
    analysis = qa._fallback(query)
    assert analysis.has_life_stage and not analysis.has_release_era
    return analysis


def _track(sid: str, gender: str, genre: str = "발라드", year: str = "2015") -> MatchingTrack:
    return MatchingTrack(id=sid, score=1.0, title=f"곡{sid}", vocal_gender=gender, genre=genre, release_date=f"{year}-01-01")


def _split_pool() -> List[MatchingTrack]:
    return [_track("a", "남성"), _track("b", "여성"), _track("c", "남성"), _track("d", "여성")]


# --- 1. 질문 선택 ------------------------------------------------------------

def test_birth_year_is_asked_before_data_slots_for_life_stage_queries() -> None:
    q = pick_question(_life_stage_analysis(), _split_pool())
    assert q is not None and q.slot == BIRTH_YEAR_SLOT
    assert [o.value for o in q.options][:2] == ["1971~1975년생", "1976~1980년생"]
    assert all(o.count == 0 for o in q.options)
    assert "중학교 때" in q.reason


def test_birth_year_is_not_asked_when_era_is_known_or_already_asked() -> None:
    pool = _split_pool()
    assert pick_question(qa._fallback("비 오는 날 남자 발라드"), pool).slot == "vocal_gender"
    assert pick_question(qa._fallback("2000년대 중학교 때 듣던 댄스곡"), pool).slot == "vocal_gender"
    assert pick_question(_life_stage_analysis(), pool, asked_slots=[BIRTH_YEAR_SLOT]).slot == "vocal_gender"
    assert pick_question(with_birth_year(_life_stage_analysis(), 2001), pool).slot == "vocal_gender"
    assert BIRTH_YEAR_SLOT not in ALLOWED_SLOTS


def test_birth_year_is_asked_even_without_remaining_candidates() -> None:
    """출생 연도는 지금 보이는 결과에도 적용되는 정보다 — 미표시 후보가 없어도 묻는다(리뷰). 데이터 슬롯은 여전히 후보가 필요하다."""
    assert pick_question(_life_stage_analysis(), []).slot == BIRTH_YEAR_SLOT
    assert pick_question(_life_stage_analysis(), [], asked_slots=[BIRTH_YEAR_SLOT]) is None
    assert pick_question(qa._fallback("비 오는 날 남자 발라드"), []) is None


# --- 2. 답 → 시기 창 ---------------------------------------------------------

@pytest.mark.parametrize("value, year", [
    ("1996~2000년생", 1998), ("2001~2005년생", 2003), ("1998", 1998), ("98년생", None), ("", None), ("모름", None),
])
def test_parse_birth_year(value: str, year: Optional[int]) -> None:
    assert parse_birth_year(value) == year


def test_with_birth_year_builds_a_soft_window_only_for_life_stage_queries() -> None:
    analysis = _life_stage_analysis()
    merged = with_birth_year(analysis, 2001)
    assert merged is not analysis
    assert (merged.release_era.start_year, merged.release_era.end_year) == (2013, 2017)
    assert merged.release_era.confidence == pytest.approx(0.6)
    assert not analysis.has_release_era, "원본은 바뀌지 않는다"
    plain = qa._fallback("비 오는 날 남자 발라드")
    assert with_birth_year(plain, 2001) is plain
    assert with_birth_year(analysis, None) is analysis
    dated = qa._fallback("2000년대 중학교 때 듣던 댄스곡")
    assert with_birth_year(dated, 2001) is dated, "절대 연도가 있으면 출생 연도는 쓰지 않는다"


def test_merge_answer_turns_birth_year_into_release_era() -> None:
    analysis = _life_stage_analysis()
    merged = merge_answer(analysis.model_copy(deep=True), ClarifyAnswer(slot=BIRTH_YEAR_SLOT, value="2001~2005년생"))
    assert (merged.release_era.start_year, merged.release_era.end_year) == (2015, 2019)
    untouched = merge_answer(analysis.model_copy(deep=True), ClarifyAnswer(slot=BIRTH_YEAR_SLOT, value="모름"))
    assert not untouched.has_release_era
    skipped = merge_answer(analysis.model_copy(deep=True), ClarifyAnswer(slot=BIRTH_YEAR_SLOT, skipped=True))
    assert not skipped.has_release_era


def test_birth_year_answer_never_matches_a_track_by_itself() -> None:
    assert not answer_matches(_track("a", "남성", year="2015"), ClarifyAnswer(slot=BIRTH_YEAR_SLOT, value="1996~2000년생"))


def test_router_entry_uses_the_window_from_answers() -> None:
    analysis = _life_stage_analysis()
    out = apply_birth_year_answers(analysis, [ClarifyAnswer(slot="vocal_gender", value="남성"),
                                              ClarifyAnswer(slot=BIRTH_YEAR_SLOT, value="1996~2000년생")])
    assert (out.release_era.start_year, out.release_era.end_year) == (2010, 2014)
    assert apply_birth_year_answers(analysis, None) is analysis
    assert apply_birth_year_answers(analysis, [ClarifyAnswer(slot="vocal_gender", value="남성")]) is analysis


# --- 4. 리랭커 프롬프트 ------------------------------------------------------

def test_reranker_sees_an_era_window_not_the_birth_year(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_RERANK_CORRECTIONS", raising=False)
    monkeypatch.delenv("CLARIFY_RERANK_INPUT_ORDER", raising=False)
    analysis = _life_stage_analysis()
    out = answers_for_reranker(analysis, [ClarifyAnswer(slot=BIRTH_YEAR_SLOT, value="1996~2000년생"),
                                          ClarifyAnswer(slot="vocal_gender", value="남성")])
    assert [(a.slot, a.value) for a in out] == [("release_era", "2010~2014년쯤"), ("vocal_gender", "남성")]
    plain = qa._fallback("비 오는 날 남자 발라드")
    assert answers_for_reranker(plain, [ClarifyAnswer(slot=BIRTH_YEAR_SLOT, value="1996~2000년생")]) == []


def test_reranker_gets_the_window_when_the_analysis_already_carries_it(monkeypatch) -> None:
    """라우터 입구(apply_birth_year_answers)와 프로필(with_birth_year) 모두 분석에 창을 먼저 넣는다 — 그 뒤에도 리랭커는 창을 받아야 한다(리뷰)."""
    monkeypatch.delenv("GEMINI_RERANK_CORRECTIONS", raising=False)
    monkeypatch.delenv("CLARIFY_RERANK_INPUT_ORDER", raising=False)
    windowed = with_birth_year(_life_stage_analysis(), 1998)
    # 답 경로: 분석에 창이 이미 들어 있고 answers에 birth_year 답이 남아 있다
    out = answers_for_reranker(windowed, [ClarifyAnswer(slot=BIRTH_YEAR_SLOT, value="1996~2000년생"),
                                          ClarifyAnswer(slot="vocal_gender", value="남성")])
    assert [(a.slot, a.value) for a in out] == [("release_era", "2010~2014년쯤"), ("vocal_gender", "남성")]
    # 프로필 경로: answers가 비어 있어도 창은 간다
    assert [(a.slot, a.value) for a in answers_for_reranker(windowed, [])] == [("release_era", "2010~2014년쯤")]
    # 질의에 절대 시기가 있는 분석은 창을 따로 보내지 않는다(프롬프트의 질의에 이미 있다)
    assert answers_for_reranker(qa._fallback("2000년대 중학교 때 듣던 댄스곡"), []) == []
    assert answers_for_reranker(qa._fallback("비 오는 날 남자 발라드"), []) == []


def test_router_passes_the_window_to_the_answer_using_reranker() -> None:
    """프로필 경로(answers 없음)와 답 경로 모두에서 라우터가 리랭커에 release_era 창을 넘긴다(리뷰)."""
    import asyncio
    from tests.test_router_rerank_input_order import _AnswerReranker, _hits
    from tests.test_search_explain import _router

    def run(analysis, answers):
        rr = _AnswerReranker()
        router = _router(_hits(), reranker=rr)
        try:
            asyncio.run(router.search(analysis, top_k=4, candidate_k=4, answers=answers))
        finally:
            router.shutdown()
        return [(a.slot, a.value) for call in rr.answers for a in call]

    assert run(with_birth_year(_life_stage_analysis(), 1998), None) == [("release_era", "2010~2014년쯤")]
    assert run(_life_stage_analysis(), [ClarifyAnswer(slot=BIRTH_YEAR_SLOT, value="1996~2000년생")]) == [("release_era", "2010~2014년쯤")]
    assert run(_life_stage_analysis(), None) == [], "창이 없으면 보낼 답도 없다"


# --- 3. 라우트: 프로필 birth_year ---------------------------------------------

class _FakeRouter:
    def __init__(self) -> None:
        self.seen: List[QueryAnalysis] = []

    async def search(self, analysis, top_k=10, candidate_ids_out=None, candidate_tracks_out=None, **kw):
        self.seen.append(analysis)
        pool = _split_pool()
        if candidate_ids_out is not None:
            candidate_ids_out.extend(t.id for t in pool)
        if candidate_tracks_out is not None:
            candidate_tracks_out.extend(pool)
        return pool[:1]


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(search_route)
    fake = _FakeRouter()
    app.dependency_overrides[get_search_router] = lambda: fake
    app.dependency_overrides[get_query_analyzer] = lambda: type("A", (), {"analyze": staticmethod(qa._fallback)})()
    return TestClient(app), fake


def test_profile_birth_year_skips_the_question_and_keeps_response_analysis_clean(client) -> None:
    tc, fake = client
    body = tc.post("/search", json={"query": "중학교 때 많이 듣던 남자 발라드", "birth_year": 2001}).json()
    searched = fake.seen[-1]
    assert (searched.release_era.start_year, searched.release_era.end_year) == (2013, 2017)
    assert body["analysis"]["release_era"]["start_year"] is None, "응답의 분석은 원본 — 프로필이 굳어 들어가지 않는다"
    assert body["clarify"]["slot"] == "vocal_gender"


def test_without_profile_the_route_asks_birth_year_first(client) -> None:
    tc, fake = client
    body = tc.post("/search", json={"query": "중학교 때 많이 듣던 남자 발라드"}).json()
    assert not fake.seen[-1].has_release_era
    assert body["clarify"]["slot"] == BIRTH_YEAR_SLOT
    assert body["clarify"]["options"][0]["value"] == "1971~1975년생"
    # 2턴: 답을 돌려보내면 라우터가 창을 받고, 다음 질문은 데이터 슬롯이다
    turn2 = tc.post("/search", json={
        "query": "중학교 때 많이 듣던 남자 발라드", "prior_analysis": body["analysis"],
        "answers": [{"slot": BIRTH_YEAR_SLOT, "value": "1996~2000년생", "skipped": False}],
        "asked_slots": [BIRTH_YEAR_SLOT], "rejected_ids": ["a"], "turn": body["turn"],
    }).json()
    assert turn2["clarify"]["slot"] == "vocal_gender"
    assert turn2["asked_slots"] == [BIRTH_YEAR_SLOT]


class _AllShownRouter(_FakeRouter):
    """후보 2곡을 모두 결과로 보여준다 — 질문에 넘길 미표시 후보가 없다."""

    async def search(self, analysis, top_k=10, candidate_ids_out=None, candidate_tracks_out=None, **kw):
        self.seen.append(analysis)
        pool = _split_pool()[:2]
        if candidate_ids_out is not None:
            candidate_ids_out.extend(t.id for t in pool)
        if candidate_tracks_out is not None:
            candidate_tracks_out.extend(pool)
        return pool


def test_birth_year_is_asked_even_when_every_candidate_is_shown() -> None:
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_search_router] = lambda: _AllShownRouter()
    app.dependency_overrides[get_query_analyzer] = lambda: type("A", (), {"analyze": staticmethod(qa._fallback)})()
    body = TestClient(app).post("/search", json={"query": "중학교 때 듣던 노래"}).json()
    assert len(body["results"]) == 2 and body["clarify"]["slot"] == BIRTH_YEAR_SLOT


def test_research_with_prior_analysis_and_profile_keeps_the_turn(client) -> None:
    """출생 연도 답 뒤의 재검색: 화면은 prior_analysis + birth_year만 보낸다(거절·답 없음) — 재분석 없이, 턴도 그대로."""
    tc, fake = client
    first = tc.post("/search", json={"query": "중학교 때 많이 듣던 남자 발라드"}).json()
    assert first["clarify"]["slot"] == BIRTH_YEAR_SLOT and first["turn"] == 1
    calls = len(fake.seen)
    again = tc.post("/search", json={"query": "중학교 때 많이 듣던 남자 발라드", "prior_analysis": first["analysis"],
                                     "birth_year": 1998, "turn": first["turn"]}).json()
    assert len(fake.seen) == calls + 1
    assert (fake.seen[-1].release_era.start_year, fake.seen[-1].release_era.end_year) == (2010, 2014)
    assert again["turn"] == 1, "거절도 답도 없는 재검색은 턴을 쓰지 않는다"
    assert again["clarify"]["slot"] == "vocal_gender"
    assert again["analysis"]["release_era"]["start_year"] is None


def test_profile_is_ignored_for_queries_without_life_stage(client) -> None:
    tc, fake = client
    tc.post("/search", json={"query": "비 오는 날 남자 발라드", "birth_year": 2001})
    assert not fake.seen[-1].has_release_era


# --- 하네스 -------------------------------------------------------------------

def _song(year: str) -> dict:
    return {"metadata": {"release_date": f"{year}-05-01", "vocal_gender": "남성"}}


def test_harness_oracle_birth_year_is_derived_from_release_year() -> None:
    analysis = _life_stage_analysis()                      # 중학교 13~15 → 중앙 14
    assert oracle_birth_year(_song("2015"), analysis) == birth_year_band(2001)   # 2015-14 = 2001
    assert noisy_birth_year(_song("2015"), analysis) == birth_year_band(2011)
    assert oracle_birth_year(_song("2015"), qa._fallback("비 오는 날 발라드")) is None
    assert oracle_value(_song("2015"), BIRTH_YEAR_SLOT, analysis) == birth_year_band(2001)
    assert oracle_value(_song("2015"), "vocal_gender") == "남성"


def test_harness_choose_answer_for_birth_year_question() -> None:
    analysis = _life_stage_analysis()
    question = pick_question(analysis, _split_pool())
    target = _track("x", "남성", year="2015")
    answer, kind = choose_answer(question, "oracle", target, _song("2015"), analysis)
    assert (kind, answer.value) == ("oracle", birth_year_band(2001))
    answer, kind = choose_answer(question, "noisy", target, _song("2015"), analysis)
    assert (kind, answer.value) == ("noisy", birth_year_band(2011))
    answer, kind = choose_answer(question, "oracle", target, _song("2015"))   # 분석 없으면 모른다 → skip
    assert kind == "skip" and answer.skipped
