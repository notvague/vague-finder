"""/search 응답의 사용자용 설명 — 실제 라우트 경로로 확인한다.

고정하는 것: explain=true면 각 결과에 user_explanation이 실린다. explain을 켜도 결과 ID·순서·점수와
분석기·리랭커 호출 수가 같다. 재질문의 정정 답변이 현재 단서다. 설명 생성이 실패해도, 만든 설명이
응답 스키마와 어긋나도 검색은 성공한다.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import src.backend.api.routes.search as route_module
from src.backend.api.dependencies import get_query_analyzer, get_search_router
from src.backend.api.routes.search import router as search_route
from src.backend.schemas.explain import UserExplanationOut, to_user_explanation_out
from tests.test_search_explain import _StubReranker, _analysis, _router, _track


class _CountingAnalyzer:
    def __init__(self, **fields):
        self.calls = 0
        self.fields = fields

    def analyze(self, query):
        self.calls += 1
        return _analysis(original_query=query, **self.fields)


def _client(router, analyzer):
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = lambda: analyzer
    app.dependency_overrides[get_search_router] = lambda: router
    return TestClient(app)


def _hits():
    return [
        _track("s1", title="곡1", vocal_gender="남성", genre="댄스", release_date="2019.03.01"),
        _track("s2", title="곡2", vocal_gender="여성", genre="발라드", release_date="2012.01.01"),
        _track("s3", title="곡3", vocal_gender="혼성", genre="인디음악, 록/메탈", release_date="2016.06.01"),
    ]


def _post(body, *, analyzer=None, hits=None):
    analyzer = analyzer or _CountingAnalyzer()
    reranker = _StubReranker()
    router = _router(hits or _hits(), reranker=reranker)
    try:
        with _client(router, analyzer) as client:
            response = client.post("/search", json={"query": "남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어", **body})
        assert response.status_code == 200, response.text
        return response.json(), analyzer, reranker
    finally:
        router.shutdown()


def test_route_attaches_a_user_explanation_to_every_result():
    body, _, _ = _post({"top_k": 3, "explain": True}, analyzer=_CountingAnalyzer(vocal_gender="남성", genre="인디"))
    for track in body["results"]:
        ux = track["explain"]["user_explanation"]
        assert ux is not None and ux["summary"]
        slots = [c["slot"] for c in ux["clue_checks"]]
        assert slots[:2] == ["vocal_gender", "genre"] and "mv_scene" in slots
        for check in ux["clue_checks"]:
            assert check["status_label"] in {"일치", "부분 일치", "다름", "미확인"}
            assert check["basis_label"]
    by_id = {t["id"]: t["explain"]["user_explanation"] for t in body["results"]}
    gender = {sid: ux["clue_checks"][0]["status"] for sid, ux in by_id.items()}
    assert gender == {"s1": "match", "s2": "mismatch", "s3": "partial"}
    # 가산 기록이 있는 곡만 "반영"이라고 말한다 — 여기서는 라우터가 남성 일치 가산을 실제로 더했다
    s1 = by_id["s1"]
    assert s1["clue_checks"][0]["basis_label"] == "검색 점수에 반영"
    assert s1["clue_checks"][0]["ranking_basis"][0]["code"] == "vocal_gender_match"
    assert by_id["s2"]["clue_checks"][0]["ranking_basis"] == []
    assert all(c["status"] == "unverified" for ux in by_id.values() for c in ux["clue_checks"] if c["slot"] == "mv_scene")


def test_route_explain_keeps_ids_order_scores_and_call_counts():
    off, a_off, r_off = _post({"top_k": 3})
    on, a_on, r_on = _post({"top_k": 3, "explain": True})
    assert [t["id"] for t in off["results"]] == [t["id"] for t in on["results"]]
    assert [t["score"] for t in off["results"]] == [t["score"] for t in on["results"]]
    assert a_off.calls == a_on.calls == 1
    assert r_off.calls == r_on.calls


def test_route_without_explain_has_no_user_explanation():
    body, _, _ = _post({"top_k": 3})
    assert all(t["explain"] is None for t in body["results"])


def test_route_uses_the_corrected_answer_as_the_current_clue_without_reanalysis():
    prior = _analysis(original_query="남자 가수 노래", vocal_gender="남성").model_dump()
    body, analyzer, _ = _post({
        "query": "남자 가수 노래", "top_k": 3, "explain": True,
        "prior_analysis": prior,
        "answers": [{"slot": "vocal_gender", "value": "여성", "skipped": False}],
        "asked_slots": ["vocal_gender"], "turn": 1,
    })
    assert analyzer.calls == 0, "prior_analysis가 있으면 재분석하지 않는다"
    by_id = {t["id"]: t for t in body["results"]}
    s1 = by_id["s1"]["explain"]
    gender_row = s1["user_explanation"]["clue_checks"][0]
    assert gender_row["clue"] == "여성 보컬" and gender_row["clue_origin"] == "answer"
    assert gender_row["status"] == "mismatch" and gender_row["ranking_basis"] == []
    # 남성 기준 가산은 상세 기록에 남는다 — 현재 단서의 근거로는 쓰지 않았다
    old = [a for a in s1["adjustments"] if a["rule"] == "vocal_gender_match"]
    assert old and old[0]["value"] == "남성"
    assert "반영" not in s1["user_explanation"]["summary"]
    s2 = by_id["s2"]["explain"]["user_explanation"]["clue_checks"][0]
    assert s2["status"] == "match" and any(b["code"] == "answer_match" for b in s2["ranking_basis"])


def test_route_survives_an_explanation_failure(monkeypatch):
    def boom(**kw):
        raise RuntimeError("설명 실패(시험)")
    monkeypatch.setattr(route_module, "build_user_explanation", boom)
    body, _, _ = _post({"top_k": 3, "explain": True})
    assert len(body["results"]) == 3
    for track in body["results"]:
        assert track["explain"] is not None and track["explain"]["user_explanation"] is None
        assert track["explain"]["summary"]


# ---------------------------------------------------------------------------
# 회차 1 검토 지적 — UFE-T1(출생 연도 창 · MV 생략 + notice), UFE-09(정정 전 가산 문구)
# ---------------------------------------------------------------------------

LIFE_STAGE = {"stage": "middle", "text": "중학교 때", "spans": [[0, 5]], "age_from": 13, "age_to": 15, "confidence": 0.8}


class _MvSkipReranker(_StubReranker):
    """Gemini listwise처럼 자료 없는 MV 장면 질의의 리랭킹을 정책으로 생략한다."""

    skips_unverifiable_mv_scene = True


def _post_with(body, *, analyzer=None, hits=None, reranker=None, query="중학교 때 듣던 노래"):
    analyzer = analyzer or _CountingAnalyzer()
    reranker = reranker or _StubReranker()
    router = _router(hits or _hits(), reranker=reranker)
    try:
        with _client(router, analyzer) as client:
            response = client.post("/search", json={"query": query, **body})
        assert response.status_code == 200, response.text
        return response.json(), analyzer, reranker
    finally:
        router.shutdown()


def _era_rows(body):
    return {t["id"]: next(c for c in t["explain"]["user_explanation"]["clue_checks"] if c["slot"] == "release_era")
            for t in body["results"]}


def test_route_profile_birth_year_window_is_an_estimated_era_clue():
    body, _, _ = _post_with({"top_k": 3, "explain": True, "birth_year": 1998},
                            analyzer=_CountingAnalyzer(life_stage=LIFE_STAGE))
    rows = _era_rows(body)
    assert all(r["clue_origin"] == "birth_year" and "2010~2014" in r["clue"] and "추정" in r["clue"] for r in rows.values())
    assert rows["s2"]["status"] == "match" and "추정 기간 안" in rows["s2"]["observed"]          # 2012
    assert rows["s2"]["ranking_basis"] and rows["s2"]["ranking_basis"][0]["code"] == "release_era"
    assert rows["s3"]["status"] == "partial" and "근접" in rows["s3"]["observed"]              # 2016
    assert rows["s1"]["status"] == "mismatch"                                                  # 2019 — 그 시절보다 나중
    assert body["analysis"]["release_era"]["start_year"] is None, "응답의 분석에는 프로필 창이 굳지 않는다"


def test_route_birth_year_answer_window_matches_what_the_router_used():
    prior = _analysis(original_query="중학교 때 듣던 노래", life_stage=LIFE_STAGE).model_dump()
    body, analyzer, _ = _post_with({
        "top_k": 3, "explain": True, "prior_analysis": prior,
        "answers": [{"slot": "birth_year", "value": "1998", "skipped": False}],
        "asked_slots": ["birth_year"], "turn": 1,
    })
    assert analyzer.calls == 0
    rows = _era_rows(body)
    assert rows["s2"]["status"] == "match" and rows["s2"]["clue_origin"] == "birth_year"
    assert rows["s2"]["ranking_basis"][0]["code"] == "release_era", "답으로 만든 창의 가산이 같은 창으로 연결된다"


def test_route_without_birth_year_keeps_life_stage_as_unverified_without_claiming_missing_data():
    body, _, _ = _post_with({"top_k": 3, "explain": True}, analyzer=_CountingAnalyzer(life_stage=LIFE_STAGE))
    rows = _era_rows(body)
    assert all(r["status"] == "unverified" and "출생 연도" in r["observed"] for r in rows.values())
    for t in body["results"]:
        assert "확인할 자료가 없" not in t["explain"]["user_explanation"]["summary"]


def test_route_mv_skip_policy_is_reflected_in_the_clue_row_and_notice():
    query = "남자 인디 노래인데 뮤비에 의자가 하늘에 떠있는 장면이 있어"
    body, _, reranker = _post_with({"top_k": 3, "explain": True}, reranker=_MvSkipReranker(), query=query)
    assert reranker.calls == 0, "정책 생략이면 모델을 부르지 않는다"
    assert body["notice"] and body["notice"]["code"] == "mv_scene_unverifiable"
    assert body["explain"]["rerank_skip_reason"] == "mv_scene_unverifiable"
    for t in body["results"]:
        mv = next(c for c in t["explain"]["user_explanation"]["clue_checks"] if c["slot"] == "mv_scene")
        assert mv["clue"] == "뮤비에 의자가 하늘에 떠있는 장면이 있어"
        assert mv["status"] == "unverified" and mv["basis_label"] == "자료가 없어 리랭킹 생략"
        assert mv["ranking_basis"][0]["kind"] == "rerank_skip"
    # 생략하지 않는 리랭커면 같은 행이 미확인이되 생략 근거는 없다
    body, _, reranker = _post_with({"top_k": 3, "explain": True}, query=query)
    assert reranker.calls == 1 and body["notice"] is None
    mv = next(c for c in body["results"][0]["explain"]["user_explanation"]["clue_checks"] if c["slot"] == "mv_scene")
    assert mv["status"] == "unverified" and mv["ranking_basis"] == []


def test_route_marks_the_pre_correction_bonus_line_in_the_details():
    prior = _analysis(original_query="남자 가수 노래", vocal_gender="남성").model_dump()
    body, _, _ = _post({
        "query": "남자 가수 노래", "top_k": 3, "explain": True, "prior_analysis": prior,
        "answers": [{"slot": "vocal_gender", "value": "여성", "skipped": False}],
        "asked_slots": ["vocal_gender"], "turn": 1,
    })
    s1 = {t["id"]: t for t in body["results"]}["s1"]["explain"]
    old = [a for a in s1["adjustments"] if a["rule"] == "vocal_gender_match"]
    assert old and old[0]["value"] == "남성" and old[0]["note"] == "정정 전 값(남성) 기준"
    assert s1["user_explanation"]["adjustment_notes"][0]["note"] == "정정 전 값(남성) 기준"
    # 상세 맨 위의 기록 요약 문장에도 같은 표시가 붙는다 — "말한 성별과 일치"만 읽히면 지금 표와 반대로 보인다
    assert "말한 성별과 일치, 남성 · 정정 전 값(남성) 기준 (+" in s1["summary"], s1["summary"]
    # 정정하지 않은 요청의 같은 줄에는 표시가 없다
    plain, _, _ = _post({"query": "남자 가수 노래", "top_k": 3, "explain": True},
                        analyzer=_CountingAnalyzer(vocal_gender="남성"))
    plain_s1 = {t["id"]: t for t in plain["results"]}["s1"]["explain"]
    assert "말한 성별과 일치, 남성 (+" in plain_s1["summary"] and "기준" not in plain_s1["summary"]
    assert all(a["note"] == "" for a in plain_s1["adjustments"])


# ---------------------------------------------------------------------------
# 회차 2 검토 지적 — UFE-12 (a): 값이 충돌한 줄의 가산은 상세에서도 기준이 적힌다 (실제 라우트)
# ---------------------------------------------------------------------------

def test_route_conflict_row_bonus_is_marked_in_the_details():
    query = "남자 가수 이별 노래가 크게 뜨고 나서 그 노래 답가로 나온 여자 버전 노래 있잖아. 2017년 겨울쯤"
    body, _, _ = _post_with({"top_k": 3, "explain": True}, analyzer=_CountingAnalyzer(vocal_gender="여성"), query=query)
    s2 = {t["id"]: t for t in body["results"]}["s2"]["explain"]          # 등록 여성 — 검색이 여성 일치 가산을 더했다
    row = s2["user_explanation"]["clue_checks"][0]
    assert row["slot"] == "vocal_gender" and row["status"] == "unverified"
    assert row["clue"] == "여성 보컬(검색이 해석한 값)" and "원문에서 읽은 값(혼성)" in row["observed"]
    assert row["basis_label"] == "검색 점수에 반영(검색이 해석한 값 기준)"
    bonus = [a for a in s2["adjustments"] if a["rule"] == "vocal_gender_match"]
    assert bonus and bonus[0]["note"] == "검색이 해석한 값(여성) 기준"
    assert "말한 성별과 일치, 여성 · 검색이 해석한 값(여성) 기준 (+" in s2["summary"], s2["summary"]
    assert "일치해" not in s2["user_explanation"]["summary"] and "달라요" not in s2["user_explanation"]["summary"]


# ---------------------------------------------------------------------------
# 회차 3 검토 지적 — UFE-14: 장면 뒤에 다른 단서의 절이 와도 MV 칸은 장면까지다 (실제 라우트)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query, scene", [
    ("뮤비에 의자가 하늘에 떠있는 장면이 있는데 남자가 불렀어", "뮤비에 의자가 하늘에 떠있는 장면이 있는데"),
    ("뮤비에서 의자가 하늘에 떠 있어 여자 목소리였어", "뮤비에서 의자가 하늘에 떠 있어"),
    ("뮤비에 의자가 하늘에 떠있는 장면이 있어 중학교 때 듣던 노래야", "뮤비에 의자가 하늘에 떠있는 장면이 있어"),
    ("뮤비에서 의자가 하늘에 떠 있어 바다가 생각나는 노래야", "뮤비에서 의자가 하늘에 떠 있어"),
])
def test_route_mv_cell_holds_only_the_scene_when_another_clue_follows(query, scene):
    body, _, reranker = _post_with({"top_k": 3, "explain": True}, reranker=_MvSkipReranker(), query=query)
    assert reranker.calls == 0 and body["notice"]["code"] == "mv_scene_unverifiable"
    for track in body["results"]:
        mv = next(c for c in track["explain"]["user_explanation"]["clue_checks"] if c["slot"] == "mv_scene")
        assert mv["clue"] == scene, mv["clue"]
        assert mv["status"] == "unverified" and mv["basis_label"] == "자료가 없어 리랭킹 생략"


# ---------------------------------------------------------------------------
# PR #52 자동 리뷰 — UFE-16: 만든 설명이 응답 스키마와 어긋나도 검색은 성공한다 (실제 라우트)
# ---------------------------------------------------------------------------

class _Misshapen:
    """생성은 됐지만 응답 스키마와 어긋난 설명 — 대조표 줄에 필수 필드가 빠졌다."""

    def __init__(self, built):
        self.built = built

    def to_dict(self):
        return {**self.built.to_dict(), "clue_checks": [{"slot": "vocal_gender"}]}


def _misshape_for(monkeypatch, song_id):
    real = route_module.build_user_explanation

    def build(**kw):
        built = real(**kw)
        return _Misshapen(built) if kw["track"].id == song_id else built
    monkeypatch.setattr(route_module, "build_user_explanation", build)


def _record_only(track):
    return {k: v for k, v in track["explain"].items() if k != "user_explanation"}


def test_route_survives_an_explanation_that_does_not_fit_the_response_schema(monkeypatch):
    request = {"top_k": 3, "explain": True}
    normal, _, _ = _post(request, analyzer=_CountingAnalyzer(vocal_gender="남성", genre="인디"))
    _misshape_for(monkeypatch, "s1")
    body, _, _ = _post(request, analyzer=_CountingAnalyzer(vocal_gender="남성", genre="인디"))   # 200은 _post가 본다
    assert [(t["id"], t["score"]) for t in body["results"]] == [(t["id"], t["score"]) for t in normal["results"]]
    before = {t["id"]: t for t in normal["results"]}
    after = {t["id"]: t for t in body["results"]}
    # 어긋난 곡은 설명만 비고, 그 곡의 기록 요약 · 경로 · 가감점은 그대로다
    s1 = after["s1"]["explain"]
    assert before["s1"]["explain"]["user_explanation"] is not None and s1["user_explanation"] is None
    assert _record_only(after["s1"]) == _record_only(before["s1"])
    assert s1["summary"] and s1["paths"]
    assert "vocal_gender_match" in [a["rule"] for a in s1["adjustments"]]
    # 다른 곡의 설명은 그대로 실린다 — 한 곡의 실패가 응답 전체를 비우지 않는다
    for song_id in ("s2", "s3"):
        assert after[song_id]["explain"]["user_explanation"] is not None
        assert after[song_id]["explain"] == before[song_id]["explain"]


def test_route_shows_the_plain_record_line_when_the_explanation_that_carried_its_note_is_dropped(monkeypatch):
    # 가산 줄의 "정정 전 값" 표시는 설명에서 나온다. 설명이 빠진 곡은 표시 없는 기록 그대로다(생성 실패와 같다)
    _misshape_for(monkeypatch, "s1")
    prior = _analysis(original_query="남자 가수 노래", vocal_gender="남성").model_dump()
    body, _, _ = _post({
        "query": "남자 가수 노래", "top_k": 3, "explain": True, "prior_analysis": prior,
        "answers": [{"slot": "vocal_gender", "value": "여성", "skipped": False}],
        "asked_slots": ["vocal_gender"], "turn": 1,
    })
    s1 = {t["id"]: t for t in body["results"]}["s1"]["explain"]
    assert s1["user_explanation"] is None
    old = [a for a in s1["adjustments"] if a["rule"] == "vocal_gender_match"]
    assert old and old[0]["value"] == "남성" and old[0]["note"] == ""
    assert "말한 성별과 일치, 남성 (+" in s1["summary"] and "기준" not in s1["summary"]


def test_a_validated_explanation_passes_through_and_a_misshapen_one_is_rejected():
    out = UserExplanationOut(summary="요약")
    assert to_user_explanation_out(out) is out, "라우트가 검증한 모델을 to_track_explain이 다시 만들지 않는다"
    assert to_user_explanation_out({"summary": "요약"}) == out
    with pytest.raises(ValidationError):
        to_user_explanation_out({"summary": "요약", "clue_checks": [{"slot": "vocal_gender"}]})
