"""v59 — 부분 실패·후보 밖 id 응답이 "성공"으로 보이던 문제(v56 §7-③)의 회귀 시험.

고정하는 것:
  A. 유효 후보 0개(후보 밖 id만·빈 ranking·자료형 오류)는 **무효 응답**이다 — 기존 재시도·공유 마감 안에서 다시 묻고,
     끝내 무효면 `failed`로 폴백한다(입력 순서·보호 규칙 그대로). 입력 순서와 같은 정상 응답은 정상이다.
     일부 누락·중복·무효가 섞인 응답은 기존 보충 순서를 유지하고 관측치만 남긴다.
  B. 최종 순서에 반영된 묶음에 성공과 실패·예산 소진이 섞이면 단계는 `model_partial`("리랭킹 일부 적용") — 평가기의
     `partial_failure`와 같은 조합 규칙. 적용 0회면 `model_failed`, 라우터 예외 폴백은 커밋 취소라 부분 적용이 아니다.
     정책 생략·비활성·상태 미보고 백엔드는 실패와 섞이지 않는다.
  C. `rerank_calls`는 묶음 기록 수(의미 그대로), 적용·실패·예산 소진 묶음 수와 백엔드가 센 실제 SDK 호출 수를 따로 낸다.
     곡별 설명은 "넘기지 않은 곡"과 "실패한 묶음의 곡", "응답에 없어 보충된 곡"을 가른다.

실제 `GeminiListwiseReranker`·`SearchRouter`·`/search` 라우트에 v56의 가짜 SDK 클라이언트만 꽂는다. 모델 호출 없음.
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.dependencies import get_query_analyzer, get_search_router
from src.backend.api.routes.search import router as search_route
from src.backend.schemas.explain import SearchExplainOut, to_track_explain
from src.backend.schemas.query import ModalityWeights
from src.retrieval import search_router as sr
from src.retrieval.evaluate_search_accuracy import (
    OUTCOME_APPLIED,
    OUTCOME_EXCEPTION,
    OUTCOME_FALLBACK,
    OUTCOME_PARTIAL_FAILURE,
    OUTCOME_POLICY_SKIPPED,
    OUTCOME_SKIPPED,
    OUTCOME_UNKNOWN,
    rerank_outcome,
    rerank_preserving_exact_lyrics,
)
from src.retrieval.explain import (
    NULL_RECORDER,
    REORDER_MODEL_APPLIED,
    REORDER_MODEL_FAILED,
    REORDER_MODEL_PARTIAL,
    REORDER_MODEL_SKIPPED,
    REORDER_MODEL_UNKNOWN,
    REORDER_NOT_ATTEMPTED,
    REORDER_POLICY_SKIPPED,
    REORDER_STAGE_LABELS,
    REORDER_STAGE_NOTES,
    RERANK_APPLIED,
    RERANK_BUDGET_EXHAUSTED,
    RERANK_FAILED,
    RERANK_SKIPPED,
    RERANK_UNKNOWN,
    ExplainRecorder,
    SearchExplain,
    render_ko,
)
from src.retrieval.gemini_listwise_reranker import (
    GeminiListwiseReranker,
    GeminiListwiseRerankerConfig,
    InvalidRankingResponse,
)
from tests.fake_gemini_client import FOREIGN_ID_PREFIX, FakeGeminiClient, _Response
from tests.rerank.test_rerank_shared_budget import ScriptedClient, _lyric_analysis, _others, _protected
from tests.search.test_search_explain import _LegacyReranker, _StubReranker, _analysis, _router, _track
from tests.api.test_user_explanation_route import _CountingAnalyzer

BUDGET = 1.0
TOLERANCE = 0.5


def _cfg(**over) -> GeminiListwiseRerankerConfig:
    base = dict(enabled=True, max_retries=3, passes=1, request_timeout_seconds=0.6, time_budget_seconds=5.0,
                pass_reserve_seconds=0.2, use_search_grounding=False, rare_fact_verification=False)
    base.update(over)
    return GeminiListwiseRerankerConfig(**base)


def _tracks(n: int = 6, prefix: str = "1000") -> list:
    return [_track(f"{prefix}{i}", title=f"곡{i}", score=1.0 - i * 0.01, retrieval_score=1.0 - i * 0.01)
            for i in range(1, n + 1)]


def _reranker(client, **cfg) -> GeminiListwiseReranker:
    return GeminiListwiseReranker(_cfg(**cfg), client=client, configured=True)


def _search(router, analysis, **kw):
    import asyncio

    started = time.monotonic()
    try:
        results = asyncio.run(router.search(analysis, **kw))
    finally:
        router.shutdown()
    return results, time.monotonic() - started


# ---------------------------------------------------------------------------
# A. 유효 후보가 없는 응답 — 파서 단위
# ---------------------------------------------------------------------------

def _parse(raw, valid, stats=None):
    return GeminiListwiseReranker._parse_response(raw, valid, stats)


@pytest.mark.parametrize("raw", [
    json.dumps({"query_confidence": 0.9, "ranking": []}),                                  # 빈 ranking
    json.dumps({"query_confidence": 0.9}),                                                 # ranking 없음
    json.dumps({"ranking": [{"id": f"{FOREIGN_ID_PREFIX}1"}, {"id": f"{FOREIGN_ID_PREFIX}2"}]}),  # 후보 밖 id만
    json.dumps({"ranking": [{"id": {"nested": "1"}}, {"id": ["1"]}, 3.5, None]}),         # id 자료형 오류·항목 자료형 오류
    json.dumps({"ranking": [{"id": True}]}),                                               # bool은 id가 아니다
])
def test_parser_rejects_responses_without_a_valid_candidate(raw):
    stats: dict = {}
    with pytest.raises(InvalidRankingResponse):
        _parse(raw, ["1", "2", "3"], stats)
    assert stats.get("valid", 0) == 0, "보충 전에 유효 수를 판별해야 한다"


@pytest.mark.parametrize("raw", [
    json.dumps([{"id": "1"}, {"id": "2"}]),             # 최상위가 배열
    json.dumps("ranking"),                               # 최상위가 문자열
    json.dumps({"ranking": {"id": "1"}}),                # ranking이 객체
    json.dumps({"ranking": "1,2,3"}),                    # ranking이 문자열(글자 단위로 돌면 안 된다)
    json.dumps({"ranking": 12}),
])
def test_parser_rejects_well_formed_json_with_wrong_types(raw):
    with pytest.raises(InvalidRankingResponse):
        _parse(raw, ["1", "2", "3"])


def test_parser_still_raises_json_error_for_non_json():
    with pytest.raises(json.JSONDecodeError):
        _parse("리랭킹 결과입니다: 1위는 첫 번째 곡", ["1", "2"])


def test_identity_order_with_every_candidate_is_a_normal_response():
    """순서가 입력과 같다는 이유로 실패 판정하지 않는다."""
    stats: dict = {}
    order, scores, confidence, _ = _parse(
        json.dumps({"query_confidence": 0.7, "ranking": [{"id": i, "relevance": 0.5} for i in ("1", "2", "3")]}),
        ["1", "2", "3"], stats,
    )
    assert order == ["1", "2", "3"] and confidence == 0.7
    assert stats == {"valid": 3, "missing": 0, "duplicate": 0, "invalid": 0}


def test_partial_duplicate_and_foreign_mix_keeps_the_existing_fill_order_and_reports_stats():
    """일부 유효 + 중복 + 후보 밖 id. 보충 정책은 그대로(응답 순서 → 빠진 id는 입력 순서로 뒤에), 관측치만 남긴다."""
    stats: dict = {}
    raw = json.dumps({"ranking": [
        {"id": "3", "relevance": 0.9}, {"id": "1", "relevance": 0.8}, {"id": "3", "relevance": 0.1},
        {"id": f"{FOREIGN_ID_PREFIX}7"}, 42,
    ]})
    order, scores, _, _ = _parse(raw, ["1", "2", "3", "4"], stats)
    assert order == ["3", "1", "2", "4"], "응답에 있던 3·1이 먼저, 빠진 2·4는 입력 순서"
    assert scores == {"3": 0.9, "1": 0.8}, "중복 항목의 점수는 첫 것만"
    assert stats == {"valid": 2, "missing": 2, "duplicate": 1, "invalid": 2}


def test_parser_tolerates_odd_scalar_types_in_a_valid_response():
    """id는 정수 표기도 받고(str로 비교), relevance·query_confidence는 bool·문자열이어도 터지지 않는다."""
    order, scores, confidence, reasons = _parse(
        json.dumps({"query_confidence": True, "ranking": [
            {"id": 2, "relevance": "0.4", "reason": ["list"]}, {"id": "1", "relevance": False},
        ]}),
        ["1", "2"],
    )
    assert order == ["2", "1"] and scores == {"2": 0.4} and confidence == 0.5 and reasons == {}


# ---------------------------------------------------------------------------
# A. 무효 응답 — 실제 리랭커의 재시도·공유 마감·폴백
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["foreign_ids", "empty_ranking", "wrong_types"])
def test_invalid_response_is_retried_with_the_existing_loop_then_fails(mode):
    client = FakeGeminiClient(mode=mode, sleep_seconds=0.05)
    reranker = _reranker(client)
    tracks = _tracks()
    run = reranker.rerank_run("q", tracks, top_k=6)
    assert run.status == RERANK_FAILED
    assert [t.id for t in run.tracks] == [t.id for t in tracks], "폴백은 입력 순서"
    assert len(client.calls) == 3, "기존 재시도 횟수(3) 그대로 — 별도 루프가 아니다"
    assert run.model_calls == 3 and run.judged_ids == [] and run.unreturned_ids == []


def test_invalid_response_then_normal_is_applied_on_retry():
    client = FakeGeminiClient(mode="invalid_then_normal", sleep_seconds=0.05)
    reranker = _reranker(client)
    tracks = _tracks()
    run = reranker.rerank_run("q", tracks, top_k=6)
    assert run.status == RERANK_APPLIED
    assert [t.id for t in run.tracks] == [t.id for t in reversed(tracks)], "재시도 응답(역순)이 반영됐다"
    assert len(client.calls) == 2 and run.model_calls == 2
    assert run.response_stats == {"valid": 6, "missing": 0, "duplicate": 0, "invalid": 0}, "반영된 응답의 관측치만"


def test_persistently_invalid_response_fails_inside_the_shared_deadline():
    """무효 응답이 계속되면 재시도 대기(1초)도 공유 마감 안에서 잘린다 — 추가 시간 예산이 생기지 않는다."""
    client = FakeGeminiClient(mode="foreign_ids", sleep_seconds=0.05)
    reranker = _reranker(client, time_budget_seconds=BUDGET)
    deadline = time.monotonic() + BUDGET
    started = time.monotonic()
    run = reranker.rerank_run("q", _tracks(), top_k=6, deadline=deadline)
    elapsed = time.monotonic() - started
    assert run.status == RERANK_FAILED
    assert elapsed < BUDGET + TOLERANCE, f"공유 마감을 넘겼다: {elapsed:.2f}s"
    assert 1 <= len(client.calls) <= 3


def test_partial_response_marks_unreturned_candidates_without_changing_order_policy():
    client = FakeGeminiClient(mode="partial_ids", sleep_seconds=0.05)
    reranker = _reranker(client)
    tracks = _tracks(6)
    ids = [t.id for t in tracks]
    run = reranker.rerank_run("q", tracks, top_k=6)
    assert run.status == RERANK_APPLIED
    # 응답: 앞 절반(1,2,3) 역순 + 1 중복 + 후보 밖 1 → 모델 순서는 3,2,1 뒤에 빠진 4,5,6 입력 순서(규칙 전 순서 기준)
    assert run.model_order_ids[:3] == [ids[2], ids[1], ids[0]] or set(run.model_order_ids[:3]) == set(ids[:3])
    assert run.model_order_ids[3:] == ids[3:], "빠진 곡은 입력 순서로 뒤에 — 기존 보충 정책"
    assert sorted(run.unreturned_ids) == sorted(ids[3:])
    assert run.judged_ids == ids, "프롬프트에는 전부 있었다(judged) — 그러나 뒤 셋은 모델이 자리를 정하지 않았다"
    assert run.response_stats == {"valid": 3, "missing": 3, "duplicate": 1, "invalid": 1}


# ---------------------------------------------------------------------------
# B. 단계 파생 — 서비스(reorder_stage)와 평가기(rerank_outcome)가 같은 조합 규칙
# ---------------------------------------------------------------------------

def _record(statuses, *, committed=True, attempted=True, policy_skip="", order_rules=False) -> SearchExplain:
    rec = SearchExplain(reorder_attempted=attempted, reorder_committed=committed, rerank_policy_skip=policy_skip,
                        order_rules_applied=order_rules)
    rec.rerank_runs.extend(statuses)
    return rec


@pytest.mark.parametrize("statuses, stage, outcome", [
    ([RERANK_APPLIED, RERANK_APPLIED, RERANK_APPLIED], REORDER_MODEL_APPLIED, OUTCOME_APPLIED),
    ([RERANK_FAILED, RERANK_APPLIED, RERANK_APPLIED], REORDER_MODEL_PARTIAL, OUTCOME_PARTIAL_FAILURE),   # 첫 그룹 실패
    ([RERANK_APPLIED, RERANK_FAILED], REORDER_MODEL_PARTIAL, OUTCOME_PARTIAL_FAILURE),                   # 뒤 그룹 실패
    ([RERANK_APPLIED, RERANK_BUDGET_EXHAUSTED], REORDER_MODEL_PARTIAL, OUTCOME_PARTIAL_FAILURE),         # 뒤 그룹 미실행
    ([RERANK_APPLIED, RERANK_FAILED, RERANK_BUDGET_EXHAUSTED], REORDER_MODEL_PARTIAL, OUTCOME_PARTIAL_FAILURE),
    ([RERANK_FAILED, RERANK_FAILED], REORDER_MODEL_FAILED, OUTCOME_FALLBACK),
    ([RERANK_FAILED, RERANK_BUDGET_EXHAUSTED, RERANK_BUDGET_EXHAUSTED], REORDER_MODEL_FAILED, OUTCOME_FALLBACK),
    ([RERANK_BUDGET_EXHAUSTED], REORDER_MODEL_FAILED, OUTCOME_FALLBACK),
    ([RERANK_APPLIED, RERANK_SKIPPED], REORDER_MODEL_APPLIED, OUTCOME_APPLIED),      # 정상 건너뜀은 실패가 아니다
    ([RERANK_APPLIED, RERANK_UNKNOWN], REORDER_MODEL_APPLIED, OUTCOME_APPLIED),      # 미보고도 실패가 아니다
    ([RERANK_UNKNOWN], REORDER_MODEL_UNKNOWN, OUTCOME_UNKNOWN),
    ([RERANK_SKIPPED], REORDER_MODEL_SKIPPED, OUTCOME_SKIPPED),
])
def test_stage_and_evaluator_outcome_agree(statuses, stage, outcome):
    rec = _record(statuses)
    assert rec.reorder_stage == stage
    assert rerank_outcome(statuses, "") == outcome
    assert stage in REORDER_STAGE_LABELS
    # 부분 적용에만 사용자 문구가 있고, 그 문구는 "검색 순서"라고 단정하지 않는다
    note = REORDER_STAGE_NOTES.get(stage, "")
    assert bool(note) == (stage == REORDER_MODEL_PARTIAL)
    assert "검색 순서" not in note


def test_stage_counts_separate_failed_from_budget_exhausted():
    rec = _record([RERANK_APPLIED, RERANK_FAILED, RERANK_BUDGET_EXHAUSTED, RERANK_BUDGET_EXHAUSTED])
    assert len(rec.rerank_runs) == 4 and rec.rerank_runs_applied == 1
    assert rec.rerank_runs_failed == 1 and rec.rerank_runs_budget_exhausted == 2 and rec.rerank_runs_not_applied == 3


def test_router_exception_after_a_success_is_total_fallback_not_partial():
    """커밋이 취소되면 성공 이력이 있어도 model_failed — reorder_committed/revoke_reorder 의미 유지. 평가기는 exception."""
    rec = _record([RERANK_APPLIED, RERANK_FAILED], committed=False)
    assert rec.reorder_stage == REORDER_MODEL_FAILED
    assert rerank_outcome(list(rec.rerank_runs), "RuntimeError: x") == OUTCOME_EXCEPTION


def test_policy_skip_and_not_attempted_are_not_failures():
    assert _record([], attempted=False, policy_skip="image_dominant").reorder_stage == REORDER_POLICY_SKIPPED
    assert rerank_outcome([], "", "image_dominant") == OUTCOME_POLICY_SKIPPED
    assert _record([], attempted=False).reorder_stage == REORDER_NOT_ATTEMPTED


# ---------------------------------------------------------------------------
# B/C. 실제 라우터 — 그룹 조합별 단계·묶음 수·곡별 설명
# ---------------------------------------------------------------------------

def test_first_group_fails_rest_succeed_is_partial_and_songs_say_which_group_failed():
    client = FakeGeminiClient(mode="fail_group_first", sleep_seconds=0.05)
    reranker = _reranker(client, max_retries=1)
    protected, others = _protected(second_tier=False), _others()
    rec = ExplainRecorder("q")
    results, _ = _search(_router(others, lyrics_hits=protected, reranker=reranker), _lyric_analysis(),
                         top_k=6, recorder=rec)
    ids = [t.id for t in results]
    assert ids[:3] == ["p1", "p2", "p3"], "실패한 보호 묶음은 리랭킹 전 순서(보호 배치는 유지)"
    assert ids[3:] == ["o3", "o2", "o1"], "성공한 일반 묶음은 모델 순서"
    assert len(ids) == len(set(ids)) == 6
    r = rec.record
    assert r.rerank_runs == [RERANK_FAILED, RERANK_APPLIED]
    assert r.reorder_stage == REORDER_MODEL_PARTIAL and r.reorder_committed
    assert r.rerank_runs_failed == 1 and r.rerank_runs_budget_exhausted == 0 and r.rerank_model_calls == 2
    # 곡별: 실패 묶음의 곡은 "넘겼는데 실패", 성공 묶음의 곡은 적용
    for sid in ("p1", "p2", "p3"):
        song = r.get(sid)
        assert song.rerank_group_status == RERANK_FAILED and not song.reorder_applied
        text = render_ko(song, reorder_stage=r.reorder_stage)
        assert "이 곡의 묶음은 리랭킹에 실패" in text, text
        assert "넘기지 않음" not in text and "리랭킹이 " not in text, text
    for sid in ("o1", "o2", "o3"):
        assert r.get(sid).rerank_group_status == RERANK_APPLIED and r.get(sid).reorder_applied


def test_first_group_succeeds_then_later_group_fails_is_partial_and_keeps_the_first_result():
    client = ScriptedClient(["normal", "exception"], sleep_seconds=0.05)
    reranker = _reranker(client, max_retries=1)
    protected, others = _protected(second_tier=False), _others()
    rec = ExplainRecorder("q")
    results, _ = _search(_router(others, lyrics_hits=protected, reranker=reranker), _lyric_analysis(),
                         top_k=6, recorder=rec)
    ids = [t.id for t in results]
    assert ids[:3] == ["p1", "p3", "p2"], "앞 묶음의 모델 순서(규칙 적용 후)는 보존"
    assert ids[3:] == ["o1", "o2", "o3"], "실패한 뒤 묶음은 리랭킹 전 순서"
    assert rec.record.rerank_runs == [RERANK_APPLIED, RERANK_FAILED]
    assert rec.record.reorder_stage == REORDER_MODEL_PARTIAL
    assert [c.outcome for c in client.calls] == ["returned", "raised"] and rec.record.rerank_model_calls == 2
    o_text = render_ko(rec.record.get("o1"), reorder_stage=rec.record.reorder_stage)
    assert "이 곡의 묶음은 리랭킹에 실패" in o_text


def test_first_group_succeeds_then_budget_exhausted_is_partial_with_unrun_group_count():
    client = ScriptedClient(["normal", "slow"], sleep_seconds=0.05)
    reranker = _reranker(client, time_budget_seconds=BUDGET)
    protected, others = _protected(), _others()
    rec = ExplainRecorder("q")
    results, elapsed = _search(_router(others, lyrics_hits=protected, reranker=reranker), _lyric_analysis(),
                               top_k=8, recorder=rec)
    assert elapsed < BUDGET + TOLERANCE
    r = rec.record
    assert r.rerank_runs == [RERANK_APPLIED, RERANK_FAILED, RERANK_BUDGET_EXHAUSTED]
    assert r.reorder_stage == REORDER_MODEL_PARTIAL
    assert len(r.rerank_runs) == 3 and r.rerank_runs_applied == 1 and r.rerank_runs_budget_exhausted == 1
    assert r.rerank_model_calls == 2, "묶음 기록 3 ≠ 실제 SDK 호출 2(성공 1 + 취소 1, 미실행 묶음 0)"
    assert [c.outcome for c in client.calls] == ["returned", "cancelled"]
    # 예산 소진 묶음(일반 후보)의 곡은 "모델을 부르지 않음"
    text = render_ko(r.get("o1"), reorder_stage=r.reorder_stage)
    assert "예산이 소진돼 모델을 부르지 않음" in text, text
    assert "넘기지 않음" not in text


def test_all_groups_failed_or_exhausted_is_failed_not_partial():
    client = ScriptedClient(["slow"], sleep_seconds=0.05)
    reranker = _reranker(client, time_budget_seconds=BUDGET)
    protected, others = _protected(), _others()
    rec = ExplainRecorder("q")
    results, _ = _search(_router(others, lyrics_hits=protected, reranker=reranker), _lyric_analysis(),
                         top_k=8, recorder=rec)
    r = rec.record
    assert r.rerank_runs == [RERANK_FAILED, RERANK_BUDGET_EXHAUSTED, RERANK_BUDGET_EXHAUSTED]
    assert r.reorder_stage == REORDER_MODEL_FAILED and r.rerank_runs_applied == 0
    assert r.rerank_model_calls == 1 and REORDER_STAGE_NOTES.get(r.reorder_stage, "") == ""
    assert [t.id for t in results] == ["p1", "p2", "p3", "p4", "p5", "o1", "o2", "o3"]
    # 전체 실패 문구는 기존 그대로(묶음별 문구가 아니다)
    assert "리랭킹" in render_ko(r.get("o1"), reorder_stage=r.reorder_stage)


def test_router_exception_after_group_success_reverts_everything_including_group_status():
    protected = [_track(f"p{i}", title=f"보호{i}", vocal_gender="남성", lyric_match_type="exact", lyric_match_score=0.95)
                 for i in (1, 2)]
    others = _others()
    rec = ExplainRecorder("q")
    stub = _StubReranker(fail_after=2)
    _search(_router(others, lyrics_hits=protected, reranker=stub), _lyric_analysis(), top_k=5, recorder=rec)
    r = rec.record
    assert RERANK_APPLIED in r.rerank_runs and r.rerank_error
    assert r.reorder_committed is False and r.reorder_stage == REORDER_MODEL_FAILED, "부분 적용이 아니다 — 전체 폴백"
    assert all(not s.reorder_applied and s.rerank_group_status == "" for s in r.songs.values())
    assert r.rerank_model_calls is None, "스텁은 호출 수를 세지 않는다 — 추정하지 않는다"


def test_policy_skip_disabled_and_unknown_backend_are_not_mixed_with_failure():
    # 정책 생략(표지 지배, Gemini listwise만 동의): 호출 0 · 단계 policy_skipped · 호출 수 0
    rec = ExplainRecorder("q")
    client = FakeGeminiClient(mode="normal", sleep_seconds=0.05)
    cover = _analysis(original_query="초록 바탕에 흰 도형이 있는 앨범 커버", has_visual_clue=True,
                      image_english_query="green cover", modality_weights=ModalityWeights(text=0.1, image=0.9, audio=0.0))
    _search(_router(_others(n=4), reranker=_reranker(client)), cover, top_k=4, recorder=rec)
    assert rec.record.rerank_policy_skip == "image_dominant", "표지 지배 생략 사례가 성립해야 한다"
    assert rec.record.reorder_stage == REORDER_POLICY_SKIPPED and rec.record.rerank_runs == [] and client.calls == []
    out = SearchExplainOut.of(rec.record)
    assert out.rerank_calls == 0 and out.rerank_groups_failed == 0 and out.rerank_model_calls is None
    assert out.reorder_stage_note == "" and out.rerank_skip_reason == "image_dominant"
    # 비활성: not_attempted
    rec2 = ExplainRecorder("q")
    _search(_router(_others(n=4), reranker=_StubReranker()), _analysis(), top_k=4, recorder=rec2, use_rerank=False)
    assert rec2.record.reorder_stage == REORDER_NOT_ATTEMPTED and len(rec2.record.rerank_runs) == 0
    # 상태 미보고 백엔드: model_unknown · 호출 수 모름
    rec3 = ExplainRecorder("q")
    _search(_router(_others(n=4), reranker=_LegacyReranker()), _analysis(), top_k=4, recorder=rec3)
    assert rec3.record.reorder_stage == REORDER_MODEL_UNKNOWN
    assert rec3.record.rerank_runs == [RERANK_UNKNOWN] and rec3.record.rerank_model_calls is None
    out = SearchExplainOut.of(rec3.record)
    assert out.rerank_calls == 1 and out.rerank_model_calls is None and out.rerank_groups_failed == 0


def test_unreturned_songs_are_not_described_as_placed_by_the_model():
    client = FakeGeminiClient(mode="partial_ids", sleep_seconds=0.05)
    reranker = _reranker(client)
    tracks = _others(n=6)
    rec = ExplainRecorder("q")
    results, _ = _search(_router(tracks, reranker=reranker), _analysis(), top_k=6, recorder=rec)
    r = rec.record
    assert r.reorder_stage == REORDER_MODEL_APPLIED
    returned = {"o1", "o2", "o3"}
    for sid in ("o4", "o5", "o6"):
        song = r.get(sid)
        assert song.reorder_applied and song.rerank_unreturned
        text = render_ko(song, reorder_stage=r.reorder_stage)
        assert "모델 응답에 이 곡이 없어" in text, text
        assert "리랭킹이 " not in text and "묶음 안에서" not in text, text
        assert to_track_explain(song, reorder_stage=r.reorder_stage).rerank_unreturned is True
    for sid in returned:
        song = r.get(sid)
        assert song.reorder_applied and not song.rerank_unreturned
        assert "모델 응답에 이 곡이 없어" not in render_ko(song, reorder_stage=r.reorder_stage)


# ---------------------------------------------------------------------------
# 평가기 — 기록을 꺼도 판정이 유지된다
# ---------------------------------------------------------------------------

def test_evaluator_judges_invalid_response_as_fallback_even_with_explain_off():
    client = FakeGeminiClient(mode="foreign_ids", sleep_seconds=0.05)
    reranker = _reranker(client, max_retries=1)
    candidates = _others(n=5)
    statuses: list = []
    errors: list = []
    final = rerank_preserving_exact_lyrics(reranker, _analysis(), candidates, 5, recorder=NULL_RECORDER,
                                           errors=errors, statuses=statuses)
    assert [t.id for t in final] == [t.id for t in candidates]
    assert statuses == [RERANK_FAILED] and not errors
    assert rerank_outcome(statuses, "") == OUTCOME_FALLBACK


def test_evaluator_partial_failure_matches_service_partial_stage_with_explain_off():
    client = FakeGeminiClient(mode="fail_group_first", sleep_seconds=0.05)
    reranker = _reranker(client, max_retries=1)
    candidates = [*_protected(second_tier=False), *_others()]
    statuses: list = []
    final = rerank_preserving_exact_lyrics(reranker, _lyric_analysis(), candidates, 6, recorder=NULL_RECORDER,
                                           statuses=statuses)
    assert [t.id for t in final] == ["p1", "p2", "p3", "o3", "o2", "o1"]
    assert statuses == [RERANK_FAILED, RERANK_APPLIED]
    assert rerank_outcome(statuses, "") == OUTCOME_PARTIAL_FAILURE
    assert _record(statuses).reorder_stage == REORDER_MODEL_PARTIAL


# ---------------------------------------------------------------------------
# API — 실제 /search 라우트의 JSON
# ---------------------------------------------------------------------------

def _client(router, analyzer):
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = lambda: analyzer
    app.dependency_overrides[get_search_router] = lambda: router
    return TestClient(app)


def _post_lyric_query(reranker, *, protected, others, top_k=8):
    analyzer = _CountingAnalyzer(vocal_gender="남성", lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])
    router = _router(others, lyrics_hits=protected, reranker=reranker)
    try:
        with _client(router, analyzer) as client:
            response = client.post("/search", json={"query": "한 소절 노래", "top_k": top_k, "explain": True})
        assert response.status_code == 200, response.text
        return response.json()
    finally:
        router.shutdown()


def test_route_reports_partial_stage_counts_and_per_song_group_status():
    client = ScriptedClient(["normal", "slow"], sleep_seconds=0.05)
    reranker = _reranker(client, time_budget_seconds=BUDGET)
    body = _post_lyric_query(reranker, protected=_protected(), others=_others())
    ex = body["explain"]
    assert ex["reorder_stage"] == REORDER_MODEL_PARTIAL
    assert ex["reorder_stage_label"] == "리랭킹 일부 적용"
    assert ex["reorder_stage_note"] == REORDER_STAGE_NOTES[REORDER_MODEL_PARTIAL]
    assert ex["rerank_calls"] == 3 and ex["rerank_calls_applied"] == 1, "호환 필드 의미 그대로(묶음 기록 수)"
    assert ex["rerank_groups_failed"] == 1 and ex["rerank_groups_budget_exhausted"] == 1
    assert ex["rerank_model_calls"] == 2
    assert ex["rerank_error"] == "" and ex["rerank_skip_reason"] == ""
    ids = [t["id"] for t in body["results"]]
    assert ids == ["p1", "p3", "p2", "p4", "p5", "o1", "o2", "o3"] and len(set(ids)) == 8
    by_id = {t["id"]: t["explain"] for t in body["results"]}
    assert by_id["p1"]["reorder_applied"] and by_id["p1"]["rerank_group_status"] == RERANK_APPLIED
    assert not by_id["p4"]["reorder_applied"] and by_id["p4"]["rerank_group_status"] == RERANK_FAILED
    assert not by_id["o1"]["reorder_applied"] and by_id["o1"]["rerank_group_status"] == RERANK_BUDGET_EXHAUSTED
    assert "예산이 소진돼 모델을 부르지 않음" in by_id["o1"]["summary"]
    assert all(t["explain"]["reorder_stage"] == REORDER_MODEL_PARTIAL for t in body["results"])
    # 내부 예외 원문·모델 응답 전문은 응답에 없다
    dumped = json.dumps(body, ensure_ascii=False)
    assert "Traceback" not in dumped and "v56 가짜 응답" not in dumped and "InvalidRankingResponse" not in dumped


def test_route_reports_failed_stage_when_only_invalid_responses_come_back():
    client = FakeGeminiClient(mode="foreign_ids", sleep_seconds=0.05)
    reranker = _reranker(client, max_retries=1)
    body = _post_lyric_query(reranker, protected=[], others=_others(n=5), top_k=5)
    ex = body["explain"]
    assert ex["reorder_stage"] == REORDER_MODEL_FAILED and ex["reorder_stage_label"] == "리랭킹 실패"
    assert ex["rerank_calls"] == 1 and ex["rerank_calls_applied"] == 0 and ex["rerank_groups_failed"] == 1
    assert ex["rerank_model_calls"] == 1 and ex["reorder_stage_note"] == ""
    assert [t["id"] for t in body["results"]] == ["o1", "o2", "o3", "o4", "o5"]
    assert not any(t["explain"]["reorder_applied"] for t in body["results"])


def test_route_reports_applied_stage_with_identity_order_as_normal():
    client = FakeGeminiClient(mode="identity", sleep_seconds=0.05)
    reranker = _reranker(client)
    body = _post_lyric_query(reranker, protected=[], others=_others(n=5), top_k=5)
    ex = body["explain"]
    assert ex["reorder_stage"] == REORDER_MODEL_APPLIED and ex["rerank_calls"] == ex["rerank_calls_applied"] == 1
    assert ex["rerank_groups_failed"] == 0 and ex["rerank_groups_budget_exhausted"] == 0 and ex["rerank_model_calls"] == 1
    assert [t["id"] for t in body["results"]] == ["o1", "o2", "o3", "o4", "o5"], "입력 순서와 같아도 정상 응답"
    assert all(t["explain"]["reorder_applied"] and not t["explain"]["rerank_unreturned"] for t in body["results"])


def test_route_keeps_response_shape_for_backends_that_do_not_count_calls():
    body = _post_lyric_query(_StubReranker(), protected=[], others=_others(n=4), top_k=4)
    ex = body["explain"]
    assert ex["reorder_stage"] == REORDER_MODEL_APPLIED and ex["rerank_model_calls"] is None
    assert ex["rerank_calls"] == 1 and ex["rerank_groups_failed"] == 0


def test_search_explain_out_rejects_nothing_new_and_keeps_legacy_fields():
    out = SearchExplainOut.of(_record([RERANK_APPLIED, RERANK_BUDGET_EXHAUSTED]))
    dumped = out.model_dump()
    for key in ("reorder_stage", "reorder_stage_label", "rerank_calls", "rerank_calls_applied", "rerank_error"):
        assert key in dumped
    assert dumped["reorder_stage"] == REORDER_MODEL_PARTIAL and dumped["rerank_groups_budget_exhausted"] == 1


# ---------------------------------------------------------------------------
# 리뷰 반영(2026-10-16)
# ---------------------------------------------------------------------------

def test_client_creation_failure_counts_zero_model_calls(monkeypatch):
    """[리뷰 1] 호출 수는 SDK `generate_content`를 실제로 부른 자리에서 센다 — 클라이언트 생성이 실패하면 0이다."""
    reranker = _reranker(FakeGeminiClient(mode="normal"), max_retries=3)

    def boom(self):
        raise RuntimeError("client factory failed (시험)")

    monkeypatch.setattr(GeminiListwiseReranker, "_new_client", boom)
    tracks = _tracks()
    run = reranker.rerank_run("q", tracks, top_k=6)
    assert run.status == RERANK_FAILED and [t.id for t in run.tracks] == [t.id for t in tracks]
    assert run.model_calls == 0, "모델을 한 번도 부르지 않았는데 호출 수가 올라가면 안 된다"


def test_cancelled_and_sync_calls_are_both_counted():
    """취소된 비동기 호출도 SDK를 불렀으니 1회다. 비동기 면이 없는 동기 가짜도 같은 자리에서 센다."""
    slow = ScriptedClient(["slow"], sleep_seconds=0.05)
    run = _reranker(slow, request_timeout_seconds=0.3, time_budget_seconds=0.5, pass_reserve_seconds=0.1).rerank_run("q", _tracks(), top_k=6)
    assert run.status == RERANK_FAILED and run.model_calls == len(slow.calls) >= 1

    class SyncOnly(FakeGeminiClient):
        def __init__(self):
            super().__init__(mode="normal")
            del self.aio

    sync = SyncOnly()
    run = _reranker(sync).rerank_run("q", _tracks(), top_k=6)
    assert run.status == RERANK_APPLIED and run.model_calls == len(sync.calls) == 1


class _LowConfidenceLastOnly(FakeGeminiClient):
    """[리뷰 2] 마지막 후보 하나만, 낮은 신뢰도로 돌려준다 — 보충된 곡이 합성 뒤 반환된 곡보다 **앞에** 놓일 수 있다."""

    def _respond(self, call, mode):
        return _Response(json.dumps({"query_confidence": 0.1, "ranking": [{"id": call.candidate_ids[-1], "relevance": 0.2}]}))


def test_unreturned_songs_may_precede_the_returned_one_and_the_text_does_not_claim_otherwise():
    client = _LowConfidenceLastOnly(mode="normal", sleep_seconds=0.05)
    tracks = _others(n=5)
    rec = ExplainRecorder("q")
    results, _ = _search(_router(tracks, reranker=_reranker(client)), _analysis(), top_k=5, recorder=rec)
    ids = [t.id for t in results]
    r = rec.record
    assert r.reorder_stage == REORDER_MODEL_APPLIED and sorted(r.songs["o5"].rerank_group_status for _ in [0]) == [RERANK_APPLIED]
    assert r.get("o5").reorder_applied and not r.get("o5").rerank_unreturned
    unreturned = [sid for sid in ids if r.get(sid).rerank_unreturned]
    assert sorted(unreturned) == ["o1", "o2", "o3", "o4"]
    # 저신뢰 합성에서 검색 점수가 앞서 보충 곡이 반환 곡보다 앞에 온다 — "묶음 뒤에 배치"는 거짓이 된다
    assert ids.index("o5") > 0, f"반환된 곡이 1위가 아니어야 이 사례가 성립한다: {ids}"
    for sid in unreturned:
        text = render_ko(r.get(sid), reorder_stage=r.reorder_stage)
        assert "입력 순서로 보충한 뒤 점수 합성·순서 규칙을 적용" in text, text
        assert "묶음 뒤" not in text and "리랭킹이 " not in text, text
