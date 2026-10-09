"""실험 스위치 CLARIFY_RERANK_INPUT_ORDER가 라우터에서 **답을 쓰는 리랭커의 입력에만** 닿는지 (results_clarify_v11 리뷰).

- 답을 쓰는 리랭커가 적용 실패를 보고하면 최종 결과는 보너스 순서(후보 목록)다
- 답을 안 받는 리랭커(CE)는 입력이 보너스 순서다
- 답을 쓰는 리랭커가 돌면 입력은 보너스 전 순서이고, 후보 목록(candidate_ids_out)은 보너스 순서 그대로다
"""
import asyncio

import pytest

from src.backend.schemas.search import ClarifyAnswer
from src.retrieval.explain import RERANK_APPLIED, RERANK_FAILED, RerankRun
from tests.test_search_explain import _analysis, _router, _track


class _AnswerReranker:
    """답을 받는 listwise 흉내. identity면 입력 순서 그대로, fail이면 Gemini처럼 예외 없이 실패를 보고한다."""

    enabled = True
    uses_clarify_answers = True

    def __init__(self, fail=False):
        self.fail = fail
        self.seen = []
        self.answers = []

    def rerank_run(self, query, tracks, top_k, answers=None):
        self.seen.append([t.id for t in tracks])
        self.answers.append(list(answers or []))
        if self.fail:
            return RerankRun(list(tracks)[:top_k], RERANK_FAILED, [])
        out = [t.model_copy(update={"rerank_score": 0.9}) for t in tracks]
        return RerankRun(out[:top_k], RERANK_APPLIED, [t.id for t in tracks])


class _PlainReranker:
    """CE처럼 답을 안 받는 리랭커."""

    enabled = True
    uses_clarify_answers = False

    def __init__(self):
        self.seen = []

    def rerank_run(self, query, tracks, top_k):
        self.seen.append([t.id for t in tracks])
        out = [t.model_copy(update={"rerank_score": 0.9}) for t in tracks]
        return RerankRun(out[:top_k], RERANK_APPLIED, [t.id for t in tracks])


def _hits():
    # 보너스(남성)는 s2·s4만 올린다 → 보너스 전 s1 s2 s3 s4, 보너스 뒤 s2 s4 s1 s3 (점수 간격이 작아 보너스가 뒤집는다)
    return [
        _track("s1", score=0.0160, vocal_gender="여성"),
        _track("s2", score=0.0159, vocal_gender="남성"),
        _track("s3", score=0.0158, vocal_gender="여성"),
        _track("s4", score=0.0157, vocal_gender="남성"),
    ]


def _search(router, **kw):
    pool, rerank_in = [], []
    try:
        out = asyncio.run(router.search(
            _analysis(original_query="남자 발라드"), top_k=4, candidate_k=4,
            answers=[ClarifyAnswer(slot="vocal_gender", value="남성")],
            candidate_ids_out=pool, rerank_input_tracks_out=rerank_in, **kw,
        ))
    finally:
        router.shutdown()
    return [t.id for t in out], pool, [t.id for t in rerank_in]


def test_pre_bonus_reaches_only_the_answer_using_reranker_input(monkeypatch):
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "pre_bonus")
    rr = _AnswerReranker()
    final, pool, rerank_in = _search(_router(_hits(), reranker=rr))
    assert pool == ["s2", "s4", "s1", "s3"], "후보 목록은 보너스 순서 그대로"
    assert rr.seen[0] == ["s1", "s2", "s3", "s4"] == rerank_in, "리랭커 입력만 보너스 전 순서"
    assert final == ["s1", "s2", "s3", "s4"]
    assert rr.answers[0] and rr.answers[0][0].value == "남성", "답은 그대로 프롬프트로 간다"


def test_reported_failure_falls_back_to_the_bonus_order(monkeypatch):
    """적용 실패를 보고하면 보너스 전 입력 순서가 아니라 후보(보너스) 순서로 돌아간다."""
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "pre_bonus")
    rr = _AnswerReranker(fail=True)
    final, pool, _ = _search(_router(_hits(), reranker=rr))
    assert rr.seen[0] == ["s1", "s2", "s3", "s4"]
    assert final == pool == ["s2", "s4", "s1", "s3"]


def test_reranker_without_answers_gets_the_bonus_order(monkeypatch):
    """CE처럼 답을 안 받는 리랭커는 스위치와 무관하게 보너스 순서를 받는다."""
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "pre_bonus")
    rr = _PlainReranker()
    final, pool, rerank_in = _search(_router(_hits(), reranker=rr))
    assert rr.seen[0] == pool == rerank_in == ["s2", "s4", "s1", "s3"]


def test_switch_off_leaves_everything_in_bonus_order(monkeypatch):
    monkeypatch.delenv("CLARIFY_RERANK_INPUT_ORDER", raising=False)
    rr = _AnswerReranker()
    final, pool, rerank_in = _search(_router(_hits(), reranker=rr))
    assert rr.seen[0] == pool == rerank_in == final == ["s2", "s4", "s1", "s3"]


def test_placebo_shuffles_only_bonus_changed_positions_and_drops_answers(monkeypatch):
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "placebo:7")
    rr = _AnswerReranker()
    final, pool, rerank_in = _search(_router(_hits(), reranker=rr))
    assert pool == ["s2", "s4", "s1", "s3"]
    assert set(rerank_in) == {"s1", "s2", "s3", "s4"} and rerank_in != ["s1", "s2", "s3", "s4"]
    assert rr.answers[0] == [], "위약에서는 답이 프롬프트에 안 간다"


class _SplitReranker:
    """보호 그룹 호출은 적용, 일반 후보 호출은 Gemini처럼 예외 없이 실패를 보고한다 (PR #30 리뷰 [참고])."""

    enabled = True
    uses_clarify_answers = True

    def __init__(self):
        self.seen = []

    def rerank_run(self, query, tracks, top_k, answers=None):
        ids = [t.id for t in tracks]
        self.seen.append(ids)
        if all(i.startswith("p") for i in ids):  # 보호 그룹
            out = [t.model_copy(update={"rerank_score": 0.9}) for t in reversed(list(tracks))]
            return RerankRun(out[:top_k], RERANK_APPLIED, ids)
        return RerankRun(list(tracks)[:top_k], RERANK_FAILED, [])


def _mixed_hits():
    # 보호 곡 p1·p2(가사 exact) + 일반 s1~s4. 보너스(남성)는 s2·s4를 올린다 → 일반 후보 보너스 순서 s2 s4 s1 s3
    return [
        _track("p1", score=0.0200, lyric_match_type="exact", lyric_match_score=0.95, vocal_gender="여성"),
        _track("p2", score=0.0199, lyric_match_type="exact", lyric_match_score=0.95, vocal_gender="여성"),
        _track("s1", score=0.0160, vocal_gender="여성"),
        _track("s2", score=0.0159, vocal_gender="남성"),
        _track("s3", score=0.0158, vocal_gender="여성"),
        _track("s4", score=0.0157, vocal_gender="남성"),
    ]


def test_partial_failure_falls_back_per_call_to_the_bonus_order(monkeypatch):
    """보호 그룹은 적용·일반 후보만 실패 → 일반 후보는 보너스 전 입력 순서가 아니라 보너스 순서로 나간다."""
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "pre_bonus")
    rr = _SplitReranker()
    pool, rerank_in = [], []
    router = _router(_mixed_hits(), reranker=rr)
    try:
        out = asyncio.run(router.search(
            _analysis(original_query="남자 발라드"), top_k=6, candidate_k=6,
            answers=[ClarifyAnswer(slot="vocal_gender", value="남성")],
            candidate_ids_out=pool, rerank_input_tracks_out=rerank_in,
        ))
    finally:
        router.shutdown()
    final = [t.id for t in out]
    others_bonus = [i for i in pool if i.startswith("s")]
    assert others_bonus == ["s2", "s4", "s1", "s3"]
    assert final[:2] == ["p2", "p1"], "보호 그룹은 적용된 결과"
    assert final[2:] == others_bonus, "일반 후보는 보너스 순서로 폴백(입력 순서 s1 s2 s3 s4가 아님)"


def test_plain_path_fallback_reports_model_failed_in_explain(monkeypatch):
    """일반 경로 폴백 뒤 설명 단계는 MODEL_FAILED여야 한다 — revoke 뒤 commit이면 다시 켜진다 (PR #30 리뷰 [권장])."""
    from src.retrieval.explain import REORDER_MODEL_FAILED, ExplainRecorder
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "pre_bonus")
    rr = _AnswerReranker(fail=True)
    rec = ExplainRecorder("q")
    router = _router(_hits(), reranker=rr)
    try:
        out = asyncio.run(router.search(
            _analysis(original_query="남자 발라드"), top_k=4, candidate_k=4,
            answers=[ClarifyAnswer(slot="vocal_gender", value="남성")], recorder=rec,
        ))
    finally:
        router.shutdown()
    assert [t.id for t in out] == ["s2", "s4", "s1", "s3"]
    assert rec.record.reorder_committed is False
    assert rec.record.reorder_stage == REORDER_MODEL_FAILED
