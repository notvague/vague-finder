"""v56 — 가짜 SDK 클라이언트를 꽂은 **실제** `GeminiListwiseReranker`와 `SearchRouter`의 실패·폴백 경로.

리랭커를 통째로 바꾼 기존 스텁(`_StubReranker`)과 달리 여기서는 SDK 클라이언트만 가짜다. 재시도·JSON 파서·
시간 예산(`asyncio.wait_for` 취소)·`RerankRun` 상태 보고·라우터의 그룹 합치기가 운영 코드 그대로 돈다.
외부 호출 없음 — Gemini 설정 없이도 `configured=True`로 켠다(가짜 클라이언트가 주입돼 네트워크가 없다).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from src.backend.schemas.query import LyricClue, QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval import search_router as sr
from src.retrieval.explain import RERANK_APPLIED, RERANK_FAILED, ExplainRecorder
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig
from tests.fake_gemini_client import FOREIGN_ID_PREFIX, FakeGeminiClient


def _cfg(**over):
    base = dict(enabled=True, max_retries=3, passes=1, request_timeout_seconds=0.6, time_budget_seconds=1.0,
                pass_reserve_seconds=0.2, use_search_grounding=False, rare_fact_verification=False)
    base.update(over)
    return GeminiListwiseRerankerConfig(**base)


def _tracks(n: int = 6, prefix: str = "1000") -> list:
    return [MatchingTrack(id=f"{prefix}{i}", score=1.0 - i * 0.01, retrieval_score=1.0 - i * 0.01,
                          title=f"곡{i}", artist=f"가수{i}") for i in range(1, n + 1)]


def _reranker(mode: str, **cfg) -> tuple[GeminiListwiseReranker, FakeGeminiClient]:
    client = FakeGeminiClient(mode=mode, sleep_seconds=0.05)
    return GeminiListwiseReranker(_cfg(**cfg), client=client, configured=True), client


# --- 리랭커 단독: 가짜 응답이 실제 파서·재시도·예산을 지난다 -----------------------------

def test_normal_response_is_applied_and_reversed():
    reranker, client = _reranker("normal")
    tracks = _tracks()
    run = reranker.rerank_run("q", tracks, top_k=6)
    assert run.status == RERANK_APPLIED
    assert client.calls[0].candidate_ids == [t.id for t in tracks]
    assert [t.id for t in run.tracks] == [t.id for t in reversed(tracks)]
    assert len(client.calls) == 1


def test_exception_retries_three_times_then_reports_failed_without_raising():
    # 재시도 사이 1초 대기가 예산 안이어야 3회가 다 돈다(운영 예산 30초). 예산이 1초면 1회 뒤 예산 초과로 끝난다.
    reranker, client = _reranker("exception", time_budget_seconds=5.0)
    tracks = _tracks()
    run = reranker.rerank_run("q", tracks, top_k=6)
    assert run.status == RERANK_FAILED
    assert [t.id for t in run.tracks] == [t.id for t in tracks]  # 입력 순서 유지
    assert len(client.calls) == 3 and all(c.outcome == "raised" for c in client.calls)


def test_slow_response_is_cancelled_by_wall_clock_and_falls_back():
    reranker, client = _reranker("slow", request_timeout_seconds=0.3, time_budget_seconds=0.5, pass_reserve_seconds=0.1)
    tracks = _tracks()
    run = reranker.rerank_run("q", tracks, top_k=6)
    assert run.status == RERANK_FAILED
    assert [t.id for t in run.tracks] == [t.id for t in tracks]
    assert client.calls and client.calls[0].outcome == "cancelled"  # wait_for 가 실제로 끊었다


def test_slow_within_budget_is_applied():
    reranker, client = _reranker("slow_within", request_timeout_seconds=2.0, time_budget_seconds=3.0)
    client.set_mode("slow_within", sleep_seconds=0.2)
    run = reranker.rerank_run("q", _tracks(), top_k=6)
    assert run.status == RERANK_APPLIED
    assert client.calls[0].elapsed_s >= 0.2


def test_invalid_json_retries_with_correction_then_fails():
    reranker, client = _reranker("invalid_json", time_budget_seconds=5.0)
    run = reranker.rerank_run("q", _tracks(), top_k=6)
    assert run.status == RERANK_FAILED
    assert len(client.calls) == 3


def test_foreign_ids_are_dropped_and_input_order_is_kept_but_status_applied():
    reranker, client = _reranker("foreign_ids")
    tracks = _tracks()
    run = reranker.rerank_run("q", tracks, top_k=6)
    assert run.status == RERANK_APPLIED
    assert [t.id for t in run.tracks] == [t.id for t in tracks]
    assert not any(t.id.startswith(FOREIGN_ID_PREFIX) for t in run.tracks)


def test_mode_file_switches_mode_between_calls(tmp_path):
    mode_file = tmp_path / "mode.json"
    mode_file.write_text(json.dumps({"mode": "exception"}), encoding="utf-8")
    client = FakeGeminiClient(mode_file=str(mode_file), log_path=str(tmp_path / "calls.jsonl"))
    reranker = GeminiListwiseReranker(_cfg(time_budget_seconds=5.0), client=client, configured=True)
    assert reranker.rerank_run("q", _tracks(), top_k=6).status == RERANK_FAILED
    mode_file.write_text(json.dumps({"mode": "normal"}), encoding="utf-8")
    assert reranker.rerank_run("q", _tracks(), top_k=6).status == RERANK_APPLIED
    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and json.loads(lines[-1])["outcome"] == "returned"


# --- 라우터: 그룹 하나만 실패했을 때 합치는 방식 -----------------------------------------

class _TextService:
    def __init__(self, hits):
        self._hits = hits

    def search_text(self, query, top_k=10, alpha=0.5, sparse_query=None, metadata_filter=None):
        return [t.model_copy() for t in self._hits[:top_k]]


class _Embedder:
    def embed_texts(self, texts, l2_normalize=True):
        return [[0.0] * 4 for _ in texts]


class _Index:
    def query(self, **kwargs):
        return {"matches": []}

    def fetch(self, ids, namespace=None, fields=None):
        return {"matches": []}


class _Client:
    def Index(self, name):  # noqa: N802
        return _Index()


class _Lyrics:
    def __init__(self, hits):
        self._hits = hits

    def search(self, clues, top_k=None, snapshot_out=None):
        return [t.model_copy() for t in self._hits]


def _lyric_track(song_id: str, title: str) -> MatchingTrack:
    return MatchingTrack(id=song_id, score=1.0, title=title, artist="가수", lyric_match_type="exact",
                         lyric_match_score=0.95, lyric_match_phrase="여보세요")


def _router(reranker, text_hits, lyric_hits):
    return sr.SearchRouter(
        search_service=_TextService(text_hits), image_embedder=_Embedder(), audio_embedder=_Embedder(),
        vector_client=_Client(), lyrics_search_service=_Lyrics(lyric_hits), reranker=reranker, max_workers=2,
    )


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query="여보세요 하고 시작하는 노래", intent_type="lyrics", korean_tags=["가사"],
        lyric_clues=[LyricClue(text="여보세요", kind="verbatim", confidence=0.95)],
        image_english_query="", audio_english_query="",
        modality_weights={"text": 1.0, "image": 0.0, "audio": 0.0}, confidence=0.9,
    )


def _run(router, **kw):
    try:
        return asyncio.run(router.search(_analysis(), **kw))
    finally:
        router.shutdown()


@pytest.mark.parametrize("mode", ["fail_group_first", "normal"])
def test_protected_group_failure_keeps_group_order_and_reranks_the_rest(mode):
    """가사 exact 보호 그룹(3곡) 호출이 실패해도 그룹은 앞자리를 지키고(입력 순서), 일반 후보만 리랭킹된다.
    상태는 호출별로 남는다(failed | applied) — 전체 실패(검색 순서 폴백)와 다르다."""
    client = FakeGeminiClient(mode=mode)
    reranker = GeminiListwiseReranker(_cfg(time_budget_seconds=5.0), client=client, configured=True)
    protected = [_lyric_track("2001", "보호1"), _lyric_track("2002", "보호2"), _lyric_track("2003", "보호3")]
    others = _tracks(5, prefix="3000")
    router = _router(reranker, text_hits=[*others], lyric_hits=protected)
    recorder = ExplainRecorder("q")
    statuses: list = []
    orig = sr.call_reranker

    def spy(reranker_, query, tracks, top_k, answers=None, recorder_=None, runs_out=None, **kw):
        out = orig(reranker_, query, tracks, top_k, answers, recorder_ or recorder, runs_out)
        statuses.append((len(tracks), runs_out[-1] if runs_out else None))
        return out

    sr.call_reranker = spy
    try:
        results = _run(router, top_k=8, recorder=recorder)
    finally:
        sr.call_reranker = orig

    ids = [t.id for t in results]
    assert len(ids) == len(set(ids)) == 8
    protected_ids = [t.id for t in protected]
    if mode == "fail_group_first":
        assert ids[:3] == protected_ids  # 그룹 호출 실패 → 그 그룹은 입력(보정) 순서 그대로
    else:
        # 그룹 호출 성공 → 가짜가 역순으로 냈지만 실물 규칙(`_protect_strong_surface_lyric_anchor`)이 검색 1위의 exact 일치를 1위로 되돌린다
        assert ids[:3] == [protected_ids[0], protected_ids[2], protected_ids[1]]
    # 일반 후보는 두 모드 모두 리랭킹(역순)됐다
    assert ids[3:] == [t.id for t in reversed(others)]
    group_status = [s for n, s in statuses if n == 3]
    assert group_status == ([RERANK_FAILED] if mode == "fail_group_first" else [RERANK_APPLIED])
    assert [s for n, s in statuses if n == 5] == [RERANK_APPLIED]
    assert len(recorder.record.rerank_runs) == 2
    assert recorder.record.rerank_runs_applied == (1 if mode == "fail_group_first" else 2)


def test_failure_state_does_not_leak_into_next_request():
    client = FakeGeminiClient(mode="exception")
    reranker = GeminiListwiseReranker(_cfg(time_budget_seconds=5.0), client=client, configured=True)
    tracks = _tracks(5)
    first = _router(reranker, text_hits=tracks, lyric_hits=[])
    rec1 = ExplainRecorder("q")
    out1 = _run(first, top_k=5, recorder=rec1)
    assert [t.id for t in out1] == [t.id for t in tracks]
    assert rec1.record.rerank_runs_applied == 0 and rec1.record.rerank_runs == [RERANK_FAILED]

    client.set_mode("normal")
    second = _router(reranker, text_hits=tracks, lyric_hits=[])
    rec2 = ExplainRecorder("q")
    out2 = _run(second, top_k=5, recorder=rec2)
    assert [t.id for t in out2] == [t.id for t in reversed(tracks)]
    assert rec2.record.rerank_runs == [RERANK_APPLIED] and not rec2.record.rerank_error
