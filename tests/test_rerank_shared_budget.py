"""v57 — 요청 하나의 리랭킹 단계가 **하나의 마감**을 공유한다(가사 exact 보호 그룹·일반 후보·재시도).

v56 운영 검증(results_v56_attr_runtime_validation §7-①)에서 보호 그룹 질의가 모델 무응답 때 그룹 수 × 30초
(n007 93초 · t018 62초 > 화면 제한 60초)가 걸렸다. 호출마다 `rerank_run`이 예산을 새로 만들었고 재시도 사이
`time.sleep(1)`도 예산을 넘겼다. 여기서는 실제 `GeminiListwiseReranker` + 실제 `SearchRouter`에 v56의 가짜 SDK
클라이언트만 꽂아 그 구조를 짧은 예산(1초)으로 재현하고, 수정 뒤 동작을 고정한다.

시간 시험은 짧은 예산과 스케줄링 여유(0.5초)를 쓰되, 시간만 보지 않는다 — 가짜 호출 수·취소 여부·호출별 제한·
상태 목록·반환 순서를 함께 확인한다. 외부 호출 없음(`configured=True` + 주입 클라이언트).
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from src.backend.schemas.search import ClarifyAnswer
from src.retrieval import search_router as sr
from src.retrieval.evaluate_search_accuracy import (
    OUTCOME_FALLBACK,
    OUTCOME_PARTIAL_FAILURE,
    rerank_outcome,
    rerank_preserving_exact_lyrics,
)
from src.retrieval.explain import (
    REORDER_MODEL_APPLIED,
    REORDER_MODEL_FAILED,
    RERANK_APPLIED,
    RERANK_BUDGET_EXHAUSTED,
    RERANK_FAILED,
    ExplainRecorder,
    SearchExplain,
)
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig
from tests.fake_gemini_client import FakeGeminiClient, candidate_ids_from_prompt
from tests.test_search_explain import _StubReranker, _analysis, _router, _track

BUDGET = 1.0          # 운영 30초의 1/30 — 구조는 같다
REQ_TIMEOUT = 0.6     # 운영 20초에 대응
TOLERANCE = 0.5       # 스케줄링·스레드 전환 여유


def _cfg(**over) -> GeminiListwiseRerankerConfig:
    base = dict(enabled=True, max_retries=3, passes=1, request_timeout_seconds=REQ_TIMEOUT,
                time_budget_seconds=BUDGET, pass_reserve_seconds=0.2, use_search_grounding=False,
                rare_fact_verification=False)
    base.update(over)
    return GeminiListwiseRerankerConfig(**base)


class ScriptedClient(FakeGeminiClient):
    """호출 순서대로 모드를 정한다 — 한 요청 안에서 그룹마다 다른 응답(정상 → 느림 …). 목록이 끝나면 마지막 모드 반복."""

    def __init__(self, script, **kw):
        super().__init__(mode=script[0], **kw)
        self.script = list(script)

    def _current(self):
        return {"mode": self.script[min(len(self.calls), len(self.script) - 1)], "sleep": self._sleep}


class PrefixScriptedClient(FakeGeminiClient):
    """후보 id 접두로 요청을 구분해 요청마다 다른 대본을 준다(동시 요청 시험). 스레드별로 대본 색인을 따로 센다."""

    def __init__(self, scripts_by_prefix, **kw):
        super().__init__(mode="normal", **kw)
        self._scripts = dict(scripts_by_prefix)
        self._cursor = {prefix: 0 for prefix in scripts_by_prefix}
        self._lock = threading.Lock()

    async def _generate_async(self, contents):
        ids = candidate_ids_from_prompt(contents)
        prefix = next((p for p in self._scripts if ids and ids[0].startswith(p)), None)
        with self._lock:
            if prefix is None:
                mode, sleep = "normal", self._sleep
            else:
                script = self._scripts[prefix]
                step = script[min(self._cursor[prefix], len(script) - 1)]
                self._cursor[prefix] += 1
                mode, sleep = (step if isinstance(step, tuple) else (step, self._sleep))
            call = self._begin(contents, mode)
        try:
            if mode == "slow":
                await asyncio.sleep(3600)
            elif mode == "slow_within":
                await asyncio.sleep(sleep)
            response = self._respond(call, mode)
        except asyncio.CancelledError:
            self._end(call, "cancelled")
            raise
        except Exception:
            self._end(call, "raised")
            raise
        self._end(call, "returned")
        return response


def _protected(prefix: str = "p", *, second_tier: bool = True):
    """같은 exact 구절의 보호 곡 — 남성 3곡(분석과 성별 일치 tier) + 여성 2곡(불일치 tier). 그룹 호출 2회가 생긴다."""
    tracks = [
        _track(f"{prefix}{i}", title=f"보호{i}", vocal_gender="남성", lyric_match_type="exact", lyric_match_score=0.95)
        for i in (1, 2, 3)
    ]
    if second_tier:
        tracks += [
            _track(f"{prefix}{i}", title=f"보호{i}", vocal_gender="여성", lyric_match_type="exact", lyric_match_score=0.95)
            for i in (4, 5)
        ]
    return tracks


def _others(prefix: str = "o", n: int = 3):
    return [_track(f"{prefix}{i}", title=f"일반{i}") for i in range(1, n + 1)]


def _lyric_analysis(**over):
    base = dict(vocal_gender="남성", lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])
    base.update(over)
    return _analysis(**base)


def _spy_statuses(monkeypatch):
    """call_reranker 호출마다 (후보 수, 보고 상태)를 남긴다 — 라우터가 호출을 시작했는지와 결과를 분리해 본다."""
    seen: list = []
    orig = sr.call_reranker

    def spy(reranker, query, tracks, top_k, answers=None, recorder=None, runs_out=None, **kw):
        out = orig(reranker, query, tracks, top_k, answers, recorder if recorder is not None else sr.NULL_RECORDER,
                   runs_out, **kw)
        seen.append((len(tracks), runs_out[-1] if runs_out else None))
        return out

    monkeypatch.setattr(sr, "call_reranker", spy)
    return seen


def _search(router, analysis, **kw):
    started = time.monotonic()
    try:
        results = asyncio.run(router.search(analysis, **kw))
    finally:
        router.shutdown()
    return results, time.monotonic() - started


# --- 재현: 보호 그룹 질의의 예산 공유 ------------------------------------------------------------

def test_all_groups_slow_share_one_budget(monkeypatch):
    """n007형(보호 tier 2 + 일반 = 호출 3회)이 전부 무응답이어도 리랭킹 단계는 예산 1회분 안에 끝난다.

    수정 전: 호출마다 예산(1.0초) + 재시도 대기(1초)로 3 × 1.6초 = 4.8초. 수정 뒤: 첫 호출이 min(0.6, 남은) 뒤 취소되고
    재시도 대기가 남은 0.4초로 잘려 1.0초에 예산이 끝난다. 나머지 두 그룹은 **모델을 부르지 않는다**.
    """
    client = ScriptedClient(["slow"])
    reranker = GeminiListwiseReranker(_cfg(), client=client, configured=True)
    protected, others = _protected(), _others()
    rec = ExplainRecorder("q")
    seen = _spy_statuses(monkeypatch)

    results, elapsed = _search(_router(others, lyrics_hits=protected, reranker=reranker), _lyric_analysis(),
                               top_k=8, recorder=rec)

    assert elapsed < BUDGET + TOLERANCE, f"리랭킹 단계가 예산을 넘겼다: {elapsed:.2f}s"
    assert [c.outcome for c in client.calls] == ["cancelled"], "첫 그룹만 불렀고 wait_for가 끊었다"
    assert client.calls[0].elapsed_s < REQ_TIMEOUT + 0.2
    assert [s for _, s in seen] == [RERANK_FAILED, RERANK_BUDGET_EXHAUSTED, RERANK_BUDGET_EXHAUSTED]
    # 폴백 순서: 보호 tier(성별 일치 → 불일치) 안은 입력 순서, 일반 후보도 입력 순서
    assert [t.id for t in results] == ["p1", "p2", "p3", "p4", "p5", "o1", "o2", "o3"]
    assert rec.record.rerank_runs == [RERANK_FAILED, RERANK_BUDGET_EXHAUSTED, RERANK_BUDGET_EXHAUSTED]
    assert rec.record.rerank_runs_applied == 0
    assert rec.record.reorder_stage == REORDER_MODEL_FAILED, "미실행 그룹이 성공이나 다른 단계로 보이면 안 된다"
    assert not rec.record.rerank_error, "예외 폴백이 아니다 — 백엔드가 상태로 보고했다"


def test_each_call_limit_is_min_of_single_limit_and_remaining_budget():
    """재시도 없이(1회) 세 그룹이 모두 불리면 셋째 호출의 제한은 남은 예산(0.2초)이다 — 단일 제한 0.3초가 아니다."""
    client = ScriptedClient(["slow"])
    reranker = GeminiListwiseReranker(_cfg(request_timeout_seconds=0.3, time_budget_seconds=0.8, max_retries=1),
                                      client=client, configured=True)
    protected, others = _protected(), _others()

    results, elapsed = _search(_router(others, lyrics_hits=protected, reranker=reranker), _lyric_analysis(), top_k=8)

    assert elapsed < 0.8 + TOLERANCE
    assert [c.outcome for c in client.calls] == ["cancelled", "cancelled", "cancelled"]
    assert client.calls[0].elapsed_s < 0.3 + 0.1 and client.calls[1].elapsed_s < 0.3 + 0.1
    assert client.calls[2].elapsed_s < 0.28, f"셋째 호출이 남은 예산(0.2초)보다 길었다: {client.calls[2].elapsed_s}"
    assert [t.id for t in results] == ["p1", "p2", "p3", "p4", "p5", "o1", "o2", "o3"]


def test_first_group_applied_then_budget_runs_out(monkeypatch):
    """앞 그룹 성공 → 다음 그룹에서 예산 소진 → 남은 그룹은 부르지 않는다. 성공 그룹의 결과는 기존 부분 실패 정책대로 남는다."""
    client = ScriptedClient(["normal", "slow"])
    reranker = GeminiListwiseReranker(_cfg(), client=client, configured=True)
    protected, others = _protected(), _others()
    rec = ExplainRecorder("q")
    seen = _spy_statuses(monkeypatch)

    results, elapsed = _search(_router(others, lyrics_hits=protected, reranker=reranker), _lyric_analysis(),
                               top_k=8, recorder=rec)

    assert elapsed < BUDGET + TOLERANCE
    assert [c.outcome for c in client.calls] == ["returned", "cancelled"]
    assert [s for _, s in seen] == [RERANK_APPLIED, RERANK_FAILED, RERANK_BUDGET_EXHAUSTED]
    ids = [t.id for t in results]
    # 성공한 tier: 가짜는 역순(p3,p2,p1)을 냈고 실물 규칙(검색 1위 exact 보호)이 p1을 1위로 되돌린다 → p1,p3,p2
    assert ids[:3] == ["p1", "p3", "p2"], "앞 그룹의 모델 순서는 보존된다"
    assert ids[3:] == ["p4", "p5", "o1", "o2", "o3"], "실패·미실행 그룹은 입력 순서"
    assert len(ids) == len(set(ids)) == 8
    # 기존 부분 실패 정책과 같다 — 한 번이라도 적용됐으면 단계는 model_applied, 호출 수에 흔적(3/1)
    assert rec.record.reorder_stage == REORDER_MODEL_APPLIED
    assert len(rec.record.rerank_runs) == 3 and rec.record.rerank_runs_applied == 1
    assert rerank_outcome(list(rec.record.rerank_runs), "") == OUTCOME_PARTIAL_FAILURE


def test_retry_wait_is_clamped_to_remaining_budget():
    """예외 재시도 사이 대기(1초)가 남은 예산(0.5초)을 넘지 않는다. 수정 전에는 예산 0.5초가 약 1.0초로 늘었다."""
    client = ScriptedClient(["exception"])
    reranker = GeminiListwiseReranker(_cfg(time_budget_seconds=0.5), client=client, configured=True)
    tracks = _others(n=5)
    started = time.monotonic()
    run = reranker.rerank_run("q", tracks, top_k=5)
    elapsed = time.monotonic() - started
    assert run.status == RERANK_FAILED
    assert elapsed < 0.5 + 0.3, f"재시도 대기가 예산을 넘겼다: {elapsed:.2f}s"
    assert [c.outcome for c in client.calls] == ["raised"], "둘째 시도는 예산 밖이라 시작하지 않는다"


def test_retries_still_happen_inside_a_sufficient_budget():
    """대조: 예산이 충분하면 재시도 3회가 그대로 돈다(기존 동작 유지)."""
    client = ScriptedClient(["exception"])
    reranker = GeminiListwiseReranker(_cfg(time_budget_seconds=5.0), client=client, configured=True)
    run = reranker.rerank_run("q", _others(n=5), top_k=5)
    assert run.status == RERANK_FAILED
    assert [c.outcome for c in client.calls] == ["raised"] * 3


# --- 리랭커 단독: 외부 마감의 의미 -----------------------------------------------------------------

class _ClockModels:
    """동기 클라이언트(aio 없음) — 응답마다 가짜 시계를 `advance`초 돌린다. 예산 뒤 도착 시험용."""

    def __init__(self, clock, advance):
        self.clock, self.advance, self.calls = clock, advance, 0

    def generate_content(self, model, contents, config=None):
        self.calls += 1
        self.clock["now"] += self.advance
        ids = candidate_ids_from_prompt(contents)
        payload = {"query_confidence": 0.9,
                   "ranking": [{"id": i, "relevance": 1.0 - 0.1 * k} for k, i in enumerate(reversed(ids))]}
        return type("Response", (), {"text": json.dumps(payload)})()


def _clock_reranker(monkeypatch, advance, **cfg):
    import src.retrieval.gemini_listwise_reranker as mod

    clock = {"now": 100.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(mod.time, "sleep", lambda s: clock.__setitem__("now", clock["now"] + s))
    models = _ClockModels(clock, advance)
    client = type("Client", (), {"models": models})()
    return GeminiListwiseReranker(_cfg(**cfg), client=client, configured=True), clock, models


def test_response_after_shared_deadline_is_not_applied(monkeypatch):
    """외부 마감(남은 5초) 뒤에 온 응답은 순위에 반영하지 않는다 — 입력 순서 + failed."""
    reranker, clock, models = _clock_reranker(monkeypatch, advance=6.0, time_budget_seconds=30.0, max_retries=1)
    tracks = _others(n=3)
    run = reranker.rerank_run("q", tracks, top_k=3, deadline=clock["now"] + 5.0)
    assert run.status == RERANK_FAILED and models.calls == 1
    assert [t.id for t in run.tracks] == ["o1", "o2", "o3"]


def test_response_inside_shared_deadline_is_applied(monkeypatch):
    reranker, clock, models = _clock_reranker(monkeypatch, advance=3.0, time_budget_seconds=30.0)
    tracks = _others(n=3)
    run = reranker.rerank_run("q", tracks, top_k=3, deadline=clock["now"] + 5.0)
    assert run.status == RERANK_APPLIED and models.calls == 1
    assert [t.id for t in run.tracks] == ["o3", "o2", "o1"]


def test_expired_shared_deadline_makes_no_call(monkeypatch):
    reranker, clock, models = _clock_reranker(monkeypatch, advance=0.0)
    run = reranker.rerank_run("q", _others(n=3), top_k=3, deadline=clock["now"] - 0.001)
    assert run.status == RERANK_FAILED and models.calls == 0


def test_standalone_call_without_deadline_uses_its_own_budget(monkeypatch):
    """마감을 주지 않으면(평가 도구·measure_rerank_batch·기존 테스트) 설정 예산(30초)을 지금부터 센다 — 외부 마감 5초와 달리 적용된다."""
    reranker, clock, models = _clock_reranker(monkeypatch, advance=6.0, time_budget_seconds=30.0)
    run = reranker.rerank_run("q", _others(n=3), top_k=3)
    assert run.status == RERANK_APPLIED and models.calls == 1
    assert reranker.rerank_budget_seconds == 30.0
    assert sr.rerank_stage_deadline(reranker) == pytest.approx(clock["now"] + 30.0)


def test_pre_pass_reserve_shrinks_with_remaining_budget():
    """앞 단계(검증) 몫은 설정 예산이 아니라 **남은** 예산의 절반까지만 — 공유 마감으로 4초 남았으면 2초."""
    reranker = GeminiListwiseReranker(_cfg(time_budget_seconds=30.0, pass_reserve_seconds=10.0,
                                           request_timeout_seconds=20.0), client=object(), configured=True)
    assert reranker._pass_reserve_seconds() == 10.0
    assert reranker._pass_reserve_seconds(4.0) == 2.0
    assert reranker._pass_reserve_seconds(0.0) == 0.0


# --- 마감이 요청 사이·동시 요청 사이에 새지 않는다 -------------------------------------------------------

def test_exhausted_request_does_not_leak_into_next_request(monkeypatch):
    client = ScriptedClient(["slow"])
    reranker = GeminiListwiseReranker(_cfg(), client=client, configured=True)
    first, _ = _search(_router(_others(), lyrics_hits=_protected(), reranker=reranker), _lyric_analysis(), top_k=8)
    assert [t.id for t in first] == ["p1", "p2", "p3", "p4", "p5", "o1", "o2", "o3"]

    client.script = ["normal"]
    rec = ExplainRecorder("q")
    second, elapsed = _search(_router(_others(), lyrics_hits=_protected(), reranker=reranker), _lyric_analysis(),
                              top_k=8, recorder=rec)
    assert rec.record.rerank_runs == [RERANK_APPLIED] * 3, "앞 요청의 마감이 남아 있으면 여기서 호출이 생략된다"
    assert [t.id for t in second] == ["p1", "p3", "p2", "p4", "p5", "o3", "o2", "o1"]
    assert elapsed < TOLERANCE


def test_concurrent_requests_keep_separate_deadlines():
    """같은 singleton 리랭커로 요청 A(무응답, 예산 소진)와 요청 B(느리지만 예산 안)를 겹쳐 돌린다.

    B는 A의 마감(t≈1.0)이 지난 뒤(t≈1.4)에 첫 응답을 받는다. 마감이 리랭커에 저장됐다면 B의 호출이 A의 마감에서
    끊겨 failed가 된다. 요청별 지역 마감이면 B는 세 그룹 모두 applied다.
    """
    client = PrefixScriptedClient({"7": ["slow"], "8": [("slow_within", 0.5), "normal", "normal"]})
    reranker = GeminiListwiseReranker(_cfg(), client=client, configured=True)
    outcome: dict = {}

    def run(label, prefix_p, prefix_o, delay):
        time.sleep(delay)
        rec = ExplainRecorder(label)
        router = _router(_others(prefix_o), lyrics_hits=_protected(prefix_p), reranker=reranker)
        results, elapsed = _search(router, _lyric_analysis(), top_k=8, recorder=rec)
        outcome[label] = (list(rec.record.rerank_runs), [t.id for t in results], elapsed)

    a = threading.Thread(target=run, args=("A", "7", "70", 0.0))
    b = threading.Thread(target=run, args=("B", "8", "80", 0.9))
    a.start(); b.start(); a.join(10); b.join(10)

    a_runs, a_ids, a_elapsed = outcome["A"]
    b_runs, b_ids, b_elapsed = outcome["B"]
    assert a_runs == [RERANK_FAILED, RERANK_BUDGET_EXHAUSTED, RERANK_BUDGET_EXHAUSTED] and a_elapsed < BUDGET + TOLERANCE
    assert a_ids == ["71", "72", "73", "74", "75", "701", "702", "703"]
    assert b_runs == [RERANK_APPLIED] * 3, f"B의 마감이 A에 끌려갔다: {b_runs}"
    # 두 보호 tier 모두 가짜 역순 뒤 검색 1위 exact 보호가 첫 곡을 되돌린다(81,83,82 · 84,85) + 일반 역순
    assert b_ids == ["81", "83", "82", "84", "85", "803", "802", "801"]
    outcomes = {c.candidate_ids[0][0]: [] for c in client.calls}
    for c in client.calls:
        outcomes[c.candidate_ids[0][0]].append(c.outcome)
    assert outcomes == {"7": ["cancelled"], "8": ["returned", "returned", "returned"]}


# --- 기존 동작 유지: 재질문 답·거절·정책 생략·단독 호출 호환 ----------------------------------------------

def test_answers_and_rejections_still_reach_the_model_with_shared_deadline(monkeypatch):
    """2턴(답변 + 거절)에서도 답이 프롬프트에 들어가고 거절 곡은 후보에서 빠진다. 마감은 지역값이라 결과에 영향이 없다."""
    client = ScriptedClient(["normal"])
    reranker = GeminiListwiseReranker(_cfg(), client=client, configured=True)
    tracks = _others(n=5)
    rec = ExplainRecorder("q")
    prompts: list = []
    orig = client._generate_async

    async def capture(contents):
        prompts.append(contents)
        return await orig(contents)

    monkeypatch.setattr(client, "_generate_async", capture)
    results, _ = _search(_router(tracks, reranker=reranker), _analysis(), top_k=4, recorder=rec,
                         answers=[ClarifyAnswer(slot="vocal_gender", value="여성")], exclude_ids=["o1"])
    assert "o1" not in [t.id for t in results]
    assert rec.record.rerank_runs == [RERANK_APPLIED]
    assert prompts and "User corrections" in prompts[0] and "여성" in prompts[0]


def test_policy_skip_makes_no_call_and_no_deadline_record():
    client = ScriptedClient(["normal"])
    reranker = GeminiListwiseReranker(_cfg(), client=client, configured=True)
    rec = ExplainRecorder("q")
    analysis = _analysis(image_english_query="red album cover with a rose", has_visual_clue=True,
                         modality_weights={"text": 0.1, "image": 0.9, "audio": 0.0})
    _search(_router(_others(n=4), reranker=reranker), analysis, top_k=4, recorder=rec)
    assert client.calls == [] and rec.record.rerank_runs == []
    assert rec.record.rerank_policy_skip == "image_dominant"


def test_backend_without_deadline_support_is_called_as_before():
    """마감을 받지 않는 백엔드(스텁·Cross-Encoder)에는 deadline 인자를 넘기지 않고, 라우터도 마감을 만들지 않는다."""
    stub = _StubReranker()
    assert sr.rerank_stage_deadline(stub) is None
    out = sr.call_reranker(stub, "q", _others(n=3), 3, deadline=None)
    assert stub.calls == 1 and [t.id for t in out] == ["o3", "o2", "o1"]
    # 마감이 이미 지났으면 어떤 백엔드든 부르지 않는다(상태 budget_exhausted)
    runs: list = []
    out = sr.call_reranker(stub, "q", _others(n=3), 3, runs_out=runs, deadline=time.monotonic() - 1.0)
    assert stub.calls == 1 and runs == [RERANK_BUDGET_EXHAUSTED] and [t.id for t in out] == ["o1", "o2", "o3"]


def test_reorder_stage_treats_budget_exhausted_like_failure():
    rec = SearchExplain(query="q")
    rec.reorder_attempted = True
    rec.reorder_committed = True
    rec.rerank_runs = [RERANK_BUDGET_EXHAUSTED]
    assert rec.reorder_stage == REORDER_MODEL_FAILED
    assert rerank_outcome([RERANK_BUDGET_EXHAUSTED], "") == OUTCOME_FALLBACK
    assert rerank_outcome([RERANK_FAILED, RERANK_BUDGET_EXHAUSTED], "") == OUTCOME_FALLBACK
    assert rerank_outcome([RERANK_APPLIED, RERANK_BUDGET_EXHAUSTED], "") == OUTCOME_PARTIAL_FAILURE


# --- 평가기 사본(`rerank_preserving_exact_lyrics`)도 같은 예산 공유 ---------------------------------------

def test_evaluator_protected_groups_share_the_budget():
    client = ScriptedClient(["normal", "slow"])
    reranker = GeminiListwiseReranker(_cfg(), client=client, configured=True)
    candidates = [*_protected(), *_others()]
    statuses: list = []
    rec = ExplainRecorder("q")
    started = time.monotonic()
    final = rerank_preserving_exact_lyrics(reranker, _lyric_analysis(), candidates, 8, recorder=rec, statuses=statuses)
    elapsed = time.monotonic() - started
    assert elapsed < BUDGET + TOLERANCE
    assert statuses == [RERANK_APPLIED, RERANK_FAILED, RERANK_BUDGET_EXHAUSTED]
    assert [c.outcome for c in client.calls] == ["returned", "cancelled"]
    assert [t.id for t in final] == ["p1", "p3", "p2", "p4", "p5", "o1", "o2", "o3"]
    assert rec.record.reorder_stage == REORDER_MODEL_APPLIED and rec.record.rerank_runs_applied == 1
