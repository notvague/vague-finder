"""v57 재현 — 라우터의 리랭킹 단계가 그룹마다 예산을 새로 만드는지 확인한다(가짜 SDK 클라이언트, 모델 호출 없음).

예산을 운영값(30초)의 1/30로 줄여 같은 구조를 재현한다: 예산 1.0초 · 호출 제한 0.6초 · 재시도 3회.
  n007형: 보호 tier 2개(남성 2곡 · 여성 2곡) + 일반 3곡 → 호출 3회
  t018형: 보호 tier 1개(2곡) + 일반 → 호출 2회
실행: venv/bin/python <this> [label]  (워크트리 루트에서)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from src.retrieval import search_router as sr
from src.retrieval.explain import ExplainRecorder
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig
from tests.fake_gemini_client import FakeGeminiClient
from tests.test_search_explain import _analysis, _router, _track

BUDGET, REQ_TIMEOUT, RETRIES = 1.0, 0.6, 3


def cfg(**over):
    base = dict(enabled=True, max_retries=RETRIES, passes=1, request_timeout_seconds=REQ_TIMEOUT,
                time_budget_seconds=BUDGET, pass_reserve_seconds=0.2, use_search_grounding=False,
                rare_fact_verification=False)
    base.update(over)
    return GeminiListwiseRerankerConfig(**base)


class Scripted(FakeGeminiClient):
    def __init__(self, script, **kw):
        super().__init__(mode=script[0], **kw)
        self.script = list(script)

    def _current(self):
        return {"mode": self.script[min(len(self.calls), len(self.script) - 1)], "sleep": self._sleep}


def fixtures(two_tiers: bool):
    protected = [
        _track("p1", title="보호1", vocal_gender="남성", lyric_match_type="exact", lyric_match_score=0.95),
        _track("p2", title="보호2", vocal_gender="남성", lyric_match_type="exact", lyric_match_score=0.95),
    ]
    if two_tiers:
        protected += [
            _track("p3", title="보호3", vocal_gender="여성", lyric_match_type="exact", lyric_match_score=0.95),
            _track("p4", title="보호4", vocal_gender="여성", lyric_match_type="exact", lyric_match_score=0.95),
        ]
    others = [_track(f"o{i}", title=f"일반{i}") for i in range(1, 4)]
    analysis = _analysis(vocal_gender="남성", lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])
    return protected, others, analysis


def run_case(name, script, two_tiers, **cfg_over):
    client = Scripted(script, sleep_seconds=0.5)
    reranker = GeminiListwiseReranker(cfg(**cfg_over), client=client, configured=True)
    protected, others, analysis = fixtures(two_tiers)
    router = _router(others, lyrics_hits=protected, reranker=reranker)
    rec = ExplainRecorder("q")
    statuses = []
    orig = sr.call_reranker

    def spy(*a, **kw):
        out = orig(*a, **kw)
        runs = kw.get("runs_out") if "runs_out" in kw else (a[6] if len(a) > 6 else None)
        statuses.append(runs[-1] if runs else None)
        return out

    sr.call_reranker = spy
    started = time.monotonic()
    try:
        results = asyncio.run(router.search(analysis, top_k=8, recorder=rec))
    finally:
        sr.call_reranker = orig
        router.shutdown()
    elapsed = time.monotonic() - started
    row = {
        "case": name, "script": script, "elapsed_s": round(elapsed, 2), "budget_s": BUDGET,
        "fake_calls": len(client.calls), "fake_outcomes": [c.outcome for c in client.calls],
        "fake_elapsed_s": [c.elapsed_s for c in client.calls],
        "statuses": statuses, "reorder_stage": rec.record.reorder_stage,
        "rerank_calls": len(rec.record.rerank_runs), "applied": rec.record.rerank_runs_applied,
        "result_ids": [t.id for t in results],
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    rows = [
        run_case("n007형 전부 느림(3그룹)", ["slow"], True),
        run_case("t018형 전부 느림(2그룹)", ["slow"], False),
        run_case("n007형 1그룹 성공 뒤 느림", ["normal", "slow"], True),
        run_case("단일 호출 예외 재시도(예산 0.5)", ["exception"], False, time_budget_seconds=0.5),
        run_case("n007형 정상", ["normal"], True),
    ]
    out = f"repro_{label}.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=1)
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
