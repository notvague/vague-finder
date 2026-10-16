"""v59 재현 — 같은 가짜 SDK 응답 대본을 **수정 전·후 코드**에 흘려 리랭킹 상태·단계·횟수·반환 순서를 비교한다.

실제 `GeminiListwiseReranker` + 실제 `SearchRouter`(+ 부분 사례는 실제 `/search` 라우트, TestClient)에 가짜 SDK 클라이언트만
꽂는다. 모델·네트워크·Mongo·Qdrant 호출 없음. 코드 판에 없는 필드는 "n/a"로 적어, 수정 전 코드에도 그대로 돈다.

    venv/bin/python experiments/reranking/results_v59_rerank_status/repro_rerank_status.py --label after --out <json>
    (수정 전은 main 466fbe8을 받은 임시 워크트리에서 같은 파일을 복사해 --label before 로)

예산은 운영 30초의 1/30(1초)·요청 제한 0.6초로 구조만 같게 줄였다(v57 시험과 같은 값). 응답 순서 자체는 가짜가 정한 역순이라
"어느 곡이 정답인가"가 아니라 "묶음별로 모델 순서가 반영됐나·상태가 무엇으로 기록되나"만 본다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.backend.api.dependencies import get_query_analyzer, get_search_router  # noqa: E402
from src.backend.api.routes.search import router as search_route  # noqa: E402
from src.backend.schemas.explain import SearchExplainOut  # noqa: E402
from src.backend.schemas.query import ModalityWeights  # noqa: E402
from src.retrieval import explain as ex  # noqa: E402
from src.retrieval.evaluate_search_accuracy import rerank_outcome  # noqa: E402
from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker, GeminiListwiseRerankerConfig  # noqa: E402
from tests.fake_gemini_client import FOREIGN_ID_PREFIX, FakeGeminiClient, _Response, normal_ranking  # noqa: E402
from tests.test_rerank_shared_budget import ScriptedClient, _lyric_analysis, _others, _protected  # noqa: E402
from tests.test_search_explain import _StubReranker, _analysis, _router  # noqa: E402
from tests.test_user_explanation_route import _CountingAnalyzer  # noqa: E402

BUDGET = 1.0
REQ_TIMEOUT = 0.6


class Client(ScriptedClient):
    """v59 사례용 응답을 이 파일 안에서 만든다 — 수정 전 코드의 가짜 클라이언트에는 없는 모드라서."""

    def _respond(self, call, mode):
        ids = call.candidate_ids
        if mode == "empty_ranking":
            return _Response(json.dumps({"query_confidence": 0.9, "ranking": []}))
        if mode == "wrong_types":
            return _Response(json.dumps([{"id": i} for i in ids]))
        if mode == "partial_ids":
            half = ids[: max(1, len(ids) // 2)]
            items = [*reversed(half), half[0], f"{FOREIGN_ID_PREFIX}9999"]
            return _Response(json.dumps({"ranking": [{"id": i, "relevance": 0.5} for i in items]}))
        return super()._respond(call, mode)


def cfg(**over):
    base = dict(enabled=True, max_retries=3, passes=1, request_timeout_seconds=REQ_TIMEOUT, time_budget_seconds=5.0,
                pass_reserve_seconds=0.2, use_search_grounding=False, rare_fact_verification=False)
    base.update(over)
    return GeminiListwiseRerankerConfig(**base)


def reranker(client, **over):
    return GeminiListwiseReranker(cfg(**over), client=client, configured=True)


def search(router, analysis, **kw):
    rec = ex.ExplainRecorder("q")
    started = time.monotonic()
    try:
        results = asyncio.run(router.search(analysis, top_k=8, recorder=rec, **kw))
    finally:
        router.shutdown()
    return results, rec.record, round(time.monotonic() - started, 3)


def summarize(record, results, client, elapsed):
    out = SearchExplainOut.of(record).model_dump()
    row = {
        "elapsed_s": elapsed,
        "statuses": list(record.rerank_runs),
        "reorder_stage": record.reorder_stage,
        "stage_label": ex.REORDER_STAGE_LABELS.get(record.reorder_stage, record.reorder_stage),
        "stage_note": out.get("reorder_stage_note", "n/a"),
        "evaluator_outcome": rerank_outcome(list(record.rerank_runs), record.rerank_error, record.rerank_policy_skip),
        "rerank_calls": out["rerank_calls"],
        "rerank_calls_applied": out["rerank_calls_applied"],
        "groups_failed": out.get("rerank_groups_failed", "n/a"),
        "groups_budget_exhausted": out.get("rerank_groups_budget_exhausted", "n/a"),
        "model_calls_reported": out.get("rerank_model_calls", "n/a"),
        "fake_sdk_calls": None if client is None else len(client.calls),
        "fake_sdk_outcomes": None if client is None else [c.outcome for c in client.calls],
        "rerank_error": record.rerank_error,
        "policy_skip": record.rerank_policy_skip,
        "top_ids": [t.id for t in results],
        "songs": {},
    }
    for sid, song in sorted(record.songs.items()):
        if sid not in row["top_ids"]:
            continue
        row["songs"][sid] = {
            "reorder_applied": song.reorder_applied,
            "group_status": getattr(song, "rerank_group_status", "n/a"),
            "unreturned": getattr(song, "rerank_unreturned", "n/a"),
            "summary": ex.render_ko(song, reorder_stage=record.reorder_stage),
        }
    return row


def run_cases():
    cases = {}

    def put(name, client, rr, analysis, protected, others, **kw):
        results, record, elapsed = search(_router(others, lyrics_hits=protected, reranker=rr), analysis, **kw)
        cases[name] = summarize(record, results, client, elapsed)

    # 그룹 3개(보호 tier 2 + 일반) — n007형
    c = Client(["normal"]); put("all_normal", c, reranker(c), _lyric_analysis(), _protected(), _others())
    c = FakeGeminiClient(mode="fail_group_first", sleep_seconds=0.05)
    put("fail_first_group", c, reranker(c, max_retries=1), _lyric_analysis(), _protected(second_tier=False), _others())
    c = Client(["normal", "exception"]); put("first_ok_then_fail", c, reranker(c, max_retries=1), _lyric_analysis(), _protected(second_tier=False), _others())
    c = Client(["normal", "slow"]); put("first_ok_then_budget", c, reranker(c, time_budget_seconds=BUDGET), _lyric_analysis(), _protected(), _others())
    c = Client(["slow"]); put("all_slow", c, reranker(c, time_budget_seconds=BUDGET), _lyric_analysis(), _protected(), _others())
    c = Client(["exception"]); put("all_exception", c, reranker(c, max_retries=1), _lyric_analysis(), _protected(second_tier=False), _others())
    # 라우터 예외(앞 그룹 성공 뒤) — 스텁 리랭커
    stub = _StubReranker(fail_after=2)
    put("router_exception_after_success", None, stub, _lyric_analysis(), _protected(second_tier=False)[:2], _others())
    # 단일 그룹(일반 후보 5곡) — 응답 내용별
    for name, mode in [("foreign_ids_only", "foreign_ids"), ("empty_ranking", "empty_ranking"), ("wrong_types", "wrong_types"),
                       ("identity_order", "identity"), ("partial_duplicate_foreign_mix", "partial_ids")]:
        c = Client([mode]); put(name, c, reranker(c, max_retries=3), _analysis(), [], _others(n=6))
    c = Client(["foreign_ids", "normal"]); put("invalid_then_normal", c, reranker(c), _analysis(), [], _others(n=6))
    c = Client(["foreign_ids"]); put("invalid_persist_shared_deadline", c, reranker(c, time_budget_seconds=BUDGET), _analysis(), [], _others(n=6))
    # 정책 생략·비활성·미보고
    c = Client(["normal"])
    cover = _analysis(original_query="초록 바탕에 흰 도형이 있는 앨범 커버", has_visual_clue=True, image_english_query="green cover",
                      modality_weights=ModalityWeights(text=0.1, image=0.9, audio=0.0))
    put("policy_skip_image_dominant", c, reranker(c), cover, [], _others(n=4))
    c = Client(["normal"]); put("rerank_disabled", c, reranker(c), _analysis(), [], _others(n=4), use_rerank=False)
    from tests.test_search_explain import _LegacyReranker
    put("unknown_status_backend", None, _LegacyReranker(), _analysis(), [], _others(n=4))
    return cases


def run_http():
    """부분 사례를 실제 라우트로 — API JSON이 서비스 기록과 같은 뜻인지."""
    client = Client(["normal", "slow"])
    rr = reranker(client, time_budget_seconds=BUDGET)
    analyzer = _CountingAnalyzer(vocal_gender="남성", lyric_clues=[{"kind": "verbatim", "text": "한 소절"}])
    router = _router(_others(), lyrics_hits=_protected(), reranker=rr)
    app = FastAPI()
    app.include_router(search_route)
    app.dependency_overrides[get_query_analyzer] = lambda: analyzer
    app.dependency_overrides[get_search_router] = lambda: router
    try:
        with TestClient(app) as tc:
            started = time.monotonic()
            res = tc.post("/search", json={"query": "한 소절 노래", "top_k": 8, "explain": True})
            elapsed = round(time.monotonic() - started, 3)
    finally:
        router.shutdown()
    body = res.json()
    explain = body.get("explain") or {}
    keep = ["reorder_stage", "reorder_stage_label", "reorder_stage_note", "rerank_calls", "rerank_calls_applied",
            "rerank_groups_failed", "rerank_groups_budget_exhausted", "rerank_model_calls", "rerank_error"]
    return {
        "status_code": res.status_code,
        "elapsed_s": elapsed,
        "explain": {k: explain.get(k, "n/a") for k in keep},
        "ids": [t["id"] for t in body.get("results", [])],
        "per_song": {t["id"]: {k: (t["explain"] or {}).get(k, "n/a") for k in ("reorder_applied", "rerank_group_status", "summary")}
                     for t in body.get("results", [])},
        "fake_sdk_calls": [c.outcome for c in client.calls],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    payload = {"label": args.label, "root": str(ROOT), "cases": run_cases(), "http_partial": run_http()}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, row in payload["cases"].items():
        print(f"{name:34s} stage={row['reorder_stage']:15s} outcome={row['evaluator_outcome']:16s} "
              f"calls={row['rerank_calls']}/{row['rerank_calls_applied']} failed={row['groups_failed']} "
              f"exhausted={row['groups_budget_exhausted']} model_calls={row['model_calls_reported']} "
              f"sdk={row['fake_sdk_calls']} {row['elapsed_s']}s")
    print("http:", json.dumps(payload["http_partial"]["explain"], ensure_ascii=False))


if __name__ == "__main__":
    main()
