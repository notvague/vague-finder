"""§3 항목 4·5 — 가짜 SDK 장애 · 기존 손실 재현 · 2턴 · 상태 누출. v56 `run_fallback.py`를 현재 응답 필드(v57·v59)에 맞춰 옮겼다.

가짜는 리랭커가 아니라 그 아래 SDK 클라이언트다. 재시도·파서·공유 마감·그룹 상태·보호 배치·답변 가산·거절 제외는 운영 코드가 돈다.
모드는 서버가 읽는 파일(`ctl/fake_mode_<arm>.json`)로 요청마다 바꾼다. 각 요청 뒤 가짜 호출 기록의 새 줄을 붙인다.
사례·모드 순서는 RUN_INFO §3-1(측정 전 고정). t014·t049는 이미 열린 v55 test의 저장 분석 — **장애 계약 재현에만** 쓴다.

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/run_fallback.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_sweep import SERVERS, LOG, fake_lines, fake_tail, summarize  # noqa: E402

BODY = {"top_k": 15, "candidate_k": 30, "explain": True, "defer_clarify": True}
OUT = HERE / "fallback_requests.jsonl"
MODES_SINGLE = ["normal", "exception", "invalid_json", "foreign_ids", "empty_ranking", "wrong_types",
                "invalid_then_normal", "partial_ids", "slow", "slow_within", "normal"]
MODES_PROTECTED = ["normal", "exception", "fail_group_first", "invalid_then_normal", "slow", "normal"]
MODES_TURN2 = ["normal", "exception", "slow", "normal"]
CASES = [
    {"query_id": "t014", "kind": "repro_v55", "modes": MODES_SINGLE},
    {"query_id": "t049", "kind": "repro_v55", "modes": MODES_SINGLE},
    {"query_id": "n007", "kind": "lyric_protected", "modes": MODES_PROTECTED},
    {"query_id": "t018", "kind": "lyric_protected", "modes": MODES_PROTECTED},
    {"query_id": "q117", "kind": "policy_skip", "modes": ["exception", "normal"]},
    {"query_id": "q112", "kind": "turn2_gender", "modes": MODES_TURN2, "answers": [{"slot": "vocal_gender", "value": "남성"}]},
    {"query_id": "n011", "kind": "turn2_genre", "modes": MODES_TURN2, "answers": [{"slot": "genre", "value": "댄스"}]},
]


def load_queries() -> Dict[str, str]:
    import v64_runtime

    analyzer = v64_runtime.CachedAnalyzer()
    return {qid: query for query, (qid, *_rest) in analyzer.by_query.items()}


def set_mode(arm: str, mode: str, sleep: float = 8.0) -> None:
    (HERE / "ctl" / f"fake_mode_{arm}.json").write_text(json.dumps({"mode": mode, "sleep": sleep}), encoding="utf-8")
    time.sleep(0.05)


def post(arm: str, body: Dict[str, Any]):
    started = time.perf_counter()
    resp = requests.post(f"{SERVERS[arm]}/api/v1/search", json=body, timeout=600)
    return resp, round((time.perf_counter() - started) * 1000, 1)


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} 이미 있다 — 덮어쓰지 않는다")
    queries = load_queries()
    with OUT.open("w", encoding="utf-8") as handle:
        for arm in ("A", "D"):
            for case in CASES:
                qid = case["query_id"]
                prior, rejected, turn1 = None, [], None
                if case.get("answers"):
                    set_mode(arm, "normal")
                    before = fake_lines(arm)
                    resp, ms = post(arm, dict(BODY, query=queries[qid]))
                    turn1 = summarize(resp, ms)
                    data = resp.json()
                    prior = data["analysis"]
                    rejected = [r["id"] for r in data["results"][:1]]
                    handle.write(json.dumps({"arm": arm, "query_id": qid, "kind": case["kind"], "step": 0, "mode": "turn1_normal",
                                             **turn1, "fake_calls": len(fake_tail(arm, before)), "rejected_ids": rejected},
                                            ensure_ascii=False) + "\n")
                for step, mode in enumerate(case["modes"], start=1):
                    set_mode(arm, mode)
                    before = fake_lines(arm)
                    body = dict(BODY, query=queries[qid])
                    if prior is not None:
                        body.update({"prior_analysis": prior, "answers": case["answers"], "rejected_ids": rejected,
                                     "asked_slots": [a["slot"] for a in case["answers"]], "turn": 1})
                    resp, ms = post(arm, body)
                    row = summarize(resp, ms)
                    calls = fake_tail(arm, before)
                    row.update({"arm": arm, "query_id": qid, "kind": case["kind"], "step": step, "mode": mode,
                                "sent_at": time.strftime("%H:%M:%S"), "fake_calls": len(calls),
                                "fake_outcomes": [c["outcome"] for c in calls], "fake_sizes": [c["n_candidates"] for c in calls],
                                "fake_elapsed_s": [c["elapsed_s"] for c in calls], "rejected_ids": rejected,
                                "turn1_candidate_ids": (turn1 or {}).get("candidate_ids")})
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"{arm} {qid:>5} {mode:>20} {row['client_ms']:>9}ms {row['status']} stage={row.get('reorder_stage')} "
                          f"groups={row.get('rerank_calls')}/{row.get('rerank_calls_applied')} failed={row.get('rerank_groups_failed')} "
                          f"budget={row.get('rerank_groups_budget_exhausted')} model_calls={row.get('rerank_model_calls')} "
                          f"fake={len(calls)}{row['fake_outcomes']}", flush=True)
                set_mode(arm, "normal")
    status = {arm: requests.get(f"{base}/v64/status", timeout=10).json() for arm, base in SERVERS.items()}
    (HERE / "fallback_status_after.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
