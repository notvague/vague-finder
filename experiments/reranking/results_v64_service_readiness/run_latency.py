"""§3 항목 3 지연 비교 — v56 `run_latency.py`와 같은 절차를 현재 코드의 A(8064)·D(8065)에 적용한다.

규칙(측정 전 고정, `latency_sample.json` 선정 뒤 바꾸지 않는다):
  - 본문은 화면과 같고 리랭킹만 끈다(use_rerank false). 합성 0곡은 prior_analysis(시기 창만 바꾼 저장 분석)를 함께 보낸다.
  - 두 서버 `/health` 의 warmup.state == ready 확인 뒤 시작. 준비 요청: 서버마다 표본 첫 질의(closed[0]) 1건(warm=1, 집계 제외).
  - 3회차 × 표본 순서대로 질의마다 A→D(홀수 회차) / D→A(짝수 회차). 동시 요청 없음. 느린 요청도 전부 포함한다.
  - 서버 구간은 SEARCH_TIMING_LOG의 request_id로 맞춘다(`aggregate_latency.py`).

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/run_latency.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

HERE = Path(__file__).resolve().parent
SERVERS = {"A": "http://127.0.0.1:8064", "D": "http://127.0.0.1:8065"}
ROUNDS = 3
BODY = {"top_k": 15, "candidate_k": 30, "explain": True, "defer_clarify": True, "use_rerank": False}
OUT = HERE / "latency_requests.csv"


def wait_ready(base: str, limit_s: float = 900) -> Dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            health = requests.get(f"{base}/health", timeout=10).json()
            if health.get("warmup", {}).get("state") == "ready":
                return health
        except requests.RequestException:
            pass
        time.sleep(2)
    raise SystemExit(f"{base} warmup not ready in {limit_s}s")


def digest(ids: List[str]) -> str:
    return hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:12]


def one_request(arm: str, item: Dict[str, Any], category: str, rnd: int, order: int, warm: int) -> Dict[str, Any]:
    body = dict(BODY, query=item["query"])
    if item.get("prior_analysis"):
        body["prior_analysis"] = item["prior_analysis"]
    row: Dict[str, Any] = {"query_id": item["query_id"], "set": item["set"], "category": category, "round": rnd, "arm": arm,
                           "order_in_pair": order, "warm": warm, "sent_at": time.strftime("%H:%M:%S")}
    started = time.perf_counter()
    try:
        resp = requests.post(f"{SERVERS[arm]}/api/v1/search", json=body, timeout=600)
        row["client_ms"] = round((time.perf_counter() - started) * 1000, 1)
        row["status"] = resp.status_code
        row["request_id"] = resp.headers.get("X-Request-Id", "")
        if resp.status_code == 200:
            data = resp.json()
            cands = data.get("candidate_ids", [])
            ex = data.get("explain") or {}
            row.update({"n_results": len(data.get("results", [])), "n_candidates": len(cands), "candidate_digest": digest(cands),
                        "reorder_stage": ex.get("reorder_stage", ""), "failed_paths": "|".join(p.get("path", "") for p in ex.get("failed_paths") or []),
                        "error": ""})
        else:
            row.update({"n_results": "", "n_candidates": "", "candidate_digest": "", "reorder_stage": "", "failed_paths": "",
                        "error": resp.text[:200]})
    except requests.RequestException as exc:
        row.update({"client_ms": round((time.perf_counter() - started) * 1000, 1), "status": "EXC", "request_id": "",
                    "n_results": "", "n_candidates": "", "candidate_digest": "", "reorder_stage": "", "failed_paths": "",
                    "error": type(exc).__name__})
    return row


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} 이미 있다 — 덮어쓰지 않는다")
    sample = json.loads((HERE / "latency_sample.json").read_text(encoding="utf-8"))["sample"]
    plan = [(c, item) for c in ("closed", "open_hits", "open_empty", "open_empty_synthetic") for item in sample.get(c, [])]
    health = {arm: wait_ready(base) for arm, base in SERVERS.items()}
    meta = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rounds": ROUNDS, "body": BODY, "plan_size": len(plan), "health": health,
            "status_before": {arm: requests.get(f"{base}/v64/status", timeout=10).json() for arm, base in SERVERS.items()}}
    rows: List[Dict[str, Any]] = []
    first_cat, first_item = plan[0]
    for arm in ("A", "D"):
        rows.append(one_request(arm, first_item, first_cat, 0, 0, warm=1))
    for rnd in range(1, ROUNDS + 1):
        arms = ("A", "D") if rnd % 2 == 1 else ("D", "A")
        for category, item in plan:
            for order, arm in enumerate(arms, start=1):
                rows.append(one_request(arm, item, category, rnd, order, warm=0))
        print(f"round {rnd} done", flush=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["status_after"] = {arm: requests.get(f"{base}/v64/status", timeout=10).json() for arm, base in SERVERS.items()}
    (HERE / "latency_runmeta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("done", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
