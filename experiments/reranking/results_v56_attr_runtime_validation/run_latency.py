"""§5 지연 비교 — 실제 HTTP 로 A(8056)·D(8057) 서버의 `/api/v1/search` 를 부른다(서비스가 쓰는 검색 경로 그대로).

규칙(측정 전에 고정, `latency_sample.json` 선정 뒤 바꾸지 않는다):
  - 요청 본문은 화면(map.js)과 같다: top_k 15 · candidate_k 30 · explain true · defer_clarify true. 리랭킹만 끈다(use_rerank false).
    합성 open_empty 는 prior_analysis(시기 창만 바꾼 저장 분석)를 함께 보낸다 — 2턴 재검색과 같은 정규 입력이다.
  - 두 서버 모두 /health 의 warmup.state == ready 를 기다린 뒤 시작한다.
  - 준비 요청: 각 서버에 표본 첫 질의(closed[0]) 1건 → `warm=1` 로 기록하고 집계에서 뺀다. 그 밖의 요청은 느려도 전부 포함한다.
  - 3회차 × (표본 순서대로 질의마다 A→D 또는 D→A 번갈아; 회차마다 먼저 가는 쪽을 바꾼다). 동시 요청 없음(순차).
  - 서버 구간 기록은 SEARCH_TIMING_LOG(JSONL)의 request_id 로 맞춘다(응답 헤더 X-Request-Id).
  - 같은 프로세스에서 반복한다(서버는 회차 사이에 재시작하지 않음). 캐시 공유: A·D는 서로 다른 프로세스, Qdrant 서버·Mongo 는 공유.

실행: venv/bin/python experiments/reranking/results_v56_attr_runtime_validation/run_latency.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

HERE = Path(__file__).resolve().parent
SERVERS = {"A": "http://127.0.0.1:8056", "D": "http://127.0.0.1:8057"}
ROUNDS = 3
BODY = {"top_k": 15, "candidate_k": 30, "explain": True, "defer_clarify": True, "use_rerank": False}
TIMEOUT = 600


def wait_ready(base: str, limit_s: float = 1800) -> Dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            health = requests.get(f"{base}/health", timeout=10).json()
            if health.get("warmup", {}).get("state") == "ready":
                return health
        except requests.RequestException:
            pass
        time.sleep(5)
    raise SystemExit(f"{base} warmup not ready in {limit_s}s")


def digest(ids: List[str]) -> str:
    return hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:12]


def one_request(arm: str, item: Dict[str, Any], category: str, rnd: int, order: int, warm: int) -> Dict[str, Any]:
    body = dict(BODY, query=item["query"])
    if item.get("prior_analysis"):
        body["prior_analysis"] = item["prior_analysis"]
    started = time.perf_counter()
    row: Dict[str, Any] = {
        "query_id": item["query_id"], "set": item["set"], "category": category, "round": rnd, "arm": arm,
        "order_in_pair": order, "warm": warm, "sent_at": time.strftime("%H:%M:%S"),
    }
    try:
        resp = requests.post(f"{SERVERS[arm]}/api/v1/search", json=body, timeout=TIMEOUT)
        row["client_ms"] = round((time.perf_counter() - started) * 1000, 1)
        row["status"] = resp.status_code
        row["request_id"] = resp.headers.get("X-Request-Id", "")
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            cands = data.get("candidate_ids", [])
            row.update({
                "n_results": len(results), "n_candidates": len(cands), "candidate_digest": digest(cands),
                "top1": results[0]["id"] if results else "", "notice": (data.get("notice") or {}).get("code", ""),
                "reorder_stage": (data.get("explain") or {}).get("reorder_stage", ""),
                "attr_filter_genre": json.dumps((data.get("explain") or {}).get("attr_filter_genre"), ensure_ascii=False)
                if (data.get("explain") or {}).get("attr_filter_genre") else "",
                "error": "",
            })
        else:
            row.update({"n_results": "", "n_candidates": "", "candidate_digest": "", "top1": "", "notice": "",
                        "reorder_stage": "", "attr_filter_genre": "", "error": resp.text[:200]})
    except requests.RequestException as exc:
        row.update({"client_ms": round((time.perf_counter() - started) * 1000, 1), "status": "EXC", "request_id": "",
                    "n_results": "", "n_candidates": "", "candidate_digest": "", "top1": "", "notice": "",
                    "reorder_stage": "", "attr_filter_genre": "", "error": f"{type(exc).__name__}: {exc}"[:200]})
    return row


def main() -> int:
    sample = json.loads((HERE / "latency_sample.json").read_text(encoding="utf-8"))["sample"]
    plan: List[tuple] = []
    for category in ("closed", "open_hits", "open_empty", "open_empty_synthetic"):
        for item in sample.get(category, []):
            plan.append((category, item))
    health = {arm: wait_ready(base) for arm, base in SERVERS.items()}
    status = {arm: requests.get(f"{base}/v56/status", timeout=10).json() for arm, base in SERVERS.items()}
    meta = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rounds": ROUNDS, "body": BODY, "plan_size": len(plan),
            "health": health, "v56_status": status}
    (HERE / "latency_runmeta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    # 준비 요청(집계 제외)
    first_cat, first_item = plan[0]
    for arm in ("A", "D"):
        rows.append(one_request(arm, first_item, first_cat, 0, 0, warm=1))
        print("warm", arm, rows[-1]["client_ms"], rows[-1]["status"], flush=True)
    for rnd in range(1, ROUNDS + 1):
        arms = ("A", "D") if rnd % 2 == 1 else ("D", "A")
        for category, item in plan:
            for order, arm in enumerate(arms, start=1):
                rows.append(one_request(arm, item, category, rnd, order, warm=0))
            a = next(r for r in rows[-2:] if r["arm"] == "A"); d = next(r for r in rows[-2:] if r["arm"] == "D")
            print(f"r{rnd} {category:>20} {item['query_id']:>5} A={a['client_ms']:>8} D={d['client_ms']:>8} "
                  f"({a['status']}/{d['status']}) cand A={a.get('n_candidates')} D={d.get('n_candidates')}", flush=True)
    fields = list(rows[0].keys())
    with (HERE / "latency_requests.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["v56_status_after"] = {arm: requests.get(f"{base}/v56/status", timeout=10).json() for arm, base in SERVERS.items()}
    (HERE / "latency_runmeta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", len(rows), "requests ->", HERE / "latency_requests.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
