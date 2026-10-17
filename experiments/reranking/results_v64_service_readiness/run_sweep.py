"""§3 항목 1·1b·2 — dev 176 전체 + 합성 0곡 10건 + 캐시 밖 질의 1건을 A·D 서버의 실제 `POST /api/v1/search`로 보낸다.

본문은 화면과 같다(top_k 15 · candidate_k 30 · explain · defer_clarify). 리랭킹은 켜고 가짜 SDK는 `normal`(후보 역순) — 품질이 아니라
경로·상태·직렬화를 본다. 질의마다 A → D 순서, 동시 요청 없음. 각 요청 뒤 가짜 호출 기록과 가사 호출 기록(`/v64/lyrics_calls`)의
새 줄을 붙인다. 원문 응답은 남기지 않고 필요한 필드만 `sweep_requests.jsonl`(Git 제외)에 쓴다.

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/run_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
SERVERS = {"A": "http://127.0.0.1:8064", "D": "http://127.0.0.1:8065"}
BODY = {"top_k": 15, "candidate_k": 30, "explain": True, "defer_clarify": True}
LOG = HERE / "logs" / "main"
OUT = HERE / "sweep_requests.jsonl"
CACHES = {
    "v06": ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v06_dev_vertex.json",
    "v09": ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v09_dev_vertex.json",
    "v10": ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v10_dev_vertex.json",
}
CACHE_MISS_QUERY = "v64 캐시에 없는 질의 — 분석 캐시 누락 확인용"


def fake_lines(arm: str) -> int:
    path = LOG / f"fake_calls_{arm}.jsonl"
    return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0


def fake_tail(arm: str, since: int) -> List[Dict[str, Any]]:
    path = LOG / f"fake_calls_{arm}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[since:]]


def lyrics_total(arm: str) -> int:
    return int(requests.get(f"{SERVERS[arm]}/v64/lyrics_calls", params={"since": 10**9}, timeout=10).json()["total"])


def lyrics_tail(arm: str, since: int) -> List[Dict[str, Any]]:
    return requests.get(f"{SERVERS[arm]}/v64/lyrics_calls", params={"since": since}, timeout=10).json()["calls"]


def summarize(resp: requests.Response, ms: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {"status": resp.status_code, "client_ms": ms, "request_id": resp.headers.get("X-Request-Id", "")}
    if resp.status_code != 200:
        row["error_body"] = resp.text[:300]
        return row
    data = resp.json()
    results = data.get("results") or []
    ids = [r["id"] for r in results]
    cands = data.get("candidate_ids") or []
    ex = data.get("explain") or {}
    tracks = []
    for r in results:
        te = r.get("explain") or {}
        tracks.append({
            "id": r["id"], "lyric_match_type": r.get("lyric_match_type"), "has_lyric_detail": r.get("lyric_match_detail") is not None,
            "rerank_group_status": te.get("rerank_group_status", ""), "rerank_unreturned": te.get("rerank_unreturned"),
            "rules": [o.get("rule") for o in (te.get("order_rules") or []) if isinstance(o, dict)],
            "has_evidence": bool(te.get("evidence")), "has_context_evidence": r.get("context_evidence") is not None,
        })
    row.update({
        "n_results": len(ids), "n_unique": len(set(ids)), "results_in_candidates": all(i in cands for i in ids),
        "n_candidates": len(cands), "n_candidates_unique": len(set(cands)), "result_ids": ids, "candidate_ids": cands,
        "reorder_stage": ex.get("reorder_stage", ""), "reorder_stage_note": ex.get("reorder_stage_note", ""),
        "rerank_calls": ex.get("rerank_calls"), "rerank_calls_applied": ex.get("rerank_calls_applied"),
        "rerank_groups_failed": ex.get("rerank_groups_failed"), "rerank_groups_budget_exhausted": ex.get("rerank_groups_budget_exhausted"),
        "rerank_model_calls": ex.get("rerank_model_calls"), "rerank_skip_reason": ex.get("rerank_skip_reason", ""),
        "rerank_error": ex.get("rerank_error", ""), "failed_paths": [p.get("path") for p in ex.get("failed_paths") or []],
        "analysis_fallback": ex.get("analysis_fallback"), "modality_weights": ex.get("modality_weights"),
        "notice": (data.get("notice") or {}).get("code", ""), "turn": data.get("turn"), "clarify_deferred": data.get("clarify_deferred"),
        "total_found": data.get("total_found"), "tracks": tracks, "response_keys": sorted(data.keys()),
    })
    return row


def send(arm: str, body: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    f0, l0 = fake_lines(arm), lyrics_total(arm)
    started = time.perf_counter()
    try:
        resp = requests.post(f"{SERVERS[arm]}/api/v1/search", json=body, timeout=600)
        row = summarize(resp, round((time.perf_counter() - started) * 1000, 1))
    except requests.RequestException as exc:
        row = {"status": "EXC", "client_ms": round((time.perf_counter() - started) * 1000, 1), "error_body": type(exc).__name__}
    calls = fake_tail(arm, f0)
    lyr = lyrics_tail(arm, l0)
    row.update({"arm": arm, **meta, "sent_at": time.strftime("%H:%M:%S"),
                "fake_calls": len(calls), "fake_outcomes": [c["outcome"] for c in calls], "fake_sizes": [c["n_candidates"] for c in calls],
                "lyrics_calls": lyr})
    return row


def main() -> int:
    from src.retrieval.analysis_cache import load_cache

    plan = []
    for set_name, path in CACHES.items():
        cache = load_cache(path)
        for qid in sorted(cache.entries):
            plan.append(({"kind": "dev", "set": set_name, "query_id": qid}, {"query": cache.entries[qid]["query"]}))
    sample = json.loads((HERE / "latency_sample.json").read_text(encoding="utf-8"))["sample"]
    for item in sample["open_empty_synthetic"]:
        plan.append(({"kind": "synthetic_empty", "set": item["set"], "query_id": item["query_id"]},
                     {"query": item["query"], "prior_analysis": item["prior_analysis"]}))
    plan.append(({"kind": "cache_miss", "set": "-", "query_id": "miss1"}, {"query": CACHE_MISS_QUERY}))
    for arm in SERVERS:
        Path(HERE / "ctl" / f"fake_mode_{arm}.json").write_text(json.dumps({"mode": "normal"}), encoding="utf-8")
    if OUT.exists():
        raise SystemExit(f"{OUT} 이미 있다 — 덮어쓰지 않는다")
    t0 = time.time()
    with OUT.open("w", encoding="utf-8") as handle:
        for i, (meta, extra) in enumerate(plan, start=1):
            for arm in ("A", "D"):
                row = send(arm, dict(BODY, **extra), meta)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            if i % 20 == 0:
                print(f"{i}/{len(plan)} {time.time() - t0:.0f}s", flush=True)
    status = {arm: requests.get(f"{base}/v64/status", timeout=10).json() for arm, base in SERVERS.items()}
    (HERE / "sweep_status_after.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", len(plan) * 2, f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
