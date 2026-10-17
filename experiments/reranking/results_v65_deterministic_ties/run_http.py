"""v65 §4-⑥ 실제 HTTP — dev 176을 실제 `POST /api/v1/search`(v65_app, 실제 FastAPI 앱)로 순차 전송한다(모델 호출 없음).

본문은 화면과 같다(top_k 15 · candidate_k 30 · explain · defer_clarify). 가짜 SDK는 `identity`(입력 순서 그대로의 정상 응답 —
평가기 offline 가짜 응답과 같은 규칙). 요청마다 새로 생긴 가짜 호출 기록(프롬프트 sha16·후보 id)과 가사 호출 기록을 붙인다.
원문 응답은 남기지 않고 필요한 필드만 `http/<run>.jsonl`(Git 제외)에 쓴다.

    venv/bin/python run_http.py --run after_server_1 --port 8067
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
BODY = {"top_k": 15, "candidate_k": 30, "explain": True, "defer_clarify": True}
CACHES = {s: ROOT / f"experiments/reranking/results_v49_vertex_baseline/analysis_cache_{s}_dev_vertex.json" for s in ("v06", "v09", "v10")}


def fake_rows(log: Path, since: int) -> List[Dict[str, Any]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()[since:]]


def main() -> int:
    from src.retrieval.analysis_cache import load_cache

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    log = HERE / "logs" / f"http_{args.run}" / "fake_calls.jsonl"
    out = HERE / "http" / f"{args.run}.jsonl"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} 이미 있다")
    for _ in range(600):
        try:
            if (requests.get(f"{base}/health", timeout=5).json().get("warmup") or {}).get("state") == "ready":
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    status_before = requests.get(f"{base}/v65/status", timeout=10).json()
    plan = []
    for set_name, path in CACHES.items():
        cache = load_cache(path)
        for qid in sorted(cache.entries):
            plan.append((set_name, qid, cache.entries[qid]["query"]))
    t0 = time.time()
    with out.open("w", encoding="utf-8") as fh:
        for set_name, qid, query in plan:
            f0 = len(fake_rows(log, 0))
            l0 = requests.get(f"{base}/v65/lyrics_calls", params={"since": 10 ** 9}, timeout=10).json()["total"]
            started = time.perf_counter()
            resp = requests.post(f"{base}/api/v1/search", json=dict(BODY, query=query), timeout=600)
            ms = round((time.perf_counter() - started) * 1000, 1)
            row: Dict[str, Any] = {"set": set_name, "query_id": qid, "status": resp.status_code, "client_ms": ms}
            if resp.status_code == 200:
                data = resp.json()
                ex = data.get("explain") or {}
                row.update({
                    "candidate_ids": data.get("candidate_ids") or [], "result_ids": [r["id"] for r in data.get("results") or []],
                    "lyric_types": [r.get("lyric_match_type") for r in data.get("results") or []],
                    "context_evidence": [bool(r.get("context_evidence")) for r in data.get("results") or []],
                    "reorder_stage": ex.get("reorder_stage", ""), "rerank_calls": ex.get("rerank_calls"),
                    "rerank_model_calls": ex.get("rerank_model_calls"), "rerank_skip_reason": ex.get("rerank_skip_reason", ""),
                    "failed_paths": [p.get("path") for p in ex.get("failed_paths") or []],
                })
            else:
                row["error_body"] = resp.text[:300]
            calls = fake_rows(log, f0)
            row["fake_calls"] = [{"sha": c.get("contents_sha16", ""), "n": c["n_candidates"], "ids": c["candidate_ids"],
                                  "outcome": c["outcome"]} for c in calls]
            row["lyrics_calls"] = requests.get(f"{base}/v65/lyrics_calls", params={"since": l0}, timeout=10).json()["calls"]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
    status_after = requests.get(f"{base}/v65/status", timeout=10).json()
    (HERE / "http" / f"{args.run}_status.json").write_text(
        json.dumps({"before": status_before, "after": status_after, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("done", args.run, len(plan), f"{time.time() - t0:.0f}s", "net", len(status_after["network_attempts"]),
          "blocked", status_after["blocked_client_calls"], "code", status_after.get("code"), status_after.get("backend_file"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
