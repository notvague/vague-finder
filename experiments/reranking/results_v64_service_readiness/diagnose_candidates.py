"""§5-2 진단 — HTTP A 후보와 v63 offline B 후보가 다른 질의에서 무엇이 차이를 만드는가(모델·Mongo 호출 0).

같은 코드·분석·가사 스냅샷으로 라우터를 직접 부르며 한 번에 한 요인만 바꾼다.
  모드: local(파일 복사본 `artifacts/qdrant_v64_src`, 이 프로세스만 연다) / server(전용 Qdrant 127.0.0.1:6364)
  호출: eval(평가기처럼 top_k=30 · use_rerank=False · 반환 목록) / route(라우트처럼 top_k=15 · candidate_ids_out)
각 조합의 후보 30 순서를 HTTP A(`sweep_requests.jsonl`)·v63 offline B와 대조한다. 반복 2회로 같은 조합 안의 변동도 본다.

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/diagnose_candidates.py <local|server> [qid,...]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

MODE = sys.argv[1]
for key in ("QDRANT_PATH", "QDRANT_URL", "QDRANT_HYBRID_SCAN_LIMIT", "MONGO_URI", "GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ.pop(key, None)
os.environ.update({
    "ATTR_FILTER_WEIGHT": "0", "CONTEXT_WEIGHT": "0.5", "AUDIO_FUSION_SCALE": "1", "SEARCH_REFERENCE_YEAR": "2026",
    "RERANKER_BACKEND": "gemini_listwise", "GEMINI_RETRIEVAL_BACKEND": "vertex", "GCP_PROJECT_ID": "v64-blocked-project",
    "GCP_LOCATION": "global", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "FEEDBACK_COLLECTION_ENABLED": "0", "FEEDBACK_INDEX_WORKER_ENABLED": "0", "FEEDBACK_RETRIEVAL_ENABLED": "0",
})
if MODE == "local":
    os.environ["QDRANT_PATH"] = "artifacts/qdrant_v64_src"
else:
    os.environ["QDRANT_URL"] = "http://127.0.0.1:6364"

import v64_runtime  # noqa: E402

v64_runtime.install_guards()
v64_runtime.verify_guards()
v64_runtime.install_lyrics_snapshot()

import src.backend.api.dependencies as deps  # noqa: E402

deps.get_reranker = deps.singleton(v64_runtime.make_fake_reranker)  # type: ignore[assignment]
from analyze_sweep import load_v63  # noqa: E402


async def run(router, analysis, how: str):
    if how == "eval":
        tracks = await router.search(analysis, top_k=30, use_rerank=False, candidate_k=30)
        return [t.id for t in tracks]
    out: list = []
    await router.search(analysis, top_k=15, use_rerank=False, candidate_k=30, candidate_ids_out=out)
    return out


def main() -> int:
    v63 = load_v63()
    http = {}
    for line in (HERE / "sweep_requests.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["arm"] == "A" and r["kind"] == "dev" and r["status"] == 200:
            http[r["query_id"]] = r["candidate_ids"]
    wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(http)
    analyzer = v64_runtime.get_cached_analyzer()
    by_id = {qid: query for query, (qid, *_rest) in analyzer.by_query.items()}
    router = deps.get_search_router()
    rows = {}
    for qid in wanted:
        analysis = analyzer.analyze(by_id[qid])
        res = {f"{how}{rep}": asyncio.run(run(router, analysis, how)) for how in ("eval", "route") for rep in (1, 2)}
        v63_ids = [c["id"] for c in v63[qid]["candidates"]]
        rows[qid] = {k: {"eq_http": v == http[qid], "eq_v63": v == v63_ids} for k, v in res.items()}
        rows[qid]["eval1_eq_route1"] = res["eval1"] == res["route1"]
        rows[qid]["rep_stable"] = res["eval1"] == res["eval2"] and res["route1"] == res["route2"]
        rows[qid]["http_eq_v63"] = http[qid] == v63_ids
    out = HERE / f"diagnose_candidates_{MODE}.json"
    summary = {
        "mode": MODE, "queries": len(rows),
        "eval_eq_v63": sum(1 for r in rows.values() if r["eval1"]["eq_v63"]),
        "eval_eq_http": sum(1 for r in rows.values() if r["eval1"]["eq_http"]),
        "route_eq_http": sum(1 for r in rows.values() if r["route1"]["eq_http"]),
        "route_eq_v63": sum(1 for r in rows.values() if r["route1"]["eq_v63"]),
        "eval_eq_route": sum(1 for r in rows.values() if r["eval1_eq_route1"]),
        "rep_stable": sum(1 for r in rows.values() if r["rep_stable"]),
        "http_eq_v63": sum(1 for r in rows.values() if r["http_eq_v63"]),
        "network_attempts": len(v64_runtime.NET_ATTEMPTS), "blocked_client_calls": list(v64_runtime.BLOCKED_CALLS),
        "per_query": rows,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_query"}, ensure_ascii=False))
    router.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
