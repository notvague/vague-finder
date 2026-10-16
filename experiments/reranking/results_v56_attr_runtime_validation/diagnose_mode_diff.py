"""§4-1 진단 — 로컬 파일 모드와 서버 모드에서 후보가 달랐던 질의의 **경로별** 결과와 dense 전수 조회(상한 5,000) 경계를 뽑는다. 모델 호출 없음.

`--mode local|server` 로 두 번 돌리고 `--compare` 로 대조한다. A 구성(보조 경로 끔) 고정.
실행: venv/bin/python …/diagnose_mode_diff.py --mode local ; --mode server ; --compare
"""
from __future__ import annotations

import argparse
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

QUERY_IDS = ["q101", "t049", "c601", "n010", "m103", "n002"]


def run(mode: str) -> None:
    import v56_runtime

    v56_runtime.block_real_models()
    if mode == "server":
        os.environ["QDRANT_URL"] = os.getenv("V56_QDRANT_URL", "http://127.0.0.1:6356")
    else:
        os.environ.pop("QDRANT_URL", None)
        os.environ["QDRANT_PATH"] = "artifacts/qdrant_eval_v56_local"
    os.environ.update({"ATTR_FILTER_WEIGHT": "0", "FEEDBACK_COLLECTION_ENABLED": "0", "SEARCH_REFERENCE_YEAR": "2026",
                       "TOKENIZERS_PARALLELISM": "false"})
    import src.backend.api.dependencies as deps

    deps.get_reranker = deps.singleton(v56_runtime.make_fake_reranker)  # type: ignore[assignment]
    from src.backend.schemas.query import QueryAnalysis
    from src.retrieval.explain import ExplainRecorder
    from src.vector_db import qdrant_backend as qb
    from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME

    analyzer = v56_runtime.CachedAnalyzer()
    by_id = {qid: a for _q, (qid, a, _f, _p) in analyzer.by_query.items()}
    router = deps.get_search_router()
    service = deps.get_search_service()
    client = deps.get_vector_client().client
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    total = client.count(collection, exact=True).count
    out = {"mode": mode, "points": total, "scan_limit": qb.HYBRID_SCAN_LIMIT, "queries": {}}
    for qid in QUERY_IDS:
        analysis = QueryAnalysis(**by_id[qid])
        recorder = ExplainRecorder(analysis.original_query)
        cands: list = []
        asyncio.run(router.search(analysis, top_k=15, use_rerank=False, candidate_k=30, candidate_ids_out=cands, recorder=recorder))
        songs = {}
        for sid in cands:
            se = recorder.record.get(sid)
            if se is None:
                continue
            d = se.model_dump() if hasattr(se, "model_dump") else (se.__dict__ if hasattr(se, "__dict__") else {})
            songs[sid] = {
                "paths": [(p.get("path"), p.get("rank"), round(float(p.get("score") or 0), 6)) if isinstance(p, dict) else str(p) for p in (d.get("paths") or [])],
                "fused_score": d.get("fused_score"), "retrieval_score": d.get("retrieval_score"),
            }
        # dense 전수 조회 경계: 상한 5,000의 마지막 20점과, 5,143 전체에서 5,000 밖에 떨어지는 점
        dense_vec = service.text_embedder.embed_passages([f"query: {analysis.search_text}"], add_e5_prefix=False, normalize=True)[0].tolist()
        params = qb.models.SearchParams(exact=True) if os.getenv("QDRANT_URL") else None
        pts = client.query_points(collection, query=dense_vec, using=qb.DENSE_VECTOR, limit=total, with_payload=False, search_params=params).points
        ranked = [(str(p.id), float(p.score)) for p in pts]
        out["queries"][qid] = {
            "candidates": cands, "songs": songs,
            "dense_total_returned": len(ranked),
            "dense_rank_4990_5010": ranked[4989:5010],
            "dense_beyond_5000_ids": sorted(i for i, _ in ranked[5000:]),
            "dense_top5000_set_sha": __import__("hashlib").sha1("|".join(sorted(i for i, _ in ranked[:5000])).encode()).hexdigest()[:12],
        }
        print(mode, qid, len(cands), "dense returned", len(ranked), "beyond5000", len(ranked[5000:]), flush=True)
    router.shutdown()
    try:
        deps.close_vector_client()
    except Exception:
        pass
    (HERE / f"mode_diff_{mode}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def compare() -> None:
    L = json.loads((HERE / "mode_diff_local.json").read_text(encoding="utf-8"))
    S = json.loads((HERE / "mode_diff_server.json").read_text(encoding="utf-8"))
    report = {"points": L["points"], "scan_limit": L["scan_limit"], "queries": {}}
    for qid in QUERY_IDS:
        a, b = L["queries"][qid], S["queries"][qid]
        same_cands = a["candidates"] == b["candidates"]
        # 경로별 비교: 각 곡의 (path, rank) 집합 차이
        path_diff = {}
        for sid in set(a["songs"]) | set(b["songs"]):
            pa = {(p[0], p[1]) for p in a["songs"].get(sid, {}).get("paths", [])}
            pb = {(p[0], p[1]) for p in b["songs"].get(sid, {}).get("paths", [])}
            if pa != pb:
                path_diff[sid] = {"local": sorted(pa), "server": sorted(pb)}
        paths_involved = sorted({p[0] for d in path_diff.values() for side in ("local", "server") for p in d[side]})
        same_dense5000 = a["dense_top5000_set_sha"] == b["dense_top5000_set_sha"]
        boundary_local = [(i, round(s, 6)) for i, s in a["dense_rank_4990_5010"]]
        boundary_server = [(i, round(s, 6)) for i, s in b["dense_rank_4990_5010"]]
        beyond_only_local = sorted(set(a["dense_beyond_5000_ids"]) - set(b["dense_beyond_5000_ids"]))
        beyond_only_server = sorted(set(b["dense_beyond_5000_ids"]) - set(a["dense_beyond_5000_ids"]))
        report["queries"][qid] = {
            "candidates_identical": same_cands, "n_songs_with_path_diff": len(path_diff), "paths_involved": paths_involved,
            "dense_top5000_set_identical": same_dense5000, "dense_beyond5000_only_local": beyond_only_local,
            "dense_beyond5000_only_server": beyond_only_server, "boundary_local_4990_5010": boundary_local,
            "boundary_server_4990_5010": boundary_server, "path_diff_sample": dict(list(path_diff.items())[:6]),
        }
        print(f"{qid}: cands_identical={same_cands} songs_with_path_diff={len(path_diff)} paths={paths_involved} "
              f"dense_top5000_same={same_dense5000} beyond_only_local={len(beyond_only_local)} beyond_only_server={len(beyond_only_server)}")
    (HERE / "mode_diff_compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "server"))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    compare() if args.compare else run(args.mode)
