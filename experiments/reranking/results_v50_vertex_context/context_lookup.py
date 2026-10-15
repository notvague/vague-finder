"""Context 조회 성공과 최종 후보 기여를 가른다 — dev 176 중 `context_clues`가 있는 질의만, 모델 호출 없음.

평가 기록(detail·explain)은 **후보에 든 곡**의 경로만 남긴다. 정답이 후보 밖이면 "Context 조회가 정답을 돌려줬는데 융합에서
밀렸는가, 조회 자체가 못 찾았는가, 정답에 사실이 없는가"를 알 수 없다. 그래서 라우터의 Context 조회(`_search_context`)만
같은 분석·같은 폭(최종 후보 30 기준)으로 다시 불러 질의별로 적는다. 순위 계산을 바꾸지 않는다.

    CONTEXT_WEIGHT=0.5 QDRANT_PATH=artifacts/qdrant_eval_b venv/bin/python <이 파일> --output context_lookup.csv

열: set · query_id · n_clues · clue_queries · n_context_hits · target_id · target_facts(나무위키 사실 수, 0=여담 없음/문서 없음) ·
target_context_rank(조회 결과 안 순위, 빈 값=조회 결과에 없음) · target_dense_rank · target_sparse_rank · target_best_fact(record_id) ·
top3_hits · cand_rank_ctx0 · cand_rank_ctx0.5(후보 측정 detail이 있으면).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"


def facts_of(song_id: str) -> tuple[int, str]:
    path = ROOT / "artifacts/context/songs" / f"{song_id}.json"
    if not path.exists():
        return 0, "no_artifact"
    data = json.loads(path.read_text(encoding="utf-8"))
    retrieval = data.get("retrieval") or {}
    return int(retrieval.get("record_count") or len(retrieval.get("records") or [])), str(data.get("status") or "")


def cand_rank(run: str, set_name: str) -> dict:
    path = HERE / run / f"{set_name}_dev" / "search_eval_dev_detail.csv"
    if not path.exists():
        return {}
    return {r["query_id"]: r.get("candidate_rank@30", "") for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=HERE / "context_lookup.csv")
    ap.add_argument("--candidate-k", type=int, default=30)
    args = ap.parse_args()
    if float(os.getenv("CONTEXT_WEIGHT", "0.5")) <= 0:
        raise SystemExit("CONTEXT_WEIGHT가 0이면 라우터가 Context를 조회하지 않는다 — 0.5로 둔다")

    from src.backend.api.dependencies import get_search_router
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.evaluate_search_accuracy import parse_relevant_ids, read_queries

    router = get_search_router()
    rows_out = []
    for set_name in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{set_name}_dev_vertex.json")
        a_ranks = cand_rank("cand_ctx0", set_name)
        b_ranks = cand_rank("cand_ctx0.5", set_name)
        for row in read_queries(ROOT / f"experiments/reranking/eval_queries_{set_name}.csv", "dev"):
            qid, query = row["query_id"].strip(), row["query"].strip()
            analysis = cache.get(qid, query)
            if not analysis.context_clues:
                continue
            hits = router._search_context(analysis, args.candidate_k)  # noqa: SLF001 - 진단용, 라우터와 같은 조회
            relevant = sorted(parse_relevant_ids(row["relevant_ids"]))
            by_id = {hit.route.song_id: (i, hit) for i, hit in enumerate(hits, 1)}
            for target in relevant:
                facts, status = facts_of(target)
                found = by_id.get(target)
                fused = found[1].route.fused_hit if found else None
                best = getattr(getattr(fused, "dense_song", None), "best_fact", None) if fused else None
                rows_out.append({
                    "set": set_name, "query_id": qid, "query": query,
                    "n_clues": len(analysis.context_clues),
                    "clue_queries": " || ".join(c.search_query for c in analysis.context_clues),
                    "clue_targets": " || ".join(c.target for c in analysis.context_clues),
                    "clue_confidence": " || ".join(f"{c.confidence:.2f}" for c in analysis.context_clues),
                    "n_context_hits": len(hits),
                    "target_id": target,
                    "target_facts": facts, "target_artifact_status": status,
                    "target_context_rank": found[0] if found else "",
                    "target_dense_rank": getattr(fused, "dense_rank", "") if fused else "",
                    "target_sparse_rank": getattr(fused, "sparse_rank", "") if fused else "",
                    "target_best_fact": getattr(best, "record_id", "") if best else "",
                    "top3_hits": "|".join(h.route.song_id for h in hits[:3]),
                    "cand_rank_ctx0": a_ranks.get(qid, ""),
                    "cand_rank_ctx0.5": b_ranks.get(qid, ""),
                })
            print(f"{set_name} {qid}: Context 조회 {len(hits)}건 · 정답 {[by_id[t][0] if t in by_id else '없음' for t in relevant]}", flush=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    n = len(rows_out)
    found = sum(1 for r in rows_out if r["target_context_rank"])
    with_facts = sum(1 for r in rows_out if r["target_facts"])
    print(f"\n정답 곡 기준 {n}건: 나무위키 사실 있음 {with_facts} · Context 조회 결과에 정답 있음 {found} · 기록 {args.output}")


if __name__ == "__main__":
    main()
