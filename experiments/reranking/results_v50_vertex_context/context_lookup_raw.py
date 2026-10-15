"""Context 조회가 정답을 못 돌려준 까닭을 가른다 — 융합 조회(필터 전) vs 매체 설명 필터(후). 모델 호출 없음.

context_lookup.csv에서 "정답에 나무위키 사실이 있는데 Context 조회 결과에 없음"인 질의만 골라, 라우터 `_search_context`와 같은
순서로 (1) 단서별 조회 문자열 → (2) `search_fused_songs`(필터 전 순위) → (3) `combine_context_clues` → (4)
`context_candidate_matches_media_description` 필터 뒤 순위를 적는다. 어느 단계에서 정답이 사라졌는지가 열 `lost_at`이다.

    CONTEXT_WEIGHT=0.5 QDRANT_PATH=artifacts/qdrant_eval_c venv/bin/python <이 파일> --output context_lookup_raw.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=HERE / "context_lookup_raw.csv")
    ap.add_argument("--all", action="store_true", help="사실 유무와 무관하게 context_clues 질의 전부")
    args = ap.parse_args()

    from src.backend.api.dependencies import get_search_router
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.context_evidence import context_candidate_matches_media_description
    from src.retrieval.context_query import context_media_target_terms, context_search_queries
    from src.retrieval.context_route import combine_context_clues
    from src.retrieval.evaluate_search_accuracy import read_queries

    targets = {}
    for r in csv.DictReader(open(HERE / "context_lookup.csv", encoding="utf-8-sig")):
        if args.all or (int(r["target_facts"]) > 0 and not r["target_context_rank"]):
            targets[(r["set"], r["query_id"])] = r["target_id"]
    router = get_search_router()
    search = router._context_search  # noqa: SLF001
    out = []
    for set_name in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{set_name}_dev_vertex.json")
        for row in read_queries(ROOT / f"experiments/reranking/eval_queries_{set_name}.csv", "dev"):
            qid, query = row["query_id"].strip(), row["query"].strip()
            if (set_name, qid) not in targets:
                continue
            target = targets[(set_name, qid)]
            analysis = cache.get(qid, query)
            rankings = []
            for clue in analysis.context_clues:
                if clue.confidence <= 0:
                    continue
                for position, q in enumerate(context_search_queries(clue, original_query=analysis.original_query)):
                    media = context_media_target_terms(clue, original_query=analysis.original_query) if position else ()
                    hits = search.search_fused_songs(q, fact_k=100, sparse_k=100, **({"media_targets": media} if media else {}))
                    raw_rank = next((i for i, h in enumerate(hits, 1) if h.song_id == target), "")
                    rankings.append((clue, hits))
                    out.append({
                        "set": set_name, "query_id": qid, "target_id": target, "stage": "fused_lookup",
                        "lookup_query": q, "media_targets": "|".join(media), "n_hits": len(hits),
                        "target_rank": raw_rank,
                        "target_dense_rank": next((h.dense_rank for h in hits if h.song_id == target), ""),
                        "target_sparse_rank": next((h.sparse_rank for h in hits if h.song_id == target), ""),
                        "top3": "|".join(h.song_id for h in hits[:3]),
                    })
            combined = combine_context_clues(rankings)
            comb_rank = next((i for i, h in enumerate(combined, 1) if h.song_id == target), "")
            filtered = tuple(h for h in combined if context_candidate_matches_media_description(h, query_clues=analysis.context_clues))
            filt_rank = next((i for i, h in enumerate(filtered, 1) if h.song_id == target), "")
            lost_at = ("fused_lookup" if not comb_rank else "media_description_filter" if not filt_rank else "kept")
            out.append({
                "set": set_name, "query_id": qid, "target_id": target, "stage": "combined→filtered",
                "lookup_query": "", "media_targets": "", "n_hits": f"{len(combined)}→{len(filtered)}",
                "target_rank": f"{comb_rank or '없음'}→{filt_rank or '없음'}", "target_dense_rank": "", "target_sparse_rank": "",
                "top3": "|".join(h.song_id for h in filtered[:3]), "lost_at": lost_at,
            })
            print(f"{set_name} {qid}: 융합 {len(combined)}건(정답 {comb_rank or '없음'}) → 필터 뒤 {len(filtered)}건(정답 {filt_rank or '없음'}) · {lost_at}", flush=True)
    fields = list(out[0]) + (["lost_at"] if "lost_at" not in out[0] else [])
    with open(args.output, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    print(f"기록: {args.output}")


if __name__ == "__main__":
    main()
