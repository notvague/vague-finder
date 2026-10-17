"""가사 경로가 실제로 돌았다는 실행 증거 — 오프라인 B1(가사 스냅샷)을 v62 `cand_fixed`(가사 경로 빈 조건, 같은 색인·캐시·구현)와 대조.

    venv/bin/python lyrics_evidence.py [--v62-root <v62 cand_fixed/cand_A>] → lyrics_evidence.json (곡 ID·건수·유형만, 가사 원문 없음)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(HERE))
from v63_runtime import LYRIC_QUERY_IDS, SETS  # noqa: E402

DEFAULT_V62 = "/private/tmp/vague-finder-hybrid-full-scan/experiments/reranking/results_v62_hybrid_full_scan/cand_fixed/cand_A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v62-root", default=DEFAULT_V62)
    parser.add_argument("--run", default=str(HERE / "offline/offline_B_rep1"))
    args = parser.parse_args()
    changed, per_query = [], {}
    total_lyrics_calls = errors = zero = 0
    for s in SETS:
        old = {}
        for line in open(f"{args.v62_root}/{s}_dev/candidates.jsonl", encoding="utf-8"):
            row = json.loads(line)
            old[row["query_id"]] = row["candidate_ids"]
        for line in open(f"{args.run}/{s}_dev/v63_queries.jsonl", encoding="utf-8"):
            r = json.loads(line)
            q = r["query_id"]
            ids = [c["id"] for c in r["candidates"]]
            if ids != old[q]:
                changed.append(q)
            total_lyrics_calls += len(r["lyrics"])
            errors += sum(1 for e in r["lyrics"] if "error" in e)
            zero += sum(1 for e in r["lyrics"] if e.get("results") == 0)
            if q in LYRIC_QUERY_IDS:
                e = r["lyrics"][0] if r["lyrics"] else {}
                types = [t["type"] for t in e.get("top", [])]
                per_query[q] = {
                    "lyrics_results": e.get("results"), "lyrics_ms": e.get("ms"),
                    "match_types_top30": {t: types.count(t) for t in sorted(set(types))},
                    "protected": len(r["protection"]["protected_ids"]),
                    "tiers": sorted(set(r["protection"]["tiers"].values())),
                    "rerank_groups": [[len(g["input_ids"]), g["top_k"]] for g in r["rerank_runs"]],
                    "policy_skip": r["protection"]["policy_skip"],
                    "candidates_changed_vs_lyrics_empty": ids != old[q],
                    "lyric_candidates_in_30": sum(1 for c in r["candidates"] if c["lyric"]),
                }
    out = {
        "baseline": args.v62_root, "run": args.run,
        "lyrics_search_calls": total_lyrics_calls, "lyrics_search_errors": errors, "lyrics_zero_result_calls": zero,
        "queries_with_candidates_changed_vs_lyrics_empty": changed,
        "non_lyric_queries_changed": [q for q in changed if q not in LYRIC_QUERY_IDS],
        "lyric_queries": per_query,
    }
    (HERE / "lyrics_evidence.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "lyric_queries"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
