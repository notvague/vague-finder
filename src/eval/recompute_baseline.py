"""
기존 측정 결과를 새 split 기준으로 재집계한다.

dev/test를 새로 나누면 "그럼 기존 기준선은 못 쓰는 것 아니냐"는 문제가 생긴다.
그렇지 않다. `evaluate_search_accuracy.py`의 detail CSV에는 질의별 지표가
그대로 남아 있으므로, 검색을 다시 돌리지 않고 부분집합으로 다시 평균 내면 된다.
v0.5 기준선(2026-08-19, 905곡)과의 연속성은 이 방식으로 유지한다.

  python -m src.eval.recompute_baseline
  python -m src.eval.recompute_baseline --detail experiments/reranking/results_v05/search_eval_dev_detail.csv

주의: 새 질의(clarify_v1)는 그 측정에 포함되지 않았으므로 여기서 집계되지 않는다.
비교 가능한 것은 측정 당시 존재하던 질의(query_set=v04)뿐이다.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.eval.loader import DEFAULT_EVAL_PATH, load_eval_set

DEFAULT_DETAIL = Path("experiments/reranking/results_v05/search_eval_dev_detail.csv")
# detail CSV는 analysis_json 때문에 176KB라 .gitignore 대상이다. 순위·지표만 남긴
# 경량본은 커밋하므로, 측정을 돌리지 않은 팀원도 재집계를 재현할 수 있다.
DEFAULT_RANKS = Path("experiments/reranking/results_v05/search_eval_dev_ranks.csv")

# 경량본에 남길 컬럼 — 질의별 분석과 split 재집계에 실제로 쓰이는 것만.
# split은 일부러 뺀다 — 측정 당시의 split이 굳어 있으면 queries.json(SSOT)과
# 어긋난다. 재집계는 항상 queries.json의 split을 기준으로 한다.
RANK_COLUMNS = [
    "query_id",
    "candidate_rank@30", "candidate_recall@30",
    "baseline_rank", "rerank_rank", "rank_change_positive_is_better",
    "baseline_hit@1", "rerank_hit@1",
    "baseline_hit@5", "rerank_hit@5",
    "baseline_hit@10", "rerank_hit@10",
    "baseline_mrr@10", "rerank_mrr@10",
    "baseline_ndcg@10", "rerank_ndcg@10",
    "rerank_confidence", "effective_rerank_weight",
]


def export_ranks(detail: Dict[str, dict], out: Path) -> int:
    """detail CSV에서 순위·지표 컬럼만 뽑아 커밋 가능한 크기로 저장한다."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RANK_COLUMNS)
        writer.writeheader()
        for row in detail.values():
            writer.writerow({c: row.get(c, "") for c in RANK_COLUMNS})
    return len(detail)

METRICS = [
    ("Hit@1", "baseline_hit@1", "rerank_hit@1"),
    ("Hit@5", "baseline_hit@5", "rerank_hit@5"),
    ("Hit@10", "baseline_hit@10", "rerank_hit@10"),
    ("MRR@10", "baseline_mrr@10", "rerank_mrr@10"),
    ("nDCG@10", "baseline_ndcg@10", "rerank_ndcg@10"),
]


def read_detail(path: Path) -> Dict[str, dict]:
    with open(path, encoding="utf-8-sig") as f:
        return {row["query_id"]: row for row in csv.DictReader(f)}


def mean(rows: List[dict], column: str) -> Optional[float]:
    values = [float(r[column]) for r in rows if (r.get(column) or "").strip()]
    return sum(values) / len(values) if values else None


def rank_bucket(row: dict) -> str:
    """정답이 어디에 있었는지. 재질문이 개입할 수 있는 구간을 가른다."""
    rr = (row["rerank_rank"] or "").strip()
    if not rr:
        return "Top-10 밖"
    return "1~5위" if int(float(rr)) <= 5 else "6~10위"


def main() -> None:
    parser = argparse.ArgumentParser(description="기존 detail CSV를 새 split으로 재집계")
    parser.add_argument("--queries", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument(
        "--detail",
        type=Path,
        default=None,
        help=f"기본값: {DEFAULT_RANKS}가 있으면 그것, 없으면 {DEFAULT_DETAIL}",
    )
    parser.add_argument(
        "--export-ranks",
        type=Path,
        nargs="?",
        const=DEFAULT_RANKS,
        help="detail CSV에서 순위·지표만 뽑아 경량 CSV로 저장한다 (커밋용)",
    )
    args = parser.parse_args()

    source = args.detail or (DEFAULT_RANKS if DEFAULT_RANKS.exists() else DEFAULT_DETAIL)
    eval_set = load_eval_set(args.queries)
    detail = read_detail(source)

    if args.export_ranks:
        n = export_ranks(detail, args.export_ranks)
        size = args.export_ranks.stat().st_size
        print(f"경량본 저장: {args.export_ranks} — {n}행 / {size:,}바이트\n")

    by_split: Dict[str, List[dict]] = defaultdict(list)
    missing: List[str] = []
    for q in eval_set.queries:
        row = detail.get(q.query_id)
        if row is None:
            # clarify_v1과 modality_v1은 이 측정에 없었다 — 정상이다.
            if q.query_set == "v04" and q.is_scorable:
                missing.append(q.query_id)
            continue
        by_split[q.split].append(row)
        by_split["전체"].append(row)

    print(f"기준선: {source}")
    print(f"질의 세트: {args.queries} (v04 부분집합만 비교 가능)\n")

    header = f"{'지표':<10}" + "".join(f"{name:>22}" for name in ("전체", "dev", "test"))
    print(header)
    print("-" * len(header))
    for label, base_col, rr_col in METRICS:
        line = f"{label:<10}"
        for split in ("전체", "dev", "test"):
            rows = by_split.get(split, [])
            b, r = mean(rows, base_col), mean(rows, rr_col)
            line += f"{'—':>22}" if b is None else f"{b:>10.3f} →{r:>9.3f}"
        print(line)

    print()
    for split in ("전체", "dev", "test"):
        rows = by_split.get(split, [])
        if not rows:
            continue
        buckets = defaultdict(int)
        for row in rows:
            buckets[rank_bucket(row)] += 1
        summary = "  ".join(f"{k} {v}" for k, v in sorted(buckets.items()))
        print(f"{split:<5} n={len(rows):<3} | {summary}")

    if missing:
        print(f"\n⚠️ detail에 없는 v04 질의 {len(missing)}개: {', '.join(missing)}")
        print("   (측정 이후 라벨이 바뀐 질의라면 해당 항목은 재집계에서 빠진다)")


if __name__ == "__main__":
    main()
