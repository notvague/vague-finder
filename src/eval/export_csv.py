"""
queries.json(SSOT) → 측정용 CSV 내보내기.

`src/retrieval/evaluate_search_accuracy.py`는 CSV를 읽는다. 그 CSV를 손으로
관리하면 라벨이 두 곳에 생기고 어긋난다 (v0.4에서 실제로 일어난 일 —
queries.json 60개와 eval_queries_v04.csv 53개가 따로 놀았다).
여기서는 항상 queries.json에서 생성한다.

  python -m src.eval.export_csv
  python -m src.eval.export_csv --query-set v04 --out experiments/reranking/eval_queries_v04_regen.csv

split 컬럼은 그대로 실어 보내므로, 실제 dev/test 선택은 평가 스크립트의
`--split`이 맡는다. 내보내기 단계에서 미리 거르지 않는 이유는 한 파일로
두 split을 모두 재현할 수 있게 하기 위해서다.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

from src.eval.loader import DEFAULT_EVAL_PATH, load_eval_set
from src.eval.schema import EvalQuery

DEFAULT_OUT = Path("experiments/reranking/eval_queries_v05.csv")

# evaluate_search_accuracy.py가 요구하는 컬럼.
COLUMNS = ["query_id", "split", "query_type", "query", "relevant_ids"]


def select(
    queries: List[EvalQuery],
    query_sets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> List[EvalQuery]:
    """정확도 집계가 가능한 질의만 고른다.

    정답이 없는 질의(개방형 추천, 단서 극빈, 표지 확인 전이라 라벨이 비어 있는
    질의)를 넣으면 Hit가 0으로 깔려 다른 질의의 개선을 가린다. 부분 라벨
    (pending_cover인데 positives가 있는 경우)은 포함한다.
    """
    picked = [q for q in queries if q.is_scorable]
    if query_sets:
        picked = [q for q in picked if q.query_set in query_sets]
    if splits:
        picked = [q for q in picked if q.split in splits]
    return picked


def to_rows(queries: List[EvalQuery]) -> List[dict]:
    return [
        {
            "query_id": q.query_id,
            "split": q.split,
            "query_type": "search",
            "query": q.query,
            "relevant_ids": "|".join(q.positives),
        }
        for q in queries
    ]


def write_csv(rows: List[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="queries.json → 측정용 CSV")
    p.add_argument("--queries", type=Path, default=DEFAULT_EVAL_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--query-set",
        action="append",
        choices=["v04", "modality_v1", "clarify_v1"],
        help="특정 출처 세트만. 반복 지정 가능. 생략하면 전부",
    )
    p.add_argument(
        "--split",
        action="append",
        choices=["dev", "test"],
        help="특정 split만. 생략하면 둘 다 내보내고 평가 스크립트가 고른다",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    eval_set = load_eval_set(args.queries)
    picked = select(eval_set.queries, args.query_set, args.split)
    write_csv(to_rows(picked), args.out)

    from collections import Counter
    print(f"{args.out} — {len(picked)}개")
    print("  split:", dict(Counter(q.split for q in picked)))
    print("  query_set:", dict(Counter(q.query_set for q in picked)))
    excluded = [q.query_id for q in eval_set.queries if not q.is_scorable]
    if excluded:
        print(f"  집계 제외 {len(excluded)}개 (정답 없음): {', '.join(excluded)}")
    unverified = [q.query_id for q in picked if not q.has_verified_label]
    if unverified:
        print(f"  라벨 일부 미확정 {len(unverified)}개 (표지 확인 전): {', '.join(unverified)}")


if __name__ == "__main__":
    main()
