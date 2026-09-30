"""
저장된 top-10으로 엄격·확장 지표를 다시 계산한다 (검색 재실행 없음).

엄격 지표는 원래 타깃(relevant_ids)만, 확장 지표는 원래 타깃 + 허용 정답(allowed_ids)을
정답으로 센다. 허용 정답은 3,010곡 코퍼스 기준 팀 라벨 검토(2026-09-30~10-01)로 정했다.
`evaluate_search_accuracy.py`의 detail CSV에 순위별 곡 ID(baseline_top_ids·rerank_top_ids)가
남아 있으므로, 라벨만 바뀌면 이 스크립트로 다시 집계하면 된다.

  python -m src.eval.relaxed_metrics --result-dir experiments/reranking/results_v22_corpus3010
  python -m src.eval.relaxed_metrics --result-dir experiments/reranking/results_v21_ce_topn/scored20_fullnorm --split test

엄격 지표는 원래 측정 요약(search_eval_{split}_summary.csv)과 같아야 한다. 다르면
detail과 라벨 파일이 어긋난 것이므로 확인하도록 경고한다.

주의: 허용 정답은 v21·v22 top-10 합집합에 든 후보만 검토했다. 다른 설정의 측정에 쓰면
검토하지 않은 곡이 top-10에 들어와도 오답으로 세지므로, 확장 지표가 낮게 나올 수 있다.
Candidate Recall@30은 후보 30개 목록이 detail에 없어 여기서 계산하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

DEFAULT_LABELS = Path("experiments/reranking/eval_queries_v07.csv")
KS = (1, 3, 5, 10)
SYSTEMS = ("baseline", "rerank")
TAG = "relaxed_v07"


def split_ids(raw: Optional[str]) -> List[str]:
    return [part.strip() for part in (raw or "").split("|") if part.strip()]


def first_rank(result_ids: Sequence[str], relevant: Set[str], k: int = 10) -> Optional[int]:
    for rank, song_id in enumerate(result_ids[:k], start=1):
        if song_id in relevant:
            return rank
    return None


def scores(result_ids: Sequence[str], relevant: Set[str]) -> Dict[str, float]:
    """Hit@k와 MRR@10. evaluate_search_accuracy.py의 정의와 같다."""
    rank = first_rank(result_ids, relevant, 10)
    out = {f"hit@{k}": float(rank is not None and rank <= k) for k in KS}
    out["mrr@10"] = 0.0 if rank is None else 1.0 / rank
    return out


def read_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def evaluate(detail: List[dict], labels: Dict[str, dict]) -> tuple[List[dict], List[dict]]:
    """질의별 행과 요약 행을 돌려준다."""
    per_query: List[dict] = []
    for row in detail:
        qid = row["query_id"]
        if qid not in labels:
            raise KeyError(f"{qid}: 라벨 파일에 없는 질의 — detail과 라벨의 질의 세트가 다르다")
        strict = set(split_ids(labels[qid]["relevant_ids"]))
        if strict != set(split_ids(row["relevant_ids"])):
            raise ValueError(f"{qid}: detail과 라벨 파일의 relevant_ids가 다르다")
        relaxed = strict | set(split_ids(labels[qid].get("allowed_ids")))
        out: dict = {"query_id": qid, "allowed_count": len(relaxed) - len(strict)}
        for system in SYSTEMS:
            top = split_ids(row[f"{system}_top_ids"])
            out[f"{system}_strict_rank"] = first_rank(top, strict) or ""
            out[f"{system}_relaxed_rank"] = first_rank(top, relaxed) or ""
            for name, value in scores(top, strict).items():
                out[f"{system}_strict_{name}"] = value
            for name, value in scores(top, relaxed).items():
                out[f"{system}_relaxed_{name}"] = value
        per_query.append(out)

    summary: List[dict] = []
    n = len(per_query)
    for name in [f"hit@{k}" for k in KS] + ["mrr@10"]:
        row = {"metric": name.replace("hit", "Hit").replace("mrr", "MRR")}
        for system in SYSTEMS:
            for kind in ("strict", "relaxed"):
                col = f"{system}_{kind}_{name}"
                row[f"{system}_{kind}"] = round(sum(q[col] for q in per_query) / n, 6) if n else 0.0
        summary.append(row)
    return per_query, summary


def check_against_original(summary: List[dict], original: Path) -> List[str]:
    """엄격 지표가 원래 측정 요약과 같은지 본다. 다른 항목을 돌려준다."""
    if not original.exists():
        return [f"원래 요약 없음: {original}"]
    base = {r["metric"]: r for r in read_csv(original)}
    diffs = []
    for row in summary:
        if row["metric"] not in base:
            continue  # Hit@3은 원래 요약에 없다
        for system in SYSTEMS:
            want = float(base[row["metric"]][system])
            got = row[f"{system}_strict"]
            if abs(want - got) > 1e-5:
                diffs.append(f"{row['metric']} {system}: 원래 {want} · 재계산 {got}")
    return diffs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="저장된 top-10으로 엄격·확장(허용 정답 포함) 지표 재계산")
    p.add_argument("--result-dir", type=Path, required=True, help="search_eval_{split}_detail.csv가 있는 폴더")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="allowed_ids 열이 있는 질의 CSV")
    p.add_argument("--split", action="append", choices=["dev", "test"], help="생략하면 dev·test 둘 다")
    return p


def main() -> None:
    args = build_parser().parse_args()
    labels = {r["query_id"]: r for r in read_csv(args.labels)}
    if not any("allowed_ids" in r for r in labels.values()):
        raise SystemExit(f"{args.labels}에 allowed_ids 열이 없다 — export_csv --with-allowed로 만든 파일을 쓸 것")

    for split in args.split or ["dev", "test"]:
        detail_path = args.result_dir / f"search_eval_{split}_detail.csv"
        if not detail_path.exists():
            print(f"[{split}] detail 없음: {detail_path} — 건너뜀")
            continue
        per_query, summary = evaluate(read_csv(detail_path), labels)

        summary_path = args.result_dir / f"search_eval_{split}_{TAG}_summary.csv"
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        ranks_path = args.result_dir / f"search_eval_{split}_{TAG}_ranks.csv"
        with open(ranks_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_query[0]))
            writer.writeheader()
            writer.writerows(per_query)

        with_allowed = sum(1 for q in per_query if q["allowed_count"])
        print(f"[{split}] 질의 {len(per_query)}개 (허용 정답 있는 질의 {with_allowed}개) — {summary_path}")
        print(f"  {'지표':<8}{'baseline 엄격':>14}{'확장':>8}{'rerank 엄격':>14}{'확장':>8}")
        for row in summary:
            print(f"  {row['metric']:<8}{row['baseline_strict']:>14.4f}{row['baseline_relaxed']:>8.4f}"
                  f"{row['rerank_strict']:>14.4f}{row['rerank_relaxed']:>8.4f}")
        diffs = check_against_original(summary, args.result_dir / f"search_eval_{split}_summary.csv")
        print("  엄격 지표 = 원래 요약" if not diffs else "  ⚠ 엄격 지표가 원래 요약과 다름: " + "; ".join(diffs))
        moved = [q for q in per_query if q["rerank_relaxed_rank"] != q["rerank_strict_rank"]]
        if moved:
            print("  허용 정답이 원래 타깃보다 먼저 나온 질의(rerank): "
                  + ", ".join(f"{q['query_id']}({q['rerank_strict_rank'] or '밖'}→{q['rerank_relaxed_rank']})" for q in moved))


if __name__ == "__main__":
    main()
