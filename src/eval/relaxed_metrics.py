"""
저장된 top-10으로 엄격·확장 지표를 다시 계산한다 (검색 재실행 없음).

엄격 지표는 원래 타깃(relevant_ids)만, 확장 지표는 원래 타깃 + 허용 정답(allowed_ids)을
정답으로 센다. 허용 정답은 3,010곡 코퍼스 기준 팀 라벨 검토(2026-09-30~10-01)로 정했다.
라벨 버전은 결과 파일 이름에 남는다(eval_queries_v08.csv → search_eval_{split}_relaxed_v08_*.csv).
v07(질의 20개·80곡)과 v08(+2곡)은 v21~v24 측정에서 같은 숫자를 낸다 — 더한 두 곡이 그 측정들의 첫 정답보다 아래에 있다.
`evaluate_search_accuracy.py`의 detail CSV에 순위별 곡 ID(baseline_top_ids·rerank_top_ids)가
남아 있으므로, 라벨만 바뀌면 이 스크립트로 다시 집계하면 된다.

  python -m src.eval.relaxed_metrics --result-dir experiments/reranking/results_v22_corpus3010
  python -m src.eval.relaxed_metrics --result-dir experiments/reranking/results_v21_ce_topn/scored20_fullnorm --split test

detail은 곡 제목·설명 문장이 들어 있어 커밋하지 않는다. 대신 곡 ID만 담은 top-10 경량본
(search_eval_{split}_top10.csv)을 커밋한다 — detail이 있으면 이 스크립트가 경량본을 새로 쓰고,
detail이 없으면 경량본을 읽는다. 측정 때는 evaluate_search_accuracy.py가 함께 쓴다.

범주형 집계: docs/eval/queries.json에서 target_scope=categorical인 질의(질의가 일반 속성만 말해
원래 타깃을 특정할 수 없는 질의)를 나눠 본 요약을 search_eval_{split}_{tag}_scope_summary.csv로 함께 낸다.
전체 지표는 바꾸지 않는다 — 범주형 질의도 전체 집계에 그대로 들어간다.

엄격 지표는 원래 측정 요약(search_eval_{split}_summary.csv)과 같아야 한다. 다르면
detail과 라벨 파일이 어긋난 것이므로 확인하도록 경고한다.

주의: 허용 정답은 v21·v22 top-10 합집합(+ v23 경로 깊이 40·v24 CE 30곡 채점의 새 top-10)에 든 후보만 검토했다. 다른 설정의 측정에 쓰면
검토하지 않은 곡이 top-10에 들어와도 오답으로 세지므로, 확장 지표가 낮게 나올 수 있다.
Candidate Recall@30은 후보 30개 목록이 detail에 없어 여기서 계산하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from src.eval.loader import DEFAULT_EVAL_PATH, load_target_scopes
from src.eval.schema import TARGET_SCOPES

DEFAULT_LABELS = Path("experiments/reranking/eval_queries_v08.csv")
KS = (1, 3, 5, 10)
SYSTEMS = ("baseline", "rerank")
# top-10 경량본의 열. 곡 ID와 질의 ID만 — 공개 레포에 올려도 되는 범위다(docs/data_policy.md).
TOP10_COLUMNS = ("query_id", "split", "relevant_ids", "baseline_top_ids", "rerank_top_ids")
# 범주형 집계의 묶음. all은 원래 요약과 같은 전체 집합이다.
SCOPES = ("all",) + TARGET_SCOPES


def tag_for(labels: Path) -> str:
    """결과 파일 이름에 넣을 라벨 표시. eval_queries_v08.csv → relaxed_v08"""
    m = re.search(r"_(v\d+)$", labels.stem)
    return f"relaxed_{m.group(1) if m else labels.stem}"


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


def write_csv(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def top10_path(result_dir: Path, split: str) -> Path:
    return result_dir / f"search_eval_{split}_top10.csv"


def write_top10(detail: List[dict], path: Path) -> None:
    """detail에서 곡 ID 열만 골라 top-10 경량본을 쓴다."""
    if detail:
        write_csv(path, [{col: row[col] for col in TOP10_COLUMNS} for row in detail])


def load_result_rows(result_dir: Path, split: str) -> Optional[tuple[List[dict], Path]]:
    """detail이 있으면 detail을, 없으면 top-10 경량본을 읽는다. 둘 다 없으면 None."""
    for path in (result_dir / f"search_eval_{split}_detail.csv", top10_path(result_dir, split)):
        if path.exists():
            return read_csv(path), path
    return None


def evaluate(
    detail: List[dict],
    labels: Dict[str, dict],
    scopes: Optional[Dict[str, str]] = None,
) -> tuple[List[dict], List[dict]]:
    """질의별 행과 요약 행을 돌려준다. scopes를 주면 질의별 행에 target_scope 열을 붙인다."""
    per_query: List[dict] = []
    for row in detail:
        qid = row["query_id"]
        if qid not in labels:
            raise KeyError(f"{qid}: 라벨 파일에 없는 질의 — detail과 라벨의 질의 세트가 다르다")
        if scopes is not None and qid not in scopes:
            raise KeyError(f"{qid}: queries.json에 없는 질의 — 범주형 여부를 알 수 없다")
        strict = set(split_ids(labels[qid]["relevant_ids"]))
        if strict != set(split_ids(row["relevant_ids"])):
            raise ValueError(f"{qid}: detail과 라벨 파일의 relevant_ids가 다르다")
        relaxed = strict | set(split_ids(labels[qid].get("allowed_ids")))
        out: dict = {"query_id": qid, "allowed_count": len(relaxed) - len(strict)}
        if scopes is not None:
            out["target_scope"] = scopes[qid]
        for system in SYSTEMS:
            top = split_ids(row[f"{system}_top_ids"])
            out[f"{system}_strict_rank"] = first_rank(top, strict) or ""
            out[f"{system}_relaxed_rank"] = first_rank(top, relaxed) or ""
            for name, value in scores(top, strict).items():
                out[f"{system}_strict_{name}"] = value
            for name, value in scores(top, relaxed).items():
                out[f"{system}_relaxed_{name}"] = value
        per_query.append(out)
    return per_query, summarize(per_query)


def summarize(per_query: List[dict]) -> List[dict]:
    """질의 평균. 질의가 없으면 값을 비운다(0으로 깔지 않는다)."""
    summary: List[dict] = []
    n = len(per_query)
    for name in [f"hit@{k}" for k in KS] + ["mrr@10"]:
        row = {"metric": name.replace("hit", "Hit").replace("mrr", "MRR")}
        for system in SYSTEMS:
            for kind in ("strict", "relaxed"):
                col = f"{system}_{kind}_{name}"
                row[f"{system}_{kind}"] = round(sum(q[col] for q in per_query) / n, 6) if n else ""
        summary.append(row)
    return summary


def scope_summary(per_query: List[dict]) -> List[dict]:
    """범주형 집계 — 전체 · 범주형 제외(specific) · 범주형으로 나눈 요약. evaluate(..., scopes)의 행을 받는다."""
    rows: List[dict] = []
    for scope in SCOPES:
        group = per_query if scope == "all" else [q for q in per_query if q["target_scope"] == scope]
        for row in summarize(group):
            rows.append({"scope": scope, "n_queries": len(group), **row})
    return rows


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
    p.add_argument(
        "--result-dir", type=Path, required=True,
        help="search_eval_{split}_detail.csv 또는 search_eval_{split}_top10.csv가 있는 폴더",
    )
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="allowed_ids 열이 있는 질의 CSV")
    p.add_argument("--queries", type=Path, default=DEFAULT_EVAL_PATH, help="target_scope를 읽을 질의 원본")
    p.add_argument("--split", action="append", choices=["dev", "test"], help="생략하면 dev·test 둘 다")
    return p


def main() -> None:
    args = build_parser().parse_args()
    labels = {r["query_id"]: r for r in read_csv(args.labels)}
    tag = tag_for(args.labels)
    if not any("allowed_ids" in r for r in labels.values()):
        raise SystemExit(f"{args.labels}에 allowed_ids 열이 없다 — export_csv --with-allowed로 만든 파일을 쓸 것")
    scopes = load_target_scopes(args.queries)

    for split in args.split or ["dev", "test"]:
        loaded = load_result_rows(args.result_dir, split)
        if loaded is None:
            print(f"[{split}] detail도 top-10 경량본도 없음: {args.result_dir} — 건너뜀")
            continue
        rows, source = loaded
        if source != top10_path(args.result_dir, split):
            write_top10(rows, top10_path(args.result_dir, split))
        per_query, summary = evaluate(rows, labels, scopes)

        summary_path = args.result_dir / f"search_eval_{split}_{tag}_summary.csv"
        write_csv(summary_path, summary)
        write_csv(args.result_dir / f"search_eval_{split}_{tag}_ranks.csv", per_query)
        by_scope = scope_summary(per_query)
        scope_path = args.result_dir / f"search_eval_{split}_{tag}_scope_summary.csv"
        write_csv(scope_path, by_scope)

        with_allowed = sum(1 for q in per_query if q["allowed_count"])
        print(f"[{split}] 질의 {len(per_query)}개 (허용 정답 있는 질의 {with_allowed}개, 입력 {source.name}) — {summary_path}")
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
        print(f"  범주형 집계 (rerank Hit@10 엄격 · 확장) — {scope_path.name}")
        for row in by_scope:
            if row["metric"] == "Hit@10":
                cell = (f"{row['rerank_strict']:.4f} · {row['rerank_relaxed']:.4f}"
                        if row["n_queries"] else "질의 없음")
                print(f"    {row['scope']:<12}{row['n_queries']:>3}개  {cell}")


if __name__ == "__main__":
    main()
