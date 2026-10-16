"""v62 — 후보 30개 단계 비교: 수정 전 A(v61 기준선) · 독립 기준 B(v61) · 수정 구현(이 폴더 `cand_fixed/cand_A`). 모델·DB 없음.

v61 `compare_candidates.py`를 그대로 import해 **출력 폴더만 이 폴더로** 돌린다(v61 폴더의 `candidate_changes.csv` 등은 건드리지 않는다).
  - `candidate_changes_fixed_vs_A.csv` · `candidate_summary_fixed_vs_A.json`  — 열 A = 수정 전(v61 A), 열 B = 수정 구현
  - `candidate_changes_fixed_vs_B.csv` · `candidate_summary_fixed_vs_B.json`  — 열 A = 독립 기준(v61 B), 열 B = 수정 구현
추가로 평가기 detail CSV의 **모든 열**을 질의별로 대조해(시간 열 제외) 어떤 열이 몇 건 달라졌는지 적고(`detail_column_diff.json`),
`candidates.jsonl`의 후보 30개 ID·순서·융합 점수를 대조한다. 리랭커는 꺼져 있었으므로 "리랭커 입력"은 곧 이 후보 목록·순서다 —
가사 보호 그룹 구성은 가사 경로가 빈 조건(Mongo 차단)에서 `baseline_lyric_match_types`·`explain_*` 열로만 간접 확인한다.
실행: venv/bin/python experiments/reranking/results_v62_hybrid_full_scan/compare_fixed.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v62_common as common  # noqa: E402

common.add_paths()
import compare_candidates as cc  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument("--fixed-root", default=str(common.HERE / "cand_fixed"), help="수정 구현의 후보 실행 루트(run_candidates_fixed.sh의 CAND_ROOT)")
_ARGS = _parser.parse_args()
FIXED_ROOT = Path(_ARGS.fixed_root).resolve()
TIME_COLUMNS = {"retrieval_ms", "rerank_only_ms"}
REQUIRED = ("candidates.jsonl", "search_eval_dev_detail.csv", "conditions.json")   # conditions.json은 실행이 끝까지 간 뒤에만 쓰인다


def _run_cc(baseline_arm: str, suffix: str) -> None:
    cc.HERE = common.HERE
    sys.argv = ["compare_candidates.py", "--baseline-root", str(common.V61_SOURCE_ROOT), "--baseline-arm", baseline_arm,
                "--compare-root", str(FIXED_ROOT), "--compare-arm", "A", "--no-reference", "--suffix", suffix]
    print(f"\n=== compare_candidates: v61 {baseline_arm} ↔ fixed ({suffix}) ===")
    cc.main()


def _detail(root: Path, arm: str, set_name: str) -> Dict[str, Dict[str, str]]:
    path = root / f"cand_{arm}" / f"{set_name}_dev" / "search_eval_dev_detail.csv"
    with path.open(encoding="utf-8-sig") as fh:
        return {row["query_id"]: row for row in csv.DictReader(fh)}


def _candidates(root: Path, arm: str, set_name: str) -> Dict[str, Dict[str, Any]]:
    path = root / f"cand_{arm}" / f"{set_name}_dev" / "candidates.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["query_id"]] = row
    return out


def _column_diff(label: str, base_arm: str) -> Dict[str, Any]:
    per_column: Counter = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = {}
    candidate_diffs: List[Dict[str, Any]] = []
    queries = 0
    for set_name in common.SETS:
        base, fixed = _detail(common.V61_SOURCE_ROOT, base_arm, set_name), _detail(FIXED_ROOT, "A", set_name)
        if set(base) != set(fixed):
            raise RuntimeError(f"{set_name}: 질의 집합이 다르다")
        cand_base, cand_fixed = _candidates(common.V61_SOURCE_ROOT, base_arm, set_name), _candidates(FIXED_ROOT, "A", set_name)
        for qid in base:
            queries += 1
            for col in base[qid]:
                if col in TIME_COLUMNS:
                    continue
                if base[qid].get(col) != fixed[qid].get(col):
                    per_column[col] += 1
                    examples.setdefault(col, [])
                    if len(examples[col]) < 5:
                        examples[col].append({"set": set_name, "query_id": qid, "before": base[qid].get(col)[:160], "after": fixed[qid].get(col)[:160]})
            cb, cf = cand_base[qid], cand_fixed[qid]
            same_ids = cb["candidate_ids"] == cf["candidate_ids"]
            score_delta = [abs(a - b) for a, b in zip(cb["candidate_scores"], cf["candidate_scores"])] if same_ids else []
            max_delta = max(score_delta) if score_delta else None
            if not same_ids or (max_delta is not None and max_delta > 1e-9):
                candidate_diffs.append({
                    "set": set_name, "query_id": qid, "same_ids_and_order": same_ids, "n_before": len(cb["candidate_ids"]), "n_after": len(cf["candidate_ids"]),
                    "changed_ids": sorted(set(cb["candidate_ids"]) ^ set(cf["candidate_ids"])),
                    "max_abs_fusion_score_delta": max_delta,
                    "positions_with_score_change": [i + 1 for i, d in enumerate(score_delta) if d > 1e-9][:10] if same_ids else None,
                })
    return {"label": label, "baseline": f"{common.V61_SOURCE_ROOT}/cand_{base_arm}", "fixed": str(FIXED_ROOT / "cand_A"), "queries": queries,
            "columns_compared": "detail CSV 전체 열(시간 열 retrieval_ms·rerank_only_ms 제외)",
            "columns_with_differences": dict(per_column), "examples": examples,
            "candidate_list_differences": candidate_diffs, "candidate_lists_identical": queries - len(candidate_diffs)}


def main() -> int:
    for set_name in common.SETS:
        folder = FIXED_ROOT / "cand_A" / f"{set_name}_dev"
        missing = [name for name in REQUIRED if not (folder / name).exists()]
        if missing:
            # conditions.json이 없으면 실행이 중간에 끊긴 것(옛 산출물만 남았을 수 있다) — 비교하지 않는다
            raise SystemExit(f"수정 구현의 후보 기록이 불완전하다: {folder} ({', '.join(missing)} 없음) — run_candidates_fixed.sh를 끝까지 돌린 뒤 비교한다")
    _run_cc("A", "fixed_vs_A")
    _run_cc("B", "fixed_vs_B")
    conditions = {}
    for set_name in common.SETS:
        path = FIXED_ROOT / "cand_A" / f"{set_name}_dev" / "conditions.json"
        c = json.loads(path.read_text(encoding="utf-8"))
        conditions[set_name] = {k: c.get(k) for k in ("git_sha", "point_count", "hybrid_scan_limit_default", "hybrid_scan_limit_used", "capture", "eval_seconds", "guard_after")}
    report = {"fixed_conditions": conditions, "vs_A": _column_diff("수정 전 A ↔ 수정 구현", "A"), "vs_B": _column_diff("독립 기준 B ↔ 수정 구현", "B")}
    (common.HERE / "detail_column_diff.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for key in ("vs_A", "vs_B"):
        r = report[key]
        print(f"\n[{r['label']}] 질의 {r['queries']}건 · 후보 목록(ID·순서·점수) 동일 {r['candidate_lists_identical']}건 · 열별 차이: {r['columns_with_differences']}")
        for d in r["candidate_list_differences"]:
            print("  후보 차이:", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
