"""진단 집계 — 단계표(`failure_stage_table.csv`) + 사람 검토(`failure_review_v67.csv`) → 원인별 규모·v50 대비·지렛대(겹침). 모델 호출 없음.

검사(하나라도 어긋나면 집계하지 않는다):
- 검토 행 = 실패 질의 집합(누락·중복·여분 0) · 단계표의 대표 줄이 질의마다 정확히 하나
- 주 원인 코드 ∈ {A,B,R,P,F,K,L,U} · 보조 원인 ∈ 같은 집합 또는 빈 값 · 주 원인 ≠ 보조 원인
- 단계와 주 원인의 정합: 후보 안 → K, 후보 밖이면 K 금지 · 자동 단계 F(융합·보정 뒤 절단) → F · P(전체 깊이 가산 뒤 ≤30) → P ·
  그 밖의 경로 밖 → A·B·R·L·U 중 하나(사람 판정)
- 경계: 후보 순위 30은 후보 안, 31 이상은 후보 밖(단계표 pool_rank가 1~30인지) · 전체 깊이 가산 뒤 30은 P, 31은 P 아님
- 정답 여럿: 대표 줄이 가장 멀리 간 곡인지(단계 순서·후보 순위) 다시 계산
- 합계: 주 원인 합계 = 후보 밖 질의 수 / 후보 내 질의 수(보조 원인·지렛대는 겹치므로 합계에 넣지 않는다)

    venv/bin/python summarize_diagnosis.py [--stage <csv>] [--review <csv>] [--out <json>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
CODES = {"A", "B", "R", "P", "F", "K", "L", "U"}
HUMAN = {"A", "B", "R", "L", "U"}
STAGE_ORDER = {"후보 안": 0, "보정 뒤 절단": 1, "융합 절단": 2, "경로 밖": 3}


class DiagnosisError(ValueError):
    pass


def expected_primary(rep: Dict[str, str]) -> set:
    if rep["stage"] == "후보 안":
        return {"K"}
    if rep["stage"] in ("융합 절단", "보정 뒤 절단"):
        return {"F"}
    bd = int(rep["boosted_rank_deep"]) if rep["boosted_rank_deep"] else None
    if bd is not None and bd <= 30:
        return {"P"}
    return HUMAN


def summarize(stage_rows: List[Dict[str, str]], review_rows: List[Dict[str, str]],
              v50_fail: set | None = None) -> Dict[str, Any]:
    by_q: Dict[str, List[Dict[str, str]]] = {}
    for r in stage_rows:
        by_q.setdefault(r["query_id"], []).append(r)
    review_ids = [r["query_id"] for r in review_rows]
    if len(review_ids) != len(set(review_ids)):
        raise DiagnosisError(f"검토 행 중복: {[q for q, n in Counter(review_ids).items() if n > 1]}")
    if set(review_ids) != set(by_q):
        raise DiagnosisError(f"검토 누락 {sorted(set(by_q) - set(review_ids))} · 여분 {sorted(set(review_ids) - set(by_q))}")
    review = {r["query_id"]: r for r in review_rows}
    reps: Dict[str, Dict[str, str]] = {}
    for q, rows in by_q.items():
        flagged = [r for r in rows if r["representative"] == "1"]
        if len(flagged) != 1:
            raise DiagnosisError(f"{q}: 대표 줄 {len(flagged)}개")
        best = min(rows, key=lambda x: (STAGE_ORDER[x["stage"]], int(x["pool_rank"] or 999), int(x["boosted_rank_deep"] or 99999)))
        if best["song_id"] != flagged[0]["song_id"]:
            raise DiagnosisError(f"{q}: 대표 곡이 가장 멀리 간 정답이 아니다")
        for r in rows:
            if r["pool_rank"] and not 1 <= int(r["pool_rank"]) <= 30:
                raise DiagnosisError(f"{q}: 후보 순위 {r['pool_rank']}가 1~30 밖")
            if (r["stage"] == "후보 안") != bool(r["pool_rank"]):
                raise DiagnosisError(f"{q}: 단계와 후보 순위 불일치")
        reps[q] = flagged[0]
    out_pool = [q for q in reps if reps[q]["stage"] != "후보 안"]
    in_pool = [q for q in reps if reps[q]["stage"] == "후보 안"]
    for q, rv in review.items():
        p, s = rv["primary"].strip(), rv["secondary"].strip()
        if p not in CODES:
            raise DiagnosisError(f"{q}: 주 원인 코드 '{p}'")
        if s and s not in CODES:
            raise DiagnosisError(f"{q}: 보조 원인 코드 '{s}'")
        if s and s == p:
            raise DiagnosisError(f"{q}: 주 원인 = 보조 원인")
        allowed = expected_primary(reps[q])
        if p not in allowed:
            raise DiagnosisError(f"{q}: 단계 '{reps[q]['stage']}'(가산 뒤 {reps[q]['boosted_rank_deep']})에 주 원인 {p} 불가 — 허용 {sorted(allowed)}")
        if not rv["evidence"].strip():
            raise DiagnosisError(f"{q}: 근거 없음")
    prim_out = Counter(review[q]["primary"] for q in out_pool)
    prim_in = Counter(review[q]["primary"] for q in in_pool)
    if sum(prim_out.values()) != len(out_pool) or sum(prim_in.values()) != len(in_pool):
        raise DiagnosisError("주 원인 합계가 질의 수와 다르다")
    levers = Counter()
    lever_q: Dict[str, List[str]] = {}
    for q in reps:
        for lv in filter(None, review[q]["lever"].split("|")):
            levers[(lv, "out" if q in out_pool else "in")] += 1
            lever_q.setdefault(lv, []).append(q)
    result: Dict[str, Any] = {
        "failures": len(reps), "out_of_pool": len(out_pool), "in_pool": len(in_pool),
        "multi_gold_failures": sorted(q for q, rows in by_q.items() if len(rows) > 1),
        "primary_out_of_pool": dict(sorted(prim_out.items(), key=lambda kv: -kv[1])),
        "primary_in_pool": dict(prim_in),
        "primary_queries": {c: sorted(q for q in reps if review[q]["primary"] == c) for c in sorted(CODES)},
        "secondary_counts_overlapping": dict(Counter(review[q]["secondary"] for q in reps if review[q]["secondary"])),
        "confidence": dict(Counter(review[q]["confidence"] for q in reps)),
        "levers_overlapping": {f"{k}:{w}": v for (k, w), v in sorted(levers.items())},
        "lever_queries": {k: sorted(v) for k, v in sorted(lever_q.items())},
        "unconfirmed_primary_U": sorted(q for q in reps if review[q]["primary"] == "U"),
    }
    if v50_fail is not None:
        cur = set(reps)
        result["v50_compare"] = {"same": len(cur & v50_fail), "new": sorted(cur - v50_fail),
                                 "resolved": sorted(v50_fail - cur)}
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, default=HERE / "diagnosis/failure_stage_table.csv")
    ap.add_argument("--review", type=Path, default=HERE / "diagnosis/failure_review_v67.csv")
    ap.add_argument("--v50", type=Path, default=HERE.parent / "results_v50_vertex_context/failure_diagnosis.csv")
    ap.add_argument("--out", type=Path, default=HERE / "diagnosis/diagnosis_summary.json")
    args = ap.parse_args()
    stage = list(csv.DictReader(open(args.stage, encoding="utf-8")))
    review = list(csv.DictReader(open(args.review, encoding="utf-8")))
    v50 = {r["query_id"] for r in csv.DictReader(open(args.v50, encoding="utf-8-sig")) if r["auto_category"] != "hit"}
    try:
        result = summarize(stage, review, v50)
    except DiagnosisError as exc:
        print(f"집계 거부: {exc}", file=sys.stderr)
        return 3
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("failures", "out_of_pool", "in_pool", "primary_out_of_pool", "primary_in_pool",
                                             "levers_overlapping", "v50_compare")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
