"""v74 남은 실패 55(후보 밖 51 + 후보 안 4) 진단표 — 작품명 ON(work_on) 추적으로 단계를 다시 계산하고 v67 사람 판정을 이어받는다. 모델·DB 호출 없음.

규칙(RUN_INFO §4, 결과 전 고정):
- 대표 정답 곡 = 가장 멀리 간 곡(후보 안 > 보정 뒤 절단 > 융합 절단 > 경로 밖, 같으면 후보 순위 → 전체 깊이 가산 뒤 순위) — v67과 같은 정의.
- 자동 단계: 후보 안 = K · 융합 절단/보정 뒤 절단 = F · 경로 밖이고 전체 깊이 가산 뒤 ≤30 = P · 그 밖 = 사람 판정 대상.
- 주 원인: 자동 단계가 P·F·K면 그 코드. 아니면 v67 사람 판정(A·B·R·L)을 이어받는다(`inherited`).
  v74 대표 곡의 자동 단계가 v67 대표 곡의 자동 단계와 다르거나, 작품명 해석이 그 질의의 입력을 바꿨으면 `review_needed`로 표시하고
  `--overrides`(사람 재검토 결과 csv: query_id,primary,secondary,confidence,evidence)가 있어야 끝난다.
- 겹치는 꼬리표(합계에 더하지 않음): v67 지렛대 + `work_input_changed`(작품명 ON이 리랭커 입력을 바꿈) + `work_status=<상태>` +
  `pool_in_work_skip`(결합 조건에서 후보 안) + `audio_path_ran`.

    venv/bin/python summarize_diagnosis.py [--overrides diagnosis/review_overrides_v74.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V67 = ROOT / "experiments/reranking/results_v67_candidate_recall_diagnosis"
STAGE_ORDER = {"후보 안": 0, "보정 뒤 절단": 1, "융합 절단": 2, "경로 밖": 3}
CODES = {"A", "B", "R", "P", "F", "K", "L", "U"}


def read_csv(path: Path) -> list:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def auto_code(t: dict) -> str:
    if t["stage"] == "후보 안":
        return "K"
    if t["stage"] in ("융합 절단", "보정 뒤 절단"):
        return "F"
    if t["boosted_rank_deep"] and int(t["boosted_rank_deep"]) <= 30:
        return "P"
    return "human"


def rep_of(rows: list) -> dict:
    return min(rows, key=lambda x: (STAGE_ORDER[x["stage"]], int(x["pool_rank"] or 999), int(x["boosted_rank_deep"] or 99999)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", type=Path, default=HERE / "diagnosis/review_overrides_v74.csv")
    args = ap.parse_args()
    fails = {r["query_id"]: r for r in read_csv(HERE / "diagnosis/baseline_failures.csv") if r["kind"] != "hit"}
    trace_on: dict = {}
    for r in read_csv(HERE / "trace/trace_work_on.csv"):
        trace_on.setdefault(r["query_id"], []).append(r)
    trace_skip: dict = {}
    for r in read_csv(HERE / "trace/trace_work_skip.csv"):
        trace_skip.setdefault(r["query_id"], []).append(r)
    v67_review = {r["query_id"]: r for r in read_csv(V67 / "diagnosis/failure_review_v67.csv")}
    v67_stage = {r["query_id"]: r for r in read_csv(V67 / "diagnosis/failure_stage_table.csv") if r["representative"] == "1"}
    cfg = {r["query_id"]: r for r in read_csv(HERE / "analysis/per_query_configs.csv")}
    overrides = {r["query_id"]: r for r in read_csv(args.overrides)} if args.overrides.exists() else {}
    problems = []
    out = []
    for qid, f in fails.items():
        rows = trace_on.get(qid)
        if not rows:
            problems.append(f"{qid}: 추적 없음")
            continue
        rep = rep_of(rows)
        auto = auto_code(rep)
        in_pool = f["kind"] == "in_pool_fail"
        if (auto == "K") != in_pool:
            problems.append(f"{qid}: 추적 단계 {rep['stage']} ↔ 기준 표 {f['kind']}")
        if in_pool and str(rep["pool_rank"]) != f["pool_rank"]:
            problems.append(f"{qid}: 추적 후보 순위 {rep['pool_rank']} ≠ 기준 {f['pool_rank']}")
        v67s = v67_stage.get(qid, {})
        v67_auto = ""
        if v67s:
            a = v67s.get("auto_stage", "")
            v67_auto = a[:1] if a[:1] in CODES else "human"
        v67r = v67_review.get(qid, {})
        work_changed = cfg[qid]["input_changed_O1"] == "1"  # work_on ↔ off_on 입력 차이(= 작품명 ON이 입력을 바꿈)
        review_needed = bool(v67s) and (auto != v67_auto) or (work_changed and not in_pool)
        if not v67s:
            review_needed = True
        if auto in ("P", "F", "K"):
            primary, source = auto, "auto"
        else:
            primary, source = v67r.get("primary", ""), "inherited_v67"
        secondary = v67r.get("secondary", "")
        confidence = v67r.get("confidence", "")
        if qid in overrides:
            o = overrides[qid]
            primary, secondary, confidence, source = o["primary"], o.get("secondary", ""), o.get("confidence", ""), "reviewed_v74"
        elif review_needed and source != "auto":
            problems.append(f"{qid}: 재검토 필요(자동 {auto} · v67 {v67_auto} · 작품명 입력 변화 {work_changed}) — overrides 없음")
        if primary not in CODES:
            problems.append(f"{qid}: 주 원인 코드 {primary!r}")
        skip_rows = trace_skip.get(qid, [])
        skip_rep = rep_of(skip_rows) if skip_rows else {}
        tags = [t for t in (v67r.get("lever", "") or "").split("|") if t]
        if work_changed:
            tags.append("work_input_changed")
        for tag in ("work_mention_digit_boundary", "work_aux_cut"):
            if qid in overrides and tag in overrides[qid].get("evidence", ""):
                tags.append(tag)
        if rep.get("work_status") not in ("", "(none)", "not_triggered"):
            tags.append(f"work_status={rep['work_status']}")
        if skip_rep.get("stage") == "후보 안":
            tags.append("pool_in_work_skip")
        if "audio:" in rep.get("path_lengths_now", ""):
            tags.append("audio_path_ran")
        best_deep = [int(x.split(":")[1]) for x in rep["paths_deep"].split("|")
                     if x and x.split(":")[0] in ("text_hybrid", "image", "audio")]
        out.append({
            "set": f["set"], "query_id": qid, "kind": f["kind"], "relevant_count": f["relevant_count"],
            "rep_song": rep["song_id"], "stage_v74": rep["stage"], "pool_rank_v74": rep["pool_rank"],
            "auto_v74": auto, "auto_v67": v67_auto, "stage_changed_vs_v67": int(bool(v67s) and auto != v67_auto),
            "primary": primary, "primary_source": source, "secondary": secondary, "confidence": confidence,
            "tags": "|".join(tags),
            "paths_now": rep["paths_now"], "paths_deep": rep["paths_deep"], "best_main_rank_deep": min(best_deep) if best_deep else "",
            "fused_rank_now": rep["fused_rank_now"], "boosted_rank_now": rep["boosted_rank_now"],
            "fused_rank_deep": rep["fused_rank_deep"], "boosted_rank_deep": rep["boosted_rank_deep"],
            "work_status": rep.get("work_status", ""),
            "work_skip_stage": skip_rep.get("stage", ""), "work_skip_pool_rank": skip_rep.get("pool_rank", ""),
            "v67_primary": v67r.get("primary", ""),
        })
    outside = [r for r in out if r["kind"] == "out_of_pool"]
    counts = Counter(r["primary"] for r in outside)
    if sum(counts.values()) != 51 or len(outside) != 51:
        problems.append(f"후보 밖 주 원인 합계 {sum(counts.values())} ≠ 51")
    if sum(1 for r in out if r["kind"] == "in_pool_fail") != 4:
        problems.append("후보 안 실패 ≠ 4")
    tag_counts = Counter(t for r in outside for t in r["tags"].split("|") if t)
    summary = {
        "failures": len(out), "out_of_pool": len(outside), "in_pool": len(out) - len(outside),
        "primary_counts_out_of_pool": dict(counts.most_common()),
        "primary_by_code": {c: sorted(r["query_id"] for r in outside if r["primary"] == c) for c in sorted(counts)},
        "primary_source": dict(Counter(r["primary_source"] for r in outside)),
        "stage_changed_vs_v67": sorted(r["query_id"] for r in out if r["stage_changed_vs_v67"]),
        "primary_changed_vs_v67": sorted(r["query_id"] for r in outside if r["v67_primary"] and r["primary"] != r["v67_primary"]),
        "overlap_tags_out_of_pool(겹침 — 합계에 더하지 않음)": dict(tag_counts.most_common()),
        "in_pool_fail": {r["query_id"]: r["pool_rank_v74"] for r in out if r["kind"] == "in_pool_fail"},
        "out_of_pool_now_in_pool_under_work_skip": sorted(r["query_id"] for r in outside if r["work_skip_stage"] == "후보 안"),
        "problems": problems,
    }
    (HERE / "diagnosis").mkdir(exist_ok=True)
    with open(HERE / "diagnosis/failure_review_v74.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (HERE / "diagnosis/diagnosis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                                          encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
