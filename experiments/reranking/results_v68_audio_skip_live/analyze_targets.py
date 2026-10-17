"""v68 대상별 분석 — `aggregate_live.py`의 질의별 표(`analysis/per_query.csv`) + v67 오프라인 목록(결과 전에 정한 대상) → 대상 묶음별 최종 결과.

모델·DB 호출 없음. 판정(RUN_INFO §5-3, 결과 전 확정):
- 질의마다 Hit@10(최종 10위 안 정답) · 최종 순위를 A1·A2·B1·B2에서 읽는다.
- **dev 관측 이득**: A1·A2 모두 10위 밖, B1·B2 모두 10위 안 · **관측 손실**: 그 반대 · 반복(A1≠A2 또는 B1≠B2)에서 Hit@10이 갈리면 **변동**.
- 순위 이득/손실(10위 안끼리의 위치 변화)도 같은 규칙(두 반복 모두 같은 방향, 반복 안에서 위치가 같을 때)으로 따로 센다.
- 원인: aggregate의 쌍별 원인(failure_fallback · policy_skip_order · input_change · model_variation · same)을 그대로 옮긴다.

    venv/bin/python analyze_targets.py --per-query analysis/per_query.csv --v67 <v67 결과 폴더> --out analysis/targets.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
PAIRS = ("A1:B1", "A2:B2", "A1:B2", "A2:B1", "A1:A2", "B1:B2")
GAINED5 = ["t058", "t077", "t078", "t100", "t111"]
INPOOL_FAIL5 = ["q203", "q205", "c705", "c707", "t044"]
RANK_DOWN6 = ["n018", "t010", "t042", "n045", "t033", "c706"]


def rank(v: str) -> Optional[int]:
    return int(float(v)) if v not in ("", None) else None


def hit(r: Optional[int]) -> bool:
    return r is not None and r <= 10


FAIL_STATUS = {"failed", "partial", "budget_exhausted"}


def labels_in(row: Dict[str, str]) -> List[str]:
    return [k for k in ("A1", "A2", "B1", "B2") if f"rank_{k}" in row]


def judge(row: Dict[str, str]) -> Dict[str, Any]:
    """RUN_INFO §5-3 + §6 앞 변경: A2가 없으면(미완료) A 반복 없이 A1 하나로 판정하고 `a_repeat_missing`을 붙인다.
    판정 순서: 실패·부분·예산 소진이 닿은 질의 → failure_touched(이득·손실로 세지 않음) → 반복 변동 → 이득/손실/같음."""
    labs = labels_in(row)
    a_labs = [k for k in labs if k.startswith("A")]
    b_labs = [k for k in labs if k.startswith("B")]
    ranks = {k: rank(row[f"rank_{k}"]) for k in labs}
    failed = [k for k in labs if row.get(f"status_{k}", "") in FAIL_STATUS]
    ha = [hit(ranks[k]) for k in a_labs]
    hb = [hit(ranks[k]) for k in b_labs]
    if failed:
        hit_verdict = "failure_touched"
    elif len(set(ha)) > 1 or len(set(hb)) > 1:
        hit_verdict = "variation"
    elif not ha[0] and hb[0]:
        hit_verdict = "gain"
    elif ha[0] and not hb[0]:
        hit_verdict = "loss"
    else:
        hit_verdict = "same_hit" if ha[0] else "same_miss"
    pos = "none"
    if not failed and all(ha) and all(hb):
        ra = {ranks[k] for k in a_labs}
        rb = {ranks[k] for k in b_labs}
        if len(ra) > 1 or len(rb) > 1:
            pos = "variation"
        elif min(rb) < min(ra):
            pos = "up"
        elif min(rb) > min(ra):
            pos = "down"
    return {"hit": hit_verdict, "position": pos, "ranks": ranks, "failed_in": failed,
            "a_repeat_missing": len(a_labs) < 2}


def summarize(rows: List[Dict[str, str]], ids: List[str], stypes: Dict[str, str] | None = None) -> Dict[str, Any]:
    by = {r["query_id"]: r for r in rows}
    missing = [q for q in ids if q not in by]
    if missing:
        raise SystemExit(f"대상 질의가 집계표에 없다: {missing}")
    labs = labels_in(rows[0])
    out = {"queries": len(ids), "hit@10": {k: sum(hit(rank(by[q][f"rank_{k}"])) for q in ids) for k in labs},
           "verdict": dict(Counter(judge(by[q])["hit"] for q in ids)),
           "position": dict(Counter(judge(by[q])["position"] for q in ids if judge(by[q])["position"] != "none")),
           **{f"cause_{p}": dict(Counter(by[q].get(f"cause_{p}", "") for q in ids)) for p in PAIRS if f"cause_{p}" in rows[0]},
           "per_query": {q: {**judge(by[q]), "cand": {k: by[q][f"cand_rank_{k}"] for k in ("A1", "B1")},
                             "status": {k: by[q][f"status_{k}"] for k in labs},
                             **{f"cause_{p}": by[q].get(f"cause_{p}", "") for p in PAIRS if f"cause_{p}" in rows[0]},
                             **({"sound_type": stypes.get(q, "")} if stypes is not None else {})}
                         for q in ids}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=Path, default=HERE / "analysis/per_query.csv")
    ap.add_argument("--v67", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "analysis/targets.json")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.per_query, encoding="utf-8")))
    if len(rows) != 176 or len({r["query_id"] for r in rows}) != 176:
        raise SystemExit("집계표 질의 수 이상")
    ab = json.loads((args.v67 / "analysis/summary_ab.json").read_text(encoding="utf-8"))["configs"]["B1"]
    changed70 = ab["v66_hits_with_changed_inputs"]
    if len(changed70) != 70:
        raise SystemExit("v67 입력 변경 목록이 70이 아니다")
    if sorted(ab["gained"]) != sorted(GAINED5) or sorted(ab["rank_down_in_pool"]) != sorted(RANK_DOWN6):
        raise SystemExit("v67 오프라인 목록과 고정 목록이 다르다")
    stypes = {r["query_id"]: (r["sound_type"] or "(소리 경로 안 돎)")
              for r in csv.DictReader(open(args.v67 / "analysis/sound_types.csv", encoding="utf-8"))}
    all_ids = [r["query_id"] for r in rows]
    result: Dict[str, Any] = {
        "all": summarize(rows, all_ids),
        "gained5": summarize(rows, GAINED5, stypes),
        "inpool_fail5": summarize(rows, INPOOL_FAIL5, stypes),
        "changed_inputs70": summarize(rows, changed70),
        "unchanged_inputs_others": summarize(rows, [q for q in all_ids if q not in set(changed70)]),
        "rank_down6": summarize(rows, RANK_DOWN6, stypes),
        "by_sound_type": {t: summarize(rows, [q for q in all_ids if stypes[q] == t])
                          for t in sorted(set(stypes.values()))},
    }
    for v in result["by_sound_type"].values():
        v.pop("per_query")
    for k in ("all", "changed_inputs70", "unchanged_inputs_others"):
        pq = result[k].pop("per_query")
        result[k]["gain_ids"] = sorted(q for q, d in pq.items() if d["hit"] == "gain")
        result[k]["loss_ids"] = sorted(q for q, d in pq.items() if d["hit"] == "loss")
        result[k]["variation_ids"] = sorted(q for q, d in pq.items() if d["hit"] == "variation")
        result[k]["failure_touched_ids"] = sorted(q for q, d in pq.items() if d["hit"] == "failure_touched")
        result[k]["position_up_ids"] = sorted(q for q, d in pq.items() if d["position"] == "up")
        result[k]["position_down_ids"] = sorted(q for q, d in pq.items() if d["position"] == "down")
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a = result["all"]
    print(json.dumps({"hit@10": a["hit@10"], "verdict": a["verdict"], "gain": a["gain_ids"], "loss": a["loss_ids"],
                      "variation": a["variation_ids"],
                      "failure_touched": a["failure_touched_ids"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
