"""v69 시간대별 분석 — `aggregate_live.py`의 질의별 표(A3·B3 + v68 A1·B1·B2) → 직접 비교 쌍·시간대별 판정·대상 묶음.

모델·DB 호출 없음. **첫 Vertex 호출 전에 작성했다**(RUN_INFO §5). 판정 기준값(10위 경계)은 v68과 같다.

시간대를 섞지 않는다 — 다른 날 실행의 순위를 평균내지 않고, 각 시간대 안에서만 판정한 뒤 나란히 놓는다.
- **v69 직접 쌍**(같은 시간대, A3 → B3): 질의마다 한쪽이라도 실패·부분·예산 소진 → `failure_touched`(이득·손실로 세지 않음),
  아니면 A3 밖·B3 안 = `gain`, 반대 = `loss`, 같으면 `same_hit`/`same_miss`. 둘 다 10위 안이면 위치 `up`/`down`/`same`.
- **v68 쌍**(A1 → B1·B2, v68 RUN_INFO §5-3 + 결과 뒤 변경 2): 실패 우선 → B1↔B2가 갈리면 `variation` → 이득/손실/같음.
- **두 시간대 나란히**(서술용 — 합치지 않는다): 두 시간대 판정이 같은 방향(`gain`·`gain`)인 질의, 뒤집힌 질의
  (`gain`↔`loss`), 한 시간대에서만 바뀐 질의를 따로 센다.
- **같은 구성 반복**: A1↔A3(다른 시간대 A 반복), B1↔B2(같은 시간대), B1↔B3·B2↔B3(다른 시간대). aggregate의 쌍별 원인
  (same · failure_fallback · input_change · model_variation)을 그대로 옮기고, 실패 없는 정답 위치 변화를 센다.

    venv/bin/python analyze_periods.py --per-query analysis/per_query.csv --v67 <v67 결과 폴더> --out analysis/targets.json
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
LABELS = ("A3", "B3", "A1", "B1", "B2")
REPEAT_PAIRS = ("A1:A3", "B1:B2", "B1:B3", "B2:B3")
FAIL_STATUS = {"failed", "partial", "budget_exhausted"}

# 결과 전에 정한 대상 목록(v67 오프라인 · v68 결과에서)
TARGETS: Dict[str, List[str]] = {
    "v68_gains4": ["t058", "t078", "t100", "c707"],
    "v68_loss1": ["q215"],
    "v68_failure_touched3": ["t107", "q107", "n019"],
    "gained5": ["t058", "t077", "t078", "t100", "t111"],
    "inpool_fail5": ["q203", "q205", "c705", "c707", "t044"],
    "rank_down6": ["n018", "t010", "t042", "n045", "t033", "c706"],
    "v68_hit1_lost5": ["q101", "q218", "m203", "n020", "n061"],
    "v68_hit1_gained4": ["n031", "n082", "t026", "t100"],
    "v68_top10_down9": ["m203", "n018", "n020", "n061", "n097", "q101", "q218", "q312", "t042"],
    "v68_top10_up5": ["m201", "n031", "n057", "n082", "t026"],
    "risk_n018_t042": ["n018", "t042"],
}


def rank(v: str) -> Optional[int]:
    return int(float(v)) if v not in ("", None) else None


def hit(r: Optional[int]) -> bool:
    return r is not None and r <= 10


def failed(row: Dict[str, str], label: str) -> bool:
    return row[f"status_{label}"] in FAIL_STATUS


def verdict_v69(row: Dict[str, str]) -> Dict[str, Any]:
    ra, rb = rank(row["rank_A3"]), rank(row["rank_B3"])
    if failed(row, "A3") or failed(row, "B3"):
        v = "failure_touched"
    elif not hit(ra) and hit(rb):
        v = "gain"
    elif hit(ra) and not hit(rb):
        v = "loss"
    else:
        v = "same_hit" if hit(ra) else "same_miss"
    pos = "none"
    if v == "same_hit":
        pos = "up" if rb < ra else ("down" if rb > ra else "same")
    h1 = "none"
    if v != "failure_touched":
        h1 = "gain" if (ra != 1 and rb == 1) else ("loss" if (ra == 1 and rb != 1) else "same")
    return {"hit": v, "position": pos, "hit1": h1}


def verdict_v68(row: Dict[str, str]) -> Dict[str, Any]:
    ra = rank(row["rank_A1"])
    rbs = [rank(row["rank_B1"]), rank(row["rank_B2"])]
    if any(failed(row, k) for k in ("A1", "B1", "B2")):
        v = "failure_touched"
    elif hit(rbs[0]) != hit(rbs[1]):
        v = "variation"
    elif not hit(ra) and hit(rbs[0]):
        v = "gain"
    elif hit(ra) and not hit(rbs[0]):
        v = "loss"
    else:
        v = "same_hit" if hit(ra) else "same_miss"
    pos = "none"
    if v == "same_hit":
        if rbs[0] != rbs[1]:
            pos = "variation"
        else:
            pos = "up" if rbs[0] < ra else ("down" if rbs[0] > ra else "same")
    return {"hit": v, "position": pos}


def side_by_side(v68: str, v69: str) -> str:
    if "failure_touched" in (v68, v69) or v68 == "variation":
        return "undetermined"
    changed68, changed69 = v68 in ("gain", "loss"), v69 in ("gain", "loss")
    if changed68 and changed69:
        return "same_direction" if v68 == v69 else "reversed"
    if changed68:
        return "v68_only"
    if changed69:
        return "v69_only"
    return "no_change_both"


def per_query(row: Dict[str, str], stypes: Dict[str, str]) -> Dict[str, Any]:
    j69, j68 = verdict_v69(row), verdict_v68(row)
    out = {"ranks": {k: rank(row[f"rank_{k}"]) for k in LABELS},
           "status": {k: row[f"status_{k}"] for k in LABELS},
           "cand_rank": {k: rank(row[f"cand_rank_{k}"]) for k in ("A3", "B3")},
           "v69": j69, "v68": j68, "side_by_side": side_by_side(j68["hit"], j69["hit"]),
           "cause_A3:B3": row.get("cause_A3:B3", ""),
           "input_change_A3:B3": row.get("input_change_A3:B3", ""),
           **{f"cause_{p}": row.get(f"cause_{p}", "") for p in REPEAT_PAIRS}}
    if stypes:
        out["sound_type"] = stypes.get(row["query_id"], "")
    return out


def summarize(rows: Dict[str, Dict[str, str]], ids: List[str], stypes: Dict[str, str], keep: bool = True) -> Dict[str, Any]:
    missing = [q for q in ids if q not in rows]
    if missing:
        raise SystemExit(f"대상 질의가 집계표에 없다: {missing}")
    pq = {q: per_query(rows[q], stypes) for q in ids}
    out: Dict[str, Any] = {
        "queries": len(ids),
        "hit@10": {k: sum(hit(pq[q]["ranks"][k]) for q in ids) for k in LABELS},
        "hit@1": {k: sum(1 for q in ids if pq[q]["ranks"][k] == 1) for k in LABELS},
        "v69_verdict": dict(Counter(pq[q]["v69"]["hit"] for q in ids)),
        "v69_position": dict(Counter(pq[q]["v69"]["position"] for q in ids if pq[q]["v69"]["position"] != "none")),
        "v69_hit1": dict(Counter(pq[q]["v69"]["hit1"] for q in ids if pq[q]["v69"]["hit1"] != "none")),
        "v68_verdict": dict(Counter(pq[q]["v68"]["hit"] for q in ids)),
        "side_by_side": dict(Counter(pq[q]["side_by_side"] for q in ids)),
    }
    if keep:
        out["per_query"] = pq
    else:
        for key, pick in (("v69_gain", lambda d: d["v69"]["hit"] == "gain"), ("v69_loss", lambda d: d["v69"]["hit"] == "loss"),
                          ("v69_failure_touched", lambda d: d["v69"]["hit"] == "failure_touched"),
                          ("v69_position_up", lambda d: d["v69"]["position"] == "up"),
                          ("v69_position_down", lambda d: d["v69"]["position"] == "down"),
                          ("v69_hit1_gain", lambda d: d["v69"]["hit1"] == "gain"),
                          ("v69_hit1_loss", lambda d: d["v69"]["hit1"] == "loss"),
                          ("v68_gain", lambda d: d["v68"]["hit"] == "gain"), ("v68_loss", lambda d: d["v68"]["hit"] == "loss"),
                          ("both_periods_same_direction", lambda d: d["side_by_side"] == "same_direction"),
                          ("reversed", lambda d: d["side_by_side"] == "reversed"),
                          ("v68_only", lambda d: d["side_by_side"] == "v68_only"),
                          ("v69_only", lambda d: d["side_by_side"] == "v69_only"),
                          ("undetermined", lambda d: d["side_by_side"] == "undetermined")):
            out[f"{key}_ids"] = sorted(q for q, d in pq.items() if pick(d))
    return out


def repeats(rows: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    out = {}
    for p in REPEAT_PAIRS:
        x, y = p.split(":")
        same10 = sum(1 for r in rows.values() if r[f"final10_{x}"] == r[f"final10_{y}"])
        causes = Counter(r[f"cause_{p}"] for r in rows.values() if r[f"cause_{p}"] != "same")
        moved = [q for q, r in rows.items() if r[f"rank_{x}"] != r[f"rank_{y}"]]
        moved_clean = [q for q in moved if not (failed(rows[q], x) or failed(rows[q], y))]
        hit_flip_clean = [q for q in moved_clean if hit(rank(rows[q][f"rank_{x}"])) != hit(rank(rows[q][f"rank_{y}"]))]
        out[p] = {"final_top10_identical": same10, "order_diff_by_cause": dict(sorted(causes.items())),
                  "gold_rank_changed": sorted(moved), "gold_rank_changed_without_failure": sorted(moved_clean),
                  "hit10_flip_without_failure": sorted(hit_flip_clean)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=Path, default=HERE / "analysis/per_query.csv")
    ap.add_argument("--v67", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "analysis/targets.json")
    args = ap.parse_args()
    table = list(csv.DictReader(open(args.per_query, encoding="utf-8")))
    rows = {r["query_id"]: r for r in table}
    if len(table) != 176 or len(rows) != 176:
        raise SystemExit("집계표 질의 수 이상")
    for k in LABELS:
        if f"rank_{k}" not in table[0]:
            raise SystemExit(f"집계표에 {k} 열이 없다")
    ab = json.loads((args.v67 / "analysis/summary_ab.json").read_text(encoding="utf-8"))["configs"]["B1"]
    changed70 = ab["v66_hits_with_changed_inputs"]
    if len(changed70) != 70 or sorted(ab["gained"]) != sorted(TARGETS["gained5"]) \
            or sorted(ab["rank_down_in_pool"]) != sorted(TARGETS["rank_down6"]):
        raise SystemExit("v67 오프라인 목록과 고정 목록이 다르다")
    stypes = {r["query_id"]: (r["sound_type"] or "(소리 경로 안 돎)")
              for r in csv.DictReader(open(args.v67 / "analysis/sound_types.csv", encoding="utf-8"))}
    all_ids = [r["query_id"] for r in table]
    result: Dict[str, Any] = {
        "rules": "analyze_periods.py 머리말(첫 Vertex 호출 전 작성)",
        "all": summarize(rows, all_ids, stypes, keep=False),
        "changed_inputs70": summarize(rows, changed70, stypes, keep=False),
        "unchanged_inputs_others": summarize(rows, [q for q in all_ids if q not in set(changed70)], stypes, keep=False),
        **{name: summarize(rows, ids, stypes) for name, ids in TARGETS.items()},
        "by_sound_type": {t: summarize(rows, [q for q in all_ids if stypes[q] == t], stypes, keep=False)
                          for t in sorted(set(stypes.values()))},
        "repeats": repeats(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a = result["all"]
    print(json.dumps({"hit@10": a["hit@10"], "hit@1": a["hit@1"], "v69_verdict": a["v69_verdict"],
                      "v69_gain": a["v69_gain_ids"], "v69_loss": a["v69_loss_ids"],
                      "v69_failure_touched": a["v69_failure_touched_ids"], "side_by_side": a["side_by_side"],
                      "same_direction": a["both_periods_same_direction_ids"], "reversed": a["reversed_ids"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
