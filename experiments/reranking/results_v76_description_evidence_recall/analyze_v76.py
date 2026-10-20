"""v76 오프라인 후보 A/B 집계 — 모델·DB 호출 없음. 검색이 끝난 기록만 읽는다(정답 라벨은 여기서 처음 붙인다).

입력: 네 실행(`runs/desc_{off,on}_server_rep{1,2}`)의 `v76_queries.jsonl`·`v76_qdrant_calls.jsonl` · 입력 대조 결과
(`compare/*.json`) · v75 질의별 표(`v75 analysis/per_query.csv` — A 최종 순위로 기존 성공 121을 정한다) · 평가 질의 CSV(정답 id).

가짜 리랭커의 최종 순서로 계산한 Hit·MRR은 만들지 않는다. 후보 회수@30(엄격 정답 하나라도)·정답 첫 후보 순위만 센다.

    venv/bin/python analyze_v76.py --v75-per-query <v75>/analysis/per_query.csv --out-dir analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")
EXPECTED = {"v06": 57, "v09": 59, "v10": 60}
RUNS = {"A1": "desc_off_server_rep1", "B1": "desc_on_server_rep1", "B2": "desc_on_server_rep2", "A2": "desc_off_server_rep2"}
# 설계(§1 조사) 때 정답 곡 자료를 읽은 질의 — 얻음·잃음을 따로 센다(RUN_INFO §5-4)
DESIGN_SEEN = {"q115", "q117", "m302", "m303", "m403", "c602", "c603", "n046", "n064", "t009", "t038", "t060", "t073",
               "t075", "t079", "t080", "t092", "t103"}


def load_rows(run: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    run_dir = HERE / "runs" / run
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"완료되지 않은 실행: {run}")
    for s in SETS:
        rows = [json.loads(line) for line in (run_dir / f"{s}_dev" / "v76_queries.jsonl").open(encoding="utf-8")]
        rows = [r for r in rows if r["query_id"] != "_none"]
        if len(rows) != EXPECTED[s]:
            raise SystemExit(f"{run} {s}: 기록 {len(rows)}건 ≠ {EXPECTED[s]}")
        calls = [json.loads(line) for line in (run_dir / f"{s}_dev" / "v76_qdrant_calls.jsonl").open(encoding="utf-8")]
        by_q: Dict[str, List[Dict[str, Any]]] = {}
        for c in calls:
            by_q.setdefault(c["query_id"], []).append(c)
        for r in rows:
            r["set"] = s
            r["calls"] = by_q.get(r["query_id"], [])
            out[r["query_id"]] = r
    if len(out) != 176:
        raise SystemExit(f"{run}: 질의 {len(out)}건 ≠ 176")
    return out


def gold() -> Dict[str, set]:
    out: Dict[str, set] = {}
    for s in SETS:
        with (ROOT / f"experiments/reranking/eval_queries_{s}.csv").open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["split"].strip() == "dev":
                    out[r["query_id"].strip()] = {x.strip() for x in r["relevant_ids"].split("|") if x.strip()}
    return out


def first_gold_rank(ids: List[str], g: set) -> Optional[int]:
    for i, sid in enumerate(ids, start=1):
        if sid in g:
            return i
    return None


def median(xs: List[float]) -> Optional[float]:
    return round(statistics.median(xs), 3) if xs else None


def pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))], 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v75-per-query", required=True)
    ap.add_argument("--out-dir", default=str(HERE / "analysis"))
    args = ap.parse_args()
    runs = {k: load_rows(v) for k, v in RUNS.items()}
    g = gold()
    with open(args.v75_per_query, encoding="utf-8") as fh:
        v75 = {r["qid"]: r for r in csv.DictReader(fh)}
    if len(v75) != 176:
        raise SystemExit("v75 질의별 표가 176건이 아니다")

    def v75_hit(q: str) -> bool:
        ranks = [v75[q][k] for k in ("A1_rank", "A2_rank")]
        hits = {bool(x) and x.isdigit() and int(x) <= 10 for x in ranks}
        if len(hits) != 1:
            raise SystemExit(f"{q}: v75 A1·A2 성공 여부가 다르다")
        return hits.pop()

    cmp_ab = json.loads((HERE / "compare/A1_vs_B1.json").read_text(encoding="utf-8"))
    changed = {f: set(v) for f, v in cmp_ab["diff_queries"].items()}
    rows_out: List[Dict[str, Any]] = []
    for q in sorted(runs["A1"], key=lambda x: (runs["A1"][x]["set"], x)):
        a, b = runs["A1"][q], runs["B1"][q]
        a_ids = [c["id"] for c in a["candidates"]]
        b_ids = [c["id"] for c in b["candidates"]]
        if [c["id"] for c in runs["A2"][q]["candidates"]] != a_ids or [c["id"] for c in runs["B2"][q]["candidates"]] != b_ids:
            raise SystemExit(f"{q}: 반복 쌍의 후보가 다르다 — 결정성 주장 불가(따로 적어야 한다)")
        gq = g[q]
        ra, rb = first_gold_rank(a_ids, gq), first_gold_rank(b_ids, gq)
        desc = b.get("desc") or {}
        entered, displaced = desc.get("entered_ids") or [], desc.get("displaced_ids") or []
        a_ms = [runs[k][q]["search_ms"] for k in ("A1", "A2")]
        b_ms = [runs[k][q]["search_ms"] for k in ("B1", "B2")]
        a_calls = len(a["calls"])
        b_calls = len(b["calls"])
        aux_calls = [c for c in b["calls"] if c.get("kind") == "index" and c.get("filtered") and c.get("hybrid")
                     and "text" in c.get("collection", "")]
        rows_out.append({
            "set": a["set"], "qid": q, "design_seen": q in DESIGN_SEEN, "v75A_hit10": v75_hit(q),
            "desc_status": b.get("desc_status") or "", "evidence_songs": desc.get("evidence_songs"),
            "aux_len": len(desc.get("aux_ids") or []), "entered": "|".join(entered), "displaced": "|".join(displaced),
            "gold_entered": bool(set(entered) & gq), "gold_displaced": bool(set(displaced) & gq),
            "gold_in_aux": bool(set(desc.get("aux_ids") or []) & gq),
            "A_cand_rank": ra or "", "B_cand_rank": rb or "",
            "recall": ("gain" if ra is None and rb is not None else "loss" if ra is not None and rb is None
                       else "both" if ra is not None else "neither"),
            "rank_move": ("" if ra is None or rb is None else "up" if rb < ra else "down" if rb > ra else "same"),
            "cand_set_changed": set(a_ids) != set(b_ids), "cand_order_changed": a_ids != b_ids,
            "cand_scores_changed": q in changed.get("cand_scores", set()),
            "prompt_changed": q in changed.get("prompts", set()),
            "input_changed": any(q in changed.get(f, set()) for f in cmp_ab["fields_compared"] if f != "final"),
            "A_search_ms_mean": round(sum(a_ms) / 2, 3), "B_search_ms_mean": round(sum(b_ms) / 2, 3),
            "delta_ms": round(sum(b_ms) / 2 - sum(a_ms) / 2, 3),
            "stage1_ms": desc.get("stage1_ms"), "stage2_ms": desc.get("stage2_ms"),
            "qdrant_calls_A": a_calls, "qdrant_calls_B": b_calls, "aux_text_calls": len(aux_calls),
            "aux_top_k": sum(int(c.get("top_k") or 0) for c in aux_calls),
        })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    def ids(pred) -> List[str]:
        return [r["qid"] for r in rows_out if pred(r)]

    base = ids(lambda r: r["v75A_hit10"])
    applied = [r for r in rows_out if r["desc_status"] == "applied"]
    status_counts: Dict[str, int] = {}
    for r in rows_out:
        status_counts[r["desc_status"]] = status_counts.get(r["desc_status"], 0) + 1
    summary = {
        "recall_at_30": {"A": sum(r["A_cand_rank"] != "" for r in rows_out),
                         "B": sum(r["B_cand_rank"] != "" for r in rows_out)},
        "gained": ids(lambda r: r["recall"] == "gain"), "lost": ids(lambda r: r["recall"] == "loss"),
        "gained_design_seen": ids(lambda r: r["recall"] == "gain" and r["design_seen"]),
        "gained_not_design_seen": ids(lambda r: r["recall"] == "gain" and not r["design_seen"]),
        "v75A_successes": len(base),
        "v75A_success_dropped_from_candidates": ids(lambda r: r["v75A_hit10"] and r["recall"] == "loss"),
        "v75A_success_rank_down": {r["qid"]: [r["A_cand_rank"], r["B_cand_rank"]] for r in rows_out
                                   if r["v75A_hit10"] and r["rank_move"] == "down"},
        "v75A_success_rank_up": {r["qid"]: [r["A_cand_rank"], r["B_cand_rank"]] for r in rows_out
                                 if r["v75A_hit10"] and r["rank_move"] == "up"},
        "all_rank_down": len(ids(lambda r: r["rank_move"] == "down")),
        "all_rank_up": len(ids(lambda r: r["rank_move"] == "up")),
        "cand_set_changed": len(ids(lambda r: r["cand_set_changed"])),
        "cand_order_changed": len(ids(lambda r: r["cand_order_changed"])),
        "cand_scores_changed": len(ids(lambda r: r["cand_scores_changed"])),
        "input_changed": len(ids(lambda r: r["input_changed"])),
        "input_changed_v75A_success": len(ids(lambda r: r["input_changed"] and r["v75A_hit10"])),
        "prompt_changed": len(ids(lambda r: r["prompt_changed"])),
        "prompt_changed_v75A_success": len(ids(lambda r: r["prompt_changed"] and r["v75A_hit10"])),
        "desc_status_counts": dict(sorted(status_counts.items())),
        "aux_len_distribution": {str(n): sum(1 for r in applied if r["aux_len"] == n) for n in range(0, 11)},
        "entered_total": sum(len(r["entered"].split("|")) for r in rows_out if r["entered"]),
        "queries_with_entrants": len(ids(lambda r: bool(r["entered"]))),
        "gold_entered_queries": ids(lambda r: r["gold_entered"]),
        "gold_displaced_queries": ids(lambda r: r["gold_displaced"]),
        "gold_in_aux_list_queries": ids(lambda r: r["gold_in_aux"]),
        "evidence_songs_median_applied": median([r["evidence_songs"] for r in applied]),
        "latency": {
            "scope": "평가 하네스 라우터 search 시간(search_ms, 리랭킹 제외) · 구성별 두 반복 평균의 같은 질의 짝 차 · 순서 A1→B1→B2→A2",
            "A_median_ms": median([r["A_search_ms_mean"] for r in rows_out]),
            "B_median_ms": median([r["B_search_ms_mean"] for r in rows_out]),
            "delta_median_all": median([r["delta_ms"] for r in rows_out]),
            "delta_median_applied": median([r["delta_ms"] for r in applied]),
            "delta_p90_applied": pct([r["delta_ms"] for r in applied], 0.9),
            "stage1_median_ms": median([r["stage1_ms"] for r in applied if r["stage1_ms"] is not None]),
            "stage1_max_ms": max([r["stage1_ms"] for r in applied if r["stage1_ms"] is not None], default=None),
            "stage2_median_ms": median([r["stage2_ms"] for r in applied if r["stage2_ms"] is not None]),
            "stage2_p90_ms": pct([r["stage2_ms"] for r in applied if r["stage2_ms"] is not None], 0.9),
        },
        "qdrant_logical_calls": {"A": sum(r["qdrant_calls_A"] for r in rows_out),
                                 "B": sum(r["qdrant_calls_B"] for r in rows_out),
                                 "aux_text_calls": sum(r["aux_text_calls"] for r in rows_out),
                                 "aux_top_k_sum": sum(r["aux_top_k"] for r in rows_out)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("recall_at_30", "gained", "lost", "v75A_success_dropped_from_candidates",
                                              "prompt_changed", "prompt_changed_v75A_success", "desc_status_counts",
                                              "queries_with_entrants", "gold_entered_queries", "gold_displaced_queries")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
