"""5,143곡 기준선(rep1·rep2) 집계 + 같은 질의의 3,010곡 기록과 질의별 대조.

3,010곡 쪽: 최종 순위는 results_v35_rare_verify_off(같은 설정 1회), 후보 순위는
results_v37_fame_supplement/pool_compare.csv의 off 열(ref_3010_pool_ranks.csv로 옮겨 둠).
코퍼스 교체 전후이므로 "설정 비교"가 아니다 — 같은 질의가 어디로 갔는지만 본다.

  venv/bin/python experiments/reranking/results_v39_corpus5143_baseline/compare.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

OUT = Path(__file__).resolve().parent
REF = OUT.parent / "results_v35_rare_verify_off"
SETS = ("v06", "v09")
REPS = ("rep1", "rep2")


def _top(path: Path) -> Dict[str, dict]:
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        rel = set(r["relevant_ids"].split("|"))
        def rank(col: str) -> Optional[int]:
            ids = [x for x in r[col].split("|") if x]
            return next((i + 1 for i, x in enumerate(ids[:10]) if x in rel), None)
        rows[r["query_id"]] = {"base": rank("baseline_top_ids"), "final": rank("rerank_top_ids"),
                               "ids": [x for x in r["rerank_top_ids"].split("|") if x][:10]}
    return rows


def _metrics(rows: Dict[str, dict], key: str = "final") -> dict:
    n = len(rows)
    rk = [r[key] for r in rows.values()]
    return {"n": n, "h1": sum(x == 1 for x in rk), "h5": sum(bool(x) and x <= 5 for x in rk),
            "h10": sum(bool(x) for x in rk), "mrr": sum(1 / x for x in rk if x) / n}


def main() -> None:
    old_ids = set()
    with open("data/all_songs_3010_backup.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                old_ids.add(str(json.loads(line)["song_id"]))
    pool3010 = {(r["set"], r["query_id"]): r["pool_rank_3010"]
                for r in csv.DictReader(open(OUT / "ref_3010_pool_ranks.csv", encoding="utf-8"))}
    table: List[dict] = []
    total = {k: {"n": 0, "h1": 0, "h5": 0, "h10": 0, "mrr_sum": 0.0, "cand": 0} for k in ("3010",) + REPS}
    print("| 세트 | 실행 | n | 후보 30 | Hit@1 | Hit@5 | Hit@10 | MRR@10 |")
    print("|---|---|---|---|---|---|---|---|")
    for s in SETS:
        ref = _top(REF / f"{s}_dev/search_eval_dev_top10.csv")
        runs = {rep: _top(OUT / rep / f"{s}_dev/search_eval_dev_top10.csv") for rep in REPS}
        drop = {r["query_id"]: r for r in csv.DictReader(open(OUT / f"candidate_drop_{s}_dev.csv", encoding="utf-8"))}
        assert set(ref) == set(runs["rep1"]) == set(runs["rep2"]) == set(drop), s
        cand5143 = sum(d["stage"] == "후보 안" for d in drop.values())
        cand3010 = sum(pool3010[(s, q)].strip().isdigit() for q in ref)
        for name, rows, cand in (("3010", ref, cand3010), ("rep1", runs["rep1"], cand5143), ("rep2", runs["rep2"], cand5143)):
            m = _metrics(rows)
            label = "3,010곡(v35)" if name == "3010" else f"5,143곡 {name}"
            print(f"| {s} dev | {label} | {m['n']} | {cand} | {m['h1']} | {m['h5']} | {m['h10']} | {m['mrr']:.3f} |")
            t = total[name]
            for k in ("n", "h1", "h5", "h10"):
                t[k] += m[k]
            t["mrr_sum"] += m["mrr"] * m["n"]
            t["cand"] += cand
        for q in ref:
            d = drop[q]
            table.append({
                "set": s, "query_id": q,
                "pool_3010": pool3010[(s, q)], "pool_5143": d["pool_rank"], "stage_5143": d["stage"],
                "text_rank": d["text_hybrid_rank"], "image_rank": d["image_rank"], "audio_rank": d["audio_rank"],
                "pool_rank_deep": d["pool_rank_deep"],
                "final_3010": ref[q]["final"] or "", "final_rep1": runs["rep1"][q]["final"] or "",
                "final_rep2": runs["rep2"][q]["final"] or "",
                "new_songs_in_top10_rep1": sum(x not in old_ids for x in runs["rep1"][q]["ids"]),
                "top10_same_rep1_rep2": int(runs["rep1"][q]["ids"] == runs["rep2"][q]["ids"]),
            })
    for name in ("3010",) + REPS:
        t = total[name]
        label = "3,010곡(v35)" if name == "3010" else f"5,143곡 {name}"
        print(f"| **합산** | {label} | {t['n']} | {t['cand']} | {t['h1']} | {t['h5']} | {t['h10']} | {t['mrr_sum']/t['n']:.3f} |")
    with open(OUT / "per_query_3010_vs_5143.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    def r(x) -> int:
        return int(x) if str(x).strip().isdigit() else 0
    print()
    print("rep1 ↔ rep2: 정답 순위 다른 질의", [(t["query_id"], t["final_rep1"] or "밖", t["final_rep2"] or "밖")
                                     for t in table if t["final_rep1"] != t["final_rep2"]],
          "· top-10 구성 다른 질의", sum(not t["top10_same_rep1_rep2"] for t in table), "/", len(table))
    both_in = lambda t: bool(r(t["final_rep1"])) and bool(r(t["final_rep2"]))
    both_out = lambda t: not r(t["final_rep1"]) and not r(t["final_rep2"])
    lost = [t for t in table if r(t["final_3010"]) and both_out(t)]
    gained = [t for t in table if not r(t["final_3010"]) and both_in(t)]
    print(f"10위 안 → 밖 (두 실행 모두): {len(lost)}건")
    for t in lost:
        print(f"   {t['query_id']} 3010: 후보 {t['pool_3010'] or '밖'}·최종 {t['final_3010']} → 5143: {t['stage_5143']}"
              f"(후보 {t['pool_5143'] or '밖'}, T{t['text_rank'] or '-'} I{t['image_rank'] or '-'} A{t['audio_rank'] or '-'},"
              f" 깊이 {t['pool_rank_deep'] or '-'}) · top-10 새 곡 {t['new_songs_in_top10_rep1']}")
    print(f"10위 밖 → 안 (두 실행 모두): {len(gained)}건", [(t["query_id"], t["final_rep1"], t["final_rep2"]) for t in gained])
    print("한 실행에서만 갈린 질의:", [(t["query_id"], t["final_3010"] or "밖", t["final_rep1"] or "밖", t["final_rep2"] or "밖")
                              for t in table if not both_in(t) and not both_out(t)])
    cand_lost = [t["query_id"] for t in table if r(t["pool_3010"]) and t["stage_5143"] != "후보 안"]
    cand_gain = [t["query_id"] for t in table if not r(t["pool_3010"]) and t["stage_5143"] == "후보 안"]
    print(f"후보 30: 안 → 밖 {len(cand_lost)}건 {cand_lost} · 밖 → 안 {len(cand_gain)}건 {cand_gain}")
    h1_lost = [(t["query_id"], t["final_rep1"], t["final_rep2"]) for t in table
               if r(t["final_3010"]) == 1 and r(t["final_rep1"]) != 1 and r(t["final_rep2"]) != 1 and both_in(t)]
    print(f"1위 → 2~10위 (두 실행 모두): {len(h1_lost)}건", h1_lost)
    in_stage = [t for t in table if t["stage_5143"] == "후보 안"]
    print("후보 안인데 두 실행 모두 10위 밖:", [t["query_id"] for t in in_stage if both_out(t)])


if __name__ == "__main__":
    main()
