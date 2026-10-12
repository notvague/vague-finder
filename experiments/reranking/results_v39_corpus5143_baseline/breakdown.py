"""3,010곡 → 5,143곡: 같은 질의 116건이 어떻게 움직였는지 층별로 본다(per_query_3010_vs_5143.csv 사용).

  venv/bin/python experiments/reranking/results_v39_corpus5143_baseline/breakdown.py
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import List

OUT = Path(__file__).resolve().parent


def r(x: str) -> int:
    return int(x) if str(x).strip().isdigit() else 0


def main() -> None:
    rows = list(csv.DictReader(open(OUT / "per_query_3010_vs_5143.csv", encoding="utf-8")))
    meta = {q["query_id"]: q for q in json.load(open("docs/eval/queries.json", encoding="utf-8"))["queries"]}
    for t in rows:
        m = meta[t["query_id"]]
        t["modality"], t["tier"], t["category"], t["scope"] = m["modality_focus"], m["tier"], m["category"], m.get("target_scope", "specific")
        a, b = r(t["final_3010"]), r(t["final_rep1"])
        t["move"] = ("계속 밖" if not a and not b else "밖 → 안" if not a else "안 → 밖" if not b
                     else "그대로" if a == b else "올라감" if b < a else "내려감")

    def line(name: str, g: List[dict]) -> str:
        n = len(g)
        c3 = sum(bool(r(t["pool_3010"])) for t in g); c5 = sum(t["stage_5143"] == "후보 안" for t in g)
        f = lambda col, k: sum(0 < r(t[col]) <= k for t in g)
        return (f"| {name} | {n} | {c3} → {c5} | {f('final_3010',1)} → {f('final_rep1',1)} | "
                f"{f('final_3010',5)} → {f('final_rep1',5)} | {f('final_3010',10)} → {f('final_rep1',10)} |")

    print("| 층 | n | 후보 30 | Hit@1 | Hit@5 | Hit@10 |")
    print("|---|---|---|---|---|---|")
    print(line("전체", rows))
    for col in ("set", "modality", "tier", "scope", "category"):
        for k in sorted({t[col] for t in rows}):
            print(line(f"{col}: {k}", [t for t in rows if t[col] == k]))
    print()
    print("최종 순위 이동:", dict(Counter(t["move"] for t in rows)))
    down = [(t["query_id"], r(t["final_3010"]), r(t["final_rep1"])) for t in rows if t["move"] == "내려감"]
    up = [(t["query_id"], r(t["final_3010"]), r(t["final_rep1"])) for t in rows if t["move"] == "올라감"]
    print("  내려감:", down)
    print("  올라감:", up)
    both = [(r(t["pool_3010"]), r(t["pool_5143"])) for t in rows if r(t["pool_3010"]) and r(t["pool_5143"])]
    print(f"후보 순위(양쪽 모두 후보 안 {len(both)}건): 중앙값 {statistics.median(a for a, _ in both)} → {statistics.median(b for _, b in both)}"
          f" · 내려감 {sum(b > a for a, b in both)} · 그대로 {sum(b == a for a, b in both)} · 올라감 {sum(b < a for a, b in both)}")
    fin = [(r(t["final_3010"]), r(t["final_rep1"])) for t in rows if r(t["final_3010"]) and r(t["final_rep1"])]
    print(f"최종 순위(양쪽 모두 10위 안 {len(fin)}건): 평균 {statistics.mean(a for a, _ in fin):.2f} → {statistics.mean(b for _, b in fin):.2f}")
    new = [int(t["new_songs_in_top10_rep1"]) for t in rows]
    print(f"top-10에 든 새 곡: 질의당 평균 {statistics.mean(new):.2f}곡(중앙값 {statistics.median(new)}) · 새 곡이 코퍼스에서 차지하는 비율 {2133/5143:.1%}")
    lost = [t for t in rows if t["move"] == "안 → 밖"]
    print("안 → 밖 9건의 top-10 새 곡 수:", [(t["query_id"], t["new_songs_in_top10_rep1"]) for t in lost])


if __name__ == "__main__":
    main()
