"""후보 단계 측정(리랭킹 끔)을 나란히 요약한다 — 후보 30 안 · 리랭킹 전 Hit@10과 질의별 후보 순위 변화. 첫 인자가 대조다.

detail CSV를 읽는다(커밋하지 않는 파일이라 측정한 자리에서만 돈다). 결과 표는 candidates.csv로 남긴다.

  venv/bin/python <이 파일> ctx0_scale1 ctx0.5_scale1 ctx0_scale0 ctx0.5_scale0
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SETS = ("v06", "v09", "v10")
CACHES = HERE.parent / "results_v46_main_after_context"


def context_queries() -> set[tuple[str, str]]:
    out = set()
    for name in SETS:
        entries = json.loads((CACHES / f"analysis_cache_{name}_dev_ctx.json").read_text(encoding="utf-8"))["entries"]
        out |= {(name, qid) for qid, entry in entries.items() if entry["analysis"].get("context_clues")}
    return out


def load(tag: str) -> dict:
    out = {}
    for name in SETS:
        path = HERE / tag / f"{name}_dev" / "search_eval_dev_detail.csv"
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            if not row["relevant_ids"].strip():
                continue
            out[(name, row["query_id"])] = {
                "cand": int(row["candidate_rank@30"]) if row["candidate_rank@30"] else None,
                "base": int(row["baseline_rank"]) if row["baseline_rank"] else None,
            }
    return out


def main() -> None:
    tags = sys.argv[1:]
    runs = {tag: load(tag) for tag in tags}
    ctx = context_queries()
    print("| 실행 | 세트 | n | 후보 30 안 | 리랭킹 전 Hit@1 | Hit@10 |")
    print("|---|---|---|---|---|---|")
    for tag, data in runs.items():
        for name in (*SETS, "합계", "context_clues 37건"):
            rows = [v for k, v in data.items() if name == "합계" or k[0] == name or (name.startswith("context") and k in ctx)]
            print(f"| {tag} | {name} | {len(rows)} | {sum(1 for r in rows if r['cand'])} | "
                  f"{sum(1 for r in rows if r['base'] == 1)} | {sum(1 for r in rows if r['base'] and r['base'] <= 10)} |")
    base = runs[tags[0]]
    show = lambda r: "밖" if r is None else str(r)  # noqa: E731
    for tag in tags[1:]:
        data = runs[tag]
        changed = [(k, base[k], v) for k, v in sorted(data.items()) if v["cand"] != base[k]["cand"]]
        gained = [c for c in changed if c[2]["cand"] and not c[1]["cand"]]
        lost = [c for c in changed if c[1]["cand"] and not c[2]["cand"]]
        print(f"\n== {tags[0]} → {tag}: 후보에 새로 든 {len(gained)} · 빠진 {len(lost)} · 후보 순위가 바뀐 질의 {len(changed)}")
        for (s, q), b, v in changed:
            mark = " [context]" if (s, q) in ctx else ""
            print(f"   {s} {q}{mark}: 후보 {show(b['cand'])} → {show(v['cand'])} · 리랭킹 전 {show(b['base'])} → {show(v['base'])}")
    with open(HERE / "candidates.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["set", "query_id", "has_context_clue", *[f"{t}_{c}" for t in tags for c in ("candidate_rank", "pre_rerank_rank")]])
        for key in sorted(base):
            writer.writerow([key[0], key[1], int(key in ctx),
                             *[runs[t][key][c] or "" for t in tags for c in ("cand", "base")]])


if __name__ == "__main__":
    main()
