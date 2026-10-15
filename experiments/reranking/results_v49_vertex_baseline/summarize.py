"""측정 폴더를 나란히 요약한다 — 엄격 지표와 질의별 순위 변화. 첫 인자가 대조다.

인자는 이 폴더 안의 실행 이름(`new_scale1_rep1`)이거나 `experiments/reranking` 아래 경로
(`results_v43_audio_path_weight/scale1_rep1`)다. 후보 순위는 detail CSV가 있을 때만 센다(커밋하지 않는 파일).

  venv/bin/python <이 파일> results_v43_audio_path_weight/scale1_rep1 old_scale1_rep1 new_scale1_rep1
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
SETS = ("v06", "v09", "v10")


def folder_of(tag: str) -> Path:
    return (HERE.parent / tag) if "/" in tag else (HERE / tag)


def load(tag: str) -> dict:
    """(세트, 질의) → {"cand": 후보 순위|None|"?", "rank": 최종 순위|None(10위 밖), "outcome": 리랭킹 결과 코드}"""
    out: dict = {}
    for name in SETS:
        folder = folder_of(tag) / f"{name}_dev"
        top10 = folder / "search_eval_dev_top10.csv"
        if not top10.exists():
            continue
        detail = folder / "search_eval_dev_detail.csv"
        rows = {r["query_id"]: r for r in csv.DictReader(open(detail, encoding="utf-8-sig"))} if detail.exists() else {}
        for row in csv.DictReader(open(top10, encoding="utf-8-sig")):
            relevant = set(filter(None, row["relevant_ids"].split("|")))
            if not relevant:
                continue  # no_target
            top = row["rerank_top_ids"].split("|")
            q = row["query_id"]
            d = rows.get(q)
            out[(name, q)] = {
                "cand": "?" if d is None else (int(d["candidate_rank@30"]) if d["candidate_rank@30"] else None),
                "rank": next((i for i, sid in enumerate(top, 1) if sid in relevant), None),
                "outcome": "" if d is None else d.get("rerank_outcome", ""),
            }
    return out


def metrics(rows: list) -> str:
    n = len(rows)
    hit = lambda k: sum(1 for r in rows if r["rank"] and r["rank"] <= k)  # noqa: E731
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    cand = "?" if any(r["cand"] == "?" for r in rows) else sum(1 for r in rows if r["cand"])
    return f"{n} | {cand} | {hit(1)} | {hit(5)} | {hit(10)} | {mrr:.3f}"


def main() -> None:
    tags = sys.argv[1:]
    runs = {tag: load(tag) for tag in tags}
    print("| 실행 | 세트 | n | 후보 30 안 | Hit@1 | Hit@5 | Hit@10 | MRR@10 |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, data in runs.items():
        for name in (*SETS, "합계"):
            rows = [v for (s, _), v in data.items() if name == "합계" or s == name]
            if rows:
                print(f"| {tag} | {name} | {metrics(rows)} |")
        failed = sorted(q for (_, q), v in data.items() if v["outcome"] not in ("applied", "policy_skipped", ""))
        if failed:
            print(f"  ! {tag}: 리랭킹이 적용되지 않은 질의 {failed}")

    base = runs[tags[0]]
    show = lambda r: "밖" if r is None else str(r)  # noqa: E731
    for tag in tags[1:]:
        data = {k: v for k, v in runs[tag].items() if k in base}
        gained = [(k, base[k], v) for k, v in data.items() if v["rank"] and not base[k]["rank"]]
        lost = [(k, base[k], v) for k, v in data.items() if not v["rank"] and base[k]["rank"]]
        moved = [(k, base[k], v) for k, v in data.items() if v["rank"] and base[k]["rank"] and v["rank"] != base[k]["rank"]]
        up = sum(1 for _, b, v in moved if v["rank"] < b["rank"])
        print(f"\n== {tags[0]} → {tag} (겹치는 {len(data)}건): 10위 안 새로 든 {len(gained)} · 빠진 {len(lost)} · 안에서 오름 {up} · 내림 {len(moved) - up}")
        for label, items in (("든", gained), ("빠진", lost), ("움직인", moved)):
            for (s, q), b, v in sorted(items):
                print(f"   {label} {s} {q}: 후보 {show(b['cand'])} → {show(v['cand'])} · 최종 {show(b['rank'])} → {show(v['rank'])}")


if __name__ == "__main__":
    main()
