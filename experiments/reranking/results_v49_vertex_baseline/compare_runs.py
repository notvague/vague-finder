"""실행 폴더들을 나란히 비교한다 — 엄격 지표 · 리랭킹 결과 코드 · 지연(detail이 있을 때) · 같은 질의의 순위 일치.

    venv/bin/python <이 파일> [--queries pilot|all] <대조 실행> <비교 실행> ...

실행 이름은 이 폴더 안의 하위 폴더(`aistudio_cache_pilot_rep1`)이거나 `experiments/reranking` 아래 경로
(`results_v46_main_after_context/new_scale1_rep1`)다. 각 실행 폴더 아래 `{v06,v09,v10}_dev/search_eval_dev_top10.csv`를 읽고,
`search_eval_dev_detail.csv`가 있으면 리랭킹 결과 코드·후보 순위·리랭킹 시간(`rerank_only_ms`)도 센다(커밋하지 않는 파일).
`--queries pilot`이면 시범 30건(세트당 10건)만 본다 — 저장된 176건 실행을 시범 30건과 견줄 때.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

HERE = Path(__file__).parent
SETS = ("v06", "v09", "v10")
PILOT = {
    "v06": "m302,q101,q112,q204,q205,q207,q209,q312,q314,q318",
    "v09": "n010,n031,n038,n057,n061,n071,n073,n082,n086,n096",
    "v10": "t024,t025,t029,t042,t075,t083,t086,t096,t097,t104",
}
PILOT_IDS = {(s, q) for s, ids in PILOT.items() for q in ids.split(",")}


def folder_of(tag: str) -> Path:
    return (HERE.parent / tag) if "/" in tag else (HERE / tag)


def load(tag: str, only_pilot: bool) -> dict:
    out: dict = {}
    for name in SETS:
        folder = folder_of(tag) / f"{name}_dev"
        top10 = folder / "search_eval_dev_top10.csv"
        if not top10.exists():
            continue
        detail = folder / "search_eval_dev_detail.csv"
        rows = {r["query_id"]: r for r in csv.DictReader(open(detail, encoding="utf-8-sig"))} if detail.exists() else {}
        for row in csv.DictReader(open(top10, encoding="utf-8-sig")):
            q = row["query_id"]
            if only_pilot and (name, q) not in PILOT_IDS:
                continue
            relevant = set(filter(None, row["relevant_ids"].split("|")))
            if not relevant:
                continue
            top = row["rerank_top_ids"].split("|")
            d = rows.get(q)
            out[(name, q)] = {
                "rank": next((i for i, sid in enumerate(top, 1) if sid in relevant), None),
                "cand": None if d is None else (int(d["candidate_rank@30"]) if d["candidate_rank@30"] else 0),
                "outcome": None if d is None else d.get("rerank_outcome", ""),
                "ms": None if d is None or not d.get("rerank_only_ms") else float(d["rerank_only_ms"]),
                "calls": None if d is None or not d.get("rerank_calls") else int(float(d["rerank_calls"])),
            }
    return out


def pct(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def metrics(rows: list) -> str:
    n = len(rows)
    hit = lambda k: sum(1 for r in rows if r["rank"] and r["rank"] <= k)  # noqa: E731
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    cand = "?" if any(r["cand"] is None for r in rows) else sum(1 for r in rows if r["cand"])
    return f"{n} | {cand} | {hit(1)} | {hit(5)} | {hit(10)} | {mrr:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", choices=["pilot", "all"], default="all")
    ap.add_argument("tags", nargs="+")
    args = ap.parse_args()
    runs = {tag: load(tag, args.queries == "pilot") for tag in args.tags}

    print("| 실행 | 세트 | n | 후보 30 안 | Hit@1 | Hit@5 | Hit@10 | MRR@10 |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, data in runs.items():
        for name in (*SETS, "합계"):
            rows = [v for (s, _), v in data.items() if name == "합계" or s == name]
            if rows:
                print(f"| {tag} | {name} | {metrics(rows)} |")
        outcomes = {}
        for v in data.values():
            if v["outcome"] is not None:
                outcomes[v["outcome"]] = outcomes.get(v["outcome"], 0) + 1
        ms = [v["ms"] for v in data.values() if v["ms"] is not None and v["outcome"] == "applied"]
        calls = [v["calls"] for v in data.values() if v["calls"] is not None]
        if outcomes:
            line = f"  {tag}: 리랭킹 결과 {outcomes}"
            if ms:
                line += (f" · 모델을 부른 질의 리랭킹 시간 중앙값 {statistics.median(ms) / 1000:.2f}초 "
                         f"· p95 {pct(ms, 0.95) / 1000:.2f}초 · 최대 {max(ms) / 1000:.2f}초")
            if calls:
                line += f" · 호출 합계 {sum(calls)}(질의당 >1이면 재시도)"
            print(line)
        failed = sorted(q for (_, q), v in data.items() if v["outcome"] not in (None, "applied", "policy_skipped", ""))
        if failed:
            print(f"  ! {tag}: 리랭킹이 적용되지 않은 질의 {failed}")

    base = runs[args.tags[0]]
    show = lambda r: "밖" if r is None else str(r)  # noqa: E731
    for tag in args.tags[1:]:
        data = {k: v for k, v in runs[tag].items() if k in base}
        same = sum(1 for k, v in data.items() if v["rank"] == base[k]["rank"])
        gained = [(k, base[k], v) for k, v in data.items() if v["rank"] and not base[k]["rank"]]
        lost = [(k, base[k], v) for k, v in data.items() if not v["rank"] and base[k]["rank"]]
        moved = [(k, base[k], v) for k, v in data.items() if v["rank"] and base[k]["rank"] and v["rank"] != base[k]["rank"]]
        up = sum(1 for _, b, v in moved if v["rank"] < b["rank"])
        better = sum(1 for k, v in data.items() if (v["rank"] or 99) < (base[k]["rank"] or 99))
        worse = sum(1 for k, v in data.items() if (v["rank"] or 99) > (base[k]["rank"] or 99))
        print(f"\n== {args.tags[0]} → {tag} (겹치는 {len(data)}건): 정답 순위 같음 {same} · 질의별 나음 {better} : 나쁨 {worse} · "
              f"10위 안 새로 든 {len(gained)} · 빠진 {len(lost)} · 안에서 오름 {up} · 내림 {len(moved) - up}")
        for label, items in (("든", gained), ("빠진", lost), ("움직인", moved)):
            for (s, q), b, v in sorted(items):
                print(f"   {label} {s} {q}: 최종 {show(b['rank'])} → {show(v['rank'])}")


if __name__ == "__main__":
    main()
