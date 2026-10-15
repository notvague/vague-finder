"""v50 실행 폴더들을 나란히 비교한다 — 후보 Recall@30 · 엄격 지표 · 리랭킹 결과 코드 · 지연 · 질의별 순위 이동. 모델 호출 없음.

    venv/bin/python <이 파일> [--sets v06,v09,v10] <대조 실행> <비교 실행> ...

실행 이름은 이 폴더 안의 하위 폴더(`rerank_ctx0_rep1`)이거나 `experiments/reranking` 아래 경로
(`results_v49_vertex_baseline/vertex_cache_rep1`)다. 각 실행 폴더 아래 `{v06,v09,v10}_dev/search_eval_dev_top10.csv`를 읽고,
`search_eval_dev_detail.csv`(git 제외)가 있으면 후보 순위·리랭킹 전 순위·결과 코드·지연(`retrieval_ms`·`rerank_only_ms`)도 센다.
리랭킹을 끈 실행(cand_*)은 `rerank_top_ids`가 검색 순서 그대로라 "리랭킹 전 Hit@k"로 읽는다.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

HERE = Path(__file__).parent
SETS = ("v06", "v09", "v10")
OK_OUTCOMES = {None, "", "applied", "policy_skipped"}


def folder_of(tag: str) -> Path:
    return (HERE.parent / tag) if "/" in tag else (HERE / tag)


def load(tag: str, sets) -> dict:
    out: dict = {}
    for name in sets:
        folder = folder_of(tag) / f"{name}_dev"
        top10 = folder / "search_eval_dev_top10.csv"
        if not top10.exists():
            continue
        detail = folder / "search_eval_dev_detail.csv"
        rows = {r["query_id"]: r for r in csv.DictReader(open(detail, encoding="utf-8-sig"))} if detail.exists() else {}
        for row in csv.DictReader(open(top10, encoding="utf-8-sig")):
            q = row["query_id"]
            relevant = set(filter(None, row["relevant_ids"].split("|")))
            if not relevant:
                continue
            top = row["rerank_top_ids"].split("|")
            base = row["baseline_top_ids"].split("|")
            d = rows.get(q)
            out[(name, q)] = {
                "rank": next((i for i, sid in enumerate(top, 1) if sid in relevant), None),
                "base_rank": next((i for i, sid in enumerate(base, 1) if sid in relevant), None),
                "cand": None if d is None else (int(d["candidate_rank@30"]) if d["candidate_rank@30"] else 0),
                "outcome": None if d is None else d.get("rerank_outcome", ""),
                "retrieval_ms": None if d is None or not d.get("retrieval_ms") else float(d["retrieval_ms"]),
                "rerank_ms": None if d is None or not d.get("rerank_only_ms") else float(d["rerank_only_ms"]),
                "calls": None if d is None or not d.get("rerank_calls") else int(float(d["rerank_calls"])),
                "digest": None if d is None else d.get("candidate_digest", ""),
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


def lat(values) -> str:
    if not values:
        return "—"
    return f"중앙값 {statistics.median(values) / 1000:.2f}초 · p95 {pct(values, 0.95) / 1000:.2f}초 · 최대 {max(values) / 1000:.2f}초"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("tags", nargs="+")
    args = ap.parse_args()
    sets = tuple(s for s in args.sets.split(",") if s)
    runs = {tag: load(tag, sets) for tag in args.tags}

    print("| 실행 | 세트 | n | 후보 30 안 | Hit@1 | Hit@5 | Hit@10 | MRR@10 |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, data in runs.items():
        for name in (*sets, "합계"):
            rows = [v for (s, _), v in data.items() if name == "합계" or s == name]
            if rows:
                print(f"| {tag} | {name} | {metrics(rows)} |")
        outcomes: dict = {}
        for v in data.values():
            if v["outcome"] is not None:
                outcomes[v["outcome"]] = outcomes.get(v["outcome"], 0) + 1
        applied = [v for v in data.values() if v["outcome"] == "applied" and v["rerank_ms"] is not None]
        total = [v["retrieval_ms"] + v["rerank_ms"] for v in data.values()
                 if v["retrieval_ms"] is not None and v["rerank_ms"] is not None]
        retrieval = [v["retrieval_ms"] for v in data.values() if v["retrieval_ms"] is not None]
        calls = [v["calls"] for v in data.values() if v["calls"] is not None]
        if outcomes:
            print(f"  {tag}: 리랭킹 결과 {outcomes}")
        if retrieval:
            print(f"  {tag}: 검색(후보까지) {lat(retrieval)}")
        if applied:
            print(f"  {tag}: 모델을 부른 질의 리랭킹 {lat([v['rerank_ms'] for v in applied])} · 전체 검색(검색+리랭킹, 176건) {lat(total)}"
                  f" · 리랭킹 호출 합계 {sum(calls)}(질의당 >1이면 하네스가 두 번 이상 부름)")
        failed = sorted(q for (_, q), v in data.items() if v["outcome"] not in OK_OUTCOMES)
        # 리랭킹을 끈 실행(cand_*)은 전부 skipped다 — 리랭킹이 한 번이라도 적용된 실행에서만 실패 목록을 낸다
        if failed and "applied" in outcomes:
            print(f"  ! {tag}: 리랭킹이 적용되지 않은 질의 {failed}")

    base = runs[args.tags[0]]
    show = lambda r: "밖" if not r else str(r)  # noqa: E731
    for tag in args.tags[1:]:
        data = {k: v for k, v in runs[tag].items() if k in base}
        same = sum(1 for k, v in data.items() if v["rank"] == base[k]["rank"])
        gained = [(k, base[k], v) for k, v in data.items() if v["rank"] and not base[k]["rank"]]
        lost = [(k, base[k], v) for k, v in data.items() if not v["rank"] and base[k]["rank"]]
        moved = [(k, base[k], v) for k, v in data.items() if v["rank"] and base[k]["rank"] and v["rank"] != base[k]["rank"]]
        up = sum(1 for _, b, v in moved if v["rank"] < b["rank"])
        better = sum(1 for k, v in data.items() if (v["rank"] or 99) < (base[k]["rank"] or 99))
        worse = sum(1 for k, v in data.items() if (v["rank"] or 99) > (base[k]["rank"] or 99))
        cand_in = [(k, base[k], v) for k, v in data.items() if v["cand"] and base[k]["cand"] == 0]
        cand_out = [(k, base[k], v) for k, v in data.items() if base[k]["cand"] and v["cand"] == 0]
        cand_moved = [(k, base[k], v) for k, v in data.items() if v["cand"] and base[k]["cand"] and v["cand"] != base[k]["cand"]]
        digest_same = sum(1 for k, v in data.items() if v["digest"] is not None and v["digest"] == base[k]["digest"])
        print(f"\n== {args.tags[0]} → {tag} (겹치는 {len(data)}건): 정답 최종 순위 같음 {same} · 질의별 나음 {better} : 나쁨 {worse} · "
              f"10위 안 새로 든 {len(gained)} · 빠진 {len(lost)} · 안에서 오름 {up} · 내림 {len(moved) - up}")
        if any(v["cand"] is not None for v in data.values()):
            print(f"   후보 30: 새로 든 {len(cand_in)} · 빠진 {len(cand_out)} · 후보 순위만 바뀐 {len(cand_moved)} · 후보 집합 지문 같음 {digest_same}/{len(data)}")
            for label, items in (("후보 든", cand_in), ("후보 빠진", cand_out), ("후보 이동", cand_moved)):
                for (s, q), b, v in sorted(items):
                    print(f"   {label} {s} {q}: 후보 {show(b['cand'])} → {show(v['cand'])} · 리랭킹 전 {show(b['base_rank'])} → {show(v['base_rank'])} · 최종 {show(b['rank'])} → {show(v['rank'])}")
        for label, items in (("든", gained), ("빠진", lost), ("움직인", moved)):
            for (s, q), b, v in sorted(items):
                print(f"   {label} {s} {q}: 후보 {show(b['cand'])} → {show(v['cand'])} · 최종 {show(b['rank'])} → {show(v['rank'])}")


if __name__ == "__main__":
    main()
