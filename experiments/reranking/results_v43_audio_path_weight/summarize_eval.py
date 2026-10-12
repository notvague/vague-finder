"""측정 폴더(scale*_rep*)를 나란히 요약한다 — 엄격 지표와 질의별 순위 변화. 첫 태그가 대조다.

  venv/bin/python <이 파일> scale1_rep1 scale0_rep1 scale0_rep2
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
SETS = ("v06", "v09", "v10")


# 측정 폴더의 detail CSV는 커밋하지 않는다(.gitignore). 없으면 같은 설정의 후보 추적표에서 후보 순위를 읽는다 —
# 배율 0과 `off` 캐시는 후보 순위가 176건 모두 같다(RUN_INFO §3).
TRACE_FOR = {"scale1": "ctrl", "scale0": "off"}


def candidates(folder: Path, tag: str, name: str) -> tuple[dict, dict]:
    """질의 → 후보 순위, 질의 → 리랭킹 결과 코드. detail이 없으면 추적표로 대신하고 결과 코드는 비운다."""
    detail = folder / "search_eval_dev_detail.csv"
    if detail.exists():
        rows = {r["query_id"]: r for r in csv.DictReader(open(detail, encoding="utf-8-sig"))}
        return (
            {q: int(r["candidate_rank@30"]) if r["candidate_rank@30"] else None for q, r in rows.items()},
            {q: r.get("rerank_outcome", "") for q, r in rows.items()},
        )
    trace = HERE / f"candidate_drop_{name}_dev_{TRACE_FOR.get(tag.split('_rep')[0], '')}.csv"
    if not trace.exists():
        return {}, {}
    best: dict = {}
    for row in csv.DictReader(open(trace, encoding="utf-8-sig")):
        rank = int(row["pool_rank"]) if row["pool_rank"] else None
        q = row["query_id"]  # 정답이 여럿인 질의는 가장 좋은 행
        if q not in best or (rank is not None and (best[q] is None or rank < best[q])):
            best[q] = rank
    return best, {}


def load(tag: str) -> dict:
    """(세트, 질의) → {"cand": 후보 순위|None, "rank": 최종 순위|None(10위 밖)}"""
    out: dict = {}
    for name in SETS:
        folder = HERE / tag / f"{name}_dev"
        cand, outcome = candidates(folder, tag, name)
        for row in csv.DictReader(open(folder / "search_eval_dev_top10.csv", encoding="utf-8-sig")):
            relevant = set(filter(None, row["relevant_ids"].split("|")))
            if not relevant:
                continue  # no_target
            top = row["rerank_top_ids"].split("|")
            rank = next((i for i, sid in enumerate(top, 1) if sid in relevant), None)
            q = row["query_id"]
            out[(name, q)] = {"cand": cand.get(q), "rank": rank, "outcome": outcome.get(q, "")}
    return out


def metrics(rows: list) -> str:
    n = len(rows)
    hit = lambda k: sum(1 for r in rows if r["rank"] and r["rank"] <= k)  # noqa: E731
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    cand = sum(1 for r in rows if r["cand"])
    return f"{n} | {cand} | {hit(1)} | {hit(5)} | {hit(10)} | {mrr:.3f}"


def main() -> None:
    tags = sys.argv[1:]
    runs = {tag: load(tag) for tag in tags}
    print("| 실행 | 세트 | n | 후보 30 안 | Hit@1 | Hit@5 | Hit@10 | MRR@10 |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, data in runs.items():
        for name in (*SETS, "합계"):
            rows = [v for (s, _), v in data.items() if name == "합계" or s == name]
            print(f"| {tag} | {name} | {metrics(rows)} |")
        failed = sorted(q for (_, q), v in data.items() if v["outcome"] not in ("applied", "policy_skipped", ""))
        if failed:
            print(f"  ! {tag}: 리랭킹이 적용되지 않은 질의 {failed}")

    base = runs[tags[0]]
    show = lambda r: "밖" if r is None else str(r)  # noqa: E731
    for tag in tags[1:]:
        data = runs[tag]
        gained = [(k, base[k], v) for k, v in data.items() if v["rank"] and not base[k]["rank"]]
        lost = [(k, base[k], v) for k, v in data.items() if not v["rank"] and base[k]["rank"]]
        moved = [(k, base[k], v) for k, v in data.items() if v["rank"] and base[k]["rank"] and v["rank"] != base[k]["rank"]]
        up = sum(1 for _, b, v in moved if v["rank"] < b["rank"])
        print(f"\n== {tags[0]} → {tag}: 10위 안 새로 든 {len(gained)} · 빠진 {len(lost)} · 안에서 오름 {up} · 내림 {len(moved) - up}")
        for label, items in (("든", gained), ("빠진", lost), ("움직인", moved)):
            for (s, q), b, v in sorted(items):
                print(f"   {label} {s} {q}: 후보 {show(b['cand'])} → {show(v['cand'])} · 최종 {show(b['rank'])} → {show(v['rank'])}")


if __name__ == "__main__":
    main()
