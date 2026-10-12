"""v35(희소 사실 검증 끔) vs 같은 설정에 검증 켠 기록(v32 nosearch_pass1 · v33 lw_pass1) 비교.

품질은 커밋되는 경량본 `search_eval_dev_top10.csv`(곡 ID만)로 계산한다. 리랭킹 지연은 git 제외 대상인
`search_eval_dev_detail.csv`가 있을 때만 집계한다(없으면 그 줄만 생략).
"""
import csv
import statistics
from pathlib import Path

ROOT = Path("experiments/reranking")
PAIRS = [
    ("v06 dev 57", ROOT / "results_v32_gemini_listwise/nosearch_pass1", ROOT / "results_v35_rare_verify_off/v06_dev"),
    ("v09 dev 59", ROOT / "results_v33_v09_dev/lw_pass1", ROOT / "results_v35_rare_verify_off/v09_dev"),
]
METRICS = ["Hit@1", "Hit@5", "Hit@10", "MRR@10", "nDCG@10"]


def summary(d: Path) -> dict[str, dict]:
    return {row["metric"]: row for row in csv.DictReader(open(d / "search_eval_dev_summary.csv", encoding="utf-8-sig"))}


def ranks(d: Path) -> dict[str, int | None]:
    """질의 → 리랭킹 뒤 원래 타깃의 Top-10 순위(1부터). 밖이면 None."""
    out: dict[str, int | None] = {}
    for row in csv.DictReader(open(d / "search_eval_dev_top10.csv", encoding="utf-8-sig")):
        relevant = {x for x in row["relevant_ids"].split("|") if x}
        top = [x for x in row["rerank_top_ids"].split("|") if x]
        hit = [i for i, song in enumerate(top, start=1) if song in relevant]
        out[row["query_id"]] = hit[0] if hit else None
    return out


def rerank_ms(d: Path) -> list[float] | None:
    path = d / "search_eval_dev_detail.csv"
    if not path.exists():
        return None
    return sorted(float(r["rerank_only_ms"]) for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["rerank_only_ms"])


def p95(xs: list[float]) -> float:
    return xs[int(round(0.95 * (len(xs) - 1)))]


total = {"on": [0, 0], "off": [0, 0]}  # [Hit@1, Hit@10]
for name, on_dir, off_dir in PAIRS:
    if not (off_dir / "search_eval_dev_summary.csv").exists():
        print(f"{name}: 아직 없음")
        continue
    s_on, s_off = summary(on_dir), summary(off_dir)
    print(f"\n## {name}")
    print("| 지표 | 검증 켬 | 검증 끔 | 차이 |\n|---|---|---|---|")
    for m in METRICS:
        a, b = float(s_on[m]["rerank"]), float(s_off[m]["rerank"])
        print(f"| {m} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    print("| 결과 분포 |", s_on["Rerank outcome distribution"]["rerank"], "|", s_off["Rerank outcome distribution"]["rerank"], "| |")

    r_on, r_off = ranks(on_dir), ranks(off_dir)
    assert set(r_on) == set(r_off), f"{name}: 질의 집합이 다르다"
    for key, rs in (("on", r_on), ("off", r_off)):
        total[key][0] += sum(1 for r in rs.values() if r == 1)
        total[key][1] += sum(1 for r in rs.values() if r is not None)
    gain = [q for q in r_off if r_on[q] is None and r_off[q] is not None]
    loss = [q for q in r_off if r_on[q] is not None and r_off[q] is None]
    print(f"Hit@10 회복 {len(gain)} {gain} / 손실 {len(loss)} {loss}")
    moved = [(q, r_on[q], r_off[q]) for q in r_off if r_on[q] != r_off[q]]
    fmt = lambda r: "-" if r is None else str(r)
    print(f"순위 바뀐 질의 {len(moved)}/{len(r_off)}:", ", ".join(f"{q} {fmt(a)}→{fmt(b)}" for q, a, b in moved))

    ms_on, ms_off = rerank_ms(on_dir), rerank_ms(off_dir)
    if ms_on and ms_off:
        print(f"리랭킹 ms 중앙값·p95: 켬 {statistics.median(ms_on):.0f}·{p95(ms_on):.0f} → 끔 {statistics.median(ms_off):.0f}·{p95(ms_off):.0f}")
    else:
        print("리랭킹 ms: detail.csv 없음(git 제외) — 생략")

print(f"\n합산 Hit@1 {total['on'][0]}→{total['off'][0]}, Hit@10 {total['on'][1]}→{total['off'][1]}")
