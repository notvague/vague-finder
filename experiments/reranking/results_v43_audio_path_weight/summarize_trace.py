"""후보 추적표를 변형별로 요약한다 — 후보 30 안 건수, 대조(ctrl) 대비 든 질의·빠진 질의, 소리 경로 순위 분포.

  venv/bin/python <이 파일> [태그 ...]      (기본: 폴더에 있는 태그 전부)
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
SETS = ("v06", "v09", "v10")


def load(tag: str) -> dict:
    """질의 id → 정답 곡들 중 가장 좋은 행. v06은 정답이 둘 이상인 질의가 있다."""
    best: dict = {}
    for name in SETS:
        path = HERE / f"candidate_drop_{name}_dev_{tag}.csv"
        if not path.exists():
            return {}
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            key = (name, row["query_id"])
            rank = int(row["pool_rank"]) if row["pool_rank"] else 10**6
            if key not in best or rank < best[key][0]:
                best[key] = (rank, row)
    return best


def num(value: str):
    return int(value) if value else None


def main() -> None:
    tags = sys.argv[1:] or sorted(
        {p.stem.rsplit("_dev_", 1)[1] for p in HERE.glob("candidate_drop_v06_dev_*.csv")}
    )
    ctrl = load("ctrl")
    assert ctrl, "ctrl 추적표가 없다"
    in_ctrl = {k for k, (rank, _) in ctrl.items() if rank <= 30}

    audio_rows = [row for _, row in ctrl.values() if "audio" in row["active_main_paths"]]
    a_ranks = [num(r["audio_rank"]) for r in audio_rows if num(r["audio_rank"])]
    t_ranks = [num(r["text_hybrid_rank"]) for r in audio_rows if num(r["text_hybrid_rank"])]
    print(f"질의 {len(ctrl)}건 · 소리 경로가 켜진 질의 {len(audio_rows)}건")
    print(f"  정답의 소리 경로 순위: 중앙값 {statistics.median(a_ranks):.0f} · 10위 안 {sum(r <= 10 for r in a_ranks)} · 30위 안 {sum(r <= 30 for r in a_ranks)} · 100위 안 {sum(r <= 100 for r in a_ranks)}")
    print(f"  정답의 텍스트 경로 순위: 중앙값 {statistics.median(t_ranks):.0f} · 10위 안 {sum(r <= 10 for r in t_ranks)} · 30위 안 {sum(r <= 30 for r in t_ranks)} · 100위 안 {sum(r <= 100 for r in t_ranks)}")
    print()
    print("| 변형 | 후보 30 안 | v06 | v09 | v10 | 새로 든 질의 | 빠진 질의 | 후보 10 안 |")
    print("|---|---|---|---|---|---|---|---|")
    for tag in tags:
        data = load(tag)
        if not data:
            continue
        inside = {k for k, (rank, _) in data.items() if rank <= 30}
        top10 = sum(1 for rank, _ in data.values() if rank <= 10)
        per = {s: sum(1 for k in inside if k[0] == s) for s in SETS}
        gained = sorted(q for _, q in inside - in_ctrl)
        lost = sorted(q for _, q in in_ctrl - inside)
        print(f"| {tag} | {len(inside)} | {per['v06']} | {per['v09']} | {per['v10']} | "
              f"{len(gained)} ({', '.join(gained)}) | {len(lost)} ({', '.join(lost)}) | {top10} |")

    print()
    print("| 변형 | 정답의 텍스트 경로 순위: 10위 안 | 30위 안 | 100위 안 | 중앙값 |")
    print("|---|---|---|---|---|")
    for tag in tags:
        data = load(tag)
        if not data:
            continue
        ranks = [num(row["text_hybrid_rank"]) or 10**6 for _, row in data.values()]
        print(f"| {tag} | {sum(r <= 10 for r in ranks)} | {sum(r <= 30 for r in ranks)} | "
              f"{sum(r <= 100 for r in ranks)} | {statistics.median(ranks):.0f} |")


if __name__ == "__main__":
    main()
