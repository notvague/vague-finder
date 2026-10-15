"""검색 지연을 '초기 적재'와 '준비된 상태'로 나눈다(v52·v53과 같은 도구, v54 실행 폴더). 모델·DB 없음. detail CSV의 retrieval_ms · rerank_only_ms를 읽는다.

초기 적재 = 각 프로세스(세트별 실행)의 첫 질의(임베딩 모델·Qdrant·분석 캐시 워밍업)와 **가사 질의의 Mongo 가사 판 적재**
(`LyricsExactSearchService`가 첫 가사 검색에서 전곡 가사를 읽고 5분 TTL로 다시 읽는다 — q209 · n007 · t003). 나머지가 준비된 상태의 지연이다.
'전체(176)'는 적재 질의까지 포함한 질의당 검색+리랭킹 합(초기 적재를 포함한 전체 지연)이다.

    venv/bin/python experiments/reranking/results_v54_attr_genre_policy/latency_split.py
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent
V53 = HERE.parent / "results_v53_attr_filter_no_genre"
RUNS = {
    "v54 후보 단계 A(끔, 3프로세스 동시)": HERE / "cand_attr0_eraplus",
    "v54 후보 단계 B(always, 동시)": HERE / "cand_attr0.5_eraplus",
    "v54 후보 단계 D(conditional, 동시)": HERE / "cand_attr0.5_eraplus_condgenre",
    "v54 B1 (순차 단일)": HERE / "rerank_attr0.5_eraplus_rep1",
    "v54 D1 (순차 단일)": HERE / "rerank_attr0.5_eraplus_condgenre_rep1",
    "v54 B2 (순차 단일)": HERE / "rerank_attr0.5_eraplus_rep2",
    "v54 D2 (순차 단일)": HERE / "rerank_attr0.5_eraplus_condgenre_rep2",
}
LOAD_THRESHOLD_S = 5.0  # 준비된 상태의 검색은 p95 1.5초 안이다. 이보다 긴 것은 적재로 분류하고 질의 ID를 적는다


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]


def main() -> None:
    print("| 실행 | 첫 질의(세트별, 검색 s) | 적재로 분류된 그 밖의 질의(검색 s) | 준비된 상태 검색 중앙값 / p95 / 최대 (n) | 리랭킹 중앙값 / p95 | 전체(적재 포함) 질의당 검색+리랭킹 중앙값 / p95 / 합계 분 |")
    print("|---|---|---|---|---|---|")
    for name, root in RUNS.items():
        firsts, loads, ready, rerank, total = [], [], [], [], []
        for s in ("v06", "v09", "v10"):
            p = root / f"{s}_dev" / "search_eval_dev_detail.csv"
            if not p.exists():
                continue
            rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
            for i, r in enumerate(rows):
                sec = float(r["retrieval_ms"]) / 1000
                rk = float(r["rerank_only_ms"] or 0) / 1000
                total.append(sec + rk)
                if i == 0:
                    firsts.append(f"{r['query_id']} {sec:.1f}")
                elif sec > LOAD_THRESHOLD_S:
                    loads.append(f"{r['query_id']} {sec:.0f}")
                else:
                    ready.append(sec)
                    if rk > 0:
                        rerank.append(rk)
        if not ready:
            continue
        rr = f"{st.median(rerank):.2f} / {pct(rerank, .95):.2f}" if rerank else "-"
        print(f"| {name} | {', '.join(firsts)} | {', '.join(loads) or '-'} | {st.median(ready):.2f} / {pct(ready, .95):.2f} / {max(ready):.2f} ({len(ready)}) | {rr} "
              f"| {st.median(total):.2f} / {pct(total, .95):.2f} / {sum(total) / 60:.1f} |")


if __name__ == "__main__":
    main()
