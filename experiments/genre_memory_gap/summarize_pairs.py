#!/usr/bin/env python
"""같은 문장에서 장르 낱말만 바꿨을 때 정답이 어디로 가는가 — 곡마다 한 줄 (모델·DB 없음).

`diagnose_genre_gap trace`의 산출물(diagnosis.csv)을 읽어, 곡마다 네 문장을 나란히 놓는다.

    registered      등록 장르 표현            (예: …남자 그룹의 힙합 노래인데…)
    perceived       체감 장르 표현            (예: …남자 그룹의 발라드 노래인데…)
    genre_omitted   장르 낱말을 뺀 같은 문장   (예: …남자 그룹의 노래인데…)
    identity_sound  가수 정체성·소리 표현      (여러 단서가 함께 바뀐다 — 참고)

문장마다 남기는 것(후보 단계 값이다 — 리랭커 없음):

    text    전체 깊이에서 본 주 텍스트 경로의 순위
    perf    보컬·편성 전용 보조 검색에서의 순위(그 검색이 돌지 않으면 `-`, 깊이 300 안에 없으면 빈칸). 서비스는 상위 30곡만 읽는다
    paths   현행 실행에서 정답이 실제로 잡힌 경로(`경로:순위`). 후보는 주 경로뿐 아니라 보조 신호로도 든다
    pool    후보 30 안의 순위(빈칸 = 후보 밖) — 현행 배율 · 가장 낮춘 배율 · (--blank를 주면) 분석의 genre 칸을 비운 실행
    deep    경로 깊이 제한이 없다면 보정 뒤 몇 위인가

체감 장르 문장이 나쁜 까닭을 나눠 보는 법:

    가산의 몫        perceived pool@현행 → pool@0     명시 가산(genre_match)이 등록 장르가 맞는 다른 곡을 올린 만큼
    genre 칸의 몫    pool@0 → pool@blank              보조 경로 질의 등에 들어가던 분석의 genre 값까지 뺀 만큼
    낱말의 몫        pool@blank → genre_omitted@0     질의문·태그에 든 장르 낱말이 텍스트 검색(주 경로와, 같은 검색문·태그를 쓰는
                                                      보조 검색)에서 등록 장르가 그 낱말인 곡을 끌어온 만큼

    venv/bin/python experiments/genre_memory_gap/summarize_pairs.py \\
        experiments/genre_memory_gap/results_v01/trace_genre_gap \\
        --blank experiments/genre_memory_gap/results_v01/trace_genre_gap_blank \\
        --output experiments/genre_memory_gap/results_v01/genre_word_decomposition.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

VARIANTS = ("registered", "perceived", "genre_omitted", "identity_sound")
POOL = 30
NOT_RUN = "×"   # 화면 표에서만 — 그 문장에서는 전용 검색이 돌지 않았다(CSV에는 trace가 적은 `-` 그대로)


def _load(folder: str) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """pair → variant → 배율 → 행. 변형마다 질의는 하나다(contrast는 여기서 보지 않는다)."""
    out: Dict[str, Dict[str, Dict[str, dict]]] = {}
    with (Path(folder) / "diagnosis.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["variant"] in VARIANTS:
                out.setdefault(row["pair_id"], {}).setdefault(row["variant"], {})[row["genre_match_scale"]] = row
    return out


def _int(value: str) -> Optional[int]:
    return int(value) if str(value).strip().isdigit() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder")
    parser.add_argument("--blank", default="", help="--blank-genre로 돈 trace 폴더(있으면 pool@blank 열을 더한다)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = _load(args.folder)
    blank = _load(args.blank) if args.blank else {}
    scales = sorted({s for variants in data.values() for rows in variants.values() for s in rows}, key=float, reverse=True)
    base, low = scales[0], scales[-1]
    lines: List[dict] = []
    for pair in sorted(data):
        variants = data[pair]
        any_row = next(iter(next(iter(variants.values())).values()))
        line = {"pair_id": pair, "group": any_row["group"], "registered_genre": any_row["target_registered_genre"]}
        for variant in VARIANTS:
            rows = variants.get(variant)
            if not rows:
                continue
            line[f"{variant}:analysis_genre"] = rows[base]["analysis_genre"]
            line[f"{variant}:text"] = rows[base]["text_hybrid_rank"]
            line[f"{variant}:perf"] = rows[base]["performance_search_rank"]
            line[f"{variant}:paths@{base}"] = rows[base]["target_paths"]
            line[f"{variant}:pool@{base}"] = rows[base]["pool_rank"]
            line[f"{variant}:pool@{low}"] = rows[low]["pool_rank"]
            if blank:
                blank_row = next(iter(blank.get(pair, {}).get(variant, {}).values()), None)
                line[f"{variant}:pool@blank"] = blank_row["pool_rank"] if blank_row else ""
            line[f"{variant}:deep@{base}"] = rows[base]["boosted_rank_deep"]
            line[f"{variant}:deep@{low}"] = rows[low]["boosted_rank_deep"]
        lines.append(line)
    columns = list(dict.fromkeys(key for line in lines for key in line))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: line.get(c, "") for c in columns} for line in lines)

    def cell(line: dict, variant: str, kind: str) -> str:
        value = str(line.get(f"{variant}:{kind}", ""))
        return NOT_RUN if value == "-" else value or "–"

    blank_head = " → genre 칸 비움" if blank else ""
    print(f"곡   등록 장르       | text 등록·체감·없음 | perf 등록·체감·없음 | pool 등록 · 체감@{base} → @{low}{blank_head} · 없음@{base} → @{low}"
          f" | 등록 문장이 들어온 길")
    for line in lines:
        blank_cell = f" → {cell(line, 'perceived', 'pool@blank'):>3}" if blank else ""
        print(f"{line['pair_id']:<4}{line['registered_genre'][:13]:<14} | "
              f"{cell(line, 'registered', 'text'):>5} {cell(line, 'perceived', 'text'):>5} {cell(line, 'genre_omitted', 'text'):>5} | "
              f"{cell(line, 'registered', 'perf'):>4} {cell(line, 'perceived', 'perf'):>4} {cell(line, 'genre_omitted', 'perf'):>4} | "
              f"{cell(line, 'registered', f'pool@{base}'):>3} · {cell(line, 'perceived', f'pool@{base}'):>3} → "
              f"{cell(line, 'perceived', f'pool@{low}'):>3}{blank_cell} · {cell(line, 'genre_omitted', f'pool@{base}'):>3} → "
              f"{cell(line, 'genre_omitted', f'pool@{low}'):>3} | {cell(line, 'registered', f'paths@{base}')}")

    gap = [line for line in lines if line["group"] == "gap"]

    def in_pool(variant: str, kind: str) -> int:
        return sum(1 for line in gap if _int(line.get(f"{variant}:{kind}", "")) is not None)

    print(f"\n– 후보 밖(perf는 깊이 300 안에 없음) · {NOT_RUN} 그 문장에서는 전용 검색이 돌지 않음")
    print(f"\n간격 사례 {len(gap)}곡 — 후보 {POOL} 안에 든 수")
    for variant, label in (("registered", "등록 장르 표현"), ("perceived", "체감 장르 표현"), ("genre_omitted", "장르 낱말 없음")):
        tail = f" · genre 칸 비움 {in_pool(variant, 'pool@blank')}" if blank else ""
        print(f"  {label:<10} 현행 {in_pool(variant, f'pool@{base}')} · 가산 끔 {in_pool(variant, f'pool@{low}')}{tail}")
    for kind, label in (("text", "주 텍스트 경로"), ("perf", "보컬·편성 전용 검색")):
        pairs = [(_int(line.get(f"registered:{kind}", "")), _int(line.get(f"perceived:{kind}", ""))) for line in gap]
        both = [(r, p) for r, p in pairs if r is not None and p is not None]
        print(f"  {label} 순위가 체감 장르 문장에서 더 나쁜 곡 {sum(1 for r, p in both if p > r)}/{len(both)}"
              f"(나아진 곡 {sum(1 for r, p in both if p < r)} · 같은 곡 {sum(1 for r, p in both if p == r)}, 두 문장 모두 순위가 있는 곡만)")
    print(f"\n기록: {out}")


if __name__ == "__main__":
    main()
