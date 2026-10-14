#!/usr/bin/env python
"""가수 정체성·체감 장르 낱말이 지금 곡 표현(검색 passage)에 들어 있는가 — 계획서 §6(C)의 선행 확인.

사용자가 '아이돌' · '인디 밴드' · '래퍼' · '트로트 가수'라고 말했을 때 그 낱말이 닿을 자리가 곡 쪽에 있는지 본다.
곡 정보를 고치지 않고, 색인에 넣는 것과 같은 함수(`passage_builder`)로 passage를 다시 만들어 낱말만 센다.

    venv/bin/python experiments/genre_memory_gap/identity_terms_check.py \\
        --corpus data/all_songs.jsonl --output-dir experiments/genre_memory_gap/results_v01/identity_terms

산출물: `corpus_terms.csv`(낱말 × 가수 형태별 곡 수) · `target_terms.csv`(비교 질의의 정답 곡마다 낱말 유무).
곡 제목·가수 이름·passage 본문은 쓰지 않는다 — 곡 ID와 건수만 남긴다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding.text.passage_builder import build_dense_passage, build_sparse_passage  # noqa: E402

# 사용자 표현 → passage에서 찾을 낱말(표기 변형). 장르 낱말은 등록 장르 문자열과 겹치므로 참고로만 본다.
TERMS: Dict[str, List[str]] = {
    "아이돌": ["아이돌"],
    "걸그룹": ["걸그룹", "걸 그룹"],
    "보이그룹": ["보이그룹", "보이 그룹"],
    "인디": ["인디"],
    "밴드 사운드": ["밴드 사운드", "밴드사운드"],
    "래퍼": ["래퍼"],
    "트로트": ["트로트"],
    "발라드": ["발라드"],
    "알앤비": ["알앤비", "R&B", "r&b"],
    "어쿠스틱": ["어쿠스틱"],
    "통기타": ["통기타"],
    "잔잔": ["잔잔"],
}


def _types(song: Dict[str, Any]) -> List[str]:
    value = (song.get("metadata") or {}).get("type") or []
    return [str(v) for v in (value if isinstance(value, list) else [value])]


def _has(text: str, variants: List[str]) -> bool:
    return any(v in text for v in variants)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--sources", default=str(ROOT / "experiments/genre_memory_gap/query_sources.json"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # 줄 단위로 읽는다 — splitlines()는 가사 속 유니코드 줄바꿈(U+2028 등)에서도 끊는다.
    with open(args.corpus, encoding="utf-8") as f:
        songs = [json.loads(line) for line in f if line.strip()]
    targets = {s["song_id"]: s["pair_id"] for s in json.loads(Path(args.sources).read_text(encoding="utf-8"))["songs"]}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    totals: Counter = Counter()
    counts: Dict[str, Counter] = defaultdict(Counter)
    target_rows: List[Dict[str, Any]] = []
    for song in songs:
        sparse, dense = build_sparse_passage(song), build_dense_passage(song)
        registered = (song.get("metadata") or {}).get("genre") or []
        registered = [str(g) for g in (registered if isinstance(registered, list) else [registered]) if str(g).strip()]
        sparse_without_genre = sparse
        for genre in registered:       # 등록 장르 표기를 하나씩 지운다(passage에는 공백으로 이어 붙어 있다)
            sparse_without_genre = sparse_without_genre.replace(genre, " ")
        groups = ["(전체)"] + sorted({t for t in _types(song) if t in ("솔로", "그룹", "밴드", "듀오")})
        for group in groups:
            totals[group] += 1
        row: Dict[str, Any] = {"pair_id": targets.get(str(song.get("song_id")), ""), "song_id": song.get("song_id")}
        for term, variants in TERMS.items():
            in_sparse, in_dense = _has(sparse, variants), _has(dense, variants)
            # 등록 장르 문자열 밖에서도 나오는가 — '인디'가 "인디음악" 장르 줄에만 있는 경우와 구분한다.
            outside_genre = _has(sparse_without_genre, variants)
            for group in groups:
                counts[term][(group, "sparse")] += in_sparse
                counts[term][(group, "dense")] += in_dense
                counts[term][(group, "sparse_outside_genre")] += outside_genre
            row[term] = "".join(["S" if in_sparse else "-", "D" if in_dense else "-"])
        if row["pair_id"]:
            target_rows.append(row)

    with (out / "corpus_terms.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "group", "songs", "in_sparse", "in_dense", "in_sparse_outside_genre", "sparse_share"])
        for term in TERMS:
            for group in ("(전체)", "솔로", "그룹", "밴드", "듀오"):
                n = totals[group]
                s = counts[term][(group, "sparse")]
                writer.writerow([term, group, n, s, counts[term][(group, "dense")],
                                 counts[term][(group, "sparse_outside_genre")], round(s / n, 4) if n else ""])
    target_rows.sort(key=lambda r: r["pair_id"])
    with (out / "target_terms.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "song_id", *TERMS])
        writer.writeheader()
        writer.writerows(target_rows)

    print(f"곡 {len(songs)} · 정답 곡 {len(target_rows)}")
    print(f"{'낱말':<10}{'전체':>8}{'그룹':>8}{'솔로':>8}{'밴드':>8}   (sparse passage에 낱말이 든 곡 수)")
    for term in TERMS:
        print(f"{term:<10}" + "".join(f"{counts[term][(g, 'sparse')]:>8}" for g in ("(전체)", "그룹", "솔로", "밴드")))
    print(f"{'곡 수':<10}" + "".join(f"{totals[g]:>8}" for g in ("(전체)", "그룹", "솔로", "밴드")))
    print("\n정답 곡 (S=sparse, D=dense):")
    for row in target_rows:
        print(f"  {row['pair_id']} " + " ".join(f"{t}={row[t]}" for t in TERMS))


if __name__ == "__main__":
    main()
