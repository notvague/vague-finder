"""가정 실험: "…이미지"로 끝나는 표지 묘사(t086·t099)를 표지 단서로 인정하면 정답이 어디에 오는가.

규칙은 바꾸지 않는다 — 이 두 질의에 한해 has_explicit_visual_clue를 참으로 바꿔 끼워 분석하고,
그 분석을 캐시(analysis_cache_v10_dev_imgprobe.json)와 2행짜리 질의 CSV로 남긴다.
  PYTHONPATH=. venv/bin/python <이 파일>
"""
from __future__ import annotations

import csv
from pathlib import Path

import src.retrieval.modality_queries as m
from src.backend.api.dependencies import get_query_analyzer
from src.retrieval.analysis_cache import new_cache

OUT = Path(__file__).resolve().parent
IDS = ("t086", "t099")


def main() -> None:
    rows = [r for r in csv.DictReader(open("experiments/reranking/eval_queries_v10.csv", encoding="utf-8")) if r["query_id"] in IDS]
    targets = {r["query"].strip() for r in rows}
    original = m.has_explicit_visual_clue
    m.has_explicit_visual_clue = lambda q: True if str(q).strip() in targets else original(q)
    src = OUT / "eval_queries_v10_imgprobe.csv"
    with src.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cache = new_cache(source=str(src), split="dev")
    analyzer = get_query_analyzer()
    for r in rows:
        a = analyzer.analyze(r["query"].strip())
        cache.put(r["query_id"], r["query"].strip(), a)
        print(r["query_id"], a.modality_weights, "|", a.image_english_query)
    cache.save(OUT / "analysis_cache_v10_dev_imgprobe.json")


if __name__ == "__main__":
    main()
