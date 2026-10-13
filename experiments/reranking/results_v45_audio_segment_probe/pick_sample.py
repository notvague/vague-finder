"""타진용 표본 곡 목록을 만든다 — 소리 경로가 도는 dev 질의의 정답 곡 전부 + 코퍼스에서 무작위로 뽑은 곡.

무작위 표본이라 장르 분포가 코퍼스와 같다. 표본 안에서의 순위는 전곡 순위에 (전곡 수 / 표본 수)를 곱한 것과 견줄 수 있다.

  PYTHONPATH=. venv/bin/python <이 파일> --artifacts artifacts --distractors 600 --out <목록 파일>
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from src.backend.schemas.query import QueryAnalysis

ROOT = Path("experiments/reranking")
SETS = {
    "v06": (ROOT / "eval_queries_v06.csv", ROOT / "analysis_cache_v06_dev.json"),
    "v09": (ROOT / "eval_queries_v09.csv", ROOT / "results_v42_modality_degrade/analysis_cache_v09_dev_degrade.json"),
    "v10": (ROOT / "eval_queries_v10.csv", ROOT / "results_v41_cover_description_image_noun/analysis_cache_v10_dev_imagenoun.json"),
}
SEED = 20261012


def audio_queries() -> list[dict]:
    out = []
    for name, (csv_path, cache_path) in SETS.items():
        entries = json.loads(cache_path.read_text(encoding="utf-8"))["entries"]
        for row in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
            if row["split"] != "dev" or not row["relevant_ids"].strip():
                continue
            analysis = QueryAnalysis(**entries[row["query_id"]]["analysis"])
            if (not analysis.audio_english_query.strip() or analysis.modality_weights.audio <= 0
                    or analysis.has_exact_lyric_clue):
                continue
            out.append({
                "set": name, "query_id": row["query_id"], "query": row["query"],
                "prompt": analysis.audio_english_query.strip(),
                "relevant": [s.strip() for s in row["relevant_ids"].replace(",", "|").split("|") if s.strip()],
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--distractors", type=int, default=600)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    corpus = sorted(f.stem for f in next((args.artifacts / "embeddings/audio").iterdir()).glob("*.npy"))
    answers = sorted({sid for q in audio_queries() for sid in q["relevant"] if sid in set(corpus)})
    rest = [sid for sid in corpus if sid not in set(answers)]
    random.Random(SEED).shuffle(rest)
    picked = answers + rest[: args.distractors]
    args.out.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"정답 곡 {len(answers)} + 무작위 {args.distractors} = {len(picked)}곡 → {args.out}")


if __name__ == "__main__":
    main()
