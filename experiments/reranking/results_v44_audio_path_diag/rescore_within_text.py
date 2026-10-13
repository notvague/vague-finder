"""소리 신호를 독립 경로가 아니라 **텍스트 후보의 재채점**에 쓰면 어떤지 본다(Gemini 없음).

지금은 소리 경로가 전곡에서 상위 30곡을 따로 뽑아 텍스트 30곡과 가중 RRF로 섞는다. 소리 경로는 장르 수준만
구분하므로(coarse_attribute_probe.py) 그 30곡은 대개 "같은 장르의 다른 곡"이고 텍스트 후보를 밀어낸다.
대안: 텍스트 경로가 뽑은 후보 안에서만 소리 유사도로 순서를 다시 매긴다 — 소리만으로 들어오는 곡이 없다.

텍스트 경로만 실제 서비스 코드로 돌리고(`SearchRouter._search_text`), 융합은 여기서 같은 식으로 계산한다.
보조 경로·가산·표지 경로는 빠져 있어 **후보 풀 전체의 재현은 아니다** — 두 방식의 방향을 보는 용도다.

  QDRANT_PATH=<색인> BM25_PARAMS_PATH=<bm25> PYTHONPATH=. venv/bin/python <이 파일> [--artifacts artifacts]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.backend.api.dependencies import get_audio_embedder, get_search_router
from src.backend.schemas.query import QueryAnalysis

ROOT = Path("experiments/reranking")
SETS = {
    "v06": (ROOT / "eval_queries_v06.csv", ROOT / "analysis_cache_v06_dev.json"),
    "v09": (ROOT / "eval_queries_v09.csv", ROOT / "results_v42_modality_degrade/analysis_cache_v09_dev_degrade.json"),
    "v10": (ROOT / "eval_queries_v10.csv", ROOT / "results_v41_cover_description_image_noun/analysis_cache_v10_dev_imagenoun.json"),
}
RRF_K = 60
DEPTHS = (30, 90)


def rrf(lists: list[tuple[float, list[str]]]) -> list[str]:
    """[(가중, 순위 목록)] → 가중 RRF로 합친 순서. search_router._rrf_fuse와 같은 식."""
    score: dict = {}
    for weight, ids in lists:
        for rank, sid in enumerate(ids, 1):
            score[sid] = score.get(sid, 0.0) + weight / (RRF_K + rank)
    return [sid for sid, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def rank_of(order: list[str], relevant: set) -> int | None:
    return next((i for i, sid in enumerate(order, 1) if sid in relevant), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=ROOT / "results_v44_audio_path_diag/rescore_within_text.csv")
    args = parser.parse_args()

    folder = next((args.artifacts / "embeddings/audio").iterdir())
    files = sorted(folder.glob("*.npy"))
    ids = [f.stem for f in files]
    row_of = {sid: i for i, sid in enumerate(ids)}
    c = np.stack([np.load(f) for f in files]).astype(np.float64)

    router = get_search_router()
    clap = get_audio_embedder()
    rows = []
    try:
        for name, (csv_path, cache_path) in SETS.items():
            entries = json.loads(cache_path.read_text(encoding="utf-8"))["entries"]
            for row in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
                if row["split"] != "dev" or not row["relevant_ids"].strip():
                    continue
                analysis = QueryAnalysis(**entries[row["query_id"]]["analysis"])
                w = analysis.modality_weights
                # 표지 경로가 같이 도는 질의는 뺀다 — 여기서는 텍스트·소리 두 경로만 재현한다.
                if (not analysis.audio_english_query.strip() or w.audio <= 0 or analysis.has_exact_lyric_clue
                        or (analysis.has_visual_clue and analysis.image_english_query.strip() and w.image > 0)):
                    continue
                relevant = {s.strip() for s in row["relevant_ids"].replace(",", "|").split("|") if s.strip()}
                text = [str(t.id) for t in router._search_text(analysis, max(DEPTHS))]
                q = clap.embed_texts([analysis.audio_english_query.strip()], l2_normalize=True)[0].astype(np.float64)
                sims = c @ q
                audio_global = [ids[i] for i in np.argsort(-sims)[:30]]
                w_text, w_audio = w.text / (w.text + w.audio), w.audio / (w.text + w.audio)
                out = {"set": name, "query_id": row["query_id"], "w_audio": round(w_audio, 3),
                       "text_rank": rank_of(text, relevant),
                       "audio_rank_all": int((sims > max(sims[row_of[s]] for s in relevant if s in row_of)).sum()) + 1}
                # 지금 방식: 텍스트 30 + 전곡 소리 30을 섞는다
                out["now"] = rank_of(rrf([(w_text, text[:30]), (w_audio, audio_global)]), relevant)
                out["text_only"] = rank_of(text[:30], relevant)
                for depth in DEPTHS:
                    pool = text[:depth]
                    by_audio = sorted(pool, key=lambda sid: -sims[row_of[sid]] if sid in row_of else 0.0)
                    out[f"rescore{depth}"] = rank_of(rrf([(w_text, pool), (w_audio, by_audio)]), relevant)
                    # 소리 가중을 낮춘 판(0.25배) — 순서를 뒤집기보다 동점을 가르는 정도
                    out[f"rescore{depth}_light"] = rank_of(rrf([(w_text, pool), (w_audio * 0.25, by_audio)]), relevant)
                rows.append(out)
    finally:
        router.shutdown()

    keys = ["now", "text_only", *(f"rescore{d}{s}" for d in DEPTHS for s in ("", "_light"))]
    n = len(rows)
    print(f"질의 {n}건(텍스트·소리 두 경로만 도는 dev 질의) · 정답이 텍스트 30위 안 {sum(1 for r in rows if r['text_rank'] and r['text_rank'] <= 30)} · 90위 안 {sum(1 for r in rows if r['text_rank'])}")
    print()
    print("| 방식 | 정답이 30위 안 | 10위 안 | 1위 | MRR@30 |")
    print("|---|---|---|---|---|")
    for key in keys:
        ranks = [r[key] for r in rows]
        inside = [x for x in ranks if x and x <= 30]
        mrr = sum(1 / x for x in inside) / n
        print(f"| {key} | {len(inside)} | {sum(1 for x in inside if x <= 10)} | {sum(1 for x in inside if x == 1)} | {mrr:.3f} |")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n질의별: {args.out}")


if __name__ == "__main__":
    main()
