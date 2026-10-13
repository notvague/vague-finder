"""소리 경로만 떼어 본다 — 저장된 곡 벡터와 질의 프롬프트로 정답 순위를 다시 계산한다(Qdrant·Gemini 없음).

곡 벡터는 `normalize(v - mean)`으로 저장돼 있고 질의는 평균을 빼지 않은 CLAP 텍스트 벡터다.
v가 단위 벡터였고 mean이 남아 있어서 v를 정확히 되살릴 수 있다:
    v = a·c + mean,  a = -(c·mean) + sqrt((c·mean)² + 1 - |mean|²)
그래서 "센터링을 안 했다면", "질의 쪽도 맞췄다면"을 재임베딩 없이 볼 수 있다.

  PYTHONPATH=. venv/bin/python <이 파일> [--artifacts artifacts] [--out <csv>]

변형:
  current        지금 서비스: 질의 t  ·  곡 c(센터링)
  raw            센터링 없음: 질의 t  ·  곡 v
  q_minus_amean  질의에서 곡 평균을 뺌: normalize(t - mean) · c
  q_center       질의에서 **질의 평균**을 뺌(자기 자신 제외): normalize(t - mean_q) · c
  q_center_raw   같은 것을 센터링 안 한 곡에: normalize(t - mean_q) · v
  csls           곡마다 "아무 질의에나 가까운 정도"를 뺌: c·t - (다른 질의들과의 평균 유사도)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from src.backend.schemas.query import QueryAnalysis
from src.embedding.models.audio_clap import CLAPAudioEmbedder

ROOT = Path("experiments/reranking")
SETS = {
    "v06": (ROOT / "eval_queries_v06.csv", ROOT / "analysis_cache_v06_dev.json"),
    "v09": (ROOT / "eval_queries_v09.csv", ROOT / "results_v42_modality_degrade/analysis_cache_v09_dev_degrade.json"),
    "v10": (ROOT / "eval_queries_v10.csv", ROOT / "results_v41_cover_description_image_noun/analysis_cache_v10_dev_imagenoun.json"),
}


def unit(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def load_queries() -> list[dict]:
    """소리 경로가 실제로 도는 dev 질의만(프롬프트 있음 · 가중 > 0 · 가사 원문 단서 없음 — 라우터와 같은 조건)."""
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
                "set": name, "query_id": row["query_id"], "prompt": analysis.audio_english_query.strip(),
                "relevant": [s for s in row["relevant_ids"].replace(",", "|").split("|") if s.strip()],
                "audio_weight": analysis.modality_weights.audio,
            })
    return out


def ranks(scores: np.ndarray, ids: list[str], relevant: list[list[str]]) -> list[int]:
    """질의마다 정답(여럿이면 가장 좋은 것)의 순위. scores: [질의, 곡]"""
    index = {sid: i for i, sid in enumerate(ids)}
    out = []
    for q, rel in enumerate(relevant):
        cols = [index[s] for s in rel if s in index]
        best = max(scores[q, c] for c in cols)
        out.append(int((scores[q] > best).sum()) + 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=ROOT / "results_v44_audio_path_diag/audio_rank_variants.csv")
    args = parser.parse_args()

    folder = next((args.artifacts / "embeddings/audio").iterdir())
    files = sorted(folder.glob("*.npy"))
    ids = [f.stem for f in files]
    c = np.stack([np.load(f) for f in files]).astype(np.float64)          # 저장된 곡 벡터(센터링됨)
    mean = np.load(args.artifacts / "audio_mean.npy").astype(np.float64)
    dot = c @ mean
    a = -dot + np.sqrt(dot**2 + 1.0 - float(mean @ mean))
    v = a[:, None] * c + mean                                              # 센터링 전 곡 벡터
    print(f"곡 {len(ids)} · |mean| {np.linalg.norm(mean):.3f} · 복원한 |v| {np.linalg.norm(v, axis=1).min():.4f}~{np.linalg.norm(v, axis=1).max():.4f}")
    print(f"센터링 전 곡끼리 평균 코사인 {float((v @ v.mean(0)).mean() / np.linalg.norm(v.mean(0))):.3f} (1에 가까울수록 한 방향에 몰려 있다)")

    queries = load_queries()
    t = CLAPAudioEmbedder().embed_texts([q["prompt"] for q in queries], l2_normalize=True).astype(np.float64)
    n = len(queries)
    relevant = [q["relevant"] for q in queries]
    print(f"질의 {n} · 질의↔곡 평균 코사인: 센터링 전 {float((t @ v.T).mean()):.3f} · 센터링 뒤 {float((t @ c.T).mean()):.3f}")
    print(f"질의끼리 평균 코사인 {float((t @ t.T)[~np.eye(n, dtype=bool)].mean()):.3f}")

    loo_mean = (t.sum(0, keepdims=True) - t) / (n - 1)                     # 자기 자신을 뺀 질의 평균
    t_c = unit(t - loo_mean)
    base = t @ c.T
    hub = (base.sum(0, keepdims=True) - base) / (n - 1)                    # 곡마다: 다른 질의들과의 평균 유사도
    variants = {
        "current": base,
        "raw": t @ v.T,
        "q_minus_amean": unit(t - mean) @ c.T,
        "q_center": t_c @ c.T,
        "q_center_raw": t_c @ v.T,
        "csls": base - hub,
    }

    # 허브: 질의마다 상위 10곡을 뽑았을 때 같은 곡이 얼마나 되풀이되나
    rows = {name: ranks(scores, ids, relevant) for name, scores in variants.items()}
    print()
    print("| 변형 | 10위 안 | 30위 안 | 100위 안 | 중앙값 | 상위 10에 한 번이라도 든 곡 수 | 가장 자주 든 곡의 등장 횟수 |")
    print("|---|---|---|---|---|---|---|")
    for name, scores in variants.items():
        r = rows[name]
        top = np.argsort(-scores, axis=1)[:, :10]
        counts = np.bincount(top.ravel(), minlength=len(ids))
        print(f"| {name} | {sum(x <= 10 for x in r)} | {sum(x <= 30 for x in r)} | {sum(x <= 100 for x in r)} | "
              f"{statistics.median(r):.0f} | {int((counts > 0).sum())} / {n * 10} | {int(counts.max())} / {n} |")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["set", "query_id", "audio_weight", *variants, "prompt"])
        for i, q in enumerate(queries):
            writer.writerow([q["set"], q["query_id"], q["audio_weight"], *(rows[name][i] for name in variants), q["prompt"]])
    print(f"\n질의별 순위: {args.out}")


if __name__ == "__main__":
    main()
