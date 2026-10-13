"""타진용: 구간 벡터를 평균내지 않고 쓰면 정답이 위로 오는가. 표본 곡 안에서의 정답 순위를 방식별로 낸다.

  PYTHONPATH=. venv/bin/python <이 파일> --segments <embed_segments.py의 출력 폴더> --artifacts artifacts [--model …] [--out <csv>]

방식(곡 점수를 어떻게 만드나 — t는 질의 프롬프트의 CLAP 텍스트 벡터):
  stored      지금 색인의 벡터(앞 120초 · 12초 구간 8개 평균 · 센터링). 표본으로만 줄인 기준선
  mean_all    곡 전체 구간의 평균 벡터
  mean_c      같은 것을 표본 평균으로 센터링
  max         구간별 유사도의 최댓값 — "그 구간"이 있으면 잡는다
  max_c       구간 벡터를 센터링한 뒤 최댓값
  top3_c      센터링한 구간 유사도 상위 3개의 평균
  intro_c     앞 30초(구간 3개)만 — "도입부" 묘사용
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np

from experiments.reranking.results_v45_audio_segment_probe.pick_sample import audio_queries
from src.embedding.models.audio_clap import CLAPAudioEmbedder


def unit(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", default="laion/clap-htsat-fused")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(args.segments.glob("*.npy"))
    ids = [f.stem for f in files]
    row_of = {sid: i for i, sid in enumerate(ids)}
    seg = [np.load(f).astype(np.float64) for f in files]                 # 곡마다 [구간, 512]
    all_seg = np.concatenate(seg)
    seg_mean = all_seg.mean(0)
    seg_c = [unit(s - seg_mean) for s in seg]
    song_mean = unit(np.stack([s.mean(0) for s in seg]))
    song_mean_c = unit(song_mean - song_mean.mean(0))
    print(f"표본 {len(ids)}곡 · 구간 {len(all_seg)}개(곡당 중앙값 {statistics.median(len(s) for s in seg):.0f})")
    print(f"구간 벡터가 평균 방향과 이루는 코사인 평균 {float((all_seg @ unit(seg_mean)).mean()):.3f} · 한 곡 안 구간끼리 평균 코사인 "
          f"{float(np.mean([(s @ s.T)[~np.eye(len(s), dtype=bool)].mean() for s in seg if len(s) > 1])):.3f}")

    stored_dir = next((args.artifacts / "embeddings/audio").iterdir())
    stored = np.stack([np.load(stored_dir / f"{sid}.npy") for sid in ids]).astype(np.float64)

    queries = [q for q in audio_queries() if any(s in row_of for s in q["relevant"])]
    t = CLAPAudioEmbedder(ckpt=args.model).embed_texts([q["prompt"] for q in queries], l2_normalize=True).astype(np.float64)
    same_model = args.model == "laion/clap-htsat-fused"

    scores = {
        "mean_all": t @ song_mean.T,
        "mean_c": t @ song_mean_c.T,
        "max": np.stack([np.array([(s @ q).max() for s in seg]) for q in t]),
        "max_c": np.stack([np.array([(s @ q).max() for s in seg_c]) for q in t]),
        "top3_c": np.stack([np.array([np.sort(s @ q)[-3:].mean() for s in seg_c]) for q in t]),
        "intro_c": np.stack([np.array([(s[:3] @ q).max() for s in seg_c]) for q in t]),
    }
    if same_model:   # 저장된 벡터는 기존 모델의 것이라 다른 모델의 질의와는 맞지 않는다
        scores = {"stored": t @ stored.T, **scores}

    ranks = {}
    for name, sc in scores.items():
        out = []
        for i, q in enumerate(queries):
            best = max(sc[i, row_of[s]] for s in q["relevant"] if s in row_of)
            out.append(int((sc[i] > best).sum()) + 1)
        ranks[name] = out

    n = len(queries)
    print(f"\n질의 {n}건 · 표본 {len(ids)}곡 안에서의 정답 순위")
    print("| 방식 | 1위 | 5위 안 | 10위 안 | 30위 안 | 중앙값 | 평균 역순위 |")
    print("|---|---|---|---|---|---|---|")
    for name, r in ranks.items():
        print(f"| {name} | {sum(x == 1 for x in r)} | {sum(x <= 5 for x in r)} | {sum(x <= 10 for x in r)} | "
              f"{sum(x <= 30 for x in r)} | {statistics.median(r):.0f} | {sum(1 / x for x in r) / n:.3f} |")

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["set", "query_id", *ranks, "prompt", "query"])
            for i, q in enumerate(queries):
                writer.writerow([q["set"], q["query_id"], *(ranks[name][i] for name in ranks), q["prompt"], q["query"]])
        print(f"\n질의별: {args.out}")


if __name__ == "__main__":
    main()
