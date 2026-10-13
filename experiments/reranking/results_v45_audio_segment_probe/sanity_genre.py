"""타진용 점검: 표본 곡의 구간 평균 벡터가 장르를 가르는가. 모델이 제대로 로드됐는지 보는 용도다.

  PYTHONPATH=. venv/bin/python <이 파일> --segments <폴더> --model <모델> [--corpus data/all_songs.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.embedding.models.audio_clap import CLAPAudioEmbedder

PROBES = [
    ("랩/힙합", "A hip-hop track with rapping."),
    ("발라드", "A slow emotional ballad."),
    ("록/메탈", "A rock song with distorted electric guitars and drums."),
    ("댄스", "An upbeat dance pop track."),
]


def unit(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    order = np.argsort(scores)
    rank = np.empty(len(scores))
    rank[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((rank[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/all_songs.jsonl"))
    args = parser.parse_args()

    files = sorted(args.segments.glob("*.npy"))
    ids = [f.stem for f in files]
    songs = unit(np.stack([np.load(f).astype(np.float64).mean(0) for f in files]))
    songs_c = unit(songs - songs.mean(0))
    genre = {}
    for line in open(args.corpus, encoding="utf-8"):
        if line.strip():
            doc = json.loads(line)
            genre[str(doc["song_id"])] = (doc.get("metadata") or {}).get("genre") or []
    t = CLAPAudioEmbedder(ckpt=args.model).embed_texts([p for _, p in PROBES], l2_normalize=True).astype(np.float64)
    t_c = unit(t - t.mean(0))
    print(f"{args.model} · 표본 {len(ids)}곡")
    print("| 장르 | 해당 곡 | 상위 50 중(그대로) · AUROC | 상위 50 중(센터링) · AUROC |")
    print("|---|---|---|---|")
    for i, (label, _prompt) in enumerate(PROBES):
        positive = np.array([label in genre[sid] for sid in ids])
        cells = []
        for scores in (t[i] @ songs.T, t_c[i] @ songs_c.T):
            cells.append(f"{int(positive[np.argsort(-scores)[:50]].sum())} · {auroc(scores, positive):.2f}")
        print(f"| {label} | {positive.mean() * 100:.0f}% | {cells[0]} | {cells[1]} |")


if __name__ == "__main__":
    main()
