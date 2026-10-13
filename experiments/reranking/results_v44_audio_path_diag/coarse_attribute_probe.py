"""소리 경로가 **굵은 속성**(장르·보컬 성별)은 알아듣는지 본다 — 곡 하나를 못 찾더라도 신호 자체가 있는가.

프롬프트 하나로 5,143곡을 줄 세우고, 곡 메타데이터(멜론 장르 · 보컬 성별)를 정답으로 삼아
상위 100곡 중 맞는 곡의 비율과 AUROC(0.5 = 무작위, 1.0 = 완벽)를 낸다. Qdrant·Gemini 없음.

  PYTHONPATH=. venv/bin/python <이 파일> [--artifacts artifacts] [--corpus data/all_songs.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.embedding.models.audio_clap import CLAPAudioEmbedder

# (라벨, 프롬프트, 메타데이터 판정)
PROBES = [
    ("장르: 랩/힙합", "A hip-hop track with rapping.", lambda m: "랩/힙합" in m["genre"]),
    ("장르: 발라드", "A slow emotional ballad.", lambda m: "발라드" in m["genre"]),
    ("장르: 댄스", "An upbeat dance pop track.", lambda m: "댄스" in m["genre"]),
    ("장르: 록/메탈", "A rock song with distorted electric guitars and drums.", lambda m: "록/메탈" in m["genre"]),
    ("장르: R&B/Soul", "A smooth R&B soul track.", lambda m: "R&B/Soul" in m["genre"]),
    ("장르: 포크/블루스", "An acoustic folk song with acoustic guitar.", lambda m: "포크/블루스" in m["genre"]),
    ("장르: 성인가요/트로트", "A Korean trot song.", lambda m: "성인가요/트로트" in m["genre"]),
    ("보컬: 여성", "A song with female vocals.", lambda m: m["vocal_gender"] == "여성"),
    ("보컬: 남성", "A song with male vocals.", lambda m: m["vocal_gender"] == "남성"),
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
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--corpus", type=Path, default=Path("data/all_songs.jsonl"))
    args = parser.parse_args()

    folder = next((args.artifacts / "embeddings/audio").iterdir())
    files = sorted(folder.glob("*.npy"))
    ids = [f.stem for f in files]
    c = np.stack([np.load(f) for f in files]).astype(np.float64)
    mean = np.load(args.artifacts / "audio_mean.npy").astype(np.float64)
    dot = c @ mean
    v = (-dot + np.sqrt(dot**2 + 1.0 - float(mean @ mean)))[:, None] * c + mean

    meta = {}
    for line in open(args.corpus, encoding="utf-8"):
        if line.strip():
            doc = json.loads(line)
            md = doc.get("metadata") or {}
            meta[str(doc["song_id"])] = {"genre": md.get("genre") or [], "vocal_gender": md.get("vocal_gender") or ""}
    metas = [meta[i] for i in ids]

    t = CLAPAudioEmbedder().embed_texts([p[1] for p in PROBES], l2_normalize=True).astype(np.float64)
    t_centered = unit(t - t.mean(0, keepdims=True))        # 프롬프트 공통 성분("a song with …")을 뺀 것
    spaces = {
        "지금(센터링한 곡)": t @ c.T,
        "센터링 없는 곡": t @ v.T,
        "프롬프트 공통 성분 제거": t_centered @ c.T,
    }
    print("| 속성 | 해당 곡 비율 | " + " | ".join(f"{name}: 상위 100 중 · AUROC" for name in spaces) + " |")
    print("|---|---|" + "---|" * len(spaces))
    for i, (label, _prompt, test) in enumerate(PROBES):
        positive = np.array([bool(test(m)) for m in metas])
        cells = []
        for scores in spaces.values():
            top = np.argsort(-scores[i])[:100]
            cells.append(f"{int(positive[top].sum())} · {auroc(scores[i], positive):.2f}")
        print(f"| {label} | {positive.mean() * 100:.0f}% ({int(positive.sum())}곡) | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
