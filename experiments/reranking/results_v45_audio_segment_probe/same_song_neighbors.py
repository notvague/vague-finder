"""타진용: 구간 벡터에 "그 곡다움"이 담겨 있는가 — 텍스트와 무관하게 오디오 쪽만 본다.

구간 하나를 골라 다른 모든 구간 중 가장 가까운 것을 찾았을 때, 그것이 같은 곡의 구간인 비율.
무작위면 (곡당 구간 수 / 전체 구간 수) ≈ 0.1%다. 높을수록 곡을 구별하는 정보가 벡터에 있다.

  venv/bin/python <이 파일> <구간 폴더> [<구간 폴더> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    for folder in map(Path, sys.argv[1:]):
        files = sorted(folder.glob("*.npy"))
        parts = [np.load(f).astype(np.float32) for f in files]
        owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
        x = np.concatenate(parts)
        x = x - x.mean(0)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        hits = adjacent = 0
        index = np.arange(len(x))
        for start in range(0, len(x), 2000):
            sims = x[start:start + 2000] @ x.T
            rows = np.arange(len(sims))
            sims[rows, index[start:start + 2000]] = -np.inf           # 자기 자신 제외
            nearest = sims.argmax(1)
            same = owner[nearest] == owner[start:start + 2000]
            hits += int(same.sum())
            adjacent += int((same & (np.abs(nearest - index[start:start + 2000]) == 1)).sum())
        n = len(x)
        print(f"{folder.name}: 구간 {n}개 · 가장 가까운 구간이 같은 곡 {hits / n * 100:.1f}% "
              f"(그중 바로 옆 구간 {adjacent / n * 100:.1f}%p) · 무작위 기대 {np.mean([len(p) - 1 for p in parts]) / n * 100:.2f}%")


if __name__ == "__main__":
    main()
