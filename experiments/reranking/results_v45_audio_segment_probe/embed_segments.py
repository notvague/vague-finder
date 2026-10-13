"""타진용: 곡을 10초 구간으로 잘라 구간마다 CLAP 벡터를 만든다(평균내지 않는다). 곡 전체를 쓴다.

지금 색인은 앞 120초에서 12초 구간 8개를 뽑아 **평균낸 벡터 하나**다. 구간을 가리키는 묘사("도입부에 사이렌 소리")가
평균에서 지워지는지 보려면 구간 벡터가 따로 있어야 한다. 전곡을 다시 임베딩하는 대신 표본 곡만 한다.

  PYTHONPATH=. venv/bin/python <이 파일> --ids <곡 id 목록 파일> --data data/raw_5143 --out <폴더> [--model laion/clap-htsat-fused]

곡마다 <out>/<song_id>.npy  [구간 수, 512] (L2 정규화, 센터링 없음). 이미 있으면 건너뛴다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.embedding.audio.audio_io import load_audio_mono
from src.embedding.models.audio_clap import CLAPAudioEmbedder

SR = 48_000
SEGMENT_SECONDS = 10      # CLAP의 기본 입력 길이. 이보다 길면 fusion(무작위 구간 합성)이 끼어든다
MAX_SECONDS = 420         # 7분에서 자른다(라이브 · 메들리 대비)
MIN_TAIL_SECONDS = 5      # 마지막 조각이 이보다 짧으면 버린다


def segments(wav: np.ndarray) -> list[np.ndarray]:
    size = SR * SEGMENT_SECONDS
    out = [wav[i:i + size] for i in range(0, len(wav), size)]
    if len(out) > 1 and len(out[-1]) < SR * MIN_TAIL_SECONDS:
        out.pop()
    return [np.ascontiguousarray(s, dtype=np.float32) for s in out]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/raw_5143"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="laion/clap-htsat-fused")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    wanted = [line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    folder_of = {p.name.rsplit("_", 1)[-1]: p for p in args.data.iterdir()}
    args.out.mkdir(parents=True, exist_ok=True)
    embedder = CLAPAudioEmbedder(ckpt=args.model)

    started, done, skipped, n_segments = time.time(), 0, [], 0
    for index, song_id in enumerate(wanted, 1):
        target = args.out / f"{song_id}.npy"
        if target.exists():
            continue
        audio = folder_of[song_id] / "audio.m4a" if song_id in folder_of else None
        try:
            if audio is None or not audio.exists():
                raise FileNotFoundError("audio.m4a 없음")
            parts = segments(load_audio_mono(audio, target_sr=SR, max_seconds=MAX_SECONDS))
            vectors = [embedder.embed_audios(parts[i:i + args.batch], sampling_rate=SR, l2_normalize=True)
                       for i in range(0, len(parts), args.batch)]
            np.save(target, np.concatenate(vectors).astype(np.float32))
        except Exception as exc:  # 한 곡 때문에 멈추지 않는다
            skipped.append((song_id, f"{type(exc).__name__}: {exc}"))
            continue
        done += 1
        n_segments += len(parts)
        if done % 25 == 0:
            rate = (time.time() - started) / done
            print(f"[{index}/{len(wanted)}] {done}곡 · 구간 {n_segments} · 곡당 {rate:.2f}초", flush=True)
    print(f"끝: 새로 {done}곡 · 구간 {n_segments} · {time.time() - started:.0f}초 · 건너뜀 {len(skipped)}")
    for song_id, reason in skipped[:20]:
        print("  건너뜀", song_id, reason)


if __name__ == "__main__":
    main()
