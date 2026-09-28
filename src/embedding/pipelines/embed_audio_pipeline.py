from pathlib import Path
import numpy as np

from src.embedding.update_mongo import mark_audio_embedded
from src.embedding.audio_embedder import embed_audio  

RAW_ROOT = Path("data/raw")
OUT_ROOT = Path("data/processed/audio_emb")


def process_song(song_dir: Path):
    audio_path = song_dir / "audio.m4a"
    if not audio_path.exists():
        return

    # 1. 임베딩 생성
    embedding = embed_audio(audio_path)

    # 2. 저장 경로 자동 생성
    out_dir = OUT_ROOT / song_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "audio_clap.npy"
    np.save(out_path, embedding)

    # 3. MongoDB 자동 업데이트
    mark_audio_embedded(
        song_id=song_dir.name,
        embedding_path=str(out_path),
    )


def main():
    for song_dir in RAW_ROOT.iterdir():
        if song_dir.is_dir():
            process_song(song_dir)


if __name__ == "__main__":
    main()
