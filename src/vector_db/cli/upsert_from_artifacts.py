"""
src/vector_db/cli/upsert_from_artifacts.py

실행 예시:
docker compose exec backend python -u -m src.vector_db.cli.upsert_from_artifacts
    --namespace dev

주의:
캐시에 존재하는 id들이 pinecone에 없으면 insert, 기존에 있으면 update
반면 넘겨주지 않은 id는 삭제되지 않고 그대로 남아있음
-> 기존 곡 데이터셋에 삭제된 곡이 있다면 그 곡만 선택하여 삭제하거나
다 지우고 다시 저장하는 과정 필요.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.vector_db.metadata_loader import build_metadata_map
from src.vector_db.upsert_from_artifacts import (
    ArtifactTags,
    upsert_from_artifacts,
    _infer_single_tag,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--namespace", type=str, default="dev")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--only", choices=["all", "text", "image", "audio"], default="all")

    p.add_argument("--text_dense_tag", type=str, default="")
    p.add_argument("--text_sparse_tag", type=str, default="")  # optional
    p.add_argument("--image_tag", type=str, default="")
    p.add_argument("--audio_tag", type=str, default="")

    p.add_argument("--data_dir", type=str, default="/app/data", help="data/**/meta.json 스캔용")
    p.add_argument("--metadata_json", type=str, default="", help="id->metadata 단일 JSON 파일")

    args = p.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    emb = artifacts_dir / "embeddings"

    do_text = args.only in ("all", "text")
    do_image = args.only in ("all", "image")
    do_audio = args.only in ("all", "audio")

    text_dense_tag = ""
    text_sparse_tag = None
    image_tag = ""
    audio_tag = ""

    if do_text:
        text_dense_tag = args.text_dense_tag.strip() or _infer_single_tag(emb / "text_dense")

        if args.text_sparse_tag.strip():
            text_sparse_tag = args.text_sparse_tag.strip()
        else:
            sparse_root = emb / "text_sparse"
            text_sparse_tag = _infer_single_tag(sparse_root) if sparse_root.exists() else None

    if do_image:
        image_tag = args.image_tag.strip() or _infer_single_tag(emb / "image")

    if do_audio:
        audio_tag = args.audio_tag.strip() or _infer_single_tag(emb / "audio")

    tags = ArtifactTags(
        text_dense_tag=text_dense_tag,
        text_sparse_tag=text_sparse_tag,
        image_tag=image_tag,
        audio_tag=audio_tag,
    )

    # metadata map
    data_dir = Path(args.data_dir) if args.data_dir.strip() else None
    metadata_json = Path(args.metadata_json) if args.metadata_json.strip() else None

    metadata_map = build_metadata_map(
        data_dir=data_dir,
        metadata_json=metadata_json,
    )
    print(f"[META] loaded metadata records={len(metadata_map)}")

    upsert_from_artifacts(
        artifacts_dir=artifacts_dir,
        tags=tags,
        metadata_map=metadata_map,
        namespace=args.namespace,
        batch_size=args.batch_size,
        do_text=do_text,
        do_image=do_image,
        do_audio=do_audio,
    )


if __name__ == "__main__":
    main()