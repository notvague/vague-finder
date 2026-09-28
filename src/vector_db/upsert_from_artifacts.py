"""
src/vector_db/upsert_from_artifacts.py

목표:
- artifacts/embeddings/ 아래에 저장된 결과물(.npy/.npz)만 사용해서 Pinecone에 upsert한다.

지원 모달리티:
- text_hybrid: text_dense(.npy) + text_sparse(.npz) -> TEXT_HYBRID_INDEX
- image: image(.npy) -> IMAGE_INDEX
- audio: audio(.npy) -> AUDIO_INDEX

- sparse npz는 {"indices": int[], "values": float[]} 포맷으로 Pinecone sparse_values에 넣는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.vector_db.indexes import ensure_text_index, ensure_image_index, ensure_audio_index
from src.vector_db.pinecone_client import get_pinecone_client
from src.vector_db.settings import (
    TEXT_HYBRID_INDEX_NAME,
    IMAGE_INDEX_NAME,
    AUDIO_INDEX_NAME,
    TEXT_DENSE_DIM,
    IMAGE_DIM,
    AUDIO_DIM,
    NAMESPACE,
    UPSERT_BATCH_SIZE,
)


# -----------------------------
# Utilities
# -----------------------------
def _chunk(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _infer_single_tag(modality_dir: Path) -> str:
    if not modality_dir.exists():
        raise FileNotFoundError(f"Missing embeddings dir: {modality_dir}")

    tags = [p.name for p in modality_dir.iterdir() if p.is_dir()]
    if len(tags) == 1:
        return tags[0]
    if len(tags) == 0:
        raise FileNotFoundError(f"No model_tag directories under: {modality_dir}")
    raise ValueError(f"Multiple model_tag dirs found under {modality_dir}. Please specify tag explicitly: {tags}")


def _load_npy_vec(path: Path, expected_dim: int) -> List[float]:
    v = np.load(str(path))
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.shape[0] != expected_dim:
        raise ValueError(f"Dimension mismatch: {path} -> {v.shape[0]} (expected {expected_dim})")
    return v.tolist()


def _load_npz_sparse(path: Path) -> Dict[str, Any]:
    """
    np.savez_compressed로 저장된 sparse 캐시 로드.
    포맷:
      - indices: int64 1D
      - values : float32 1D
    """
    z = np.load(str(path))
    idx = np.asarray(z["indices"], dtype=np.int64).reshape(-1)
    val = np.asarray(z["values"], dtype=np.float32).reshape(-1)
    if idx.shape[0] != val.shape[0]:
        raise ValueError(f"Sparse length mismatch: {path} -> indices={idx.shape[0]} values={val.shape[0]}")
    # Pinecone payload 안전성을 위해 list로 변환
    return {"indices": [int(i) for i in idx.tolist()], "values": [float(v) for v in val.tolist()]}


def _list_ids_from_dir(dir_path: Path, ext: str) -> List[str]:
    """
    <id>.<ext> 형태 파일에서 id 목록 추출.
    """
    if not dir_path.exists():
        return []
    out: List[str] = []
    for p in dir_path.glob(f"*.{ext}"):
        out.append(p.stem)
    out.sort()
    return out


# -----------------------------
# Core: Build records from artifacts
# -----------------------------
@dataclass(frozen=True)
class ArtifactTags:
    text_dense_tag: str
    text_sparse_tag: Optional[str]
    image_tag: str
    audio_tag: str


def build_text_hybrid_records(
    artifacts_dir: Path,
    *,
    text_dense_tag: str,
    text_sparse_tag: Optional[str],
    metadata_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    text hybrid 인덱스에 넣을 vectors payload를 artifacts에서 구성한다.

    - dense는 필수 (text_dense/<tag>/<id>.npy)
    - sparse는 있으면 추가 (text_sparse/<tag>/<id>.npz)
      * sparse tag가 None이거나 파일이 없으면 dense-only로 upsert

    반환 레코드 형식:
      {"id": "...", "values": [...], "sparse_values": {...}} (sparse는 optional)
    """
    base = Path(artifacts_dir) / "embeddings"

    dense_dir = base / "text_dense" / text_dense_tag
    dense_ids = _list_ids_from_dir(dense_dir, "npy")
    if not dense_ids:
        raise FileNotFoundError(f"No text_dense npy files found: {dense_dir}")

    # sparse는 optional
    sparse_dir = None
    sparse_ids_set = set()
    if text_sparse_tag:
        sparse_dir = base / "text_sparse" / text_sparse_tag
        sparse_ids = _list_ids_from_dir(sparse_dir, "npz")
        sparse_ids_set = set(sparse_ids)

    records: List[Dict[str, Any]] = []

    for tid in dense_ids:
        dense_path = dense_dir / f"{tid}.npy"
        dense_values = _load_npy_vec(dense_path, expected_dim=TEXT_DENSE_DIM)

        rec: Dict[str, Any] = {"id": tid, "values": dense_values}

        if sparse_dir is not None and tid in sparse_ids_set:
            sp_path = sparse_dir / f"{tid}.npz"
            rec["sparse_values"] = _load_npz_sparse(sp_path)

        md = metadata_map.get(tid)
        if md:
            rec["metadata"] = md

        records.append(rec)

    return records


def build_dense_only_records(
    artifacts_dir: Path,
    *,
    modality: str,
    model_tag: str,
    expected_dim: int,
    metadata_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    image/audio 같이 dense-only 모달리티 레코드 생성.
    """
    base = Path(artifacts_dir) / "embeddings" / modality / model_tag
    ids = _list_ids_from_dir(base, "npy")
    if not ids:
        raise FileNotFoundError(f"No {modality} npy files found: {base}")

    records: List[Dict[str, Any]] = []
    for tid in ids:
        path = base / f"{tid}.npy"
        values = _load_npy_vec(path, expected_dim=expected_dim)

        rec: Dict[str, Any] = {"id": tid, "values": values}
        md = metadata_map.get(tid)
        if md:
            rec["metadata"] = md

        records.append(rec)

    return records


# -----------------------------
# Public API: Upsert
# -----------------------------
def upsert_from_artifacts(
    *,
    artifacts_dir: Path,
    tags: ArtifactTags,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    namespace: str = NAMESPACE,
    batch_size: int = UPSERT_BATCH_SIZE,
    do_text: bool = True,
    do_image: bool = True,
    do_audio: bool = True,
) -> None:
    """
    artifacts/embeddings/* 만 읽어서 Pinecone upsert 수행.
    """
    metadata_map = metadata_map or {}

    pc = get_pinecone_client()

    # 1) ensure indexes
    if do_text:
        ensure_text_index(pc)
    if do_image:
        ensure_image_index(pc)
    if do_audio:
        ensure_audio_index(pc)

    # 2) build records
    if do_text:
        text_records = build_text_hybrid_records(
            artifacts_dir,
            text_dense_tag=tags.text_dense_tag,
            text_sparse_tag=tags.text_sparse_tag,
            metadata_map=metadata_map,
        )
        idx = pc.Index(TEXT_HYBRID_INDEX_NAME)
        for batch in _chunk(text_records, batch_size):
            idx.upsert(vectors=batch, namespace=namespace)
        print(f"[OK] upserted text records={len(text_records)} index={TEXT_HYBRID_INDEX_NAME} ns={namespace}")

    if do_image:
        image_records = build_dense_only_records(
            artifacts_dir,
            modality="image",
            model_tag=tags.image_tag,
            expected_dim=IMAGE_DIM,
            metadata_map=metadata_map,
        )
        idx = pc.Index(IMAGE_INDEX_NAME)
        for batch in _chunk(image_records, batch_size):
            idx.upsert(vectors=batch, namespace=namespace)
        print(f"[OK] upserted image records={len(image_records)} index={IMAGE_INDEX_NAME} ns={namespace}")

    if do_audio:
        audio_records = build_dense_only_records(
            artifacts_dir,
            modality="audio",
            model_tag=tags.audio_tag,
            expected_dim=AUDIO_DIM,
            metadata_map=metadata_map,
        )
        idx = pc.Index(AUDIO_INDEX_NAME)
        for batch in _chunk(audio_records, batch_size):
            idx.upsert(vectors=batch, namespace=namespace)
        print(f"[OK] upserted audio records={len(audio_records)} index={AUDIO_INDEX_NAME} ns={namespace}")