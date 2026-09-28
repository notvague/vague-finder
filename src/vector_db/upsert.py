"""
src/vector_db/upsert.py

songs(이미 임베딩이 채워진 리스트)를 Pinecone에 저장

요구사항:
- 텍스트 인덱스 1개: text_dense_values + text_sparse_values 함께 저장
- 이미지 인덱스 1개: image_values 저장
- 오디오 인덱스 1개: audio_values 저장
- metadata는 allowlist만 저장
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

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
    METADATA_ALLOWLIST,
    MAX_METADATA_STR_LEN,
)


# -----------------------------
# Utils
# -----------------------------
def _to_float_list(vec: Any) -> List[float]:
    """np.ndarray -> list[float] 변환(또는 iterable -> list[float])."""
    if vec is None:
        return []
    if isinstance(vec, np.ndarray):
        return vec.astype(np.float32).tolist()
    return [float(x) for x in vec]


def _chunk(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    """업서트 batch 분할."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _sanitize_metadata_value(v: Any) -> Any:
    """
    Pinecone metadata는 JSON-serializable이어야 한다.
    - str/int/float/bool/list/dict 형태만 안전.
    - 그 외는 문자열로 변환(최후수단).

    또한 문자열은 길이 제한을 건다(요약/컨텍스트가 길어질 수 있어서).
    """
    if v is None:
        return None

    # numpy scalar 처리
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)

    if isinstance(v, str):
        s = v.strip()
        return s[:MAX_METADATA_STR_LEN] if len(s) > MAX_METADATA_STR_LEN else s

    if isinstance(v, (int, float, bool)):
        return v

    if isinstance(v, list):
        return [_sanitize_metadata_value(x) for x in v if x is not None]

    if isinstance(v, dict):
        out: Dict[str, Any] = {}
        for kk, vv in v.items():
            if kk is None:
                continue
            out[str(kk)] = _sanitize_metadata_value(vv)
        return out

    # fallback: string
    s = str(v)
    return s[:MAX_METADATA_STR_LEN] if len(s) > MAX_METADATA_STR_LEN else s


def _build_metadata(song: Dict[str, Any]) -> Dict[str, Any]:
    """
    song["metadata"]에서 allowlist 필드만 뽑아서 Pinecone metadata로 구성한다.
    - title, artist, release_date, album_summary, comment_context, mood_tags, scene_summary, youtube_url
    """
    md = song.get("metadata", {})
    if not isinstance(md, dict):
        return {}

    out: Dict[str, Any] = {}
    for k in METADATA_ALLOWLIST:
        if k in md and md[k] is not None:
            sv = _sanitize_metadata_value(md[k])
            if sv is not None:
                out[k] = sv

    return out


def _normalize_sparse(sparse: Any) -> Optional[Dict[str, Any]]:
    """
    BM25 sparse 벡터를 Pinecone 포맷으로 정규화.
      {"indices":[...], "values":[...]} 형태.

    Fail-fast:
    - 길이 불일치/중복 인덱스는 구축 단계에서 바로 에러로 터뜨려 원인 파악.
    """
    if sparse is None:
        return None

    if not isinstance(sparse, dict):
        raise TypeError(f"text_sparse_values must be dict, got: {type(sparse)}")

    indices = sparse.get("indices")
    values = sparse.get("values")

    if indices is None or values is None:
        raise ValueError("text_sparse_values must contain 'indices' and 'values'")

    if len(indices) != len(values):
        raise ValueError(f"sparse length mismatch: indices={len(indices)} values={len(values)}")

    idx_list = [int(i) for i in indices]
    val_list = [float(v) for v in values]

    # 중복 인덱스 존재하면, 점수 합산 등의 정책이 필요해져서 여기서는 에러로 처리.
    if len(set(idx_list)) != len(idx_list):
        raise ValueError("sparse indices contain duplicates. Deduplicate/merge before upsert.")

    # 재현성/디버깅을 위해 정렬
    pairs = sorted(zip(idx_list, val_list), key=lambda x: x[0])
    return {"indices": [p[0] for p in pairs], "values": [p[1] for p in pairs]}


# -----------------------------
# Record builders
# -----------------------------
def _build_text_records(songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    TEXT_HYBRID 인덱스용 레코드 생성.
    - values: dense
    - sparse_values: BM25
    - metadata: allowlist
    """
    records: List[Dict[str, Any]] = []

    for s in songs:
        song_id = s.get("id")
        if not song_id:
            raise ValueError("song must have non-empty 'id'")

        dense = s.get("text_dense_values")
        if dense is None:
            # 텍스트 임베딩이 없으면 텍스트 인덱스에 넣을 수 없음
            continue

        dense_list = _to_float_list(dense)
        if len(dense_list) != TEXT_DENSE_DIM:
            raise ValueError(
                f"text_dense_values dimension mismatch: got {len(dense_list)}, expected {TEXT_DENSE_DIM}. "
                "Ko-E5 출력 차원과 인덱스 차원을 맞춰야 함."
            )

        sparse_norm = _normalize_sparse(s.get("text_sparse_values"))
        meta = _build_metadata(s)

        rec: Dict[str, Any] = {
            "id": str(song_id),
            "values": dense_list,
        }
        if sparse_norm is not None:
            rec["sparse_values"] = sparse_norm
        if meta:
            rec["metadata"] = meta

        records.append(rec)

    return records


def _build_image_records(songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    IMAGE 인덱스용 레코드 생성.
    - values: image embedding
    - metadata: (동일 allowlist)
    """
    records: List[Dict[str, Any]] = []

    for s in songs:
        song_id = s.get("id")
        if not song_id:
            raise ValueError("song must have non-empty 'id'")

        img = s.get("image_values")
        if img is None:
            continue

        img_list = _to_float_list(img)
        if len(img_list) != IMAGE_DIM:
            raise ValueError(
                f"image_values dimension mismatch: got {len(img_list)}, expected {IMAGE_DIM}. "
                "SigLIP2 출력 차원과 인덱스 차원을 맞춰야 함."
            )

        meta = _build_metadata(s)

        rec: Dict[str, Any] = {"id": str(song_id), "values": img_list}
        if meta:
            rec["metadata"] = meta

        records.append(rec)

    return records


def _build_audio_records(songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    AUDIO 인덱스용 레코드 생성.
    - values: audio embedding (CLAP mean, 512 dim)
    - metadata: (동일 allowlist)
    """
    records: List[Dict[str, Any]] = []

    for s in songs:
        song_id = s.get("id")
        if not song_id:
            raise ValueError("song must have non-empty 'id'")

        aud = s.get("audio_values")
        if aud is None:
            continue

        aud_list = _to_float_list(aud)
        if len(aud_list) != AUDIO_DIM:
            raise ValueError(
                f"audio_values dimension mismatch: got {len(aud_list)}, expected {AUDIO_DIM}. "
                "CLAP 출력 차원과 인덱스 차원을 맞춰야 함."
            )

        meta = _build_metadata(s)

        rec: Dict[str, Any] = {"id": str(song_id), "values": aud_list}
        if meta:
            rec["metadata"] = meta

        records.append(rec)

    return records


# -----------------------------
# Public API
# -----------------------------
def upsert_songs_to_pinecone(songs: List[Dict[str, Any]]) -> None:
    """
    인덱스 생성 + 업서트 수행.
    """
    pc = get_pinecone_client()

    # 1) ensure indexes
    ensure_text_index(pc)
    ensure_image_index(pc)
    ensure_audio_index(pc)

    text_index = pc.Index(TEXT_HYBRID_INDEX_NAME)
    image_index = pc.Index(IMAGE_INDEX_NAME)
    audio_index = pc.Index(AUDIO_INDEX_NAME)

    # 2) build records
    text_records = _build_text_records(songs)
    image_records = _build_image_records(songs)
    audio_records = _build_audio_records(songs)

    # 3) upsert
    for batch in _chunk(text_records, UPSERT_BATCH_SIZE):
        text_index.upsert(vectors=batch, namespace=NAMESPACE)

    for batch in _chunk(image_records, UPSERT_BATCH_SIZE):
        image_index.upsert(vectors=batch, namespace=NAMESPACE)

    for batch in _chunk(audio_records, UPSERT_BATCH_SIZE):
        audio_index.upsert(vectors=batch, namespace=NAMESPACE)