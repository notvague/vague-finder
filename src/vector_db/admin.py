"""
src/vector_db/admin.py

Vector DB 운영/관리 기능:
1) fetch/get_by_id  : 특정 레코드 확인(sparse_values 포함 여부 검증)
2) stats            : namespace별 vector count 확인
3) delete           : ids 삭제 / filter 삭제 / namespace 전체 삭제
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from src.vector_db.pinecone_client import get_pinecone_client
from src.vector_db.indexes import ensure_text_index, ensure_image_index, ensure_audio_index
from src.vector_db.settings import (
    TEXT_HYBRID_INDEX_NAME,
    IMAGE_INDEX_NAME,
    AUDIO_INDEX_NAME,
    TEXT_DENSE_DIM,
    IMAGE_DIM,
    AUDIO_DIM,
    NAMESPACE,
)

IndexKind = Literal["text", "image", "audio"]


def _get_index(kind: IndexKind):
    """
    kind에 따라 Pinecone Index 객체를 반환.
    - ensure_index까지 수행해서, 인덱스가 없는 경우도 자동 생성되게 함.
    """
    pc = get_pinecone_client()

    if kind == "text":
        ensure_text_index(pc)
        return pc.Index(TEXT_HYBRID_INDEX_NAME)

    if kind == "image":
        ensure_image_index(pc)
        return pc.Index(IMAGE_INDEX_NAME)

    if kind == "audio":
        ensure_audio_index(pc)
        return pc.Index(AUDIO_INDEX_NAME)

    raise ValueError(f"Unknown index kind: {kind}")


# -----------------------------
# 1) fetch
# -----------------------------
def fetch_by_id(kind: IndexKind, ids: List[str], namespace: str = NAMESPACE) -> Dict[str, Any]:
    """
    특정 id들의 레코드를 fetch해서 반환.

    기대:
    - text 인덱스의 경우 vectors[id]에 sparse_values가 포함되어 내려오면
      "sparse가 실제로 들어갔다"를 100% 확인 가능.
    """
    index = _get_index(kind)
    return index.fetch(ids=ids, namespace=namespace)


def fetch_one(kind: IndexKind, id_: str, namespace: str = NAMESPACE) -> Dict[str, Any] | None:
    """
    id 하나 fetch해서 vectors[id]만 반환. 없으면 None.
    """
    res = fetch_by_id(kind, [id_], namespace=namespace)
    vectors = res.get("vectors") or {}
    return vectors.get(id_)


# -----------------------------
# 2) stats
# -----------------------------
def describe_stats(kind: IndexKind, namespace: str = NAMESPACE) -> Dict[str, Any]:
    """
    인덱스 통계를 반환.
    - namespace별 vector count 확인 가능
    """
    index = _get_index(kind)
    return index.describe_index_stats(namespace=namespace)


# -----------------------------
# 3) delete
# -----------------------------
def delete_ids(kind: IndexKind, ids: List[str], namespace: str = NAMESPACE) -> None:
    """
    특정 ids 삭제.
    """
    index = _get_index(kind)
    index.delete(ids=ids, namespace=namespace)


def delete_by_filter(kind: IndexKind, filter_: Dict[str, Any], namespace: str = NAMESPACE) -> None:
    """
    metadata filter로 삭제.
    예) {"artist": {"$eq": "화사 (HWASA)"}}  같은 형식 권장.

    주의:
    - Pinecone filter 문법은 버전에 따라 차이가 있을 수 있음.
    """
    index = _get_index(kind)
    index.delete(filter=filter_, namespace=namespace)


def delete_all(kind: IndexKind, namespace: str = NAMESPACE) -> None:
    """
    namespace 전체 삭제(dev 초기화용).
    """
    index = _get_index(kind)
    index.delete(delete_all=True, namespace=namespace)