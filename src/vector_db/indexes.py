"""
src/vector_db/indexes.py

- 인덱스가 없으면 생성하는 ensure_index.
- 하이브리드 검색을 쓰더라도 인덱스 생성은 "dense dimension" 기준으로 생성.
  sparse는 upsert/query에서 sparse_values로 같이 넣는 방식.
"""
from __future__ import annotations

import time

from typing import Any
from pinecone import Pinecone, ServerlessSpec

from src.vector_db.settings import (
    PINECONE_CLOUD,
    PINECONE_REGION,
    TEXT_HYBRID_METRIC,
    IMAGE_METRIC,
    AUDIO_METRIC,
    TEXT_HYBRID_INDEX_NAME,
    IMAGE_INDEX_NAME,
    AUDIO_INDEX_NAME,
    TEXT_DENSE_DIM,
    IMAGE_DIM,
    AUDIO_DIM,
)


def _read_field(obj: Any, field: str) -> Any:
    """
    Pinecone SDK 응답이 dict 또는 attribute object 형태로 올 수 있어서 둘 다 처리한다.
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        return obj.get(field)

    if hasattr(obj, field):
        return getattr(obj, field)

    if hasattr(obj, "to_dict"):
        d = obj.to_dict()
        if isinstance(d, dict):
            return d.get(field)

    return None


def _wait_until_ready(
    pc: Pinecone,
    name: str,
    *,
    timeout_s: int = 120,
    interval_s: int = 3,
) -> None:
    """
    Pinecone index 생성 직후 ready 상태가 될 때까지 대기한다.
    """
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        desc = pc.describe_index(name)
        status = _read_field(desc, "status")

        ready = None
        if isinstance(status, dict):
            ready = status.get("ready")
        elif hasattr(status, "ready"):
            ready = getattr(status, "ready")

        if ready is True:
            print(f"[INDEX] ready name={name}")
            return

        time.sleep(interval_s)

    raise TimeoutError(f"Pinecone index did not become ready within {timeout_s}s: {name}")


def ensure_index(
    pc: Pinecone,
    *,
    name: str,
    dimension: int,
    metric: str,
) -> None:
    """
    index가 없으면 생성하고, 있으면 dimension/metric이 맞는지 검증한다.
    """
    required_metric = str(metric).lower()

    if pc.has_index(name):
        desc = pc.describe_index(name)

        current_metric = _read_field(desc, "metric")
        current_dim = _read_field(desc, "dimension")

        if current_metric is not None and str(current_metric).lower() != required_metric:
            raise RuntimeError(
                f"Pinecone index metric mismatch: index={name!r}, "
                f"current_metric={current_metric!r}, required_metric={metric!r}. "
                "Pinecone index metric cannot be changed after creation. "
                "Delete and recreate this index, or use another index name. "
                "For text hybrid dense+sparse search, metric must be 'dotproduct'."
            )

        if current_dim is not None and int(current_dim) != int(dimension):
            raise RuntimeError(
                f"Pinecone index dimension mismatch: index={name!r}, "
                f"current_dimension={current_dim!r}, required_dimension={dimension!r}. "
                "Delete and recreate this index, or use another index name."
            )

        print(
            f"[INDEX] exists name={name} "
            f"dimension={current_dim or dimension} metric={current_metric or metric}"
        )
        return

    pc.create_index(
        name=name,
        vector_type="dense",
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )

    print(f"[INDEX] created name={name} dimension={dimension} metric={metric}")
    _wait_until_ready(pc, name)


def ensure_text_index(pc: Pinecone) -> None:
    ensure_index(
        pc,
        name=TEXT_HYBRID_INDEX_NAME,
        dimension=TEXT_DENSE_DIM,
        metric=TEXT_HYBRID_METRIC,
    )


def ensure_image_index(pc: Pinecone) -> None:
    ensure_index(
        pc,
        name=IMAGE_INDEX_NAME,
        dimension=IMAGE_DIM,
        metric=IMAGE_METRIC,
    )


def ensure_audio_index(pc: Pinecone) -> None:
    ensure_index(
        pc,
        name=AUDIO_INDEX_NAME,
        dimension=AUDIO_DIM,
        metric=AUDIO_METRIC,
    )