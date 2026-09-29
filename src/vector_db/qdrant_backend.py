"""
Qdrant 벡터 클라이언트.

검색 코드가 벡터 DB를 부르는 곳은 세 군데뿐이다.

    search_service.py   텍스트 하이브리드 (dense + BM25 sparse)
    search_router.py    이미지 / 오디오

세 곳 모두 `client.Index(name).query(...)` 결과에서 `matches`의
`id / score / metadata`만 읽는다. 이 호출 모양은 2026-09 Pinecone에서 옮겨 올 때
검색·재질문·리랭킹 코드를 바꾸지 않으려고 그대로 둔 것이다.

점수 계산
---------
- 텍스트: 하이브리드 점수는 alpha로 스케일한 dense·sparse 내적의 합이다. Qdrant에는
  그 합산이 없으므로 dense와 sparse를 따로 조회해 **코드에서 더한다.** 두 조회 모두
  컬렉션 전체를 훑어 합산하므로(HYBRID_SCAN_LIMIT) 잘린 상위 목록만 더해서 생기는
  차이가 없다.
- 이미지·오디오: 저장된 벡터가 전부 L2 정규화되어 있어 cosine == dot이다.
- 정확 검색(`exact=True`)을 쓴다. 근사 검색은 측정할 때마다 순위가 흔들릴 수 있다.

로컬 모드(파일 경로)가 기본이다. QDRANT_URL을 주면 서버에 붙는다.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient, models

from src.vector_db.settings import (
    AUDIO_INDEX_NAME,
    IMAGE_INDEX_NAME,
    NAMESPACE,
    TEXT_HYBRID_INDEX_NAME,
)

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# 인덱스 이름 → 거리 척도.
# (모든 벡터가 L2 정규화되어 있어 cosine과 dot은 같은 순서를 준다)
INDEX_DISTANCE: Dict[str, models.Distance] = {
    TEXT_HYBRID_INDEX_NAME: models.Distance.DOT,
    IMAGE_INDEX_NAME: models.Distance.COSINE,
    AUDIO_INDEX_NAME: models.Distance.COSINE,
}

# 하이브리드 합산을 위해 한 번에 훑을 상한. 코퍼스(905곡)보다 크면 전수 합산이 된다.
HYBRID_SCAN_LIMIT = int(os.getenv("QDRANT_HYBRID_SCAN_LIMIT", "5000"))


def collection_name(index_name: str, namespace: str = NAMESPACE) -> str:
    """(인덱스 이름, 네임스페이스)로 Qdrant 컬렉션 이름 `<인덱스>__<네임스페이스>`를 만든다."""
    return f"{index_name}__{namespace or 'default'}"


def point_id(song_id: str) -> int | str:
    """Qdrant point id는 정수 또는 UUID만 허용한다. 곡 id는 숫자 문자열이다."""
    text = str(song_id).strip()
    if text.isdigit():
        return int(text)
    # 숫자가 아닌 id가 들어오면 UUID로 접는다. payload의 song_id가 원본을 보존한다.
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


_RANGE_OPS = {"$gt": "gt", "$gte": "gte", "$lt": "lt", "$lte": "lte"}


def _field_conditions(key: str, value: Any) -> List[models.Condition]:
    """한 필드의 조건을 Qdrant 조건으로 옮긴다."""
    if not isinstance(value, dict):
        return [models.FieldCondition(key=key, match=models.MatchValue(value=value))]

    conditions: List[models.Condition] = []
    range_kwargs: Dict[str, Any] = {}
    for op, operand in value.items():
        if op == "$eq":
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=operand))
            )
        elif op == "$ne":
            conditions.append(
                models.Filter(
                    must_not=[
                        models.FieldCondition(key=key, match=models.MatchValue(value=operand))
                    ]
                )
            )
        elif op == "$in":
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=list(operand)))
            )
        elif op == "$nin":
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchExcept(**{"except": list(operand)}))
            )
        elif op in _RANGE_OPS:
            range_kwargs[_RANGE_OPS[op]] = operand
        else:
            raise ValueError(f"지원하지 않는 필터 조건입니다: {key}={{{op!r}: ...}}")

    if range_kwargs:
        conditions.append(models.FieldCondition(key=key, range=models.Range(**range_kwargs)))
    return conditions


def translate_filter(metadata_filter: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
    """검색 코드의 메타데이터 필터(`$eq`·`$and` 같은 표기)를 Qdrant 필터로 옮긴다.

    실제로 쓰는 것은 제목 구조 경로의 `$eq`·`$and`·`$gte`(문장형 제목)·`$or`
    (의성어 제목)와 연주 단서 경로의 `$eq`다. 범위·OR를 빠뜨리면 그 경로가
    예외로 죽고 라우터가 빈 결과로 넘겨 **제목 단서 후보가 통째로 사라진다**.

    모르는 연산자는 **조용히 무시하지 않고 예외를 낸다** — 필터가 빠진 채로
    검색되면 결과가 달라진 것을 눈치채기 어렵다.
    """
    conditions = _filter_conditions(metadata_filter)
    return models.Filter(must=conditions) if conditions else None


def _filter_conditions(metadata_filter: Optional[Dict[str, Any]]) -> List[models.Condition]:
    if not metadata_filter:
        return []

    conditions: List[models.Condition] = []
    for key, value in metadata_filter.items():
        if key == "$and":
            for sub in value or []:
                conditions.extend(_filter_conditions(sub))
        elif key == "$or":
            branches: List[models.Condition] = []
            for sub in value or []:
                sub_conditions = _filter_conditions(sub)
                # 한 갈래 안의 조건들은 AND로 묶어야 OR 의미가 유지된다
                branches.append(
                    sub_conditions[0]
                    if len(sub_conditions) == 1
                    else models.Filter(must=sub_conditions)
                )
            if branches:
                conditions.append(models.Filter(should=branches))
        elif key == "$not":
            conditions.append(models.Filter(must_not=_filter_conditions(value)))
        elif key.startswith("$"):
            raise ValueError(f"지원하지 않는 필터 연산자입니다: {key}")
        else:
            conditions.extend(_field_conditions(key, value))
    return conditions


class QdrantIndex:
    """검색 코드가 쓰는 `query()`를 제공한다."""

    def __init__(self, client: QdrantClient, index_name: str, exact: bool = True):
        self._client = client
        self._index_name = index_name
        # 로컬 모드는 항상 전수 비교라 search_params가 무시된다(경고만 남는다).
        # 서버 모드에서만 정확 검색을 명시한다 — 근사 검색은 측정 순위를 흔든다.
        self._search_params = models.SearchParams(exact=True) if exact else None

    # 검색 코드의 호출 인자를 그대로 받는다 (include_values 등 미사용 인자는 무시)
    def query(
        self,
        *,
        vector: Sequence[float],
        top_k: int = 10,
        include_metadata: bool = True,
        namespace: str = NAMESPACE,
        filter: Optional[Dict[str, Any]] = None,  # noqa: A002 - 호출부 인자명을 따른다
        sparse_vector: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        collection = collection_name(self._index_name, namespace)
        query_filter = translate_filter(filter)

        dense_hits = self._search(
            collection,
            query=list(vector),
            using=DENSE_VECTOR,
            limit=top_k if sparse_vector is None else HYBRID_SCAN_LIMIT,
            query_filter=query_filter,
            with_payload=include_metadata,
        )

        if sparse_vector is None:
            return {"matches": [self._match(hit, include_metadata) for hit in dense_hits[:top_k]]}

        sparse_hits = self._search(
            collection,
            query=models.SparseVector(
                indices=[int(i) for i in sparse_vector.get("indices", [])],
                values=[float(v) for v in sparse_vector.get("values", [])],
            ),
            using=SPARSE_VECTOR,
            limit=HYBRID_SCAN_LIMIT,
            query_filter=query_filter,
            with_payload=include_metadata,
        )

        # dense + sparse 점수 합산 = 하이브리드 점수
        merged: Dict[Any, Dict[str, Any]] = {}
        for hit in dense_hits:
            merged[hit.id] = {"hit": hit, "score": float(hit.score)}
        for hit in sparse_hits:
            entry = merged.get(hit.id)
            if entry is None:
                merged[hit.id] = {"hit": hit, "score": float(hit.score)}
            else:
                entry["score"] += float(hit.score)

        ordered = sorted(merged.values(), key=lambda e: e["score"], reverse=True)[:top_k]
        return {
            "matches": [
                self._match(entry["hit"], include_metadata, score=entry["score"])
                for entry in ordered
            ]
        }

    def _search(
        self,
        collection: str,
        *,
        query: Any,
        using: str,
        limit: int,
        query_filter: Optional[models.Filter],
        with_payload: bool,
    ) -> List[models.ScoredPoint]:
        try:
            response = self._client.query_points(
                collection_name=collection,
                query=query,
                using=using,
                limit=max(1, limit),
                query_filter=query_filter,
                with_payload=with_payload,
                search_params=self._search_params,
            )
        except Exception as exc:
            logger.error("[Qdrant] %s/%s 조회 실패: %s", collection, using, exc)
            raise
        return list(response.points)

    @staticmethod
    def _match(
        hit: models.ScoredPoint,
        include_metadata: bool,
        score: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload = dict(hit.payload or {})
        song_id = str(payload.pop("song_id", hit.id))
        return {
            "id": song_id,
            "score": float(hit.score if score is None else score),
            "metadata": payload if include_metadata else {},
        }


class QdrantVectorClient:
    """검색 코드가 쓰는 벡터 DB 클라이언트. `Index(name)`으로 컬렉션을 연다."""

    def __init__(self, path: Optional[str] = None, url: Optional[str] = None, api_key: Optional[str] = None):
        url = url or os.getenv("QDRANT_URL", "").strip()
        if url:
            self.client = QdrantClient(url=url, api_key=api_key or os.getenv("QDRANT_API_KEY") or None)
            self.is_local = False
            logger.info("[Qdrant] 서버 모드: %s", url)
        elif (path or os.getenv("QDRANT_PATH", "")) == ":memory:":
            # 테스트용 — 디스크에 남기지 않는다
            self.client = QdrantClient(location=":memory:")
            self.is_local = True
            logger.info("[Qdrant] 메모리 모드")
        else:
            storage = Path(path or os.getenv("QDRANT_PATH", "artifacts/qdrant"))
            storage.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage))
            self.is_local = True
            logger.info("[Qdrant] 로컬 모드: %s", storage)

    def Index(self, index_name: str) -> QdrantIndex:  # noqa: N802 - 호출부가 쓰는 이름을 따른다
        return QdrantIndex(self.client, index_name, exact=not self.is_local)

    # --- 적재용 ---------------------------------------------------------
    def ensure_collection(
        self,
        index_name: str,
        *,
        dim: int,
        namespace: str = NAMESPACE,
        with_sparse: bool = False,
        recreate: bool = False,
    ) -> str:
        name = collection_name(index_name, namespace)
        if recreate and self.client.collection_exists(name):
            self.client.delete_collection(name)
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=dim,
                        distance=INDEX_DISTANCE.get(index_name, models.Distance.COSINE),
                    )
                },
                sparse_vectors_config=(
                    {SPARSE_VECTOR: models.SparseVectorParams()} if with_sparse else None
                ),
            )
        return name

    def close(self) -> None:
        self.client.close()


def get_qdrant_client() -> QdrantVectorClient:
    return QdrantVectorClient()
