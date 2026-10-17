"""승인 사례의 Qdrant 컬렉션. 곡 컬렉션·BM25 통계와 분리돼 있다.

포인트 하나 = 승인된 사례 하나. 벡터는 **운영자가 쓴 재사용 문장**의 KoE5 임베딩이고, 페이로드에는
사례를 가리키는 값만 둔다 — 원질의·사용자 사유·재사용 문장 본문은 Qdrant에 복사하지 않는다.

    feedback_id · song_id · source_revision · review_version · embedding_version ·
    expires_at_ts(초) · expires_at(ISO) · source

이 컬렉션은 **검색용 사본**이다. 기준은 Mongo이므로 여기서 찾은 사례는 반드시 Mongo의 최신 문서와
판을 대조한 뒤에 쓴다(`retrieval`). 그래서 포인트가 낡았더라도 검색 결과가 틀어지지 않는다.

삭제는 두 가지뿐이다.

- 만료: `expires_at_ts`가 지난 포인트. Mongo 원본이 TTL로 먼저 사라져도 이 값으로 지운다.
- 특정 판: `feedback_id` + 두 판 번호가 **모두 같은** 포인트. 그 사이 새 판이 같은 자리를 덮었으면
  조건이 맞지 않아 지워지지 않는다 — 늦게 도착한 삭제가 재승인 포인트를 지우지 못한다.

로컬 Qdrant는 한 컬렉션을 여러 스레드가 동시에 읽고 쓰는 것을 막아 주지 않는다. 검색 요청(읽기)과
색인 작업(쓰기)이 같은 클라이언트를 쓰므로 이 클래스의 잠금 하나로 줄 세운다.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from qdrant_client import models

from src.feedback.review import CaseVersion, as_utc, point_uuid
from src.vector_db.qdrant_backend import DENSE_VECTOR, INDEX_DISTANCE, collection_name, complete_top_points
from src.vector_db.settings import FEEDBACK_INDEX_NAME, NAMESPACE, TEXT_DENSE_DIM

logger = logging.getLogger(__name__)

# 컬렉션이 아직 없을 때 존재 여부를 다시 묻기까지의 간격. 서버 모드에서는 물을 때마다 왕복이다.
_MISSING_RECHECK_SECONDS = 30.0


class CaseIndexBusy(Exception):
    """잠금을 시간 안에 얻지 못했다(색인 작업이 쓰는 중). 검색은 이 사례 조회를 건너뛴다."""


@dataclass(frozen=True)
class CaseHit:
    feedback_id: str
    song_id: str
    source_revision: int
    review_version: int
    embedding_version: str
    expires_at_ts: float
    similarity: float

    def payload(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id, "song_id": self.song_id,
            "source_revision": self.source_revision, "review_version": self.review_version,
        }


def _version_filter(feedback_id: str, source_revision: int, review_version: int) -> models.Filter:
    return models.Filter(must=[
        models.FieldCondition(key="feedback_id", match=models.MatchValue(value=feedback_id)),
        models.FieldCondition(key="source_revision", match=models.MatchValue(value=source_revision)),
        models.FieldCondition(key="review_version", match=models.MatchValue(value=review_version)),
    ])


def _expired_filter(now_ts: float) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="expires_at_ts", range=models.Range(lte=now_ts))])


class FeedbackCaseIndex:
    def __init__(self, vector_client: Any, *, dim: int = TEXT_DENSE_DIM, namespace: str = NAMESPACE):
        # qdrant_backend.QdrantVectorClient. **만들 때는 아무것도 열거나 읽지 않는다** — 검색 반영과 색인
        # 작업이 모두 꺼진 서버에서도 이 객체는 만들어지므로, Qdrant는 실제로 쓰는 순간에만 만진다.
        self._vector_client = vector_client
        self._dim = dim
        self._namespace = namespace
        self.collection = collection_name(FEEDBACK_INDEX_NAME, namespace)
        self._lock = threading.RLock()
        self._exists = False
        self._missing_until = 0.0

    @property
    def _client(self) -> Any:
        return self._vector_client.client             # qdrant_client.QdrantClient

    # ------------------------------------------------------------------
    def ensure(self) -> None:
        """컬렉션을 만들거나, 이미 있으면 **차원·거리 설정이 맞는지** 확인한다. 다르면 쓰지 않는다."""
        with self._lock:
            self._vector_client.ensure_collection(FEEDBACK_INDEX_NAME, dim=self._dim, namespace=self._namespace)
            params = self._client.get_collection(self.collection).config.params.vectors
            dense = params.get(DENSE_VECTOR) if isinstance(params, dict) else None
            expected = INDEX_DISTANCE[FEEDBACK_INDEX_NAME]
            if dense is None or dense.size != self._dim or dense.distance != expected:
                raise RuntimeError(
                    f"{self.collection} 컬렉션의 벡터 설정이 다르다(기대: {self._dim}차원 · {expected.value})"
                )
            self._exists = True

    def exists(self) -> bool:
        if self._exists:
            return True
        now = time.monotonic()
        if now < self._missing_until:
            return False
        with self._lock:
            self._exists = bool(self._client.collection_exists(self.collection))
        if not self._exists:
            self._missing_until = now + _MISSING_RECHECK_SECONDS
        return self._exists

    # ------------------------------------------------------------------
    def upsert(self, entries: Sequence[Tuple[CaseVersion, Sequence[float], datetime]], *,
               embedding_version: str, source: str = "shown_result") -> None:
        """사례를 올린다. 포인트 ID가 사례마다 정해져 있어 같은 사례를 다시 올리면 덮어쓴다(중복 없음)."""
        points = []
        for case, vector, expires_at in entries:
            values = [float(value) for value in vector]
            if len(values) != self._dim or not all(math.isfinite(value) for value in values):
                raise ValueError("사례 벡터의 차원이 다르거나 유한하지 않은 값이 있다")
            expires = as_utc(expires_at)
            if expires is None:
                raise ValueError("만료 시각이 없는 사례는 색인하지 않는다")
            points.append(models.PointStruct(
                id=point_uuid(case.feedback_id),
                vector={DENSE_VECTOR: values},
                payload={
                    "feedback_id": case.feedback_id, "song_id": case.song_id,
                    "source_revision": case.source_revision, "review_version": case.review_version,
                    "embedding_version": embedding_version,
                    # 원본 검색 기록의 만료 시각을 그대로 싣는다. 색인·재색인으로 늘지 않는다.
                    "expires_at_ts": expires.timestamp(), "expires_at": expires.isoformat(),
                    "source": source,
                },
            ))
        if not points:
            return
        with self._lock:
            self._client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_version(self, feedback_id: str, source_revision: int, review_version: int) -> None:
        """이 사례의 **이 판** 포인트만 지운다. 다른 판이 올라와 있으면 건드리지 않는다."""
        with self._lock:
            self._client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=_version_filter(feedback_id, source_revision, review_version)),
                wait=True,
            )

    def count(self) -> int:
        with self._lock:
            return int(self._client.count(collection_name=self.collection, exact=True).count)

    def count_expired(self, now_ts: float) -> int:
        with self._lock:
            return int(self._client.count(
                collection_name=self.collection, count_filter=_expired_filter(now_ts), exact=True).count)

    def delete_expired(self, now_ts: float) -> int:
        """만료 포인트를 지우고 지운 수를 돌려준다. Mongo 문서가 남아 있는지는 보지 않는다."""
        with self._lock:
            expired = self.count_expired(now_ts)
            if expired:
                self._client.delete(
                    collection_name=self.collection,
                    points_selector=models.FilterSelector(filter=_expired_filter(now_ts)), wait=True,
                )
            return expired

    def scroll(self, offset: Any, limit: int) -> Tuple[List[Dict[str, Any]], Any]:
        """페이로드를 한 묶음 읽는다. (페이로드 목록, 다음 위치) — 다음 위치가 None이면 끝이다."""
        with self._lock:
            records, next_offset = self._client.scroll(
                collection_name=self.collection, limit=limit, offset=offset,
                with_payload=True, with_vectors=False,
            )
        return [dict(record.payload or {}) for record in records], next_offset

    def retrieve(self, feedback_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """피드백 ID → 지금 올라와 있는 포인트의 페이로드. 없는 사례는 결과에 없다."""
        ids = [point_uuid(feedback_id) for feedback_id in dict.fromkeys(feedback_ids)]
        if not ids:
            return {}
        with self._lock:
            records = self._client.retrieve(
                collection_name=self.collection, ids=ids, with_payload=True, with_vectors=False)
        return {str(record.payload["feedback_id"]): dict(record.payload)
                for record in records if record.payload and "feedback_id" in record.payload}

    # ------------------------------------------------------------------
    def search(self, vector: Sequence[float], *, top_k: int, now_ts: float, embedding_version: str,
               min_similarity: float, lock_timeout: Optional[float] = None) -> List[CaseHit]:
        """유사 사례. **미만료 · 같은 임베딩 판 · 문턱 이상**만 돌려준다(유사도 내림차순).

        잠금을 `lock_timeout` 안에 얻지 못하면 `CaseIndexBusy` — 검색은 사례 조회 없이 기존 경로로 간다.
        """
        if not self.exists():
            return []
        query_filter = models.Filter(must=[
            models.FieldCondition(key="expires_at_ts", range=models.Range(gt=now_ts)),
            models.FieldCondition(key="embedding_version", match=models.MatchValue(value=embedding_version)),
        ])
        acquired = self._lock.acquire(timeout=lock_timeout) if lock_timeout is not None else self._lock.acquire()
        if not acquired:
            raise CaseIndexBusy()
        search_params = None if self._vector_client.is_local else models.SearchParams(exact=True)
        query = [float(value) for value in vector]
        by_point: Dict[Any, CaseHit] = {}

        def identity(point: Any) -> Optional[str]:
            hit = _hit(point.payload or {}, float(point.score))
            # 문턱과 만료는 Qdrant 필터가 걸렀지만 한 번 더 본다 — 필터가 빠져도 넓은 사례가 새지 않게.
            if hit is None or hit.similarity < min_similarity or hit.expires_at_ts <= now_ts:
                return None
            by_point[point.id] = hit
            return hit.feedback_id

        try:
            # 동점 순서(v65): 유사도 내림차순 → feedback_id. limit에서 잘린 경계 동점까지 받은 뒤 자른다
            # (`complete_top_points`) — 받은 top_k만 다시 정렬하면 로컬·서버가 경계에서 다른 사례를 고를 수 있다.
            ranked = complete_top_points(
                lambda limit: self._client.query_points(
                    collection_name=self.collection, query=query, using=DENSE_VECTOR,
                    limit=limit, query_filter=query_filter, with_payload=True,
                    score_threshold=min_similarity, search_params=search_params,
                ).points,
                max(1, top_k),
                identity,
            )
        finally:
            self._lock.release()
        return [by_point[point.id] for point, _key in ranked]


def _hit(payload: Mapping[str, Any], score: float) -> Optional[CaseHit]:
    """페이로드가 온전한 포인트만 사례로 읽는다. 값이 빠졌거나 형이 다르면 버린다."""
    try:
        revision, version = payload["source_revision"], payload["review_version"]
        expires = float(payload["expires_at_ts"])
        if isinstance(revision, bool) or isinstance(version, bool) or not math.isfinite(score) or not math.isfinite(expires):
            return None
        return CaseHit(
            feedback_id=str(payload["feedback_id"]), song_id=str(payload["song_id"]),
            source_revision=int(revision), review_version=int(version),
            embedding_version=str(payload["embedding_version"]), expires_at_ts=expires, similarity=score,
        )
    except (KeyError, TypeError, ValueError):
        return None
