"""v61 — 하이브리드 점수의 **독립 정확 합산 기준**.

서비스 구현(`src/vector_db/qdrant_backend.py`의 `QdrantIndex.query`)은 dense·sparse를 각각 `HYBRID_SCAN_LIMIT`개까지
조회해 코드에서 더한다. 이 모듈은 그 구현을 쓰지 않고, **컬렉션의 벡터 전부를 읽어** 질의 벡터와 직접 내적한 값을
기준으로 삼는다. 상한을 크게 올린 조회는 기준이 아니다 — 그것이 맞는지도 이 기준으로 확인한다.

- 읽기: `scroll` 페이지네이션(점수 조회 없음). 페이지마다 중복 id를 검사하고, 끝났을 때 `count(exact)`와 수가 같아야 한다.
  중간에 멈추면(`next_page_offset`가 None이 아닌데 빈 페이지) 예외.
- dense: float64 행렬곱 `M @ q`. 질의 벡터는 서비스가 넘기는 그대로(alpha가 이미 곱해진 값)라 여기서 다시 가중하지 않는다.
- sparse: 역색인(인덱스 → (행, 값)) 합. 질의 sparse도 서비스가 넘기는 그대로((1-alpha)가 곱해진 값).
- 합산 = dense + sparse. sparse에 걸리지 않은 곡은 sparse 0으로 dense 값만 남는다(서비스 구현과 같은 의미).
- 필터: 서비스가 쓰는 `translate_filter` 결과를 `scroll(scroll_filter=…)`로 적용해 허용 id 집합을 만든다.
  **필터 의미 자체는 Qdrant에 맡긴다** — 이 진단의 대상은 합산 누락이고 필터 변환은 기존 테스트가 덮는다.
- 허용 오차: Qdrant는 f32로 누적하고 여기는 f64다. dense는 단위 벡터 내적(|값| ≤ 1)이라 1e-5, sparse·합산은 BM25 값
  크기(수십)를 감안해 1e-4를 쓴다(`DENSE_TOL`, `TOTAL_TOL`). 비교 코드는 이 범위 안의 차이를 "같다"로 본다.
- 동점: 합산이 같은 곡들의 순서는 정의하지 않는다. 비교에서 동점 묶음 안의 순서 차이는 누락이 아니다(`compare_*` 참조).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

DENSE_TOL = 1e-5
TOTAL_TOL = 1e-4


class ScrollIntegrityError(RuntimeError):
    """페이지네이션으로 읽은 벡터가 컬렉션 수와 다르거나 중복됐다."""


@dataclass
class ScoreTable:
    """한 질의의 곡별 점수. 배열은 모두 같은 길이·같은 순서(`point_ids`)다."""

    point_ids: np.ndarray          # int64
    song_ids: List[str]
    dense: np.ndarray              # float64
    sparse: np.ndarray             # float64 (미일치 0)
    total: np.ndarray              # float64
    sparse_matched: np.ndarray     # bool — 질의 sparse 인덱스와 겹치는 항이 하나라도 있는가

    def ranked(self, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """합산 내림차순. 동점은 point id 오름차순으로만 **결정적으로** 정렬한다(의미 없는 순서지만 재현되게)."""
        order = np.lexsort((self.point_ids, -self.total))
        if top_k is not None:
            order = order[: max(0, top_k)]
        return [
            {
                "id": self.song_ids[i],
                "point_id": int(self.point_ids[i]),
                "score": float(self.total[i]),
                "dense": float(self.dense[i]),
                "sparse": float(self.sparse[i]),
                "sparse_matched": bool(self.sparse_matched[i]),
            }
            for i in order
        ]

    def ranks(self) -> Dict[str, Dict[str, int]]:
        """곡별 dense 순위 · sparse 순위(일치한 곡만, 1부터) · 합산 순위. 진단 분류용(구현 재현이 아님)."""
        dense_order = np.lexsort((self.point_ids, -self.dense))
        total_order = np.lexsort((self.point_ids, -self.total))
        matched = np.flatnonzero(self.sparse_matched)
        sparse_order = matched[np.lexsort((self.point_ids[matched], -self.sparse[matched]))]
        out: Dict[str, Dict[str, int]] = {sid: {} for sid in self.song_ids}
        for rank, i in enumerate(dense_order, start=1):
            out[self.song_ids[i]]["dense_rank"] = rank
        for rank, i in enumerate(total_order, start=1):
            out[self.song_ids[i]]["total_rank"] = rank
        for rank, i in enumerate(sparse_order, start=1):
            out[self.song_ids[i]]["sparse_rank"] = rank
        return out


@dataclass
class ExactHybridReference:
    collection: str
    point_ids: np.ndarray                      # int64, N
    song_ids: List[str]                        # N
    dense: np.ndarray                          # float64, N×D
    sparse_postings: Dict[int, Tuple[np.ndarray, np.ndarray]]  # 인덱스 → (행 번호 int64, 값 float64)
    sparse_nnz: int
    row_of_point: Dict[int, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ 구성
    @classmethod
    def from_client(
        cls,
        client: Any,
        collection: str,
        *,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
        page_size: int = 256,
        song_id_field: str = "song_id",
    ) -> "ExactHybridReference":
        """컬렉션 전체를 scroll로 읽는다. 중복·누락·중단을 검사한다."""
        expected = int(client.count(collection, exact=True).count)
        ids: List[int] = []
        seen: Set[int] = set()
        song_ids: List[str] = []
        dense_rows: List[np.ndarray] = []
        postings_idx: Dict[int, List[int]] = {}
        postings_val: Dict[int, List[float]] = {}
        nnz = 0
        offset = None
        pages = 0
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=page_size,
                offset=offset,
                with_payload=[song_id_field],
                with_vectors=[dense_name, sparse_name],
            )
            pages += 1
            if not points and offset is not None:
                raise ScrollIntegrityError(f"{collection}: 빈 페이지인데 다음 offset이 남아 있다(페이지 {pages})")
            for point in points:
                pid = int(point.id)
                if pid in seen:
                    raise ScrollIntegrityError(f"{collection}: 중복 point id {pid}(페이지 {pages})")
                seen.add(pid)
                vectors = point.vector if isinstance(point.vector, dict) else {}
                dense_vec = vectors.get(dense_name)
                if dense_vec is None:
                    raise ScrollIntegrityError(f"{collection}: point {pid}에 dense 벡터가 없다")
                row = len(ids)
                ids.append(pid)
                payload = point.payload or {}
                song_ids.append(str(payload.get(song_id_field, pid)))
                dense_rows.append(np.asarray(dense_vec, dtype=np.float64))
                sparse_vec = vectors.get(sparse_name)
                if sparse_vec is not None:
                    for index, value in zip(sparse_vec.indices, sparse_vec.values):
                        postings_idx.setdefault(int(index), []).append(row)
                        postings_val.setdefault(int(index), []).append(float(value))
                        nnz += 1
            if offset is None:
                break
        if len(ids) != expected:
            raise ScrollIntegrityError(f"{collection}: 읽은 수 {len(ids)} ≠ count {expected}")
        if not ids:
            raise ScrollIntegrityError(f"{collection}: 비어 있다")
        dims = {row.shape[0] for row in dense_rows}
        if len(dims) != 1:
            raise ScrollIntegrityError(f"{collection}: dense 차원이 섞여 있다 {sorted(dims)}")
        dense = np.vstack(dense_rows)
        postings = {
            index: (np.asarray(postings_idx[index], dtype=np.int64), np.asarray(postings_val[index], dtype=np.float64))
            for index in postings_idx
        }
        point_arr = np.asarray(ids, dtype=np.int64)
        return cls(
            collection=collection,
            point_ids=point_arr,
            song_ids=song_ids,
            dense=dense,
            sparse_postings=postings,
            sparse_nnz=nnz,
            row_of_point={pid: row for row, pid in enumerate(ids)},
        )

    # ------------------------------------------------------------------ 지문
    def fingerprint(self) -> Dict[str, Any]:
        """내용 지문 — point id 오름차순으로 (id, dense f32, sparse 인덱스·값)을 해시한다. 비교 중 색인이 바뀌지 않았음을 적는다."""
        order = np.argsort(self.point_ids)
        digest = hashlib.sha256()
        # sparse를 행 기준으로 다시 모은다(역색인은 인덱스 기준이라서)
        per_row: Dict[int, List[Tuple[int, float]]] = {}
        for index, (rows, values) in self.sparse_postings.items():
            for row, value in zip(rows.tolist(), values.tolist()):
                per_row.setdefault(row, []).append((index, value))
        for row in order.tolist():
            digest.update(int(self.point_ids[row]).to_bytes(8, "little", signed=False))
            digest.update(self.dense[row].astype(np.float32).tobytes())
            for index, value in sorted(per_row.get(row, [])):
                digest.update(int(index).to_bytes(4, "little", signed=False))
                digest.update(np.float32(value).tobytes())
        return {
            "collection": self.collection,
            "points": int(len(self.point_ids)),
            "dense_dim": int(self.dense.shape[1]),
            "sparse_nnz": int(self.sparse_nnz),
            "sparse_vocab": int(len(self.sparse_postings)),
            "content_sha256_16": digest.hexdigest()[:16],
        }

    # ------------------------------------------------------------------ 점수
    def score(
        self,
        dense_query: Sequence[float],
        sparse_query: Optional[Dict[str, Any]],
        *,
        allowed_point_ids: Optional[Iterable[int]] = None,
    ) -> ScoreTable:
        """곡별 dense·sparse·합산. `allowed_point_ids`를 주면 그 곡만 남긴다(필터)."""
        q = np.asarray(list(dense_query), dtype=np.float64)
        if q.shape[0] != self.dense.shape[1]:
            raise ValueError(f"dense 질의 차원 {q.shape[0]} ≠ 색인 {self.dense.shape[1]}")
        dense = self.dense @ q
        sparse = np.zeros(len(self.point_ids), dtype=np.float64)
        matched = np.zeros(len(self.point_ids), dtype=bool)
        if sparse_query:
            indices = [int(i) for i in sparse_query.get("indices", [])]
            values = [float(v) for v in sparse_query.get("values", [])]
            if len(indices) != len(values):
                raise ValueError("sparse 질의의 indices·values 길이가 다르다")
            for index, value in zip(indices, values):
                posting = self.sparse_postings.get(index)
                if posting is None:
                    continue
                rows, doc_values = posting
                np.add.at(sparse, rows, doc_values * value)
                matched[rows] = True
        keep = np.arange(len(self.point_ids))
        if allowed_point_ids is not None:
            allowed = set(int(p) for p in allowed_point_ids)
            keep = np.asarray([row for row, pid in enumerate(self.point_ids.tolist()) if pid in allowed], dtype=np.int64)
        return ScoreTable(
            point_ids=self.point_ids[keep],
            song_ids=[self.song_ids[i] for i in keep.tolist()],
            dense=dense[keep],
            sparse=sparse[keep],
            total=(dense + sparse)[keep],
            sparse_matched=matched[keep],
        )

    def top(
        self,
        dense_query: Sequence[float],
        sparse_query: Optional[Dict[str, Any]],
        top_k: int,
        *,
        allowed_point_ids: Optional[Iterable[int]] = None,
    ) -> List[Dict[str, Any]]:
        return self.score(dense_query, sparse_query, allowed_point_ids=allowed_point_ids).ranked(top_k)


def allowed_point_ids(client: Any, collection: str, qdrant_filter: Any, *, page_size: int = 1024) -> Set[int]:
    """필터에 맞는 point id 집합을 scroll로 모은다(점수 조회 없음). 중복이면 예외."""
    out: Set[int] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_filter,
            limit=page_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in points:
            pid = int(point.id)
            if pid in out:
                raise ScrollIntegrityError(f"{collection}: 필터 scroll에 중복 id {pid}")
            out.add(pid)
        if offset is None:
            return out


# ---------------------------------------------------------------------- 비교 도구

def same_score(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def tie_groups(ranked: List[Dict[str, Any]], tol: float = TOTAL_TOL) -> List[List[str]]:
    """합산 점수가 허용 오차 안에서 이어지는 묶음. 묶음 안의 순서는 비교에서 '차이'로 세지 않는다."""
    groups: List[List[str]] = []
    for item in ranked:
        if groups and same_score(item["score"], groups[-1][-1] if False else _last_score(groups, ranked), tol):
            groups[-1].append(item["id"])
        else:
            groups.append([item["id"]])
    return groups


def _last_score(groups: List[List[str]], ranked: List[Dict[str, Any]]) -> float:
    last_id = groups[-1][-1]
    for item in ranked:
        if item["id"] == last_id:
            return float(item["score"])
    raise KeyError(last_id)


def compare_rankings(
    actual: List[Dict[str, Any]],
    expected: List[Dict[str, Any]],
    *,
    tol: float = TOTAL_TOL,
) -> Dict[str, Any]:
    """구현 결과(actual)와 기준(expected)의 상위 목록을 대조한다.

    - `missing`: 기준 상위에는 있는데 구현 상위에 없는 곡
    - `extra`: 구현 상위에만 있는 곡
    - `score_mismatch`: 양쪽에 다 있지만 합산 점수가 허용 오차를 넘는 곡(점수 일부 누락의 직접 증거)
    - `order_differs_beyond_ties`: 집합·점수가 같아도 순서가 다른데, 그 차이가 동점 묶음 안의 순서로 설명되지 않을 때
    """
    actual_scores = {item["id"]: float(item["score"]) for item in actual}
    expected_scores = {item["id"]: float(item["score"]) for item in expected}
    missing = [sid for sid in expected_scores if sid not in actual_scores]
    extra = [sid for sid in actual_scores if sid not in expected_scores]
    mismatch = [
        {"id": sid, "actual": actual_scores[sid], "expected": expected_scores[sid],
         "delta": actual_scores[sid] - expected_scores[sid]}
        for sid in expected_scores
        if sid in actual_scores and not same_score(actual_scores[sid], expected_scores[sid], tol)
    ]
    order_differs = False
    if not missing and not extra and not mismatch:
        actual_ids = [item["id"] for item in actual]
        expected_ids = [item["id"] for item in expected]
        if actual_ids != expected_ids:
            # 동점 묶음 단위로 같은 집합이면 동점 순서 차이일 뿐이다
            groups = tie_groups(expected, tol)
            cursor = 0
            for group in groups:
                window = actual_ids[cursor: cursor + len(group)]
                if sorted(window) != sorted(group):
                    order_differs = True
                    break
                cursor += len(group)
    return {
        "identical_ids": [i["id"] for i in actual] == [i["id"] for i in expected],
        "same_set": not missing and not extra,
        "missing": missing,
        "extra": extra,
        "score_mismatch": mismatch,
        "order_differs_beyond_ties": order_differs,
        "max_abs_score_delta_common": max(
            (abs(actual_scores[s] - expected_scores[s]) for s in expected_scores if s in actual_scores),
            default=0.0,
        ),
    }
