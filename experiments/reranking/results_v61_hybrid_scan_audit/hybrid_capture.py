"""v61 — 서비스의 하이브리드 조회(`QdrantIndex.query`)를 **그대로 두고 감싸서** 입력·출력·시간을 기록하고, 구성(arm)에 따라
결과를 바꿔 끼운다. 서비스 코드는 수정하지 않는다.

  A: 현행 구현 그대로(`HYBRID_SCAN_LIMIT` 기본 5000). 기록만 한다.
  C: 현행 구현에 상한만 실제 point 수로(`run_candidates.py`가 모듈 상수를 바꾼다). 기록만 한다.
  B: 하이브리드 호출(sparse_vector가 있는 호출)만 `exact_hybrid.ExactHybridReference`로 계산한다. dense 전용 호출(이미지·오디오·
     sparse 없는 텍스트)은 원래 구현이 처리한다. payload는 원래 구현의 `_payloads`로 상위 top_k만 읽는다(서비스와 같은 모양).

기록(JSONL 한 줄 = 하이브리드 호출 하나): 질의 ID(라우터 `search`를 감싸 현재 질의를 안다 — 평가기는 질의를 순차로 돈다),
collection·top_k·필터·include_metadata, dense 질의 노름(=alpha, 단위 벡터에 alpha를 곱한 값이므로), sparse 항 수, 두 조회가
돌려준 건수와 dense 점수 최소·최대(A·C만 — 실제 조회 결과에서 읽는다), 결과 id·점수, 소요 ms. 질의 벡터는 npz로 따로 저장해
오프라인 재계산(`compare_text_calls.py`)에 쓴다. 벡터 파일은 커밋하지 않는다.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.vector_db import qdrant_backend as qb

from exact_hybrid import ExactHybridReference, allowed_point_ids


class HybridCapture:
    def __init__(self, arm: str, *, reference: Optional[ExactHybridReference] = None,
                 record_path: Optional[Path] = None, vectors_path: Optional[Path] = None) -> None:
        if arm not in {"A", "B", "C"}:
            raise ValueError(arm)
        if arm == "B" and reference is None:
            raise ValueError("B 구성은 기준 구현이 필요하다")
        self.arm = arm
        self.reference = reference
        self.record_path = record_path
        self.vectors_path = vectors_path
        self._lock = threading.Lock()
        self._local = threading.local()
        self.records: List[Dict[str, Any]] = []
        self.vectors: Dict[str, np.ndarray] = {}
        self.current_query_id: Optional[str] = None
        self.current_query_text: Optional[str] = None
        self.dense_only_calls = 0
        self._seq = 0
        self._orig_query = None
        self._orig_search = None
        self._orig_router_search = None
        self._installed = False
        self._filter_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------ 설치
    def install(self) -> None:
        if self._installed:
            return
        capture = self
        self._orig_query = qb.QdrantIndex.query
        self._orig_search = qb.QdrantIndex._search

        def wrapped_search(index_self, collection, *, query, using, limit, query_filter, with_payload):
            hits = capture._orig_search(index_self, collection, query=query, using=using, limit=limit,
                                        query_filter=query_filter, with_payload=with_payload)
            ctx = getattr(capture._local, "ctx", None)
            if ctx is not None:
                scores = [float(h.score) for h in hits]
                ctx[f"{using}_hits"] = len(hits)
                ctx[f"{using}_limit"] = int(limit)
                ctx[f"{using}_score_min"] = min(scores) if scores else None
                ctx[f"{using}_score_max"] = max(scores) if scores else None
                ctx[f"{using}_hit_ids"] = [int(h.id) for h in hits]
            return hits

        def wrapped_query(index_self, *, vector, top_k=10, include_metadata=True, namespace=qb.NAMESPACE,
                          filter=None, sparse_vector=None, **ignored):  # noqa: A002
            if sparse_vector is None:
                with capture._lock:
                    capture.dense_only_calls += 1
                return capture._orig_query(index_self, vector=vector, top_k=top_k, include_metadata=include_metadata,
                                           namespace=namespace, filter=filter, sparse_vector=None, **ignored)
            with capture._lock:
                capture._seq += 1
                seq = capture._seq
                query_id = capture.current_query_id
                query_text = capture.current_query_text
            ctx: Dict[str, Any] = {}
            capture._local.ctx = ctx
            started = time.perf_counter()
            try:
                if capture.arm == "B":
                    result = capture._exact_query(index_self, vector=vector, top_k=top_k, include_metadata=include_metadata,
                                                  namespace=namespace, filter=filter, sparse_vector=sparse_vector, ctx=ctx)
                else:
                    result = capture._orig_query(index_self, vector=vector, top_k=top_k, include_metadata=include_metadata,
                                                 namespace=namespace, filter=filter, sparse_vector=sparse_vector, **ignored)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000
                capture._local.ctx = None
            dense_arr = np.asarray(list(vector), dtype=np.float32)
            sparse_idx = np.asarray([int(i) for i in sparse_vector.get("indices", [])], dtype=np.int64)
            sparse_val = np.asarray([float(v) for v in sparse_vector.get("values", [])], dtype=np.float32)
            record = {
                "seq": seq,
                "arm": capture.arm,
                "query_id": query_id,
                "query": query_text,
                "collection": qb.collection_name(index_self._index_name, namespace),
                "top_k": int(top_k),
                "include_metadata": bool(include_metadata),
                "filter": json.dumps(filter, ensure_ascii=False, sort_keys=True) if filter else "",
                "dense_norm": float(np.linalg.norm(dense_arr.astype(np.float64))),
                "sparse_terms": int(len(sparse_idx)),
                "sparse_value_sum": float(sparse_val.astype(np.float64).sum()) if len(sparse_val) else 0.0,
                "elapsed_ms": round(elapsed_ms, 3),
                "result_ids": [m["id"] for m in result["matches"]],
                "result_scores": [round(float(m["score"]), 9) for m in result["matches"]],
                **{k: v for k, v in ctx.items() if not k.endswith("_hit_ids")},
            }
            # A·C에서 실제 두 조회가 돌려준 id 집합은 npz에만(크다) — 누락 분류에 쓴다
            with capture._lock:
                capture.records.append(record)
                capture.vectors[f"call{seq}_dense"] = dense_arr
                capture.vectors[f"call{seq}_sparse_idx"] = sparse_idx
                capture.vectors[f"call{seq}_sparse_val"] = sparse_val
                for using in (qb.DENSE_VECTOR, qb.SPARSE_VECTOR):
                    ids = ctx.get(f"{using}_hit_ids")
                    if ids is not None:
                        capture.vectors[f"call{seq}_{using}_hit_ids"] = np.asarray(ids, dtype=np.int64)
            return result

        qb.QdrantIndex.query = wrapped_query  # type: ignore[assignment]
        qb.QdrantIndex._search = wrapped_search  # type: ignore[assignment]

        # 현재 질의 ID — 평가기는 질의를 순차로 돌리므로 라우터 search 진입 때 한 번 적는다
        from src.retrieval import search_router as sr

        self._orig_router_search = sr.SearchRouter.search

        async def wrapped_router_search(router_self, analysis, *args, **kwargs):
            capture.current_query_text = getattr(analysis, "original_query", None)
            capture.current_query_id = capture.query_ids.get(capture.current_query_text) if capture.query_ids else None
            started = time.perf_counter()
            try:
                result = await capture._orig_router_search(router_self, analysis, *args, **kwargs)
            finally:
                query_id, query_text = capture.current_query_id, capture.current_query_text
                capture.current_query_text = None
                capture.current_query_id = None
            # 평가기가 받는 후보 목록(리랭킹 전 30개) 전체를 남긴다 — detail CSV에는 지문과 상위 10개만 있다
            with capture._lock:
                capture.candidates.append({
                    "query_id": query_id, "query": query_text,
                    "candidate_ids": [str(getattr(t, "id", t)) for t in result],
                    "candidate_scores": [round(float(getattr(t, "score", 0.0) or 0.0), 9) for t in result],
                    "search_ms": round((time.perf_counter() - started) * 1000, 3),
                })
            return result

        sr.SearchRouter.search = wrapped_router_search  # type: ignore[assignment]
        self.query_ids: Dict[str, str] = {}
        self.candidates: List[Dict[str, Any]] = []
        self._installed = True

    def set_query_ids(self, mapping: Dict[str, str]) -> None:
        """질의 원문 → query_id."""
        self.query_ids = dict(mapping)

    # ------------------------------------------------------------------ B 구성
    def _exact_query(self, index_self, *, vector, top_k, include_metadata, namespace, filter, sparse_vector, ctx):  # noqa: A002
        assert self.reference is not None
        collection = qb.collection_name(index_self._index_name, namespace)
        if collection != self.reference.collection:
            raise RuntimeError(f"B: 기준 구현의 컬렉션({self.reference.collection})과 호출({collection})이 다르다")
        allowed = None
        if filter:
            key = json.dumps(filter, ensure_ascii=False, sort_keys=True)
            allowed = self._filter_cache.get(key)
            if allowed is None:
                allowed = allowed_point_ids(index_self._client, collection, qb.translate_filter(filter))
                self._filter_cache[key] = allowed
            ctx["filter_allowed"] = len(allowed)
        ranked = self.reference.top(vector, sparse_vector, top_k, allowed_point_ids=allowed)
        ctx["exact_candidates"] = len(self.reference.point_ids) if allowed is None else len(allowed)
        payloads = (
            index_self._payloads(collection, [item["point_id"] for item in ranked]) if include_metadata else {}
        )
        matches = []
        for item in ranked:
            payload = dict(payloads.get(item["point_id"]) or {})
            song_id = str(payload.pop("song_id", item["id"]))
            matches.append({"id": song_id, "score": float(item["score"]), "metadata": payload if include_metadata else {}})
        return {"matches": matches}

    # ------------------------------------------------------------------ 저장
    def save(self) -> Dict[str, Any]:
        if self.record_path is not None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open("w", encoding="utf-8") as fh:
                for record in self.records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.vectors_path is not None:
            self.vectors_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.vectors_path, **self.vectors)
        if self.record_path is not None:
            with (self.record_path.parent / "candidates.jsonl").open("w", encoding="utf-8") as fh:
                for row in self.candidates:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        elapsed = [r["elapsed_ms"] for r in self.records]
        return {
            "hybrid_calls": len(self.records),
            "dense_only_calls": self.dense_only_calls,
            "hybrid_elapsed_ms_median": float(np.median(elapsed)) if elapsed else None,
            "hybrid_elapsed_ms_p95": float(np.percentile(elapsed, 95)) if elapsed else None,
            "dense_hits_total": int(sum(r.get("dense_hits", 0) or 0 for r in self.records)),
            "sparse_hits_total": int(sum(r.get("sparse_hits", 0) or 0 for r in self.records)),
            "queries_seen": len({r["query_id"] for r in self.records if r["query_id"]}),
        }
