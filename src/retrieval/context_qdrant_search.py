"""Query the published Namuwiki fact and song-profile collections.

The retrieval boundary returns individual facts and sparse song profiles;
their raw scores are not comparable. Song aggregation is available as a
separate ranking step; Dense/Sparse fusion belongs to a later stage.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qdrant_client import models

from src.embedding.context_dense import CONTEXT_DENSE_MANIFEST_VERSION, safe_model_tag
from src.embedding.context_sparse import (
    DEFAULT_CONTEXT_SPARSE_OUTPUT,
    ContextBM25QueryEncoder,
    load_context_bm25_query_encoder,
)
from src.embedding.models.text_koe5 import DEFAULT_KOE5_MODEL, KoE5Embedder
from src.retrieval.context_ranking import (
    ContextDenseSongHit,
    aggregate_dense_facts_by_song,
)
from src.vector_db.context_qdrant import (
    CONTEXT_QDRANT_MANIFEST_VERSION,
    CONTEXT_QDRANT_PAYLOAD_VERSION,
)
from src.vector_db.qdrant_backend import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    QdrantVectorClient,
    collection_name,
)
from src.vector_db.settings import (
    CONTEXT_DENSE_DIM,
    CONTEXT_DENSE_INDEX_NAME,
    CONTEXT_SPARSE_INDEX_NAME,
    NAMESPACE,
)


@dataclass(frozen=True)
class ContextFactHit:
    song_id: str
    record_id: str
    score: float
    fact_text: str
    source_url: str
    title: str
    artists: tuple[str, ...]
    category: str
    section: str
    quality: str
    source_fact_indices: tuple[int, ...]


@dataclass(frozen=True)
class ContextProfileHit:
    song_id: str
    profile_id: str
    score: float
    title: str
    artists: tuple[str, ...]


@dataclass(frozen=True)
class ContextSearchHits:
    dense_facts: tuple[ContextFactHit, ...]
    sparse_profiles: tuple[ContextProfileHit, ...]


@dataclass(frozen=True)
class _Snapshot:
    dense_collection: str
    sparse_collection: str
    build_id: str
    dimension: int
    dense_manifest_sha256: str
    sparse_manifest_sha256: str


def _manifest(path: Path, schema: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"context search manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid context search manifest: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise RuntimeError(f"unsupported context search manifest: {path}")
    expected = value.get("manifest_content_sha256")
    content = dict(value)
    content.pop("manifest_content_sha256", None)
    actual = hashlib.sha256(json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if not expected or expected != actual:
        raise RuntimeError(f"context search manifest hash mismatch: {path}")
    return value


def _required(payload: dict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"context Qdrant hit has invalid {key}")
    return value.strip()


def _artists(payload: dict) -> tuple[str, ...]:
    value = payload.get("artists", [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError("context Qdrant hit has invalid artists")
    return tuple(value)


class ContextQdrantSearch:
    """Low-level context search using one consistent pair of Qdrant generations.

    A catalogue that has never published context returns empty results. If a
    publication manifest exists, missing, partial or stale aliases are errors:
    a wrong Qdrant path must not silently disable background-knowledge search.
    """

    def __init__(
        self,
        vector_client: QdrantVectorClient,
        *,
        namespace: str = NAMESPACE,
        dense_dir: Path | None = None,
        sparse_dir: Path = DEFAULT_CONTEXT_SPARSE_OUTPUT,
        state_manifest: Path | None = None,
        text_embedder: Any | None = None,
        hash_fn: Callable[[str], int] | None = None,
        expected_dense_dim: int = CONTEXT_DENSE_DIM,
    ) -> None:
        self.client = vector_client.client
        self.dense_alias = collection_name(CONTEXT_DENSE_INDEX_NAME, namespace)
        self.sparse_alias = collection_name(CONTEXT_SPARSE_INDEX_NAME, namespace)
        self.dense_dir = Path(dense_dir or (
            Path("artifacts/embeddings/context_dense") / safe_model_tag(DEFAULT_KOE5_MODEL)
        ))
        self.sparse_dir = Path(sparse_dir)
        self.state_manifest = Path(state_manifest or (
            Path("artifacts/vector_db/context_qdrant") / namespace / "manifest.json"
        ))
        self.expected_dense_dim = expected_dense_dim
        self._embedder = text_embedder
        self._injected_embedder = text_embedder is not None
        self._embedder_build_id: str | None = None
        self._hash_fn = hash_fn
        self._sparse_encoder: ContextBM25QueryEncoder | None = None
        self._sparse_manifest_sha256: str | None = None
        self._lock = threading.Lock()

    def _snapshot(self) -> _Snapshot | None:
        aliases = {
            item.alias_name: item.collection_name
            for item in self.client.get_aliases().aliases
        }
        dense = aliases.get(self.dense_alias)
        sparse = aliases.get(self.sparse_alias)
        if dense is None and sparse is None:
            if self.state_manifest.exists() or self.state_manifest.is_symlink():
                raise RuntimeError("published context Qdrant aliases are missing")
            return None
        if dense is None or sparse is None:
            raise RuntimeError("context Qdrant aliases are only partially published")

        state = _manifest(self.state_manifest, CONTEXT_QDRANT_MANIFEST_VERSION)
        info = state.get("qdrant", {})
        hashes = state.get("source_hashes", {})
        if (
            state.get("status") != "ok"
            or info.get("dense_alias") != self.dense_alias
            or info.get("sparse_alias") != self.sparse_alias
            or info.get("dense_collection") != dense
            or info.get("sparse_collection") != sparse
            or info.get("dense_vector_name") != DENSE_VECTOR
            or info.get("sparse_vector_name") != SPARSE_VECTOR
            or info.get("payload_schema_version") != CONTEXT_QDRANT_PAYLOAD_VERSION
            or info.get("dense_dimension") != self.expected_dense_dim
            or not state.get("context_build_id")
            or not hashes.get("dense_manifest_content_sha256")
            or not hashes.get("sparse_manifest_content_sha256")
        ):
            raise RuntimeError("context Qdrant publication differs from its state manifest")
        return _Snapshot(
            dense_collection=dense,
            sparse_collection=sparse,
            build_id=state["context_build_id"],
            dimension=self.expected_dense_dim,
            dense_manifest_sha256=hashes["dense_manifest_content_sha256"],
            sparse_manifest_sha256=hashes["sparse_manifest_content_sha256"],
        )

    def _text_embedder(self, snapshot: _Snapshot) -> Any:
        with self._lock:
            if self._embedder_build_id != snapshot.build_id:
                dense = _manifest(
                    self.dense_dir / "manifest.json", CONTEXT_DENSE_MANIFEST_VERSION
                )
                config = dense.get("embedding", {})
                if (
                    dense["manifest_content_sha256"] != snapshot.dense_manifest_sha256
                    or config.get("dimension") != snapshot.dimension
                    or not config.get("model_name")
                    or config.get("e5_prefix") != "passage: "
                ):
                    raise RuntimeError("context dense model differs from the published corpus")
                if self._injected_embedder:
                    if isinstance(self._embedder, KoE5Embedder) and (
                        self._embedder.model_name != config["model_name"]
                        or self._embedder.revision != config.get("model_revision")
                    ):
                        raise RuntimeError("context query embedder differs from the published model")
                else:
                    self._embedder = KoE5Embedder(
                        model_name=config["model_name"], revision=config.get("model_revision")
                    )
                self._embedder_build_id = snapshot.build_id
            return self._embedder

    def _query_encoder(self, snapshot: _Snapshot) -> ContextBM25QueryEncoder:
        with self._lock:
            if self._sparse_manifest_sha256 != snapshot.sparse_manifest_sha256:
                encoder = load_context_bm25_query_encoder(
                    self.sparse_dir, hash_fn=self._hash_fn
                )
                if encoder.manifest_content_sha256 != snapshot.sparse_manifest_sha256:
                    raise RuntimeError("context BM25 parameters differ from the published corpus")
                self._sparse_encoder = encoder
                self._sparse_manifest_sha256 = snapshot.sparse_manifest_sha256
            assert self._sparse_encoder is not None
            return self._sparse_encoder

    @staticmethod
    def _payload(hit: models.ScoredPoint, snapshot: _Snapshot, kind: str) -> dict:
        payload = hit.payload
        if (
            not isinstance(payload, dict)
            or payload.get("context_build_id") != snapshot.build_id
            or payload.get("payload_schema_version") != CONTEXT_QDRANT_PAYLOAD_VERSION
            or payload.get("point_kind") != kind
            or not math.isfinite(float(hit.score))
        ):
            raise RuntimeError("context Qdrant hit is incompatible with the active generation")
        return payload

    def _dense(self, snapshot: _Snapshot, query: str, limit: int) -> tuple[ContextFactHit, ...]:
        vector = np.asarray(self._text_embedder(snapshot).embed_passages(
            [f"query: {query}"], add_e5_prefix=False, normalize=True
        ), dtype=np.float32)
        if (
            vector.shape != (1, snapshot.dimension)
            or not np.isfinite(vector).all()
            or not math.isclose(float(np.linalg.norm(vector[0])), 1.0, abs_tol=2e-3)
        ):
            raise RuntimeError("context query embedding has invalid dimension or norm")
        hits = self.client.query_points(
            collection_name=snapshot.dense_collection,
            query=vector[0].tolist(),
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points
        results = []
        for hit in hits:
            payload = self._payload(hit, snapshot, "context_fact")
            indices = payload.get("source_fact_indices", [])
            if not isinstance(indices, list) or not all(type(i) is int for i in indices):
                raise RuntimeError("context fact has invalid source indices")
            results.append(ContextFactHit(
                song_id=_required(payload, "song_id"),
                record_id=_required(payload, "record_id"),
                score=float(hit.score),
                fact_text=_required(payload, "fact_text"),
                source_url=str(payload.get("source_url") or ""),
                title=str(payload.get("title") or ""),
                artists=_artists(payload),
                category=str(payload.get("category") or ""),
                section=str(payload.get("section") or ""),
                quality=str(payload.get("quality") or ""),
                source_fact_indices=tuple(indices),
            ))
        return tuple(results)

    def _sparse(self, snapshot: _Snapshot, query: str, limit: int) -> tuple[ContextProfileHit, ...]:
        sparse = self._query_encoder(snapshot).encode(query)
        if not sparse["indices"]:
            return ()
        hits = self.client.query_points(
            collection_name=snapshot.sparse_collection,
            query=models.SparseVector(**sparse),
            using=SPARSE_VECTOR,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points
        results = []
        for hit in hits:
            payload = self._payload(hit, snapshot, "context_song_profile")
            results.append(ContextProfileHit(
                song_id=_required(payload, "song_id"),
                profile_id=_required(payload, "profile_id"),
                score=float(hit.score),
                title=str(payload.get("title") or ""),
                artists=_artists(payload),
            ))
        return tuple(results)

    def search_dense_facts(self, query: str, *, limit: int = 50) -> tuple[ContextFactHit, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        snapshot = self._snapshot()
        return self._dense(snapshot, query.strip(), limit) if snapshot and query.strip() else ()

    def search_dense_songs(
        self, query: str, *, fact_k: int = 50, song_k: int | None = None
    ) -> tuple[ContextDenseSongHit, ...]:
        """Retrieve ``fact_k`` facts, then keep the best fact for each song.

        ``song_k`` applies after aggregation. Choose a sufficiently large
        ``fact_k`` when multiple facts from one song could crowd the results.
        """
        if song_k is not None and song_k < 1:
            raise ValueError("song_k must be positive")
        return aggregate_dense_facts_by_song(
            self.search_dense_facts(query, limit=fact_k), limit=song_k
        )

    def search_sparse_profiles(self, query: str, *, limit: int = 50) -> tuple[ContextProfileHit, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        snapshot = self._snapshot()
        return self._sparse(snapshot, query.strip(), limit) if snapshot and query.strip() else ()

    def search(self, query: str, *, dense_k: int = 50, sparse_k: int = 50) -> ContextSearchHits:
        if dense_k < 1 or sparse_k < 1:
            raise ValueError("dense_k and sparse_k must be positive")
        snapshot = self._snapshot()
        query = query.strip()
        if snapshot is None or not query:
            return ContextSearchHits((), ())
        return ContextSearchHits(
            dense_facts=self._dense(snapshot, query, dense_k),
            sparse_profiles=self._sparse(snapshot, query, sparse_k),
        )
