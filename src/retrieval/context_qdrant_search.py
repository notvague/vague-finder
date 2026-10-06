"""Query the published Namuwiki fact and song-profile collections.

The retrieval boundary returns individual facts and sparse song profiles;
their raw scores are not comparable. Dense facts are aggregated by song, then
the two Context rankings can be fused into one candidate path with RRF.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import unicodedata
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
    CONTEXT_RRF_K,
    ContextDenseSongHit,
    ContextFusedSongHit,
    aggregate_dense_facts_by_song,
    fuse_context_song_candidates,
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
class ContextSongCandidates:
    """Separate song rankings to use in Context-only rank fusion.

    ``dense_facts`` retains the same query's retrieved sentences for the
    final evidence check. Even ``dense_songs[].best_fact`` is only a candidate
    until that check. A sparse profile hit cannot substantiate a sentence.
    """

    dense_songs: tuple[ContextDenseSongHit, ...]
    sparse_songs: tuple[ContextProfileHit, ...]
    # Keep the same snapshot's fact hits for later evidence checking. Ranking
    # still uses only one maximum-score fact per song.
    dense_facts: tuple[ContextFactHit, ...] = ()


@dataclass(frozen=True)
class _Snapshot:
    dense_collection: str
    sparse_collection: str
    build_id: str
    dimension: int
    dense_manifest_sha256: str
    sparse_manifest_sha256: str


def _media_key(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold()
                   if c.isalnum())


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
        self._media_index_key: tuple[str, str] | None = None
        self._media_index_rows: tuple[tuple[Any, str], ...] = ()
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

    def _media_fact_ids(
        self, snapshot: _Snapshot, targets: tuple[str, ...],
    ) -> list[Any]:
        """Cache literal fact text once per generation, never the query ranks.

        Korean work names often carry particles ("...에서") and varying spaces.
        MatchText tokenization varies across Qdrant versions/index settings.
        Match normalized names here, then use the matching point IDs as a
        Dense filter. The fact corpus is read without vectors or song metadata.
        """
        key = (snapshot.dense_collection, snapshot.build_id)
        with self._lock:
            if self._media_index_key != key:
                rows, seen, offset = [], set(), None
                while True:
                    points, next_offset = self.client.scroll(
                        collection_name=snapshot.dense_collection,
                        limit=512, offset=offset, with_vectors=False,
                        with_payload=["fact_text", "context_build_id",
                                      "payload_schema_version", "point_kind"],
                    )
                    for point in points:
                        payload = point.payload
                        if (not isinstance(payload, dict)
                                or payload.get("context_build_id") != snapshot.build_id
                                or payload.get("payload_schema_version") != CONTEXT_QDRANT_PAYLOAD_VERSION
                                or payload.get("point_kind") != "context_fact"
                                or point.id in seen):
                            raise RuntimeError("context media fact index has incompatible or duplicate points")
                        seen.add(point.id)
                        rows.append((point.id, _media_key(_required(payload, "fact_text"))))
                    if next_offset is None:
                        break
                    if next_offset == offset:
                        raise RuntimeError("context media fact index scroll made no progress")
                    offset = next_offset
                self._media_index_rows = tuple(rows)
                self._media_index_key = key
            names = tuple(_media_key(name) for name in targets)
            return [point_id for point_id, text in self._media_index_rows
                    if any(name in text for name in names)]

    def _dense(
        self, snapshot: _Snapshot, query: str, limit: int,
        media_targets: tuple[str, ...] = (),
    ) -> tuple[ContextFactHit, ...]:
        ids = self._media_fact_ids(snapshot, media_targets) if media_targets else None
        if ids == []:
            return ()
        vector = np.asarray(self._text_embedder(snapshot).embed_passages(
            [f"query: {query}"], add_e5_prefix=False, normalize=True
        ), dtype=np.float32)
        if (
            vector.shape != (1, snapshot.dimension)
            or not np.isfinite(vector).all()
            or not math.isclose(float(np.linalg.norm(vector[0])), 1.0, abs_tol=2e-3)
        ):
            raise RuntimeError("context query embedding has invalid dimension or norm")
        # This optional focused lookup searches facts containing the supplied
        # work name. It is separate from the router's unrestricted recollection
        # lookup; a text filter contributes no score or extra outer RRF vote.
        query_filter = models.Filter(must=[models.HasIdCondition(has_id=ids)]) if ids is not None else None
        hits = self.client.query_points(
            collection_name=snapshot.dense_collection,
            query=vector[0].tolist(),
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            **({"query_filter": query_filter} if query_filter is not None else {}),
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

    def search(
        self, query: str, *, dense_k: int = 50, sparse_k: int = 50,
        media_targets: tuple[str, ...] = (),
    ) -> ContextSearchHits:
        if dense_k < 1 or sparse_k < 1:
            raise ValueError("dense_k and sparse_k must be positive")
        if (not isinstance(media_targets, tuple) or len(media_targets) > 4
                or any(not isinstance(name, str) or len(_media_key(name)) < 2
                       or len(name) > 80 for name in media_targets)):
            raise ValueError("media_targets must contain at most four nonempty work names")
        snapshot = self._snapshot()
        query = query.strip()
        if snapshot is None or not query:
            return ContextSearchHits((), ())
        return ContextSearchHits(
            dense_facts=self._dense(snapshot, query, dense_k, media_targets),
            sparse_profiles=self._sparse(snapshot, query, sparse_k),
        )

    def search_song_candidates(
        self,
        query: str,
        *,
        fact_k: int = 50,
        sparse_k: int = 50,
        song_k: int | None = None,
        media_targets: tuple[str, ...] = (),
    ) -> ContextSongCandidates:
        """Retrieve both rankings against one published Qdrant generation.

        ``fact_k`` limits Dense facts before song aggregation; ``song_k``
        limits the resulting Dense songs. Sparse results already represent
        one profile per song. Keep both score scales and rankings separate so
        the fusion step can combine their *ranks* without counting multiple
        profiles for one song or treating profile overlap as an evidence quote.
        """
        if song_k is not None and song_k < 1:
            raise ValueError("song_k must be positive")
        hits = self.search(query, dense_k=fact_k, sparse_k=sparse_k,
                           **({"media_targets": media_targets} if media_targets else {}))
        sparse_ids = [hit.song_id for hit in hits.sparse_profiles]
        if len(sparse_ids) != len(set(sparse_ids)):
            raise RuntimeError("context sparse search returned more than one profile per song")
        return ContextSongCandidates(
            dense_songs=aggregate_dense_facts_by_song(hits.dense_facts, limit=song_k),
            sparse_songs=hits.sparse_profiles,
            dense_facts=hits.dense_facts,
        )

    def search_fused_songs(
        self,
        query: str,
        *,
        fact_k: int = 50,
        sparse_k: int = 50,
        song_k: int | None = None,
        limit: int | None = None,
        rrf_k: int = CONTEXT_RRF_K,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        media_targets: tuple[str, ...] = (),
    ) -> tuple[ContextFusedSongHit, ...]:
        """Return one Context ranking from the same published collection pair.

        ``fact_k`` is a number of facts, not songs. ``song_k`` caps Dense
        songs after aggregation; ``limit`` caps the fused ranking. The RRF
        score is internal: the SearchRouter integration should give this
        entire ranking *one* Context path contribution based on its rank.
        Retrieved Dense facts are not display-ready evidence until a later
        relevance check accepts them.
        """
        candidates = self.search_song_candidates(
            query, fact_k=fact_k, sparse_k=sparse_k, song_k=song_k,
            **({"media_targets": media_targets} if media_targets else {}),
        )
        return fuse_context_song_candidates(
            candidates,
            limit=limit,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
