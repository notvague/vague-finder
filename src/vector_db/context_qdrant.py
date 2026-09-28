"""Validated, failure-safe Qdrant publication for Namuwiki context vectors.

Context retrieval deliberately has two different document units:

* dense: one point per fact (``record_id``), aggregated by ``song_id`` later;
* sparse: one point per song profile (``profile_id``).

They therefore live in two collections.  A load is published to generation
collections first and both stable aliases are switched in one atomic Qdrant
alias operation only after every point has been verified.  An interrupted load
cannot replace the currently searchable pair with a half-written generation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np
from qdrant_client import models

from src.embedding.context_dense import (
    CONTEXT_DENSE_MANIFEST_VERSION,
    iter_context_dense_embeddings,
    load_context_dense_input,
)
from src.embedding.context_sparse import (
    CONTEXT_SPARSE_MANIFEST_VERSION,
    iter_context_sparse_embeddings,
)
from src.vector_db.qdrant_backend import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    QdrantVectorClient,
    collection_name,
    point_id,
)
from src.vector_db.settings import (
    CONTEXT_DENSE_DIM,
    CONTEXT_DENSE_INDEX_NAME,
    CONTEXT_QDRANT_BATCH_SIZE,
    CONTEXT_SPARSE_INDEX_NAME,
    NAMESPACE,
)

CONTEXT_QDRANT_MANIFEST_VERSION = "context_qdrant_manifest_v1"
CONTEXT_QDRANT_PAYLOAD_VERSION = "context_qdrant_payload_v1"
_GENERATION_MARKER = "__generation_"


@dataclass(frozen=True)
class ContextQdrantPlan:
    context_dir: Path
    dense_dir: Path
    sparse_dir: Path
    namespace: str
    dense_alias: str
    sparse_alias: str
    build_id: str
    source_scope_complete: bool
    source_song_count: int
    dense_record_count: int
    sparse_profile_count: int
    dense_dimension: int
    context_manifest_sha256: str
    dense_manifest_sha256: str
    sparse_manifest_sha256: str
    dense_manifest_content_sha256: str
    sparse_manifest_content_sha256: str
    dense_point_ids: frozenset[str]
    sparse_point_ids: frozenset[str]
    dense_song_ids: frozenset[str]
    sparse_song_ids: frozenset[str]

    def generation_collection(self, alias: str, repair_token: str | None = None) -> str:
        name = f"{alias}{_GENERATION_MARKER}{self.build_id[:16]}"
        return f"{name}_repair_{repair_token}" if repair_token else name


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _manifest_content_digest(manifest: Mapping) -> str:
    value = dict(manifest)
    value.pop("manifest_content_sha256", None)
    return _json_digest(value)


def _read_manifest(path: Path, schema_version: str) -> tuple[dict, str]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"manifest is missing or unsafe: {source}")
    resolved = source.resolve()
    if not resolved.is_file():
        raise ValueError(f"manifest is missing or unsafe: {resolved}")
    raw = resolved.read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != schema_version:
        raise ValueError(f"unsupported manifest schema: {resolved}")
    expected = manifest.get("manifest_content_sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"manifest content hash is missing: {resolved}")
    if expected != _manifest_content_digest(manifest):
        raise ValueError(f"manifest content hash mismatch: {resolved}")
    return manifest, _sha256_bytes(raw)


def _point_key(value: int | str) -> str:
    return str(value)


def _require_nonempty_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"context vector row has empty {label}")
    return text


def _validate_dense_rows(
    context_dir: Path,
    dense_dir: Path,
    *,
    dimension: int,
) -> tuple[frozenset[str], frozenset[str], int]:
    point_ids: set[str] = set()
    song_ids: set[str] = set()
    count = 0
    for row in iter_context_dense_embeddings(
        context_dir=context_dir,
        dense_dir=dense_dir,
    ):
        record_id = _require_nonempty_text(row.get("record_id"), "record_id")
        song_id = _require_nonempty_text(row.get("song_id"), "song_id")
        _require_nonempty_text(row.get("dense_text"), "dense_text")
        _require_nonempty_text(row.get("fact_text"), "fact_text")
        key = _point_key(point_id(record_id))
        if key in point_ids:
            raise ValueError(f"duplicate Qdrant dense point id for record_id={record_id}")
        vector = np.asarray(row.get("values"), dtype=np.float32).reshape(-1)
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid dense vector for record_id={record_id}")
        norm = float(np.linalg.norm(vector))
        if not math.isclose(norm, 1.0, rel_tol=2e-3, abs_tol=2e-3):
            raise ValueError(f"dense vector is not L2-normalized for record_id={record_id}")
        point_ids.add(key)
        song_ids.add(song_id)
        count += 1
    return frozenset(point_ids), frozenset(song_ids), count


def _validate_sparse_rows(
    context_dir: Path,
    sparse_dir: Path,
) -> tuple[frozenset[str], frozenset[str], int]:
    point_ids: set[str] = set()
    song_ids: set[str] = set()
    count = 0
    for row in iter_context_sparse_embeddings(
        context_dir=context_dir,
        sparse_dir=sparse_dir,
    ):
        profile_id = _require_nonempty_text(row.get("profile_id"), "profile_id")
        song_id = _require_nonempty_text(row.get("song_id"), "song_id")
        key = _point_key(point_id(profile_id))
        if key in point_ids:
            raise ValueError(f"duplicate Qdrant sparse point id for profile_id={profile_id}")
        sparse = row.get("sparse_values")
        if not isinstance(sparse, Mapping):
            raise ValueError(f"missing sparse vector for profile_id={profile_id}")
        indices = [int(value) for value in sparse.get("indices", [])]
        values = np.asarray(sparse.get("values", []), dtype=np.float64)
        if (
            not indices
            or len(indices) != len(values)
            or indices != sorted(indices)
            or len(indices) != len(set(indices))
            or any(index < 0 or index > 0xFFFFFFFF for index in indices)
            or not np.isfinite(values).all()
            or not np.all(values > 0)
        ):
            raise ValueError(f"invalid sparse vector for profile_id={profile_id}")
        point_ids.add(key)
        song_ids.add(song_id)
        count += 1
    return frozenset(point_ids), frozenset(song_ids), count


def validate_context_qdrant_inputs(
    *,
    context_dir: Path,
    dense_dir: Path,
    sparse_dir: Path,
    namespace: str = NAMESPACE,
    expected_dense_dim: int = CONTEXT_DENSE_DIM,
    require_complete_scope: bool = False,
) -> ContextQdrantPlan:
    """Validate every source artifact and vector without opening Qdrant."""
    if expected_dense_dim < 1:
        raise ValueError("expected_dense_dim must be positive")
    namespace = str(namespace).strip()
    if not namespace:
        raise ValueError("namespace must not be empty")

    context_dir = Path(context_dir).resolve()
    dense_dir = Path(dense_dir).resolve()
    sparse_dir = Path(sparse_dir).resolve()

    source = load_context_dense_input(
        context_dir,
        require_complete_scope=require_complete_scope,
    )
    context_path = context_dir / "manifest.json"
    context_manifest_sha256 = _sha256_file(context_path)
    dense_manifest, dense_manifest_sha256 = _read_manifest(
        dense_dir / "manifest.json", CONTEXT_DENSE_MANIFEST_VERSION
    )
    sparse_manifest, sparse_manifest_sha256 = _read_manifest(
        sparse_dir / "manifest.json", CONTEXT_SPARSE_MANIFEST_VERSION
    )

    for label, manifest in (("dense", dense_manifest), ("sparse", sparse_manifest)):
        if manifest.get("complete_for_source_manifest") is not True:
            raise ValueError(f"context {label} publication is incomplete")
        if manifest.get("source_manifest_sha256") != context_manifest_sha256:
            raise ValueError(f"context {label} publication targets another context manifest")
        if bool(manifest.get("source_scope_complete")) != source.source_scope_complete:
            raise ValueError(f"context {label} scope-completion flag is stale")

    dense_config = dense_manifest.get("embedding")
    if not isinstance(dense_config, Mapping):
        raise ValueError("context dense embedding config is missing")
    if (
        int(dense_config.get("dimension", -1)) != expected_dense_dim
        or dense_config.get("embedding_unit") != "fact"
        or dense_config.get("metric") != "dotproduct"
        or dense_config.get("l2_normalized") is not True
    ):
        raise ValueError("context dense embedding config is incompatible with Qdrant")

    sparse_config = sparse_manifest.get("bm25")
    if not isinstance(sparse_config, Mapping) or (
        sparse_config.get("document_unit") != "song"
        or sparse_config.get("vector_format") != "indices_values_dotproduct"
        or sparse_config.get("input_field") != "retrieval.sparse_profile.terms"
    ):
        raise ValueError("context sparse BM25 config is incompatible with Qdrant")

    dense_point_ids, dense_song_ids, dense_count = _validate_dense_rows(
        context_dir, dense_dir, dimension=expected_dense_dim
    )
    sparse_point_ids, sparse_song_ids, sparse_count = _validate_sparse_rows(
        context_dir, sparse_dir
    )

    if dense_count != int(dense_manifest.get("record_count", -1)):
        raise ValueError("dense emitted record count differs from its manifest")
    if sparse_count != int(sparse_manifest.get("profile_count", -1)):
        raise ValueError("sparse emitted profile count differs from its manifest")
    if len(source.songs) != int(dense_manifest.get("song_count", -1)):
        raise ValueError("dense song count differs from the context manifest")
    if len(source.songs) != int(sparse_manifest.get("source_song_count", -1)):
        raise ValueError("sparse source song count differs from the context manifest")
    if not dense_song_ids.issubset({song.song_id for song in source.songs}):
        raise ValueError("dense rows contain a song absent from the context manifest")
    if not sparse_song_ids.issubset({song.song_id for song in source.songs}):
        raise ValueError("sparse rows contain a song absent from the context manifest")

    dense_content_sha = _require_nonempty_text(
        dense_manifest.get("manifest_content_sha256"),
        "dense manifest_content_sha256",
    )
    sparse_content_sha = _require_nonempty_text(
        sparse_manifest.get("manifest_content_sha256"),
        "sparse manifest_content_sha256",
    )
    dense_alias = collection_name(CONTEXT_DENSE_INDEX_NAME, namespace)
    sparse_alias = collection_name(CONTEXT_SPARSE_INDEX_NAME, namespace)
    build_id = _json_digest({
        "payload_schema_version": CONTEXT_QDRANT_PAYLOAD_VERSION,
        "namespace": namespace,
        "dense_alias": dense_alias,
        "sparse_alias": sparse_alias,
        "context_manifest_sha256": context_manifest_sha256,
        "dense_manifest_content_sha256": dense_content_sha,
        "sparse_manifest_content_sha256": sparse_content_sha,
        "dense_dimension": expected_dense_dim,
        "dense_vector_name": DENSE_VECTOR,
        "sparse_vector_name": SPARSE_VECTOR,
    })

    return ContextQdrantPlan(
        context_dir=context_dir,
        dense_dir=dense_dir,
        sparse_dir=sparse_dir,
        namespace=namespace,
        dense_alias=dense_alias,
        sparse_alias=sparse_alias,
        build_id=build_id,
        source_scope_complete=source.source_scope_complete,
        source_song_count=len(source.songs),
        dense_record_count=dense_count,
        sparse_profile_count=sparse_count,
        dense_dimension=expected_dense_dim,
        context_manifest_sha256=context_manifest_sha256,
        dense_manifest_sha256=dense_manifest_sha256,
        sparse_manifest_sha256=sparse_manifest_sha256,
        dense_manifest_content_sha256=dense_content_sha,
        sparse_manifest_content_sha256=sparse_content_sha,
        dense_point_ids=dense_point_ids,
        sparse_point_ids=sparse_point_ids,
        dense_song_ids=dense_song_ids,
        sparse_song_ids=sparse_song_ids,
    )


def _batches(iterator: Iterator[dict], size: int) -> Iterator[list[dict]]:
    if size < 1:
        raise ValueError("batch_size must be positive")
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def _dense_payload(row: Mapping, plan: ContextQdrantPlan) -> dict:
    return {
        "payload_schema_version": CONTEXT_QDRANT_PAYLOAD_VERSION,
        "point_kind": "context_fact",
        "context_build_id": plan.build_id,
        "record_schema_version": row.get("record_schema_version"),
        "record_id": str(row["record_id"]),
        "song_id": str(row["song_id"]),
        "title": str(row.get("title", "")),
        "artists": [str(value) for value in row.get("artists", [])],
        "category": str(row.get("category", "")),
        "section": str(row.get("section", "")),
        "source_fact_indices": [int(value) for value in row.get("source_fact_indices", [])],
        "quality": str(row.get("quality", "")),
        "fact_text": str(row.get("fact_text", "")),
        "dense_text": str(row.get("dense_text", "")),
        "keywords": [str(value) for value in row.get("keywords", [])],
        "source_url": str(row.get("source_url", "")),
        "collected_at": str(row.get("collected_at", "")),
    }


def _sparse_payload(row: Mapping, plan: ContextQdrantPlan) -> dict:
    terms = [str(value) for value in row.get("sparse_terms", [])]
    return {
        "payload_schema_version": CONTEXT_QDRANT_PAYLOAD_VERSION,
        "point_kind": "context_song_profile",
        "context_build_id": plan.build_id,
        "profile_id": str(row["profile_id"]),
        "song_id": str(row["song_id"]),
        "title": str(row.get("title", "")),
        "artists": [str(value) for value in row.get("artists", [])],
        "term_count": len(terms),
        "sparse_terms": terms,
    }


def _dense_points(plan: ContextQdrantPlan) -> Iterator[models.PointStruct]:
    for row in iter_context_dense_embeddings(
        context_dir=plan.context_dir,
        dense_dir=plan.dense_dir,
    ):
        yield models.PointStruct(
            id=point_id(str(row["record_id"])),
            vector={DENSE_VECTOR: np.asarray(row["values"], dtype=np.float32).tolist()},
            payload=_dense_payload(row, plan),
        )


def _sparse_points(plan: ContextQdrantPlan) -> Iterator[models.PointStruct]:
    for row in iter_context_sparse_embeddings(
        context_dir=plan.context_dir,
        sparse_dir=plan.sparse_dir,
    ):
        sparse = row["sparse_values"]
        yield models.PointStruct(
            id=point_id(str(row["profile_id"])),
            vector={
                SPARSE_VECTOR: models.SparseVector(
                    indices=[int(value) for value in sparse["indices"]],
                    values=[float(value) for value in sparse["values"]],
                )
            },
            payload=_sparse_payload(row, plan),
        )


def _alias_targets(client: QdrantVectorClient) -> dict[str, str]:
    return {
        str(alias.alias_name): str(alias.collection_name)
        for alias in client.client.get_aliases().aliases
    }


def _collection_config_matches(
    client: QdrantVectorClient,
    collection: str,
    *,
    kind: str,
    dimension: int,
) -> bool:
    info = client.client.get_collection(collection)
    vectors = info.config.params.vectors
    sparse = info.config.params.sparse_vectors or {}
    if kind == "dense":
        if not isinstance(vectors, Mapping) or set(vectors) != {DENSE_VECTOR} or sparse:
            return False
        config = vectors[DENSE_VECTOR]
        return int(config.size) == dimension and config.distance == models.Distance.DOT
    return isinstance(vectors, Mapping) and not vectors and set(sparse) == {SPARSE_VECTOR}


def _verify_collection(
    client: QdrantVectorClient,
    collection: str,
    *,
    kind: str,
    expected_ids: frozenset[str],
    build_id: str,
    dimension: int,
) -> bool:
    if not client.client.collection_exists(collection):
        return False
    if not _collection_config_matches(
        client, collection, kind=kind, dimension=dimension
    ):
        return False
    if int(client.client.count(collection_name=collection, exact=True).count) != len(expected_ids):
        return False

    actual_ids: set[str] = set()
    offset = None
    while True:
        rows, offset = client.client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=["context_build_id", "point_kind"],
            with_vectors=False,
        )
        for row in rows:
            payload = row.payload or {}
            if payload.get("context_build_id") != build_id:
                return False
            expected_kind = "context_fact" if kind == "dense" else "context_song_profile"
            if payload.get("point_kind") != expected_kind:
                return False
            key = _point_key(row.id)
            if key in actual_ids:
                return False
            actual_ids.add(key)
        if offset is None:
            break
    return actual_ids == set(expected_ids)


def _create_generation_collections(
    client: QdrantVectorClient,
    plan: ContextQdrantPlan,
    dense_collection: str,
    sparse_collection: str,
) -> None:
    client.client.create_collection(
        collection_name=dense_collection,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=plan.dense_dimension,
                distance=models.Distance.DOT,
            )
        },
    )
    try:
        client.client.create_collection(
            collection_name=sparse_collection,
            vectors_config={},
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )
    except Exception:
        # Neither collection is active yet.  Remove the sibling so a retry sees
        # one coherent staging pair instead of a misleading partial build.
        client.client.delete_collection(dense_collection)
        raise


def _create_payload_indexes(
    client: QdrantVectorClient,
    dense_collection: str,
    sparse_collection: str,
) -> None:
    # Embedded/local Qdrant performs an exact scan and explicitly ignores
    # payload indexes.  Create them only for server mode, where category and
    # song filters benefit from the index at full-catalogue scale.
    if client.is_local:
        return
    for collection, fields in (
        (dense_collection, ("song_id", "category", "context_build_id")),
        (sparse_collection, ("song_id", "context_build_id")),
    ):
        for field in fields:
            client.client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )


def _upsert_points(
    client: QdrantVectorClient,
    collection: str,
    iterator: Iterator[models.PointStruct],
    *,
    batch_size: int,
    label: str,
    total: int,
    progress: Callable[[str], None] | None,
) -> None:
    loaded = 0
    for batch in _batches(iterator, batch_size):
        client.client.upsert(
            collection_name=collection,
            points=batch,
            wait=True,
        )
        loaded += len(batch)
        if progress:
            percentage = loaded / total * 100.0 if total else 100.0
            progress(f"{label} points={loaded}/{total} ({percentage:.1f}%)")
    if loaded != total:
        raise RuntimeError(f"{label} source changed while loading: {loaded} != {total}")


def _switch_alias_pair(
    client: QdrantVectorClient,
    *,
    dense_alias: str,
    dense_collection: str,
    sparse_alias: str,
    sparse_collection: str,
) -> dict[str, str | None]:
    before = _alias_targets(client)
    if client.client.collection_exists(dense_alias) and dense_alias not in before:
        raise RuntimeError(f"dense alias name is occupied by a collection: {dense_alias}")
    if client.client.collection_exists(sparse_alias) and sparse_alias not in before:
        raise RuntimeError(f"sparse alias name is occupied by a collection: {sparse_alias}")
    if (
        before.get(dense_alias) == dense_collection
        and before.get(sparse_alias) == sparse_collection
    ):
        return {
            "previous_dense_collection": before.get(dense_alias),
            "previous_sparse_collection": before.get(sparse_alias),
        }

    operations: list[models.CreateAliasOperation | models.DeleteAliasOperation] = []
    for alias in (dense_alias, sparse_alias):
        if alias in before:
            operations.append(models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=alias)
            ))
    operations.extend([
        models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name=dense_collection,
            alias_name=dense_alias,
        )),
        models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name=sparse_collection,
            alias_name=sparse_alias,
        )),
    ])
    client.client.update_collection_aliases(change_aliases_operations=operations)
    after = _alias_targets(client)
    if (
        after.get(dense_alias) != dense_collection
        or after.get(sparse_alias) != sparse_collection
    ):
        raise RuntimeError("Qdrant context alias verification failed after switch")
    return {
        "previous_dense_collection": before.get(dense_alias),
        "previous_sparse_collection": before.get(sparse_alias),
    }


def _prune_inactive_generations(
    client: QdrantVectorClient,
    plan: ContextQdrantPlan,
    active: set[str],
) -> list[str]:
    alias_targets = set(_alias_targets(client).values())
    prefixes = (
        f"{plan.dense_alias}{_GENERATION_MARKER}",
        f"{plan.sparse_alias}{_GENERATION_MARKER}",
    )
    removed: list[str] = []
    for description in client.client.get_collections().collections:
        name = str(description.name)
        if (
            name not in active
            and name not in alias_targets
            and name.startswith(prefixes)
        ):
            client.client.delete_collection(name)
            removed.append(name)
    return sorted(removed)


def publish_context_qdrant(
    plan: ContextQdrantPlan,
    client: QdrantVectorClient,
    *,
    batch_size: int = CONTEXT_QDRANT_BATCH_SIZE,
    force: bool = False,
    prune_old_generations: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Publish a validated plan and atomically activate its dense/sparse pair."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    aliases_before = _alias_targets(client)
    dense_collection = plan.generation_collection(plan.dense_alias)
    sparse_collection = plan.generation_collection(plan.sparse_alias)

    dense_valid = _verify_collection(
        client,
        dense_collection,
        kind="dense",
        expected_ids=plan.dense_point_ids,
        build_id=plan.build_id,
        dimension=plan.dense_dimension,
    )
    sparse_valid = _verify_collection(
        client,
        sparse_collection,
        kind="sparse",
        expected_ids=plan.sparse_point_ids,
        build_id=plan.build_id,
        dimension=plan.dense_dimension,
    )

    active_targets = set(aliases_before.values())
    reuse_generation = not force and dense_valid and sparse_valid
    if force or (not reuse_generation and (
        dense_collection in active_targets or sparse_collection in active_targets
    )):
        repair = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:6]
        dense_collection = plan.generation_collection(plan.dense_alias, repair)
        sparse_collection = plan.generation_collection(plan.sparse_alias, repair)
        reuse_generation = False
    elif not reuse_generation:
        # Deterministic inactive staging names are safe to clean and rebuild.
        for collection in (dense_collection, sparse_collection):
            if client.client.collection_exists(collection):
                client.client.delete_collection(collection)

    already_active = (
        reuse_generation
        and aliases_before.get(plan.dense_alias) == dense_collection
        and aliases_before.get(plan.sparse_alias) == sparse_collection
    )
    if already_active:
        action = "reuse"
        alias_history = {
            "previous_dense_collection": dense_collection,
            "previous_sparse_collection": sparse_collection,
        }
    else:
        if not reuse_generation:
            _create_generation_collections(
                client, plan, dense_collection, sparse_collection
            )
            _upsert_points(
                client,
                dense_collection,
                _dense_points(plan),
                batch_size=batch_size,
                label="dense",
                total=plan.dense_record_count,
                progress=progress,
            )
            _upsert_points(
                client,
                sparse_collection,
                _sparse_points(plan),
                batch_size=batch_size,
                label="sparse",
                total=plan.sparse_profile_count,
                progress=progress,
            )
            _create_payload_indexes(client, dense_collection, sparse_collection)
            if not _verify_collection(
                client,
                dense_collection,
                kind="dense",
                expected_ids=plan.dense_point_ids,
                build_id=plan.build_id,
                dimension=plan.dense_dimension,
            ):
                raise RuntimeError("loaded dense context collection failed verification")
            if not _verify_collection(
                client,
                sparse_collection,
                kind="sparse",
                expected_ids=plan.sparse_point_ids,
                build_id=plan.build_id,
                dimension=plan.dense_dimension,
            ):
                raise RuntimeError("loaded sparse context collection failed verification")
            action = "rebuild" if force else "build"
        else:
            action = "activate_existing"

        alias_history = _switch_alias_pair(
            client,
            dense_alias=plan.dense_alias,
            dense_collection=dense_collection,
            sparse_alias=plan.sparse_alias,
            sparse_collection=sparse_collection,
        )

    removed = (
        _prune_inactive_generations(
            client, plan, {dense_collection, sparse_collection}
        )
        if prune_old_generations
        else []
    )
    return {
        "status": "ok",
        "action": action,
        "context_build_id": plan.build_id,
        "source_scope_complete": plan.source_scope_complete,
        "source_song_count": plan.source_song_count,
        "dense_record_count": plan.dense_record_count,
        "sparse_profile_count": plan.sparse_profile_count,
        "dense_alias": plan.dense_alias,
        "sparse_alias": plan.sparse_alias,
        "dense_collection": dense_collection,
        "sparse_collection": sparse_collection,
        **alias_history,
        "pruned_collections": removed,
    }


def _state_manifest(plan: ContextQdrantPlan, result: Mapping) -> dict:
    manifest = {
        "schema_version": CONTEXT_QDRANT_MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": result.get("status"),
        "action": result.get("action"),
        "context_build_id": plan.build_id,
        "source_scope_complete": plan.source_scope_complete,
        "source_song_count": plan.source_song_count,
        "dense_record_count": plan.dense_record_count,
        "sparse_profile_count": plan.sparse_profile_count,
        "source_hashes": {
            "context_manifest_sha256": plan.context_manifest_sha256,
            "dense_manifest_file_sha256": plan.dense_manifest_sha256,
            "dense_manifest_content_sha256": plan.dense_manifest_content_sha256,
            "sparse_manifest_file_sha256": plan.sparse_manifest_sha256,
            "sparse_manifest_content_sha256": plan.sparse_manifest_content_sha256,
        },
        "qdrant": {
            "namespace": plan.namespace,
            "dense_alias": result.get("dense_alias"),
            "sparse_alias": result.get("sparse_alias"),
            "dense_collection": result.get("dense_collection"),
            "sparse_collection": result.get("sparse_collection"),
            "dense_dimension": plan.dense_dimension,
            "dense_metric": "dotproduct",
            "dense_vector_name": DENSE_VECTOR,
            "sparse_vector_name": SPARSE_VECTOR,
            "payload_schema_version": CONTEXT_QDRANT_PAYLOAD_VERSION,
        },
    }
    manifest["manifest_content_sha256"] = _manifest_content_digest(manifest)
    return manifest


def _atomic_write_json(destination: Path, value: Mapping) -> None:
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError(f"refusing to overwrite state-manifest symlink: {destination}")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".context-qdrant-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((
                json.dumps(value, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _preflight_state_destination(destination: Path) -> None:
    """Fail before the alias switch when the audit manifest cannot be written."""
    target = Path(destination)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError(f"state manifest destination is unsafe: {target}")
    parent = target.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".context-qdrant-preflight-", suffix=".tmp", dir=parent
    )
    os.close(descriptor)
    os.unlink(temporary)


def load_context_qdrant(
    *,
    context_dir: Path,
    dense_dir: Path,
    sparse_dir: Path,
    state_manifest: Path,
    namespace: str = NAMESPACE,
    qdrant_path: str | None = None,
    expected_dense_dim: int = CONTEXT_DENSE_DIM,
    batch_size: int = CONTEXT_QDRANT_BATCH_SIZE,
    require_complete_scope: bool = False,
    dry_run: bool = False,
    force: bool = False,
    prune_old_generations: bool = False,
    vector_client: QdrantVectorClient | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Validate and publish context vectors; ``dry_run`` never opens Qdrant."""
    plan = validate_context_qdrant_inputs(
        context_dir=context_dir,
        dense_dir=dense_dir,
        sparse_dir=sparse_dir,
        namespace=namespace,
        expected_dense_dim=expected_dense_dim,
        require_complete_scope=require_complete_scope,
    )
    if progress:
        progress(
            f"validated songs={plan.source_song_count} dense={plan.dense_record_count} "
            f"sparse={plan.sparse_profile_count} scope_complete={plan.source_scope_complete}"
        )
    if dry_run:
        return {
            "schema_version": CONTEXT_QDRANT_MANIFEST_VERSION,
            "status": "dry_run",
            "context_build_id": plan.build_id,
            "source_scope_complete": plan.source_scope_complete,
            "source_song_count": plan.source_song_count,
            "dense_record_count": plan.dense_record_count,
            "sparse_profile_count": plan.sparse_profile_count,
            "dense_alias": plan.dense_alias,
            "sparse_alias": plan.sparse_alias,
            "planned_dense_collection": plan.generation_collection(plan.dense_alias),
            "planned_sparse_collection": plan.generation_collection(plan.sparse_alias),
            "qdrant_opened": False,
        }

    _preflight_state_destination(state_manifest)

    owns_client = vector_client is None
    try:
        client = vector_client or QdrantVectorClient(path=qdrant_path)
    except Exception as exc:
        message = str(exc)
        if "already accessed" in message or "already in use" in message:
            raise RuntimeError(
                "Qdrant storage is already open. Stop the backend before a local-file "
                "load, or use QDRANT_URL server mode."
            ) from exc
        raise

    try:
        result = publish_context_qdrant(
            plan,
            client,
            batch_size=batch_size,
            force=force,
            prune_old_generations=prune_old_generations,
            progress=progress,
        )
    finally:
        if owns_client:
            client.close()

    state = _state_manifest(plan, result)
    _atomic_write_json(state_manifest, state)
    return {
        "schema_version": CONTEXT_QDRANT_MANIFEST_VERSION,
        **result,
        "state_manifest": str(Path(state_manifest).resolve()),
        "state_manifest_content_sha256": state["manifest_content_sha256"],
    }
