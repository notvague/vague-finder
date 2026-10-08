"""Validated, corpus-wide BM25 vectors for Namuwiki song context.

Dense context is embedded per fact.  Sparse context deliberately has a
different unit: one curated lexical profile per song.  BM25 document
frequencies must be fit over *all* current profiles at once, so any corpus
change rebuilds this compact corpus bundle rather than reusing stale per-song
weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Callable, Iterator, Mapping

import numpy as np

from src.crawler.context.schemas import now
from src.embedding.context_artifacts import iter_context_sparse_profiles
from src.embedding.context_dense import load_context_dense_input
from src.embedding.models.context_bm25 import (
    CONTEXT_BM25_DOCUMENT_TOKENIZER,
    CONTEXT_BM25_HASH_ALGORITHM,
    CONTEXT_BM25_PARAMS_VERSION,
    DEFAULT_CONTEXT_BM25_B,
    DEFAULT_CONTEXT_BM25_K1,
    ContextBM25Encoder,
    HashFunction,
)
from src.embedding.text.context_bm25_tokenizer import (
    CONTEXT_BM25_TOKENIZER_VERSION,
    tokenize_context_for_bm25,
)

CONTEXT_SPARSE_MANIFEST_VERSION = "context_sparse_manifest_v1"
CONTEXT_SPARSE_BUNDLE_VERSION = "context_sparse_bundle_v1"
DEFAULT_CONTEXT_SPARSE_OUTPUT = Path("artifacts/embeddings/context_sparse/bm25")


@dataclass(frozen=True)
class ContextSparseProfileInput:
    profile_id: str
    song_id: str
    title: str
    artists: tuple[str, ...]
    artifact_ref: str
    artifact_content_sha256: str
    terms: tuple[str, ...]

    @property
    def terms_sha256(self) -> str:
        return _json_digest(list(self.terms))


@dataclass(frozen=True)
class ContextSparseInput:
    manifest: dict
    source_manifest_sha256: str
    source_input_sha256: str
    source_scope_complete: bool
    source_song_count: int
    profiles: tuple[ContextSparseProfileInput, ...]

    @property
    def term_count(self) -> int:
        return sum(len(profile.terms) for profile in self.profiles)


@dataclass
class ContextBM25QueryEncoder:
    """Query-side encoder pinned to one published context corpus."""

    encoder: ContextBM25Encoder
    manifest_content_sha256: str

    def tokenize(self, query: str) -> list[str]:
        return tokenize_context_for_bm25(query, expand_synonyms=True)

    def encode(self, query: str) -> dict[str, list]:
        return self.encoder.encode_query_terms(self.tokenize(query))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(raw.encode("utf-8"))


def _package_version(name: str) -> str:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return "unknown"


def build_context_bm25_config(
    *,
    b: float = DEFAULT_CONTEXT_BM25_B,
    k1: float = DEFAULT_CONTEXT_BM25_K1,
) -> dict:
    # Constructor validation is cheap and keeps CLI dry-runs honest without
    # importing mmh3 or fitting a corpus.
    ContextBM25Encoder(b=b, k1=k1, hash_fn=lambda _term: 0)
    config = {
        "implementation": "context_pretokenized_bm25_v1",
        "b": float(b),
        "k1": float(k1),
        "hash_algorithm": CONTEXT_BM25_HASH_ALGORITHM,
        "document_tokenizer": CONTEXT_BM25_DOCUMENT_TOKENIZER,
        "query_tokenizer": CONTEXT_BM25_TOKENIZER_VERSION,
        "document_synonym_expansion": False,
        "query_synonym_expansion": True,
        "document_unit": "song",
        "input_field": "retrieval.sparse_profile.terms",
        "vector_format": "indices_values_dotproduct",
        "document_weight": "bm25_tf_length_norm",
        "query_weight": "normalized_bm25_idf",
        "runtime_versions": {
            "mmh3": _package_version("mmh3"),
            "numpy": _package_version("numpy"),
        },
    }
    config["config_sha256"] = _json_digest(config)
    return config


def _source_projection(manifest: Mapping, profiles: tuple[ContextSparseProfileInput, ...]) -> dict:
    policy = manifest.get("retrieval_policy")
    coverage = manifest.get("coverage")
    return {
        "manifest_schema_version": manifest.get("schema_version"),
        "artifact_schema_version": manifest.get("artifact_schema_version"),
        "record_schema_version": manifest.get("record_schema_version"),
        # Scope completion is part of the publication contract even though it
        # is not a vector feature.  Stable counters prevent a pilot manifest
        # from being reused as the final build without making generated_at
        # changes trigger needless refits.
        "coverage": {
            key: coverage.get(key)
            for key in (
                "scope_total",
                "terminal_meta",
                "artifact_ready",
                "pending",
                "coverage_error_count",
            )
        } if isinstance(coverage, Mapping) else None,
        "sparse_policy": {
            "sparse_embedding_unit": policy.get("sparse_embedding_unit")
            if isinstance(policy, Mapping) else None,
            "sparse_input": policy.get("sparse_input")
            if isinstance(policy, Mapping) else None,
        },
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "song_id": profile.song_id,
                "title": profile.title,
                "artists": list(profile.artists),
                "terms_sha256": profile.terms_sha256,
                "term_count": len(profile.terms),
            }
            for profile in profiles
        ],
    }


def load_context_sparse_input(
    context_dir: Path,
    *,
    require_complete_scope: bool = False,
) -> ContextSparseInput:
    """Validate the manifest/artifacts and return every song-level profile."""
    validated = load_context_dense_input(
        context_dir,
        require_complete_scope=require_complete_scope,
    )
    manifest = validated.manifest
    policy = manifest.get("retrieval_policy")
    if not isinstance(policy, Mapping) or (
        policy.get("sparse_embedding_unit") != "song"
        or policy.get("sparse_input") != "retrieval.sparse_profile.terms"
    ):
        raise ValueError("context manifest sparse retrieval policy is incompatible")

    entries = {
        str(entry["song_id"]): entry
        for entry in manifest.get("songs", [])
    }
    profiles: list[ContextSparseProfileInput] = []
    profile_ids: set[str] = set()
    song_ids: set[str] = set()
    for raw in iter_context_sparse_profiles(Path(context_dir)):
        profile_id = str(raw.get("profile_id", ""))
        song_id = str(raw.get("song_id", ""))
        entry = entries.get(song_id)
        if entry is None:
            raise ValueError("context sparse profile belongs to a song absent from manifest")
        if profile_id in profile_ids or song_id in song_ids:
            raise ValueError("duplicate context sparse profile identity")
        profile_ids.add(profile_id)
        song_ids.add(song_id)
        terms = tuple(str(term) for term in raw.get("sparse_terms", []))
        if (
            not terms
            or len(terms) != len(set(terms))
            or " ".join(terms) != raw.get("sparse_passage")
            or len(terms) != int(entry.get("sparse_term_count", -1))
        ):
            raise ValueError(f"invalid context sparse terms for song_id={song_id}")
        profiles.append(ContextSparseProfileInput(
            profile_id=profile_id,
            song_id=song_id,
            title=str(raw.get("title", "")).strip(),
            artists=tuple(str(value).strip() for value in raw.get("artists", [])),
            artifact_ref=str(entry.get("artifact_ref", "")),
            artifact_content_sha256=str(entry.get("artifact_content_sha256", "")),
            terms=terms,
        ))

    expected = int(manifest.get("sparse_profile_count", -1))
    if len(profiles) != expected:
        raise ValueError("context sparse profile count differs from manifest")
    manifest_profile_songs = {
        str(entry["song_id"])
        for entry in manifest.get("songs", [])
        if int(entry.get("sparse_term_count", 0)) > 0
    }
    if song_ids != manifest_profile_songs:
        raise ValueError("context sparse profile songs differ from manifest")

    ordered = tuple(profiles)
    return ContextSparseInput(
        manifest=dict(manifest),
        source_manifest_sha256=validated.source_manifest_sha256,
        source_input_sha256=_json_digest(_source_projection(manifest, ordered)),
        source_scope_complete=validated.source_scope_complete,
        source_song_count=len(validated.songs),
        profiles=ordered,
    )


def _corpus_stats(profiles: tuple[ContextSparseProfileInput, ...]) -> dict:
    lengths = [len(profile.terms) for profile in profiles]
    document_frequency: Counter[str] = Counter()
    for profile in profiles:
        document_frequency.update(set(profile.terms))
    document_count = len(profiles)
    common = sorted(document_frequency.items(), key=lambda item: (-item[1], item[0]))[:20]
    return {
        "document_count": document_count,
        "term_occurrence_count": sum(lengths),
        "unique_term_count": len(document_frequency),
        "singleton_term_count": sum(value == 1 for value in document_frequency.values()),
        "min_document_length": min(lengths) if lengths else 0,
        "average_document_length": (
            sum(lengths) / document_count if document_count else 0.0
        ),
        "max_document_length": max(lengths) if lengths else 0,
        "top_document_frequency_terms": [
            {
                "term": term,
                "document_count": count,
                "document_ratio": round(count / document_count, 6),
            }
            for term, count in common
        ] if document_count else [],
    }


def _hash_collision_stats(
    profiles: tuple[ContextSparseProfileInput, ...],
    encoder: ContextBM25Encoder,
) -> dict:
    buckets: dict[int, set[str]] = defaultdict(set)
    for term in {term for profile in profiles for term in profile.terms}:
        buckets[encoder.hash_term(term)].add(term)
    collisions = [
        {"index": index, "terms": sorted(terms)}
        for index, terms in sorted(buckets.items())
        if len(terms) > 1
    ]
    return {
        "unique_hash_count": len(buckets),
        "collision_bucket_count": len(collisions),
        "hash_collision_count": sum(len(item["terms"]) - 1 for item in collisions),
        "collision_samples": collisions[:20],
    }


def _safe_output_path(root: Path, ref: str) -> Path:
    relative = Path(ref)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("invalid context sparse output reference")
    target = root / relative
    resolved = target.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("context sparse output reference escapes output root")
    return target


def _atomic_write_json(destination: Path, value: Mapping) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"refusing to overwrite output symlink: {destination}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".context-sparse-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _canonical_vector(vector: Mapping) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw_indices = list(vector["indices"])
        raw_values = list(vector["values"])
    except (KeyError, TypeError) as exc:
        raise ValueError("context BM25 returned an invalid sparse vector") from exc
    if len(raw_indices) != len(raw_values):
        raise ValueError("context BM25 sparse indices and values differ in length")
    combined: dict[int, float] = {}
    for raw_index, raw_value in zip(raw_indices, raw_values):
        index, value = int(raw_index), float(raw_value)
        if not 0 <= index <= 0xFFFFFFFF or not math.isfinite(value) or value < 0.0:
            raise ValueError("context BM25 sparse vector contains an invalid value")
        if value:
            combined[index] = combined.get(index, 0.0) + value
    pairs = sorted(combined.items())
    return (
        np.asarray([index for index, _ in pairs], dtype=np.uint32),
        np.asarray([value for _, value in pairs], dtype=np.float32),
    )


def _pack_vectors(vectors: list[Mapping]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr = [0]
    all_indices: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    for vector in vectors:
        indices, values = _canonical_vector(vector)
        if not len(indices):
            raise ValueError("context BM25 document vector is empty")
        all_indices.append(indices)
        all_values.append(values)
        indptr.append(indptr[-1] + len(indices))
    return (
        np.asarray(indptr, dtype=np.int64),
        np.concatenate(all_indices) if all_indices else np.asarray([], dtype=np.uint32),
        np.concatenate(all_values) if all_values else np.asarray([], dtype=np.float32),
    )


def _atomic_write_bundle(
    destination: Path,
    *,
    inputs: ContextSparseInput,
    config: Mapping,
    vectors: list[Mapping],
) -> None:
    indptr, indices, values = _pack_vectors(vectors)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"refusing to overwrite output symlink: {destination}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".context-sparse-corpus-", suffix=".npz", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                bundle_schema_version=np.asarray(CONTEXT_SPARSE_BUNDLE_VERSION),
                source_input_sha256=np.asarray(inputs.source_input_sha256),
                encoder_config_sha256=np.asarray(config["config_sha256"]),
                profile_ids=np.asarray([p.profile_id for p in inputs.profiles]),
                song_ids=np.asarray([p.song_id for p in inputs.profiles]),
                terms_sha256=np.asarray([p.terms_sha256 for p in inputs.profiles]),
                indptr=indptr,
                indices=indices,
                values=values,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar context sparse bundle field")
    return str(array.reshape(-1)[0])


def _load_bundle(
    path: Path,
    *,
    inputs: ContextSparseInput,
    config: Mapping,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("context sparse corpus bundle is missing or unsafe")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            required = {
                "bundle_schema_version", "source_input_sha256",
                "encoder_config_sha256", "profile_ids", "song_ids",
                "terms_sha256", "indptr", "indices", "values",
            }
            if not required.issubset(bundle.files):
                raise ValueError("context sparse corpus bundle fields are incomplete")
            if _scalar_text(bundle["bundle_schema_version"]) != CONTEXT_SPARSE_BUNDLE_VERSION:
                raise ValueError("context sparse corpus bundle schema mismatch")
            if _scalar_text(bundle["source_input_sha256"]) != inputs.source_input_sha256:
                raise ValueError("context sparse corpus bundle is stale")
            if _scalar_text(bundle["encoder_config_sha256"]) != config["config_sha256"]:
                raise ValueError("context sparse corpus encoder config mismatch")
            if [str(value) for value in bundle["profile_ids"].tolist()] != [
                profile.profile_id for profile in inputs.profiles
            ]:
                raise ValueError("context sparse profile order mismatch")
            if [str(value) for value in bundle["song_ids"].tolist()] != [
                profile.song_id for profile in inputs.profiles
            ]:
                raise ValueError("context sparse song order mismatch")
            if [str(value) for value in bundle["terms_sha256"].tolist()] != [
                profile.terms_sha256 for profile in inputs.profiles
            ]:
                raise ValueError("context sparse term hash mismatch")
            indptr = np.asarray(bundle["indptr"])
            indices = np.asarray(bundle["indices"])
            values = np.asarray(bundle["values"])
            if indptr.dtype != np.int64 or indptr.shape != (len(inputs.profiles) + 1,):
                raise ValueError("context sparse CSR indptr is invalid")
            if indices.dtype != np.uint32 or values.dtype != np.float32:
                raise ValueError("context sparse CSR value dtypes are invalid")
            if indices.ndim != 1 or values.ndim != 1 or len(indices) != len(values):
                raise ValueError("context sparse CSR arrays are invalid")
            if (
                not len(indptr)
                or indptr[0] != 0
                or np.any(indptr[1:] < indptr[:-1])
                or indptr[-1] != len(indices)
            ):
                raise ValueError("context sparse CSR offsets are invalid")
            if not np.isfinite(values).all() or np.any(values <= 0.0):
                raise ValueError("context sparse CSR weights are invalid")
            for row in range(len(inputs.profiles)):
                start, stop = int(indptr[row]), int(indptr[row + 1])
                if start == stop or np.any(indices[start + 1:stop] <= indices[start:stop - 1]):
                    raise ValueError("context sparse CSR row indices are invalid")
            return indptr.copy(), indices.copy(), values.copy()
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid context sparse corpus bundle {path}: {exc}") from exc


def _manifest_digest(manifest: Mapping) -> str:
    value = dict(manifest)
    value.pop("manifest_content_sha256", None)
    return _json_digest(value)


def _read_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("context sparse manifest is missing or unsafe")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CONTEXT_SPARSE_MANIFEST_VERSION:
        raise ValueError("unsupported context sparse manifest schema")
    if manifest.get("manifest_content_sha256") != _manifest_digest(manifest):
        raise ValueError("context sparse manifest content hash mismatch")
    if manifest.get("complete_for_source_manifest") is not True:
        raise ValueError("context sparse manifest is incomplete")
    config = manifest.get("bm25")
    if not isinstance(config, Mapping) or config.get("config_sha256") != _json_digest({
        key: value for key, value in config.items() if key != "config_sha256"
    }):
        raise ValueError("context sparse BM25 config hash mismatch")
    return manifest


def _read_params(
    root: Path,
    manifest: Mapping,
    *,
    hash_fn: HashFunction | None = None,
) -> tuple[dict, ContextBM25Encoder]:
    ref = manifest.get("params_ref")
    if not isinstance(ref, str):
        raise ValueError("context sparse BM25 params reference is missing")
    path = _safe_output_path(root, ref)
    if path.is_symlink() or not path.is_file():
        raise ValueError("context sparse BM25 params are missing or unsafe")
    if _sha256_file(path) != manifest.get("params_sha256"):
        raise ValueError("context sparse BM25 params hash mismatch")
    params = json.loads(path.read_text(encoding="utf-8"))
    encoder = ContextBM25Encoder.from_params(params, hash_fn=hash_fn)
    if encoder.n_docs != int(manifest.get("profile_count", -1)):
        raise ValueError("context sparse BM25 params document count mismatch")
    return params, encoder


def _profile_entries(inputs: ContextSparseInput) -> list[dict]:
    return [
        {
            "row": row,
            "profile_id": profile.profile_id,
            "song_id": profile.song_id,
            "artifact_ref": profile.artifact_ref,
            "artifact_content_sha256": profile.artifact_content_sha256,
            "term_count": len(profile.terms),
            "terms_sha256": profile.terms_sha256,
        }
        for row, profile in enumerate(inputs.profiles)
    ]


def _validate_published(
    root: Path,
    *,
    inputs: ContextSparseInput,
    expected_config: Mapping | None = None,
    hash_fn: HashFunction | None = None,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    manifest = _read_manifest(root)
    if manifest.get("source_input_sha256") != inputs.source_input_sha256:
        raise ValueError("context artifacts changed after context BM25 fitting")
    # Artifact-only runs may republish an otherwise identical context catalogue
    # with a new generated_at. Its searchable profiles are unchanged, but the
    # final Qdrant loader requires both vector manifests to target these exact
    # source-manifest bytes. Refit/re-publish rather than reusing a stale hash.
    if manifest.get("source_manifest_sha256") != inputs.source_manifest_sha256:
        raise ValueError("context sparse publication targets another context manifest")
    config = manifest["bm25"]
    if expected_config is not None and config.get("config_sha256") != expected_config.get(
        "config_sha256"
    ):
        raise ValueError("context BM25 build config changed")
    if int(manifest.get("source_song_count", -1)) != inputs.source_song_count:
        raise ValueError("context sparse source song count mismatch")
    if int(manifest.get("profile_count", -1)) != len(inputs.profiles):
        raise ValueError("context sparse profile count mismatch")
    if int(manifest.get("term_occurrence_count", -1)) != inputs.term_count:
        raise ValueError("context sparse term count mismatch")
    if manifest.get("profiles") != _profile_entries(inputs):
        raise ValueError("context sparse manifest profile entries differ")

    if not inputs.profiles:
        if manifest.get("params_ref") is not None or manifest.get("documents_ref") is not None:
            raise ValueError("empty context sparse corpus unexpectedly has output files")
        return manifest, None

    params, encoder = _read_params(root, manifest, hash_fn=hash_fn)
    if params.get("schema_version") != CONTEXT_BM25_PARAMS_VERSION:
        raise ValueError("context sparse BM25 parameter schema mismatch")
    expected_avgdl = inputs.term_count / len(inputs.profiles)
    if not math.isclose(float(encoder.avgdl), expected_avgdl, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("context sparse BM25 average document length mismatch")
    ref = manifest.get("documents_ref")
    if not isinstance(ref, str):
        raise ValueError("context sparse documents reference is missing")
    path = _safe_output_path(root, ref)
    if _sha256_file(path) != manifest.get("documents_sha256"):
        raise ValueError("context sparse documents hash mismatch")
    bundle = _load_bundle(path, inputs=inputs, config=config)
    return manifest, bundle


def build_context_bm25(
    *,
    context_dir: Path,
    output_dir: Path,
    b: float = DEFAULT_CONTEXT_BM25_B,
    k1: float = DEFAULT_CONTEXT_BM25_K1,
    force: bool = False,
    dry_run: bool = False,
    require_complete_scope: bool = False,
    encoder_factory: Callable[..., ContextBM25Encoder] | None = None,
    hash_fn: HashFunction | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Fit all profiles, encode every song and atomically publish a manifest."""
    report = progress or (lambda _message: None)
    inputs = load_context_sparse_input(
        context_dir,
        require_complete_scope=require_complete_scope,
    )
    config = build_context_bm25_config(b=b, k1=k1)
    root = Path(output_dir).resolve()
    stats = _corpus_stats(inputs.profiles)

    reusable = False
    cache_rejection: str | None = None
    if not force and (root / "manifest.json").exists():
        try:
            _validate_published(
                root,
                inputs=inputs,
                expected_config=config,
                hash_fn=hash_fn,
            )
        except (OSError, ValueError) as exc:
            cache_rejection = str(exc)
        else:
            reusable = True

    action = "reuse" if reusable else ("empty" if not inputs.profiles else "build")
    summary = {
        "schema_version": CONTEXT_SPARSE_MANIFEST_VERSION,
        "status": "dry_run" if dry_run else "ok",
        "action": action,
        "context_dir": str(Path(context_dir).resolve()),
        "output_dir": str(root),
        "source_scope_complete": inputs.source_scope_complete,
        "source_song_count": inputs.source_song_count,
        "source_profile_count": len(inputs.profiles),
        "source_term_count": inputs.term_count,
        "documents_to_fit": 0 if reusable else len(inputs.profiles),
        "bm25_config": config,
        "corpus_stats": stats,
        "cache_rejection_count": int(cache_rejection is not None),
    }
    report(
        f"validated songs={inputs.source_song_count} profiles={len(inputs.profiles)} "
        f"terms={inputs.term_count} scope_complete={inputs.source_scope_complete}"
    )
    report(
        f"plan action={action} documents_to_fit={summary['documents_to_fit']}"
    )
    if dry_run:
        if cache_rejection:
            summary["cache_rejection_reason"] = cache_rejection
        return summary
    if reusable:
        manifest = _read_manifest(root)
        summary.update({
            "reused_corpus": True,
            "output_manifest": str(root / "manifest.json"),
            "manifest_content_sha256": manifest["manifest_content_sha256"],
        })
        return summary

    entries = _profile_entries(inputs)
    if not inputs.profiles:
        manifest = {
            "schema_version": CONTEXT_SPARSE_MANIFEST_VERSION,
            "generated_at": now(),
            "complete_for_source_manifest": True,
            "source_scope_complete": inputs.source_scope_complete,
            "source_manifest_ref": str(Path(context_dir).resolve() / "manifest.json"),
            "source_manifest_sha256": inputs.source_manifest_sha256,
            "source_input_sha256": inputs.source_input_sha256,
            "bm25": config,
            "source_song_count": inputs.source_song_count,
            "profile_count": 0,
            "term_occurrence_count": 0,
            "corpus_stats": stats,
            "hash_stats": {
                "unique_hash_count": 0,
                "collision_bucket_count": 0,
                "hash_collision_count": 0,
                "collision_samples": [],
            },
            "params_ref": None,
            "params_sha256": None,
            "documents_ref": None,
            "documents_sha256": None,
            "profiles": entries,
        }
        manifest["manifest_content_sha256"] = _manifest_digest(manifest)
        _atomic_write_json(root / "manifest.json", manifest)
        summary.update({
            "reused_corpus": False,
            "output_manifest": str(root / "manifest.json"),
            "manifest_content_sha256": manifest["manifest_content_sha256"],
        })
        return summary

    active_encoder = (
        encoder_factory(b=b, k1=k1)
        if encoder_factory is not None
        else ContextBM25Encoder(b=b, k1=k1, hash_fn=hash_fn)
    )
    documents = [profile.terms for profile in inputs.profiles]
    report(f"fitting corpus documents={len(documents)}")
    active_encoder.fit(documents)
    vectors = active_encoder.encode_documents(documents)
    report(f"encoded documents={len(vectors)}/{len(documents)} (100.0%)")

    params = active_encoder.get_params()
    hash_stats = _hash_collision_stats(inputs.profiles, active_encoder)
    build_key = f"{inputs.source_input_sha256[:20]}-{config['config_sha256'][:12]}"
    params_ref = f"params/{build_key}.json"
    documents_ref = f"corpora/{build_key}.npz"
    params_path = _safe_output_path(root, params_ref)
    documents_path = _safe_output_path(root, documents_ref)
    _atomic_write_json(params_path, params)
    _atomic_write_bundle(
        documents_path,
        inputs=inputs,
        config=config,
        vectors=vectors,
    )

    manifest = {
        "schema_version": CONTEXT_SPARSE_MANIFEST_VERSION,
        "generated_at": now(),
        "complete_for_source_manifest": True,
        "source_scope_complete": inputs.source_scope_complete,
        "source_manifest_ref": str(Path(context_dir).resolve() / "manifest.json"),
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "source_input_sha256": inputs.source_input_sha256,
        "bm25": config,
        "source_song_count": inputs.source_song_count,
        "profile_count": len(inputs.profiles),
        "term_occurrence_count": inputs.term_count,
        "corpus_stats": stats,
        "hash_stats": hash_stats,
        "params_ref": params_ref,
        "params_sha256": _sha256_file(params_path),
        "documents_ref": documents_ref,
        "documents_sha256": _sha256_file(documents_path),
        "profiles": entries,
    }
    manifest["manifest_content_sha256"] = _manifest_digest(manifest)
    _atomic_write_json(root / "manifest.json", manifest)
    _validate_published(
        root,
        inputs=inputs,
        expected_config=config,
        hash_fn=hash_fn,
    )
    summary.update({
        "reused_corpus": False,
        "fitted_document_count": len(inputs.profiles),
        "encoded_document_count": len(vectors),
        "params_ref": params_ref,
        "documents_ref": documents_ref,
        "hash_stats": hash_stats,
        "output_manifest": str(root / "manifest.json"),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
    })
    if cache_rejection:
        summary["cache_rejection_reason"] = cache_rejection
    return summary


def iter_context_sparse_embeddings(
    *,
    context_dir: Path,
    sparse_dir: Path,
) -> Iterator[dict]:
    """Yield validated song metadata and BM25 sparse values for DB loading."""
    inputs = load_context_sparse_input(context_dir)
    manifest, bundle = _validate_published(
        Path(sparse_dir).resolve(),
        inputs=inputs,
    )
    if bundle is None:
        return
    indptr, indices, values = bundle
    for row, profile in enumerate(inputs.profiles):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        yield {
            "profile_id": profile.profile_id,
            "song_id": profile.song_id,
            "title": profile.title,
            "artists": list(profile.artists),
            "sparse_terms": list(profile.terms),
            "sparse_passage": " ".join(profile.terms),
            "sparse_values": {
                "indices": [int(value) for value in indices[start:stop]],
                "values": [float(value) for value in values[start:stop]],
            },
            "context_sparse_manifest_sha256": manifest["manifest_content_sha256"],
        }


def load_context_bm25_query_encoder(
    sparse_dir: Path,
    *,
    hash_fn: HashFunction | None = None,
) -> ContextBM25QueryEncoder:
    """Load the exact corpus statistics and tokenizer contract used at build time."""
    root = Path(sparse_dir).resolve()
    manifest = _read_manifest(root)
    if int(manifest.get("profile_count", 0)) == 0:
        raise ValueError("context sparse corpus has no searchable profiles")
    _params, encoder = _read_params(root, manifest, hash_fn=hash_fn)
    return ContextBM25QueryEncoder(
        encoder=encoder,
        manifest_content_sha256=str(manifest["manifest_content_sha256"]),
    )
