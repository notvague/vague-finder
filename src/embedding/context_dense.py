"""Validated, resumable KoE5 embeddings for Namuwiki context facts.

The crawler artifact and the vector cache have deliberately different units:

* input artifact: one JSON per song, containing fact-level ``dense_text``;
* dense cache: one NPZ per song, containing one vector per fact;
* future vector DB: one point per fact, grouped by ``song_id`` at query time.

Keeping all fact vectors for one song in one NPZ avoids tens of thousands of
tiny files when the catalogue grows.  Every run validates the context manifest,
artifact hashes and lossless source-fact coverage before loading KoE5.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from importlib import metadata as package_metadata
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator, Mapping

import numpy as np

from src.crawler.context.schemas import now
from src.embedding.context_artifacts import (
    CONTEXT_MANIFEST_VERSION,
    ContextArtifactStore,
    iter_context_artifact_records,
)
from src.embedding.models.text_koe5 import DEFAULT_KOE5_MODEL, KoE5Embedder
from src.embedding.text.namuwiki_passage import (
    CONTEXT_ARTIFACT_VERSION,
    CONTEXT_RECORD_VERSION,
    DENSE_TEXT_FORMAT,
)

CONTEXT_DENSE_MANIFEST_VERSION = "context_dense_manifest_v1"
CONTEXT_DENSE_BUNDLE_VERSION = "context_dense_bundle_v1"
DEFAULT_CONTEXT_DENSE_DIM = 1024
DEFAULT_CONTEXT_DENSE_BATCH_SIZE = 32
_MODEL_TAG = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ContextDenseSongInput:
    song_id: str
    title: str
    artists: tuple[str, ...]
    artifact_ref: str
    artifact_content_sha256: str
    records: tuple[dict, ...]


@dataclass(frozen=True)
class ContextDenseInput:
    manifest: dict
    source_manifest_sha256: str
    source_input_sha256: str
    source_scope_complete: bool
    songs: tuple[ContextDenseSongInput, ...]

    @property
    def record_count(self) -> int:
        return sum(len(song.records) for song in self.songs)


def safe_model_tag(model_name: str) -> str:
    value = (
        str(model_name).strip().replace("/", "__").replace("\\", "__")
        .replace(":", "_").replace(" ", "_")
    )
    if not value or not _MODEL_TAG.fullmatch(value):
        raise ValueError("model tag may contain only letters, numbers, dot, underscore and hyphen")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(raw)


def _runtime_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("sentence-transformers", "transformers", "torch", "numpy"):
        try:
            result[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            result[package] = "unknown"
    return result


def build_embedding_config(
    *,
    model_name: str = DEFAULT_KOE5_MODEL,
    model_revision: str | None = None,
    model_tag: str | None = None,
    expected_dim: int = DEFAULT_CONTEXT_DENSE_DIM,
) -> dict:
    if expected_dim < 1:
        raise ValueError("expected_dim must be positive")
    resolved_tag = safe_model_tag(model_tag or model_name)
    config = {
        "model_name": str(model_name).strip(),
        "model_revision": str(model_revision).strip() if model_revision else None,
        "model_tag": resolved_tag,
        "dimension": int(expected_dim),
        "input_field": "retrieval.records[].dense_text",
        "embedding_unit": "fact",
        "e5_prefix": "passage: ",
        "l2_normalized": True,
        "metric": "dotproduct",
        "runtime_versions": _runtime_versions(),
    }
    config["config_sha256"] = _json_digest(config)
    return config


def _source_input_projection(manifest: Mapping) -> dict:
    songs = []
    for entry in manifest.get("songs", []):
        songs.append({
            "song_id": str(entry.get("song_id")),
            "status": entry.get("status"),
            "artifact_ref": entry.get("artifact_ref"),
            "artifact_content_sha256": entry.get("artifact_content_sha256"),
            "record_count": entry.get("record_count"),
            "source_fact_count": entry.get("source_fact_count"),
            "covered_source_fact_count": entry.get("covered_source_fact_count"),
        })
    return {
        "schema_version": manifest.get("schema_version"),
        "artifact_schema_version": manifest.get("artifact_schema_version"),
        "record_schema_version": manifest.get("record_schema_version"),
        "retrieval_policy": manifest.get("retrieval_policy"),
        "songs": songs,
    }


def _scope_is_complete(coverage: Mapping) -> bool:
    try:
        scope_total = int(coverage["scope_total"])
        ready = int(coverage["artifact_ready"])
        pending = int(coverage["pending"])
        errors = int(coverage.get("coverage_error_count", 0))
    except (KeyError, TypeError, ValueError):
        return False
    return scope_total >= 0 and ready == scope_total and pending == 0 and errors == 0


def _validate_catalogue_coverage(coverage: Mapping, manifest: Mapping) -> None:
    """Reject internally inconsistent progress even for a partial pilot run."""
    required = (
        "scope_total", "terminal_meta", "artifact_ready", "pending",
        "context_record_count", "coverage_error_count",
    )
    try:
        values = {key: int(coverage[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("context catalogue coverage is incomplete or invalid") from exc
    if any(value < 0 for value in values.values()):
        raise ValueError("context catalogue coverage contains negative counts")
    if values["artifact_ready"] > values["scope_total"]:
        raise ValueError("context catalogue artifact_ready exceeds scope_total")
    if values["terminal_meta"] < values["artifact_ready"]:
        raise ValueError("context catalogue terminal_meta is below artifact_ready")
    if values["pending"] != values["scope_total"] - values["artifact_ready"]:
        raise ValueError("context catalogue pending count is inconsistent")
    if values["artifact_ready"] != int(manifest.get("song_count", -1)):
        raise ValueError("context catalogue artifact_ready differs from manifest song_count")
    if values["context_record_count"] != int(manifest.get("record_count", -1)):
        raise ValueError("context catalogue record count differs from manifest record_count")
    errors = coverage.get("coverage_errors")
    if not isinstance(errors, list) or values["coverage_error_count"] != len(errors):
        raise ValueError("context catalogue coverage error count is inconsistent")
    status_counts = coverage.get("status_counts")
    if not isinstance(status_counts, Mapping):
        raise ValueError("context catalogue status_counts is invalid")
    try:
        status_total = sum(int(value) for value in status_counts.values())
    except (TypeError, ValueError) as exc:
        raise ValueError("context catalogue status_counts is invalid") from exc
    if status_total != values["terminal_meta"]:
        raise ValueError("context catalogue status_counts differs from terminal_meta")
    try:
        percentage = float(coverage["completion_percent"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("context catalogue completion_percent is invalid") from exc
    expected = round(
        values["artifact_ready"] / values["scope_total"] * 100.0
        if values["scope_total"] else 100.0,
        1,
    )
    if abs(percentage - expected) > 0.05:
        raise ValueError("context catalogue completion_percent is inconsistent")


def load_context_dense_input(
    context_dir: Path,
    *,
    require_complete_scope: bool = False,
) -> ContextDenseInput:
    """Validate every current artifact before any model or output is touched.

    ``require_complete_scope=False`` supports a pilot subset.  It never relaxes
    per-artifact hashes or 100% source-fact coverage; it only allows the crawler
    catalogue's ``coverage.pending`` value to be non-zero.
    """
    root = Path(context_dir).resolve()
    store = ContextArtifactStore(root)
    manifest_path = store.path("manifest.json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"context manifest not found or unsafe: {manifest_path}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))

    if manifest.get("schema_version") != CONTEXT_MANIFEST_VERSION:
        raise ValueError("unsupported context manifest schema")
    if manifest.get("artifact_schema_version") != CONTEXT_ARTIFACT_VERSION:
        raise ValueError("context artifact schema version mismatch")
    if manifest.get("record_schema_version") != CONTEXT_RECORD_VERSION:
        raise ValueError("context record schema version mismatch")
    if manifest.get("artifacts_valid") is not True or int(
        manifest.get("invalid_artifact_count", -1)
    ) != 0:
        raise ValueError("context manifest contains invalid artifacts")

    entries = manifest.get("songs")
    if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
        raise ValueError("context manifest songs must be a list")
    if int(manifest.get("song_count", -1)) != len(entries):
        raise ValueError("context manifest song_count differs from songs")
    song_ids = [str(entry.get("song_id", "")) for entry in entries]
    if any(not song_id.isdigit() for song_id in song_ids):
        raise ValueError("context manifest contains a non-numeric song_id")
    if len(song_ids) != len(set(song_ids)):
        raise ValueError("context manifest contains duplicate song_id values")
    artifact_refs = [entry.get("artifact_ref") for entry in entries]
    if any(not isinstance(ref, str) for ref in artifact_refs):
        raise ValueError("context manifest contains an invalid artifact_ref")
    if len(artifact_refs) != len(set(artifact_refs)):
        raise ValueError("context manifest contains duplicate artifact_ref values")

    policy = manifest.get("retrieval_policy")
    expected_policy = {
        "dense_embedding_unit": "fact",
        "dense_text_format": DENSE_TEXT_FORMAT,
        "dense_input": "records[].dense_text",
        "dense_group_by": "song_id",
        "dense_score_aggregation": "max_per_song",
    }
    if not isinstance(policy, Mapping) or any(
        policy.get(key) != value for key, value in expected_policy.items()
    ):
        raise ValueError("context manifest dense retrieval policy is incompatible")

    manifest_status_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("status", ""))
        manifest_status_counts[status] = manifest_status_counts.get(status, 0) + 1
    if manifest_status_counts != manifest.get("status_counts"):
        raise ValueError("context manifest status_counts differs from song entries")
    if int(manifest.get("sparse_profile_count", -1)) != sum(
        int(entry.get("sparse_term_count", 0)) > 0 for entry in entries
    ):
        raise ValueError("context manifest sparse_profile_count differs from song entries")
    invalid_artifacts = manifest.get("invalid_artifacts")
    if not isinstance(invalid_artifacts, list) or len(invalid_artifacts) != int(
        manifest.get("invalid_artifact_count", -1)
    ):
        raise ValueError("context manifest invalid_artifact_count is inconsistent")
    orphan_artifacts = manifest.get("orphan_artifacts")
    if not isinstance(orphan_artifacts, list) or len(orphan_artifacts) != int(
        manifest.get("orphan_artifact_count", -1)
    ):
        raise ValueError("context manifest orphan_artifact_count is inconsistent")

    coverage = manifest.get("coverage") if isinstance(manifest.get("coverage"), Mapping) else {}
    _validate_catalogue_coverage(coverage, manifest)
    coverage_errors = int(coverage.get("coverage_error_count", 0))
    if coverage_errors or coverage.get("coverage_errors"):
        raise ValueError("context catalogue coverage contains errors")
    scope_complete = _scope_is_complete(coverage)
    if require_complete_scope and not scope_complete:
        raise ValueError(
            "context catalogue is incomplete; finish artifact generation before the final embedding run"
        )
    if require_complete_scope and int(manifest.get("orphan_artifact_count", 0)) != 0:
        raise ValueError("final embedding refuses a context manifest with orphan artifacts")

    # This iterator validates each artifact schema, self-hash, manifest binding,
    # record count and lossless source-fact coverage.
    records_by_song = {song_id: [] for song_id in song_ids}
    record_ids: set[str] = set()
    for record in iter_context_artifact_records(root):
        song_id = str(record.get("song_id", ""))
        if song_id not in records_by_song:
            raise ValueError("context iterator emitted a song absent from manifest")
        record_id = str(record.get("record_id", ""))
        if record_id in record_ids:
            raise ValueError(f"duplicate context record_id across artifacts: {record_id}")
        record_ids.add(record_id)
        records_by_song[song_id].append(record)

    songs: list[ContextDenseSongInput] = []
    total_records = 0
    for entry in entries:
        song_id = str(entry["song_id"])
        records = tuple(records_by_song[song_id])
        expected = int(entry.get("record_count", -1))
        if expected != len(records):
            raise ValueError(f"context record count mismatch for song_id={song_id}")
        if int(entry.get("source_fact_count", -1)) != int(
            entry.get("covered_source_fact_count", -2)
        ):
            raise ValueError(f"context source facts are not fully covered for song_id={song_id}")
        total_records += len(records)
        songs.append(ContextDenseSongInput(
            song_id=song_id,
            title=str(entry.get("title", "")).strip(),
            artists=tuple(str(value).strip() for value in entry.get("artists", [])),
            artifact_ref=str(entry["artifact_ref"]),
            artifact_content_sha256=str(entry.get("artifact_content_sha256", "")),
            records=records,
        ))
    if total_records != int(manifest.get("record_count", -1)):
        raise ValueError("context manifest record_count differs from validated records")

    return ContextDenseInput(
        manifest=dict(manifest),
        source_manifest_sha256=_sha256_bytes(raw),
        source_input_sha256=_json_digest(_source_input_projection(manifest)),
        source_scope_complete=scope_complete,
        songs=tuple(songs),
    )


def _bundle_ref(song_id: str) -> str:
    if not str(song_id).isdigit():
        raise ValueError("numeric song_id is required for a dense bundle")
    return f"songs/{song_id}.npz"


def _safe_output_path(root: Path, ref: str) -> Path:
    relative = Path(ref)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("invalid context dense reference")
    target = root / relative
    resolved = target.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("context dense reference escapes output root")
    return target


def _record_ids(song: ContextDenseSongInput) -> list[str]:
    return [str(record["record_id"]) for record in song.records]


def _dense_text_hashes(song: ContextDenseSongInput) -> list[str]:
    return [_sha256_text(str(record["dense_text"])) for record in song.records]


def _scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar bundle field")
    return str(array.reshape(-1)[0])


def _validate_vectors(vectors: np.ndarray, *, rows: int, dim: int) -> np.ndarray:
    value = np.asarray(vectors)
    if value.dtype != np.float32:
        raise ValueError("context vectors must be float32")
    if value.shape != (rows, dim):
        raise ValueError(f"context vector shape is {value.shape}, expected {(rows, dim)}")
    if not np.isfinite(value).all():
        raise ValueError("context vectors contain NaN or Inf")
    if rows:
        norms = np.linalg.norm(value, axis=1)
        if not np.allclose(norms, np.ones(rows), rtol=2e-3, atol=2e-3):
            raise ValueError("context vectors are not L2-normalized")
    return value


def _load_valid_bundle(
    path: Path,
    *,
    song: ContextDenseSongInput,
    config: Mapping,
) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ValueError("context dense bundle is missing or unsafe")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            required = {
                "bundle_schema_version", "song_id", "artifact_content_sha256",
                "embedding_config_sha256", "record_ids", "dense_text_sha256", "vectors",
            }
            if not required.issubset(bundle.files):
                raise ValueError("context dense bundle fields are incomplete")
            if _scalar_text(bundle["bundle_schema_version"]) != CONTEXT_DENSE_BUNDLE_VERSION:
                raise ValueError("context dense bundle schema mismatch")
            if _scalar_text(bundle["song_id"]) != song.song_id:
                raise ValueError("context dense bundle song_id mismatch")
            if _scalar_text(bundle["artifact_content_sha256"]) != song.artifact_content_sha256:
                raise ValueError("context dense bundle is stale")
            if _scalar_text(bundle["embedding_config_sha256"]) != config["config_sha256"]:
                raise ValueError("context dense bundle embedding config mismatch")
            if [str(value) for value in bundle["record_ids"].tolist()] != _record_ids(song):
                raise ValueError("context dense bundle record order mismatch")
            if [str(value) for value in bundle["dense_text_sha256"].tolist()] != _dense_text_hashes(song):
                raise ValueError("context dense bundle text hash mismatch")
            return _validate_vectors(
                bundle["vectors"], rows=len(song.records), dim=int(config["dimension"])
            ).copy()
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid context dense bundle {path}: {exc}") from exc


def _atomic_write_bundle(
    destination: Path,
    *,
    song: ContextDenseSongInput,
    config: Mapping,
    vectors: np.ndarray,
) -> None:
    vectors = _validate_vectors(
        np.asarray(vectors, dtype=np.float32),
        rows=len(song.records),
        dim=int(config["dimension"]),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"refusing to overwrite bundle symlink: {destination}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".context-dense-", suffix=".npz", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                bundle_schema_version=np.asarray(CONTEXT_DENSE_BUNDLE_VERSION),
                song_id=np.asarray(song.song_id),
                artifact_content_sha256=np.asarray(song.artifact_content_sha256),
                embedding_config_sha256=np.asarray(config["config_sha256"]),
                record_ids=np.asarray(_record_ids(song)),
                dense_text_sha256=np.asarray(_dense_text_hashes(song)),
                vectors=vectors,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _manifest_digest(manifest: Mapping) -> str:
    value = dict(manifest)
    value.pop("manifest_content_sha256", None)
    return _json_digest(value)


def _atomic_write_json(destination: Path, value: Mapping) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"refusing to overwrite manifest symlink: {destination}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".context-dense-manifest-", suffix=".tmp", dir=destination.parent
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


def _chunks(iterator: Iterator[tuple[ContextDenseSongInput, dict]], size: int):
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def embed_context_dense(
    *,
    context_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_KOE5_MODEL,
    model_revision: str | None = None,
    model_tag: str | None = None,
    expected_dim: int = DEFAULT_CONTEXT_DENSE_DIM,
    batch_size: int = DEFAULT_CONTEXT_DENSE_BATCH_SIZE,
    force: bool = False,
    dry_run: bool = False,
    require_complete_scope: bool = False,
    embedder: KoE5Embedder | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Validate, plan, embed and atomically publish a dense-cache manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    report = progress or (lambda _message: None)
    inputs = load_context_dense_input(
        context_dir, require_complete_scope=require_complete_scope
    )
    config = build_embedding_config(
        model_name=model_name,
        model_revision=model_revision,
        model_tag=model_tag,
        expected_dim=expected_dim,
    )
    root = Path(output_dir).resolve()

    cached: list[ContextDenseSongInput] = []
    pending: list[ContextDenseSongInput] = []
    empty: list[ContextDenseSongInput] = []
    cache_rejections: dict[str, str] = {}
    for song in inputs.songs:
        if not song.records:
            empty.append(song)
            continue
        path = _safe_output_path(root, _bundle_ref(song.song_id))
        if force:
            pending.append(song)
            continue
        if not path.exists() and not path.is_symlink():
            # A first run is normal work, not a rejected cache entry.  Keep
            # cache_rejection_count for files that existed but failed an
            # integrity/freshness check so operators can spot real damage.
            pending.append(song)
            continue
        try:
            _load_valid_bundle(path, song=song, config=config)
        except ValueError as exc:
            pending.append(song)
            cache_rejections[song.song_id] = str(exc)
        else:
            cached.append(song)

    summary = {
        "schema_version": CONTEXT_DENSE_MANIFEST_VERSION,
        "status": "dry_run" if dry_run else "ok",
        "context_dir": str(Path(context_dir).resolve()),
        "output_dir": str(root),
        "source_scope_complete": inputs.source_scope_complete,
        "source_song_count": len(inputs.songs),
        "source_record_count": inputs.record_count,
        "cached_song_count": len(cached),
        "pending_song_count": len(pending),
        "empty_song_count": len(empty),
        "records_to_embed": sum(len(song.records) for song in pending),
        "embedding_config": config,
    }
    report(
        f"validated songs={len(inputs.songs)} records={inputs.record_count} "
        f"scope_complete={inputs.source_scope_complete}"
    )
    report(
        f"plan cached={len(cached)} pending={len(pending)} empty={len(empty)} "
        f"records_to_embed={summary['records_to_embed']}"
    )
    if dry_run:
        return summary

    entries_by_song: dict[str, dict] = {}
    for song in empty:
        entries_by_song[song.song_id] = {
            "song_id": song.song_id,
            "artifact_ref": song.artifact_ref,
            "artifact_content_sha256": song.artifact_content_sha256,
            "record_count": 0,
            "vector_ref": None,
            "vector_sha256": None,
            "state": "no_records",
        }
    for song in cached:
        ref = _bundle_ref(song.song_id)
        path = _safe_output_path(root, ref)
        entries_by_song[song.song_id] = {
            "song_id": song.song_id,
            "artifact_ref": song.artifact_ref,
            "artifact_content_sha256": song.artifact_content_sha256,
            "record_count": len(song.records),
            "vector_ref": ref,
            "vector_sha256": _sha256_file(path),
            "state": "reused",
        }

    if pending:
        active_embedder = embedder or KoE5Embedder(
            model_name=model_name, revision=model_revision
        )
        flattened = (
            (song, record)
            for song in pending
            for record in song.records
        )
        buffers: dict[str, list[np.ndarray]] = {}
        embedded_records = 0
        total_pending_records = int(summary["records_to_embed"])
        last_reported_percent = -1
        for batch in _chunks(iter(flattened), batch_size):
            passages = [str(record["dense_text"]) for _, record in batch]
            vectors = active_embedder.embed_passages(
                passages,
                add_e5_prefix=True,
                normalize=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            vectors = _validate_vectors(
                np.asarray(vectors, dtype=np.float32),
                rows=len(batch),
                dim=int(config["dimension"]),
            )
            for (song, _record), vector in zip(batch, vectors):
                bucket = buffers.setdefault(song.song_id, [])
                bucket.append(vector.copy())
                if len(bucket) == len(song.records):
                    ref = _bundle_ref(song.song_id)
                    path = _safe_output_path(root, ref)
                    _atomic_write_bundle(
                        path,
                        song=song,
                        config=config,
                        vectors=np.stack(bucket).astype(np.float32),
                    )
                    entries_by_song[song.song_id] = {
                        "song_id": song.song_id,
                        "artifact_ref": song.artifact_ref,
                        "artifact_content_sha256": song.artifact_content_sha256,
                        "record_count": len(song.records),
                        "vector_ref": ref,
                        "vector_sha256": _sha256_file(path),
                        "state": "embedded",
                    }
                    del buffers[song.song_id]
            embedded_records += len(batch)
            percent = int(embedded_records * 100 / total_pending_records)
            if percent > last_reported_percent or embedded_records == total_pending_records:
                report(
                    f"embedded records={embedded_records}/{total_pending_records} "
                    f"({embedded_records * 100 / total_pending_records:.1f}%)"
                )
                last_reported_percent = percent
        if buffers:
            raise RuntimeError("context dense embedding ended with incomplete song buffers")

    # Re-read every current bundle before publishing the only manifest later
    # stages are allowed to consume.  A crash before here leaves the previous
    # manifest intact and the valid per-song bundles are reusable on retry.
    for song in inputs.songs:
        entry = entries_by_song.get(song.song_id)
        if entry is None:
            raise RuntimeError(f"missing dense output entry for song_id={song.song_id}")
        if song.records:
            path = _safe_output_path(root, str(entry["vector_ref"]))
            _load_valid_bundle(path, song=song, config=config)
            entry["vector_sha256"] = _sha256_file(path)

    ordered_entries = [entries_by_song[song.song_id] for song in inputs.songs]
    manifest = {
        "schema_version": CONTEXT_DENSE_MANIFEST_VERSION,
        "generated_at": now(),
        "complete_for_source_manifest": True,
        "source_scope_complete": inputs.source_scope_complete,
        "source_manifest_ref": str(Path(context_dir).resolve() / "manifest.json"),
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "source_input_sha256": inputs.source_input_sha256,
        "embedding": config,
        "song_count": len(inputs.songs),
        "embedded_song_count": sum(bool(song.records) for song in inputs.songs),
        "empty_song_count": len(empty),
        "record_count": inputs.record_count,
        "songs": ordered_entries,
    }
    manifest["manifest_content_sha256"] = _manifest_digest(manifest)
    _atomic_write_json(root / "manifest.json", manifest)

    summary.update({
        "embedded_song_count": len(pending),
        "reused_song_count": len(cached),
        "output_manifest": str(root / "manifest.json"),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "cache_rejection_count": len(cache_rejections),
    })
    return summary


def iter_context_dense_embeddings(
    *,
    context_dir: Path,
    dense_dir: Path,
) -> Iterator[dict]:
    """Yield validated fact metadata plus vectors for a future DB upsert."""
    inputs = load_context_dense_input(context_dir)
    root = Path(dense_dir).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("context dense manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CONTEXT_DENSE_MANIFEST_VERSION:
        raise ValueError("unsupported context dense manifest schema")
    if manifest.get("manifest_content_sha256") != _manifest_digest(manifest):
        raise ValueError("context dense manifest content hash mismatch")
    if manifest.get("complete_for_source_manifest") is not True:
        raise ValueError("context dense manifest is incomplete")
    if manifest.get("source_input_sha256") != inputs.source_input_sha256:
        raise ValueError("context artifacts changed after dense embedding")
    config = manifest.get("embedding")
    if not isinstance(config, Mapping) or config.get("config_sha256") != _json_digest({
        key: value for key, value in config.items() if key != "config_sha256"
    }):
        raise ValueError("context dense embedding config hash mismatch")
    entries = manifest.get("songs")
    if not isinstance(entries, list) or len(entries) != len(inputs.songs):
        raise ValueError("context dense manifest song count mismatch")
    entry_by_song = {str(entry.get("song_id")): entry for entry in entries}
    if len(entry_by_song) != len(entries):
        raise ValueError("context dense manifest contains duplicate song_id values")

    emitted = 0
    for song in inputs.songs:
        entry = entry_by_song.get(song.song_id)
        if entry is None or int(entry.get("record_count", -1)) != len(song.records):
            raise ValueError(f"context dense entry mismatch for song_id={song.song_id}")
        if not song.records:
            continue
        ref = entry.get("vector_ref")
        if not isinstance(ref, str):
            raise ValueError(f"context dense vector_ref missing for song_id={song.song_id}")
        path = _safe_output_path(root, ref)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"context dense vector file is missing or unsafe for song_id={song.song_id}"
            )
        if _sha256_file(path) != entry.get("vector_sha256"):
            raise ValueError(f"context dense vector file hash mismatch for song_id={song.song_id}")
        vectors = _load_valid_bundle(path, song=song, config=config)
        for record, vector in zip(song.records, vectors):
            emitted += 1
            yield {**record, "values": vector.copy()}
    if emitted != int(manifest.get("record_count", -1)):
        raise ValueError("context dense manifest record_count differs from emitted vectors")
