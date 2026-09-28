"""Atomic storage and loading for hybrid song-context artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from pydantic import ValidationError

from src.crawler.context.schemas import NAMUWIKI_META_VERSION, now
from src.embedding.text.namuwiki_passage import (
    CONTEXT_ARTIFACT_VERSION,
    CONTEXT_RECORD_VERSION,
    DENSE_TEXT_FORMAT,
    ContextSongArtifact,
    build_namuwiki_song_artifact,
    context_source_digest,
    embedding_record_projection,
    sparse_profile_projection,
)

CONTEXT_MANIFEST_VERSION = "context_manifest_v6"
CONTEXT_PROGRESS_VERSION = "context_progress_v1"
CONTEXT_UNRESOLVED_VERSION = "context_unresolved_v1"


@dataclass(frozen=True)
class ContextArtifactState:
    applicable: bool
    ready: bool
    reason: str
    artifact_ref: str | None = None
    record_count: int = 0
    source_fact_count: int = 0
    covered_source_fact_count: int = 0
    sparse_term_count: int = 0
    source_digest: str | None = None
    status: str | None = None

    def as_dict(self) -> dict:
        return {
            "applicable": self.applicable,
            "ready": self.ready,
            "reason": self.reason,
            "artifact_ref": self.artifact_ref,
            "record_count": self.record_count,
            "source_fact_count": self.source_fact_count,
            "covered_source_fact_count": self.covered_source_fact_count,
            "sparse_term_count": self.sparse_term_count,
            "source_digest": self.source_digest,
            "status": self.status,
        }


class ContextArtifactStore:
    """One writer, deterministic song files, and an embedding manifest."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def path(self, ref: str) -> Path:
        relative = Path(ref)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("invalid context artifact reference")
        candidate = self.root / relative
        resolved = candidate.resolve()
        if resolved == self.root or not resolved.is_relative_to(self.root):
            raise ValueError("context artifact reference escapes storage root")
        # Return the lexical final path so callers can still detect and reject
        # a final-component symlink. Parent symlinks were constrained above by
        # the resolved containment check.
        return candidate

    @staticmethod
    def song_ref(song_id: str) -> str:
        value = str(song_id).strip()
        if not value.isdigit():
            raise ValueError("numeric song_id is required for an artifact path")
        return f"songs/{value}.json"

    def read(self, ref: str):
        path = self.path(ref)
        if path.is_symlink():
            raise ValueError(f"refusing to read artifact symlink: {ref}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _payload(self, value: object) -> bytes:
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    def _write_atomic(self, ref: str, value: object) -> str:
        destination = self.path(ref)
        if destination.is_symlink():
            raise ValueError(f"refusing to overwrite artifact symlink: {ref}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload(value)
        existed = destination.exists()
        if existed and destination.read_bytes() == payload:
            return "unchanged"

        descriptor, temp_path = tempfile.mkstemp(
            prefix=".context-",
            suffix=".tmp",
            dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, destination)
            temp_path = ""
            return "updated" if existed else "created"
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    @contextmanager
    def writer(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.path("writer.lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(
                "context artifact writer.lock exists; another writer or an interrupted run must be inspected"
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "started_at": now()}, stream)
            yield
        finally:
            lock.unlink(missing_ok=True)

    def inspect(self, song: Mapping) -> ContextArtifactState:
        """Check freshness without running Kiwi unless a rewrite is required."""
        try:
            source_digest = context_source_digest(song)
            song_id = str(song.get("id") or song.get("song_id") or "").strip()
            ref = self.song_ref(song_id)
            context = song.get("namuwiki")
            status = context.get("status") if isinstance(context, Mapping) else None
        except (TypeError, ValueError, ValidationError) as exc:
            return ContextArtifactState(
                applicable=False,
                ready=False,
                reason=f"not_terminal_or_invalid_source:{type(exc).__name__}",
            )

        destination = self.path(ref)
        if not destination.exists():
            return ContextArtifactState(
                applicable=True,
                ready=False,
                reason="artifact_missing",
                artifact_ref=ref,
                source_digest=source_digest,
                status=status,
            )
        if destination.is_symlink():
            return ContextArtifactState(
                applicable=True,
                ready=False,
                reason="artifact_symlink_rejected",
                artifact_ref=ref,
                source_digest=source_digest,
                status=status,
            )
        try:
            artifact = ContextSongArtifact.model_validate(self.read(ref))
        except (OSError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            return ContextArtifactState(
                applicable=True,
                ready=False,
                reason=f"artifact_invalid:{type(exc).__name__}",
                artifact_ref=ref,
                source_digest=source_digest,
                status=status,
            )
        if artifact.source.digest != source_digest:
            return ContextArtifactState(
                applicable=True,
                ready=False,
                reason="artifact_stale",
                artifact_ref=ref,
                record_count=artifact.retrieval.record_count,
                source_fact_count=artifact.retrieval.source_fact_count,
                covered_source_fact_count=artifact.retrieval.covered_source_fact_count,
                sparse_term_count=(
                    artifact.retrieval.sparse_profile.term_count
                    if artifact.retrieval.sparse_profile
                    else 0
                ),
                source_digest=source_digest,
                status=status,
            )
        return ContextArtifactState(
            applicable=True,
            ready=True,
            reason="artifact_current",
            artifact_ref=ref,
            record_count=artifact.retrieval.record_count,
            source_fact_count=artifact.retrieval.source_fact_count,
            covered_source_fact_count=artifact.retrieval.covered_source_fact_count,
            sparse_term_count=(
                artifact.retrieval.sparse_profile.term_count
                if artifact.retrieval.sparse_profile
                else 0
            ),
            source_digest=source_digest,
            status=artifact.status,
        )

    def sync_song(self, song: Mapping) -> dict:
        artifact = build_namuwiki_song_artifact(song)
        ref = self.song_ref(artifact["song"]["song_id"])
        existed = self.path(ref).exists()
        write_result = self._write_atomic(ref, artifact)
        if write_result != "unchanged":
            write_result = "updated" if existed else "created"
        return {
            "artifact_ref": ref,
            "artifact_action": write_result,
            "record_count": artifact["retrieval"]["record_count"],
            "source_fact_count": artifact["retrieval"]["source_fact_count"],
            "covered_source_fact_count": artifact["retrieval"]["covered_source_fact_count"],
            "sparse_term_count": (
                artifact["retrieval"].get("sparse_profile") or {}
            ).get("term_count", 0),
            "source_digest": artifact["source"]["digest"],
            "status": artifact["status"],
        }

    def write_progress(self, value: Mapping) -> str:
        payload = {
            "schema_version": CONTEXT_PROGRESS_VERSION,
            **dict(value),
            "updated_at": now(),
        }
        self._write_atomic("progress.json", payload)
        return "progress.json"

    def write_unresolved(self, value: Mapping) -> str:
        """Publish the current non-ready song queue atomically.

        This is deliberately separate from ``manifest.json``: the manifest is
        trusted embedding input, while unresolved songs must never be consumed
        by an embedder until collection or manual binding review is complete.
        """
        payload = {
            "schema_version": CONTEXT_UNRESOLVED_VERSION,
            **dict(value),
            "updated_at": now(),
        }
        self._write_atomic("unresolved.json", payload)
        return "unresolved.json"

    def publish_manifest(
        self,
        *,
        coverage: Mapping | None = None,
        active_song_ids: set[str] | None = None,
    ) -> dict:
        entries, invalid, orphan = [], [], []
        active = (
            {str(song_id) for song_id in active_song_ids}
            if active_song_ids is not None
            else None
        )
        songs_dir = self.path("songs")
        if songs_dir.is_symlink():
            invalid.append({"artifact_ref": "songs", "reason": "symlink_rejected"})
        elif songs_dir.exists():
            for path in sorted(songs_dir.glob("*.json")):
                ref = str(path.relative_to(self.root)).replace("\\", "/")
                path_song_id = path.stem
                # A targeted backfill must not delete unrelated artifacts, and
                # a removed/core-invalid raw song must not poison the active
                # embedding manifest.  Preserve such files as recoverable
                # orphans without trusting or parsing their contents.
                if active is not None and path_song_id not in active:
                    orphan.append(
                        {
                            "song_id": path_song_id if path_song_id.isdigit() else None,
                            "artifact_ref": ref,
                            "reason": "not_an_active_core_valid_raw_song",
                        }
                    )
                    continue
                if path.is_symlink():
                    invalid.append({"artifact_ref": ref, "reason": "symlink_rejected"})
                    continue
                try:
                    artifact = ContextSongArtifact.model_validate(
                        json.loads(path.read_text(encoding="utf-8"))
                    )
                except (OSError, ValueError, json.JSONDecodeError, ValidationError) as exc:
                    invalid.append(
                        {
                            "artifact_ref": ref,
                            "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                        }
                    )
                    continue
                if artifact.song.song_id != path_song_id:
                    invalid.append(
                        {
                            "artifact_ref": ref,
                            "reason": "artifact_song_id_does_not_match_filename",
                        }
                    )
                    continue
                entries.append(
                    {
                        "song_id": artifact.song.song_id,
                        "title": artifact.song.title,
                        "artists": artifact.song.artists,
                        "status": artifact.status,
                        "record_count": artifact.retrieval.record_count,
                        "source_fact_count": artifact.retrieval.source_fact_count,
                        "covered_source_fact_count": artifact.retrieval.covered_source_fact_count,
                        "sparse_term_count": (
                            artifact.retrieval.sparse_profile.term_count
                            if artifact.retrieval.sparse_profile
                            else 0
                        ),
                        "artifact_ref": ref,
                        "source_digest": artifact.source.digest,
                        "artifact_content_sha256": artifact.artifact_content_sha256,
                    }
                )

        entries.sort(key=lambda item: (int(item["song_id"]), item["artifact_ref"]))
        orphan.sort(
            key=lambda item: (
                int(item["song_id"]) if str(item.get("song_id", "")).isdigit() else -1,
                item["artifact_ref"],
            )
        )
        status_counts: dict[str, int] = {}
        for entry in entries:
            status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1
        manifest = {
            "schema_version": CONTEXT_MANIFEST_VERSION,
            "artifact_schema_version": CONTEXT_ARTIFACT_VERSION,
            "record_schema_version": CONTEXT_RECORD_VERSION,
            "source_schema_version": NAMUWIKI_META_VERSION,
            "retrieval_policy": {
                "dense_embedding_unit": "fact",
                "dense_text_format": DENSE_TEXT_FORMAT,
                "sparse_embedding_unit": "song",
                "dense_input": "records[].dense_text",
                "sparse_input": "retrieval.sparse_profile.terms",
                "dense_group_by": "song_id",
                "dense_score_aggregation": "max_per_song",
                "hybrid_fusion": "rrf",
            },
            "generated_at": now(),
            "song_count": len(entries),
            "record_count": sum(entry["record_count"] for entry in entries),
            "sparse_profile_count": sum(
                1 for entry in entries if entry["sparse_term_count"] > 0
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "invalid_artifact_count": len(invalid),
            "artifacts_valid": not invalid,
            "coverage": dict(coverage or {}),
            "songs": entries,
            "invalid_artifacts": invalid,
            # Orphans are preserved on disk for recovery but are deliberately
            # excluded from the embedding input manifest.
            "orphan_artifact_count": len(orphan),
            "orphan_artifacts": orphan,
        }
        self._write_atomic("manifest.json", manifest)
        return manifest


def iter_context_artifact_records(root: Path) -> Iterator[dict]:
    """Validated fact-level dense inputs for the context embedding pipeline."""
    store = ContextArtifactStore(root)
    manifest = store.read("manifest.json")
    if manifest.get("schema_version") != CONTEXT_MANIFEST_VERSION:
        raise ValueError("unsupported or missing context manifest schema")
    if manifest.get("record_schema_version") != CONTEXT_RECORD_VERSION:
        raise ValueError("context record schema version mismatch")
    if manifest.get("invalid_artifact_count"):
        raise ValueError("context manifest contains invalid artifacts")

    expected_total = int(manifest.get("record_count", -1))
    emitted = 0
    for entry in manifest.get("songs", []):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("artifact_ref"), str):
            raise ValueError("invalid context manifest song entry")
        artifact = ContextSongArtifact.model_validate(store.read(entry["artifact_ref"]))
        if (
            artifact.song.song_id != str(entry.get("song_id"))
            or artifact.source.digest != entry.get("source_digest")
            or (
                artifact.artifact_content_sha256
                != entry.get("artifact_content_sha256")
            )
            or artifact.retrieval.record_count != entry.get("record_count")
            or artifact.retrieval.source_fact_count != entry.get("source_fact_count")
            or (
                artifact.retrieval.covered_source_fact_count
                != entry.get("covered_source_fact_count")
            )
            or (
                (
                    artifact.retrieval.sparse_profile.term_count
                    if artifact.retrieval.sparse_profile
                    else 0
                )
                != entry.get("sparse_term_count")
            )
        ):
            raise ValueError("context manifest and song artifact differ")
        for record in artifact.retrieval.records:
            emitted += 1
            yield embedding_record_projection(artifact, record)
    if emitted != expected_total:
        raise ValueError("context manifest record_count differs from emitted records")


def iter_context_sparse_profiles(root: Path) -> Iterator[dict]:
    """Validated one-per-song sparse inputs for the context BM25 corpus."""
    store = ContextArtifactStore(root)
    manifest = store.read("manifest.json")
    if manifest.get("schema_version") != CONTEXT_MANIFEST_VERSION:
        raise ValueError("unsupported or missing context manifest schema")
    if manifest.get("record_schema_version") != CONTEXT_RECORD_VERSION:
        raise ValueError("context record schema version mismatch")
    if manifest.get("invalid_artifact_count"):
        raise ValueError("context manifest contains invalid artifacts")

    expected_total = int(manifest.get("sparse_profile_count", -1))
    emitted = 0
    for entry in manifest.get("songs", []):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("artifact_ref"), str):
            raise ValueError("invalid context manifest song entry")
        artifact = ContextSongArtifact.model_validate(store.read(entry["artifact_ref"]))
        profile = sparse_profile_projection(artifact)
        if profile is None:
            if entry.get("sparse_term_count") != 0:
                raise ValueError("manifest expects a missing sparse profile")
            continue
        if (
            artifact.song.song_id != str(entry.get("song_id"))
            or artifact.source.digest != entry.get("source_digest")
            or artifact.artifact_content_sha256 != entry.get("artifact_content_sha256")
            or len(profile["sparse_terms"]) != entry.get("sparse_term_count")
        ):
            raise ValueError("context manifest and sparse profile differ")
        emitted += 1
        yield profile
    if emitted != expected_total:
        raise ValueError("context manifest sparse_profile_count differs from emitted profiles")
