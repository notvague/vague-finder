"""Single-writer, content-addressed snapshots and atomic manifest publication."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .schemas import SourceSnapshot, digest, now


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, ref: str) -> Path:
        result = (self.root / ref).resolve()
        if result == self.root or not result.is_relative_to(self.root):
            raise ValueError("context reference escapes storage root")
        return result

    def read(self, ref: str):
        return json.loads(self.path(ref).read_text(encoding="utf-8"))

    def write_bytes(self, ref: str, payload: bytes) -> None:
        destination = self.path(ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".pending-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, destination)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def write(self, ref: str, value: object) -> None:
        self.write_bytes(ref, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    def immutable(self, ref: str, value: object) -> None:
        if self.path(ref).exists():
            if self.read(ref) != value:
                raise ValueError(f"immutable snapshot conflict: {ref}")
            return
        self.write(ref, value)

    @contextmanager
    def writer(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.path("writer.lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("writer.lock exists: another writer or interrupted run; inspect before removing") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "started_at": now()}, stream)
            yield
        finally:
            lock.unlink()

    def save_snapshot(self, raw: bytes, *, requested_url: str, acquisition_method: str,
                      final_url=None, http_status=None, redirects=None, headers=None,
                      parent_snapshot_ref=None, publish_cache=True) -> tuple[str, SourceSnapshot]:
        raw_hash = hashlib.sha256(raw).hexdigest()
        document_id = digest(final_url or requested_url)[:24]
        # Acquisition method matters: saved HTML must never masquerade as an HTTP success.
        snapshot_id = digest([document_id, requested_url, raw_hash, acquisition_method])
        prefix = f"sources/{document_id}/{snapshot_id}"
        ref = f"{prefix}/snapshot.json"
        if self.path(ref).exists():
            snapshot = SourceSnapshot.model_validate(self.read(ref))
            self.raw(snapshot)  # Refuse corrupt cache; do not silently bless it.
            if publish_cache:
                self.write(f"url_cache/{digest(requested_url)}.json", {"snapshot_ref": ref})
            return ref, snapshot
        headers = headers or {}
        raw_ref = f"{prefix}/page.html.gz"
        self.write_bytes(raw_ref, gzip.compress(raw, mtime=0))
        snapshot = SourceSnapshot(
            snapshot_id=snapshot_id, document_id=document_id, requested_url=requested_url,
            final_url=final_url, acquisition_method=acquisition_method, fetched_at=now(),
            http_status=http_status, raw_sha256=raw_hash, raw_html_ref=raw_ref,
            redirects=redirects or [], content_type=headers.get("Content-Type"),
            etag=headers.get("ETag"), last_modified=headers.get("Last-Modified"),
            parent_snapshot_ref=parent_snapshot_ref,
        )
        self.immutable(ref, snapshot.model_dump())
        if publish_cache:
            self.write(f"url_cache/{digest(requested_url)}.json", {"snapshot_ref": ref})
        return ref, snapshot

    def raw(self, snapshot: SourceSnapshot) -> bytes:
        with gzip.open(self.path(snapshot.raw_html_ref), "rb") as stream:
            raw = stream.read(12_000_001)
        if len(raw) > 12_000_000 or hashlib.sha256(raw).hexdigest() != snapshot.raw_sha256:
            raise ValueError("raw snapshot size/hash mismatch")
        return raw

    def cached(self, url: str, *, http_only: bool = False) -> str | None:
        index = self.path(f"url_cache/{digest(url)}.json")
        if not index.exists():
            return None
        ref = self.read(str(index.relative_to(self.root)))["snapshot_ref"]
        snapshot = SourceSnapshot.model_validate(self.read(ref))
        if http_only and snapshot.acquisition_method not in {"http", "browser_rendered"}:
            return None
        self.raw(snapshot)
        return ref

    def refs(self, folder: str) -> list[str]:
        return [str(p.relative_to(self.root)).replace("\\", "/")
                for p in sorted(self.path(folder).glob("*.json"))]
