"""Correlate a real HTTP request with its own retrieval execution spans.

No model, Qdrant or backend imports. A Top-10 contribution and successful
embedding/index execution are separate observations in a mixed query.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time


def timing_path(project: Path, registration: dict) -> Path:
    raw = os.getenv("SEARCH_TIMING_LOG", "").strip()
    if not raw:
        raw = str(registration.get("environment", {}).get("SEARCH_TIMING_LOG") or "").strip()
    if not raw:
        raise ValueError("SEARCH_TIMING_LOG is absent; enable the server timing log before API verification")
    path = Path(raw)
    target = path if path.is_absolute() else project / path
    resolved = target.resolve()
    base = (project / "artifacts").resolve()
    if base not in resolved.parents or target.is_symlink() or resolved.suffix != ".jsonl":
        raise ValueError("SEARCH_TIMING_LOG must be a JSONL file inside project artifacts")
    return resolved


def read_request_timing(path: Path, offset: int, request_id: str, query: str,
                        *, wait_seconds: float = 3) -> dict:
    """Read only new complete lines, matched by response X-Request-Id and query."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("Search response is missing X-Request-Id; timing cannot be correlated")
    deadline = time.monotonic() + wait_seconds
    while True:
        if path.is_file():
            if path.stat().st_size < offset:
                raise ValueError("Server timing log was rotated during API verification")
            with path.open("rb") as stream:
                stream.seek(offset)
                lines = stream.read().splitlines(keepends=True)
            matched = []
            for line in lines:
                # A concurrent writer may have started, but not finished, its line.
                if not line.endswith(b"\n") or not line.strip():
                    continue
                value = json.loads(line.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("Server timing record is not an object")
                if value.get("request_id") == request_id:
                    matched.append(value)
            if len(matched) > 1:
                raise ValueError("Duplicate timing records for one HTTP request ID")
            if matched:
                record = matched[0]
                if record.get("query") != query or record.get("attrs", {}).get("status") != 200:
                    raise ValueError("Timing record does not describe this successful source query")
                return record
        if time.monotonic() >= deadline:
            raise ValueError("No new timing record for X-Request-Id; check SEARCH_TIMING_LOG on the running server")
        time.sleep(0.05)


def verify_media_execution(analysis: dict, record: dict, name: str) -> dict:
    """Require a positive dedicated modality and embed/query descendants of its job."""
    if name not in {"image", "audio"}:
        raise ValueError("Media execution check supports image and audio only")
    prompt = analysis.get(name + "_english_query")
    raw_weight = analysis.get("modality_weights", {}).get(name)
    if (not isinstance(prompt, str) or not prompt.strip()
            or isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float))
            or not math.isfinite(raw_weight) or raw_weight <= 0):
        raise ValueError(f"New API analysis has no positive dedicated {name} query")
    spans = record.get("spans")
    if not isinstance(spans, list) or not all(isinstance(span, dict) for span in spans):
        raise ValueError("Timing spans are absent or malformed")
    by_id = {}
    for span in spans:
        sid = span.get("id")
        if isinstance(sid, bool) or not isinstance(sid, int) or sid <= 0 or sid in by_id:
            raise ValueError("Timing span IDs must be unique positive integers")
        by_id[sid] = span
    for sid, span in by_id.items():
        parent = span.get("parent")
        if (isinstance(parent, bool) or not isinstance(parent, int) or parent < 0
                or parent > 0 and parent not in by_id):
            raise ValueError("Timing spans must reference a recorded parent or root zero")
        seen, cursor = {sid}, parent
        while cursor > 0:
            if cursor in seen:
                raise ValueError("Timing span ancestry contains a cycle")
            seen.add(cursor)
            cursor = by_id[cursor].get("parent")
            if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
                raise ValueError("Timing spans have an invalid parent")
            if cursor > 0 and cursor not in by_id:
                raise ValueError("Timing spans have an unrecorded parent")
    branches = [span for span in spans if span.get("name") == "path." + name]
    if not branches:
        raise ValueError(f"{name} path did not execute in this HTTP request")

    def is_descendant(child: dict, ancestor: int) -> bool:
        parent, seen = child.get("parent"), set()
        while isinstance(parent, int) and parent > 0 and parent not in seen:
            if parent == ancestor:
                return True
            seen.add(parent)
            parent = by_id.get(parent, {}).get("parent")
        return False

    relevant = [span for span in spans
                if span in branches or span.get("name") in {name + ".embed", name + ".query"}]
    for span in relevant:
        if span.get("error"):
            raise ValueError(f"{name} execution failed: {span['name']}: {span['error']}")
        for field in ("run_ms", "wait_ms"):
            value = span.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} span has an invalid {field}")
    # Both stages must belong to the same actual job, not an unrelated job
    # with similarly named spans elsewhere in this request.
    proof = next((branch for branch in branches if all(
        any(span.get("name") == name + suffix and is_descendant(span, branch["id"])
            for span in spans) for suffix in (".embed", ".query"))), None)
    if proof is None:
        raise ValueError(f"{name} path lacks embedding and index-query stages under the same job")
    children = [span for span in relevant if span is not proof and is_descendant(span, proof["id"])]
    return dict(verified=True, weight=float(raw_weight), job_id=proof["id"],
                spans=[{key: span[key] for key in ("id", "name", "parent", "run_ms", "wait_ms")}
                       for span in [proof, *children]])
