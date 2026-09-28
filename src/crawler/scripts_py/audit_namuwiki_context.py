"""Offline quality audit for Namuwiki metadata and context artifacts.

This command never fetches or mutates ``data/raw``.  It validates every final
song state, correlates the latest crawl binding evidence, checks artifact
freshness, and emits a small risk-based manual-review queue.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from src.crawler.context.document_parser import compact
from src.crawler.context.schemas import NAMUWIKI_META_VERSION, NamuwikiMeta, now
from src.embedding.context_artifacts import (
    ContextArtifactStore,
    iter_context_sparse_profiles,
)

ROOT = Path(__file__).resolve().parents[3]
AUDIT_SCHEMA_VERSION = "namuwiki_context_audit_v1"
TERMINAL = {"ok", "no_trivia", "not_found"}
SEVERITY = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _default_artifact_dir(data_dir: Path) -> Path:
    data_dir = Path(data_dir).resolve()
    project = (
        data_dir.parent.parent
        if data_dir.name == "raw" and data_dir.parent.name == "data"
        else data_dir.parent
    )
    return (project / "artifacts/context").resolve()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".namuwiki-audit-",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    _atomic_bytes(path, payload.encode("utf-8"))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".namuwiki-audit-",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _song_id(meta: dict, folder: Path) -> str:
    value = str(meta.get("song_id") or meta.get("id") or "").strip()
    if not value:
        suffix = folder.name.rsplit("_", 1)[-1]
        value = suffix if suffix.isdigit() else ""
    if not value.isdigit():
        raise ValueError("numeric song_id is required")
    return value


def _artists(meta: dict) -> list[str]:
    metadata = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    values = metadata.get("artist") or []
    values = [values] if isinstance(values, str) else values
    return [str(value).strip() for value in values if str(value).strip()]


def _artist_variants(artists: list[str]) -> list[str]:
    values: list[str] = []
    for artist in artists:
        values.append(artist)
        base = re.sub(r"\([^)]*\)", "", artist).strip()
        if base:
            values.append(base)
        values.extend(
            part.strip()
            for part in re.findall(r"\(([^)]*)\)", artist)
            if part.strip()
        )
        values.extend(
            part.strip() for part in re.split(r"[/·|]", artist) if part.strip()
        )
    return list(dict.fromkeys(values))


def _youtube_views(meta: dict) -> int:
    try:
        value = meta["community_feedback"]["popularity"]["youtube_view_count"]
        return max(0, int(value or 0))
    except (KeyError, TypeError, ValueError):
        return 0


def _latest_run_audits(cache_dir: Path) -> tuple[dict[str, dict], list[dict]]:
    latest: dict[str, dict] = {}
    errors: list[dict] = []
    runs = cache_dir / "raw_runs"
    if not runs.is_dir():
        return latest, errors
    for path in sorted(runs.glob("*.json"), key=lambda item: (item.stat().st_mtime_ns, item.name)):
        if path.is_symlink():
            errors.append({"path": str(path), "reason": "symlink_rejected"})
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "reason": type(exc).__name__})
            continue
        for audit in report.get("audits", []):
            if not isinstance(audit, dict):
                continue
            song_id = str(audit.get("song_id") or "")
            if song_id.isdigit():
                latest[song_id] = {**audit, "run_report": str(path)}
    return latest, errors


def _binding_for_meta(audit: dict | None, source_url: str | None) -> dict | None:
    if not isinstance(audit, dict):
        return None
    attempts = [row for row in audit.get("attempts", []) if isinstance(row, dict)]
    if source_url:
        matching = [
            row for row in attempts
            if row.get("source_url") == source_url
            or row.get("url") == source_url
            or row.get("fetch", {}).get("final_url") == source_url
        ]
        if matching:
            attempts = matching
    for attempt in reversed(attempts):
        identity = attempt.get("identity")
        if isinstance(identity, dict):
            return identity
    return None


def _binding_contract(binding: dict | None) -> str:
    if not isinstance(binding, dict):
        return "missing"
    required = {
        "verified",
        "reason",
        "confidence",
        "scope_section_id",
        "evidence",
        "candidate_section_ids",
    }
    return "current" if required.issubset(binding) else "legacy"


def _source_slug(url: str | None) -> str:
    if not url:
        return ""
    try:
        path = unquote(urlsplit(url).path)
    except ValueError:
        return ""
    return path[3:] if path.startswith("/w/") else ""


def _source_identity_shape(title: str, artists: list[str], source_url: str | None) -> str:
    slug = compact(_source_slug(source_url))
    title_key = compact(title)
    artist_keys = [
        compact(value) for value in _artist_variants(artists) if compact(value)
    ]
    if not slug:
        return "missing"
    if any(slug == title_key + artist for artist in artist_keys):
        return "title_artist"
    if slug == title_key + compact("노래"):
        return "generic_song"
    if slug == title_key:
        return "title_only"
    return "different_title"


def _stable_key(row: dict) -> tuple:
    digest = hashlib.sha256(str(row.get("song_id", "")).encode()).hexdigest()
    return (-int(row.get("youtube_view_count") or 0), digest)


def _add_issue(row: dict, severity: str, code: str) -> None:
    row["issues"].append(code)
    if SEVERITY[severity] > SEVERITY[row["risk_level"]]:
        row["risk_level"] = severity


def _validate_manifest(artifact_dir: Path) -> dict:
    """Exercise the same hash/coverage contract used before dense embedding."""
    try:
        # Import lazily so the crawler's audit --help stays lightweight and
        # cannot load the embedding model.  load_context_dense_input itself
        # validates schemas, manifest bindings, self-hashes, record counts and
        # lossless source-fact coverage without instantiating KoE5.
        from src.embedding.context_dense import load_context_dense_input

        dense = load_context_dense_input(
            artifact_dir,
            require_complete_scope=False,
        )
        sparse_count = sum(1 for _ in iter_context_sparse_profiles(artifact_dir))
        expected_sparse = int(dense.manifest.get("sparse_profile_count", -1))
        if sparse_count != expected_sparse:
            raise ValueError("validated sparse profile count differs from manifest")
        coverage = dense.manifest.get("coverage") or {}
        return {
            "status": "ok",
            "scope_complete": dense.source_scope_complete,
            "song_count": len(dense.songs),
            "record_count": dense.record_count,
            "sparse_profile_count": sparse_count,
            "pending": int(coverage.get("pending", 0)),
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def _audit_song(
    *,
    path: Path,
    meta: dict,
    artifacts: ContextArtifactStore,
    latest_audit: dict | None,
) -> dict:
    metadata = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    song_id = _song_id(meta, path.parent)
    title = str(metadata.get("title") or "").strip()
    artists = _artists(meta)
    context = meta.get("namuwiki") if isinstance(meta.get("namuwiki"), dict) else {}
    status = str(context.get("status") or "missing_namuwiki")
    facts = context.get("facts") if isinstance(context.get("facts"), list) else []
    source_url = context.get("source_url") if isinstance(context.get("source_url"), str) else None
    error_code = context.get("error_code") if isinstance(context.get("error_code"), str) else None
    binding = _binding_for_meta(latest_audit, source_url)
    binding_contract = _binding_contract(binding)
    artifact = artifacts.inspect(meta)
    artifact_ref = artifacts.song_ref(song_id)
    artifact_path = artifacts.path(artifact_ref)
    artifact_file_exists = artifact_path.exists() or artifact_path.is_symlink()

    row = {
        "song_id": song_id,
        "title": title,
        "artists": artists,
        "album": metadata.get("album"),
        "release_date": metadata.get("release_date"),
        "youtube_view_count": _youtube_views(meta),
        "status": status,
        "error_code": error_code,
        "source_url": source_url,
        "source_identity_shape": _source_identity_shape(title, artists, source_url),
        "fact_count": len(facts),
        "artifact_ready": artifact.ready,
        "artifact_ref": artifact_ref,
        "artifact_file_exists": artifact_file_exists,
        "artifact_reason": artifact.reason,
        "artifact_record_count": artifact.record_count,
        "artifact_source_fact_count": artifact.source_fact_count,
        "binding": binding,
        "binding_contract": binding_contract,
        "run_report": latest_audit.get("run_report") if latest_audit else None,
        "meta_path": str(path),
        "risk_level": "none",
        "issues": [],
    }

    if not context:
        _add_issue(row, "critical", "namuwiki_metadata_missing")
        return row
    try:
        NamuwikiMeta.model_validate(context)
    except (ValueError, TypeError, ValidationError):
        _add_issue(row, "critical", "namuwiki_schema_invalid")
    if context.get("schema_version") != NAMUWIKI_META_VERSION:
        _add_issue(row, "high", "namuwiki_schema_not_current")

    if status in TERMINAL:
        if not artifact.ready:
            _add_issue(row, "critical", "terminal_artifact_not_current")
        if status == "ok":
            if not facts:
                _add_issue(row, "critical", "ok_without_facts")
            if artifact.ready and artifact.source_fact_count != len(facts):
                _add_issue(row, "high", "artifact_source_fact_count_mismatch")
        elif facts:
            _add_issue(row, "critical", "non_ok_status_has_facts")
        if status in {"ok", "no_trivia"} and not source_url:
            _add_issue(row, "critical", "resolved_status_without_source_url")
    elif status == "needs_review":
        if artifact_file_exists:
            _add_issue(row, "critical", "unresolved_status_has_artifact_file")
    elif status == "error":
        _add_issue(row, "high", "collection_error")
        if artifact_file_exists:
            _add_issue(row, "critical", "unresolved_status_has_artifact_file")
    else:
        _add_issue(row, "critical", "unknown_namuwiki_status")

    if source_url:
        try:
            parsed = urlsplit(source_url)
            valid_source = (
                parsed.scheme == "https"
                and parsed.netloc.lower() == "namu.wiki"
                and parsed.path.startswith("/w/")
            )
        except ValueError:
            valid_source = False
        if not valid_source:
            _add_issue(row, "critical", "invalid_source_url")

    if status in {"ok", "no_trivia"}:
        if binding is not None and not binding.get("verified", False):
            _add_issue(row, "high", "latest_binding_not_verified")
        scope = binding.get("scope_section_id") if isinstance(binding, dict) else None
        if (
            row["source_identity_shape"] in {"title_only", "generic_song"}
            and binding_contract != "current"
        ):
            _add_issue(row, "high", "identity_binding_requires_current_recheck")
        if (
            row["source_identity_shape"] == "different_title"
            and not scope
            and binding_contract != "current"
        ):
            _add_issue(row, "high", "different_title_source_without_verified_scope")
        elif binding is None:
            # Old reports can be absent after a cache move.  This is an audit
            # limitation, not evidence that a previously accepted page is bad.
            _add_issue(row, "low", "binding_audit_missing")

    return row


def run_audit(args) -> dict:
    data_dir = Path(args.data_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    artifact_dir = (
        Path(args.artifact_dir).resolve()
        if args.artifact_dir
        else _default_artifact_dir(data_dir)
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (artifact_dir / "audit").resolve()
    )
    if not data_dir.is_dir():
        raise ValueError(f"raw directory not found: {data_dir}")
    if output_dir == data_dir or output_dir.is_relative_to(data_dir):
        raise ValueError("audit output must be outside raw")
    if output_dir.is_symlink():
        raise ValueError("audit output directory cannot be a symlink")

    latest_audits, run_report_errors = _latest_run_audits(cache_dir)
    artifacts = ContextArtifactStore(artifact_dir)
    manifest_validation = _validate_manifest(artifact_dir)
    rows: list[dict] = []
    catalogue_errors: list[dict] = []
    seen: set[str] = set()

    for path in sorted(data_dir.glob("*/meta.json")):
        if path.is_symlink() or path.parent.is_symlink():
            catalogue_errors.append({"meta_path": str(path), "reason": "symlink_rejected"})
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(meta, dict):
                raise ValueError("meta must be an object")
            song_id = _song_id(meta, path.parent)
            if song_id in seen:
                raise ValueError("duplicate song_id")
            seen.add(song_id)
            rows.append(_audit_song(
                path=path,
                meta=meta,
                artifacts=artifacts,
                latest_audit=latest_audits.get(song_id),
            ))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            catalogue_errors.append({
                "meta_path": str(path),
                "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
            })

    rows.sort(key=lambda row: int(row["song_id"]))
    status_counts = Counter(row["status"] for row in rows)
    error_counts = Counter(
        row["error_code"] or "unknown"
        for row in rows
        if row["status"] == "needs_review"
    )
    risk_counts = Counter(row["risk_level"] for row in rows)
    issue_counts = Counter(issue for row in rows for issue in row["issues"])

    review_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] == "needs_review":
            review_groups[row["error_code"] or "unknown"].append(row)

    manual: list[dict] = []
    selected: set[str] = set()

    def add(rows_to_add: list[dict], reason: str) -> None:
        for row in rows_to_add:
            if row["song_id"] in selected:
                continue
            selected.add(row["song_id"])
            manual.append({**row, "manual_review_reason": reason})

    add(
        [row for row in rows if row["risk_level"] in {"critical", "high"}],
        "automatic_high_risk",
    )
    for code, grouped in sorted(review_groups.items()):
        add(
            sorted(grouped, key=_stable_key)[: args.sample_per_review_code],
            f"needs_review_sample:{code}",
        )
    add(
        sorted([row for row in rows if row["status"] == "not_found"], key=_stable_key)[
            : args.not_found_sample
        ],
        "not_found_priority_sample",
    )
    for status in ("ok", "no_trivia"):
        add(
            sorted([row for row in rows if row["status"] == status], key=_stable_key)[
                : args.resolved_sample
            ],
            f"{status}_quality_sample",
        )

    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": now(),
        "status": (
            "issues_found"
            if (
                catalogue_errors
                or risk_counts["critical"]
                or risk_counts["high"]
                or manifest_validation["status"] != "ok"
                or manifest_validation.get("scope_complete") is not True
            )
            else "ok"
        ),
        "data_dir": str(data_dir),
        "cache_dir": str(cache_dir),
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "song_count": len(rows),
        "status_counts": dict(status_counts.most_common()),
        "needs_review_error_counts": dict(error_counts.most_common()),
        "risk_counts": dict(sorted(risk_counts.items(), key=lambda item: -SEVERITY[item[0]])),
        "issue_counts": dict(issue_counts.most_common()),
        "manual_review_count": len(manual),
        "catalogue_error_count": len(catalogue_errors),
        "run_report_error_count": len(run_report_errors),
        "manifest_validation": manifest_validation,
        "files": {
            "inventory": "inventory.jsonl",
            "needs_review_groups": "needs_review_groups.json",
            "manual_review": "manual_review.csv",
            "catalogue_errors": "catalogue_errors.json",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "inventory.jsonl", rows)
    _write_json(
        output_dir / "needs_review_groups.json",
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "groups": {
                code: grouped for code, grouped in sorted(review_groups.items())
            },
        },
    )
    csv_rows = [{
        **row,
        "artists": " | ".join(row["artists"]),
        "issues": " | ".join(row["issues"]),
        "binding": json.dumps(row["binding"], ensure_ascii=False) if row["binding"] else "",
        "decision": "",
        "manual_url": "",
        "review_note": "",
    } for row in manual]
    _write_csv(
        output_dir / "manual_review.csv",
        csv_rows,
        [
            "song_id",
            "title",
            "artists",
            "album",
            "release_date",
            "youtube_view_count",
            "status",
            "error_code",
            "risk_level",
            "issues",
            "source_url",
            "source_identity_shape",
            "fact_count",
            "artifact_ready",
            "artifact_ref",
            "artifact_file_exists",
            "artifact_reason",
            "artifact_record_count",
            "binding",
            "binding_contract",
            "manual_review_reason",
            "decision",
            "manual_url",
            "review_note",
            "meta_path",
        ],
    )
    _write_json(
        output_dir / "catalogue_errors.json",
        {
            "catalogue_errors": catalogue_errors,
            "run_report_errors": run_report_errors,
        },
    )
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Namuwiki bindings and context artifacts without network access."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("VAGUEFINDER_DATA_DIR", str(ROOT / "data/raw"))),
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/context")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-per-review-code", type=int, default=20)
    parser.add_argument("--not-found-sample", type=int, default=80)
    parser.add_argument("--resolved-sample", type=int, default=50)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        for name in (
            "sample_per_review_code",
            "not_found_sample",
            "resolved_sample",
        ):
            if getattr(args, name) < 0:
                raise ValueError(f"{name.replace('_', '-')} must be non-negative")
        summary = run_audit(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.strict and (
            summary["catalogue_error_count"]
            or summary["risk_counts"].get("critical", 0)
            or summary["risk_counts"].get("high", 0)
            or summary["manifest_validation"]["status"] != "ok"
            or summary["manifest_validation"].get("scope_complete") is not True
        ):
            return 1
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
