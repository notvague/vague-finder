"""Append compact Namuwiki facts and publish per-song context artifacts.

Music, comments, images and audio are never crawled again. This command only
scans data/raw/*/meta.json and never moves a song to failed_raw. Terminal
Namuwiki results are projected to artifacts/context/songs/<song_id>.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from src.crawler.context.document_parser import compact
from src.crawler.context.fetcher import Fetcher, article_url
from src.crawler.context.pipeline import collect_target
from src.crawler.context.schemas import NAMUWIKI_META_VERSION, NamuwikiMeta, SongTarget, digest, now
from src.crawler.context.store import Store
from src.crawler.context.trivia import extract_legacy_trivia_facts, refine_existing_facts
from src.embedding.context_artifacts import ContextArtifactStore
from src.embedding.fixtures.meta_validation import validate_meta_document

ROOT = Path(__file__).resolve().parents[3]
TERMINAL = {"ok", "no_trivia", "not_found"}
DEFAULT_CATALOG_RUN_LIMIT = 40
MAX_ROUTINE_REPORT_ROWS = 20
COMPACT_LEGACY_VERSIONS = {"namuwiki_v1", "namuwiki_v2"}
REEXTRACT_FROM_CACHE_VERSIONS = {"namuwiki_v2"}
LEGACY_SUCCESS = {"collected", "extracted", "no_relevant_fact"}
KNOWN = {
    "837567": ("사랑했나봐", "윤도현", "https://namu.wiki/w/사랑했나봐(윤도현)"),
    "1698598": ("거짓말", "BIGBANG", "https://namu.wiki/w/거짓말(BIGBANG)"),
}


def default_artifact_dir(data_dir: Path) -> Path:
    """Keep derived artifacts beside a custom raw-data sandbox."""
    data_dir = Path(data_dir).resolve()
    project = (
        data_dir.parent.parent
        if data_dir.name == "raw" and data_dir.parent.name == "data"
        else data_dir.parent
    )
    return (project / "artifacts/context").resolve()


def read_meta(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    meta = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(meta, dict):
        raise ValueError("meta JSON must be an object")
    return raw, meta


def current_namuwiki(meta: dict) -> dict:
    value = meta.get("namuwiki")
    return value if isinstance(value, dict) else {}


def legacy_namuwiki(meta: dict) -> dict:
    external = meta.get("external_context")
    value = external.get("namuwiki") if isinstance(external, dict) else None
    return value if isinstance(value, dict) else {}


def identify(meta: dict, folder: Path) -> tuple[str, str, list[str]]:
    song_id = str(meta.get("song_id") or meta.get("id") or "").strip()
    suffix = re.search(r"_(\d+)$", folder.name)
    song_id = song_id or (suffix.group(1) if suffix else "")
    if not re.fullmatch(r"\d+", song_id):
        raise ValueError("valid numeric song_id/id is required")
    if suffix and suffix.group(1) != song_id:
        raise ValueError("folder song_id and meta song_id differ")

    metadata = meta.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    title, artists = str(metadata.get("title", "")).strip(), metadata.get("artist", [])
    artists = [artists] if isinstance(artists, str) else artists
    if (
        not title
        or not isinstance(artists, list)
        or not artists
        or not all(isinstance(artist, str) and artist.strip() for artist in artists)
    ):
        raise ValueError("metadata.title and metadata.artist are required")
    return song_id, title, [artist.strip() for artist in artists]


def _artist_variants(artists: list[str]) -> list[str]:
    """Prefer credited names and bounded typography-only URL aliases."""
    values: list[str] = []

    for artist in artists:
        base = re.sub(r"\([^)]*\)", "", artist).strip()
        if base:
            values.append(base)
        for part in re.findall(r"\(([^)]*)\)", artist):
            values.append(part.strip())
        for part in re.split(r"[/·|]", artist):
            values.append(part.strip())
        values.append(artist.strip())

    # Keep ordinary aliases ahead of typography-only fallbacks.  Otherwise a
    # collapsed Latin name could displace a useful parenthesized Korean alias
    # from the two bounded URL candidates (for example ALPHA DRIVE ONE).
    canonical = list(dict.fromkeys(value for value in values if value))
    collapsed = [
        re.sub(r"\s+", "", value)
        for value in canonical
        if re.sub(r"\s+", "", value) != value
    ]
    return list(dict.fromkeys([*canonical, *collapsed]))


def _artist_keys(artists: list[str]) -> set[str]:
    return {compact(value) for value in _artist_variants(artists) if compact(value)}


_TITLE_QUALIFIER = re.compile(
    r"(?<![A-Za-z])(?:feat(?:uring)?\.?|with|prod(?:uced)?\.?|ver(?:sion)?\.?|"
    r"remaster(?:ed)?|live|inst(?:rumental)?\.?|edit|original|dialog|ost|"
    r"acoustic|karaoke)(?![A-Za-z])",
    re.I,
)


def _title_variants(title: str) -> list[str]:
    """Return bounded discovery aliases while preserving the credited title."""
    values = [str(title).strip()]
    value = values[0]

    # Version/feature suffixes frequently exist in Melon titles but not in a
    # NamuWiki article title.  They are fallback aliases only; page binding
    # still requires independent artist/album evidence.
    without_qualified_parens = re.sub(
        r"\s*[\[(]([^)\]]{0,80})[)\]]\s*$",
        lambda match: "" if _TITLE_QUALIFIER.search(match.group(1)) else match.group(0),
        value,
    ).strip()
    if without_qualified_parens and without_qualified_parens != value:
        values.append(without_qualified_parens)

    without_feature = re.sub(
        r"\s+(?:[-–—]\s*)?(?:feat(?:uring)?\.?|with|prod(?:uced)?\.?)\s+.+$",
        "",
        value,
        flags=re.I,
    ).strip()
    if without_feature and without_feature != value:
        values.append(without_feature)

    return list(dict.fromkeys(item for item in values if item))[:3]


def _binding_metadata(meta: dict, title: str) -> tuple[str | None, int | None, list[str]]:
    metadata = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    album_value = metadata.get("album")
    album = str(album_value).strip() if album_value is not None else ""
    release_value = str(metadata.get("release_date") or "")
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", release_value)
    release_year = int(match.group(1)) if match else None
    title_aliases = _title_variants(title)[1:]
    return album or None, release_year, title_aliases


def _previous_urls(meta: dict) -> list[str]:
    urls = []
    current, legacy = current_namuwiki(meta), legacy_namuwiki(meta)
    # A current non-terminal URL was never identity-verified. Reusing it would
    # pin retries to the same unrelated same-title page.
    current_url = current.get("source_url") if current.get("status") in TERMINAL else None
    for value in (current_url, legacy.get("source_url"), legacy.get("page_url")):
        if isinstance(value, str) and value.strip():
            urls.append(value)
    for source in legacy.get("sources", []):
        if isinstance(source, dict) and isinstance(source.get("url"), str):
            urls.append(source["url"])
    for attempt in legacy.get("attempts", []):
        if isinstance(attempt, dict) and isinstance(attempt.get("url"), str):
            urls.append(attempt["url"])
    return list(dict.fromkeys(urls))


def candidate_target(meta: dict, folder: Path, url_map: dict) -> SongTarget:
    song_id, title, artists = identify(meta, folder)
    album, release_year, title_aliases = _binding_metadata(meta, title)
    supplied, known = url_map.get(song_id), KNOWN.get(song_id)
    explicit = supplied is not None

    if supplied is not None:
        urls = [supplied] if isinstance(supplied, str) else supplied
    elif known and compact(title) == compact(known[0]) and compact(known[1]) in _artist_keys(artists):
        urls, explicit = [known[2]], True
    else:
        # These are candidates, not search-engine results. A disambiguation page
        # is accepted only after visible title/artist verification.
        disambiguators = _artist_variants(artists)[:2]
        previous = _previous_urls(meta)
        urls = list(previous)
        explicit = bool(previous)
        for index, candidate_title in enumerate(_title_variants(title)):
            artist_candidates = disambiguators if index == 0 else disambiguators[:1]
            urls.extend(
                "https://namu.wiki/w/"
                + quote(f"{candidate_title}({artist})", safe="()")
                for artist in artist_candidates
            )
            if not re.search(r"\(\s*노래\s*\)\s*$", candidate_title):
                urls.append(
                    "https://namu.wiki/w/"
                    + quote(f"{candidate_title}(노래)", safe="()")
                )
            urls.append(
                "https://namu.wiki/w/" + quote(candidate_title, safe="()")
            )

    if not isinstance(urls, list) or not urls:
        raise ValueError("URL map/history must contain Namuwiki URLs")
    normalized = list(dict.fromkeys(article_url(str(url)) for url in urls))
    if supplied is not None and len(normalized) > 8:
        raise ValueError("URL map may contain at most 8 Namuwiki URLs per song")
    normalized = normalized[:8]
    return SongTarget(
        song_id=song_id,
        title=title,
        artists=artists,
        album=album,
        release_year=release_year,
        title_aliases=title_aliases,
        urls=normalized,
        explicit_urls=explicit,
    )


def should_process(
    meta: dict,
    *,
    retry_not_found: bool = False,
    retry_needs_review: bool = False,
    force: bool = False,
) -> tuple[str, str]:
    """Return (collect|reextract|migrate|skip, reason)."""
    if force:
        return "collect", "forced"

    current = current_namuwiki(meta)
    if current.get("schema_version") == NAMUWIKI_META_VERSION:
        if not str(current.get("collected_at", "")).strip():
            return "migrate", "repair_missing_collected_at"
        try:
            status = NamuwikiMeta.model_validate(current).status
        except ValueError:
            return "collect", "repair_invalid_current_schema"
        if status == "not_found" and retry_not_found:
            return "collect", "retry_not_found"
        if status == "needs_review" and not retry_needs_review:
            return "skip", "needs_review_manual_resolution"
        if status in TERMINAL:
            return "skip", "already_complete"
        return "collect", "retry_incomplete_or_error"

    if current.get("schema_version") in REEXTRACT_FROM_CACHE_VERSIONS:
        status = current.get("status")
        if status == "not_found":
            if retry_not_found:
                return "collect", "retry_not_found"
            return "migrate", "upgrade_compact_namuwiki_schema"
        if status in {"ok", "no_trivia"}:
            # v2 may already have discarded dependent source sentences. Re-run
            # projection from its cached HTML snapshot, never from the network.
            return "reextract", "upgrade_refinement_rules_from_cached_source"
        return "collect", "retry_incomplete_compact_legacy"

    if current.get("schema_version") in COMPACT_LEGACY_VERSIONS:
        status = current.get("status")
        if status == "not_found" and retry_not_found:
            return "collect", "retry_not_found"
        if status in TERMINAL:
            return "migrate", "upgrade_compact_namuwiki_schema"
        return "collect", "retry_incomplete_compact_legacy"
    if current:
        return "collect", "repair_unknown_current_schema"

    legacy = legacy_namuwiki(meta)
    if legacy:
        if legacy.get("status") in LEGACY_SUCCESS and isinstance(legacy.get("sources"), list):
            if any(isinstance(source, dict) and source.get("blocks") for source in legacy["sources"]):
                return "migrate", "migrate_verbose_legacy_context"
        if legacy.get("status") == "not_found" or legacy.get("match_status") == "not_found":
            return "migrate", "migrate_legacy_not_found"
        return "collect", "retry_legacy_incomplete_or_error"
    return "collect", "missing_namuwiki"


def migrate_legacy(meta: dict) -> dict | None:
    legacy = legacy_namuwiki(meta)
    if legacy.get("status") == "not_found" or legacy.get("match_status") == "not_found":
        return NamuwikiMeta(status="not_found", source_url=legacy.get("source_url"), facts=[]).model_dump(
            exclude_none=True
        )
    facts, trivia_found, source_url = extract_legacy_trivia_facts(legacy.get("sources", []))
    if not facts and not trivia_found and not any(
        isinstance(source, dict) and source.get("blocks") for source in legacy.get("sources", [])
    ):
        return None
    status = "ok" if facts else "no_trivia"
    return NamuwikiMeta(status=status, source_url=source_url, facts=facts).model_dump(exclude_none=True)


def migrate_existing(meta: dict) -> dict | None:
    """Upgrade either compact v1 or the older verbose nested representation."""
    current = current_namuwiki(meta)
    if current.get("schema_version") == NAMUWIKI_META_VERSION:
        repaired = copy.deepcopy(current)
        repaired["collected_at"] = str(repaired.get("collected_at", "")).strip() or now()
        return NamuwikiMeta.model_validate(repaired).model_dump(exclude_none=True)
    if current.get("schema_version") not in COMPACT_LEGACY_VERSIONS:
        return migrate_legacy(meta)

    status = current.get("status")
    source_url = current.get("source_url")
    collected_at = current.get("collected_at") or now()
    if status == "not_found":
        return NamuwikiMeta(
            status="not_found",
            source_url=source_url,
            collected_at=collected_at,
            facts=[],
        ).model_dump(exclude_none=True)
    if status == "no_trivia":
        if not source_url:
            return None
        return NamuwikiMeta(
            status="no_trivia",
            source_url=source_url,
            collected_at=collected_at,
            facts=[],
        ).model_dump(exclude_none=True)
    if status != "ok" or not source_url or not isinstance(current.get("facts"), list):
        return None

    metadata = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    facts = refine_existing_facts(
        current["facts"],
        page_title=str(metadata.get("title", "")).strip() or None,
    )
    projected_status = "ok" if facts else "no_trivia"
    return NamuwikiMeta(
        status=projected_status,
        source_url=source_url,
        collected_at=collected_at,
        facts=facts,
    ).model_dump(exclude_none=True)


def merge_namuwiki(meta: dict, value: dict) -> dict:
    """Remove the old nested payload and append one compact top-level key."""
    value = NamuwikiMeta.model_validate(value).model_dump(exclude_none=True)
    updated = copy.deepcopy(meta)
    external = updated.get("external_context")
    if external is not None and not isinstance(external, dict):
        raise ValueError("external_context is not an object; refusing overwrite")
    if isinstance(external, dict) and "namuwiki" in external:
        external = copy.deepcopy(external)
        external.pop("namuwiki", None)
        if external:
            updated["external_context"] = external
        else:
            updated.pop("external_context", None)
    updated.pop("namuwiki", None)
    updated["namuwiki"] = value
    return updated


def write_meta(path: Path, expected: bytes, updated: dict, store: Store, song_id: str) -> str:
    """Back up exact bytes, reject concurrent edits, then atomically replace."""
    lock, temp_path = path.with_name("meta.namuwiki.lock"), None
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.close(descriptor)
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError("meta_changed_during_collection; refusing stale overwrite")
        backup = f"meta_backups/{song_id}/{hashlib.sha256(expected).hexdigest()}.json"
        if not store.path(backup).exists():
            store.write_bytes(backup, expected)
        payload = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        descriptor, temp_path = tempfile.mkstemp(prefix=".namuwiki-", suffix=".tmp", dir=path.parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, path.stat().st_mode & 0o777)
        if path.read_bytes() != expected:
            raise ValueError("meta_changed_before_commit; refusing stale overwrite")
        os.replace(temp_path, path)
        temp_path = None
        return backup
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        lock.unlink(missing_ok=True)


def _summary(row: dict, value: dict) -> None:
    facts = value.get("facts", [])
    row.update(
        status=value.get("status"),
        source_url=value.get("source_url"),
        fact_count=len(facts),
    )
    if facts:
        row["categories"] = dict(Counter(fact["category"] for fact in facts))
    if value.get("error_code"):
        row["error_code"] = value["error_code"]


def _artifact_summary(row: dict, value) -> None:
    if hasattr(value, "as_dict"):
        data = value.as_dict()
        row.update(
            artifact_ref=data.get("artifact_ref"),
            artifact_ready=data.get("ready", False),
            artifact_reason=data.get("reason"),
            context_record_count=data.get("record_count", 0),
            artifact_source_fact_count=data.get("source_fact_count", 0),
            artifact_covered_fact_count=data.get("covered_source_fact_count", 0),
            artifact_sparse_term_count=data.get("sparse_term_count", 0),
        )
        return
    row.update(
        artifact_ref=value.get("artifact_ref"),
        artifact_ready=True,
        artifact_action=value.get("artifact_action"),
        context_record_count=value.get("record_count", 0),
        artifact_source_fact_count=value.get("source_fact_count", 0),
        artifact_covered_fact_count=value.get("covered_source_fact_count", 0),
        artifact_sparse_term_count=value.get("sparse_term_count", 0),
        artifact_source_digest=value.get("source_digest"),
    )


def _percent(done: int, total: int) -> float:
    return round((done / total * 100.0) if total else 100.0, 1)


def _coverage(paths: list[Path], artifacts: ContextArtifactStore) -> dict:
    terminal_meta = artifact_ready = records = 0
    status_counts: Counter = Counter()
    errors = []
    for path in paths:
        try:
            _, meta = read_meta(path)
            current = current_namuwiki(meta)
            if current.get("schema_version") == NAMUWIKI_META_VERSION:
                try:
                    parsed = NamuwikiMeta.model_validate(current)
                except ValueError:
                    parsed = None
                if parsed is not None and parsed.status in TERMINAL:
                    terminal_meta += 1
                    status_counts[parsed.status] += 1
            state = artifacts.inspect(meta)
            if state.ready:
                artifact_ready += 1
                records += state.record_count
        except (OSError, ValueError, TypeError) as exc:
            errors.append(
                {
                    "meta_path": str(path),
                    "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            )
    total = len(paths)
    return {
        "scope_total": total,
        "terminal_meta": terminal_meta,
        "artifact_ready": artifact_ready,
        "pending": max(0, total - artifact_ready),
        "completion_percent": _percent(artifact_ready, total),
        "context_record_count": records,
        "status_counts": dict(sorted(status_counts.items())),
        "coverage_error_count": len(errors),
        "coverage_errors": errors,
    }


def _unresolved_catalog(paths: list[Path], artifacts: ContextArtifactStore) -> dict:
    """Return a compact, deterministic queue of songs without ready artifacts."""
    songs = []
    reason_counts: Counter = Counter()
    for path in paths:
        try:
            _, meta = read_meta(path)
            song_id, title, artists = identify(meta, path.parent)
            state = artifacts.inspect(meta)
            if state.ready:
                continue

            current = current_namuwiki(meta)
            status = None
            error_code = None
            attempted_url = None
            if current.get("schema_version") == NAMUWIKI_META_VERSION:
                try:
                    parsed = NamuwikiMeta.model_validate(current)
                except ValueError:
                    reason = "invalid_context_metadata"
                else:
                    status = parsed.status
                    error_code = parsed.error_code
                    attempted_url = parsed.source_url
                    if status == "needs_review":
                        reason = "needs_review"
                    elif status == "error":
                        reason = "collection_error"
                    elif status in TERMINAL:
                        reason = "artifact_pending"
                    else:
                        reason = "pending_collection"
            elif current or legacy_namuwiki(meta):
                reason = "migration_pending"
            else:
                reason = "pending_collection"

            reason_counts[reason] += 1
            row = {
                "song_id": song_id,
                "title": title,
                "artists": artists,
                "reason": reason,
                "meta_path": str(path),
                "artifact_reason": state.reason,
            }
            if status:
                row["namuwiki_status"] = status
            if error_code:
                row["error_code"] = error_code
            if attempted_url:
                row["attempted_url"] = attempted_url
            songs.append(row)
        except (OSError, ValueError, TypeError) as exc:
            folder_match = re.search(r"_(\d+)$", path.parent.name)
            song_id = folder_match.group(1) if folder_match else None
            reason_counts["catalog_error"] += 1
            songs.append(
                {
                    **({"song_id": song_id} if song_id else {}),
                    "reason": "catalog_error",
                    "meta_path": str(path),
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            )

    songs.sort(
        key=lambda row: (
            int(row["song_id"]) if str(row.get("song_id", "")).isdigit() else -1,
            row["meta_path"],
        )
    )

    def ids_for(reason: str) -> list[str]:
        return [
            row["song_id"]
            for row in songs
            if row.get("reason") == reason and row.get("song_id")
        ]

    return {
        "scope_total": len(paths),
        "unresolved_count": len(songs),
        "reason_counts": dict(sorted(reason_counts.items())),
        "unresolved_song_ids": [row["song_id"] for row in songs if row.get("song_id")],
        "needs_review_song_ids": ids_for("needs_review"),
        "collection_error_song_ids": ids_for("collection_error"),
        "artifact_pending_song_ids": ids_for("artifact_pending"),
        "pending_collection_song_ids": ids_for("pending_collection"),
        "songs": songs,
    }


def _show_progress(args, *, processed: int, selected: int, coverage: dict, row: dict) -> None:
    if getattr(args, "quiet_progress", False):
        return
    print(
        "[NAMUWIKI] "
        f"run {processed}/{selected} | "
        f"artifacts {coverage['artifact_ready']}/{coverage['scope_total']} "
        f"({coverage['completion_percent']:.1f}%) | "
        f"song_id={row.get('song_id', '?')} | "
        f"action={row.get('action', '?')}",
        file=sys.stderr,
        flush=True,
    )


def run_backfill(args, *, fetcher_factory=Fetcher) -> dict:
    data_dir, cache_dir = args.data_dir.resolve(), args.cache_dir.resolve()
    configured_artifact_dir = getattr(args, "artifact_dir", None)
    artifact_dir = (
        Path(configured_artifact_dir).resolve()
        if configured_artifact_dir is not None
        else default_artifact_dir(data_dir)
    )
    if not data_dir.is_dir():
        raise ValueError(f"raw directory not found: {data_dir}")
    if cache_dir == data_dir or cache_dir.is_relative_to(data_dir):
        raise ValueError("cache-dir must be outside raw")
    if artifact_dir == data_dir or artifact_dir.is_relative_to(data_dir):
        raise ValueError("artifact-dir must be outside raw")
    if (
        artifact_dir == cache_dir
        or artifact_dir.is_relative_to(cache_dir)
        or cache_dir.is_relative_to(artifact_dir)
    ):
        raise ValueError("artifact-dir and cache-dir must be separate")

    url_map_path = getattr(args, "url_map", None)
    url_map = json.loads(url_map_path.read_text(encoding="utf-8-sig")) if url_map_path else {}
    if not isinstance(url_map, dict):
        raise ValueError("url-map must be an object: song_id -> URL or URLs")

    requested = {str(song_id) for song_id in (getattr(args, "song_ids", None) or [])}
    configured_limit = getattr(args, "limit", None)
    effective_limit = (
        configured_limit
        if configured_limit is not None
        else (None if requested else DEFAULT_CATALOG_RUN_LIMIT)
    )
    artifacts = ContextArtifactStore(artifact_dir)
    rows, targets, seen = [], [], set()
    eligible_paths: list[Path] = []
    all_eligible_paths: list[Path] = []
    active_artifact_song_ids: set[str] = set()
    ready_records: dict[str, int] = {}
    catalog_entries = []
    song_locations: dict[str, list[str]] = {}
    catalog_raw_meta_count = 0
    catalog_core_incomplete_count = 0
    catalog_error_count = 0
    queued_after_limit = 0
    routine_rows_reported = 0
    routine_rows_omitted = 0

    def append_row(row: dict, *, routine: bool = False) -> None:
        nonlocal routine_rows_reported, routine_rows_omitted
        # Full-catalog runs would otherwise print thousands of identical
        # already-complete/manual-review rows in the final JSON. Targeted runs
        # retain every requested row for diagnosis.
        if routine and not requested and routine_rows_reported >= MAX_ROUTINE_REPORT_ROWS:
            routine_rows_omitted += 1
            return
        rows.append(row)
        if routine and not requested:
            routine_rows_reported += 1

    # Build the global active-song set before applying --song-ids. Otherwise a
    # two-song test run would make every already-generated artifact for other
    # valid raw songs look orphaned in the embedding manifest.
    for path in sorted(data_dir.glob("*/meta.json")):
        if path.is_symlink() or path.parent.is_symlink():
            continue
        catalog_raw_meta_count += 1
        try:
            raw, meta = read_meta(path)
            song_id, title, _ = identify(meta, path.parent)
            issues = validate_meta_document(meta, song_dir=path.parent, require_media_files=True)
            song_locations.setdefault(song_id, []).append(str(path))
            if issues:
                catalog_core_incomplete_count += 1
            else:
                active_artifact_song_ids.add(song_id)
                all_eligible_paths.append(path)
            # Retain only lightweight catalog data for this run's scope. A
            # two-song test over a large raw catalog stays bounded in memory.
            if not requested or song_id in requested:
                catalog_entries.append((path, song_id, title, issues, None))
        except (ValueError, OSError, TypeError) as exc:
            catalog_error_count += 1
            folder_match = re.search(r"_(\d+)$", path.parent.name)
            inferred_id = folder_match.group(1) if folder_match else None
            if not requested or inferred_id in requested:
                catalog_entries.append((path, inferred_id, None, None, str(exc)))

    duplicates = {
        song_id: locations
        for song_id, locations in song_locations.items()
        if len(locations) > 1
    }
    if duplicates:
        detail = "; ".join(
            f"{song_id}: {', '.join(locations)}"
            for song_id, locations in sorted(duplicates.items())
        )
        raise ValueError("duplicate song_id in raw; resolve before backfill: " + detail)

    for path, song_id, title, issues, catalog_error in catalog_entries:
        if requested and song_id not in requested:
            continue
        if song_id:
            seen.add(song_id)
        if catalog_error:
            append_row(
                {
                    **({"song_id": song_id} if song_id else {}),
                    "meta_path": str(path),
                    "action": "skip",
                    "reason": catalog_error,
                }
            )
            continue

        # Re-read only scoped songs. The first pass deliberately retains no
        # full meta payload, keeping a targeted run bounded on a large corpus.
        try:
            raw, meta = read_meta(path)
            current_song_id, current_title, _ = identify(meta, path.parent)
            if current_song_id != song_id or current_title != title:
                raise ValueError("meta identity changed during catalog scan")
        except (ValueError, OSError, TypeError) as exc:
            append_row(
                {
                    "song_id": song_id,
                    "title": title,
                    "meta_path": str(path),
                    "action": "skip",
                    "reason": str(exc),
                }
            )
            continue

        row = {
            "song_id": song_id,
            "title": title,
            "meta_path": str(path),
        }
        try:
            # Background context is optional, but artifacts must only be made
            # for a song whose core metadata/media are valid. This never moves
            # the song folder.
            if issues:
                row.update(
                    action="skip",
                    reason="core_music_incomplete",
                    issues=[{"path": issue.path, "reason": issue.reason} for issue in issues],
                )
                append_row(row)
                continue
            eligible_paths.append(path)

            artifact_state = artifacts.inspect(meta)
            if artifact_state.ready:
                ready_records[song_id] = artifact_state.record_count

            artifacts_only = bool(getattr(args, "artifacts_only", False))
            if artifacts_only:
                _summary(row, current_namuwiki(meta))
                if not artifact_state.applicable:
                    row.update(action="skip", reason="terminal_namuwiki_metadata_required")
                    _artifact_summary(row, artifact_state)
                    append_row(row, routine=True)
                    continue
                if artifact_state.ready and not getattr(args, "force", False):
                    row.update(action="skip", reason="artifact_already_current")
                    _artifact_summary(row, artifact_state)
                    append_row(row, routine=True)
                    continue
                mode = "artifact"
                reason = (
                    "forced_artifact_rebuild"
                    if getattr(args, "force", False)
                    else artifact_state.reason
                )
            else:
                mode, reason = should_process(
                    meta,
                    retry_not_found=getattr(args, "retry_not_found", False),
                    retry_needs_review=getattr(args, "retry_needs_review", False),
                    force=getattr(args, "force", False),
                )
            row.update(action=mode, reason=reason)
            if mode == "skip":
                _summary(row, current_namuwiki(meta))
                _artifact_summary(row, artifact_state)
                if artifact_state.applicable and not artifact_state.ready:
                    mode = "artifact"
                    row.update(
                        action=mode,
                        reason=artifact_state.reason,
                    )
                else:
                    append_row(row, routine=True)
                    continue

            target = (
                None
                if mode in {"migrate", "artifact"}
                else candidate_target(meta, path.parent, url_map)
            )
            if target is not None:
                row["candidate_urls"] = target.urls
                if getattr(args, "html_dir", None) and (
                    not target.explicit_urls or len(target.urls) != 1
                ):
                    raise ValueError(
                        "saved HTML requires one explicit URL per song via --url-map/history"
                    )
            if effective_limit is not None and len(targets) >= effective_limit:
                queued_after_limit += 1
                if requested:
                    row.update(action="queued", reason="run_limit")
                    append_row(row)
                continue
            append_row(row)
            targets.append((mode, path, raw, meta, target, row))
        except (ValueError, OSError, TypeError) as exc:
            row.update(action="skip", reason=str(exc))
            append_row(row)

    # Cache-only migrations/artifact repairs do not consume the HTTP budget;
    # complete them before starting network collection.
    targets.sort(key=lambda item: item[0] == "collect")

    missing = sorted(requested - seen)
    if missing:
        raise ValueError("requested song IDs not found in raw: " + ", ".join(missing))

    before = _coverage(eligible_paths, artifacts)
    catalog_before = _coverage(all_eligible_paths, artifacts)
    report = {
        "data_dir": str(data_dir),
        "artifact_dir": str(artifact_dir),
        "dry_run": bool(getattr(args, "dry_run", False)),
        "catalog": {
            "raw_meta_count": catalog_raw_meta_count,
            "core_valid_song_count": len(active_artifact_song_ids),
            "core_incomplete_song_count": catalog_core_incomplete_count,
            "catalog_error_count": catalog_error_count,
        },
        "catalog_progress": {
            "completed_before": catalog_before["artifact_ready"],
            "scope_total": catalog_before["scope_total"],
            "pending_before": catalog_before["pending"],
            "completion_percent_before": catalog_before["completion_percent"],
            "completed_after": catalog_before["artifact_ready"],
            "pending_after": catalog_before["pending"],
            "completion_percent_after": catalog_before["completion_percent"],
        },
        "eligible_count": len(eligible_paths),
        "selected_count": len(targets),
        "batch": {
            "limit": effective_limit,
            "queued_after_limit": queued_after_limit,
            "stopped_early": False,
            "stop_reason": None,
            "routine_song_rows_omitted": routine_rows_omitted,
        },
        "progress": {
            "completed_before": before["artifact_ready"],
            "scope_total": before["scope_total"],
            "pending_before": before["pending"],
            "completion_percent_before": before["completion_percent"],
            "selected_this_run": len(targets),
            "processed_this_run": 0,
            "succeeded_this_run": 0,
            "deferred_this_run": 0,
            "queued_this_run": 0,
            "failed_this_run": 0,
            "completed_after": before["artifact_ready"],
            "pending_after": before["pending"],
            "completion_percent_after": before["completion_percent"],
        },
        "songs": rows,
    }
    if report["dry_run"]:
        return report

    started_at = now()
    run_id = digest([started_at, os.getpid(), str(data_dir), sorted(requested)])[:24]

    def progress_payload(state: str, *, current_song_id: str | None = None) -> dict:
        return {
            "run_id": run_id,
            "state": state,
            "started_at": started_at,
            "finished_at": now() if state in {"completed", "interrupted"} else None,
            "data_dir": str(data_dir),
            "artifact_dir": str(artifact_dir),
            "requested_song_ids": sorted(requested),
            "current_song_id": current_song_id,
            "catalog_progress": report["catalog_progress"],
            **report["progress"],
        }

    def publish_unresolved() -> None:
        unresolved = _unresolved_catalog(all_eligible_paths, artifacts)
        report["artifact_unresolved_ref"] = artifacts.write_unresolved(unresolved)
        report["unresolved"] = {
            "scope_total": unresolved["scope_total"],
            "unresolved_count": unresolved["unresolved_count"],
            "reason_counts": unresolved["reason_counts"],
            "needs_review_count": len(unresolved["needs_review_song_ids"]),
            "collection_error_count": len(unresolved["collection_error_song_ids"]),
            "artifact_pending_count": len(unresolved["artifact_pending_song_ids"]),
            "pending_collection_count": len(unresolved["pending_collection_song_ids"]),
        }

    # A no-op run still publishes a trustworthy manifest/progress snapshot.
    if not targets:
        with artifacts.writer():
            after = _coverage(eligible_paths, artifacts)
            catalog_after = _coverage(all_eligible_paths, artifacts)
            report["progress"].update(
                completed_after=after["artifact_ready"],
                pending_after=after["pending"],
                completion_percent_after=after["completion_percent"],
                context_record_count=after["context_record_count"],
            )
            report["catalog_progress"].update(
                completed_after=catalog_after["artifact_ready"],
                pending_after=catalog_after["pending"],
                completion_percent_after=catalog_after["completion_percent"],
            )
            manifest = artifacts.publish_manifest(
                coverage=catalog_after,
                active_song_ids=active_artifact_song_ids,
            )
            report["artifact_manifest_ref"] = "manifest.json"
            report["artifact_progress_ref"] = artifacts.write_progress(
                progress_payload("completed")
            )
            publish_unresolved()
            report["artifact_manifest"] = {
                "song_count": manifest["song_count"],
                "record_count": manifest["record_count"],
                "sparse_profile_count": manifest["sparse_profile_count"],
                "invalid_artifact_count": manifest["invalid_artifact_count"],
                "orphan_artifact_count": manifest["orphan_artifact_count"],
            }
        return report

    store = Store(cache_dir)
    audits = []
    counters = {"processed": 0, "succeeded": 0, "deferred": 0, "failed": 0}
    with store.writer():
        with artifacts.writer():
            report["artifact_progress_ref"] = artifacts.write_progress(
                progress_payload("running")
            )
            needs_fetcher = any(mode in {"collect", "reextract"} for mode, *_ in targets)
            fetcher = (
                fetcher_factory(
                    store,
                    interval=getattr(args, "interval", 3.0),
                    timeout=getattr(args, "timeout", 20.0),
                    max_requests=getattr(args, "max_requests", 100),
                    render=not getattr(args, "no_render", False),
                    render_timeout=getattr(args, "render_timeout", 30.0),
                )
                if needs_fetcher
                else None
            )
            interrupted = True
            try:
                for target_index, (mode, path, raw, meta, target, row) in enumerate(targets):
                    if mode == "collect" and (
                        fetcher is None
                        or fetcher.stopped
                        or fetcher.requests_made >= getattr(args, "max_requests", 100)
                    ):
                        stop_reason = (
                            "access_stop"
                            if fetcher is not None and fetcher.stopped
                            else "request_budget_exhausted"
                        )
                        remaining = targets[target_index:]
                        for _, _, _, _, _, queued_row in remaining:
                            queued_row.update(
                                action="queued",
                                reason=f"{stop_reason}_no_meta_change",
                                meta_unchanged=True,
                            )
                        counters["deferred"] += len(remaining)
                        report["batch"].update(
                            stopped_early=True,
                            stop_reason=stop_reason,
                        )
                        report["progress"].update(
                            deferred_this_run=counters["deferred"],
                            queued_this_run=len(remaining),
                        )
                        artifacts.write_progress(progress_payload("running"))
                        break

                    meta_committed = False
                    outcome = "failed"
                    try:
                        if mode == "artifact":
                            result = artifacts.sync_song(meta)
                            row.update(
                                action="artifact_synced",
                                reason="context_artifact_reconciled",
                            )
                            _summary(row, current_namuwiki(meta))
                            _artifact_summary(row, result)
                            ready_records[row["song_id"]] = result["record_count"]
                            outcome = "succeeded"
                        else:
                            if mode == "migrate":
                                value = migrate_existing(meta)
                                if value is None:
                                    row.update(
                                        action="deferred",
                                        reason="legacy_payload_has_no_reusable_facts",
                                    )
                                    outcome = "deferred"
                                    continue
                                audit = {
                                    "song_id": row["song_id"],
                                    "status": value["status"],
                                    "method": "offline_schema_projection",
                                }
                            else:
                                value, audit = collect_target(
                                    store,
                                    fetcher,
                                    target,
                                    html_dir=getattr(args, "html_dir", None),
                                    # v2 upgrades are cache-only and make zero
                                    # Namuwiki network requests.
                                    offline=getattr(args, "offline", False)
                                    or mode == "reextract",
                                    refresh=(
                                        getattr(args, "refresh", False)
                                        if mode == "collect"
                                        else False
                                    ),
                                )
                            audits.append(audit)

                            previous = current_namuwiki(meta)
                            if (
                                previous.get("status") in TERMINAL
                                and value.get("status") not in TERMINAL
                            ):
                                reason = (
                                    "cached_reextract_failed_kept_previous"
                                    if mode == "reextract"
                                    else "failed_refresh_kept_previous_success"
                                )
                                row.update(action="kept_previous", reason=reason)
                                _summary(row, previous)
                                previous_state = artifacts.inspect(meta)
                                if previous_state.applicable:
                                    result = artifacts.sync_song(meta)
                                    _artifact_summary(row, result)
                                    ready_records[row["song_id"]] = result["record_count"]
                                    outcome = "succeeded"
                                else:
                                    _artifact_summary(row, previous_state)
                                    outcome = "deferred"
                            elif value.get("status") == "error":
                                # Network/access/request-budget failures are
                                # run diagnostics, not durable song context.
                                # Leaving meta untouched makes the next batch
                                # resume this song naturally.
                                row.update(
                                    action="deferred",
                                    reason=value.get("error_code") or "collection_error",
                                    meta_unchanged=True,
                                    namuwiki_committed=False,
                                    artifact_ready=False,
                                )
                                _summary(row, value)
                                outcome = "deferred"
                            else:
                                updated = merge_namuwiki(meta, value)
                                backup = write_meta(
                                    path,
                                    raw,
                                    updated,
                                    store,
                                    row["song_id"],
                                )
                                meta_committed = True
                                action = {
                                    "migrate": "migrated",
                                    "reextract": "reprocessed",
                                }.get(mode, "updated")
                                row.update(
                                    action=action,
                                    namuwiki_action=action,
                                    backup_ref=backup,
                                    namuwiki_committed=True,
                                    meta_unchanged=False,
                                )
                                _summary(row, value)
                                if value.get("status") not in TERMINAL:
                                    # A same-title page that fails identity
                                    # binding is a normal manual-review result,
                                    # not an artifact write failure. Only
                                    # terminal context can enter embeddings.
                                    row.update(
                                        action="needs_review",
                                        reason=(
                                            value.get("error_code")
                                            or "manual_binding_review_required"
                                        ),
                                        artifact_ready=False,
                                    )
                                    _artifact_summary(row, artifacts.inspect(updated))
                                    outcome = "deferred"
                                else:
                                    result = artifacts.sync_song(updated)
                                    _artifact_summary(row, result)
                                    ready_records[row["song_id"]] = result["record_count"]
                                    outcome = "succeeded"
                    except Exception as exc:
                        ready_records.pop(row.get("song_id", ""), None)
                        row.update(
                            action="artifact_error" if (mode == "artifact" or meta_committed) else "error",
                            reason=type(exc).__name__,
                            error_message=(
                                str(exc)[:300]
                                if isinstance(exc, (ValueError, OSError, RuntimeError))
                                else None
                            ),
                            meta_unchanged=not meta_committed,
                            namuwiki_committed=meta_committed,
                            artifact_ready=False,
                        )
                        audits.append(
                            {
                                "song_id": row.get("song_id"),
                                "status": "error",
                                "stage": (
                                    "artifact"
                                    if (mode == "artifact" or meta_committed)
                                    else "collection"
                                ),
                                "error_type": type(exc).__name__,
                                "message": (
                                    str(exc)[:300]
                                    if isinstance(exc, (ValueError, OSError, RuntimeError))
                                    else None
                                ),
                            }
                        )
                        outcome = "failed"
                    finally:
                        counters["processed"] += 1
                        counters[outcome] += 1
                        live_total = len(eligible_paths)
                        live_ready = len(ready_records)
                        report["progress"].update(
                            processed_this_run=counters["processed"],
                            succeeded_this_run=counters["succeeded"],
                            deferred_this_run=counters["deferred"],
                            failed_this_run=counters["failed"],
                            completed_after=live_ready,
                            pending_after=max(0, live_total - live_ready),
                            completion_percent_after=_percent(live_ready, live_total),
                            context_record_count=sum(ready_records.values()),
                        )
                        artifacts.write_progress(
                            progress_payload(
                                "running",
                                current_song_id=row.get("song_id"),
                            )
                        )
                        _show_progress(
                            args,
                            processed=counters["processed"],
                            selected=len(targets),
                            coverage={
                                "artifact_ready": live_ready,
                                "scope_total": live_total,
                                "completion_percent": _percent(live_ready, live_total),
                            },
                            row=row,
                        )

                after = _coverage(eligible_paths, artifacts)
                catalog_after = _coverage(all_eligible_paths, artifacts)
                report["progress"].update(
                    completed_after=after["artifact_ready"],
                    pending_after=after["pending"],
                    completion_percent_after=after["completion_percent"],
                    context_record_count=after["context_record_count"],
                )
                report["catalog_progress"].update(
                    completed_after=catalog_after["artifact_ready"],
                    pending_after=catalog_after["pending"],
                    completion_percent_after=catalog_after["completion_percent"],
                )
                manifest = artifacts.publish_manifest(
                    coverage=catalog_after,
                    active_song_ids=active_artifact_song_ids,
                )
                report["artifact_manifest_ref"] = "manifest.json"
                report["artifact_manifest"] = {
                    "song_count": manifest["song_count"],
                    "record_count": manifest["record_count"],
                    "sparse_profile_count": manifest["sparse_profile_count"],
                    "invalid_artifact_count": manifest["invalid_artifact_count"],
                    "orphan_artifact_count": manifest["orphan_artifact_count"],
                }
                publish_unresolved()
                artifacts.write_progress(progress_payload("completed"))
                interrupted = False
            finally:
                if fetcher is not None:
                    fetcher.close()
                if interrupted:
                    try:
                        after = _coverage(eligible_paths, artifacts)
                        catalog_after = _coverage(all_eligible_paths, artifacts)
                        report["progress"].update(
                            completed_after=after["artifact_ready"],
                            pending_after=after["pending"],
                            completion_percent_after=after["completion_percent"],
                            context_record_count=after["context_record_count"],
                        )
                        report["catalog_progress"].update(
                            completed_after=catalog_after["artifact_ready"],
                            pending_after=catalog_after["pending"],
                            completion_percent_after=catalog_after["completion_percent"],
                        )
                        artifacts.publish_manifest(
                            coverage=catalog_after,
                            active_song_ids=active_artifact_song_ids,
                        )
                        publish_unresolved()
                        artifacts.write_progress(progress_payload("interrupted"))
                    except Exception:
                        pass

            stored_report = {**report, "audits": audits, "finished_at": now()}
            ref = f"raw_runs/{digest([now(), stored_report])}.json"
            store.write(ref, stored_report)
            report["report_ref"] = ref
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append only useful Namuwiki trivia facts to existing raw song metadata."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("VAGUEFINDER_DATA_DIR", str(ROOT / "data/raw"))),
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/context")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=(
            Path(os.environ["VAGUEFINDER_CONTEXT_ARTIFACT_DIR"])
            if os.getenv("VAGUEFINDER_CONTEXT_ARTIFACT_DIR")
            else None
        ),
        help=(
            "Per-song derived context output. If omitted, it is resolved from "
            "--data-dir so temporary/test raw data cannot overwrite project artifacts."
        ),
    )
    parser.add_argument("--song-ids", nargs="+", help="Exact IDs to test; failed_raw is never scanned")
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Maximum processable songs this run. Full-catalog runs default to "
            f"{DEFAULT_CATALOG_RUN_LIMIT}; targeted --song-ids runs are uncapped."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only: no files, HTTP or browser")
    parser.add_argument("--url-map", type=Path, help="JSON object: song_id -> Namuwiki URL or URL list")
    parser.add_argument("--html-dir", type=Path, help="Saved <song_id>.html directory")
    parser.add_argument("--offline", action="store_true", help="Use saved HTML/cache only")
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Rebuild search artifacts from terminal meta.json context without crawling",
    )
    parser.add_argument("--retry-not-found", action="store_true")
    parser.add_argument(
        "--retry-needs-review",
        action="store_true",
        help="Retry v3 needs_review songs; normally they remain in unresolved.json",
    )
    parser.add_argument("--force", action="store_true", help="Refresh even a completed song")
    parser.add_argument("--refresh", action="store_true", help="Revalidate the HTTP cache")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--no-render", action="store_true", help="Disable Chromium fallback")
    parser.add_argument("--render-timeout", type=float, default=30.0)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Keep stderr quiet; the final JSON still contains progress",
    )
    return parser


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be positive")
        if args.refresh and (args.offline or args.html_dir):
            raise ValueError("refresh requires HTTP mode")
        if args.artifacts_only and (
            args.refresh
            or args.html_dir
            or args.retry_not_found
            or args.retry_needs_review
        ):
            raise ValueError(
                "artifacts-only cannot be combined with refresh, html-dir, "
                "retry-not-found or retry-needs-review"
            )
        result = run_backfill(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        has_error = any(
            row.get("action") in {"error", "artifact_error"}
            or (
                row.get("status") == "error"
                and row.get("action") not in {"deferred", "queued"}
            )
            for row in result["songs"]
        )
        invalid_artifacts = result.get("artifact_manifest", {}).get(
            "invalid_artifact_count",
            0,
        )
        return 1 if has_error or invalid_artifacts else 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
