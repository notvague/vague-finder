"""Lean collection pipeline: fetch, parse, verify, keep only trivia facts."""
from __future__ import annotations

from pathlib import Path

from .document_parser import parse_document
from .fetcher import Fetcher, article_url
from .schemas import (
    NAMUWIKI_META_VERSION,
    PARSER_VERSION,
    NamuwikiMeta,
    ParsedDocument,
    SongTarget,
    SourceSnapshot,
    now,
)
from .store import Store
from .trivia import bind_song_page, extract_trivia_facts


def parse_snapshot(store: Store, snapshot_ref: str) -> tuple[ParsedDocument, str]:
    snapshot = SourceSnapshot.model_validate(store.read(snapshot_ref))
    document = parse_document(
        store.raw(snapshot),
        snapshot.snapshot_id,
        snapshot.final_url or snapshot.requested_url,
    )
    document_ref = f"sources/{snapshot.document_id}/{snapshot.snapshot_id}/document_{PARSER_VERSION}.json"
    store.immutable(document_ref, document.model_dump())
    return document, document_ref


def _meta(status: str, *, source_url: str | None = None, facts=None, error_code: str | None = None) -> dict:
    value = NamuwikiMeta(
        schema_version=NAMUWIKI_META_VERSION,
        status=status,
        source_url=source_url,
        collected_at=now(),
        facts=facts or [],
        error_code=error_code,
    )
    return value.model_dump(exclude_none=True)


def collect_target(
    store: Store,
    fetcher: Fetcher,
    target: SongTarget,
    *,
    html_dir: Path | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> tuple[dict, dict]:
    """Collect one song and return (small meta projection, detailed audit)."""
    audit = {"song_id": target.song_id, "started_at": now(), "attempts": []}

    for raw_url in target.urls:
        url = article_url(raw_url)
        row = {"url": url}
        audit["attempts"].append(row)

        html_path = None
        if html_dir is not None:
            html_path = html_dir / f"{target.song_id}.html"
            if not html_path.is_file():
                row.update(status="pending", stage="input", error_code="saved_html_missing",
                           expected_html=str(html_path))
                continue

        attempt = fetcher.fetch(
            url,
            html_path=html_path,
            allow_http=not offline and html_dir is None,
            refresh=refresh,
        )
        row["fetch"] = attempt.model_dump()
        row["status"] = attempt.status
        if attempt.status != "ok":
            if fetcher.stopped or attempt.status in {"blocked", "rate_limited", "pending"}:
                break
            continue

        try:
            document, document_ref = parse_snapshot(store, attempt.snapshot_ref)
        except (ValueError, OSError) as exc:
            row.update(status="structure_error", stage="parse", error_code=type(exc).__name__)
            continue

        row.update(
            document_ref=document_ref,
            parse_status=document.parse_status,
            page_title=document.page_title,
            parser_warnings=document.diagnostics.get("warnings", []),
        )
        if document.parse_status == "structure_error":
            row.update(status="structure_error", stage="parse", error_code="document_structure_error")
            continue

        binding = bind_song_page(target, document)
        row["identity"] = binding.as_dict()
        if not binding.verified:
            row.update(status="needs_review", stage="identity", error_code=binding.reason)
            continue

        facts, trivia_section_found = extract_trivia_facts(
            document,
            scope_section_id=binding.scope_section_id,
        )
        source_url = document.canonical_url or attempt.final_url or url
        status = "ok" if facts else "no_trivia"
        row.update(
            status=status,
            stage="complete",
            source_url=source_url,
            binding_scope_section_id=binding.scope_section_id,
            trivia_section_found=trivia_section_found,
            fact_count=len(facts),
        )
        audit["finished_at"] = now()
        audit["status"] = status
        return _meta(status, source_url=source_url, facts=facts), audit

    audit["finished_at"] = now()
    rows = audit["attempts"]
    statuses = [row.get("status") for row in rows]
    if rows and all(status == "not_found" for status in statuses):
        status, code = "not_found", None
    elif any(status in {"needs_review", "structure_error"} for status in statuses):
        status = "needs_review"
        code = next(
            (row.get("error_code") for row in reversed(rows) if row.get("error_code")),
            "page_identity_or_structure_needs_review",
        )
    else:
        status = "error"
        code = next(
            (
                row.get("fetch", {}).get("error_code") or row.get("error_code")
                for row in reversed(rows)
                if row.get("fetch", {}).get("error_code") or row.get("error_code")
            ),
            "no_usable_source",
        )
    audit.update(status=status, error_code=code)
    # No candidate was identity-verified. Persisting the last attempted URL as
    # a source would turn an unrelated same-title page into the preferred URL
    # on the next run. Attempt URLs remain available in the detailed audit.
    return _meta(status, error_code=code), audit
