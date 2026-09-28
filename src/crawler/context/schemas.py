"""Small contracts for collecting and parsing Namuwiki song trivia.

Only raw snapshots and parsed documents use these detailed contracts. The
song's ``meta.json`` receives the deliberately small ``namuwiki`` projection
defined by :class:`NamuwikiMeta`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION = "context_v1"
# A new parser version prevents an old immutable parse result from colliding
# with the trivia-only pipeline when an existing HTML snapshot is reused.
PARSER_VERSION = "namuwiki_dom_v3"
NAMUWIKI_META_VERSION = "namuwiki_v3"

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FactText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=400),
]
Category = Literal[
    "production",
    "media_usage",
    "music_video",
    "performance",
    "meme",
    "record",
    "influence",
    "version",
    "musical_detail",
    "other",
]
NamuwikiStatus = Literal[
    "ok",
    "no_trivia",
    "not_found",
    "needs_review",
    "error",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SongTarget(Contract):
    song_id: Annotated[str, StringConstraints(pattern=r"^\d+$")]
    title: Text
    artists: list[Text] = Field(min_length=1)
    # Binding uses the catalogue identity as independent evidence.  These
    # values never become search facts; they only prevent a same-title page or
    # a neighbouring track section from being attached to the wrong song.
    album: str | None = None
    release_year: int | None = None
    title_aliases: list[Text] = Field(default_factory=list, max_length=6)
    # Automatic discovery tries artist-qualified titles first, followed by
    # NamuWiki's common ``<title>(노래)`` disambiguator and the bare title.
    # A cleaned title (for example, with ``Feat.`` or ``Live Ver.`` removed)
    # may add three fallback candidates.  The list stays deliberately bounded
    # so one song cannot consume an unbounded crawl budget.
    urls: list[Text] = Field(min_length=1, max_length=8)
    explicit_urls: bool = False


class NamuwikiFact(Contract):
    category: Category
    section: Text
    text: FactText


class NamuwikiMeta(Contract):
    schema_version: Literal["namuwiki_v3"] = NAMUWIKI_META_VERSION
    status: NamuwikiStatus
    source_url: str | None = None
    collected_at: str = Field(default_factory=now)
    facts: list[NamuwikiFact] = Field(default_factory=list)
    # Only unsuccessful states expose one compact reason. Full fetch/parse
    # diagnostics stay in data/context/raw_runs, not in each song metadata file.
    error_code: str | None = None

    @model_validator(mode="after")
    def consistent_projection(self):
        if self.status == "ok" and not self.facts:
            raise ValueError("ok Namuwiki metadata requires at least one fact")
        if self.status != "ok" and self.facts:
            raise ValueError("only ok Namuwiki metadata may contain facts")
        if self.status in {"needs_review", "error"} and not self.error_code:
            raise ValueError("unsuccessful Namuwiki metadata requires error_code")
        if self.status in {"ok", "no_trivia"} and not self.source_url:
            raise ValueError("resolved Namuwiki page requires source_url")
        return self


class Link(Contract):
    text: str
    url: str


class SourceSnapshot(Contract):
    schema_version: str = SCHEMA_VERSION
    snapshot_id: str
    document_id: str
    source_type: Literal["namuwiki"] = "namuwiki"
    requested_url: str
    final_url: str | None = None
    canonical_url: str | None = None
    page_title: str | None = None
    acquisition_method: Literal["http", "user_saved_html", "browser_rendered"]
    parent_snapshot_ref: str | None = None
    fetched_at: str
    http_status: int | None = None
    revision_id: str | None = None
    raw_sha256: str
    raw_html_ref: str
    content_type: str | None = None
    redirects: list[str] = Field(default_factory=list)
    etag: str | None = None
    last_modified: str | None = None
    fetch_status: Literal["ok"] = "ok"


class FetchAttempt(Contract):
    requested_url: str
    acquisition_method: Literal["http", "user_saved_html", "browser_rendered"]
    stage: str = "fetch"
    initial_snapshot_ref: str | None = None
    browser_requests: int = 0
    status: Literal[
        "ok",
        "not_found",
        "blocked",
        "rate_limited",
        "network_error",
        "invalid_response",
        "pending",
    ]
    checked_at: str = Field(default_factory=now)
    http_status: int | None = None
    final_url: str | None = None
    redirects: list[str] = Field(default_factory=list)
    error_code: str | None = None
    retry_after: str | None = None
    retry_at: str | None = None
    requests_made: int = 0
    cache_hit: bool = False
    snapshot_ref: str | None = None


class Section(Contract):
    section_id: str
    parent_id: str | None
    heading: str
    level: int
    own_block_ids: list[str] = Field(default_factory=list)


class Block(Contract):
    block_id: str
    section_id: str
    order: int
    block_type: Literal["paragraph", "list_item", "table_row", "quote"]
    source_text: str
    normalized_text: str
    links: list[Link] = Field(default_factory=list)
    footnote_refs: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class Footnote(Contract):
    footnote_id: str
    source_text: str
    links: list[Link] = Field(default_factory=list)


class ParsedDocument(Contract):
    snapshot_id: str
    parser_version: str = PARSER_VERSION
    parse_status: Literal["ok", "partial", "structure_error"]
    page_title: str | None = None
    canonical_url: str | None = None
    article_content_hash: str
    sections: list[Section]
    blocks: list[Block]
    footnotes: list[Footnote]
    diagnostics: dict = Field(default_factory=dict)
