"""Build auditable hybrid-retrieval records from Namuwiki metadata.

Namuwiki facts intentionally stay out of the existing song-level mood/lyrics
passages. Dense retrieval uses one self-contained fact per vector, while sparse
retrieval uses one de-duplicated lexical profile per song. This prevents one
fact-rich song from flooding BM25 results with near-identical documents.

Every curated source fact remains traceable even when dependent or duplicate
facts are merged. The artifact stores exact future index inputs, not a second
copy of the raw crawler document.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.crawler.context.schemas import (
    Category,
    NAMUWIKI_META_VERSION,
    NamuwikiMeta,
    digest,
)
from src.crawler.context.trivia import (
    MAX_FACT_CHARS,
    clean_fact_text,
)
from src.embedding.text.context_bm25_tokenizer import tokenize_context_for_bm25

CONTEXT_RECORD_VERSION = "context_record_v6"
CONTEXT_ARTIFACT_VERSION = "context_artifact_v6"
DENSE_TEXT_FORMAT = "artist_title_category_fact_v1"
TERMINAL_CONTEXT_STATUSES = {"ok", "no_trivia", "not_found"}

MAX_RECORD_KEYWORDS = 12
MAX_SPARSE_PROFILE_TERMS = 80

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NumericId = Annotated[str, StringConstraints(pattern=r"^\d+$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RecordId = Annotated[str, StringConstraints(pattern=r"^nw:\d+:[0-9a-f]{20}$")]

# A category label gives the dense encoder a relation type without inventing
# facts. Generic media usage is deliberately not labelled OST/BGM.
CATEGORY_LABELS: dict[str, str] = {
    "production": "제작 배경",
    "media_usage": "매체 사용",
    "music_video": "뮤직비디오",
    "performance": "무대·공연",
    "meme": "밈·인터넷 유행",
    "record": "기록·수상",
    "influence": "영향·평가",
    "version": "버전·리메이크",
    "musical_detail": "음악적 특징·난이도",
    "other": "기타 배경 정보",
}

# Terms with little power to distinguish one song from another. Relation
# words such as 삽입/사용/제작/방영 are intentionally not included.
_SPARSE_STOP_TERMS = {
    "여담", "곡", "노래", "음악", "배경", "내용", "제목", "영상",
    "당시", "이후", "이전", "현재", "실제", "참고", "경우", "정도",
    "부분", "자체", "처음", "시청", "때", "말", "사람", "관련",
    "연유", "덕분", "위엄", "전", "후", "중", "하나", "본", "해당",
    "상당", "상당히", "바로", "생각", "이것", "그것", "사실", "식",
    "국내", "설명", "도중", "스스로", "시기", "존재", "가능",
    "본격", "반응", "마음", "기대", "동시", "자신", "자리", "유명",
    "사례", "담당", "사전", "자료", "공급", "추정", "주력", "흐름",
    "대다수", "채널", "상반기", "후반", "초반", "중반", "학년",
    "so", "sorry", "but", "love", "you",
}
_GENERIC_SECTION_NAMES = {"여담", "기타", "개요"}
_NUMBER_WITH_UNIT = re.compile(
    r"^(\d+)(?:기|화|회|편|위|년|월|일|번|차|주|세|명|개|옥타브)$"
)
_PLAIN_NUMBER = re.compile(r"^\d+$")
_LATIN_TERM = re.compile(r"^[a-z]+$")
_LATIN_WITH_PARTICLE = re.compile(
    r"^[a-z](?:만|은|는|이|가|을|를|과|와|로)$"
)
_KOREAN_TERM = re.compile(r"^[가-힣]{2,}$")
_RELATION_KEYWORD_ROOTS = (
    "사용", "삽입", "제작", "작곡", "작사", "녹음", "프로듀싱", "수록",
    "패러디", "합성", "수상", "선정", "유행", "공개", "발매", "기록",
    "등장", "방영", "연주", "리메이크", "평가", "차지", "선곡", "흐르",
    "만들", "부르", "선보", "종식", "재편", "삭제",
)

# Search records must stand on their own. These expressions point to context
# that is neither the song identity nor the section label, so a neighboring
# source fact is required. ``이 곡`` and ``본 곡`` are excluded: the compact
# embedding identity below resolves those without merging unrelated facts.
_DEPENDENT_SENTENCE_START = re.compile(
    r"^(?:라고|이러한|그런데|그러나|하지만|그리고|또한|한편|그렇게|"
    r"덕분에|심지어|의외로|그 결과|그로 인해|이때|그때|"
    r"이 때문에|이를 통해|영상뿐만 아니라|그래서(?:인지)?)(?:\s|[,，])"
)
_UNRESOLVED_REFERENCE = re.compile(
    r"(?:^|\s)(?:이|그|해당)\s*(?:음|음정|코드)"
    r"(?:은|는|이|가|을|를|에|에서|와|과|\s)"
)
_REFERENCE_ANCHORS = (
    (re.compile(r"영상뿐만 아니라|(?:이|그|해당)\s*영상"), ("영상",)),
    (re.compile(r"(?:이|그|해당)\s*(?:음|음정)"), ("최고음", "음정", "옥타브", "코드")),
    (re.compile(r"(?:이|그|해당)\s*코드"), ("코드", "원키")),
    (re.compile(r"(?:이|그|해당)\s*장면"), ("장면", "에피소드")),
)


@dataclass(frozen=True)
class _RetrievalUnit:
    source_fact_indices: tuple[int, ...]
    category: str
    section: str
    text: str
    quality: str = "standalone"


class ContextContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextRecord(ContextContract):
    record_id: RecordId
    source_fact_indices: list[int] = Field(min_length=1)
    category: Category
    section: Text
    evidence_text: Text
    dense_text: Text
    keywords: list[Text] = Field(default_factory=list, max_length=MAX_RECORD_KEYWORDS)
    quality: Literal[
        "standalone", "merged_context", "section_resolved", "context_dependent"
    ] = "standalone"

    @model_validator(mode="after")
    def consistent_record(self):
        if (
            any(index < 0 for index in self.source_fact_indices)
            or self.source_fact_indices != sorted(set(self.source_fact_indices))
        ):
            raise ValueError("source_fact_indices must be sorted, unique and non-negative")
        if len(self.evidence_text) > MAX_FACT_CHARS:
            raise ValueError("evidence_text exceeds the fact hard limit")
        if len(self.keywords) != len(set(self.keywords)):
            raise ValueError("keywords must be unique")
        return self


class ContextSparseProfile(ContextContract):
    profile_id: Annotated[str, StringConstraints(pattern=r"^nws:\d+$")]
    term_count: int = Field(ge=1, le=MAX_SPARSE_PROFILE_TERMS)
    terms: list[Text] = Field(min_length=1, max_length=MAX_SPARSE_PROFILE_TERMS)

    @model_validator(mode="after")
    def consistent_profile(self):
        if self.term_count != len(self.terms):
            raise ValueError("sparse profile term_count does not match terms")
        if len(self.terms) != len(set(self.terms)):
            raise ValueError("sparse profile terms must be unique")
        return self


class ContextSongIdentity(ContextContract):
    song_id: NumericId
    title: Text
    artists: list[Text] = Field(min_length=1)


class ContextSource(ContextContract):
    schema_version: Literal["namuwiki_v3"] = NAMUWIKI_META_VERSION
    digest: Sha256
    url: str | None = None
    collected_at: Text


class ContextRetrievalSector(ContextContract):
    record_schema_version: Literal["context_record_v6"] = CONTEXT_RECORD_VERSION
    dense_embedding_unit: Literal["fact"] = "fact"
    dense_text_format: Literal[
        "artist_title_category_fact_v1"
    ] = DENSE_TEXT_FORMAT
    sparse_embedding_unit: Literal["song"] = "song"
    score_aggregation: Literal["max_per_song"] = "max_per_song"
    source_fact_count: int = Field(ge=0)
    covered_source_fact_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    sparse_profile: ContextSparseProfile | None = None
    records: list[ContextRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_sector(self):
        if self.record_count != len(self.records):
            raise ValueError("record_count does not match records")
        if len({record.record_id for record in self.records}) != len(self.records):
            raise ValueError("duplicate context record_id")
        covered = [index for record in self.records for index in record.source_fact_indices]
        if len(covered) != len(set(covered)):
            raise ValueError("a source fact may belong to only one retrieval record")
        if any(index >= self.source_fact_count for index in covered):
            raise ValueError("source_fact_indices exceed source_fact_count")
        if self.covered_source_fact_count != len(set(covered)):
            raise ValueError("covered_source_fact_count does not match records")
        if self.covered_source_fact_count != self.source_fact_count:
            raise ValueError("context artifacts must not silently omit source facts")
        if self.records and self.sparse_profile is None:
            raise ValueError("searchable context records require one song sparse profile")
        if not self.records and self.sparse_profile is not None:
            raise ValueError("empty context cannot contain a sparse profile")
        return self


class ContextSongArtifact(ContextContract):
    artifact_schema_version: Literal["context_artifact_v6"] = CONTEXT_ARTIFACT_VERSION
    artifact_content_sha256: Sha256
    status: Literal["ok", "no_trivia", "not_found"]
    song: ContextSongIdentity
    source: ContextSource
    retrieval: ContextRetrievalSector

    @model_validator(mode="after")
    def consistent_artifact(self):
        records = self.retrieval.records
        if self.status == "ok" and not records:
            raise ValueError("ok context artifact requires records")
        if self.status != "ok" and records:
            raise ValueError("only ok context artifacts may contain records")
        expected_prefix = f"nw:{self.song.song_id}:"
        if any(not record.record_id.startswith(expected_prefix) for record in records):
            raise ValueError("context record_id belongs to another song")
        dense_identity = f"{' / '.join(self.song.artists)} - {self.song.title}"
        for record in records:
            expected_dense = (
                f"{dense_identity} | {CATEGORY_LABELS[record.category]}: "
                f"{record.evidence_text}"
            )
            if record.dense_text != expected_dense:
                raise ValueError(
                    "dense_text must contain the declared song identity, category and evidence"
                )
        expected_content_hash = _artifact_content_digest(
            status=self.status,
            song=self.song,
            source=self.source,
            retrieval=self.retrieval,
        )
        if self.artifact_content_sha256 != expected_content_hash:
            raise ValueError("artifact_content_sha256 does not match artifact content")
        return self


def _text(value) -> str:
    return str(value).strip() if value is not None else ""


def _identity(song: Mapping) -> tuple[str, str, list[str]]:
    song_id = _text(song.get("id") or song.get("song_id"))
    metadata = song.get("metadata") if isinstance(song.get("metadata"), Mapping) else {}
    title = _text(metadata.get("title"))
    artists_value = metadata.get("artist", [])
    if isinstance(artists_value, str):
        artists = [artists_value.strip()] if artists_value.strip() else []
    elif isinstance(artists_value, list):
        artists = [_text(value) for value in artists_value if _text(value)]
    else:
        artists = []
    if not song_id.isdigit() or not title or not artists:
        raise ValueError("numeric song_id, metadata.title and metadata.artist are required")
    return song_id, title, artists


def _context(song: Mapping) -> NamuwikiMeta:
    value = song.get("namuwiki")
    if not isinstance(value, Mapping):
        raise ValueError("current compact namuwiki metadata is required")
    # Persistent artifacts must not manufacture a new timestamp on each load.
    if not _text(value.get("collected_at")):
        raise ValueError("namuwiki.collected_at must be explicit for deterministic artifacts")
    context = NamuwikiMeta.model_validate(value)
    if context.status not in TERMINAL_CONTEXT_STATUSES:
        raise ValueError("only terminal Namuwiki metadata can produce artifacts")
    return context


def _section_parts(section: str) -> list[str]:
    parts = [part.strip() for part in section.split(">") if part.strip()]
    if not parts:
        raise ValueError("fact section is empty")
    return parts


def _section_hint(section_path: list[str]) -> str:
    leaf = section_path[-1]
    normalized = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", leaf).strip()
    return "" if normalized in _GENERIC_SECTION_NAMES else normalized


def _identity_aliases(title: str, artists: list[str]) -> list[str]:
    """Return bounded title/artist spellings without an external alias table."""
    aliases: list[str] = [title]
    for raw in artists:
        base = re.sub(r"\([^)]*\)", "", raw).strip()
        if base:
            aliases.append(base)
        aliases.extend(
            part.strip()
            for part in re.findall(r"\(([^)]*)\)", raw)
            if part.strip()
        )
        aliases.extend(
            part.strip()
            for part in re.split(r"[/·|]", raw)
            if part.strip() and "(" not in part and ")" not in part
        )
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _identity_sparse_terms(title: str, artists: list[str]) -> list[str]:
    aliases = _identity_aliases(title, artists)
    terms: list[str] = []
    # Identity is a compact discriminator, not another bag of generic words.
    # Store exact normalized spellings only: morphology would reduce the title
    # ``사랑했나봐`` to the corpus-wide term ``사랑``.
    for alias in aliases:
        normalized = unicodedata.normalize("NFKC", alias).strip().lower()
        normalized = re.sub(r"[^0-9a-z가-힣#\-]+", "_", normalized)
        normalized = normalized.strip("_")
        if (
            normalized
            and len(normalized.replace("_", "")) <= 50
            and normalized not in terms
        ):
            terms.append(normalized)
    return list(dict.fromkeys(term for term in terms if term))


def _needs_antecedent(text: str) -> bool:
    return bool(
        _DEPENDENT_SENTENCE_START.search(text)
        or _UNRESOLVED_REFERENCE.search(text)
    )


def _reference_terms(text: str) -> tuple[str, ...]:
    for pattern, anchors in _REFERENCE_ANCHORS:
        if pattern.search(text):
            return anchors
    return ()


def _join_units(units: list[_RetrievalUnit]) -> _RetrievalUnit:
    first = units[0]
    return _RetrievalUnit(
        source_fact_indices=tuple(sorted({
            index for unit in units for index in unit.source_fact_indices
        })),
        category=first.category,
        section=first.section,
        text=clean_fact_text(" ".join(unit.text for unit in units)),
        quality="merged_context",
    )


def _section_subject(section: str) -> str:
    """Extract a concrete media title from a specific section heading."""
    leaf = _section_hint(_section_parts(section))
    if not leaf:
        return ""
    subject = re.sub(
        r"\s*(?:국내판\s*)?(?:삽입곡|배경\s*음악|BGM|OST)\s*$",
        "",
        leaf,
        flags=re.I,
    ).strip()
    if not subject or subject == leaf or len(subject) > 50:
        return ""
    return subject


def _resolve_section_antecedent(unit: _RetrievalUnit) -> _RetrievalUnit:
    """Make a bounded ``이때`` fact stand alone using its own section title."""
    if unit.category != "media_usage":
        return unit
    match = re.match(r"^(?:이때|그때)(?:는|에도)?\s*[,，]?\s*", unit.text)
    if not match:
        return unit
    subject = _section_subject(unit.section)
    remainder = unit.text[match.end():].strip()
    if not subject or not remainder:
        return unit
    resolved_text = clean_fact_text(f"{subject}에서 {remainder}")
    if len(resolved_text) > MAX_FACT_CHARS:
        return unit
    return _RetrievalUnit(
        source_fact_indices=unit.source_fact_indices,
        category=unit.category,
        section=unit.section,
        text=resolved_text,
        quality="section_resolved",
    )


def _prepare_retrieval_units(context: NamuwikiMeta) -> list[_RetrievalUnit]:
    """Keep each curated fact whole and resolve only true antecedent links."""
    atomic: list[_RetrievalUnit] = []
    for index, fact in enumerate(context.facts):
        category = _text(fact.category)
        section = _text(fact.section)
        text = clean_fact_text(fact.text)
        if not text or len(text) > MAX_FACT_CHARS:
            raise ValueError(f"invalid source fact at index {index}")
        atomic.append(_RetrievalUnit((index,), category, section, text))

    resolved: list[_RetrievalUnit] = []
    for original in atomic:
        unit = _resolve_section_antecedent(original)
        if not _needs_antecedent(unit.text):
            resolved.append(unit)
            continue

        same_group_positions = [
            position
            for position in range(max(0, len(resolved) - 4), len(resolved))
            if resolved[position].category == unit.category
            and resolved[position].section == unit.section
        ]
        anchors = _reference_terms(unit.text)
        anchor_position: int | None = None
        if anchors:
            for position in reversed(same_group_positions):
                if any(anchor in resolved[position].text for anchor in anchors):
                    anchor_position = position
                    break
        elif same_group_positions:
            anchor_position = same_group_positions[-1]

        if anchor_position is None:
            resolved.append(_RetrievalUnit(
                unit.source_fact_indices, unit.category, unit.section, unit.text,
                "context_dependent",
            ))
            continue
        combined = _join_units([resolved[anchor_position], unit])
        if len(combined.text) <= MAX_FACT_CHARS:
            resolved[anchor_position] = combined
        else:
            resolved.append(_RetrievalUnit(
                unit.source_fact_indices, unit.category, unit.section, unit.text,
                "context_dependent",
            ))
    return resolved


def _record_id(song_id: str, category: str, section: str, fact_text: str) -> str:
    value = hashlib.sha256(
        f"{song_id}\0{category}\0{section}\0{fact_text}".encode("utf-8")
    ).hexdigest()
    return f"nw:{song_id}:{value[:20]}"


def context_source_digest(song: Mapping) -> str:
    """Cheap v3 change detector used before tokenization."""
    song_id, title, artists = _identity(song)
    context = _context(song)
    return digest(
        {
            "artifact_schema_version": CONTEXT_ARTIFACT_VERSION,
            "record_schema_version": CONTEXT_RECORD_VERSION,
            "source_schema_version": NAMUWIKI_META_VERSION,
            "song_id": song_id,
            "title": title,
            "artists": artists,
            "namuwiki": context.model_dump(mode="json", exclude_none=True),
        }
    )


def _artifact_content_digest(*, status: str, song, source, retrieval) -> str:
    def dump(value):
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        return dict(value)

    return digest(
        {
            "artifact_schema_version": CONTEXT_ARTIFACT_VERSION,
            "status": status,
            "song": dump(song),
            "source": dump(source),
            "retrieval": dump(retrieval),
        }
    )


def _identity_terms(title: str, artists: list[str]) -> set[str]:
    terms = set(_identity_sparse_terms(title, artists))
    # Morphological aliases are used only to remove identity repetition from a
    # fact passage. They are deliberately not emitted as extra sparse aliases.
    for alias in _identity_aliases(title, artists):
        terms.update(tokenize_context_for_bm25(alias, expand_synonyms=False))
    return terms


def _clean_sparse_terms(terms: list[str], *, identity_terms: set[str]) -> list[str]:
    """Remove repetition and corpus-wide noise without changing source facts."""
    normalized = [
        re.sub(
            r"^[^0-9a-z가-힣#&_\-]+|[^0-9a-z가-힣#&_\-]+$",
            "",
            unicodedata.normalize("NFKC", term).strip().lower(),
        )
        for term in terms
    ]
    normalized = [term for term in normalized if term]
    # Surface recovery may retain a copular ending (배경음악이라) while Kiwi
    # also emits the clean noun (배경음악). Prefer the exact noun.
    raw_terms = set(normalized)
    canonicalized: list[str] = []
    for term in normalized:
        for suffix in (
            "이라는", "이라고", "에서는", "으로는", "에게서", "까지는",
            "부터는", "라는", "라고", "이라", "에게", "한테", "으로",
            "부터", "까지", "에서", "은", "는", "이", "가", "을", "를",
            "의", "에", "와", "과", "로", "도", "만",
        ):
            base = term[:-len(suffix)] if term.endswith(suffix) else ""
            if len(base) >= 2 and base in raw_terms:
                term = base
                break
        canonicalized.append(term)
    normalized = canonicalized
    compounds = [term for term in normalized if "_" in term or "-" in term]
    compound_parts = {
        part
        for compound in compounds
        for part in re.split(r"[_-]", compound)
        if len(part) > 1
    }
    numbered = {
        match.group(1)
        for term in normalized
        for match in [_NUMBER_WITH_UNIT.fullmatch(term)]
        if match
    }
    result: list[str] = []
    seen: set[str] = set()
    seen_spacing_variants: set[str] = set()
    for term in normalized:
        spacing_key = term.replace("_", "")
        if (
            term in seen
            or spacing_key in seen_spacing_variants
            or term in identity_terms
            or term in _SPARSE_STOP_TERMS
        ):
            continue
        if term in compound_parts and "_" not in term:
            continue
        if _LATIN_TERM.fullmatch(term) and any(
            term in part and term != part for part in compound_parts
        ):
            continue
        if _LATIN_WITH_PARTICLE.fullmatch(term):
            continue
        if _KOREAN_TERM.fullmatch(term) and (
            any(
                term in part and len(part) > len(term)
                for part in compound_parts
            )
            or any(
                term in other
                and len(other) > len(term)
                and _KOREAN_TERM.fullmatch(other)
                for other in normalized
            )
        ):
            continue
        # A bare number is weaker than 6기/15화/2005년. Keep long standalone
        # identifiers such as karaoke catalogue number 46034.
        if _PLAIN_NUMBER.fullmatch(term):
            if term in numbered or len(term) < 4:
                continue
        # Kiwi can leave isolated chord letters after D-E-F#m. The protected
        # compound is more useful and avoids four noisy one-character terms.
        if len(term) == 1 and term.isalpha():
            continue
        if term == "옥타브" and any(value.endswith("옥타브") for value in normalized):
            continue
        seen.add(term)
        seen_spacing_variants.add(spacing_key)
        result.append(term)
    return result


def _keyword_priority(term: str) -> int:
    """Rank exact anchors ahead of ordinary nouns without corpus statistics."""
    if "_" in term:
        return 60
    if _NUMBER_WITH_UNIT.fullmatch(term) or any(char.isdigit() for char in term):
        return 55
    if "-" in term:
        return 50
    if re.fullmatch(r"[a-z][a-z0-9#&]*", term):
        return 45
    if any(root in term for root in _RELATION_KEYWORD_ROOTS):
        return 40
    if _KOREAN_TERM.fullmatch(term) and 2 <= len(term) <= 12:
        return 30
    return 20


def _select_keywords(terms: list[str]) -> list[str]:
    """Keep a bounded, high-precision explanation set for one dense fact."""
    ranked = sorted(
        enumerate(terms),
        key=lambda value: (-_keyword_priority(value[1]), value[0]),
    )
    selected = {index for index, _ in ranked[:MAX_RECORD_KEYWORDS]}
    return [term for index, term in enumerate(terms) if index in selected]


def _build_sparse_profile(
    *,
    song_id: str,
    title: str,
    artists: list[str],
    context: NamuwikiMeta,
    records: list[ContextRecord],
) -> ContextSparseProfile | None:
    """Build one de-duplicated lexical document for the whole song.

    Song identity and section anchors occur once. Remaining slots are filled
    round-robin from fact keywords so early or unusually long facts cannot
    consume the entire sparse budget.
    """
    if not records:
        return None

    terms: list[str] = []
    seen: set[str] = set()

    def add(values) -> None:
        for value in values:
            if value and value not in seen and len(terms) < MAX_SPARSE_PROFILE_TERMS:
                terms.append(value)
                seen.add(value)

    identity = _identity_sparse_terms(title, artists)
    add(identity)

    section_terms: list[str] = []
    seen_sections: set[str] = set()
    for fact in context.facts:
        section = _text(fact.section)
        if section in seen_sections:
            continue
        seen_sections.add(section)
        hint = _section_hint(_section_parts(section))
        if not hint:
            continue
        cleaned = _clean_sparse_terms(
            tokenize_context_for_bm25(hint, expand_synonyms=False),
            identity_terms=set(identity),
        )
        section_terms.extend(_select_keywords(cleaned)[:4])
    add(section_terms)

    for offset in range(MAX_RECORD_KEYWORDS):
        add(
            record.keywords[offset]
            for record in records
            if offset < len(record.keywords)
        )
        if len(terms) >= MAX_SPARSE_PROFILE_TERMS:
            break

    if not terms:
        raise ValueError("ok context produced an empty sparse profile")
    return ContextSparseProfile(
        profile_id=f"nws:{song_id}",
        term_count=len(terms),
        terms=terms,
    )


def _build_record(
    *,
    song_id: str,
    title: str,
    artists: list[str],
    unit: _RetrievalUnit,
) -> ContextRecord:
    category = unit.category
    section = unit.section
    fact_text = clean_fact_text(unit.text)
    section_path = _section_parts(section)
    if category not in CATEGORY_LABELS:
        raise ValueError(f"unsupported context category: {category}")
    # Each fact is embedded independently. Metadata stored beside the vector
    # is invisible to KoE5, so give every passage the same compact identity.
    # This also stops a fact that merely repeats the artist name from beating
    # the actually relevant fact within the same song.
    identity = f"{' / '.join(artists)} - {title}"
    dense = f"{identity} | {CATEGORY_LABELS[category]}: {fact_text}"

    body_terms = _clean_sparse_terms(
        tokenize_context_for_bm25(fact_text, expand_synonyms=False),
        identity_terms=_identity_terms(title, artists),
    )
    keywords = _select_keywords(body_terms)

    serialized_section = " > ".join(section_path)
    return ContextRecord(
        record_id=_record_id(song_id, category, serialized_section, fact_text),
        source_fact_indices=list(unit.source_fact_indices),
        category=category,
        section=serialized_section,
        evidence_text=fact_text,
        dense_text=dense,
        keywords=keywords,
        quality=unit.quality,
    )


def _records_for_context(
    *,
    song_id: str,
    title: str,
    artists: list[str],
    context: NamuwikiMeta,
) -> list[ContextRecord]:
    records: list[ContextRecord] = []
    positions: dict[str, int] = {}
    if context.status == "ok":
        for unit in _prepare_retrieval_units(context):
            record = _build_record(
                song_id=song_id, title=title, artists=artists, unit=unit,
            )
            if record.record_id in positions:
                position = positions[record.record_id]
                previous = records[position]
                records[position] = previous.model_copy(update={
                    "source_fact_indices": sorted(set(
                        previous.source_fact_indices + record.source_fact_indices
                    ))
                })
                continue
            positions[record.record_id] = len(records)
            records.append(record)
    return records


def build_namuwiki_song_artifact(song: Mapping) -> dict:
    """Build one deterministic, lossless v5 artifact for a terminal v3 song."""
    song_id, title, artists = _identity(song)
    context = _context(song)
    records = _records_for_context(
        song_id=song_id,
        title=title,
        artists=artists,
        context=context,
    )
    covered = {index for record in records for index in record.source_fact_indices}

    source = ContextSource(
        digest=context_source_digest(song),
        url=context.source_url,
        collected_at=context.collected_at,
    )
    identity = ContextSongIdentity(song_id=song_id, title=title, artists=artists)
    sparse_profile = _build_sparse_profile(
        song_id=song_id,
        title=title,
        artists=artists,
        context=context,
        records=records,
    )
    retrieval = ContextRetrievalSector(
        source_fact_count=len(context.facts),
        covered_source_fact_count=len(covered),
        record_count=len(records),
        sparse_profile=sparse_profile,
        records=records,
    )
    artifact_status = context.status
    artifact = ContextSongArtifact(
        artifact_content_sha256=_artifact_content_digest(
            status=artifact_status,
            song=identity,
            source=source,
            retrieval=retrieval,
        ),
        status=artifact_status,
        song=identity,
        source=source,
        retrieval=retrieval,
    )
    return artifact.model_dump(mode="json", exclude_none=True)


def embedding_record_projection(
    artifact: ContextSongArtifact,
    record: ContextRecord,
) -> dict:
    """Flatten one fact-level dense document and its audit metadata."""
    return {
        "record_schema_version": CONTEXT_RECORD_VERSION,
        "record_id": record.record_id,
        "song_id": artifact.song.song_id,
        "title": artifact.song.title,
        "artists": artifact.song.artists,
        "category": record.category,
        "section": record.section,
        "source_fact_indices": record.source_fact_indices,
        "quality": record.quality,
        "fact_text": record.evidence_text,
        "dense_text": record.dense_text,
        "keywords": record.keywords,
        "dense_passage": record.dense_text,
        "evidence_passage": record.evidence_text,
        "source_url": artifact.source.url,
        "collected_at": artifact.source.collected_at,
    }


def sparse_profile_projection(artifact: ContextSongArtifact) -> dict | None:
    """Flatten the one-per-song sparse document used by context BM25."""
    profile = artifact.retrieval.sparse_profile
    if profile is None:
        return None
    return {
        "record_schema_version": CONTEXT_RECORD_VERSION,
        "profile_id": profile.profile_id,
        "song_id": artifact.song.song_id,
        "title": artifact.song.title,
        "artists": artifact.song.artists,
        "sparse_terms": profile.terms,
        "sparse_passage": " ".join(profile.terms),
        "source_url": artifact.source.url,
        "collected_at": artifact.source.collected_at,
    }


def iter_namuwiki_fact_records(song: Mapping) -> Iterator[dict]:
    """Yield clean embedding projections; malformed optional context is ignored."""
    try:
        song_id, title, artists = _identity(song)
        value = song.get("namuwiki")
        if not isinstance(value, Mapping):
            return
        # This read-only compatibility path may receive old in-memory values
        # without an explicit timestamp. Persistent artifacts never do.
        context = NamuwikiMeta.model_validate(value)
    except (TypeError, ValueError):
        return
    if context.status != "ok":
        return

    source = ContextSource(
        digest=digest(context.model_dump(mode="json", exclude_none=True)),
        url=context.source_url,
        collected_at=context.collected_at,
    )
    identity = ContextSongIdentity(song_id=song_id, title=title, artists=artists)
    records = _records_for_context(
        song_id=song_id,
        title=title,
        artists=artists,
        context=context,
    )
    sparse_profile = _build_sparse_profile(
        song_id=song_id,
        title=title,
        artists=artists,
        context=context,
        records=records,
    )
    retrieval = ContextRetrievalSector(
        source_fact_count=len(context.facts),
        covered_source_fact_count=len({
            index for record in records for index in record.source_fact_indices
        }),
        record_count=len(records),
        sparse_profile=sparse_profile,
        records=records,
    )
    artifact_status = context.status
    artifact = ContextSongArtifact(
        artifact_content_sha256=_artifact_content_digest(
            status=artifact_status,
            song=identity,
            source=source,
            retrieval=retrieval,
        ),
        status=artifact_status,
        song=identity,
        source=source,
        retrieval=retrieval,
    )
    for record in records:
        yield embedding_record_projection(artifact, record)
