"""Strict BM25 tokenizer for factual song context.

This intentionally differs from the existing mood/lyrics tokenizer.  Context
records retain concrete nouns, names, English/numbers and a small allow-list of
relation verbs.  Synonyms are expanded only in this context path.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from itertools import groupby
from typing import List, Tuple

from kiwipiepy import Kiwi

CONTEXT_BM25_TOKENIZER_VERSION = "context_bm25_tokenizer_v1"

_KEEP_TAGS = {"NNG", "NNP", "SL", "SN", "SH"}
_SPECIAL_TAGS = {"W_HASHTAG", "W_SERIAL"}
_VERB_TAGS = {"VV", "XSV"}

_RELATION_ROOTS = {
    "사용", "삽입", "제작", "작곡", "작사", "녹음", "프로듀싱", "수록",
    "패러디", "합성", "수상", "선정", "유행", "공개", "발매", "기록",
    "등장", "방영", "연주", "리메이크", "평가", "차지", "재생", "선곡",
    "흐르", "쓰이", "알려지", "만들", "부르", "선보이", "영향",
}
_RELATION_LEMMAS = {
    "사용하다", "사용되다", "삽입하다", "삽입되다", "제작하다", "제작되다",
    "작곡하다", "작사하다", "녹음하다", "프로듀싱하다", "수록하다", "수록되다",
    "패러디하다", "합성하다", "수상하다", "선정하다", "유행하다", "공개하다",
    "발매하다", "기록하다", "등장하다", "방영하다", "연주하다", "리메이크하다",
    "평가하다", "차지하다", "재생하다", "선곡하다", "흐르다", "쓰이다",
    "알려지다", "만들다", "부르다", "선보이다", "영향받다",
}

# Frequently occurring multi-word entities that ordinary morphology splits too
# aggressively. Quoted multi-word names are protected generically below too.
_PROTECTED_PHRASES = (
    "짱구는 못말려",
    "리그 오브 레전드",
    "NEW 아기공룡 둘리",
    "아기공룡 둘리",
    "미스터리 음악쇼 복면가왕",
    "불후의 명곡",
    "펌프 잇 업",
    "강남스타일",
)
_LITERAL_COMPOUNDS = (
    "배경음악",
    "뮤직비디오",
    "삽입곡",
    "대표곡",
    "히트곡",
    "대중음악",
    "음원차트",
    "실시간검색어",
)
_QUOTED_PHRASE = re.compile(r"[〈《「『\"“']([^〉》」』\"”']{2,60})[〉》」』\"”']")
_MAX_QUOTED_COMPOUND_CHARS = 36
_MAX_QUOTED_COMPOUND_WORDS = 5
_NUMBERED_UNIT = re.compile(
    r"(?<!\d)(\d{1,4})\s*(기|화|회|편|위|년|월|일|번|차|주|세(?!대)|명|개)"
)
_OCTAVE = re.compile(r"(?<!\d)(\d+)\s*옥타브")
_PITCH = re.compile(r"(?<![A-Za-z0-9])([A-Ga-g][#b]?\d)(?![A-Za-z0-9])")
_CHORD = r"[A-Ga-g](?:[#♯b♭])?(?:maj|min|sus|dim|aug|m)?\d*"
_CHORD_SEQUENCE = re.compile(
    rf"(?<![A-Za-z])({_CHORD}(?:\s*-\s*{_CHORD})+)(?![A-Za-z0-9#♯b♭])"
)
_HYPHENATED = re.compile(
    # One-letter segments are valid in artist names (G-DRAGON) and musical
    # notation (D-E-F#m).  The old {2,} rule silently reduced G-DRAGON to
    # the generic token ``dragon``.
    r"(?<![0-9A-Za-z])([0-9A-Za-z]+"
    r"(?:\s*-\s*[0-9A-Za-z#♯♭b♭]+)+)"
    r"(?![0-9A-Za-z#♯♭b♭])"
)
_NAMED_ROLE_PHRASE = re.compile(
    r"(?<![가-힣])([가-힣]{2,12})\s+"
    r"(선생님|감독|작곡가|작사가|프로듀서|가수|래퍼)"
)

# Surface-word recovery is deliberately conservative.  Kiwi occasionally
# splits a new proper name into one-character common nouns (e.g. 메타톤
# 잼민이).  We preserve the original word and join an NNP-headed word with
# the following name-like word, but never generate arbitrary sentence n-grams.
_PARTICLE_SUFFIXES = tuple(sorted({
    "으로부터", "에게서", "까지는", "부터는", "이라는", "라는",
    "에서는", "으로는", "에게", "한테", "으로", "부터", "까지",
    "이라고", "라고", "이라", "은", "는", "이", "가", "을", "를", "의", "에", "에서", "와", "과", "로", "도", "만",
}, key=len, reverse=True))
_ENTITY_ROLE_WORDS = {
    "선생님", "감독", "작곡가", "작사가", "프로듀서", "가수", "래퍼",
    "그룹", "밴드", "프로게임단", "사건", "축제", "대회", "프로그램",
}

_SYNONYM_GROUPS = (
    (re.compile(r"\bOST\b|삽입곡|배경\s*음악|\bBGM\b", re.I),
     ("ost", "삽입곡", "배경음악", "bgm")),
    (re.compile(r"뮤직\s*비디오|뮤비|\bMV\b", re.I),
     ("뮤직비디오", "뮤비", "mv")),
    (re.compile(r"밈|패러디|합성|챌린지"),
     ("밈", "패러디", "합성", "챌린지")),
    (re.compile(r"무대|공연|경연|라이브"),
     ("무대", "공연", "경연", "라이브")),
    (re.compile(r"제작|작곡|녹음|데모"),
     ("제작", "작곡", "녹음", "데모")),
)


def _base_tag(tag: str) -> str:
    return tag.split("-", 1)[0]


def _lemma(form: str) -> str:
    value = form.strip().lower()
    return value if not value or value.endswith("다") else value + "다"


def _is_relation(term: str) -> bool:
    return term in _RELATION_LEMMAS or any(root in term for root in _RELATION_ROOTS)


@lru_cache(maxsize=1)
def _get_kiwi() -> Kiwi:
    kiwi = Kiwi()
    for phrase in _PROTECTED_PHRASES:
        # This helps contiguous names; the underscore projection below remains
        # authoritative for names that contain spaces or particles.
        if " " not in phrase:
            kiwi.add_user_word(phrase, "NNP", 0.0)
    return kiwi


def _normalized_term(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower().lstrip("#@")
    value = re.sub(r"[\s/·|]+", "_", value)
    value = re.sub(r"[^0-9a-z가-힣#♯♭&_'’-]+", "", value)
    return value.strip("_")


def _quoted_compound(value: str) -> str | None:
    """Keep short names/titles, not an entire quoted lyric sentence."""
    value = re.sub(r"\s+", " ", value.strip())
    words = value.split()
    if (
        not 2 <= len(value) <= _MAX_QUOTED_COMPOUND_CHARS
        or not 2 <= len(words) <= _MAX_QUOTED_COMPOUND_WORDS
        or re.search(r"[,;:!?，；：！？。]", value)
    ):
        return None
    return _normalized_term(value)


def _strip_surface_particle(value: str) -> str:
    value = re.sub(r"^[^0-9A-Za-z가-힣]+|[^0-9A-Za-z가-힣#♯♭b♭-]+$", "", value)
    for suffix in _PARTICLE_SUFFIXES:
        if value.endswith(suffix) and len(value) - len(suffix) >= 2:
            return value[:-len(suffix)]
    return value


def _surface_name_compounds(text: str, analyzed) -> list[str]:
    """Recover bounded proper-name compounds from Kiwi word boundaries."""
    found: list[str] = []
    infos: list[tuple[int, int, str, bool, bool, bool]] = []
    grouped = groupby(
        analyzed,
        key=lambda token: (token.line_number, token.sent_position, token.word_position),
    )
    for (line, sentence, word_position), group in grouped:
        word = list(group)
        nounish = [token for token in word if _base_tag(token.tag) in _KEEP_TAGS]
        if not nounish:
            continue
        fragmented = sum(len(token.form) == 1 for token in nounish) >= 2
        start = min(token.start for token in word)
        end = max(token.start + token.len for token in word)
        raw_surface = text[start:end].strip()
        # Parenthesized notation such as ``라(A4)`` already has a precise A4
        # token.  Treating the whole surface as a name manufactured ``라(a4``.
        # Quoted words are handled by the quote extractor/base morphology; the
        # DOM punctuation otherwise leaked continuations such as ``lie'지``.
        if any(mark in raw_surface for mark in "()[]{}'\"“”‘’〈〉《》「」『』~∼"):
            continue
        particle_tokens = [token for token in word if token.tag.startswith("J")]
        has_particle = bool(particle_tokens)
        # Kiwi can parse the last syllable of an unseen name as JKS ``이``
        # (잼+민+이). Two one-syllable noun fragments are a strong signal that
        # the original surface is the entity, not a noun plus case particle.
        if (
            fragmented
            and raw_surface.endswith("이")
            and particle_tokens
            and particle_tokens[-1].form == "이"
        ):
            has_particle = False
        has_serial_suffix = bool(
            re.search(r"[0-9A-Za-z#♯♭-][가-힣]+$", raw_surface)
        )
        surface = (
            _strip_surface_particle(raw_surface)
            if has_particle or has_serial_suffix
            else re.sub(
                r"^[^0-9A-Za-z가-힣]+|[^0-9A-Za-z가-힣#♯♭&-]+$",
                "",
                raw_surface,
            )
        )
        surface = _normalized_term(surface)
        if not 2 <= len(surface.replace("_", "")) <= 30:
            continue
        has_named = any(
            _base_tag(token.tag) in {"NNP", "SH"}
            or (
                _base_tag(token.tag) == "SL"
                and token.form
                and (token.form.isupper() or token.form[0].isupper())
            )
            for token in nounish
        )
        ascii_sentence_word = bool(
            re.fullmatch(r"[A-Za-z']+", surface)
            and not has_named
        )
        if (len(nounish) >= 2 or has_named or fragmented) and not ascii_sentence_word:
            if surface not in found:
                found.append(surface)
        infos.append((
            sentence,
            word_position,
            surface,
            has_named,
            fragmented,
            has_particle,
        ))

    for left, right in zip(infos, infos[1:]):
        left_sentence, left_position, left_surface, left_named, _, left_particle = left
        right_sentence, right_position, right_surface, right_named, right_fragmented, _ = right
        if left_sentence != right_sentence or right_position != left_position + 1:
            continue
        # A case particle means these are grammatical neighbors, not one name:
        # ``김영민과 윤형빈`` and ``릴스와 틱톡`` must stay separate entities.
        if left_particle:
            continue
        # A recovered compound is already a complete word-level entity. Do not
        # chain it to neighbors (초코비_액션가면_DVD) merely due to adjacency.
        if "_" in left_surface or "_" in right_surface:
            continue
        if not left_named:
            continue
        if not (
            right_named
            or right_fragmented
            or right_surface in _ENTITY_ROLE_WORDS
        ):
            continue
        compound = f"{left_surface}_{right_surface}"
        if len(compound.replace("_", "")) <= 30 and compound not in found:
            found.append(compound)
    return found


def _compound_entities(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    found: list[str] = []

    def add(value: str) -> None:
        value = re.sub(r"\s+", "_", value.strip().lower())
        if "_" in value and value not in found:
            found.append(value)

    for phrase in _PROTECTED_PHRASES:
        if re.search(re.escape(phrase), normalized, re.I):
            add(phrase)
    for term in _LITERAL_COMPOUNDS:
        if term.casefold() in normalized.casefold() and term not in found:
            found.append(term)
    for match in _NAMED_ROLE_PHRASE.finditer(normalized):
        name = _normalized_term(match.group(1))
        role = _normalized_term(match.group(2))
        for value in (name, f"{name}_{role}"):
            if value not in found:
                found.append(value)
    for match in _QUOTED_PHRASE.finditer(normalized):
        quoted = _quoted_compound(match.group(1))
        if quoted and quoted not in found:
            found.append(quoted)
    # Numeric counters are much more discriminative as one term ("15화")
    # than as two generic terms ("15", "화").  Keep the ordinary Kiwi
    # terms too so older query normalization remains compatible.
    for match in _NUMBERED_UNIT.finditer(normalized):
        value = "".join(match.groups())
        if value not in found:
            found.append(value)
    for match in _OCTAVE.finditer(normalized):
        value = f"{match.group(1)}옥타브"
        if value not in found:
            found.append(value)
    for match in _PITCH.finditer(normalized):
        value = match.group(1).lower()
        if value not in found:
            found.append(value)
    for match in _CHORD_SEQUENCE.finditer(normalized):
        value = re.sub(r"\s*-\s*", "-", match.group(1)).lower()
        if value not in found:
            found.append(value)
    for match in _HYPHENATED.finditer(normalized):
        value = re.sub(r"\s*-\s*", "-", match.group(1)).lower()
        if value not in found:
            found.append(value)
    return found


@lru_cache(maxsize=50_000)
def _tokenize_base_cached(text: str) -> Tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text)
    analyzed = _get_kiwi().tokenize(normalized, normalize_coda=True)
    result: List[str] = _compound_entities(normalized)
    for term in _surface_name_compounds(normalized, analyzed):
        if term not in result:
            result.append(term)
    protected = set(result)

    for _, group in groupby(
        analyzed,
        key=lambda token: (token.line_number, token.sent_position, token.word_position),
    ):
        word = list(group)
        word_terms: List[str] = []

        def add(term: str) -> None:
            term = _normalized_term(term)
            if term and term not in protected and term not in word_terms:
                word_terms.append(term)

        for token in word:
            tag = _base_tag(token.tag)
            if tag in _KEEP_TAGS:
                add(token.form)
            elif tag in _SPECIAL_TAGS:
                add(token.form)
            elif tag in _VERB_TAGS:
                lemma = _lemma(token.form)
                if _is_relation(lemma):
                    add(lemma)

        if any(_base_tag(token.tag) in _VERB_TAGS for token in word):
            compound_stem = "".join(
                token.form for token in word
                if _base_tag(token.tag) in (_KEEP_TAGS | _VERB_TAGS)
            )
            compound = _lemma(compound_stem)
            if compound and _is_relation(compound):
                add(compound)
        result.extend(word_terms)

    return tuple(result)


def tokenize_context_for_bm25(
    text: str,
    *,
    expand_synonyms: bool = True,
) -> List[str]:
    """Return strict factual tokens.

    Query normalization keeps bounded synonym expansion by default. Stored
    document passages pass ``expand_synonyms=False`` so an occurrence of
    ``삽입곡`` does not manufacture OST/BGM terms in every record.
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", str(text))
    result = list(_tokenize_base_cached(normalized))
    if not expand_synonyms:
        return result

    # Query-side expansion tokens are added once per synonym family.
    present = set(result)
    for pattern, synonyms in _SYNONYM_GROUPS:
        if pattern.search(normalized):
            for synonym in synonyms:
                if synonym not in present:
                    result.append(synonym)
                    present.add(synonym)
    return result


def normalize_context_for_bm25(text: str, *, expand_synonyms: bool = True) -> str:
    """Return the whitespace-separated form expected by a context BM25 index."""
    return " ".join(
        tokenize_context_for_bm25(text, expand_synonyms=expand_synonyms)
    )
