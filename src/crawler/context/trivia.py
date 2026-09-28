"""Project Namuwiki trivia into small, deterministic, search-ready facts.

The raw HTML, parsed blocks, links and footnotes remain in ``data/context``.
Only complete, bounded facts are copied to a song's compact ``meta.json``.
No LLM is used here, so a rerun produces the same projection for the same
parsed document.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .document_parser import compact, section_path
from .schemas import NamuwikiFact, ParsedDocument, SongTarget

TRIVIA_HEADINGS = {"여담", "기타사항", "트리비아", "trivia"}

# 40..300 characters is the preferred retrieval unit. Useful short musical
# facts are allowed, and a single complete sentence may use the 400 hard cap.
PREFERRED_MIN_FACT_CHARS = 40
PREFERRED_MAX_FACT_CHARS = 300
MAX_FACT_CHARS = 400
MIN_FACT_CHARS = 12

_CITATION_MARKER = re.compile(r"\[(?:주\s*)?\d{1,4}(?:\s*[-–~]\s*\d{1,4})?\]")
_STANDALONE_HASH = re.compile(r"(?<![\w])#(?![\w])")
# Sentence boundaries are detected by a small scanner below. A regular
# expression cannot distinguish the question mark in "'으에? 이걸요?'" from
# the end of the surrounding sentence.
_SENTENCE_TERMINALS = {".", "!", "?", "。", "！", "？"}
_TRAILING_SENTENCE_CLOSERS = {'"', "'", "”", "’", ")", "]", "}"}
_LEADING_MEDIA_CAPTION = re.compile(
    r"^\(\s*[^()\n]{0,100}?(?:\d{1,2}:)?\d{1,2}:\d{2}\s*(?:[~〜～\-–—]\s*)?"
    r"[^()\n]{0,40}\)\s*"
)
_INCOMPLETE_END = re.compile(
    r"(?:[,，:：;；/]|으로|하며|이고|인데|지만|때문에|어조로)\s*$"
)
_DEPENDENT_START = re.compile(
    r"^(?:라고|이러한|그런데|그러나|하지만|그리고|또한|한편|그렇게|덕분에|심지어|의외로|"
    r"그 결과|그로 인해|"
    r"이 때문에|이를 통해|그래서(?:인지)?)(?:\s|[,，])"
)
# References whose antecedent is not the song itself. ``이 곡``/``이 노래`` are
# intentionally excluded because the artifact's embedding identity resolves
# those safely; ``이 음`` cannot be understood without the preceding pitch fact.
_UNRESOLVED_REFERENCE = re.compile(
    r"(?:^|\s)(?:이|그|해당)\s*(?:음|음정|코드)"
    r"(?:은|는|이|가|을|를|에|에서|와|과|\s)"
)
# A dependent sentence is joined only when it still carries a concrete song,
# media, production, meme or performance relation. This preserves useful
# continuations while keeping unrelated episode plot and road-sign details out.
_DEPENDENT_MERGE_SIGNAL = re.compile(
    r"곡|노래|음원|앨범|싱글|발매|삽입|배경\s*음악|\bBGM\b|\bOST\b|"
    r"사용|수록|제작|작곡|작사|녹음|데모|프로듀싱|무대|공연|경연|라이브|"
    r"패러디|합성|밈|드립|유머|낙서|챌린지|수상|선정|차트|기록|"
    r"평가|영향|인기|주목|검색어|실검|화제|호평|떼창|가사|제목|버전|"
    r"리메이크|편곡|연주|가창|코드|최고음|음역|조성|템포|노래방|눈물|울음|"
    r"뮤직\s*비디오|뮤비|\bMV\b",
    re.I,
)
_DATE_AT_END = re.compile(
    r"(?:19|20)\d{2}[./-]\s*\d{1,2}(?:[./-]\s*\d{1,2})?\.?\s*$"
)
_RELATION_SIGNAL = re.compile(
    r"곡|노래|음원|앨범|싱글|발매|공개|방영|삽입|배경\s*음악|BGM|OST|"
    r"사용|수록|제작|작곡|작사|녹음|데모|프로듀싱|무대|공연|경연|라이브|"
    r"패러디|합성|밈|유행|챌린지|수상|선정|차트|기록|평가|영향|대표곡|"
    r"가사|제목|버전|리메이크|편곡|연주|가창|코드|최고음|음역|조성|템포|"
    r"수익|재투자|보컬|뮤직\s*비디오|뮤비|MV",
    re.I,
)
_CONCRETE_SIGNAL = re.compile(
    r"\d|[A-Z][A-Za-z0-9-]*|[〈《「『\"“'][^〉》」』\"”']{2,}[〉》」』\"”']"
)
_CONCRETE_DOMAIN_TERMS = re.compile(
    r"가수|그룹|밴드|보컬|프로듀서|작곡가|방송사|프로그램|게임|드라마|영화|"
    r"애니메이션|만화|에피소드|캐릭터|노래방|홈쇼핑|축제|시상식|설문조사|"
    r"수익|원키|옥타브|코러스|후렴|저작권"
)
_LOW_VALUE_MEDIA_DETAIL = re.compile(
    r"교통\s*상황|표지판|진입로|이정표|\d+\s*km|\d+분\s*소요",
    re.I,
)
_DIRECT_MEDIA_RELATION = re.compile(
    r"삽입|배경\s*음악|BGM|OST|사운드트랙|사용|수록|선곡|곡이\s*흐|노래가\s*흐|"
    r"재생|방영|선공개|뮤직\s*비디오|뮤비|MV",
    re.I,
)
_MEDIA_SCENE_SIGNAL = re.compile(
    r"장면|에피소드|\d+기\s*\d+화|\d+화|캐릭터|주인공|연인|이별|사랑|눈물|"
    r"울음|공항|버스|재더빙|원판|내수판|DVD|저작권"
)

_ATTRIBUTION_PATTERNS = (
    re.compile(r"^이 문서의 내용 중 전체 또는 일부는"),
    re.compile(r"문서의\s*r?\d+\s*판에서 가져왔습니다"),
    re.compile(r"이전 역사 보러 가기"),
    re.compile(r"^나무위키는 백과사전이 아니며"),
    re.compile(r"^자세한 내용은 .{0,80} 문서(?:를)? 참조"),
)

# Text classification is used after a section default has been considered.
# Specific musical and meme clues precede generic words such as "노래방",
# "영상" and "경연" that otherwise caused the sample misclassifications.
_CATEGORY_PATTERNS = (
    ("musical_detail", re.compile(
        r"코드\s*(?:진행)?|원키|키\s*변경|최고음|최저음|옥타브|음역|조성|전조|"
        r"BPM|(?:빠른|느린)\s*템포|템포(?:가|는|를|의|변화)|박자|리듬(?:이|은|을|의|패턴)|"
        r"연주\s*(?:난이도|자체)|가창\s*난이도|부르기|"
        r"창법(?:이|은|을|의|으로)|코러스|후렴|브리지|벌스|곡\s*구조|반복\s*(?:구조|진행)",
        re.I,
    )),
    ("version", re.compile(r"영어\s*제목|복수형|싱글판|앨범판|재녹음|리메이크|편곡|개사")),
    ("meme", re.compile(
        r"밈|패러디|합성|낚시\s*영상|챌린지|짤방|짤로|말장난|유머|드립|"
        r"띄어쓰기|낙서|인터넷에서|온라인에서|릴스|틱톡|유행하|화제가\s*되"
    )),
    ("music_video", re.compile(r"뮤직\s*비디오|\bM\.?V\.?\b|music\s*video", re.I)),
    ("production", re.compile(
        r"프로듀싱|프로듀서|작곡|작사|제작\s*배경|제작\s*과정|녹음|"
        r"솔로곡으로|타이틀곡으로|곡을\s*만들|노래를\s*만들|발매\s*배경|"
        r"기획|데모곡|데모\s*곡|처음에는|원래는"
    )),
    ("media_usage", re.compile(
        r"\bOST\b|\bBGM\b|삽입곡|배경\s*음악|사운드트랙|"
        r"드라마|영화|애니메이션|만화|방송|광고|게임|홈쇼핑|벨소리|"
        r"통화\s*연결음|사용되|수록되|삽입되|곡이\s*흐|노래가\s*흐",
        re.I,
    )),
    ("performance", re.compile(
        r"무대|공연|콘서트|경연|복면가왕|불후의\s*명곡|라이브|"
        r"커버\s*무대|시상식|떼창"
    )),
    ("record", re.compile(
        r"\d+위|수상|차트|기록|판매량|조회수|다운로드|스트리밍|"
        r"설문조사|선정한|대상\s*수상|인기\s*곡"
    )),
    ("influence", re.compile(
        r"영향|흐름을\s*바|대중화|상징하|대표곡|스타덤|전성시대|"
        r"음악을\s*시작|랩을\s*시작|평가를\s*받|체제를\s*종식|위기감을\s*느"
    )),
    ("version", re.compile(r"버전|배속|국내판|해외판|원곡판")),
)

_SECTION_PATTERNS = (
    ("meme", re.compile(r"밈|패러디|합성|챌린지")),
    ("media_usage", re.compile(
        r"삽입곡|OST|BGM|배경\s*음악|사운드트랙|드라마|영화|애니메이션|만화",
        re.I,
    )),
    ("musical_detail", re.compile(
        r"음악적\s*특징|곡\s*분석|코드|음역|최고음|조성|템포|연주|가창"
    )),
    ("music_video", re.compile(r"뮤직\s*비디오|뮤비|\bMV\b", re.I)),
    ("production", re.compile(r"제작|작곡|작사|녹음|데모|비하인드")),
    ("performance", re.compile(r"무대|공연|라이브|콘서트|경연")),
    ("record", re.compile(r"성적|기록|수상|차트")),
)


def _clean_heading(value: str) -> str:
    value = re.sub(r"\s*\[편집\]\s*$", "", str(value))
    # Strip a real section ordinal (``2.1. 여담``), not an artist/title whose
    # identity starts with a digit (``10CM``, ``2AM``, ``2019년``).
    return re.sub(
        r"^\s*\d+(?:\.\d+)*(?:\.\s*|\s+)",
        "",
        value,
    ).strip()


def _heading_key(value: str) -> str:
    return compact(_clean_heading(value))


def is_trivia_path(path: Iterable[str]) -> bool:
    return any(_heading_key(part) in TRIVIA_HEADINGS for part in path)


def _trivia_path(path: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    found = False
    for part in path:
        label = _clean_heading(str(part))
        if not found and _heading_key(label) in TRIVIA_HEADINGS:
            found = True
        if found and label:
            cleaned.append(label)
    return cleaned


def _name_variants(values: Iterable[str]) -> list[str]:
    """Expand ``BIGBANG (빅뱅)`` into safe local aliases without a lookup table."""
    variants = []
    for raw in values:
        value = str(raw).strip()
        if not value:
            continue
        variants.append(value)
        base = re.sub(r"\([^)]*\)", "", value).strip()
        if base:
            variants.append(base)
        variants.extend(part.strip() for part in re.findall(r"\(([^)]*)\)", value) if part.strip())
        variants.extend(part.strip() for part in re.split(r"[/·|]", value) if part.strip())
    return list(dict.fromkeys(variants))


@dataclass(frozen=True)
class BindingDecision:
    """Auditable page/section binding without expanding compact ``meta.json``."""

    verified: bool
    reason: str
    confidence: str = "none"
    scope_section_id: str | None = None
    evidence: tuple[str, ...] = ()
    candidate_section_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "verified": self.verified,
            "reason": self.reason,
            "confidence": self.confidence,
            "scope_section_id": self.scope_section_id,
            "evidence": list(self.evidence),
            "candidate_section_ids": list(self.candidate_section_ids),
        }


_MUSIC_IDENTITY = re.compile(
    r"노래|곡|싱글|타이틀곡|수록곡|음원|트랙|발매|가창|보컬|OST|앨범",
    re.I,
)
_IDENTITY_HEADINGS = {
    "개요",
    "상세",
    "소개",
    "곡정보",
    "음원정보",
    "앨범정보",
    "음반정보",
    "트랙리스트",
}
_IDENTITY_EXCLUDED_HEADINGS = {
    "가사",
    "여담",
    "기타사항",
    "트리비아",
    "관련문서",
    "둘러보기",
}


def _section_ids(document: ParsedDocument, root_id: str) -> set[str]:
    """Return one section subtree without relying on DOM/list ordering."""
    allowed = {root_id}
    changed = True
    while changed:
        changed = False
        for section in document.sections:
            if section.parent_id in allowed and section.section_id not in allowed:
                allowed.add(section.section_id)
                changed = True
    return allowed


def _identity_text(document: ParsedDocument, scope_section_id: str | None = None) -> str:
    """Bound identity evidence to infobox/intro text, never lyrics or trivia."""
    allowed = (
        _section_ids(document, scope_section_id)
        if scope_section_id is not None
        else None
    )
    values: list[str] = []
    for block in document.blocks:
        if allowed is not None and block.section_id not in allowed:
            continue
        if "lyrics" in block.flags:
            continue
        path = section_path(document, block.section_id)
        keys = [_heading_key(part) for part in path]
        if any(key in _IDENTITY_EXCLUDED_HEADINGS for key in keys):
            continue
        if allowed is None:
            # Whole-page evidence is deliberately limited to root/intro and
            # table rows.  An artist mentioned much later in a cover or trivia
            # section must not verify an unrelated same-title page.
            if path and _heading_key(path[0]) not in _IDENTITY_HEADINGS and block.block_type != "table_row":
                continue
        values.append(block.normalized_text)
        if len(values) >= 40 or sum(map(len, values)) >= 6000:
            break
    return " ".join(values)


def _alias_in_text(text: str, alias: str) -> bool:
    """Match credited names conservatively, including punctuation-rich Latin names."""
    raw = unicodedata.normalize("NFKC", str(alias)).casefold().strip()
    key = compact(raw)
    if not key:
        return False
    # The leading hash is part of artist identities such as ``#안녕``.  If it
    # is discarded, an ordinary greeting ("안녕하세요") becomes false artist
    # evidence on an unrelated same-title page.
    if raw.startswith("#"):
        return raw in unicodedata.normalize("NFKC", text).casefold()
    if re.fullmatch(r"[0-9a-z]+", key):
        chunks = re.findall(r"[0-9a-z]+", raw)
        if not chunks:
            return False
        pattern = r"(?<![0-9a-z])" + r"[^0-9a-z가-힣]*".join(
            re.escape(chunk) for chunk in chunks
        ) + r"(?![0-9a-z])"
        return bool(re.search(pattern, unicodedata.normalize("NFKC", text).casefold()))
    # One-character Hangul aliases (for example, 비) create too many substring
    # false positives.  A dedicated title(artist) page can still verify them.
    if len(key) < 2:
        return False
    return key in compact(text)


def _matching_aliases(text: str, aliases: Iterable[str]) -> list[str]:
    return [alias for alias in aliases if _alias_in_text(text, alias)]


def _title_keys(title: str, title_aliases: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(
        key
        for key in (compact(title), *(compact(value) for value in title_aliases))
        if key
    ))


_ARTIST_CREDIT_SEPARATOR = re.compile(r"\s*[,，/·&＆+]\s*")


def _page_title_artist_match(
    page_title: str,
    title_keys: Iterable[str],
    artists: Iterable[str],
) -> tuple[tuple[str, ...], bool] | None:
    """Match a final ``title(artist credits)`` disambiguator exactly.

    NamuWiki sometimes names a collaboration page with every credited act,
    for example ``원더우먼(씨야, 다비치, 티아라)``.  Compacting the whole
    suffix cannot match a catalogue row that names only one credited act.

    Artist credits are therefore split only on explicit list punctuation and
    compared as complete normalized tokens.  Substring matches are forbidden,
    so ``티아`` cannot verify a page credited to ``티아라``.  When the
    catalogue has multiple artist entries, every entry must be represented in
    the page title.
    """
    value = unicodedata.normalize("NFKC", str(page_title)).strip()
    match = re.fullmatch(r"(.+)\(([^()]*)\)\s*", value)
    if not match or compact(match.group(1)) not in set(title_keys):
        return None

    credit_text = match.group(2).strip()
    if not credit_text:
        return None
    credit_parts = [
        part.strip()
        for part in _ARTIST_CREDIT_SEPARATOR.split(credit_text)
        if part.strip()
    ]
    credit_keys = {
        key
        for key in (
            compact(credit_text),
            *(compact(part) for part in credit_parts),
        )
        if key
    }

    matched: list[str] = []
    for artist in artists:
        aliases = _name_variants([str(artist)])
        alias = next(
            (candidate for candidate in aliases if compact(candidate) in credit_keys),
            None,
        )
        if alias is None:
            return None
        matched.append(alias)

    return tuple(matched), len({compact(part) for part in credit_parts}) > 1


def _independent_album_keys(album: str | None, title_keys: Iterable[str]) -> list[str]:
    """Return bounded album aliases that still carry independent identity."""
    value = str(album or "").strip()
    keys: list[str] = []
    raw_key = compact(value)
    if len(raw_key) >= 3:
        keys.append(raw_key)

    # Melon commonly appends ``OST Part.2`` while NamuWiki names the parent
    # document ``작품명/OST``.  Removing only the numbered part preserves the
    # work/album identity and avoids reducing a generic album to bare ``OST``.
    without_part = re.sub(
        r"\s*(?:part|파트)\s*[._-]?\s*\d+\s*$",
        "",
        value,
        flags=re.I,
    ).strip()
    normalized_key = compact(without_part)
    if len(normalized_key) >= 4:
        keys.append(normalized_key)

    title_key_set = set(title_keys)
    return list(dict.fromkeys(
        key for key in keys if key not in title_key_set
    ))


def _ancestor_heading_text(document: ParsedDocument, section_id: str) -> str:
    by_id = {section.section_id: section for section in document.sections}
    headings: list[str] = []
    current = by_id.get(section_id)
    visited: set[str] = set()
    while current is not None and current.section_id not in visited:
        visited.add(current.section_id)
        if current.heading:
            headings.append(_clean_heading(current.heading))
        current = by_id.get(current.parent_id) if current.parent_id else None
    return " ".join(reversed(headings))


def _year_in_text(text: str, release_year: int | None) -> bool:
    return bool(
        release_year
        and re.search(rf"(?<!\d){int(release_year)}(?!\d)", text)
    )


def _section_candidates(
    *,
    document: ParsedDocument,
    title_keys: list[str],
    artist_aliases: list[str],
    album_keys: list[str],
    release_year: int | None,
) -> list[dict]:
    page_key = compact(document.page_title or "")
    global_identity = _identity_text(document)
    global_artists = _matching_aliases(global_identity, artist_aliases)
    page_is_album = any(page_key == key for key in album_keys)
    candidates: list[dict] = []

    for section in document.sections:
        if not section.heading:
            continue
        heading = _clean_heading(section.heading)
        heading_key = compact(heading)
        if not heading_key or _heading_key(heading) in _IDENTITY_EXCLUDED_HEADINGS:
            continue

        heading_title_exact = any(heading_key == key for key in title_keys)
        heading_title_contains = any(
            len(key) >= 4 and key in heading_key for key in title_keys
        )
        ancestor_text = _ancestor_heading_text(document, section.section_id)
        direct_heading_artists = _matching_aliases(heading, artist_aliases)
        heading_artists = _matching_aliases(ancestor_text, artist_aliases)
        artist_song_heading = bool(
            direct_heading_artists and re.search(r"노래|곡|싱글|음원", heading)
        )
        if not (heading_title_exact or heading_title_contains or artist_song_heading):
            continue

        local_text = _identity_text(document, section.section_id)
        local_artists = _matching_aliases(local_text, artist_aliases)
        artist_matches = list(dict.fromkeys([*heading_artists, *local_artists]))
        album_match = any(
            key in compact(ancestor_text + " " + local_text)
            for key in album_keys
        )
        year_match = _year_in_text(local_text, release_year)
        music_signal = bool(_MUSIC_IDENTITY.search(heading + " " + local_text))

        evidence: list[str] = []
        if heading_title_exact:
            evidence.append("section_title_exact")
        elif heading_title_contains:
            evidence.append("section_title_contains")
        if artist_song_heading:
            evidence.append("artist_song_heading")
        if artist_matches:
            evidence.append("section_artist:" + ",".join(artist_matches))
        if album_match:
            evidence.append("section_album")
        if year_match:
            evidence.append("section_release_year")
        if page_is_album:
            evidence.append("page_album_title")
        if global_artists:
            evidence.append("page_artist:" + ",".join(global_artists))
        if music_signal:
            evidence.append("music_relation")

        reason = None
        score = 0
        if artist_song_heading and any(page_key == key for key in title_keys):
            reason, score = "artist_song_subsection_match", 100
        elif heading_title_exact and artist_matches and music_signal:
            reason, score = "title_section_artist_match", 95
        elif (
            heading_title_exact
            and page_is_album
            and global_artists
            and music_signal
        ):
            reason, score = "album_track_section_match", 90
        elif heading_title_contains and artist_matches and music_signal:
            reason, score = "title_artist_section_match", 85

        if reason:
            # Year agreement raises audit confidence but can never verify a
            # page by itself.  Release dates are frequently absent on NamuWiki.
            score += 2 if year_match else 0
            candidates.append({
                "section_id": section.section_id,
                "reason": reason,
                "score": score,
                "evidence": tuple(evidence),
            })

    return sorted(candidates, key=lambda row: (-row["score"], row["section_id"]))


def decide_song_page(
    title: str,
    artists: Iterable[str],
    document: ParsedDocument,
    *,
    album: str | None = None,
    release_year: int | None = None,
    title_aliases: Iterable[str] = (),
) -> BindingDecision:
    """Bind only a dedicated page or one uniquely identified song subtree."""
    if document.parse_status == "structure_error" or not document.page_title:
        return BindingDecision(False, "page_structure_or_title_missing")

    artists = list(artists)
    artist_aliases = _name_variants(artists)
    artist_keys = [compact(value) for value in artist_aliases if compact(value)]
    title_keys = _title_keys(title, title_aliases)
    page_key = compact(document.page_title)
    album_keys = _independent_album_keys(album, title_keys)

    # A dedicated title(artist) page is the strongest available local signal.
    # Keep the original compact whole-suffix comparison first: besides being
    # backwards compatible, it also handles punctuation-rich artist names
    # whose own name contains parentheses or credit separators.
    if any(
        page_key == title_key + artist_key
        for title_key in title_keys
        for artist_key in artist_keys
    ):
        return BindingDecision(
            True,
            "title_and_artist_match",
            "high",
            evidence=("page_title", "artist_disambiguator"),
        )

    # A collaboration page may list co-artists.  Each catalogue artist must
    # match one complete credit token, so a same-title page credited only to a
    # different artist cannot pass this fallback.
    page_artist_match = _page_title_artist_match(
        document.page_title,
        title_keys,
        artists,
    )
    if page_artist_match is not None:
        matched_page_artists, has_multiple_credits = page_artist_match
        return BindingDecision(
            True,
            (
                "title_and_coartist_match"
                if has_multiple_credits
                else "title_and_artist_match"
            ),
            "high",
            evidence=(
                "page_title",
                "artist_disambiguator:" + ",".join(matched_page_artists),
                *(() if not has_multiple_credits else ("coartist_disambiguator",)),
            ),
        )

    scoped = _section_candidates(
        document=document,
        title_keys=title_keys,
        artist_aliases=artist_aliases,
        album_keys=album_keys,
        release_year=release_year,
    )
    if scoped:
        top_score = scoped[0]["score"]
        best = [row for row in scoped if row["score"] == top_score]
        if len(best) == 1:
            winner = best[0]
            return BindingDecision(
                True,
                winner["reason"],
                "high" if top_score >= 90 else "medium",
                scope_section_id=winner["section_id"],
                evidence=winner["evidence"],
                candidate_section_ids=tuple(row["section_id"] for row in scoped),
            )
        return BindingDecision(
            False,
            "multiple_equally_strong_song_sections",
            candidate_section_ids=tuple(row["section_id"] for row in best),
            evidence=tuple(
                sorted({item for row in best for item in row["evidence"]})
            ),
        )

    title_only_page = any(page_key == key for key in title_keys)
    generic_song_page = any(page_key == key + compact("노래") for key in title_keys)
    identity_text = _identity_text(document)
    matched_artists = _matching_aliases(identity_text, artist_aliases)
    music_signal = bool(_MUSIC_IDENTITY.search(identity_text))
    album_match = any(key in compact(identity_text) for key in album_keys)
    evidence = []
    if matched_artists:
        evidence.append("page_artist:" + ",".join(matched_artists))
    if music_signal:
        evidence.append("music_relation")
    if album_match:
        evidence.append("page_album")
    if _year_in_text(identity_text, release_year):
        evidence.append("page_release_year")

    # A whole-page fallback must never swallow neighbouring song sections.
    # If a unique section had enough metadata it would already have returned
    # above; multiple unresolved song subtrees remain a manual-review case.
    song_subsections = [
        section.section_id
        for section in document.sections
        if re.search(r"의\s*(?:노래|곡)\s*$", section.heading)
    ]
    if len(song_subsections) >= 2:
        return BindingDecision(
            False,
            "disambiguation_page_multiple_song_sections",
            evidence=tuple(evidence),
            candidate_section_ids=tuple(song_subsections),
        )

    if (title_only_page or generic_song_page) and matched_artists and (
        music_signal or album_match
    ):
        return BindingDecision(
            True,
            (
                "generic_song_page_with_local_artist_identity"
                if generic_song_page
                else "title_only_page_with_local_artist_identity"
            ),
            "high" if album_match else "medium",
            evidence=tuple(evidence),
        )

    if generic_song_page:
        return BindingDecision(
            False,
            "generic_song_page_artist_not_verified",
            evidence=tuple(evidence),
        )
    if title_only_page:
        return BindingDecision(
            False,
            "title_only_page_artist_not_verified",
            evidence=tuple(evidence),
        )
    return BindingDecision(
        False,
        "page_title_mismatch",
        evidence=tuple(evidence),
        candidate_section_ids=tuple(row["section_id"] for row in scoped),
    )


def bind_song_page(target: SongTarget, document: ParsedDocument) -> BindingDecision:
    return decide_song_page(
        target.title,
        target.artists,
        document,
        album=target.album,
        release_year=target.release_year,
        title_aliases=target.title_aliases,
    )


def verify_song_page(
    title: str,
    artists: Iterable[str],
    document: ParsedDocument,
    *,
    album: str | None = None,
    release_year: int | None = None,
    title_aliases: Iterable[str] = (),
) -> tuple[bool, str]:
    """Compatibility wrapper for callers that only need yes/no and reason."""
    decision = decide_song_page(
        title,
        artists,
        document,
        album=album,
        release_year=release_year,
        title_aliases=title_aliases,
    )
    return decision.verified, decision.reason


def _ascii_apostrophe(text: str, index: int) -> bool:
    """Return true only for an apostrophe inside an ASCII word such as I'm."""
    if not 0 < index < len(text) - 1:
        return False
    before, after = text[index - 1], text[index + 1]
    return before.isascii() and after.isascii() and before.isalnum() and after.isalnum()


def _sentence_units(text: str) -> list[str]:
    """Split only at terminal punctuation outside quotes and brackets."""
    parts: list[str] = []
    start = 0
    pending_end: int | None = None
    round_depth = square_depth = curly_depth = 0
    straight_double = straight_single = False
    curly_double = curly_single = False
    index = 0

    def scoped() -> bool:
        return bool(
            round_depth
            or square_depth
            or curly_depth
            or straight_double
            or straight_single
            or curly_double
            or curly_single
        )

    while index < len(text):
        char = text[index]

        if char == '"' and (index == 0 or text[index - 1] != "\\"):
            straight_double = not straight_double
        elif char == "'" and not _ascii_apostrophe(text, index):
            straight_single = not straight_single
        elif char == "“":
            curly_double = True
        elif char == "”":
            curly_double = False
        elif char == "‘":
            curly_single = True
        elif char == "’":
            curly_single = False
        elif not (straight_double or straight_single or curly_double or curly_single):
            if char == "(":
                round_depth += 1
            elif char == ")" and round_depth:
                round_depth -= 1
            elif char == "[":
                square_depth += 1
            elif char == "]" and square_depth:
                square_depth -= 1
            elif char == "{":
                curly_depth += 1
            elif char == "}" and curly_depth:
                curly_depth -= 1

        if char in _SENTENCE_TERMINALS:
            pending_end = index + 1
        elif pending_end is not None:
            if char in _TRAILING_SENTENCE_CLOSERS:
                pending_end = index + 1
            elif char.isspace():
                next_index = index
                while next_index < len(text) and text[next_index].isspace():
                    next_index += 1
                if next_index < len(text) and not scoped():
                    part = text[start:pending_end].strip()
                    if part:
                        parts.append(part)
                    start = next_index
                    pending_end = None
                    index = next_index
                    continue
            else:
                # More text continued inside a quote/bracket, so the earlier
                # punctuation was not the surrounding sentence boundary.
                pending_end = None
        index += 1

    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def clean_fact_text(value: str) -> str:
    """Remove presentation-only markers while preserving the source wording."""
    text = unicodedata.normalize("NFKC", str(value)).replace("\u200b", "")
    text = _CITATION_MARKER.sub("", text)
    text = _STANDALONE_HASH.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _LEADING_MEDIA_CAPTION.sub("", text, count=1).strip()
    text = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", text)
    return text


def split_fact_text(text: str, *, limit: int = MAX_FACT_CHARS) -> list[str]:
    """Split at quote-aware sentence boundaries without truncating a clause."""
    if limit < 1 or limit > MAX_FACT_CHARS:
        raise ValueError(f"fact limit must be between 1 and {MAX_FACT_CHARS}")
    text = clean_fact_text(text)
    if not text:
        return []
    sentences = [clean_fact_text(part) for part in _sentence_units(text) if clean_fact_text(part)]
    # A source sentence over the hard limit cannot be split without inventing
    # syntax or emitting an incomplete clause, so it is excluded from indexing.
    return [sentence for sentence in sentences if len(sentence) <= limit]


def _section_category(path: Iterable[str]) -> str | None:
    local = _trivia_path(path)
    # The most specific descendant heading wins. Plain "여담" has no default.
    for heading in reversed(local[1:]):
        for category, pattern in _SECTION_PATTERNS:
            if pattern.search(heading):
                return category
    return None


def _text_category(text: str) -> str:
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return "other"


def classify(text: str, section: Iterable[str] | str | None = None) -> str:
    """Classify by the leaf section first, then by sentence-level evidence."""
    if isinstance(section, str):
        path = [part.strip() for part in section.split(">") if part.strip()]
    else:
        path = list(section or [])
    return _section_category(path) or _text_category(clean_fact_text(text))


def _balanced_delimiters(text: str) -> bool:
    for opening, closing in (("(", ")"), ("[", "]"), ("“", "”"), ("‘", "’")):
        if text.count(opening) != text.count(closing):
            return False
    if text.count('"') % 2:
        return False
    # Ignore only ASCII word apostrophes. Korean single-quoted speech followed
    # by a particle (e.g. '한곡 정도야'라고) must still count as a quote.
    straight_single_quotes = sum(
        char == "'" and not _ascii_apostrophe(text, index)
        for index, char in enumerate(text)
    )
    return straight_single_quotes % 2 == 0


def _title_or_date_only(text: str) -> bool:
    if re.fullmatch(r"[\s(]*(?:19|20)\d{2}(?:[./-]\d{1,2}){0,2}[.)\s]*", text):
        return True
    if _DATE_AT_END.search(text) and (" - " in text or " / " in text):
        return not _RELATION_SIGNAL.search(text)
    if len(text) <= 120 and " - " in text and not _RELATION_SIGNAL.search(text):
        return True
    return False


def _other_has_concrete_relation(text: str) -> bool:
    if not _RELATION_SIGNAL.search(text):
        return False
    concrete_terms = _CONCRETE_DOMAIN_TERMS.findall(text)
    return bool(_CONCRETE_SIGNAL.search(text) or len(concrete_terms) >= 2)


def is_indexable_fact(
    text: str,
    *,
    category: str | None = None,
    section: Iterable[str] | str | None = None,
    block_type: str = "paragraph",
    page_title: str | None = None,
) -> bool:
    """Defensive quality gate shared by collection and future context indexing."""
    text = clean_fact_text(text)
    if not MIN_FACT_CHARS <= len(text) <= MAX_FACT_CHARS:
        return False
    if page_title and compact(text) == compact(page_title):
        return False
    if any(pattern.search(text) for pattern in _ATTRIBUTION_PATTERNS):
        return False
    if block_type == "quote":
        return False
    if _INCOMPLETE_END.search(text) or _DEPENDENT_START.search(text):
        return False
    if not _balanced_delimiters(text) or _title_or_date_only(text):
        return False

    resolved = category or classify(text, section)
    if resolved == "other":
        return _other_has_concrete_relation(text)

    path = (
        [part.strip() for part in section.split(">") if part.strip()]
        if isinstance(section, str)
        else list(section or [])
    )
    if resolved == "media_usage" and _section_category(path) == "media_usage":
        if _LOW_VALUE_MEDIA_DETAIL.search(text) and not _DIRECT_MEDIA_RELATION.search(text):
            return False
        # A nested insertion section may contain a useful scene description
        # without repeating "the song plays" in every sentence.
        if _text_category(text) == "other" and not _MEDIA_SCENE_SIGNAL.search(text):
            return False
    return True


def _facts_from_rows(
    rows: Iterable[tuple[list[str], str, str]],
    *,
    page_title: str | None = None,
) -> list[dict]:
    candidates: list[dict] = []
    last_index: int | None = None
    pending_quote_index: int | None = None
    active_path: tuple[str, ...] | None = None

    for path, block_type, raw_text in rows:
        local_path = _trivia_path(path)
        path_key = tuple(local_path)
        if not local_path:
            last_index = pending_quote_index = None
            active_path = None
            continue
        if path_key != active_path:
            last_index = pending_quote_index = None
            active_path = path_key
        # Quotation boxes are intentionally not indexed and must act as a
        # barrier. Otherwise text on either side could be joined across an
        # omitted quotation.
        if block_type == "quote":
            last_index = pending_quote_index = None
            continue

        parts = split_fact_text(raw_text)
        if not parts:
            last_index = pending_quote_index = None
            continue

        for part in parts:
            if pending_quote_index is not None:
                previous = candidates[pending_quote_index]
                combined = clean_fact_text(previous["text"] + " " + part)
                if (
                    previous["path_key"] == path_key
                    and len(combined) <= MAX_FACT_CHARS
                    and _balanced_delimiters(combined)
                ):
                    previous["text"] = combined
                    last_index = pending_quote_index
                    pending_quote_index = None
                    continue
                pending_quote_index = None

            candidate = {
                "path": local_path,
                "path_key": path_key,
                "block_type": block_type,
                "text": part,
            }

            if not _balanced_delimiters(part):
                candidates.append(candidate)
                pending_quote_index = len(candidates) - 1
                last_index = None
                continue

            dependent = bool(
                _DEPENDENT_START.search(part)
                or _UNRESOLVED_REFERENCE.search(part)
            )
            if dependent:
                if (
                    last_index is not None
                    and candidates[last_index]["path_key"] == path_key
                    and _DEPENDENT_MERGE_SIGNAL.search(part)
                ):
                    combined = clean_fact_text(candidates[last_index]["text"] + " " + part)
                    if len(combined) <= MAX_FACT_CHARS:
                        candidates[last_index]["text"] = combined
                        continue
                candidates.append(candidate)
                last_index = None
                continue

            candidates.append(candidate)
            index = len(candidates) - 1
            # Invalid fragments and boilerplate are barriers. A later dependent
            # sentence must not make them look valid by attaching to them.
            if (
                _INCOMPLETE_END.search(part)
                or _title_or_date_only(part)
                or any(pattern.search(part) for pattern in _ATTRIBUTION_PATTERNS)
            ):
                last_index = None
            else:
                last_index = index

    facts: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        local_path = candidate["path"]
        block_type = candidate["block_type"]
        part = candidate["text"]
        section = " > ".join(local_path)
        category = classify(part, local_path)
        if not is_indexable_fact(
            part,
            category=category,
            section=local_path,
            block_type=block_type,
            page_title=page_title,
        ):
            continue
        key = compact(part)
        if not key or key in seen:
            continue
        seen.add(key)
        facts.append(NamuwikiFact(category=category, section=section, text=part).model_dump())
    return facts


def extract_trivia_facts(
    document: ParsedDocument,
    *,
    scope_section_id: str | None = None,
) -> tuple[list[dict], bool]:
    """Extract trivia from the verified song subtree only.

    Dedicated song pages use the full document.  Disambiguation, album and OST
    pages must provide ``scope_section_id`` so facts belonging to neighbouring
    songs can never leak into the target song's artifact.
    """
    paths = {
        section.section_id: section_path(document, section.section_id)
        for section in document.sections
    }
    if scope_section_id is not None:
        if scope_section_id not in paths:
            raise ValueError("binding scope section is missing from parsed document")
        allowed = _section_ids(document, scope_section_id)
    else:
        allowed = set(paths)
    found = any(
        section_id in allowed and is_trivia_path(path)
        for section_id, path in paths.items()
    )
    rows = (
        (paths.get(block.section_id, []), block.block_type, block.source_text)
        for block in document.blocks
        if block.section_id in allowed
    )
    return _facts_from_rows(rows, page_title=document.page_title), found


def refine_existing_facts(facts: Iterable[Mapping], *, page_title: str | None = None) -> list[dict]:
    """Upgrade compact v1 facts offline, without another Namuwiki request."""
    rows = []
    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        section = str(fact.get("section", ""))
        path = [part.strip() for part in section.split(">") if part.strip()]
        rows.append((path, "paragraph", str(fact.get("text", ""))))
    return _facts_from_rows(rows, page_title=page_title)


def extract_legacy_trivia_facts(sources: Iterable[Mapping]) -> tuple[list[dict], bool, str | None]:
    """Project the previous verbose metadata without another web request."""
    rows, found, source_url, page_title = [], False, None, None
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        source_url = source_url or source.get("url")
        page_title = page_title or source.get("page_title")
        for block in source.get("blocks", []):
            if not isinstance(block, Mapping):
                continue
            path = block.get("section_path", [])
            if isinstance(path, list) and is_trivia_path(path):
                found = True
            rows.append((
                path if isinstance(path, list) else [],
                str(block.get("block_type", "paragraph")),
                str(block.get("text", "")),
            ))
    return _facts_from_rows(rows, page_title=page_title), found, source_url
