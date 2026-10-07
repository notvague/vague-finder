"""Conservative, source-bound evidence selection for Context search results.

Dense similarity retrieves candidates; it does not certify that a sentence
answers the user's clue. Only a matching document subject and fact-level event
are shown. Unsupported paraphrases abstain rather than inventing an explanation.
This checks relevance to a query, not the truth of a third-party article.
"""

from __future__ import annotations

import re
import unicodedata
from math import ceil
from typing import TYPE_CHECKING, Sequence
from urllib.parse import urlsplit

from src.backend.schemas.query import ContextClue
from src.retrieval.context_query import (
    context_media_description_requires_support, context_media_target_terms,
    is_media_usage_relation,
)
from src.retrieval.context_media_match import media_name_in_text

if TYPE_CHECKING:
    from src.backend.schemas.search import ContextEvidence
    from src.retrieval.context_qdrant_search import ContextFactHit
    from src.retrieval.context_route import ContextRouteHit


# Limited, explicit work-title equivalences. Similarity or a matching song
# title is not allowed to manufacture an equivalence between two works.
_ALIASES = (
    ("크레용 신짱", "짱구는 못말려", "짱구 애니메이션"),
)
_MEDIA_SUBJECT = re.compile(
    r"^(?:옛날\s+)?(?:(?:드라마|영화|애니(?:메이션)?|게임)\s+)?"
    r"(.{2,45}?)\s+(?:OST|오에스티|삽입곡|배경\s*음악|BGM|주제가|테마곡)(?:\s|$)",
    re.I,
)
_GENERIC = {
    "노래", "곡", "음악", "가수", "남자", "여자", "기억", "제목", "찾아줘",
    "있었는데", "있었다", "있어", "같아", "같은", "거기", "옛날", "그때",
    "드라마", "영화", "애니메이션", "애니", "게임", "무슨", "어떤",
    "나중에", "당시", "OST", "BGM", "삽입곡", "배경음악", "제작",
    "뮤직비디오", "뮤비", "MV", "밈", "무대", "공연", "커버", "버전",
    "장면", "작품", "그곳", "그런", "나오는", "나왔던", "노래였는데",
    "축제", "앙코르", "앵콜", "다시", "했다", "올라", "불렀다",
    "보며", "울던", "이야기", "같은데", "거였어", "있었어",
}
_PARTICLES = ("으로는", "에서는", "에게서", "으로", "에서", "에게", "까지", "라는", "이라", "하고", "인데", "였다", "으로", "에는", "이랑", "를", "을", "은", "는", "이", "가", "에", "의", "도")
_NEGATIVE = re.compile(r"(?:아니었|아니라|아니고|아닌|아니다|없었|없다|않았|않는다)")
_DETAIL_NOISE = {
    "노래", "곡이", "곡이었는데", "기억나", "기억", "뭐였지", "있었어",
    "있었는데", "있었다고", "들었어", "들었다", "했는데", "했던", "그전",
    "그때", "전부", "나중", "다시", "말이", "중에", "도무지", "생각",
    "같은데", "이야기였어", "들려", "찾아줘", "나온", "부른", "남자가",
    "여자가", "바람", "하다", "곡을", "곡에", "개나", "에도", "까지",
}
_FLEXIBLE_ENDINGS = (
    "하자고", "했다고", "했는데", "한다고", "하려고", "이었다고",
    "였는데", "었는데", "았는데", "어서는", "어서", "하던", "하는",
    "했다", "한", "된", "로", "와", "과", "이", "가", "에",
)
_QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,40})[\"'“”‘’]")
_ACRONYM = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,}(?![A-Za-z0-9])")
_PUBLIC_OFFICE = re.compile(
    r"[가-힣]{2,10}(?:시장|도지사|구청장)|대통령|국회의원"
)
_NUMERIC = re.compile(r"(?<![A-Za-z0-9])\d+(?:,\d{3})*(?![A-Za-z0-9])")
_NUMERIC_UNIT = re.compile(r"\s*(년|월|일|위|개|회|억|만|분|초|집)")
_NAMED_WORK = re.compile(r"[<〈《]([^<>〈〉《》]{2,80})[>〉》]")
# Specific event conditions cannot be replaced by generic shared names or a
# broad heading. These are relation cues, not song-ID or query-ID exceptions.
_EVENT_CONDITIONS = (
    (re.compile(r"위문열차"), re.compile(r"위문열차")),
    (re.compile(r"군\s*(?:복무|입대)|입대"), re.compile(r"군\s*(?:복무|입대)|입대|복무")),
    (re.compile(r"올킬|all\s*kill", re.I), re.compile(r"올킬|all\s*kill", re.I)),
    (re.compile(r"디자인|설계"), re.compile(r"디자인|설계")),
    (re.compile(r"(?:판권|저작권|라이선스)[^,;.!?。！？\n]{0,25}"
                r"(?:만료|종료|끝났|끝난)|"
                r"(?:만료|종료)[^,;.!?。！？\n]{0,15}(?:판권|저작권|라이선스)"),
     re.compile(r"(?:판권|저작권|라이선스)[^;.!?。！？\n]{0,25}"
                r"(?:만료|종료|끝났|끝난)|"
                r"(?:만료|종료)[^;.!?。！？\n]{0,15}(?:판권|저작권|라이선스)")),
    (re.compile(r"삭제(?:된|됐|되었|되|한|했)|제거(?:된|됐|한|했)"),
     re.compile(r"삭제|제거")),
)

_REPORTED_SPEECH = re.compile(
    r"(?:라고|다고|라며|다며)\s*(?:말(?:했|했던|하|하기)|"
    r"이야기(?:했|했던|하)|밝(?:혔|힌|히)|설명(?:했|한|하)|언급(?:했|한|하))",
)
_SPEECH_NOISE = {
    "그렇게", "이렇게", "정말", "너무", "아주", "이번", "함께", "그때",
    "그것", "이것", "그런", "이런", "우리", "노래", "가수", "곡이",
}
_SPEECH_ENDINGS = ("다면서요", "다면서", "다고", "이라며", "라며", "다며")
_SPEECH_PARTICLES = ("들한테", "들에게", "들에겐", "한테", "에겐", "들")


def _speech_supported(query: str, clause: str) -> bool:
    """Match remembered speech content, rather than just its concert/venue.

    Only explicit reported-speech syntax is used. Quoted phrases are handled
    by the existing quotation check too. Unsupported paraphrases abstain; no
    model-generated explanation or new retrieval request is introduced.
    """
    reports = list(_REPORTED_SPEECH.finditer(query))
    for report in reports:
        prefix = query[:report.start()].rstrip(" \"'“”‘’?!")
        quotes = list(_QUOTED.finditer(query[:report.start()]))
        if quotes and not query[quotes[-1].end():report.start()].strip():
            content = quotes[-1].group(1)
        else:
            # The remembered assertion follows the location/performer. Do not
            # let shared words such as a concert name satisfy its contents.
            content = re.split(r"[;.!?。！？\n]|(?:에서|에서는|에선)\s+", prefix)[-1]
            content = re.sub(r"^.{2,35}?(?:이|가|은|는)\s+", "", content)
        terms = set()
        for term in _details(content):
            for ending in (*_SPEECH_ENDINGS, *_SPEECH_PARTICLES):
                if term.endswith(ending):
                    term = term[:-len(ending)]
                    break
            if len(term) >= 2 and term not in _SPEECH_NOISE:
                terms.add(term)
        if not terms:
            return False
        minimum = min(len(terms), max(2, ceil(len(terms) * 0.50)))
        if sum(_term_in_fact(term, _compact(clause)) for term in terms) < minimum:
            return False
    return True

# A survey/ranking about OSTs does not assert use in the remembered work.
_MEDIA_USAGE_ASSERTION = re.compile(
    r"삽입|배경\s*음악|BGM|주제가|테마곡|등장곡|로고송|선거송|"
    r"엔딩곡|오프닝곡|사운드\s*트랙|흐르|흐른|흘러|흘렀|사용|수록|"
    r"나왔|나오던|모티브|개표방송|춤을\s*추|춤추|"
    r"OST\s*(?:로|으로|이다|였다|였|이었|인\s*곡)", re.I,
)
_ADAPTATION_ASSERTION = re.compile(
    r"개사|(?:가사|노랫말|가삿말).{0,40}(?:바꿔|바꾸|바꾼|변경|고쳐|수정)",
)
_WITHDRAWAL_REQUEST = re.compile(r"삭제(?:된|됐|되었|되|한|했)|제거(?:된|됐|한|했)")
_TRACK_WITHDRAWAL = re.compile(r"(?:곡|노래|음원)[^;.!?。！？\n]{0,35}(?:삭제|제거)")


def _media_assertion_supported(clue: ContextClue, clause: str) -> bool:
    if _MEDIA_USAGE_ASSERTION.search(clause):
        return True
    if (_ADAPTATION_ASSERTION.search(clue.search_query)
            and _ADAPTATION_ASSERTION.search(clause)):
        return True
    # Withdrawal/adaptation statements answer those events only. They cannot
    # turn an unrelated article deletion into a generic OST-use citation.
    return bool(_WITHDRAWAL_REQUEST.search(clue.search_query)
                and _TRACK_WITHDRAWAL.search(clause))
_MEDIA_DETAIL_NOISE = {
    "초반", "중반", "후반", "년대", "옛", "오래된", "유명", "유명한",
    "특별한", "가진", "주인공", "배경", "남성", "여성", "보컬", "솔로",
    "혼자", "같이", "밴드", "랩", "발라드", "리메이크", "원곡", "없이",
    "노래", "곡이었어", "있었던", "있다고", "때문", "그랬어",
}


def _media_details_supported(clue: ContextClue, clause: str, anchors: set[str]) -> bool:
    if not context_media_description_requires_support(clue):
        return True
    distinctive = {
        term for term in anchors
        if term not in _MEDIA_DETAIL_NOISE
        and not term.isdigit()
        and not re.fullmatch(r"\d+(?:년|년대|월|일|위|회|집)?", term)
    }
    compact = _compact(clause)
    # At least two supplied event details, in the same sentence. A date and
    # the generic word OST must not satisfy a description of a particular work.
    return (len(distinctive) >= 2
            and sum(_term_in_fact(term, compact) for term in distinctive) >= 2)

# Relation cues must occur in the FACT TEXT, not merely the section heading.
# A shared heading plus an unrelated vocal-range fact must not become an OST
# citation. The categories are a second independent check, not proof by itself.
_FAMILIES = (
    (re.compile(r"제작.?활동|밴드\s*해체|활동에\s*얽힌\s*사건", re.I),
     {"influence", "production", "other"},
     re.compile(r"해체|분양|역주행|재개|복귀|음악을\s*하|활동|제작", re.I)),
    (re.compile(r"삽입곡|배경음악|OST|오에스티|BGM|주제가|테마곡|등장곡|"
                r"로고송|선거송|엔딩곡|오프닝곡|사운드\s*트랙|사용", re.I),
     {"media_usage", "influence", "version", "other"},
     re.compile(r"삽입|배경\s*음악|OST|BGM|주제가|테마곡|등장곡|"
                r"로고송|선거송|엔딩곡|오프닝곡|사운드\s*트랙|"
                r"흐르|흐른|흘러|흘렀|사용|수록|나왔|나오던|모티브|"
                r"개표방송|춤을\s*추|춤추|삭제|제거|개사|"
                r"(?:가사|노랫말).{0,40}(?:바꿔|바꾸|바꾼|변경|고쳐|수정)", re.I)),
    (re.compile(r"뮤직비디오|뮤비|MV|촬영|NG", re.I),
     {"music_video", "other"},
     re.compile(r"뮤직\s*비디오|뮤비|\bMV\b|촬영|장면|안무|NG|영상", re.I)),
    (re.compile(r"제작|발매|작곡|작사|녹음|기획|활동", re.I),
     {"production", "musical_detail", "record", "other"},
     re.compile(r"제작|작곡|작사|녹음|발매|기획|데모|가이드|퇴짜|버전|"
                r"완성|원키|공개|후속곡|타이틀|변경", re.I)),
    (re.compile(r"보컬\s*참여|코러스|백\s*보컬", re.I),
     {"musical_detail", "production", "other"},
     re.compile(r"코러스|백\s*보컬|목소리|참여|피처링", re.I)),
    (re.compile(r"음원.?보컬\s*여담", re.I),
     {"musical_detail", "other"},
     re.compile(r"발음|목소리|음원|소리", re.I)),
    (re.compile(r"무대|공연|방송|축제|페스티벌|앙코르|콘서트", re.I),
     {"performance", "musical_detail", "other"},
     re.compile(r"무대|공연|방송|축제|페스티벌|앙코르|앵콜|불렀|안무|우승|"
                r"콘서트|라이브", re.I)),
    (re.compile(r"커버|리메이크|답가|원곡|버전|번안", re.I),
     {"version", "performance", "other"},
     re.compile(r"커버|리메이크|답가|원곡|번안|버전|불렀|재해석", re.I)),
    (re.compile(r"밈|유행|패러디|챌린지|역주행", re.I),
     {"meme", "other"},
     re.compile(r"밈|유행|패러디|챌린지|역주행|회자|잘못\s*부|화제", re.I)),
    (re.compile(r"기록|차트|1위|우승|영향|평가", re.I),
     {"record", "influence", "media_usage", "other"},
     re.compile(r"기록|차트|\d+위|우승|영향|평가|선정|인기|재개", re.I)),
)


def _compact(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFKC", value).casefold())


def _forms(value: str) -> tuple[str, ...]:
    key = _compact(value)
    for group in _ALIASES:
        if key in {_compact(alias) for alias in group}:
            return tuple(_compact(alias) for alias in group)
    return (key,)


def _subject(clue: ContextClue) -> str:
    if clue.target.strip():
        value = clue.target.strip()
        return "" if value in _GENERIC else value
    match = _MEDIA_SUBJECT.search(clue.search_query.strip())
    value = match.group(1).strip() if match else ""
    return "" if value in _GENERIC else value


def _details(query: str) -> set[str]:
    tokens = set()
    for raw in re.findall(r"[가-힣]{2,}|[A-Za-z0-9]{2,}", query):
        value = raw.casefold()
        for suffix in _PARTICLES:
            if value.endswith(suffix) and len(value) - len(suffix) >= 2:
                value = value[:-len(suffix)]
                break
        if len(value) >= 2 and value.upper() not in _GENERIC and value not in _GENERIC:
            tokens.add(value)
    return tokens


def _anchors(clue: ContextClue, subject: str) -> set[str]:
    """Distinctive words supplied by the user, excluding the work and boilerplate.

    This is an abstention rule, never a claim that token overlap proves a fact.
    The final citation must also pass the subject, relation, and URL checks.
    """
    subject_key = _compact(subject)
    terms = _details(clue.search_query) - _details(clue.relation)
    return {
        term for term in terms
        if term not in _DETAIL_NOISE
        and term not in _GENERIC
        and not (subject_key and _compact(term) in subject_key)
    }


def _term_in_fact(term: str, fact: str) -> bool:
    key = _compact(term)
    if len(key) < 2:
        return False
    if key in fact:
        return True
    # Korean particles and verbal endings differ across query and citation.
    # Never shorten an entity such as 서울시장 or a foreign title such as FIESTA.
    if len(key) >= 3 and all('가' <= c <= '힣' for c in key):
        for ending in _FLEXIBLE_ENDINGS:
            if key.endswith(ending) and len(key) - len(ending) >= 2:
                return key[:-len(ending)] in fact
    return False


def _specificity(clue: ContextClue) -> int:
    return len(_anchors(clue, _subject(clue)))


def _safe_namuwiki_url(value: str) -> bool:
    if not value or any(ord(c) < 33 for c in value):
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "https" and parsed.hostname == "namu.wiki"
                and parsed.username is None and parsed.password is None
                and parsed.port in (None, 443) and parsed.path.startswith("/w/"))
    except ValueError:
        return False


def _numbers_supported(query: str, clause: str) -> bool:
    """Match whole numeric values and the stated unit, never digit substrings."""
    normalized = re.sub(r"(?<=\d),(?=\d)", "", clause)
    for match in _NUMERIC.finditer(query):
        number = match.group().replace(",", "")
        unit = _NUMERIC_UNIT.match(query, match.end())
        suffix = r"\s*" + re.escape(unit.group(1)) if unit else ""
        if not re.search(r"(?<!\d)" + re.escape(number) + r"(?!\d)" + suffix,
                         normalized):
            return False
    return True


def _related(clue: ContextClue, fact: ContextFactHit, song_id: str) -> bool:
    if (clue.confidence <= 0 or fact.song_id != song_id
            or not fact.record_id.startswith(f"nw:{song_id}:")
            or not fact.fact_text.strip() or not _safe_namuwiki_url(fact.source_url)):
        return False

    relation = clue.relation.strip()
    # Grounded model-only relations can retain the user's verb ("쓰인",
    # "나온", ...). They must receive the same media checks as an OST label.
    family_relation = "다른 작품에 사용" if is_media_usage_relation(relation) else relation
    family = next((item for item in _FAMILIES if item[0].search(family_relation)), None)
    if family is None or fact.category not in family[1]:
        return False
    text = unicodedata.normalize("NFKC", fact.fact_text)
    if _NEGATIVE.search(text) or _NEGATIVE.search(clue.search_query):
        # A bare token match cannot tell which part of a negated statement
        # applies; abstain instead of displaying it as positive evidence.
        return False

    clauses = [part for part in re.split(r"(?<=[.!?。])\s+|[;；]", text)
               if family[2].search(part)]
    if not clauses:
        return False
    subject = _subject(clue)
    # The catalogue already binds this fact to its song and artist. Their
    # names need not be repeated in the article sentence that supplies the
    # event; require the *remaining* event details in that same sentence.
    identity = (fact.title, *fact.artists)
    identity_compact = tuple(_compact(value) for value in identity)
    identity_terms = set().union(*(_details(value) for value in identity))
    anchors = _anchors(clue, subject) - identity_terms
    quoted = [_compact(value) for value in _QUOTED.findall(clue.search_query)
              if _compact(value) not in identity_compact]
    acronyms = [_compact(value) for value in _ACRONYM.findall(clue.search_query)
                if value not in {"OST", "BGM", "MV"}
                and not any(_compact(value) in key for key in identity_compact)]
    offices = [_compact(value) for value in _PUBLIC_OFFICE.findall(clue.search_query)]
    # Work + relation alone is sufficient only for a genuinely simple clue.
    # For a specific event, require multiple details in the *same* fact clause.
    minimum = (min(len(anchors), 1) if subject and len(anchors) <= 2
               else min(len(anchors), max(2, ceil(len(anchors) * 0.30))))
    if not subject and not anchors:
        return False
    for part in clauses:
        compact = _compact(part)
        if is_media_usage_relation(relation):
            if not _media_assertion_supported(clue, part):
                continue
            if not _media_details_supported(clue, part, anchors):
                continue
        if subject and not any(form in compact for form in _forms(subject)
                               if len(form) >= 2):
            continue
        if any(value not in compact for value in (*quoted, *acronyms, *offices)):
            continue
        if not _numbers_supported(clue.search_query, part):
            continue
        if any(cue.search(clue.search_query) and not required.search(part)
               for cue, required in _EVENT_CONDITIONS):
            continue
        if not _speech_supported(clue.search_query, part):
            continue
        named_works = _NAMED_WORK.findall(part)
        if (named_works and re.search(r"제작|발매|후속|리메이크|답가|버전", relation)
                and fact.title.strip()
                and not any(_compact(fact.title) == _compact(work) for work in named_works)
                and not re.search(r"이\s*(?:노래|곡)|해당\s*곡|본\s*곡", part)):
            # An article about A can discuss B/C. A catalogue binding alone
            # does not turn that sentence into an event about A.
            continue
        if sum(_term_in_fact(term, compact) for term in anchors) >= minimum:
            return True
    return False


def context_candidate_matches_media_description(
    hit: ContextRouteHit, *, query_clues: Sequence[ContextClue],
) -> bool:
    """Gate media promotion against its literal work or remembered details.

    Sparse-only hits remain useful for other queries. They cannot substantiate
    a qualified description of an unnamed work or independently add an OST
    vote to an unrelated production fact. No new index request is made here.
    """
    snapshots = (hit.fused_hit, *(fused for _, fused in hit.alternatives))
    facts = {fact.record_id: fact
             for fused in snapshots if fused.song_id == hit.song_id
             and fused.dense_song is not None
             for fact in (fused.dense_facts or (fused.dense_song.best_fact,))}
    required = [clue for clue in query_clues if clue.confidence > 0
                and context_media_description_requires_support(clue)]
    # A named-work query may retain Sparse-only recall, but only when that
    # same song's retrieved profile actually includes the supplied name.
    # Matching a generic OST token is insufficient. A Dense sentence can
    # supply the anchor too; it still needs _related() before being displayed.
    for clue in query_clues:
        if clue.confidence <= 0:
            continue
        names = context_media_target_terms(clue)
        if not names:
            continue
        profile_terms = (
            term for fused in snapshots if fused.song_id == hit.song_id
            and getattr(fused, "sparse_profile", None) is not None
            and fused.sparse_profile.song_id == hit.song_id
            for term in getattr(fused.sparse_profile, "sparse_terms", ())
        )
        anchored = any(media_name_in_text(name, term) for term in profile_terms for name in names)
        if not anchored:
            anchored = any(media_name_in_text(name, fact.fact_text)
                           for fact in facts.values() if fact.song_id == hit.song_id
                           for name in names)
        if not anchored:
            return False
    return all(any(_related(clue, fact, hit.song_id) for fact in facts.values())
               for clue in required)


def select_context_fact_for_result(
    hit: ContextRouteHit, *, query_clues: Sequence[ContextClue] | None = None,
) -> ContextFactHit | None:
    """Select one relevant Dense fact from this request, or abstain.

    No new Qdrant search: every fact belongs to the same snapshot that made
    the candidate ranking. The winning clue is tested first; another clue for
    the same song can supply evidence without becoming an extra ranking vote.
    """
    snapshots = ((hit.clue, hit.fused_hit), *hit.alternatives)
    choices = snapshots
    if query_clues:
        strongest = max((_specificity(clue) for clue in query_clues
                         if clue.confidence > 0), default=0)
        # A generic OST fact cannot substitute for a separately stated,
        # specific production or performance event in the same request.
        choices = tuple(
            (clue, fused) for clue in query_clues if clue.confidence > 0
            and _specificity(clue) == strongest
            for _, fused in snapshots
        )
    seen: set[tuple[str, str, str, str]] = set()
    for clue, fused in choices:
        if fused.song_id != hit.song_id or fused.dense_song is None:
            continue  # Sparse profiles can never be fact evidence.
        facts = fused.dense_facts or (fused.dense_song.best_fact,)
        for fact in facts:
            key = (fact.record_id, clue.target, clue.relation, clue.search_query)
            if key in seen:
                continue
            seen.add(key)
            if _related(clue, fact, hit.song_id):
                return fact
    return None


def context_evidence_for_result(
    hit: ContextRouteHit, *, query_clues: Sequence[ContextClue] | None = None,
) -> ContextEvidence | None:
    """Keep the API evidence shape while separating selection from rendering."""
    fact = select_context_fact_for_result(hit, query_clues=query_clues)
    if fact is None:
        return None
    from src.backend.schemas.search import ContextEvidence

    return ContextEvidence(
        record_id=fact.record_id, fact_text=fact.fact_text,
        source_url=fact.source_url, section=fact.section, category=fact.category,
    )
