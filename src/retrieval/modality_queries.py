"""Modality-specific query prompt safeguards.

The text-to-image and text-to-audio encoders do not consume the same evidence.
This module keeps their prompts separate, validates strong cross-modal leakage,
and provides a conservative rule-based fallback when Gemini is unavailable.

The fallback intentionally prefers an empty modality prompt over forwarding the
whole Korean query.  Text retrieval still receives the complete original query,
so an unknown visual/audio expression degrades to text-only retrieval instead of
polluting another embedding space.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.retrieval.lyrics_query import unquoted_lyric_start


class ModalityQueryValidationError(ValueError):
    """Raised when a generated embedding prompt violates modality boundaries.

    ``modality``("image"·"audio")와 ``kind``("missing"·"leak")는 어느 쪽이 어떻게
    어긋났는지다. 분석기는 재시도를 다 쓴 뒤 이 응답을 버리지 않고 그 모달리티만
    내려놓는다(`apply_modality_query_safeguards(..., degrade=True)`).
    """

    def __init__(self, message: str, *, modality: str = "", kind: str = "") -> None:
        super().__init__(message)
        self.modality = modality
        self.kind = kind


# Literal lyric words remain Text evidence even if they name a mood,
# instrument or album picture. Mask locally attributed lyric spans only.
# Preserve offsets for Context's artwork gate; the original query is retained.
_LYRIC_QUOTATION_RE = re.compile(
    r'"[^"]*"|“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|`[^`]+`|'
    r"(?<!\w)'(?:[^']|(?<=[A-Za-z])'(?=[A-Za-z]))*'(?![A-Za-z0-9])"
)
_LYRIC_QUOTE_PREFIX_RE = re.compile(
    r"(?:가사|가삿말|노랫말|후렴(?:구)?)"
    r"[^\"'“”‘’「」『』`.,;!?。！？\n]{0,28}$"
)
_LYRIC_QUOTE_SUFFIX_RE = re.compile(
    r"^\s*(?:(?:이)?라는|(?:이)?라고|같은|비슷한|이런)?\s*"
    r"(?:가사|가삿말|노랫말|구절)"
)
_SUNG_QUOTE_SUFFIX_RE = re.compile(
    r"^\s*(?:(?:이)?라는|(?:이)?라고|하고)?\s*"
    r"(?:가사|구절|말|나오|나와|나왔|들리|들렸|반복)"
)
_LYRIC_QUOTE_GAP_RE = re.compile(r"^[\s,，/]*(?:(?:와|과|또는|그리고|및|또)[\s,，/]*)?$")
_SEPARATE_DESCRIPTOR_RE = re.compile(
    r"보컬|목소리|음색|반주|템포|리듬|음질|표지|커버|자켓|재킷|색감|곡명|제목"
)
_NEGATED_LYRIC_ATTRIBUTION_RE = re.compile(r"아닌|아니라|아니고|말고|대신")
_UNQUOTED_LYRIC_RE = re.compile(
    r"(?:가사|가삿말|노랫말|후렴(?:구)?)(?:에는|에서|에|이|가|은|는)\s*"
    r"(?P<phrase>[^\"'“”‘’「」『』`.,;!?。！？\n]{2,180}?)\s*"
    r"(?:이라는|이라고|라고|라는|하고)\s*"
    r"(?:말|가사|가삿말|노랫말|구절|부분|소리|나오|나와|나왔|들리|들렸)"
)
_LYRIC_FRAGMENT_BEFORE_RE = re.compile(
    r"(?:^|[.,;!?。！？\n]\s*)(?P<phrase>[^\"'“”‘’「」『』`.,;!?。！？\n]{2,160}?)\s*"
    r"(?:이라는|라는)\s*(?:가사|가삿말|노랫말|구절)"
)
_LYRIC_LINE_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:가사|가삿말|노랫말)\s*[:：]\s*"
    r"(?P<phrase>[^\n]+?)(?=\s+(?:보컬|반주|음향|표지|앨범|커버)\s*[:：]|\n|$)"
)


def _mask_lyric_literals(query: str) -> str:
    """Blank lyric literals once, at a public entry point.

    Do not mask the result again: once a quotation is blanked, the description
    before it reads like an unquoted fragment. Helpers take the masked text.
    """
    text = str(query or "")
    spans: list[tuple[int, int]] = []
    active_end: int | None = None
    for quote in _LYRIC_QUOTATION_RE.finditer(text):
        prefix = _LYRIC_QUOTE_PREFIX_RE.search(text[:quote.start()])
        direct = (prefix is not None
                  and _SEPARATE_DESCRIPTOR_RE.search(prefix.group()) is None
                  and _NEGATED_LYRIC_ATTRIBUTION_RE.search(prefix.group()) is None)
        if direct and prefix.group().startswith("후렴") and not re.search(r"가사|가삿말|노랫말", prefix.group()):
            # A quoted chorus sound is not a lyric without a sung-words predicate.
            direct = _SUNG_QUOTE_SUFFIX_RE.search(text[quote.end():]) is not None
        continued = (active_end is not None
                     and _LYRIC_QUOTE_GAP_RE.fullmatch(text[active_end:quote.start()]) is not None)
        if direct or continued or _LYRIC_QUOTE_SUFFIX_RE.search(text[quote.end():]):
            spans.append(quote.span())
            active_end = quote.end()
        else:
            active_end = None
    spans.extend(match.span("phrase") for match in _LYRIC_LINE_HEADER_RE.finditer(text))
    # An unquoted fragment has no marked left edge: a leading clause that
    # describes the song ("피아노 반주가 있는 발라드인데 사랑해라는 가사") is not
    # sung. Text after a lyric marker ("가사에 …") stays literal as a whole.
    for pattern, anchored in ((_UNQUOTED_LYRIC_RE, True), (_LYRIC_FRAGMENT_BEFORE_RE, False)):
        for match in pattern.finditer(text):
            start, end = match.span("phrase")
            spans.append((unquoted_lyric_start(text, start, end, anchored=anchored), end))
    masked = list(text)
    for start, end in spans:
        masked[start:end] = [c if c in "\r\n" else " " for c in text[start:end]]
    return "".join(masked)


# 리메이크를 뜻하는 "커버곡 / 커버 노래 / 커버 버전"을 표지 단서에서 제외한다.
# 단독 "커버"에만 적용한다 — "앨범 커버"는 앞의 '앨범'이 이미 표지를 확정하므로
# 뒤에 노래/곡/음원이 와도("빨간 앨범 커버 노래") 표지 질의로 봐야 한다.
# 리메이크를 뜻하는 동사·명사류. "앨범 커버곡 / 앨범 커버한 / 앨범 커버 버전"처럼
# 앞에 '앨범'이 붙어도 표지가 아니므로 양쪽 분기에 모두 적용한다.
_COVER_VERB_SUFFIX = (
    r"(?:곡|송|한|했|하는|하던|해서|해본|해봤|"
    # "커버해 우승" is a cover performance. Keep the bare ending bounded so
    # "앨범 커버 해상도" still means album artwork.
    r"해(?=\s|$|[,.!?])|해도|해줘|"
    r"할|불러|불렀|부른|를\s*(?:한|했|하는|할|불러|불렀|부른)|했던|버전)"
)

# 위 + 명사류. "커버 노래"는 리메이크지만 "앨범 커버 노래"는 표지 질의라서
# 노래/음원/무대는 단독 '커버'에만 적용한다.
_COVER_SONG_SUFFIX = _COVER_VERB_SUFFIX + r"|노래|음원|무대"

# A cover dance is a performance event, in either Korean word order. Do not
# turn its bare "커버" into album artwork or its "댄스" into a recording genre.
# Keep the original query intact for Context and inspect each cover occurrence
# separately, so an independent album-cover description still opens image.
COVER_DANCE_PATTERN = (
    r"(?:커버\s*(?:댄스|댄싱|춤|안무)|(?:댄스|댄싱|춤|안무)\s*커버)"
)
_COVER_DANCE_RE = re.compile(COVER_DANCE_PATTERN, re.IGNORECASE)

# Cover competitions and recorded performances use the same Korean noun as
# album artwork. Share their spans with Context; do not special-case a song,
# platform, programme or evaluation question. Quotation marks and line breaks
# can occur within a competition name.
COVER_CONTEST_PATTERN = r"커버(?:['\"’”])?[\s·-]*(?:서바이벌|공모전|경연|대회|콘테스트)"
COVER_RECORDING_PATTERN = r"커버\s*(?:영상|라이브|연주|공연)"
_COVER_EVENT_RE = re.compile(
    rf"(?:{COVER_CONTEST_PATTERN}|{COVER_RECORDING_PATTERN})", re.IGNORECASE,
)
_COVER_PERFORMANCE_RE = re.compile(
    rf"(?:{COVER_DANCE_PATTERN}|{COVER_CONTEST_PATTERN}|{COVER_RECORDING_PATTERN})",
    re.IGNORECASE,
)
_ALBUM_ART_BEFORE_COVER_RE = re.compile(
    r"(?:앨범(?:의)?|표지|자켓|재킷|아트워크)\s*$", re.IGNORECASE,
)
# Bare "커버" is ambiguous. Require a physical appearance description before
# using it as an image anchor; "가수의 커버가 화제" is a rendition, and a bare
# recollection without either kind of evidence stays available to Text.
_BARE_COVER_ART_DETAIL_RE = re.compile(
    r"(?:색깔|색감|색상|디자인|아트(?:워크)?|이미지|사진|그림|일러스트|"
    r"삽화|드로잉|수채화|유화|초상|얼굴|실루엣|글씨|글자|손글씨|"
    r"필기체|폰트|타이포|질감|인쇄|종이|캔버스|꽃|꽃잎|도형|"
    r"흑백|단색|파스텔|분홍|핑크|파랑|파란|파랗|푸른|보라|"
    r"빨강|빨간|빨갛|붉은|노랑|노란|노랗|초록|녹색|주황|"
    r"검정|검은|하양|하얀|하얗|흰색|회색|베이지|갈색)", re.IGNORECASE,
)
_COVER_LITERAL_TEXT_START_RE = re.compile(
    r"^\s*(?:가사|노랫말|제목|곡명)(?:에|에는|속에)\s*", re.IGNORECASE,
)
_COVER_SOUND_ANALOGY_RE = re.compile(
    r"(?:그림|사진|색|빛|꽃).{0,16}(?:처럼|같은|같이).{0,16}"
    r"(?:소리|음색|음질|사운드|멜로디|보컬|노래|음악)", re.IGNORECASE,
)
_BARE_COVER_MOOD_RE = re.compile(
    r"커버(?:가|는|의)?\s*(?:어둡|어두운|밝|차갑|차가운|따뜻|몽환|화려|심플|미니멀)"
    r"[^,;.!?。！？\n]{0,30}(?:느낌|분위기|톤)(?:이|였)|"
    r"(?:어두운|밝은|차가운|따뜻한|몽환적인|화려한|심플한|미니멀한)\s*커버",
    re.IGNORECASE,
)
_COVER_HEARD_DESCRIPTOR_RE = re.compile(
    r"(?:소리|음색|음질|사운드|멜로디|보컬|반주|창법)", re.IGNORECASE,
)


def normalize_cover_performance_spacing(text: str) -> str:
    """Fold formatting inside a performance phrase, not sentence boundaries.

    Context splits independent clauses at newlines. A newline in '커버\n대회'
    should not split this single event. The caller keeps the original query.
    """
    return _COVER_PERFORMANCE_RE.sub(lambda match: " ".join(match.group().split()), text)

_COVER_CONTEXT_RE = re.compile(
    # 1) "앨범 표지 / 앨범 커버 / 앨범 자켓 ..."
    #    '앨범'이 붙어도 리메이크 표현(커버곡/커버한/커버 버전)은 제외한다.
    r"(?:앨범\s*(?:표지|커버(?!\s*(?:" + _COVER_VERB_SUFFIX + r"))|"
    r"자켓|재킷|아트|사진|이미지|앞면)|"
    # 2) 단독 "표지 / 커버 / 자켓" — 명사류까지 제외한다.
    r"(?:표지|커버(?!\s*(?:" + _COVER_SONG_SUFFIX + r"))|"
    r"자켓|재킷)(?:에|가|는|의|에서|였|였던|처럼|인데)?)",
    re.IGNORECASE,
)


def _bare_cover_has_appearance(text: str, cover: re.Match[str]) -> bool:
    left, right = 0, len(text)
    for boundary in _CLAUSE_BREAK_RE.finditer(text):
        if boundary.end() <= cover.start():
            left = boundary.end()
        elif boundary.start() >= cover.end():
            right = boundary.start()
            break
    # Each occurrence owns its local evidence. A later independent album
    # cover must not retroactively turn an earlier cover performance into art.
    for other in _COVER_CONTEXT_RE.finditer(text, cover.end()):
        right = min(right, other.start())
        break
    clause = text[left:right].strip()
    if _COVER_LITERAL_TEXT_START_RE.match(clause):
        return False
    nearby = text[max(left, cover.start() - 45):min(right, cover.end() + 55)]
    nearby = _mask_performed_venue_names(nearby)
    appearance = _BARE_COVER_ART_DETAIL_RE.search(nearby) is not None
    mood = (_BARE_COVER_MOOD_RE.search(nearby) is not None
            and not _COVER_HEARD_DESCRIPTOR_RE.search(nearby))
    return not _COVER_SOUND_ANALOGY_RE.search(nearby) and (appearance or mood)


def _find_album_cover_context(text: str) -> re.Match[str] | None:
    dances = list(_COVER_DANCE_RE.finditer(text))
    events = list(_COVER_EVENT_RE.finditer(text))
    for cover in _COVER_CONTEXT_RE.finditer(text):
        if any(
            cover.start() < event.end() and event.start() < cover.end()
            for event in dances
        ):
            continue
        explicit_album_art = (
            cover.group().startswith("앨범")
            or _ALBUM_ART_BEFORE_COVER_RE.search(text[:cover.start()]) is not None
        )
        if not explicit_album_art and any(
            cover.start() < event.end() and event.start() < cover.end()
            for event in events
        ):
            continue
        if (not explicit_album_art and cover.group().startswith("커버")
                and not _bare_cover_has_appearance(text, cover)):
            continue
        return cover
    return None


def _artwork_reference_start(text: str) -> int | None:
    text = _NEGATED_COVER_CONTEXT_RE.sub(lambda match: " " * len(match.group()), text)
    match = _find_album_cover_context(text)
    return match.start() if match else None


def explicit_artwork_reference_start(query: str) -> int | None:
    """Return a literal artwork offset, ignoring negation and cover events."""
    return _artwork_reference_start(_mask_lyric_literals(query))


def has_explicit_artwork_reference(query: str) -> bool:
    """Share literal artwork gating with Context without interpreting scenes."""
    return explicit_artwork_reference_start(query) is not None

# 한 글자 낱말은 다른 낱말의 조각으로도 나온다. "초록·수록·알록달록"의 록, "손가락"의 락은
# 장르가 아니고, "이별·준비·기억해·달리는"의 별·비·해·달은 그림 속 사물이 아니다.
# 조각을 단서로 읽으면 모델이 내지 않은(낼 이유가 없는) 프롬프트를 요구하게 되고,
# 그 불일치는 분석 전체를 규칙 폴백으로 떨어뜨린다(results_v38: t009·t071·t086·t099, v09 n002).
# 그래서 앞에 한글이 붙어 있으면 받지 않는다. 붙여 쓰는 장르 이름만 따로 연다.
# "-록"으로 끝나는 일반 낱말(기록·목록·수록·초록…)은 끝이 없고 장르 이름은 정해져 있으므로,
# 막을 낱말이 아니라 **열 낱말**을 적는다. 접두는 겹칠 수 있다("팝펑크록", "얼터너티브 펑크 록").
_ROCK = (
    r"(?<![가-힣])(?:(?:하드|소프트|모던|펑크|포크|팝|인디|얼터너티브|얼터|브릿|"
    r"개러지|포스트|글램|프로그레시브|사이키델릭|블루스|아트|서던|일렉트로닉|일렉|"
    r"컨트리|재즈|심포닉|고딕|스페이스|서프|노이즈|매스|하드코어)\s*)*(?:록|락)"
)

# 조사. "도"와 "로"는 넣지 않는다 — "별도·해도·별로"가 흔한 다른 낱말이다.
_NOUN_PARTICLES = (
    r"(?:이|가|을|를|은|는|과|와|의|만|에|에서|에는|까지|랑|이랑|하고|처럼|같은|같이)"
)


def _one_syllable_noun(
    syllable: str,
    *,
    predicates: tuple[str, ...] = (),
    compounds: tuple[str, ...] = (),
) -> str:
    """한 글자 명사를 낱말로 쓰였을 때만 맞춘다.

    - 낱말 첫 글자이고 뒤가 조사·낱말 끝: "별이", "별들이", "별무늬", "달에서"
    - 그 명사의 서술어를 붙여 쓴 꼴(predicates): "비오는", "해뜨는" — "비 오는"과 다르게 읽지 않는다
    - 그 명사가 뒤에 붙는 합성 명사(compounds): "보름달", "별똥별"
    """
    word = (
        r"(?<![가-힣])" + syllable + r"(?:(?:들)?(?:무늬|그림|모양|빛)?"
        + _NOUN_PARTICLES + r"?(?![가-힣])"
        + (r"|(?=" + "|".join(predicates) + r")" if predicates else "")
        + r")"
    )
    return "(?:" + "|".join((*compounds, word)) + ")"


_STAR = _one_syllable_noun(
    "별",
    predicates=("뜨는", "뜬", "빛나는", "반짝이는", "쏟아지는"),
    compounds=("별똥별", "별자리", "샛별"),
)
_MOON = _one_syllable_noun(
    "달",
    predicates=("뜨는", "뜬"),
    compounds=("보름달", "초승달", "반달", "그믐달", "달밤"),
)
_SUN = _one_syllable_noun(
    "해",
    predicates=("뜨는", "뜬", "지는"),
    compounds=("해질녘", "해돋이", "해넘이"),
)
_RAIN = _one_syllable_noun(
    "비",
    predicates=("오는", "온", "오던", "내리는", "내린", "내리던", "맞는"),
    compounds=("봄비", "소낙비", "장대비", "이슬비", "보슬비", "가랑비"),
)

_VISUAL_DETAIL_RE = re.compile(
    r"(?:색|빛|컬러|흑백|단색|사진|그림|일러스트|삽화|스케치|드로잉|"
    r"수채화|유화|도트|픽셀|초상|얼굴|인물|실루엣|배경(?!\s*음악)|바탕|전경|"
    r"중앙|가운데|왼쪽|오른쪽|위쪽|아래쪽|한쪽|사선|구도|배치|글자|"
    r"글씨|손글씨|필기체|문자|숫자|폰트|타이포|질감|종이|캔버스|"
    r"하트|" + _STAR + "|" + _MOON + "|" + _SUN + r"|밤하늘|꽃|꽃잎|구름|"
    + _RAIN + r"|우산|바다|해변|도시|"
    r"건물|자동차|동물|고양이|강아지|콜라주|추상|기하학)",
    re.IGNORECASE,
)

# 사용자는 표지/커버라는 명사를 생략하고 정적인 아트워크만 묘사하기도 한다.
# 단일 색상이나 막연한 심상만으로 image 경로를 열지 않도록 서로 독립적인
# 시각 증거 축을 세어 판정한다.
_IMPLICIT_ARTWORK_EVIDENCE = {
    "medium": re.compile(
        r"(?:손(?:으로)?\s*그린|(?:연필|펜|붓|크레용|목탄)(?:으)?로\s*"
        r"(?:그린|그려)|그려\s*(?:진|져|놓)|그림|일러스트|삽화|스케치|"
        r"드로잉|수채화|유화|사진|도트|픽셀|콜라주|실루엣|초상|"
        # 칠한 재료·붓자국도 그림이라는 형태 근거다("물감을 붓으로 덧칠한 숲").
        r"물감|덧칠|붓질|붓\s*터치)",
        re.IGNORECASE,
    ),
    "surface": re.compile(
        r"(?:(?:종이|캔버스|인쇄|필름)\s*(?:의\s*)?질감|"
        r"종이(?:처럼|같이|같은)\s*(?:거칠|구겨|낡)|"
        r"(?:낡은|구겨진|거친)\s*종이|종이\s*같은\s*거친\s*질감|"
        r"질감(?:이|은|처럼)?\s*(?:거칠|매끈|오래|빈티지)|"
        r"거친\s*(?:종이|캔버스|표면))",
        re.IGNORECASE,
    ),
    "typography": re.compile(
        r"(?:손글씨|붓글씨|필기체|캘리그라피|폰트|타이포(?:그래피)?|"
        r"글씨\s*디자인|"
        r"(?:제목|글자|글씨|문자|숫자|알파벳|대문자).{0,18}"
        r"(?:적혀|쓰여|써\s*있|찍혀|배치)|"
        r"(?:적혀|쓰여|써\s*있|찍혀).{0,18}"
        r"(?:제목|글자|글씨|문자|숫자|알파벳|대문자))",
        re.IGNORECASE,
    ),
    "layout": re.compile(
        r"(?:배경(?!\s*음악)|바탕|전경|중앙|가운데|왼쪽|오른쪽|위쪽|"
        r"아래쪽|한쪽|"
        r"사선|대각선|구도|배치|화면.{0,8}나뉘|흩어져|겹쳐|"
        r"(?:작|크)게\s*(?:보이|그려|찍혀|배치|들어가))",
        re.IGNORECASE,
    ),
    "palette": re.compile(
        r"(?:흑백|단색|컬러풀|알록달록|파스텔|네온|형광|"
        r"분홍|핑크|파랑|파란|푸른|보라|빨강|빨간|붉은|노랑|노란|"
        r"초록|녹색|주황|오렌지|검정|검은|하양|하얀|흰색|회색|그레이|"
        r"베이지|갈색|브라운|남색|네이비)",
        re.IGNORECASE,
    ),
    "subject": re.compile(
        r"(?:하트|" + _STAR + r"|밤하늘|꽃|꽃잎|"
        r"구름|우산|바다|해변|도시|건물|자동차|동물|고양이|강아지|"
        r"얼굴|인물|사람|남자|여자|도형|원형|삼각형|사각형|기하학|"
        r"(?:동그란|빨간|파란|검은|하얀|큰|작은)\s*원)",
        re.IGNORECASE,
    ),
}

# "…한 이미지"로 끝나는 묘사 — '표지'라는 말 없이 그림 자체를 설명한 질의다
# (results_v38 t086·t099: 이 꼴을 못 읽어 모델이 낸 표지 프롬프트를 지웠다).
# '이미지'는 "청순한 이미지의 가수"처럼 **인상**도 뜻한다. 규칙이 참인데 모델이 표지 프롬프트를
# 내지 않으면 분석 전체가 폴백으로 떨어지므로, 애매하면 받지 않는다:
# - **절이 '이미지'(와 그 서술어)에서 끝나야 한다.** 받는 꼴을 늘어놓고 새는 꼴을 하나씩 막는 대신
#   "그 뒤로 절이 이어지지 않는다"를 조건으로 건다. 그래서 뒤 명사를 꾸미는 꼴은 종류와 상관없이
#   빠진다 — "이미지의 가수", "이미지였던 가수", "이미지가 있는 가수", "이미지 같아 보이는 가수",
#   "이미지가 있어 보이는 가수", "이미지였다고 하는 가수".
#   · 서술어 없이 '이미지'로 끝나면 그 뒤는 절의 끝이거나 말을 잇는 부사(근데·그래서 …)여야 한다
#     ("이미지 변신", "이미지 좋은"을 막는다).
#   · 문장을 맺는 서술어(`_IMAGE_FINAL`)는 **정말 맺었을 때만** 받는다 — 글자가 붙어 이어지거나
#     ("였다고", "야말로", "임에도") 그 어미의 보조 서술어가 따라오면("같아 보이는", "같지 않은",
#     "이미지다 보니", "였다 하는") 맺은 것이 아니다.
#     맺은 뒤에 새 문장이 오는 것은 마침표가 있든 없든 같다("이미지였어 제목이 기억 안 나").
#   · 다음 절로 넘어가는 서술어(`_IMAGE_CONNECTIVE`: -는데·-지만)는 그 자체가 절의 경계다.
#     `_CLAUSE_BREAK_RE`가 "이미지인데/이미지였고"는 이미 끊어 주고, 여기서 나머지를 받는다.
# - 같은 절에서 '이미지' 앞에 노래·듣기 낱말이 오면 노래가 주는 심상이다 — "노래가 … 같은 이미지야".
# - 시각 축은 그 절의 '이미지' **앞**에서만 센다(`_has_implicit_artwork_description`).
#
# 맺음꼴은 어간과 어미를 따로 적어 곱한다. 꼴을 하나씩 적으면 "같았지·있었지"처럼 빠지는 것이 생긴다.
_IMAGE_COPULA = r"(?:였|이었)"
_IMAGE_EXISTS = r"가\s*(?:있|그려져\s*있|찍혀\s*있)"
_IMAGE_RECALLED = _IMAGE_COPULA + r"던\s*(?:거|것|걸|듯)"   # "이미지였던 거 (같아)"
# 어간: 같-·같았-, 였-, 가 있-·가 있었- ("였던 거 같-" 포함)
_IMAGE_STEM = (
    r"(?:(?:" + _IMAGE_RECALLED + r"\s*)?같(?:았)?|" + _IMAGE_COPULA + r"|"
    + _IMAGE_EXISTS + r"(?:었)?)"
)
# 맺음 뒤에 붙는 높임·감탄 글자.
_ENDING_FILLER = r"[요ㅠㅜㅋㅎ~]*"


def _closed(auxiliary: str = "") -> str:
    """어미가 정말 문장을 맺었는가: 한글이 붙어 이어지지 않고, 그 어미의 보조 서술어가 따라오지 않는다.

    맺은 뒤에 새 문장이 오는 것은 막지 않는다 — 마침표가 있든 없든 같아야 한다.
    """
    follows = r"(?!\s*(?:" + auxiliary + r"))" if auxiliary else ""
    return _ENDING_FILLER + r"(?![가-힣])" + follows


# 어미마다 뒤에 올 수 있는 보조 서술어가 정해져 있다(닫힌 목록). 그것이 따라오면 맺은 것이 아니다.
#   -어/-아 + 보이다           "같아 보이는", "있어 보이는"
#   -지     + 않다·못하다      "같지 않은", "있지 못한"
#   -다     + 보니·보면·하다   "이미지다 보니", "였다 하는"
# -음·-야·-임·합쇼체에는 보조 서술어가 붙지 않는다.
_AFTER_EO = _closed(r"보[이여였]")
_AFTER_JI = _closed(r"않|못")
_AFTER_DA = _closed(r"보[니면]|하[는던니며고]")
_AFTER_NEUTRAL = _closed()
_IMAGE_FINAL = (
    r"(?:" + _IMAGE_STEM
    + r"(?:[어아]" + _AFTER_EO + r"|지" + _AFTER_JI + r"|다" + _AFTER_DA + r"|음" + _AFTER_NEUTRAL + r")|"
    r"(?:이)?다" + _AFTER_DA + r"|"
    r"(?:(?:이)?야|임|입니다|이에요|예요)" + _AFTER_NEUTRAL + r")"
)
_IMAGE_CONNECTIVE = (
    r"(?:(?:" + _IMAGE_RECALLED + r"\s*)?같(?:은데|지만|았는데|았지만)|"
    + _IMAGE_COPULA + r"(?:는데|지만)|"
    + _IMAGE_EXISTS + r"(?:는데|지만|었는데|었지만))"
)
# 서술어 없이 '이미지'(또는 "이미지였던 거")로 끝난 절: 절의 끝이거나, 말을 잇는 부사가 새 절을 연다.
_BARE_CLAUSE_END = r"\s*" + _ENDING_FILLER + r"\s*(?:$|근데|그래서|그치만|암튼|아무튼)"
_IMAGE_HEAD_NOUN_RE = re.compile(
    r"이미지\s*(?:" + _IMAGE_CONNECTIVE + r"|" + _IMAGE_FINAL
    + r"|(?:" + _IMAGE_RECALLED + r")?" + _BARE_CLAUSE_END + r")"
)

_LISTENING_BEFORE_IMAGE_RE = re.compile(
    r"(?:노래|음악|멜로디|사운드|(?<![가-힣])곡(?:이|은|는|을|도)?(?=\s)|"
    r"들으면|들을\s*때|듣다\s*보면|듣고\s*있으면|들었을\s*때)"
)

_STATIC_VISUAL_PREDICATE_RE = re.compile(
    r"(?:그려|적혀|쓰여|써\s*있|찍혀|찍은|보였|보이|들어가|배치|붙어|"
    r"나뉘|겹쳐|휘어져|흩어져|서\s*있|놓여|바탕|배경(?!\s*음악)|질감)",
    re.IGNORECASE,
)

# A programme/venue name can contain a visual noun (e.g. a name ending in
# "스케치북"). It is not an artwork medium when it is the location of a past
# performance. Mask only that noun for implicit image detection, never the
# original Text/Context query or the rest of a mixed artwork description.
_PERFORMED_VENUE_NAME_RE = re.compile(
    r"(?<![가-힣A-Za-z0-9])(?P<name>"
    r"['\"‘“][^'\"’”\n,;.!?]{2,60}['\"’”]|[가-힣A-Za-z0-9·&_-]{2,40})"
    r"['\"‘’“”]?에서(?:는|도)?"
    r"(?P<between>[^,;.!?。！？\n]{0,70}?)"
    r"(?:부른|부르던|불렀|가창(?:한|했)|연주(?:한|했)|공연(?:한|했)|"
    r"커버(?:한|했|해(?=\s)))",
    re.IGNORECASE,
)


def _mask_performed_venue_names(text: str) -> str:
    masked = list(text)
    for venue in _PERFORMED_VENUE_NAME_RE.finditer(text):
        name = venue.group("name").strip("'\"‘’“”")
        # "사진에서 남자가 노래를 부른 모습" still describes a real image.
        # Likewise, a drawing/layout predicate before the performance verb
        # supplies independent visual evidence and must remain inspectable.
        if (_IMPLICIT_ARTWORK_EVIDENCE["medium"].fullmatch(name)
                or _STATIC_VISUAL_PREDICATE_RE.search(venue.group("between"))):
            continue
        start, end = venue.span("name")
        masked[start:end] = " " * (end - start)
    return "".join(masked)

# 이미지라는 단어 자체는 앨범 아트를 뜻하지 않는다. 뮤직비디오/무대/작품 속
# 장면이나 음악을 들으며 떠올린 심상은 text 단서로 남기고 SigLIP2에는 보내지 않는다.
_NON_COVER_VISUAL_CONTEXT_RE = re.compile(
    r"(?:" + COVER_DANCE_PATTERN + r"|" + COVER_CONTEST_PATTERN + r"|" + COVER_RECORDING_PATTERN
    + r"|뮤직\s*비디오|뮤비|\bmv\b|티저\s*(?:영상|장면)?|"
    r"무대\s*(?:배경|영상|장면|의상)?|공연\s*(?:영상|장면|배경)?|"
    r"콘서트\s*(?:영상|장면|배경)?|방송\s*(?:화면|영상|장면)?|"
    r"애니(?:메이션)?\s*(?:속|에서|장면|영상)|"
    r"(?:드라마|영화)\s*(?:속|에서|장면|영상)|"
    r"가사\s*(?:에|속|에서).{0,30}(?:그림|이미지|장면|꽃|별|배경)|"
    r"(?:머릿속|마음속|생각(?:나|난|났|날)|떠오르|떠올|연상|상상).{0,30}"
    r"(?:그림|이미지|장면|색|빛|배경|분위기|느낌)|"
    r"(?:그림|이미지|장면|색|빛|배경|분위기|느낌).{0,30}"
    r"(?:생각(?:나|난|났|날)|떠오르|떠올|연상|상상)|"
    r"(?:그림|사진|색|빛|꽃|배경).{0,18}(?:같은|처럼).{0,18}"
    r"(?:멜로디|보컬|음색|음질|사운드|노래|음악)|"
    r"(?:멜로디|보컬|음색|음질|사운드|노래|음악).{0,18}"
    r"(?:그림|사진|색|빛|꽃).{0,10}(?:같|처럼)|"
    r"(?:그림(?:을)?\s*그(?:리|릴|렸|려)|사진(?:을)?\s*찍).{0,20}"
    r"(?:듣|노래|음악)|"
    r"(?:제목|곡명|노래\s*이름).{0,25}(?:뜻|의미|단어\s*뜻)|"
    r"(?:프로필|인스타|SNS|기사|인터뷰)\s*(?:사진|이미지))",
    re.IGNORECASE,
)

_NEGATED_COVER_CONTEXT_RE = re.compile(
    r"(?:앨범\s*)?(?:표지|커버|자켓|재킷|아트|사진|이미지)"
    r"(?:는|은|를|가|도)?\s*(?:전혀\s*)?"
    r"(?:기억(?:이)?\s*안\s*나(?:고|는데|지만)?|"
    r"기억나지\s*않(?:고|는데|지만)?|모르겠(?:고|는데|지만)?|"
    r"못\s*봤(?:고|는데|지만)?|안\s*봤(?:고|는데|지만)?|상관없(?:고|는데|지만)?)",
    re.IGNORECASE,
)

# Genre words can describe a TV show or a person instead of the recording.
# Keep the song genre itself, e.g. "힙합 그룹의 곡" or "클래식 음악".
_NON_AUDITORY_GENRE_CONTEXT_RE = re.compile(
    COVER_DANCE_PATTERN + r"|" + COVER_CONTEST_PATTERN + r"|" + COVER_RECORDING_PATTERN
    + r"|(?:힙합|재즈|클래식|록|락)(?=\s*(?:경연|대회|프로그램|방송|예능|"
    r"연주자(?:들)?))",
    re.IGNORECASE,
)

# An unrecorded arrangement discussed during planning is a Context fact, not
# evidence of what the released track sounds like. Remove only an explicitly
# hypothetical performance span; preserve the rest of the query, including any
# later description of the actual recording. Text and Context see the original.
_UNREALIZED_PERFORMANCE_PLAN_RE = re.compile(
    r"(?:(?:초기|당초)\s*(?:기획|계획)(?:\s*(?:당시|단계|때))?|"
    r"(?:기획|계획)\s*(?:당시|초기|단계|때))"
    r"[^.!?。！？\n]{0,90}?(?:듀엣|부르|가창|보컬|랩|녹음|연주|반주|편곡|"
    r"피아노|기타|드럼|소리)"
    r"[^.!?。！？\n]{0,80}?(?:"
    r"(?:안|계획|구상)(?:도|이|은|을)?\s*"
    r"(?:있었|나왔|검토(?:됐|했)|세웠|잡혔|이었|였)|"
    r"예정(?:이었|였))(?:다(?:더라|고)?|지만|고|어|네)?",
    re.IGNORECASE,
)

_STRONG_AUDIO_RE = re.compile(
    r"(?:발라드|댄스곡?|재즈|알앤비|r\s*&\s*b|힙합|랩(?:핑|파트)?|" + _ROCK + r"|"
    r"메탈|트로트|포크|클래식|edm|케이팝|k-?pop|보컬|목소리|음색|가창|무반주|"
    r"멜로디|리듬|비트|템포|bpm|"
    r"사운드(?!\s*트랙)|소리|반주|편곡|화음|코러스|고음|저음|미성|허스키|"
    r"바이브레이션|기교|피아노|건반|기타|드럼|퍼커션|타악기|신스|"
    r"신디사이저|베이스|바이올린|첼로|현악|스트링|오케스트라|브라스|"
    r"트럼펫|색소폰|플루트|플룻|피리|하모니카|아카펠라|휘파람|휘슬|"
    r"비프음|삐\s*소리|벨소리|종소리|박수\s*소리|핑거\s*스냅|"
    r"사이렌|전화음|빗소리|파도\s*소리|새\s*소리|바람\s*소리|"
    r"빠른\s*(?:곡|노래|템포|비트)|느린\s*(?:곡|노래|템포|비트))",
    re.IGNORECASE,
)

_AUDIO_MOOD_RE = re.compile(
    r"(?:신나|잔잔|몽환|애절|감성적|감성|웅장|강렬|경쾌|밝은|어두운|"
    r"차분|따뜻|쓸쓸|슬픈|희망적|편안|부드러운|격정적|청량|그루브|"
    r"로맨틱|긴장감|무거운|가벼운)",
    re.IGNORECASE,
)

_MUSIC_CONTEXT_RE = re.compile(
    r"(?:노래|곡|음악|멜로디|리듬|비트|템포|사운드|보컬|목소리|반주)",
    re.IGNORECASE,
)

_AUDIO_INSTRUMENT_RE = re.compile(
    r"(?:피아노|건반|기타|드럼|퍼커션|타악기|신스|신디사이저|베이스|"
    r"바이올린|첼로|현악|스트링|오케스트라|관현악|브라스|트럼펫|"
    r"트롬본|색소폰|플루트|플룻|피리|하모니카|가야금|해금|대금|장구|"
    r"휘파람|휘슬|비프음|삐\s*소리|벨소리|종소리|박수|손뼉|"
    r"핑거\s*스냅|사이렌|전화음|빗소리|파도\s*소리|새\s*소리|"
    r"바람\s*소리|밴드\s*(?:사운드|편곡|반주|연주))",
    re.IGNORECASE,
)

_AUDIO_BASIC_VOCAL_RE = re.compile(
    r"(?:남(?:자|성)|여(?:자|성)|혼성).{0,18}(?:보컬|목소리|부르|노래|랩|가창)|"
    r"(?:보컬|목소리|부르|노래|랩|가창).{0,18}(?:남(?:자|성)|여(?:자|성)|혼성)|"
    r"(?:듀엣|독창|솔로\s*보컬|다중\s*보컬|합창|콰이어|코러스|나레이션)",
    re.IGNORECASE,
)

# A performance described as an event in a work, or a later rendition of an
# older song, is Context history. Model-inferred vocal roles from that scene
# must not turn it into a CLAP query by themselves.
_NON_AUDITORY_PERFORMANCE_HISTORY_RE = re.compile(
    r"(?:작품|드라마|영화|방송).{0,48}(?:노래하는|부르는|부르던).{0,22}장면|"
    r"(?:부르던|불렀던).{0,42}(?:다시|나중에).{0,26}(?:부른|불렀|커버)",
    re.IGNORECASE,
)

_AUDIO_VOCAL_DETAIL_RE = re.compile(
    r"(?:호소(?:하듯|하는|력)|애원하듯|절규하듯|울부짖|감정(?:을)?\s*쏟|"
    r"속삭이듯|속삭이는|숨소리\s*섞|말하듯|담담하게|힘있게|파워풀|"
    r"허스키|미성|얇은\s*목소리|가느다란\s*목소리|고음|저음|"
    r"바이브레이션|기교|꺾기|거친\s*목소리|청아한\s*목소리)",
    re.IGNORECASE,
)

_AUDIO_RHYTHM_DYNAMICS_RE = re.compile(
    r"(?:빠른|느린)\s*(?:템포|비트|리듬)|템포가\s*(?:빠|느)|bpm|"
    r"박자가\s*(?:빠|느|엇갈|변)|리듬이\s*(?:빠|느|강|복잡)|"
    r"점점\s*(?:커지|강해|빨라|느려)|서서히\s*(?:커지|강해)|"
    r"(?:후렴|후반부?).{0,18}(?:터지|폭발|커지|강해|웅장|고조)",
    re.IGNORECASE,
)

_AUDIO_GENRE_RE = re.compile(
    r"(?:발라드|댄스곡?|재즈|알앤비|r\s*&\s*b|힙합|" + _ROCK + r"|메탈|"
    r"트로트|포크|클래식|edm|케이팝|k-?pop|아카펠라)",
    re.IGNORECASE,
)

_AUDIO_PRODUCTION_RE = re.compile(
    r"(?:반주|편곡|레이어|겹쳐|쌓이|깔리|받쳐|악기\s*구성|"
    r"로파이|lo-?fi|리버브|에코|잔향|왜곡|디스토션|오토튠|"
    r"먹먹한\s*음질|거친\s*음질|드라이한\s*보컬|공간감|"
    r"미니멀(?:한)?\s*(?:반주|편곡|사운드)|단출한\s*(?:반주|편곡)|무반주)",
    re.IGNORECASE,
)

# CLAP에는 악기 이름의 나열뿐 아니라 악기 간 관계와 시간적 전개가 중요하다.
# Gemini가 자주 축약하는 고정보존 단서를 영어 보충 구절로 되살린다.
_AUDIO_RELATIONAL_FEATURES = (
    (
        re.compile(
            r"(?:(?:반주|편곡)(?:가|는|은)?\s*.{0,16})?"
            r"(?:거의|사실상)?\s*(?:피아노|건반|기타|통기타|보컬|악기)"
            r"(?:\s*(?:하나|한\s*대))?\s*(?:뿐|만)|"
            r"(?:미니멀|단출|비어\s*있|간소)한?\s*(?:반주|편곡|사운드)",
            re.IGNORECASE,
        ),
        "a sparse, minimal accompaniment",
        re.compile(r"\b(?:sparse|minimal|stripped[-\s]?back|bare)\b", re.IGNORECASE),
    ),
    (
        re.compile(
            r"(?:드럼|퍼커션|비트)\s*(?:이|가|은|는)?\s*"
            r"(?:전혀\s*|거의\s*)?"
            r"(?:없이|없고|없는|안\s*들리|빠진)",
            re.IGNORECASE,
        ),
        "an arrangement without drums or percussion",
        re.compile(r"\b(?:without|no)\s+(?:drums?|percussion|beat|instrument)", re.IGNORECASE),
    ),
    (
        re.compile(
            r"(?:반주|악기)\s*(?:이|가|은|는)?\s*(?:전혀\s*|거의\s*)?"
            r"(?:없이|없고|없는|안\s*들리|빠진)|무반주(?:\s*보컬)?",
            re.IGNORECASE,
        ),
        "an unaccompanied, nearly a cappella texture",
        re.compile(r"\b(?:unaccompanied|a\s+cappella|without\s+accompaniment)", re.IGNORECASE),
    ),
    (
        re.compile(
            r"(?:피아노|건반|기타|통기타|드럼|베이스|신스|스트링|현악|"
            r"오케스트라)(?:가|를|이)?\s*(?:중심|위주|메인|주도)|"
            r"(?:반주|편곡)(?:가|는|은)?\s*.{0,18}(?:중심|위주)",
            re.IGNORECASE,
        ),
        "a clearly foregrounded, instrument-led arrangement",
        re.compile(
            r"\b(?:instrument[-\s]?led|foregrounded|dominant\s+instrument|"
            r"led\s+by|cent(?:er|re)ed\s+(?:around|on))",
            re.IGNORECASE,
        ),
    ),
    (
        re.compile(
            r"(?:뒤(?:에서|에)|밑에서|아래에서|안쪽에서).{0,24}"
            r"(?:깔|받치|받쳐|들리|들렸|쌓|겹|레이어)|"
            r"(?:깔|받치|받쳐|쌓|겹|레이어).{0,24}"
            r"(?:뒤에서|밑에서|아래에서)",
            re.IGNORECASE,
        ),
        "subtle underlying instrumental layers",
        re.compile(
            r"\b(?:"
            r"underlying|underneath|"
            r"subtle.{0,48}(?:layer|orchestr\w*|strings?)|"
            r"(?:layer|orchestr\w*|strings?).{0,48}subtle"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        re.compile(
            r"(?:조금씩|점점|서서히|차츰|갈수록|후반(?:부)?로\s*갈수록).{0,28}"
            r"(?:깔|쌓|겹|커지|강해|들어오|붙|추가|고조)|"
            r"(?:깔|쌓|겹|커지|강해|들어오|붙|추가|고조).{0,20}"
            r"(?:조금씩|점점|서서히|차츰|갈수록)",
            re.IGNORECASE,
        ),
        "a gradual build as instrumental layers enter",
        re.compile(r"\b(?:gradual(?:ly)?|builds?\b|progressive(?:ly)?|layers?\s+(?:enter|build))", re.IGNORECASE),
    ),
    (
        re.compile(
            r"(?:후렴|후반부?).{0,22}(?:터지|터졌|폭발|"
            r"확\s*커(?:지|졌|져)|강(?:해|했|해졌)|웅장|고조)",
            re.IGNORECASE,
        ),
        "a powerful dynamic lift in the chorus",
        re.compile(r"\b(?:chorus|refrain).{0,24}(?:lift|explode|power|dynamic|build)", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:호소(?:하듯|하는|력)|애원하듯|감정(?:을)?\s*쏟)", re.IGNORECASE),
        "emotionally expressive, pleading vocal delivery",
        re.compile(r"\b(?:pleading|impassioned|emotionally\s+expressive|yearning\s+vocal)", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:속삭이듯|속삭이는|숨소리\s*섞)", re.IGNORECASE),
        "soft, breathy, whisper-like vocal delivery",
        re.compile(r"\b(?:whisper|breathy|hushed)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:담담하게|담백하게|절제(?:된|해서|된\s*듯)|힘을\s*빼고)", re.IGNORECASE),
        "restrained, understated vocal delivery",
        re.compile(r"\b(?:restrained|understated|matter-of-fact|subdued\s+vocal)", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:파워풀|힘있게|절규하듯|울부짖|폭발적인\s*가창)", re.IGNORECASE),
        "powerful, forceful vocal delivery",
        re.compile(r"\b(?:powerful|forceful|belting|belted)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:허스키|거친\s*목소리)", re.IGNORECASE),
        "a husky, raspy vocal timbre",
        re.compile(r"\b(?:husky|raspy|gravelly)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:미성|얇은\s*목소리|가느다란\s*목소리|청아한\s*목소리)", re.IGNORECASE),
        "a light, clear, thin vocal timbre",
        re.compile(
            r"\b(?:(?:light|thin|clear|delicate).{0,30}"
            r"(?:vocal|voice|timbre)|high[-\s]?pitched)",
            re.IGNORECASE,
        ),
    ),
    (
        re.compile(r"(?:가성|팔세토|falsetto)", re.IGNORECASE),
        "prominent falsetto singing",
        re.compile(r"\b(?:falsetto|head\s+voice)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:로파이|lo-?fi|먹먹한\s*음질|빈티지한\s*음질)", re.IGNORECASE),
        "a lo-fi, muffled production texture",
        re.compile(r"\b(?:lo-?fi|muffled|vintage\s+(?:sound|production))\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:리버브|에코|잔향).{0,12}(?:강|크|많|긴|깊)|(?:강한|긴|깊은)\s*(?:리버브|에코|잔향)", re.IGNORECASE),
        "prominent reverb and a lingering echo",
        re.compile(r"\b(?:reverb|reverberant|echo(?:ing)?)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:도입부|인트로|시작).{0,24}(?:휘파람|휘슬)", re.IGNORECASE),
        "whistling in the intro",
        re.compile(r"\b(?:intro|opening).{0,20}whistl|whistl.{0,20}(?:intro|opening)", re.IGNORECASE),
    ),
)

_STRONG_AUDIO_RELATION_PHRASES = {
    "a sparse, minimal accompaniment",
    "an arrangement without drums or percussion",
    "an unaccompanied, nearly a cappella texture",
    "a clearly foregrounded, instrument-led arrangement",
    "subtle underlying instrumental layers",
    "a gradual build as instrumental layers enter",
    "a powerful dynamic lift in the chorus",
    "whistling in the intro",
}

_CLAUSE_BREAK_RE = re.compile(
    r"[,;.!?。！？…\n]+|(?:그런데|하지만|반면|그리고)|"
    r"(?:있었고|보였고|였고|이고|있고|인데|한데)",
    re.IGNORECASE,
)

# Most anchors below are physically visual. Spatial words also describe the
# audible mix ("background vocals", "guitar in the foreground"); validate those
# against a bounded audio phrase instead of rejecting the word by itself.
# Broad mood words such as "dreamy" or "dark" can describe sound too.
_VISUAL_LEAK_IN_AUDIO_RE = re.compile(
    r"\b(?:album\s+cover|cover\s+art|cover\s+image|on\s+the\s+cover|"
    r"illustration|drawing|sketch|line\s+art|paint(?:ed|ing)|watercolou?r|"
    r"photograph(?:s|y|ic)?|photos?|portraits?|pictures?|images?|artwork|"
    r"pictured|handwrit(?:ten|ing)|"
    r"calligraph(?:y|ic)|paper[-\s]+texture|textured\s+paper|"
    r"depict(?:s|ed|ing)?|visual(?:s|ly)?|background|foreground|typography|"
    r"font|lettering|on\s+the\s+(?:left|right)|night[-\s]sky|starry|stars?|"
    r"pink|purple|violet|orange|yellow|green|blue|red|beige|brown|"
    r"black|white|gr[ae]y|black[-\s]and[-\s]white|monochrome|flowers?|"
    r"floral|diagonal\s+(?:layout|split)|geometric\s+(?:shapes?|art))\b",
    re.IGNORECASE,
)

# Restrict spatial exemptions to explicit audible nouns and a short set of
# sound modifiers/predicates. Never extend an exemption across arbitrary text
# or a sentence boundary: a later "background image" must still be rejected.
_SPATIAL_AUDIO_NOUN = (
    r"(?:backing\s+vocals?|vocal\s+harmon(?:y|ies)|instrumental\s+layers?|"
    r"vocals?|voices?|chorus(?:es)?|choirs?|harmon(?:y|ies)|singing|chants?|"
    r"whispers?|whistling|instrumentation|instruments?|accompaniment|"
    r"music|melod(?:y|ies)|sounds?|noises?|ambi(?:ence|ance)|beats?|rhythms?|"
    r"percussion|drums?|piano|guitars?|bass|synth(?:esizer)?s?|strings?)"
)
_SPATIAL_AUDIO_MODIFIER = (
    r"(?:male|female|mixed|soft|subtle|quiet|faint|low|high|pitched|deep|"
    r"prominent|layered|harmonized|choral|wordless|breathy|smooth|raspy|"
    r"instrumental|electronic|acoustic|distorted|ambient|warm|bright|dark)"
)
_SPATIAL_AUDIO_PREDICATE = (
    r"(?:is|are|was|were|remains?|sit(?:s|ting)?|sound(?:s|ing)?|"
    r"play(?:s|ing|ed)?|heard|audible|sing(?:s|ing)?|sung|"
    r"mov(?:e[sd]?|ing)|gradually|slowly|softly|quietly|gently|subtly|faintly|clearly)"
)
_SPATIAL_AUDIO_CONTEXT_RE = re.compile(
    r"\b(?:background|foreground)\b[-\s]+"
    r"(?:" + _SPATIAL_AUDIO_MODIFIER + r"\b[-\s]+){0,3}"
    + _SPATIAL_AUDIO_NOUN + r"\b|"
    r"\b" + _SPATIAL_AUDIO_NOUN + r"\b"
    r"(?:\s+" + _SPATIAL_AUDIO_PREDICATE + r"\b){0,4}"
    r"\s+(?:in|into|from)\s+(?:the\s+)?(?:background|foreground)\b",
    re.IGNORECASE,
)


def _find_visual_leak_in_audio(prompt: str) -> re.Match[str] | None:
    """Ignore only audible uses of spatial words; inspect every other anchor."""
    audio_spans = [match.span() for match in _SPATIAL_AUDIO_CONTEXT_RE.finditer(prompt)]
    for leak in _VISUAL_LEAK_IN_AUDIO_RE.finditer(prompt):
        if leak.group(0).lower() in {"background", "foreground"} and any(
            start <= leak.start() and leak.end() <= end
            for start, end in audio_spans
        ):
            continue
        return leak
    return None


# Likewise, only phrases that unambiguously describe the waveform are rejected
# from an image prompt.  "A portrait of a singer holding a guitar" remains valid.
_AUDIO_LEAK_IN_IMAGE_RE = re.compile(
    r"\b(?:vocals?|ballad|tempo|bpm|melody|rhythm|instrumentation|lyrics?|"
    r"sung|singing|rap\s+verse|rapping|audio|sonic|drum\s+(?:beat|groove)|"
    r"guitar\s+(?:riff|sound)|synth(?:esizer)?\s+(?:sound|line)|"
    r"piano[-\s](?:led|driven)|jazz\s+(?:track|song|sound)|"
    r"hip[-\s]?hop\s+(?:track|song|sound)|r\s*&\s*b\s+(?:track|song|sound)|"
    r"k[-\s]?pop\s+(?:track|song|sound)|dance\s+track|edm\s+(?:track|song))\b",
    re.IGNORECASE,
)

_VISIBLE_TEXT_CONTEXT_RE = re.compile(
    r"\b(?:word|text|title|quote|lettering|typography|printed|written|reads?|"
    r"says?|spells?)\b",
    re.IGNORECASE,
)


def _clean_prompt(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _has_implicit_artwork_description(text: str) -> bool:
    """Detect a detailed static artwork description without a cover noun.

    A strong artwork/form anchor plus another independent visual axis is enough.
    A description that ends on the head noun "이미지" plus two axes is enough.
    Without such an anchor, three axes and a static-layout predicate are required.
    This keeps vague listening imagery out while covering descriptions such as
    hand-drawn flowers, rough paper texture, and handwritten titles.
    """
    evidence_text_parts: list[str] = []
    image_head_axes = 0
    for raw_part in _CLAUSE_BREAK_RE.split(text):
        part = raw_part.strip()
        if not part or _NON_COVER_VISUAL_CONTEXT_RE.search(part):
            continue
        evidence_text_parts.append(part)
        head = _IMAGE_HEAD_NOUN_RE.search(part)
        if head is None:
            continue
        # 묘사는 '이미지' 앞에 온다. 다른 절의 가수 설명("…이미지였어, 노란 옷 입은 여자 가수")이
        # 축을 채우지 않도록 그 절의 앞부분만 본다.
        described = part[: head.start()]
        if _LISTENING_BEFORE_IMAGE_RE.search(described):
            continue
        axes = sum(
            1
            for pattern in _IMPLICIT_ARTWORK_EVIDENCE.values()
            if pattern.search(described)
        )
        image_head_axes = max(image_head_axes, axes)
    evidence_text = " ".join(evidence_text_parts)
    if not evidence_text:
        return False

    matched = {
        name
        for name, pattern in _IMPLICIT_ARTWORK_EVIDENCE.items()
        if pattern.search(evidence_text)
    }
    form_anchor = bool(matched & {"medium", "surface", "typography"})
    if form_anchor and len(matched) >= 2:
        return True
    # '이미지'로 맺은 묘사는 그 자체가 형태 근거다. 그 묘사 안에 다른 축이 둘 있어야 한다.
    if image_head_axes >= 2:
        return True
    return bool(
        len(matched) >= 3
        and "layout" in matched
        and _STATIC_VISUAL_PREDICATE_RE.search(evidence_text)
    )


def has_explicit_visual_clue(query: str) -> bool:
    """Return whether the user described album-art appearance.

    The public name is kept for compatibility.  "Explicit" includes a detailed
    static artwork description whose cover noun is naturally omitted.
    """
    return _has_visual_clue(_mask_lyric_literals(query))


def _has_visual_clue(text: str) -> bool:
    positive_text = _NEGATED_COVER_CONTEXT_RE.sub("", text)
    if _artwork_reference_start(positive_text) is not None:
        return True
    positive_text = _mask_performed_venue_names(positive_text)
    # "다음 앨범" may describe a release teased in an MV, while a different
    # clause mentions a visual scene. Do not join those independent clauses
    # into an album-art prompt. A real album appearance description must
    # include a concrete form or color close to the album noun itself.
    if "앨범" in positive_text:
        for raw_part in _CLAUSE_BREAK_RE.split(positive_text):
            part = raw_part.strip()
            if not part or _NON_COVER_VISUAL_CONTEXT_RE.search(part):
                continue
            for album in re.finditer("앨범", part):
                after = part[album.end() : album.end() + 40]
                if (
                    re.search(r"(?:색(?:깔|감)?|디자인|이미지|아트워크)", after)
                    or any(
                        _IMPLICIT_ARTWORK_EVIDENCE[name].search(after)
                        for name in ("medium", "surface", "typography", "palette")
                    )
                ):
                    return True
                # A Korean artwork description often comes before the album
                # noun: "노란 달이 떠 있는 앨범". Keep the evidence in this same
                # clause so a release mentioned beside an MV does not acquire
                # the MV's visual details.
                before = part[max(0, album.start() - 60) : album.start()]
                if _VISUAL_DETAIL_RE.search(before):
                    return True
    return _has_implicit_artwork_description(positive_text)


def extract_audio_evidence_text(query: str) -> str:
    """Remove unrealized plans and album-cover clauses from audio evidence.

    This prevents visible instruments or people (for example, ``표지에 피아노
    그림``) from becoming CLAP/performance clues.  Mixed clauses are split on
    Korean connective endings, and any non-cover remainder with explicit music
    context is preserved.
    """
    text = _UNREALIZED_PERFORMANCE_PLAN_RE.sub("", _mask_lyric_literals(query))
    if not _has_visual_clue(text):
        return _NON_AUDITORY_GENRE_CONTEXT_RE.sub("", text)

    kept: list[str] = []
    for raw_part in _CLAUSE_BREAK_RE.split(text):
        part = raw_part.strip()
        if not part:
            continue
        cover = _find_album_cover_context(part)
        if cover is not None:
            outside = [part[: cover.start()], part[cover.end() :]]
            for segment in outside:
                segment = segment.strip()
                if not segment:
                    continue
                if (
                    _VISUAL_DETAIL_RE.search(segment)
                    and not _MUSIC_CONTEXT_RE.search(segment)
                ):
                    continue
                if (
                    _STRONG_AUDIO_RE.search(segment)
                    or _MUSIC_CONTEXT_RE.search(segment)
                ):
                    kept.append(segment)
            continue
        if (
            _VISUAL_DETAIL_RE.search(part)
            and not _MUSIC_CONTEXT_RE.search(part)
        ):
            continue
        kept.append(part)
    return _NON_AUDITORY_GENRE_CONTEXT_RE.sub("", " ".join(kept))


def _audio_detail_evidence_text(query: str) -> str:
    """Return only clauses suitable for judging audio-query specificity.

    External scenes can contain words such as ``감성적`` or ``배경`` without
    describing the waveform.  They remain available to text retrieval but must
    not be used to raise CLAP's fusion weight.
    """
    text = extract_audio_evidence_text(query)
    kept: list[str] = []
    for raw_part in _CLAUSE_BREAK_RE.split(text):
        part = raw_part.strip()
        if not part:
            continue
        if _NON_COVER_VISUAL_CONTEXT_RE.search(part):
            # A scene clause may still explicitly describe a heard sound
            # ("MV에서 기타 소리가 강했어"). Preserve only that case.
            audible_anchor = re.search(
                r"(?:소리|반주|편곡|보컬|목소리|음색|멜로디|리듬|비트|"
                r"템포|가창|랩|휘파람|악기)",
                part,
                re.IGNORECASE,
            )
            if audible_anchor is None:
                continue
        kept.append(part)
    return " ".join(kept)


def _relational_audio_details(query: str) -> list[tuple[str, re.Pattern[str]]]:
    """Extract deterministic English clauses for easily-lost audio relations."""
    audio_text = _audio_detail_evidence_text(query)
    return [
        (phrase, coverage)
        for pattern, phrase, coverage in _AUDIO_RELATIONAL_FEATURES
        if pattern.search(audio_text)
    ]


def _supplement_audio_prompt(query: str, prompt: str) -> str:
    """Append explicit Korean audio relations omitted by the generated prompt."""
    clean = _clean_prompt(prompt)
    missing = [
        phrase
        for phrase, coverage in _relational_audio_details(query)
        if coverage.search(clean) is None
    ]
    missing = list(dict.fromkeys(missing))
    if not clean or not missing:
        return clean

    # QueryAnalysis caps each dedicated prompt at 600 characters. Keep the
    # deterministic high-information details intact and trim only an unusually
    # verbose generated preface when the combined prompt would exceed that cap.
    selected: list[str] = []
    for phrase in missing:
        candidate = f" — {', '.join([*selected, phrase])}."
        if len(candidate) > 420:
            continue
        selected.append(phrase)
    if not selected:
        return clean
    missing = selected
    suffix = f" — {', '.join(missing)}."
    max_base_length = max(1, 600 - len(suffix))
    base = clean.rstrip(". ")
    if len(base) > max_base_length:
        base = base[:max_base_length].rstrip(" ,;—")
        if " " in base:
            base = base.rsplit(" ", 1)[0]
    return f"{base}{suffix}"


def has_detailed_audio_description(
    query: str,
    performance_clues: object = None,
) -> bool:
    """Return whether retrieval should give CLAP at least equal text weight.

    This is evidence-based rather than query-specific: an arrangement relation
    is independently strong, while ordinary descriptions require three audible
    axes.  A male ballad or a single instrument alone therefore stays balanced
    toward text, whereas sparse instrumentation plus vocal delivery and a build
    is treated as an audio-dominant memory.
    """
    text = _audio_detail_evidence_text(query)
    if not text or not has_explicit_audio_clue(query, performance_clues):
        return False
    details = {
        phrase for phrase, _coverage in _relational_audio_details(query)
    }
    if details & _STRONG_AUDIO_RELATION_PHRASES:
        return True

    axes: set[str] = set()
    if _AUDIO_INSTRUMENT_RE.search(text):
        axes.add("instrumentation")
    if _AUDIO_BASIC_VOCAL_RE.search(text):
        axes.add("vocal")
    if _AUDIO_VOCAL_DETAIL_RE.search(text):
        axes.add("vocal_delivery")
    if _AUDIO_RHYTHM_DYNAMICS_RE.search(text):
        axes.add("rhythm_dynamics")
    if _AUDIO_PRODUCTION_RE.search(text):
        axes.add("production")
    if _AUDIO_GENRE_RE.search(text):
        axes.add("genre")
    if _AUDIO_MOOD_RE.search(text):
        axes.add("auditory_mood")

    return bool(
        len(axes) >= 3
        and axes & {
            "instrumentation",
            "vocal_delivery",
            "rhythm_dynamics",
            "production",
        }
    )


def has_explicit_audio_clue(
    query: str,
    performance_clues: object = None,
) -> bool:
    """Return whether the user supplied evidence that can be heard."""
    clue = _clean_mapping(performance_clues)
    text = extract_audio_evidence_text(query)
    if (_STRONG_AUDIO_RE.search(text) or _AUDIO_RHYTHM_DYNAMICS_RE.search(text)
            or _AUDIO_BASIC_VOCAL_RE.search(text)):
        return True

    if text.strip() and not _NON_AUDITORY_PERFORMANCE_HISTORY_RE.search(text) and (
        clue.get("vocal_count")
        or clue.get("vocal_roles")
        or clue.get("sound_ensemble")
    ):
        return True

    mood = _AUDIO_MOOD_RE.search(text)
    if mood is None:
        return False
    if not has_explicit_visual_clue(query):
        return True

    # In a mixed cover query, a generic mood word belongs to audio only when
    # the user also anchors it to music ("노래는 잔잔했어").
    start = max(0, mood.start() - 24)
    end = min(len(text), mood.end() + 24)
    return bool(_MUSIC_CONTEXT_RE.search(text[start:end]))


def _normalize_weights(raw: object, *, use_image: bool, use_audio: bool) -> dict:
    value = _clean_mapping(raw)

    def number(key: str, default: float) -> float:
        try:
            return max(0.0, float(value.get(key, default)))
        except (TypeError, ValueError):
            return default

    text = number("text", 0.6)
    image = number("image", 0.2) if use_image else 0.0
    audio = number("audio", 0.2) if use_audio else 0.0

    # A valid dedicated prompt must not be assigned a dead route.  These are
    # small floors; the model still controls the relative balance above them.
    if use_image and image == 0.0:
        image = 0.10
    if use_audio and audio == 0.0:
        audio = 0.10

    total = text + image + audio
    if total <= 0:
        return {"text": 1.0, "image": 0.0, "audio": 0.0}
    return {
        "text": round(text / total, 6),
        "image": round(image / total, 6),
        "audio": round(audio / total, 6),
    }


def _has_text_identity_evidence(raw: Mapping[str, Any]) -> bool:
    """Return whether a precise text clue should outrank audio-description detail."""
    if any(
        str(raw.get(key) or "").strip()
        for key in (
            "song_title",
            "artist_name",
            "lyric_semantic_query",
        )
    ):
        return True
    if raw.get("lyric_keywords") or raw.get("lyric_clues"):
        return True
    title = _clean_mapping(raw.get("title_constraints"))
    return any(
        title.get(key) is not None
        for key in (
            "script",
            "char_count",
            "word_count",
            "contains_number",
            "repeated_char",
            "repeated_word",
        )
    )


def _apply_detailed_audio_weight_floor(
    query: str,
    raw: Mapping[str, Any],
    weights: dict,
    *,
    use_image: bool,
    use_audio: bool,
) -> dict:
    """Protect richly described sound from an arbitrary low model weight.

    Exact title/artist/lyrics evidence remains text-led.  Album-art mixtures
    retain their three-way model balance.  For a purely text+audio search whose
    distinguishing memory is a detailed waveform description, CLAP receives at
    least 0.5, matching the pre-separation behavior that recovered q115.
    """
    if (
        not use_audio
        or use_image
        or _has_text_identity_evidence(raw)
        or not has_detailed_audio_description(
            query,
            raw.get("performance_clues"),
        )
        or float(weights.get("audio", 0.0)) >= 0.5
    ):
        return weights

    return {"text": 0.5, "image": 0.0, "audio": 0.5}


def _visual_leak_in_audio(prompt: str) -> str:
    """소리 프롬프트에 섞인 시각 전용 낱말. 없으면 빈 문자열."""
    leak = _find_visual_leak_in_audio(prompt)
    return leak.group(0) if leak else ""


def _audio_leak_in_image(prompt: str) -> str:
    """표지 프롬프트에 섞인 소리 전용 낱말. 표지에 적힌 글자를 옮긴 것이면 누출이 아니다."""
    leak = _AUDIO_LEAK_IN_IMAGE_RE.search(prompt)
    if leak is None:
        return ""
    window = prompt[max(0, leak.start() - 45) : min(len(prompt), leak.end() + 45)]
    return "" if _VISIBLE_TEXT_CONTEXT_RE.search(window) else leak.group(0)


def apply_modality_query_safeguards(
    query: str,
    raw: dict,
    *,
    degrade: bool = False,
) -> dict:
    """Validate and normalize Gemini's modality-specific embedding prompts.

    The legacy ``english_translation`` key is intentionally ignored.  Reusing it
    would silently restore the original cross-modal contamination bug.

    ``degrade=False``(기본)는 경계를 어긴 응답에 예외를 낸다 — 분석기가 정정을 붙여
    다시 묻는다. ``degrade=True``는 재시도를 다 쓴 뒤의 길이다. 예외 대신 **어긴
    모달리티만** 내려놓고 나머지 분석(가사·제목·가수·태그·다른 모달리티)은 살린다.
    예전에는 여기서 분석 전체를 규칙 폴백으로 바꿨다.

    - 누락(규칙은 단서가 있다고 보는데 모델이 프롬프트를 안 냄): 그 경로를 끈다.
      정정을 받고도 안 낸 것이라 모델 판단을 따른다. 규칙 프롬프트로 채우면 규칙이
      틀렸을 때("초록"의 록 → "a rock style") 그 오염이 그대로 검색에 들어간다.
    - 누출(프롬프트에 다른 모달리티 낱말이 섞임): 단서가 있다는 데는 규칙과 모델이
      같으므로 규칙이 만든 프롬프트로 바꾼다. 규칙이 만들 것이 없으면 끈다.

    내려놓은 모달리티는 ``degraded_modalities``에 남는다.
    """
    enriched = dict(raw)
    performance_clues = enriched.get("performance_clues")
    visual_allowed = has_explicit_visual_clue(query)
    audio_allowed = has_explicit_audio_clue(query, performance_clues)

    image_query = _clean_prompt(enriched.get("image_english_query"))
    audio_query = _clean_prompt(enriched.get("audio_english_query"))
    degraded: list[str] = []

    def violated(modality: str, kind: str, message: str) -> None:
        if not degrade:
            raise ModalityQueryValidationError(message, modality=modality, kind=kind)
        degraded.append(modality)

    if not visual_allowed:
        image_query = ""
    elif not image_query:
        violated("image", "missing", "album-art visual clue requires image_english_query")

    if not audio_allowed:
        audio_query = ""
    elif not audio_query:
        violated("audio", "missing", "explicit auditory clue requires audio_english_query")

    if audio_query:
        audio_query = _supplement_audio_prompt(query, audio_query)

    if audio_query:
        leak = _visual_leak_in_audio(audio_query)
        if leak:
            violated(
                "audio",
                "leak",
                f"audio_english_query contains visual-only wording: {leak!r}",
            )
            audio_query = build_fallback_modality_queries(query, performance_clues)[1]
            if _visual_leak_in_audio(audio_query):
                audio_query = ""

    if image_query:
        leak = _audio_leak_in_image(image_query)
        if leak:
            violated(
                "image",
                "leak",
                f"image_english_query contains audio-only wording: {leak!r}",
            )
            image_query = build_fallback_modality_queries(query, performance_clues)[0]
            if _audio_leak_in_image(image_query):
                image_query = ""

    # 모델 응답이 이 키를 실어 보내도 여기서 덮는다.
    enriched["degraded_modalities"] = [m for m in ("image", "audio") if m in degraded]
    enriched["has_visual_clue"] = bool(visual_allowed and image_query)
    enriched["image_english_query"] = image_query
    enriched["audio_english_query"] = audio_query
    weights = _normalize_weights(
        enriched.get("modality_weights"),
        use_image=bool(image_query),
        use_audio=bool(audio_query),
    )
    enriched["modality_weights"] = _apply_detailed_audio_weight_floor(
        query,
        enriched,
        weights,
        use_image=bool(image_query),
        use_audio=bool(audio_query),
    )
    # Do not let an obsolete model response survive into serialized analysis.
    enriched.pop("english_translation", None)
    return enriched


_VISUAL_FEATURES = (
    (
        r"손(?:으로)?\s*그린|(?:연필|펜|붓|크레용|목탄)(?:으)?로\s*그린|"
        r"핸드\s*드로잉",
        "hand-drawn artwork",
    ),
    (r"(?:검은|블랙)\s*(?:선|라인)|선화|라인\s*드로잉", "black line art"),
    (
        r"(?:종이|캔버스)\s*(?:의\s*)?질감|거친\s*(?:종이|캔버스|표면)|"
        r"(?:낡은|구겨진)\s*종이|종이\s*같은\s*거친\s*질감",
        "a rough paper texture",
    ),
    (r"(?:손글씨|붓글씨|필기체|캘리그라피)", "handwritten typography"),
    (r"(?:폰트|타이포(?:그래피)?|글씨\s*디자인)", "distinctive typography"),
    (r"(?:사선|대각선).{0,12}(?:나뉘|분할)", "a diagonal split layout"),
    (r"흑백", "black-and-white photography"),
    (r"(?:분홍|핑크)", "pink tones"),
    (r"(?:남색|네이비)", "navy-blue tones"),
    (r"(?:파란|파랑|푸른|청색)", "blue tones"),
    (r"(?:보라|자주)", "purple tones"),
    (r"(?:빨간|빨강|붉은|적색)", "red tones"),
    (r"(?:노란|노랑|황색)", "yellow tones"),
    (r"(?:초록|녹색)", "green tones"),
    (r"(?:주황|오렌지)", "orange tones"),
    (r"(?:검정|검은|블랙)", "black tones"),
    (r"(?:하얀|흰색|화이트)", "white tones"),
    (r"(?:회색|그레이)", "gray tones"),
    (r"(?:베이지|갈색|브라운)", "beige and brown tones"),
    (r"(?:컬러풀|알록달록)", "a colorful palette"),
    (r"(?:파스텔|연한\s*색)", "soft pastel colors"),
    (r"(?:네온|형광)", "neon colors"),
    (r"(?:어둡|다크)", "a dark visual mood"),
    (r"(?:밝고|밝은|환한)", "a bright visual mood"),
    (r"(?:귀여|아기자기)", "a cute visual style"),
    (r"밤하늘", "a night sky"),
    (r"(?<!이)별(?:이|을|과|들|무늬|그림)?", "stars"),
    (r"달(?:이|을|과|그림|무늬)?", "the moon"),
    (r"하트", "a heart"),
    (r"(?:꽃잎|꽃\s*그림자)", "flower petals and floral shadows"),
    (r"꽃(?:이|을|과|그림|무늬)?", "flowers"),
    (r"구름", "clouds"),
    (r"(?:바다|해변|바닷가)", "the sea or a beach"),
    (r"(?:도시|건물)", "a city or buildings"),
    (r"(?:여자|여성)(?:의)?\s*(?:얼굴|인물|사진|초상)", "a woman's portrait"),
    (r"(?:남자|남성)(?:의)?\s*(?:얼굴|인물|사진|초상)", "a man's portrait"),
    (r"(?:얼굴|인물|초상)", "a portrait"),
    (r"(?:사진|포토)", "photographic imagery"),
    (r"(?:일러스트|삽화|그림)", "an illustration"),
    (r"(?:도트|픽셀)", "pixel art"),
    (r"콜라주", "a collage"),
    (r"실루엣", "a silhouette"),
    (r"(?:추상|기하학)", "abstract geometric art"),
    (r"(?:영어|알파벳)\s*(?:대문자)?\s*한\s*글자", "one large Latin letter"),
    (r"(?:큰|커다란)\s*(?:글자|문자|숫자)", "large typography"),
    (r"(?:왼쪽|좌측)", "positioned on the left"),
    (r"(?:오른쪽|우측)", "positioned on the right"),
    (r"(?:중앙|가운데)", "centered composition"),
    (r"(?:미니멀|단순한|심플)", "a minimal composition"),
    (r"(?:빈티지|레트로)", "a vintage visual style"),
)

_AUDIO_FEATURES = (
    (r"(?:남자|남성)\s*(?:보컬|가수|노래|발라드|랩)", "male vocals"),
    (r"(?:여자|여성)\s*(?:보컬|가수|노래|발라드|랩)", "female vocals"),
    (r"발라드", "a ballad style"),
    (r"재즈", "a jazz style"),
    (r"(?:알앤비|r\s*&\s*b)", "an R&B style"),
    (r"힙합", "a hip-hop style"),
    (_ROCK + r"(?:\s*음악|\s*느낌|\s*곡|\s*노래)?", "a rock style"),
    (r"트로트", "a trot style"),
    (r"edm|일렉트로닉", "an electronic dance style"),
    (r"(?:빠른\s*템포|템포가\s*빠|빠른\s*비트)", "a fast tempo"),
    (r"(?:느린\s*템포|템포가\s*느|느린\s*비트)", "a slow tempo"),
    (r"신나", "an upbeat energetic mood"),
    (r"잔잔", "a calm gentle mood"),
    (r"몽환", "a dreamy atmospheric sound"),
    (r"애절", "a yearning emotional mood"),
    (r"웅장", "a grand powerful sound"),
    (r"(?:경쾌|청량)", "a bright refreshing mood"),
    (r"(?:차분|편안|부드러운)", "a relaxed soft sound"),
    (r"(?:쓸쓸|슬픈)", "a melancholic mood"),
    (r"허스키", "a husky vocal timbre"),
    (r"(?:미성|얇은\s*목소리|가느다란\s*목소리)", "a light thin vocal timbre"),
    (r"(?:고음|높은\s*목소리)", "high-register vocals"),
    (r"(?:기교|바이브레이션)", "prominent vocal runs and vibrato"),
)

_VOCAL_COUNT_FEATURES = {
    "solo": "one lead vocalist",
    "duet": "two duet vocalists",
    "multiple": "multiple vocalists",
    "choir": "choir vocals",
}

_VOCAL_ROLE_FEATURES = {
    "남성랩": "male rap vocals",
    "여성랩": "female rap vocals",
    "남성노래": "male singing vocals",
    "남성보컬": "male singing vocals",
    "남성가창": "male singing vocals",
    "여성노래": "female singing vocals",
    "여성보컬": "female singing vocals",
    "여성가창": "female singing vocals",
    "피처링보컬": "a featured guest vocal",
    "나레이션": "spoken narration",
    "코러스": "backing chorus vocals",
    "합창": "choir vocals",
}

_ENSEMBLE_FEATURES = {
    "밴드사운드": "a full band arrangement",
    "어쿠스틱": "acoustic instrumentation",
    "오케스트라": "an orchestral arrangement",
    "전자음악": "electronic synthesizer production",
    "라이브": "a live-performance sound",
    "아카펠라": "a cappella vocals",
    "브라스": "brass instruments",
    "스트링": "string instruments",
    "피아노": "piano",
    "기타": "guitar",
    "퍼커션": "drums and percussion",
    "휘파람": "whistling",
    "비프음": "electronic beeps",
    "벨소리": "bell or chime sounds",
    "박수": "hand claps",
    "핑거스냅": "finger snaps",
    "사이렌": "a siren sound effect",
    "전화음": "telephone sound effects",
    "자연음": "nature ambience",
    "플루트": "flute",
    "하모니카": "harmonica",
    "국악": "traditional Korean instruments",
    "인도풍": "Indian-influenced instrumentation",
    "중동풍": "Middle Eastern-influenced instrumentation",
}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def build_fallback_modality_queries(
    query: str,
    performance_clues: object = None,
) -> tuple[str, str]:
    """Build conservative, English modality prompts without an LLM."""
    text = _mask_lyric_literals(query)
    audio_text = extract_audio_evidence_text(query)
    visual_features: list[str] = []
    if has_explicit_visual_clue(query):
        for pattern, phrase in _VISUAL_FEATURES:
            if re.search(pattern, text, re.IGNORECASE):
                visual_features.append(phrase)
    visual_features = _dedupe(visual_features)
    image_query = (
        "Album cover with " + ", ".join(visual_features) + "."
        if visual_features
        else ""
    )

    audio_features: list[str] = []
    clue = _clean_mapping(performance_clues)
    vocal_count = str(clue.get("vocal_count") or "")
    if vocal_count in _VOCAL_COUNT_FEATURES:
        audio_features.append(_VOCAL_COUNT_FEATURES[vocal_count])

    for role in clue.get("vocal_roles") or []:
        phrase = _VOCAL_ROLE_FEATURES.get(str(role))
        if phrase:
            audio_features.append(phrase)
    for ensemble in clue.get("sound_ensemble") or []:
        phrase = _ENSEMBLE_FEATURES.get(str(ensemble))
        if phrase:
            audio_features.append(phrase)
    for pattern, phrase in _AUDIO_FEATURES:
        if re.search(pattern, audio_text, re.IGNORECASE):
            audio_features.append(phrase)

    audio_features = _dedupe(audio_features)
    audio_query = (
        "An audio track with " + ", ".join(audio_features) + "."
        if audio_features and has_explicit_audio_clue(query, clue)
        else ""
    )
    if audio_query:
        audio_query = _supplement_audio_prompt(query, audio_query)
    return image_query, audio_query


def fallback_modality_payload(
    query: str,
    performance_clues: object = None,
) -> dict:
    """Return prompts, gating, and safe default weights for fallback mode."""
    image_query, audio_query = build_fallback_modality_queries(
        query,
        performance_clues,
    )
    if image_query and audio_query:
        weights = {"text": 0.4, "image": 0.3, "audio": 0.3}
    elif image_query:
        weights = {"text": 0.5, "image": 0.5, "audio": 0.0}
    elif audio_query:
        if has_detailed_audio_description(query, performance_clues):
            weights = {"text": 0.5, "image": 0.0, "audio": 0.5}
        else:
            weights = {"text": 0.6, "image": 0.0, "audio": 0.4}
    else:
        weights = {"text": 1.0, "image": 0.0, "audio": 0.0}
    return {
        "image_english_query": image_query,
        "audio_english_query": audio_query,
        "has_visual_clue": bool(image_query),
        "modality_weights": weights,
    }
