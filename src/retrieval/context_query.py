"""외부 배경 사실 단서를 원문에 묶어 검증한다.

Gemini가 검색 문장을 생성하더라도 사용자가 말하지 않은 곡·작품명을 보태면
잘못된 곡을 확신하게 된다. 고정밀 규칙으로 배경 사건을 먼저 확인하고,
모델 출력 중 원문으로 뒷받침되는 표현만 채택한다.
"""
from __future__ import annotations

import re

from src.retrieval.modality_queries import (
    COVER_CONTEST_PATTERN,
    COVER_DANCE_PATTERN,
    COVER_RECORDING_PATTERN,
    explicit_artwork_reference_start,
    has_explicit_artwork_reference,
    normalize_cover_performance_spacing,
)

_LYRIC_START = re.compile(
    r"^(?:(?:그|이|저)\s*노래(?:의)?\s*)?"
    r"(?:가사(?:에|에는|에서|에선)?|노랫말|가삿말|후렴(?:에|에서|의)?)"
)
_LYRIC_LITERAL_SCOPE = re.compile(
    r"^(?:(?:그|이|저)\s*노래(?:의)?\s*)?"
    r"(?:가사|노랫말|가삿말)(?:에|에는|에서|에선)\s*",
)
_TITLE_AS_EVENT = re.compile(
    r"^(?:(?:그|이|저)\s*노래\s*)?(?:제목|곡명)\s*"
    r"(?:이|은|에|가)?\s*['\"‘’]?가요제(?:라는|인|이)"
)
_MV_AUDIO_ANALOGY = re.compile(
    r"(?:뮤직\s*비디오|뮤비|(?<![A-Za-z])MV(?![A-Za-z]))(?:처럼|같이|같은).{0,40}"
    r"(?:들리|소리|음악|사운드|리듬)", re.I,
)
_OST_AUDIO_ANALOGY = re.compile(
    r"(?:OST|오에스티|BGM)\s*(?:처럼|같은|같이|풍|느낌).{0,45}"
    r"(?:들리|소리|음악|사운드|분위기|노래)", re.I,
)
_EVENT_AUDIO_ANALOGY = re.compile(
    r"(?:차트|멜론|공연|콘서트|라이브|무대).{0,35}"
    r"(?:처럼|같은|같이|풍|느낌).{0,35}"
    r"(?:들리|소리|음악|사운드|분위기|EDM|노래)", re.I,
)
_LYRIC_QUOTING_EVENT = re.compile(
    r"(?:후속곡|더블\s*타이틀|차트\s*\d+위).{0,40}"
    r"(?:가사|노랫말).{0,15}(?:나오|담긴|적힌)", re.I,
)
_NAMED_CHORUS = re.compile(
    r"(?:코러스|백\s*보컬).{0,25}?\s"
    r"(?P<person>[가-힣]{2,5})(?:의)?\s*목소리", re.I,
)
_GENERIC_VOICES = frozenset({"남자", "여자", "사람", "가수", "보컬"})
_REPORTED_VOICE_DETAIL = re.compile(
    r"(?=.*(?:여담|일화))(?=.*(?:발음|목소리|소리))"
    r"(?=.*(?:\d+\s*분\s*\d+\s*초|[가-힣]{2,5}만|"
    r"['\"‘’][^'\"‘’]{2,30}['\"‘’]))", re.I,
)
# Audible chorus texture alone is an Audio clue. A past recording action by
# another/named participant is an external event, even when the model omits it.
_REPORTED_RECORDING_PARTICIPATION = re.compile(
    r"(?=.*(?:다른\s*(?:가수|보컬)|친한\s*가수|즉흥|여담|일화))"
    r"(?:코러스|백\s*보컬).{0,85}녹음(?:한|했)|"
    r"녹음(?:에|에서).{0,50}(?:가수|보컬|[가-힣]{2,5}(?:이|가))"
    r".{0,30}참여(?:한|했)", re.I,
)
_LITERAL_EVENT_WORDING = re.compile(
    r"(?:제목|곡명|가사|노랫말).{0,100}(?:라는|라고|이란|이라는)"
    r".{0,12}(?:말|단어|글자|문장|구절|제목).{0,25}"
    r"(?:나오|있|들어|적혀|쓰여)|"
    r"(?:제목|곡명).{0,100}(?:다는|다고)\s*(?:말|단어|글자|문장|구절)"
    r".{0,25}(?:나오|있|들어|적혀|쓰여)|"
    r"(?:라는|이란|이라는)\s*(?:제목|곡명)(?:의|인|이)", re.I,
)
_EXTERNAL_LYRIC_REVISION = re.compile(
    r"(?:작사|작곡|제작|가이드|녹음).{0,100}"
    r"(?:바꿨|바꾼|바꾸었|바뀌|바뀐|변경했|고쳤|퇴짜|보냈|만들었)", re.I,
)
# An adaptation for an advertisement/cheering song is an external event,
# even when the user describes it as "changed the lyrics and sang it" instead
# of using the technical word 개사. Require a completed action in the same
# clause; a request to rewrite lyrics for a future advertisement is not trivia.
_COMPLETED_LYRIC_ADAPTATION = (
    r"(?:가사|노랫말|가삿말)(?:의)?(?:\s*(?:일부|구절|내용))?(?:를|을)?"
    r"[^,;.!?。！？\n]{0,25}(?:"
    r"(?:바꿔|바꾸어|고쳐)(?:서)?\s*(?:부른|불렀|불렸|부르던|"
    r"사용(?:한|했|된|됐)|녹음(?:한|했))|"
    r"바꾼|바꿨|바꾸었|고친|고쳤|변경(?:한|했|된|됐)|수정(?:한|했|된|됐))"
)
_REPORTED_LYRIC_ADAPTATION = re.compile(
    r"(?:광고|응원가|로고송|CM송|CF)[^,;.!?。！？\n]{0,85}"
    + _COMPLETED_LYRIC_ADAPTATION + "|" + _COMPLETED_LYRIC_ADAPTATION
    + r"[^,;.!?。！？\n]{0,85}(?:광고|응원가|로고송|CM송|CF)", re.I,
)
_EXPLICIT_MEDIA_USAGE = re.compile(
    r"(?:드라마|영화|애니(?:메이션)?|게임|예능)\s+"
    r"(?:[가-힣A-Za-z0-9·_-]+\s+){1,3}"
    r"(?:OST|오에스티|삽입곡|배경\s*음악)(?:였|이었|로|이고|으로|야)",
    re.I,
)
_REPORTED_COVER_CONTEST = re.compile(
    COVER_CONTEST_PATTERN
    + r".{0,85}(?:열(?:었|린|렸)|개최(?:한|했|된|됐)|우승(?:한|했)|"
    r"참(?:가|여)(?:한|했)|선정(?:한|된|됐)|부른|불렀|사용(?:된|됐|한|했))", re.I,
)
_REPORTED_COVER_RECORDING = re.compile(
    COVER_RECORDING_PATTERN
    + r".{0,85}(?:공개(?:한|했|된|됐)|촬영(?:한|했|된)|찍(?:은|었)|"
    r"제작(?:한|했)|올(?:린|렸)|부른|불렀)", re.I,
)
_REPORTED_COVER_RENDITION = re.compile(
    r"커버\s*(?:(?:를|을)\s*)?(?:"
    r"해(?=\s|$|[,.!?])|해(?:서|주|온|오던|봤|본)|했|한|했던|하던|"
    r"불렀|불러|부른|된|되었|공개(?:된|됐|한|했)|곡|송|버전|노래|음원)|"
    r"커버[^,;.!?。！？\n]{0,70}(?:"
    r"화제(?:가)?\s*(?:돼|됐|되었|된)|역주행|재해석|"
    r"(?:차트|순위)[^,;.!?。！？\n]{0,35}(?:올라|상승|재진입|복귀))", re.I,
)
_COVER_EVENT_SONG_END = re.compile(
    r"(?:곡|노래)(?:인데|이야|였어|의|야|이고|이었어|이었고|였고)?\s*[,;]?\s*$",
)
_PLANNED_COVER_ACTIVITY = re.compile(
    rf"(?:{COVER_CONTEST_PATTERN}|{COVER_RECORDING_PATTERN}|{COVER_DANCE_PATTERN}|커버)"
    r"[^,;.!?。！？\n]{0,85}(?:"
    r"(?:참가|참여|공개|제작|촬영|패러디)(?:할|하려|하기\s*좋)|"
    r"열(?:려고|려|\s*예정)|만들(?:려고|려|\s*계획)|"
    r"(?:부를|낼)\s*(?:곡|노래)|할\s*(?:곡|노래)|하려고|"
    r"해\s*보(?:려고|고\s*싶)|불러\s*볼)", re.I,
)
_RELATIONS = (
    ("삽입곡·배경음악", re.compile(
        r"(?<![A-Za-z])OST(?![A-Za-z])|오에스티|삽입곡|배경\s*음악|"
        r"(?<![A-Za-z])BGM(?![A-Za-z])|사운드\s*트랙|"
        r"주제가|테마곡|엔딩곡|오프닝곡|등장곡|로고송|선거송", re.I,
    )),
    ("다른 작품에 사용", re.compile(
        r"(?:애니(?:메이션)?|드라마|영화|게임|예능|방송|코미디빅리그)"
        r".{0,90}(?:장면|나왔|나온|나오던|흘러|틀어|사용|수록|모티브|삽입)|"
        r"(?:\d+\s*박\s*\d+\s*일|[가-힣A-Za-z0-9]{2,30}\s*편(?:의|에서)?)"
        r".{0,70}장면.{0,20}(?:삽입|사용|흘러|나왔)|"
        r"(?:게임|펌프\s*잇\s*업|FIESTA).{0,55}수록|"
        r"(?:선생님|캐릭터).{0,70}(?:장면|흘러|나오던)|"
        r"(?:판권|라이선스).{0,65}(?:만료|삭제(?:된|됐))|"
        r"(?:광고|응원가).{0,70}개사(?:한|했|된|됐)|"
        r"개사(?:한|했|된|됐).{0,40}(?:광고|응원가)", re.I,
    )),
    ("다른 작품에 사용", _REPORTED_LYRIC_ADAPTATION),
    ("뮤직비디오 속 사건", re.compile(
        r"(?:뮤직\s*비디오|뮤비|(?<![A-Za-z])MV(?![A-Za-z]))", re.I,
    )),
    ("보컬 참여 일화", _NAMED_CHORUS),
    ("보컬 참여 일화", _REPORTED_RECORDING_PARTICIPATION),
    ("음원·보컬 여담", _REPORTED_VOICE_DETAIL),
    ("제작·발매 비화", re.compile(
        r"제작진|제작\s*비화|작곡(?:했|한|할|가|을|하)|작사(?:했|한|할|가|을|하)|"
        r"작(?:사|곡)\s*(?:비화|일화|능력)|"
        r"(?:프로듀서|작곡가|작사가).{0,65}데뷔곡|"
        r"데모곡?|가이드(?:도|를|가|로)?\s*(?:녹음|버전|아르바이트)|퇴짜|"
        r"가이드(?:가|는|의\s*제목이).{0,55}(?:였|이었|불렸)|"
        r"녹음.{0,25}(?:하라|하려|했다|했다던|과정|주장)|"
        r"원키.{0,20}(?:녹음|주장)|미공개|공개되지\s*않|"
        r"원래.{0,25}(?:예정|부르|발매|공개)|"
        r"(?:초기|처음|원래)\s*기획|"
        r"후속곡.{0,65}(?:후보|선정|밀|바뀌|된|됐|고르)|"
        r"(?:더블|공동)\s*타이틀.{0,30}(?:변경|바뀌|바꿔|바꾼|바꾸|추가)|"
        r"(?:에게|한테).{0,25}예정(?:이었|였).{0,80}"
        r"(?:돌아갔|넘어갔|바뀌|변경됐)", re.I,
    )),
    ("방송·공연 일화", _REPORTED_COVER_CONTEST),
    ("방송·공연 일화", re.compile(
        r"가요제|(?:축제|페스티벌|콘서트|시상식|불후의\s*명곡|V앱|브이라이브|"
        r"SUPER\s*SHOW|위문열차)"
        r".{0,110}(?:무대|앙코르|앵콜|우승|공개|불렀|커버|사고|실수|"
        r"처음|안무|공연|라이브|파트|버전\s*음원|버전이\s*따로)|"
        + COVER_DANCE_PATTERN
        + r"(?:를|을)?\s*(?:했|한|하던|했던|해본|해봤|영상|무대)|"
        r"(?:방송|라디오|콘서트|명명식|경연|팬미팅|쇼케이스|음악회|무대)"
        r".{0,110}(?:부른|부르던|불렀|연주(?:한|했)|가창(?:한|했)|"
        r"말했던|말했)|방송\s*사고.{0,35}(?:봤|보았|있었|일어났)", re.I,
    )),
    ("커버·리메이크·답가", _REPORTED_COVER_RECORDING),
    ("커버·리메이크·답가", _REPORTED_COVER_RENDITION),
    ("커버·리메이크·답가", re.compile(
        r"(?:답가|리메이크|번안|원곡|커버(?:해|했|한|무대|곡|버전)|"
        r"여성\s*버전|남성\s*버전)", re.I,
    )),
    ("유행·밈", re.compile(
        r"(?:밈|패러디|챌린지|역주행|바이럴|드립|유행했|화제가\s*됐|"
        r"거리\s*두기.{0,40}(?:농담|회자|예언)|"
        r"제목이?.{0,85}(?:잘못\s*부(?:르|른)|부르기\s*어렵)|"
        r"발음(?:하기)?\s*어려.{0,85}잘못\s*부(?:르|른))", re.I,
    )),
    ("기록·영향", re.compile(
        r"(?:차트|음악\s*방송|지상파|빌보드|인기가요|뮤직뱅크|"
        r"음악중심|멜론|가온|지니|벅스|음원\s*사이트)"
        r".{0,70}(?:\d+위|우승|기록|올킬)|"
        r"모티브가\s*됐|영향을\s*(?:줬|받았)|"
        r"(?:후보|당선).{0,60}(?:선거송|로고송)", re.I,
    )),
    ("제작·활동에 얽힌 사건", re.compile(
        r"(?:월드\s*투어|군\s*입대|소속사|밴드\s*해체)"
        r".{0,95}(?:헤어|복귀|완성|재회|다시\s*만나|다시\s*음악|"
        r"작곡|작사|감정|인기)|프로젝트.{0,55}참여(?:한|했)", re.I,
    )),
)
_WORK_BEFORE = re.compile(
    r"([가-힣A-Za-z0-9]+(?:\s+[가-힣A-Za-z0-9]+){0,2})\s+"
    r"(?:애니(?:메이션)?|드라마|영화|게임|예능)"
)
_WORK_AFTER = re.compile(
    r"(?:애니(?:메이션)?|드라마|영화|게임|예능)\s+"
    r"([가-힣A-Za-z0-9]+(?:\s+[가-힣A-Za-z0-9]+){0,2})"
)
_EVENT_BEFORE = re.compile(r"([가-힣A-Za-z0-9·_-]{2,30})\s+가요제")
_PERSON = re.compile(r"([가-힣]{2,5})\s*(?:선생님|감독|배우|선수|작곡가)")
_GENERIC = {
    "옛날", "예전", "무슨", "어떤", "청춘", "사극", "유명한", "한국",
    "옛날에", "예전에", "그때", "남자", "여자", "중", "중에", "곡", "노래",
}
_NONENTITY_TARGETS = {"봄", "여름", "가을", "겨울", "드라마", "영화", "애니메이션", "예능", "프로그램"}
_WORK_BOUNDARY = re.compile(
    r"(?:OST|BGM|오에스티|삽입곡|배경음악|주제가|사운드트랙|"
    r"사용|수록|쓰였|나왔|나온|나오던|나오는|장면|곡|노래|중에)", re.I,
)
_NOT_WORK_WORD = re.compile(
    r"^(?:\d{2,4}년(?:대)?|초반|중반|후반|가진|없는|없이|부른|"
    r"출연(?:한|했던)|출현(?:한|했던)|나오는|나온|배경으로|한|"
    r"주인공(?:이|은|을)?|노래(?:가|를|인데)?|"
    r"기억(?:이|에)?|옛날에|예전에)$"
)
# 검색 문장에 추가할 수 있는 일반 명사만 허용한다. '삽입곡', '녹음' 같은
# 관계 단어를 원문 없이 허용하면 모델이 뮤직비디오를 OST로 바꿔 버릴 수 있다.
_SEARCH_WORDS = {"노래", "곡"}
_REQUEST_END = re.compile(
    r"\s*(?:뭐(?:였|야|지)|무슨\s*(?:곡|노래)(?:이었|였|이지|이야|이니|인지)|"
    r"제목이?\s*기억(?:이)?\s*안\s*나|찾고\s*싶어|찾아줘|알려줘)"
    r"[^,.!?]*[?!.]?\s*$"
)
_UNCERTAIN = re.compile(
    r"(?:같아|같은데|듯|아마|헷갈|쯤|줄\s*알았|(?:인지|맞는지)\s*모르)",
)
_PRELUDE = re.compile(
    r"(?:스토리|내용|이야기|감옥|죄|장면|안무|촬영|NG|경연|탈락|"
    r"드라마|애니메이션|영화|게임|방송|음원)", re.I,
)
_FOLLOWUP_ONLY = re.compile(
    r"^(?:그|이|저)\s*(?:원곡|답가|커버곡|"
    r"(?:뮤직\s*비디오|뮤비|MV)(?:의)?\s*(?:곡|노래))"
    r"(?:이|가|은|을|의)?\s*"
    r"(?:뭐|어떤|찾|알려)",
    re.I,
)
_RELATED_CONTINUATION = re.compile(
    r"^(?:나중에|그다음|이후|그전(?:의|에)?|그리고|그때).{0,110}"
    r"(?:부른\s*버전|들려|커버|무대|힘들|어렵|재회|다시\s*만나)",
)
_MODEL_EXTERNAL_CUE = re.compile(
    r"(?:애니(?:메이션)?|드라마|영화|게임|예능|방송|웹툰|광고|"
    r"뮤지컬|가요제|축제|공연|뮤직\s*비디오|뮤비|제작|녹음|시상식)"
    r".{0,100}(?P<verb>나온|나오던|흘러|쓰였|쓰인|사용|수록|삽입|"
    r"만든|제작|부른|녹음|등장|공개|공연|참여|수상)", re.I,
)

# User-supplied work aliases. Append rather than replace: Dense sees the
# original recollection and BM25 can match the catalogue's official title.
_SHINCHAN_CUE = re.compile(
    r"(?:크레용\s*신짱|짱구\s*(?:애니(?:메이션)?|만화|OST|삽입곡|장면)|"
    r"짱구(?:는\s*못말려|에서))", re.I,
)


def expand_context_search_query(query: str) -> str:
    """Append a vetted work title only when its alias appears in the clue."""
    query = query.strip()
    if (_SHINCHAN_CUE.search(query)
            and "짱구는 못말려" not in re.sub(r"\s+", " ", query)):
        return f"{query} 짱구는 못말려"
    return query


_NAMED_MEDIA_RELATION = re.compile(
    r"(?:다른\s*작품에\s*사용|OST|오에스티|삽입곡?|배경\s*음악|BGM|주제가|테마곡)",
    re.I,
)
_MEDIA_USE_IN_USER_WORDS = re.compile(
    r"(?<![A-Za-z])OST(?![A-Za-z])|삽입|흘러|나오(?:던|는|았다)|나왔|"
    r"사용|쓰였|수록|주제가|배경\s*음악|선거송|로고송", re.I,
)


def has_grounded_media_work_clue(analysis: object) -> bool:
    """Allow a separate media-work weight trial only for one explicit work cue.

    The target must occur in the user's original words. Mixed external events
    are excluded so a generic OST mention cannot outweigh a production clue.
    """
    clues = [clue for clue in getattr(analysis, "context_clues", ())
             if clue.confidence > 0]
    if len(clues) != 1:
        return False
    clue = clues[0]
    target = re.sub(r"\s+", "", clue.target).casefold()
    raw = getattr(analysis, "original_query", "")
    original = re.sub(r"\s+", "", raw).casefold()
    return bool(target and len(target) >= 2 and target in original
                and has_specific_media_target(clue, original_query=raw)
                and _NAMED_MEDIA_RELATION.search(clue.relation)
                and _MEDIA_USE_IN_USER_WORDS.search(raw))


def context_path_weight(
    weight: float, confidence: float, named_media_multiplier: float, analysis: object,
) -> float:
    """Give a named work one Context path vote, never a second vote."""
    return (weight * confidence *
            (named_media_multiplier if has_grounded_media_work_clue(analysis) else 1.0))


def is_media_usage_relation(relation: str) -> bool:
    return bool(_NAMED_MEDIA_RELATION.search(relation) or re.search(
        r"등장곡|로고송|선거송|사운드\s*트랙|게임\s*음악|"
        r"(?<![가-힣])(?:쓰인|쓰였(?:던)?|쓰이는|쓰여|사용(?:된|한|했|됐)?|"
        r"활용(?:된|한|됐)|나온|나오던|흘러)(?![가-힣])", relation, re.I,
    ))


def _context_media_target(clue: object) -> str:
    target = str(getattr(clue, "target", "") or "").strip()
    if target:
        return target
    # A rule fallback may keep the supplied work in search_query without a
    # separate target field. Only the simple "work + relation" syntax qualifies.
    match = re.match(
        r"^(?:(?:드라마|영화|애니(?:메이션)?|게임)\s+)?"
        r"(.{2,45}?)\s+(?:OST|오에스티|삽입곡|배경\s*음악|BGM|주제가|테마곡)"
        r"(?:\s|$)", str(getattr(clue, "search_query", "")).strip(), re.I,
    )
    candidate = match.group(1).strip() if match else ""
    # A long plot/era recollection before "OST" is not a recovered work name.
    # Explicit targets remain subject to original-query grounding below.
    return candidate if len(candidate.split()) <= 3 else ""


def has_specific_media_target(clue: object, *, original_query: str | None = None) -> bool:
    """A supplied name, rather than a season, genre or a plot description.

    This validates a literal user anchor; it does not identify an unnamed work.
    """
    target = _context_media_target(clue)
    source = original_query if original_query is not None else str(
        getattr(clue, "search_query", "") or ""
    )
    if not target or not _grounded_target(target, source):
        return False
    if target in _NONENTITY_TARGETS or target.casefold() in {
        "애니", "게임", "방송", "광고", "뮤지컬", "ost", "bgm", "오에스티",
        "삽입곡", "배경음악", "주제가", "테마곡", "노래", "음악", "곡",
    }:
        return False
    if re.fullmatch(
        r"(?:봄|여름|가을|겨울|청춘|사극|추억|옛날|유명한)\s*"
        r"(?:드라마|영화|애니(?:메이션)?|게임|예능)", target,
    ):
        return False
    return not bool(re.search(
        r"(?:하는|나가는|가진|부른|나오는|출연한|출현한)(?:\s|$)", target,
    ))


def context_media_description_requires_support(clue: object) -> bool:
    """An unnamed, qualified media recollection needs fact-level support.

    Broad two-word searches can enumerate OST songs. A detailed description
    cannot promote a different song merely because its profile also says OST.
    """
    if (not is_media_usage_relation(str(getattr(clue, "relation", "")))
            or has_specific_media_target(clue)):
        return False
    terms = [re.sub(r"(?:에서|에는|의|에)$", "", term.casefold()) for term in re.findall(
        r"[가-힣A-Za-z0-9]+", str(getattr(clue, "search_query", "")),
    )]
    enumeration = {
        "드라마", "영화", "애니", "애니메이션", "게임", "예능", "프로그램",
        "ost", "bgm", "오에스티", "삽입곡", "배경음악", "주제가", "테마곡",
        "노래", "곡", "음악", "추천", "찾아줘", "사운드트랙",
        "쓰인", "쓰였던", "쓰이는", "사용된", "사용한", "나온", "나오던",
        "흘러", "쓰였", "활용된",
    }
    return len(terms) >= 3 and any(term not in enumeration for term in terms)


def context_media_target_terms(
    clue: object, *, original_query: str | None = None,
) -> tuple[str, ...]:
    """Literal supplied work names and a vetted equivalent, for Dense filtering.

    Removing a leading/trailing media type does not infer an unnamed work.
    Do not use artist/title metadata or an evaluation's expected song here.
    """
    if (not is_media_usage_relation(str(getattr(clue, "relation", "")))
            or not has_specific_media_target(clue, original_query=original_query)):
        return ()
    target = re.sub(r"\s+", " ", _context_media_target(clue))
    if _SHINCHAN_CUE.search(target) or target == "짱구":
        return ("짱구는 못말려", "크레용 신짱")
    name = re.sub(r"^(?:드라마|영화|애니(?:메이션)?|게임)\s+|"
                  r"\s+(?:드라마|영화|애니(?:메이션)?|게임)$", "", target).strip()
    return (name,) if len(name) >= 2 else ()


def context_search_queries(clue: object, *, original_query: str | None = None) -> tuple[str, ...]:
    """Retain the recollection and optionally search its grounded name alone.

    Both rankings later compete for one Context vote per song. The focused
    query removes uncertain scene details without guessing a song or work.
    """
    primary = expand_context_search_query(str(getattr(clue, "search_query", "")))
    if not primary:
        return ()
    targets = context_media_target_terms(clue, original_query=original_query)
    if not targets:
        return (primary,)
    canonical = targets[0]
    if canonical == _context_media_target(clue) and len(primary.split()) <= 3:
        return (primary,)
    relation = str(getattr(clue, "relation", ""))
    terms = []
    if re.search(r"OST|오에스티", relation, re.I):
        terms.append("OST")
    if re.search(r"BGM|배경\s*음악", relation, re.I):
        terms.append("배경음악")
    if re.search(r"사운드\s*트랙", relation, re.I):
        terms.append("사운드트랙")
    terms.append("삽입곡")
    focused = f"{canonical} {' '.join(terms)}"
    return tuple(dict.fromkeys((primary, focused)))


def _grounded(text: str, query: str, *, allow_search_words: bool = False) -> bool:
    compact_query = re.sub(r"\s+", "", query).casefold()
    for token in re.findall(r"[가-힣A-Za-z0-9_]{2,}", text):
        if token.casefold() in query.casefold() or token.casefold() in compact_query:
            continue
        if allow_search_words and token in _SEARCH_WORDS:
            continue
        return False
    return True


def _grounded_target(target: str, span: str) -> bool:
    """두 글자 작품명도 원문에 있어야 하며 다른 사건에서 빌려오지 않는다."""
    compact = re.sub(r"\s+", "", target).casefold()
    words = target.split()
    return (
        len(compact) >= 2
        and not any(word in _GENERIC or _NOT_WORK_WORD.match(word) for word in words)
        and target not in _NONENTITY_TARGETS
        and compact in re.sub(r"\s+", "", span).casefold()
    )


def _target(span: str) -> str:
    # 작품 종류 앞의 '어떤 청춘'과 같은 불확실한 수식어가 인접한 가수명을
    # 작품명처럼 보이게 할 수 있다. 이런 경우에는 작품명을 비워 둔다.
    for pattern in (_WORK_BEFORE, _WORK_AFTER, _EVENT_BEFORE, _PERSON):
        match = pattern.search(span)
        if not match:
            continue
        words = match.group(1).strip().split()
        if pattern is _WORK_BEFORE and any(word in {"어떤", "무슨"} for word in words):
            continue
        if pattern is _EVENT_BEFORE and (
            words[0].endswith(("에서", "으로", "하고"))
            or words[0] in {"예능", "방송", "학교", "동네", "프로그램"}
        ):
            continue
        if pattern is _WORK_AFTER:
            # 작품명 뒤에 실제 사용 관계가 보이지 않으면 '드라마 배경으로 한
            # 추억' 같은 일반 문장을 작품명으로 추정하지 않는다.
            boundary = next((
                i if _WORK_BOUNDARY.match(word) else i + 1
                for i, word in enumerate(words)
                if _WORK_BOUNDARY.match(word) or word.endswith(("에서", "에는", "에"))
            ), None)
            if boundary is None:
                continue
            words = words[:boundary]
        words = [word for word in words if word not in _GENERIC]
        if any(_NOT_WORK_WORD.match(word) for word in words):
            continue
        words = [word for word in words if word.upper() not in {"OST", "BGM", "MV"}]
        if words:
            name = " ".join(words).strip("'\"“”‘’· ")
            # "도깨비에서"처럼 작품명에 붙은 장소 조사는 원문 관계에 남긴다.
            name = re.sub(r"(?:에서|에게|의|에는|에)$", "", name)
            if name in _NONENTITY_TARGETS:
                continue
            return name[:120]
    return ""


def _literal_event_wording_only(span: str) -> bool:
    literal = _LITERAL_EVENT_WORDING.search(span)
    if not literal:
        return False
    outside_quotes = re.sub(r"['\"‘“][^'\"’”]*['\"’”]", "", span)
    if (_EXTERNAL_LYRIC_REVISION.search(outside_quotes)
            or _REPORTED_LYRIC_ADAPTATION.search(outside_quotes)):
        return False
    # Text shown inside an MV or quoted in a past performance is still an
    # external scene. A lyric/title containing the same words is not.
    preceding = span[:literal.start()]
    return not (
        not _LYRIC_START.match(span)
        and re.search(r"뮤직\s*비디오|뮤비|\bMV\b|방송|라디오|콘서트|무대", preceding, re.I)
    )


def _rule_clues(query: str) -> list[dict]:
    clues: list[dict] = []
    # 영문 약어(M.O.M)와 숫자(07.5)의 마침표는 분할하지 않는다.
    # 쉼표/한글 문장 끝은 서로 다른 사건을 분리하되 관련된 앞 문장은 보존한다.
    previous = ""
    source = normalize_cover_performance_spacing(query)
    for part in re.split(r"[,，\n]+|(?<![A-Za-z0-9])\.(?=\s|$)", source):
        span = part.strip()
        if not span:
            continue
        relation = next((name for name, pattern in _RELATIONS if pattern.search(span)), "")
        if relation == "보컬 참여 일화":
            voice = _NAMED_CHORUS.search(span)
            if voice and voice.group("person") in _GENERIC_VOICES:
                relation = ""
        if not relation:
            if clues and not previous and _RELATED_CONTINUATION.search(span):
                clue = clues[-1]
                combined = clue["_span"] + ", " + span
                clue["_span"] = combined
                clue["search_query"] = _REQUEST_END.sub("", combined).strip(" ,.?！! ")[:400]
                if _UNCERTAIN.search(span):
                    clue["confidence"] = min(clue["confidence"], 0.6)
                continue
            previous = span
            continue
        if _FOLLOWUP_ONLY.match(span):
            previous = span
            continue
        if relation == "방송·공연 일화" and _TITLE_AS_EVENT.match(span):
            previous = span
            continue
        if _literal_event_wording_only(span):
            previous = span
            continue
        if (relation in {"방송·공연 일화", "커버·리메이크·답가", "유행·밈"}
                and _PLANNED_COVER_ACTIVITY.search(span)
                and not (_REPORTED_COVER_CONTEST.search(span)
                         or _REPORTED_COVER_RECORDING.search(span))):
            previous = span
            continue
        # A past cover event followed by an independent album-art description
        # can share one clause. Keep only the event prefix for Context, while
        # the original query still supplies the image evidence. Require a song
        # anchor: a performer's picture printed on a cover is artwork alone.
        artwork_start = explicit_artwork_reference_start(span)
        if artwork_start is not None and relation in {"방송·공연 일화", "커버·리메이크·답가"}:
            prefix = span[:artwork_start].strip()
            events = ((_REPORTED_COVER_CONTEST,) if relation == "방송·공연 일화"
                      else (_REPORTED_COVER_RECORDING, _REPORTED_COVER_RENDITION))
            if any(event.search(prefix) for event in events) and _COVER_EVENT_SONG_END.search(prefix):
                span = prefix
        if _LYRIC_START.match(span) or has_explicit_artwork_reference(span):
            # '가사에 X가 나오고 드라마 Y OST였어'처럼 같은 문장에 단서가
            # 섞인 경우 실제 사용 사실 뒤만 취한다. 표지에 적힌 OST 문구는 제외.
            media = _EXPLICIT_MEDIA_USAGE.search(span)
            if media and not re.search(
                r"(?:라는|라고).{0,10}(?:글자|문구|적혀|쓰여)",
                span[media.end():media.end() + 40],
            ):
                span = span[media.start():]
                relation = next((name for name, pattern in _RELATIONS if pattern.search(span)), "")
            elif (_LYRIC_START.match(span)
                  and relation not in {"유행·밈", "제작·발매 비화"}
                  and not (not _LYRIC_LITERAL_SCOPE.match(span)
                           and _REPORTED_LYRIC_ADAPTATION.search(span))):
                previous = span
                continue
        if relation == "뮤직비디오 속 사건" and _MV_AUDIO_ANALOGY.search(span):
            previous = span
            continue
        if relation == "삽입곡·배경음악" and _OST_AUDIO_ANALOGY.search(span):
            previous = span
            continue
        if (_EVENT_AUDIO_ANALOGY.search(span)
                or _LYRIC_QUOTING_EVENT.search(span)):
            previous = span
            continue
        # 표지·앨범 커버의 묘사에서 드라마/뮤비라는 말이 등장해도 외부 사실이
        # 되지는 않는다. 다른 절의 삽입곡·제작 단서는 별도로 살아남는다.
        if has_explicit_artwork_reference(span) and not re.search(
            r"(?:OST|삽입곡|배경\s*음악|BGM|주제가|제작진|작곡|작사|가이드|녹음)"
            r".{0,35}(?:사용|쓰였|삽입|수록|나왔|흘러|불렀|퇴짜|보냈)",
            span, re.I,
        ):
            previous = span
            continue
        if (
            previous and relation in {"뮤직비디오 속 사건", "제작·발매 비화"}
            and _PRELUDE.search(previous)
            and not _LYRIC_START.match(previous)
            and not has_explicit_artwork_reference(previous)
        ):
            span = previous + ", " + span
        cleaned = _REQUEST_END.sub("", span).strip(" ,.?！! ") or span
        clues.append({
            "target": _target(span),
            "relation": relation,
            "search_query": cleaned[:400],
            "confidence": 0.6 if _UNCERTAIN.search(span) else 0.8,
            "_span": span,
        })
        if len(clues) >= 4:
            break
        previous = ""
    return clues


def _model_only_clues(query: str, model_clues: list[dict], used: set[int], rules: list[dict]) -> list[dict]:
    """채택되지 않은 모델 단서를 원문 사건에 다시 연결한다.

    규칙에 없는 외부 사용 표현을 허용하되, 모델만의 상식이나 앨범·가사 묘사를
    근거로 새 검색 경로가 열리지 않도록 관계·검색어·대상을 한 절에서 확인한다.
    """
    from src.retrieval.context_clue_specificity import preserve_context_media_search
    source = normalize_cover_performance_spacing(query)
    spans = [part.strip() for part in re.split(r"[,，\n]+|(?<![A-Za-z0-9])\.(?=\s|$)", source)]
    extras: list[dict] = []
    for index, item in enumerate(model_clues):
        if index in used:
            continue
        target = str(item.get("target") or "").strip()
        relation = str(item.get("relation") or "").strip()
        search = " ".join(str(item.get("search_query") or "").split())
        if not relation or not search or len(relation) > 160 or len(search) > 400 or len(target) > 120:
            continue
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            continue
        if not 0 < confidence <= 1:
            continue
        for span in spans:
            cue = _MODEL_EXTERNAL_CUE.search(span)
            search_relation = next(
                (name for name, pattern in _RELATIONS if pattern.search(search)), "",
            )
            already_covered = any(
                _grounded(search, rule["_span"], allow_search_words=True)
                and (
                    search == rule["search_query"]
                    or (search_relation == rule["relation"]
                        and (not target or target == rule["target"]))
                )
                for rule in rules
            )
            if (
                not cue
                or _PLANNED_COVER_ACTIVITY.search(span)
                or _LYRIC_START.match(span)
                or _literal_event_wording_only(span)
                or has_explicit_artwork_reference(span)
                or _MV_AUDIO_ANALOGY.search(span)
                or _OST_AUDIO_ANALOGY.search(span)
                or not _grounded(search, span, allow_search_words=True)
                or not _grounded(cue.group("verb"), search)
                or (target and not _grounded_target(target, span))
                or already_covered
            ):
                continue
            if any(existing["search_query"] == search for existing in extras):
                break
            extras.append({
                "target": target,
                "relation": relation if _grounded(relation, span) else cue.group("verb"),
                "search_query": preserve_context_media_search(
                    search, source_query=span,
                    relation=relation if _grounded(relation, span) else cue.group("verb"),
                    target=target,
                ),
                "confidence": min(confidence, 0.5),
            })
            break
    return extras


def apply_context_query_safeguards(query: str, raw: dict) -> dict:
    """명시된 외부 사건만 Context 단서로 보존하며 fallback도 제공한다."""
    from src.retrieval.context_clue_specificity import preserve_context_media_search
    enriched = dict(raw)
    rules = _rule_clues(query)
    model_clues = enriched.get("context_clues") or []
    if isinstance(model_clues, dict):
        model_clues = [model_clues]
    if not isinstance(model_clues, list):
        model_clues = []
    model_clues = [item for item in model_clues if isinstance(item, dict)]

    final: list[dict] = []
    used: set[int] = set()
    rule_sources = [dict(rule) for rule in rules]
    for rule in rules:
        span = rule["_span"]
        chosen = None
        # 작품·인물과 검색어가 어느 사건에 붙는지 명확할 때만 모델의
        # 관계/질의 표현을 받아들인다. 매칭되지 않는 사건에는 규칙값을 사용한다.
        for index, item in enumerate(model_clues):
            if index in used:
                continue
            candidate_target = str(item.get("target") or "").strip()
            candidate_query = str(item.get("search_query") or "").strip()
            candidate_relation = next(
                (name for name, pattern in _RELATIONS if pattern.search(candidate_query)), "",
            )
            if (
                candidate_query
                and _grounded(candidate_query, span, allow_search_words=True)
                and (not candidate_target or _grounded_target(candidate_target, span))
                and (candidate_relation == rule["relation"]
                     or (candidate_target and candidate_target == rule["target"]))
            ):
                chosen, used_index = item, index
                break
        if chosen is not None:
            used.add(used_index)
            accepted = False
            model_target = str(chosen.get("target") or "").strip()
            if model_target and _grounded_target(model_target, span):
                rule["target"] = model_target[:120]
                accepted = True
            relation = str(chosen.get("relation") or "").strip()
            if relation and len(relation) <= 160 and _grounded(relation, span):
                rule["relation"] = relation
                accepted = True
            search = " ".join(str(chosen.get("search_query") or "").split())
            if search and len(search) <= 400 and _grounded(
                search, span, allow_search_words=True,
            ):
                rule["search_query"] = preserve_context_media_search(search, source_query=rule["search_query"], relation=rule["relation"], target=rule["target"])
                accepted = True
            try:
                model_confidence = float(chosen.get("confidence"))
            except (ValueError, TypeError):
                model_confidence = rule["confidence"]
            if accepted and 0 <= model_confidence <= 1:
                rule["confidence"] = min(rule["confidence"], model_confidence)
        if _UNCERTAIN.search(span):
            rule["confidence"] = min(rule["confidence"], 0.6)
        rule.pop("_span")
        final.append(rule)
    final.extend(_model_only_clues(query, model_clues, used, rule_sources))
    enriched["context_clues"] = final[:4]
    return enriched
