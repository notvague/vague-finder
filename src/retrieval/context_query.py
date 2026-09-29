"""외부 배경 사실 단서를 원문에 묶어 검증한다.

Gemini가 검색 문장을 생성하더라도 사용자가 말하지 않은 곡·작품명을 보태면
잘못된 곡을 확신하게 된다. 고정밀 규칙으로 배경 사건을 먼저 확인하고,
모델 출력 중 원문으로 뒷받침되는 표현만 채택한다.
"""
from __future__ import annotations

import re


_ARTWORK = re.compile(
    r"(?:앨범\s*(?:커버|표지|아트|자켓|재킷|사진)|"
    r"(?:표지|커버)(?!곡|한|했|해|버전)|음반\s*표지)"
)
_LYRIC_START = re.compile(r"^(?:가사(?:에|에는|에서|에선)?|노랫말|가삿말|후렴(?:에|에서|의)?)")
_MV_AUDIO_ANALOGY = re.compile(
    r"(?:뮤직\s*비디오|(?<![A-Za-z])MV(?![A-Za-z]))(?:처럼|같이|같은).{0,40}"
    r"(?:들리|소리|음악|사운드|리듬)", re.I,
)
_OST_AUDIO_ANALOGY = re.compile(
    r"(?:OST|오에스티|BGM)\s*(?:처럼|같은|같이|풍|느낌).{0,45}"
    r"(?:들리|소리|음악|사운드|분위기|노래)", re.I,
)
_EXPLICIT_MEDIA_USAGE = re.compile(
    r"(?:드라마|영화|애니(?:메이션)?|게임|예능)\s+"
    r"(?:[가-힣A-Za-z0-9·_-]+\s+){1,3}"
    r"(?:OST|오에스티|삽입곡|배경\s*음악)(?:였|이었|로|이고|으로|야)",
    re.I,
)
_RELATIONS = (
    ("삽입곡·배경음악", re.compile(
        r"(?<![A-Za-z])OST(?![A-Za-z])|오에스티|삽입곡|배경\s*음악|"
        r"(?<![A-Za-z])BGM(?![A-Za-z])|사운드\s*트랙|"
        r"주제가|테마곡|엔딩곡|오프닝곡|등장곡|로고송|선거송", re.I,
    )),
    ("다른 작품에 사용", re.compile(
        r"(?:애니(?:메이션)?|드라마|영화|게임|예능|방송|코미디빅리그)"
        r".{0,90}(?:장면|나왔|나오던|흘러|틀어|사용|수록|모티브|삽입)|"
        r"(?:게임|펌프\s*잇\s*업|FIESTA).{0,55}수록|"
        r"(?:선생님|캐릭터).{0,70}(?:장면|흘러|나오던)", re.I,
    )),
    ("뮤직비디오 속 사건", re.compile(
        r"(?:뮤직\s*비디오|(?<![A-Za-z])MV(?![A-Za-z]))", re.I,
    )),
    ("제작·발매 비화", re.compile(
        r"제작진|제작\s*비화|작곡(?:했|한|가|을|하)|작사(?:했|한|가|을|하)|"
        r"데모곡?|가이드(?:도|를|가|로)?\s*(?:녹음|버전|아르바이트)|퇴짜|"
        r"녹음.{0,25}(?:하라|하려|했다|했다던|과정|주장)|"
        r"원키.{0,20}(?:녹음|주장)|미공개|공개되지\s*않|"
        r"원래.{0,25}(?:예정|부르|발매|공개)|"
        r"(?:초기|처음|원래)\s*기획", re.I,
    )),
    ("방송·공연 일화", re.compile(
        r"(?:축제|페스티벌|콘서트|시상식|불후의\s*명곡|V앱|브이라이브)"
        r".{0,110}(?:무대|앙코르|앵콜|우승|공개|불렀|커버|사고|실수|"
        r"처음|안무|공연)", re.I,
    )),
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
        r"(?:차트|음악\s*방송|지상파|빌보드).{0,65}(?:1위|우승|기록)|"
        r"모티브가\s*됐|영향을\s*(?:줬|받았)|"
        r"(?:후보|당선).{0,60}(?:선거송|로고송)", re.I,
    )),
    ("제작·활동에 얽힌 사건", re.compile(
        r"(?:월드\s*투어|군\s*입대|소속사|밴드\s*해체)"
        r".{0,95}(?:헤어|복귀|완성|재회|다시\s*만나|다시\s*음악|"
        r"작곡|작사|감정|인기)", re.I,
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
_PERSON = re.compile(r"([가-힣]{2,5})\s*(?:선생님|감독|배우|선수|작곡가)")
_GENERIC = {
    "옛날", "예전", "무슨", "어떤", "청춘", "유명한", "한국",
    "남자", "여자", "중", "중에", "곡", "노래",
}
# 검색 문장에 추가할 수 있는 일반 명사만 허용한다. '삽입곡', '녹음' 같은
# 관계 단어를 원문 없이 허용하면 모델이 뮤직비디오를 OST로 바꿔 버릴 수 있다.
_SEARCH_WORDS = {"노래", "곡"}
_REQUEST_END = re.compile(
    r"\s*(?:뭐(?:였|야|지)|무슨\s*(?:곡|노래)(?:이었|였|이지|이야|이니|인지)|"
    r"제목이?\s*기억(?:이)?\s*안\s*나|찾고\s*싶어|찾아줘|알려줘)"
    r"[^,.!?]*[?!.]?\s*$"
)
_UNCERTAIN = re.compile(r"(?:같아|같은데|듯|아마|헷갈|쯤|줄\s*알았)")
_PRELUDE = re.compile(
    r"(?:스토리|내용|이야기|감옥|죄|장면|안무|촬영|NG|경연|탈락|"
    r"드라마|애니메이션|영화|게임|방송|음원)", re.I,
)
_FOLLOWUP_ONLY = re.compile(
    r"^(?:그|이|저)\s*(?:원곡|답가|커버곡)(?:이|은|을|의)?\s*"
    r"(?:뭐|어떤|찾|알려)",
)
_RELATED_CONTINUATION = re.compile(
    r"^(?:나중에|그다음|이후|그전(?:의|에)?|그리고|그때).{0,110}"
    r"(?:부른\s*버전|들려|커버|무대|힘들|어렵|재회|다시\s*만나)",
)


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
    return re.sub(r"\s+", "", target).casefold() in re.sub(r"\s+", "", span).casefold()


def _target(span: str) -> str:
    # 작품 종류 앞의 '어떤 청춘'과 같은 불확실한 수식어가 인접한 가수명을
    # 작품명처럼 보이게 할 수 있다. 이런 경우에는 작품명을 비워 둔다.
    for pattern in (_WORK_BEFORE, _WORK_AFTER, _PERSON):
        match = pattern.search(span)
        if not match:
            continue
        words = match.group(1).strip().split()
        if pattern is _WORK_BEFORE and any(word in {"어떤", "무슨"} for word in words):
            continue
        if pattern is _WORK_AFTER:
            # '드라마 도깨비 OST로 사용됐던'에서 관계 설명을 작품명에 섞지 않는다.
            words = words[:next(
                (i for i, word in enumerate(words)
                 if re.match(r"(?:OST|BGM|사용|수록|쓰였|나왔|곡|노래|장면|중에)", word, re.I)),
                len(words),
            )]
        words = [word for word in words if word not in _GENERIC]
        words = [word for word in words if word.upper() not in {"OST", "BGM", "MV"}]
        if words:
            name = " ".join(words).strip("'\"“”‘’· ")
            # "도깨비에서"처럼 작품명에 붙은 장소 조사는 원문 관계에 남긴다.
            name = re.sub(r"(?:에서|에게|의|에는|에)$", "", name)
            return name[:120]
    return ""


def _rule_clues(query: str) -> list[dict]:
    clues: list[dict] = []
    # 영문 약어(M.O.M)와 숫자(07.5)의 마침표는 분할하지 않는다.
    # 쉼표/한글 문장 끝은 서로 다른 사건을 분리하되 관련된 앞 문장은 보존한다.
    previous = ""
    for part in re.split(r"[,，\n]+|(?<![A-Za-z0-9])\.(?=\s|$)", query):
        span = part.strip()
        if not span:
            continue
        relation = next((name for name, pattern in _RELATIONS if pattern.search(span)), "")
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
        if _LYRIC_START.match(span) or _ARTWORK.search(span):
            # '가사에 X가 나오고 드라마 Y OST였어'처럼 같은 문장에 단서가
            # 섞인 경우 실제 사용 사실 뒤만 취한다. 표지에 적힌 OST 문구는 제외.
            media = _EXPLICIT_MEDIA_USAGE.search(span)
            if media and not re.search(
                r"(?:라는|라고).{0,10}(?:글자|문구|적혀|쓰여)",
                span[media.end():media.end() + 40],
            ):
                span = span[media.start():]
                relation = next((name for name, pattern in _RELATIONS if pattern.search(span)), "")
            elif _LYRIC_START.match(span) and relation not in {"유행·밈", "제작·발매 비화"}:
                previous = span
                continue
        if relation == "뮤직비디오 속 사건" and _MV_AUDIO_ANALOGY.search(span):
            previous = span
            continue
        if relation == "삽입곡·배경음악" and _OST_AUDIO_ANALOGY.search(span):
            previous = span
            continue
        # 표지·앨범 커버의 묘사에서 드라마/뮤비라는 말이 등장해도 외부 사실이
        # 되지는 않는다. 다른 절의 삽입곡·제작 단서는 별도로 살아남는다.
        if _ARTWORK.search(span) and not re.search(
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
            and not _ARTWORK.search(previous)
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


def apply_context_query_safeguards(query: str, raw: dict) -> dict:
    """명시된 외부 사건만 Context 단서로 보존하며 fallback도 제공한다."""
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
    for rule in rules:
        span = rule.pop("_span")
        chosen = None
        # 작품·인물과 검색어가 어느 사건에 붙는지 명확할 때만 모델의
        # 관계/질의 표현을 받아들인다. 매칭되지 않는 사건에는 규칙값을 사용한다.
        for index, item in enumerate(model_clues):
            if index in used:
                continue
            candidate_target = str(item.get("target") or "").strip()
            candidate_query = str(item.get("search_query") or "").strip()
            if candidate_target and candidate_target.casefold() in span.casefold():
                chosen, used_index = item, index
                break
            words = re.findall(r"[가-힣A-Za-z0-9_]{3,}", candidate_query)
            overlap = sum(word.casefold() in span.casefold() for word in words)
            if words and overlap >= 2 and overlap >= len(words) / 2:
                chosen, used_index = item, index
                break
        if chosen is None and len(rules) == 1 and len(model_clues) == 1:
            chosen, used_index = model_clues[0], 0
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
                rule["search_query"] = search
                accepted = True
            try:
                model_confidence = float(chosen.get("confidence"))
            except (ValueError, TypeError):
                model_confidence = rule["confidence"]
            if accepted and 0 <= model_confidence <= 1:
                rule["confidence"] = min(rule["confidence"], model_confidence)
        if _UNCERTAIN.search(span):
            rule["confidence"] = min(rule["confidence"], 0.6)
        final.append(rule)
    enriched["context_clues"] = final
    return enriched
