"""가사 단서 추출과 표면 문자열 정규화 유틸리티.

LLM 분석과 별개로 동작하는 고정밀 안전망이다. 따옴표가 있는 인용뿐 아니라
``가사에 ...라는 말``, ``...라는 가사가 있었다`` 같은 무인용 표현도 추출한다.
의미를 요약한 설명은 여기서 억지로 정확 구절로 만들지 않고 QueryAnalyzer의
semantic lyric 분석에 맡긴다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Literal


LyricClueKind = Literal["verbatim", "partial", "phonetic", "semantic"]


@dataclass(frozen=True)
class ExtractedLyricClue:
    text: str
    kind: LyricClueKind
    confidence: float
    source: str


_LYRIC_CUES = (
    "가사",
    "가삿말",
    "노랫말",
    "구절",
    "후렴",
    "도입부",
    "브릿지",
)

_QUOTED_RE = re.compile(
    r"[\"'“”‘’]([^\"'“”‘’\n]{2,120})[\"'“”‘’]"
)

# ``가사에 너를 사랑해도 되겠니 라는 말이 있었어``
_CUE_BEFORE_RE = re.compile(
    r"(?:가사(?:에|에는|에서|중에?)?|가삿말(?:에|에는)?|노랫말(?:에|에는)?|"
    r"후렴(?:에|에는|에서)?|도입부(?:에|에는|에서)?)"
    r"\s*(?:이|가|은|는|도)?\s*[:：-]?\s*"
    r"(?P<phrase>[^,.!?\n]{2,100}?)\s*"
    r"(?:이라는|라는|이라고|라고|이런|그런|비슷한)\s*"
    r"(?:말|가사|가삿말|노랫말|구절|부분|소리)"
)

# ``너를 사랑해도 되겠니라는 가사가 있었어``
_PHRASE_BEFORE_RE = re.compile(
    r"(?:^|[,.!?]\s*|(?:근데|그리고|또)\s+)"
    r"(?P<phrase>[^,.!?\n]{2,100}?)\s*"
    r"(?:이라는|라는|이라고|라고)\s*"
    r"(?:가사|가삿말|노랫말|구절|말|소리)"
    r"(?:이|가|은|는|도|을|를)?\s*"
    r"(?:있|나오|들리|반복|시작|기억|했|한)"
)

# ``너를 사랑해도 되겠니 가사가 있었어``처럼 인용 표지가 생략된 표현.
_BARE_PHRASE_RE = re.compile(
    r"(?:^|[,.!?]\s*)"
    r"(?P<phrase>[^,.!?\n]{3,80}?)\s+"
    r"(?:가사|가삿말|노랫말|구절)"
    r"(?:이|가|은|는|도|을|를)?\s*"
    r"(?:있|나오|들리|반복|시작|기억|였|이었)"
)

# ``후렴이 너를 사랑해도 되겠니 하고 나왔어``
_CUE_COPULA_RE = re.compile(
    r"(?:가사|가삿말|노랫말|후렴|도입부)(?:이|가|은|는)[ \t]*"
    r"(?P<phrase>[^,.!?\n]{2,80}?)[ \t]*"
    r"(?:하고[ \t]*(?:나오|나왔|들리|들렸|시작)|"
    r"라고[ \t]*(?:했|나오|나왔|들리|들렸))"
)

_SEMANTIC_MARKERS = (
    "내용",
    "식의",
    "식으로",
    "이야기",
    "주제",
    "분위기",
    "느낌",
    "의미",
    "횟수",
)

_SEMANTIC_ENDINGS = (
    "다는",
    "한다는",
    "했다는",
    "였다는",
    "있다는",
    "없다는",
    "했던",
    "하는",
    "하는 내용",
)

_LEADING_FILLERS = re.compile(
    r"^(?:(?:가사|가삿말|노랫말|후렴)(?:에|에는|에서|중에|는|가)?|"
    r"(?:그|이)?\s*노래(?:인데|에서|에|는)?|"
    r"곡(?:인데|에서|에|은|는)?|아마|대충|뭔가|그리고|근데)\s*"
)


def normalize_lyric_surface(text: str) -> str:
    """활용형과 문자 순서는 유지하고 표기상 잡음만 제거한다."""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    # Unicode 영문/숫자/한글을 모두 보존해 외국어 가사와 음차 단서도 같은 경로로
    # 처리한다. 공백·구두점만 제거하고 어간/활용/문자 순서는 바꾸지 않는다.
    return "".join(character for character in normalized if character.isalnum())


def normalize_with_origins(text: str) -> tuple[str, list[int]]:
    """정규화 결과와, 글자마다 **원문의 어느 자리에서 왔는지**를 함께 돌려준다.

    왜 필요한가. 일치 판정은 정규화된 문자열 위에서 하는데(`normalize_lyric_surface`),
    인용하려면 원문의 대소문자·띄어쓰기·줄바꿈이 살아 있는 구간이 필요하다. 그
    다리가 이 목록이다 — `origins[i]`는 정규화 결과 i번째 글자를 만든 원문 인덱스다.

    **글자 단위로 정규화한다.** 문자열 전체를 NFKC 하면 길이가 달라질 수 있어
    (합자 분해, 전각→반각) 위치를 맞출 수 없다. 대신 글자 단위 NFKC가 전체 NFKC와
    **항상 같지는 않다** — 결합 문자가 앞 글자와 합쳐지는 경우가 그렇다. 그래서
    이 함수를 쓰는 쪽은 되짚은 구간을 **다시 정규화해 원래 구절과 같은지 확인**해야
    한다(`LyricsExactSearchService.excerpt`가 그렇게 한다). 확인이 실패하면 인용을
    내지 않는다. 틀린 인용보다 없는 편이 낫다.
    """
    characters: list[str] = []
    origins: list[int] = []
    for index, character in enumerate(text or ""):
        for piece in unicodedata.normalize("NFKC", character).lower():
            if piece.isalnum():
                characters.append(piece)
                origins.append(index)
    return "".join(characters), origins


def char_ngrams(text: str, n: int = 3) -> frozenset[str]:
    normalized = normalize_lyric_surface(text)
    if not normalized:
        return frozenset()
    if len(normalized) <= n:
        return frozenset({normalized})
    return frozenset(
        normalized[index : index + n]
        for index in range(len(normalized) - n + 1)
    )


def _has_lyric_cue(text: str) -> bool:
    return any(cue in text for cue in _LYRIC_CUES)


def _clean_phrase(value: str) -> str:
    phrase = unicodedata.normalize("NFKC", value or "").strip()
    phrase = _LEADING_FILLERS.sub("", phrase)
    phrase = re.sub(r"\s+(?:이런|그런|비슷한)$", "", phrase)
    phrase = phrase.strip(" \t\r\n\"'“”‘’~…-–—")
    return re.sub(r"\s+", " ", phrase)


def _is_safe_surface_phrase(phrase: str, *, quoted: bool) -> bool:
    if not phrase:
        return False
    if len(phrase) < 2 or len(phrase) > 100:
        return False
    if not quoted and len(phrase.split()) > 14:
        return False
    if not quoted and any(mark in phrase for mark in ("\"", "'", "“", "”", "‘", "’")):
        return False
    # 다른 규칙이 문장 시작부터 ``... 가사에 실제 구절`` 전체를 다시 잡는 경우를
    # 차단한다. 고정밀 규칙 경로에서는 검색 문맥이 섞인 문자열을 원문 가사로 쓰지 않는다.
    if not quoted and _has_lyric_cue(phrase):
        return False
    if not quoted and any(marker in phrase for marker in _SEMANTIC_MARKERS):
        return False
    if not quoted and phrase.endswith(_SEMANTIC_ENDINGS):
        return False
    if normalize_lyric_surface(phrase).isdigit():
        return False
    return True


def _dedupe(clues: Iterable[ExtractedLyricClue]) -> List[ExtractedLyricClue]:
    best: dict[str, ExtractedLyricClue] = {}
    for clue in clues:
        key = normalize_lyric_surface(clue.text)
        if not key:
            continue
        current = best.get(key)
        if current is None or clue.confidence > current.confidence:
            best[key] = clue
    return list(best.values())


def extract_lyric_clues(query: str) -> List[ExtractedLyricClue]:
    """질의에서 고신뢰 표면 가사 구절을 추출한다.

    따옴표만 있다고 가사로 간주하지 않는다. 인용 주변이나 전체 질의에 가사 단서가
    있어야 하며, 제목을 인용한 일반 질의의 오탐을 피한다.
    """
    if not query or not _has_lyric_cue(query):
        return []

    clues: List[ExtractedLyricClue] = []

    for match in _QUOTED_RE.finditer(query):
        start = max(0, match.start() - 36)
        end = min(len(query), match.end() + 36)
        context = query[start:end]
        phrase = _clean_phrase(match.group(1))
        if _has_lyric_cue(context) and _is_safe_surface_phrase(phrase, quoted=True):
            clues.append(
                ExtractedLyricClue(
                    text=phrase,
                    kind="verbatim",
                    confidence=1.0,
                    source="quoted_rule",
                )
            )

    for pattern, confidence, source in (
        (_CUE_BEFORE_RE, 0.92, "cue_before_rule"),
        (_PHRASE_BEFORE_RE, 0.90, "phrase_before_rule"),
        (_BARE_PHRASE_RE, 0.82, "bare_phrase_rule"),
        (_CUE_COPULA_RE, 0.82, "cue_copula_rule"),
    ):
        for match in pattern.finditer(query):
            phrase = _clean_phrase(match.group("phrase"))
            if _is_safe_surface_phrase(phrase, quoted=False):
                clues.append(
                    ExtractedLyricClue(
                        text=phrase,
                        kind="partial",
                        confidence=confidence,
                        source=source,
                    )
                )

    return _dedupe(clues)
