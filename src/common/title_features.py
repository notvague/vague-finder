from __future__ import annotations

import re
from typing import Any, Dict, Optional


_HANJA_RE = re.compile(r"[\u4E00-\u9FFF]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
_TITLE_ANNOTATION_RE = re.compile(r"[\(\[]([^\)\]]+)[\)\]]")
_CREDIT_ANNOTATION_RE = re.compile(
    r"^(?:feat(?:uring)?\.?|with|prod(?:uced)?\.?|remix|remaster(?:ed)?|"
    r"live|inst(?:rumental)?\.?|ost|ver(?:sion)?\.?)\b",
    re.IGNORECASE,
)
_PERSON_NAME_SUFFIXES = (
    "a",
    "ia",
    "na",
    "ella",
    "elle",
    "ina",
    "ine",
    "ette",
    "ita",
    "ito",
    "o",
    "us",
)


def base_title(title: str) -> str:
    """
    제목 구조 판정에 사용할 대표 제목.

    예:
    TTL (Time To Love) -> TTL
    I (Feat. 버벌진트) -> I
    """
    title = (title or "").strip()

    if not title:
        return ""

    # 첫 괄호 이전을 primary title로 사용
    base = re.split(r"\s*[\(\[]", title, maxsplit=1)[0].strip()

    return base or title


def analyze_title_structure(title: str) -> Dict[str, Any]:
    """
    실제 곡 제목에서 구조적 특징을 결정론적으로 계산한다.
    """
    base = base_title(title)

    if not base:
        return {
            "script": None,
            "char_count": None,
            "word_count": None,
            "contains_number": False,
            "repeated_char": False,
            "repeated_word": False,
            "has_latin_anywhere": False,
            "has_hangul_anywhere": False,
            "has_hanja_anywhere": False,
            "has_number_anywhere": False,
        }

    # 제목의 실질적인 문자만 사용.
    # 공백, 괄호, 하이픈 등의 punctuation은 글자 수에서 제외.
    chars = re.findall(
        r"[A-Za-z0-9가-힣\u4E00-\u9FFF]",
        base,
    )

    compact = "".join(chars)

    # 단어
    words = re.findall(
        r"[A-Za-z0-9가-힣\u4E00-\u9FFF]+",
        base,
    )

    has_latin = bool(re.search(r"[A-Za-z]", compact))
    has_hangul = bool(re.search(r"[가-힣]", compact))
    has_hanja = bool(_HANJA_RE.search(compact))
    has_number = bool(re.search(r"[0-9]", compact))

    kinds = sum([
        has_latin,
        has_hangul,
        has_hanja,
        has_number,
    ])

    if has_latin and kinds == 1:
        script: Optional[str] = "latin"
    elif has_hangul and kinds == 1:
        script = "hangul"
    elif has_hanja and kinds == 1:
        script = "hanja"
    elif has_number and kinds == 1:
        script = "numeric"
    elif kinds > 1:
        script = "mixed"
    else:
        script = None

    lower_compact = compact.lower()

    repeated_char = (
        len(lower_compact) >= 2
        and len(set(lower_compact)) == 1
    )

    lowered_words = [w.lower() for w in words]

    repeated_word = (
        len(lowered_words) >= 2
        and len(set(lowered_words)) == 1
    )

    # 기존 구조 판정은 대표 제목(base) 기준으로 그대로 둔다. 별칭/부제가 든
    # 괄호까지 포함한 문자 존재 여부만 별도 메타데이터로 추가해 기존 질의의
    # 결과를 바꾸지 않으면서 '무제(無題)' 같은 케이스를 보조 검색할 수 있게 한다.
    full_title = title or ""

    return {
        "script": script,
        "char_count": len(compact) if compact else None,
        "word_count": len(words) if words else None,
        "contains_number": has_number,
        "repeated_char": repeated_char,
        "repeated_word": repeated_word,
        "has_latin_anywhere": bool(re.search(r"[A-Za-z]", full_title)),
        "has_hangul_anywhere": bool(re.search(r"[가-힣]", full_title)),
        "has_hanja_anywhere": bool(_HANJA_RE.search(full_title)),
        "has_number_anywhere": bool(re.search(r"[0-9]", full_title)),
    }


def has_title_constraints(constraints: Any) -> bool:
    if constraints is None:
        return False

    return any([
        constraints.script is not None,
        constraints.char_count is not None,
        constraints.word_count is not None,
        constraints.contains_number is not None,
        constraints.repeated_char is not None,
        constraints.repeated_word is not None,
    ])


def build_title_metadata_filter(constraints: Any) -> Optional[Dict[str, Any]]:
    """
    벡터 DB 메타데이터 필터 생성.

    이것은 전체 검색에 적용하는 hard filter가 아니라
    '제목 구조 보조 검색 경로'에만 사용한다.
    """
    if not has_title_constraints(constraints):
        return None

    clauses = []

    if constraints.script is not None:
        clauses.append({
            "title_script": {"$eq": constraints.script}
        })

    if constraints.char_count is not None:
        clauses.append({
            "title_char_count": {"$eq": constraints.char_count}
        })

    if constraints.word_count is not None:
        clauses.append({
            "title_word_count": {"$eq": constraints.word_count}
        })

    if constraints.contains_number is not None:
        clauses.append({
            "title_contains_number": {
                "$eq": constraints.contains_number
            }
        })

    if constraints.repeated_char is not None:
        clauses.append({
            "title_repeated_char": {
                "$eq": constraints.repeated_char
            }
        })

    if constraints.repeated_word is not None:
        clauses.append({
            "title_repeated_word": {
                "$eq": constraints.repeated_word
            }
        })

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


def build_title_presence_metadata_filter(
    constraints: Any,
) -> Optional[Dict[str, Any]]:
    """대표 제목 밖의 한자 표기를 위한 좁고 낮은 가중치의 보조 필터.

    영문 별칭은 거의 모든 K-pop 제목에 붙을 수 있어 후보를 오염시키므로 의도적으로
    확장하지 않는다. 한자는 희소하고 q118처럼 사용자가 '한자 제목'이라고 기억할 때
    괄호 속 정식 한자 표기를 놓치는 문제만 해결한다.
    """
    if constraints is None or constraints.script != "hanja":
        return None
    return {"title_has_hanja_anywhere": {"$eq": True}}


def build_title_meaning_metadata_filter(
    clue: Any,
) -> Optional[Dict[str, Any]]:
    """제목 의미 기억 중 기존 메타데이터로 안전하게 보조 소환할 수 있는 것.

    이 필터는 전체 검색의 hard filter가 아니라 별도 additive 경로에만 쓰인다.
    사람/장소 이름인지는 현재 카탈로그에 정답 라벨이 없으므로 추측하지 않는다.
    """
    if clue is None or not getattr(clue, "kind", None):
        return None
    if clue.kind == "foreign_person_name":
        # 외국 이름 기억은 라틴 표기가 제목 어딘가에 존재한다는 약한 근사치다.
        return {"title_has_latin_anywhere": {"$eq": True}}
    if clue.kind == "sentence":
        return {"title_word_count": {"$gte": 3}}
    if clue.kind == "onomatopoeia":
        return {
            "$or": [
                {"title_repeated_char": {"$eq": True}},
                {"title_repeated_word": {"$eq": True}},
            ]
        }
    return None


def title_presence_match_ratio(title: str, constraints: Any) -> float:
    """현재 지원하는 희소 보조 단서(괄호 포함 한자)의 일치 여부."""
    if constraints is None or constraints.script != "hanja":
        return 0.0
    return float(analyze_title_structure(title)["has_hanja_anywhere"])


def title_meaning_match_ratio(title: str, clue: Any) -> float:
    """제목 문자열에서 직접 확인할 수 있는 의미/유형 단서의 약한 적합도."""
    if clue is None or not getattr(clue, "kind", None):
        return 0.0
    features = analyze_title_structure(title)
    if clue.kind == "foreign_person_name":
        if not features["has_latin_anywhere"]:
            return 0.0

        full_title = (title or "").strip()
        primary = base_title(full_title)
        primary_latin_words = _LATIN_WORD_RE.findall(primary)
        primary_has_hangul = bool(re.search(r"[가-힣]", primary))
        primary_has_latin = bool(primary_latin_words)

        # 괄호 안 Feat./With/Remix 같은 크레딧은 제목 의미의 증거가 아니다.
        # 기존 has_latin_anywhere만 사용하면 거의 모든 피처링 곡에 같은 점수가
        # 붙어 foreign_person_name 단서의 변별력이 사라진다.
        alias_words = []
        for annotation in _TITLE_ANNOTATION_RE.findall(full_title):
            annotation = annotation.strip()
            if _CREDIT_ANNOTATION_RE.search(annotation):
                continue
            words = _LATIN_WORD_RE.findall(annotation)
            if words:
                alias_words.append(words)

        def _single_name_like_score(words: list[str]) -> float:
            if len(words) != 1:
                return 0.0
            token = words[0]
            letters = token.replace("-", "").replace("'", "")
            if len(letters) < 3 or letters.isupper():
                return 0.0
            if not (letters[0].isupper() and letters[1:].islower()):
                return 0.0
            if letters.lower().endswith(_PERSON_NAME_SUFFIXES):
                return 1.0
            return 0.65

        # '까탈레나 (Catallena)'처럼 한글 대표 제목 뒤에 한 단어짜리
        # 외국어 이름 표기가 붙은 경우를 가장 강한 양성 증거로 본다.
        if primary_has_hangul and not primary_has_latin:
            alias_score = max(
                (_single_name_like_score(words) for words in alias_words),
                default=0.0,
            )
            if alias_score > 0:
                return 0.85 + 0.15 * alias_score

        # 'Maria'처럼 대표 제목 자체가 한 단어 이름형인 경우도 지원한다.
        primary_score = _single_name_like_score(primary_latin_words)
        if primary_score > 0:
            return 0.45 + 0.45 * primary_score

        # 라틴 문자가 문장/크레딧에만 있는 경우에는 후보를 제거하지 않고
        # 매우 약한 근사 신호만 남긴다.
        return 0.20
    if clue.kind == "sentence":
        return float((features["word_count"] or 0) >= 3)
    if clue.kind == "question":
        return float("?" in (title or ""))
    if clue.kind == "onomatopoeia":
        return float(
            features["repeated_char"] or features["repeated_word"]
        )
    return 0.0


def title_constraint_match_ratio(
    title: str,
    constraints: Any,
) -> float:
    """
    후보 제목이 사용자가 기억한 제목 조건과 얼마나 일치하는지
    0.0 ~ 1.0 반환.
    """
    if not has_title_constraints(constraints):
        return 0.0

    actual = analyze_title_structure(title)

    checks = []

    if constraints.script is not None:
        checks.append(
            actual["script"] == constraints.script
        )

    if constraints.char_count is not None:
        checks.append(
            actual["char_count"] == constraints.char_count
        )

    if constraints.word_count is not None:
        checks.append(
            actual["word_count"] == constraints.word_count
        )

    if constraints.contains_number is not None:
        checks.append(
            actual["contains_number"]
            == constraints.contains_number
        )

    if constraints.repeated_char is not None:
        checks.append(
            actual["repeated_char"]
            == constraints.repeated_char
        )

    if constraints.repeated_word is not None:
        checks.append(
            actual["repeated_word"]
            == constraints.repeated_word
        )

    if not checks:
        return 0.0

    return sum(checks) / len(checks)
