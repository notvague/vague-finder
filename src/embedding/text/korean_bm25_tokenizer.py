"""
형태소 분리 모듈
"""

from __future__ import annotations

from functools import lru_cache
from itertools import groupby
from typing import List, Tuple

from kiwipiepy import Kiwi


# --------------------------------------------------
# 남길 품사
# --------------------------------------------------

_KEEP_TAGS = {
    "NNG",  # 일반명사: 밤, 사진, 영화
    "NNP",  # 고유명사: 인명, 가수명, 지명
    "NNB",  # 의존명사
    "NP",   # 대명사
    "NR",   # 수사

    "MAG",  # 일반부사: 텅, 오래
    "MAJ",  # 접속부사
    "MM",   # 관형사: 새, 옛

    "XR",   # 어근: 흐릿

    "SL",   # 영문
    "SN",   # 숫자
    "SH",   # 한자

    "IC",   # 감탄사 및 일부 고유 표현
}


# 기본형으로 변환해 남길 용언
_PREDICATE_TAGS = {
    "VV",  # 동사: 바라보 -> 바라보다
    "VA",  # 형용사: 늦 -> 늦다
    "VX",  # 보조용언: 있 -> 있다
}


# 앞 형태소와 결합해야 하는 파생 접미사
_DERIVATIONAL_TAGS = {
    "XSV",  # 동사 파생: 사랑 + 하 → 사랑하다
    "XSA",  # 형용사 파생: 흐릿 + 하 → 흐릿하다
}


# Kiwi가 URL, 해시태그 등에 붙이는 태그
_SPECIAL_TAGS = {
    "W_URL",
    "W_EMAIL",
    "W_HASHTAG",
    "W_MENTION",
    "W_SERIAL",
}


# 복합 용언을 구성할 때 사용할 품사
_COMPOUND_PART_TAGS = (
    _KEEP_TAGS
    | _PREDICATE_TAGS
    | _DERIVATIONAL_TAGS
)


# 복합 용언 생성을 시작할 품사
_COMPOUND_TRIGGER_TAGS = (
    _PREDICATE_TAGS
    | _DERIVATIONAL_TAGS
)


_DOMAIN_USER_WORDS = {
    "위로": "NNG",
}


# --------------------------------------------------
# Kiwi 인스턴스
# --------------------------------------------------
@lru_cache(maxsize=1)
def _get_kiwi() -> Kiwi:
    kiwi = Kiwi()

    for form, tag in _DOMAIN_USER_WORDS.items():
        kiwi.add_user_word(
            form,
            tag,
            0.0,
        )

    return kiwi


# --------------------------------------------------
# 내부 유틸리티
# --------------------------------------------------
def _base_tag(tag: str) -> str:
    return tag.split("-", 1)[0]


def _to_lemma(form: str) -> str:
    """
    동사와 형용사 어간을 기본형으로 변환

    예:
    - 늦 -> 늦다
    - 바라보 -> 바라보다
    - 있 -> 있다
    """
    form = form.strip()

    if not form:
        return ""

    if form.endswith("다"):
        return form

    return f"{form}다"


# --------------------------------------------------
# 실제 형태소 분석
# --------------------------------------------------
@lru_cache(maxsize=50_000)
def _tokenize_cached(text: str) -> Tuple[str, ...]:
    """
    형태소 분석 결과를 tuple로 캐싱
    """
    kiwi = _get_kiwi()

    analyzed = kiwi.tokenize(
        text,
        normalize_coda=True,
    )

    result: List[str] = []

    # Kiwi의 word_position을 이용해 원래 어절 단위로 묶기
    for _, token_group in groupby(
        analyzed,
        key=lambda token: (
            token.line_number,
            token.sent_position,
            token.word_position,
        ),
    ):
        word_tokens = list(token_group)
        word_terms: List[str] = []

        def add_term(term: str) -> None:
            normalized = term.strip().lower()

            if not normalized:
                return

            # 같은 어절 안에서 인위적으로 생성된 중복만 제거
            if normalized not in word_terms:
                word_terms.append(normalized)

        # ------------------------------------------
        # 1. 개별 형태소 처리
        # ------------------------------------------
        for token in word_tokens:
            tag = _base_tag(token.tag)
            form = token.form

            if tag in _KEEP_TAGS:
                # 명사, 부사, 어근 등은 원형 그대로 유지
                add_term(form)

            elif tag in _PREDICATE_TAGS:
                # 동사와 형용사는 기본형으로 변환
                add_term(_to_lemma(form))

            elif tag in _SPECIAL_TAGS:
                # '#안녕'과 '안녕'이 동일하게 검색되도록
                # 맨 앞의 # 또는 @를 제거
                add_term(form.lstrip("#@"))

        # ------------------------------------------
        # 2. 같은 어절의 복합 용언 생성
        # ------------------------------------------

        has_predicate = any(
            _base_tag(token.tag) in _COMPOUND_TRIGGER_TAGS
            for token in word_tokens
        )

        if has_predicate:
            compound_stem = "".join(
                token.form
                for token in word_tokens
                if _base_tag(token.tag) in _COMPOUND_PART_TAGS
            )

            if compound_stem:
                add_term(_to_lemma(compound_stem))

        # 어절 안의 토큰을 전체 결과에 추가
        # 여기서는 문장 전체 중복을 제거하지 않음
        # 같은 단어가 여러 번 등장하면 BM25의 TF에 반영
        result.extend(word_terms)

    return tuple(result)


# --------------------------------------------------
# 외부 공개 함수
# --------------------------------------------------
def tokenize_for_bm25(text: str) -> List[str]:
    """
    한국어 텍스트를 BM25 검색용 토큰 리스트로 변환

    예:
        tokenize_for_bm25("사진을 바라보는 모습")
        -> ["사진", "바라보다", "모습"]
    """
    if not text:
        return []

    return list(_tokenize_cached(str(text)))


def normalize_for_bm25(text: str) -> str:
    """
    BM25Encoder에 전달할 공백 구분 문자열을 반환

    예:
        normalize_for_bm25("사진을 바라보는 모습")
        -> "사진 바라보다 모습"
    """
    return " ".join(tokenize_for_bm25(text))