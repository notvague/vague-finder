"""사용자용 검색 근거 설명 — "내가 기억한 어떤 단서가 이 곡과 연결됐는가".

`src/retrieval/explain.py`의 실행 기록은 "이 곡은 이런 계산으로 이 자리에 왔다"를 숫자로 말한다.
이 모듈은 그 기록과 곡 자료를 **사용자가 말한 단서** 쪽에서 다시 읽는다 — 단서마다 곡에서 무엇을
확인했는지와, 그것이 실제 점수·순서에 반영됐는지를 따로 적는다.

## 원칙

- **모델·DB를 부르지 않는다.** 입력은 같은 요청의 원문·분석·답변·결과 곡·실행 기록·가사 인용이고,
  어느 것도 변경하지 않는다. 문장은 전부 템플릿이다 — 기록에 없는 주장은 만들 수 없다.
- **정보 일치와 순위 기여는 독립이다.** 곡에 남성이라고 등록돼 있어도 `vocal_gender_match` 기록이
  없으면 "검색 점수에 반영됐다"고 쓰지 않는다. 반대로 기록이 없다고 "그 단서를 쓰지 않았다"고도
  단정하지 않는다 — "순위 반영 기록 없음"까지만 말한다.
- **자료 없음은 미확인이다.** 장면·속성이 없다는 판정으로 바꾸지 않는다. 인디를 말했는데 등록 장르가
  댄스여도 "인디가 아니다"가 아니라 "인디 여부를 확인할 자료가 없다"다(분류 축이 다르다).
- **현재 단서는 유효한 최신 답변이 우선한다.** 남성→여성으로 정정했으면 표는 여성 기준이다. 검색이
  남성 기준으로 더한 가산은 상세 기록에 남지만(지우지 않는다) 여성 단서의 근거로 연결하지 않는다 —
  가산 기록의 `slot`·`value`(검색이 실제로 비교한 값)와 현재 단서가 같을 때만 연결하고, 다른 줄에는
  "정정 전 값(남성) 기준" 같은 표시를 붙인다.
- **원문 규칙값과 검색이 쓴 값이 다르면 단정하지 않는다.** 원문에서 읽은 성별·장르(`known_slot_values`)가
  검색용 분석의 값과 어긋나는 질의가 있다("남자 가수 … 답가로 나온 여자 버전" → 규칙 혼성, 분석 여성).
  그때는 어느 쪽이 사용자 뜻인지 모르므로 일치·다름을 말하지 않고 미확인으로 둔다.
- **없는 인용을 만들지 않는다.** 가사 원문 인용은 같은 실행이 되짚은 것(`LyricExcerpt`)만 쓴다.
  exact·음차·근사 일치는 다른 말로 적고, 발췌·요약·키워드 겹침은 "가사에서 확인"이라고 하지 않는다.
  의미 검색 유사도는 "비슷하게 평가했다"까지만 말한다.
- 모델의 자유 서술(`model_notes`)은 쓰지 않는다. 가사·상상 이미지·표지를 MV 장면의 근거로 쓰지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.retrieval.explain import (
    LYRIC_ORDER_RULES,
    RERANK_SKIP_MV_SCENE_UNVERIFIABLE,
    RULE_LABELS,
    ScoreAdjustment,
    SongExplain,
)
from src.retrieval.known_slots import genre_aliases, known_genre_values, known_slot_values
from src.retrieval.rerank_policy import mv_scene_phrases

# ---------------------------------------------------------------------------
# 상태 · 라벨
# ---------------------------------------------------------------------------

STATUS_MATCH = "match"
STATUS_PARTIAL = "partial"
STATUS_MISMATCH = "mismatch"
STATUS_UNVERIFIED = "unverified"
STATUS_LABELS: Dict[str, str] = {
    STATUS_MATCH: "일치",
    STATUS_PARTIAL: "부분 일치",
    STATUS_MISMATCH: "다름",
    STATUS_UNVERIFIED: "미확인",
}

# 단서 종류(행의 slot). 초기 범위 — 성별·장르·가사·시기·MV 장면.
SLOT_GENDER = "vocal_gender"
SLOT_GENRE = "genre"
SLOT_LYRICS = "lyrics"
SLOT_ERA = "release_era"
SLOT_MV = "mv_scene"

# 단서의 출처. 원문(query) · 재질문 답변(answer) · 분석이 뽑거나 해석한 값(analysis) · 출생 연도로 유도(birth_year).
ORIGIN_QUERY = "query"
ORIGIN_ANSWER = "answer"
ORIGIN_ANALYSIS = "analysis"
ORIGIN_BIRTH_YEAR = "birth_year"

# 행의 세부 종류(detail_kind) — 요약 문장이 "무엇을 확인했는가"를 행에서 읽는 열쇠. 문구를 파싱하지 않기 위한 것.
KIND_LYRIC_EXACT = "lyric_exact"             # 가사 원문 경로 exact 일치
KIND_LYRIC_PHONETIC = "lyric_phonetic"       # 가사 원문 경로 음차 일치
KIND_LYRIC_FUZZY = "lyric_fuzzy"             # 가사 원문 경로 근사 일치
KIND_LYRIC_HIGHLIGHT = "lyric_highlight"     # 가사 발췌(highlight)에서 일치
KIND_LYRIC_HIGHLIGHT_PHONETIC = "lyric_highlight_phonetic"   # 음차 추정 표기가 발췌에서 일치
KIND_LYRIC_SUMMARY = "lyric_summary"         # 가사 요약에서 일치 — 요약은 가사가 아니다
KIND_LYRIC_SUMMARY_PHONETIC = "lyric_summary_phonetic"
KIND_LYRIC_COMBINED = "lyric_combined"       # 발췌·요약을 이어 붙인 경계에서 일치
KIND_LYRIC_COMBINED_PHONETIC = "lyric_combined_phonetic"
KIND_LYRIC_UNATTRIBUTED = "lyric_unattributed"   # 가사 경로 기여는 있는데 어느 구절인지 기록이 없다
KIND_LYRIC_KEYWORDS = "lyric_keywords"       # 키워드 토큰 겹침
KIND_LYRIC_SEMANTIC = "lyric_semantic"       # 뜻으로 기억한 가사
KIND_LYRIC_UNMATCHED = "lyric_unmatched"     # 표면 단서인데 일치를 확인하지 못함
KIND_SLOT_CONFLICT = "slot_conflict"         # 원문 규칙값 ≠ 검색이 쓴 값
KIND_ERA_IN_WINDOW = "era_in_window"
KIND_ERA_NEAR = "era_near"
KIND_ERA_LATER = "era_later"                 # 창보다 나중 발매
KIND_ERA_EARLIER_THAN_ESTIMATE = "era_earlier_than_estimate"  # 추정 창보다 먼저 — 그 시절에 들었을 수 있다
KIND_ERA_NO_DATE = "era_no_date"
KIND_ERA_NEEDS_BIRTH_YEAR = "era_needs_birth_year"

# 순위 기여의 요약 문구. 기록이 있을 때만 "반영"을 쓴다.
BASIS_SCORED = "검색 점수에 반영"
BASIS_SCORED_SEARCH_VALUE = "검색 점수에 반영(검색이 해석한 값 기준)"
BASIS_PENALIZED_SEARCH_VALUE = "검색 점수에서 감점(검색이 해석한 값 기준)"
BASIS_PENALIZED = "검색 점수에서 감점"
BASIS_PROTECTED = "먼저 배치"
BASIS_SCORED_AND_PROTECTED = "검색 점수에 반영 · 먼저 배치"
BASIS_NEAR = "근접 가산(기간 안 일치는 아님)"
BASIS_SEMANTIC = "의미 검색 유사도로 반영(구절 확인 아님)"
BASIS_SUPPLEMENTARY = "순위 계산에 쓰이지 않음"
BASIS_RERANK_SKIPPED = "자료가 없어 리랭킹 생략"
BASIS_NONE = "순위 반영 기록 없음"
# 이전 사용자 제보로 후보에 들어온 곡의 가산·감점. 그 기록은 "후보에 드는가"를 가릴 때만 쓰였고 자리는 후보 끝으로
# 고정됐다(`search_router.pin_feedback_entrants`). "검색 점수에 반영"이라고 하면 순위를 만든 것처럼 읽힌다.
BASIS_ENTRY_ONLY = "후보에 드는 데만 쓰임 · 자리는 후보 끝(순위는 리랭커가 정함)"

# 제보로 들어온 곡의 설명은 이 문장으로 시작한다. 출처만 말한다 — 제보 내용을 사실로 확인했다거나 이 곡이
# 정답이라는 뜻으로 쓰지 않는다.
FEEDBACK_SOURCE_SENTENCE = "이전 사용자 제보를 참고해 후보에 포함한 곡이에요(제보 내용을 사실로 확인한 것은 아니에요)."

# RankingBasis.kind
BASIS_KIND_ADJUSTMENT = "adjustment"
BASIS_KIND_PATH = "path"
BASIS_KIND_ORDER_RULE = "order_rule"
BASIS_KIND_RERANK_SKIP = "rerank_skip"

_GENDERS = ("남성", "여성", "혼성")
_GENDER_RULES = ("vocal_gender_match", "vocal_gender_partial", "vocal_gender_mismatch")
_GENDER_RULE_STATUS = {
    "vocal_gender_match": STATUS_MATCH,
    "vocal_gender_partial": STATUS_PARTIAL,
    "vocal_gender_mismatch": STATUS_MISMATCH,
}

# 사용자 장르 표현(known_slots가 정규화한 값) → 코퍼스 장르 표기. 한 표현이 여러 등록 표기에 걸칠 수 있다.
# 여기 없는 값(재질문 선택지에서 고른 등록 표기 등)은 그 값 자체로 비교한다.
_GENRE_CORPUS: Dict[str, Tuple[str, ...]] = {
    "발라드": ("발라드",),
    "댄스": ("댄스",),
    "힙합": ("랩/힙합",),
    "알앤비": ("R&B/Soul",),
    "소울": ("R&B/Soul",),
    "록": ("록/메탈",),
    "메탈": ("록/메탈",),
    "인디": ("인디음악",),
    "포크": ("포크/블루스",),
    "블루스": ("포크/블루스",),
    "일렉트로니카": ("일렉트로니카",),
    "트로트": ("성인가요/트로트",),
    "재즈": ("재즈",),
}

# 경로 코드 → 제한된 설명에서 부를 이름("…이 비슷하게 평가한 후보예요").
_PATH_SEARCH_NAMES: Dict[str, str] = {
    "text_hybrid": "텍스트 의미·키워드 검색",
    "lyrics_surface": "가사 구절 검색",
    "image": "앨범 이미지 검색",
    "audio": "소리 검색",
    "balanced_semantic": "의미 검색",
    "title": "제목 구조 검색",
    "title_presence": "제목 표기 검색",
    "title_meaning": "제목 의미 검색",
    "performance": "보컬·편성 단서 검색",
    "performance_metadata": "편성 메타데이터 검색",
    "metadata": "메타데이터 검색",
    "feedback_case": "이전 사용자 제보",
}

# 사실 일치 규칙 — "직접 확인한 것은 아니다"를 붙이지 않는다(제목·가수·형태는 등록 정보와 맞춘 것이다).
_FACT_RULES = frozenset({"title_exact", "artist_match", "artist_type", "title_hanja_presence", "title_constraints"})
# 이 단서 범위 안에서 다루는 규칙 — 제한 문장의 "그 밖의 기록"에서 뺀다.
_SLOT_RULES = frozenset({*_GENDER_RULES, "genre_match", "release_era", "answer_match"})

# 범위 밖이지만 가산이 걸리는 거리(search_router._release_era_similarity와 같다: 1·2·3년).
_ERA_NEAR_YEARS = 3


# ---------------------------------------------------------------------------
# 결과 구조
# ---------------------------------------------------------------------------


@dataclass
class RankingBasis:
    """이 단서와 **연결되는** 실제 기록 하나 — 가산·경로 기여·순서 규칙·정책 생략."""

    kind: str
    code: str
    label: str
    delta: Optional[float] = None


@dataclass
class ClueCheck:
    """기억한 단서 한 줄: 무엇을 말했고, 곡에서 무엇을 확인했고, 그것이 순위에 어떻게 닿았는가."""

    slot: str
    clue: str
    clue_origin: str
    observed: str
    status: str
    status_label: str
    source: str
    ranking_basis: List[RankingBasis] = field(default_factory=list)
    basis_label: str = BASIS_NONE
    detail_kind: str = ""


@dataclass
class AdjustmentNote:
    """상세 기록의 가산 줄에 붙일 서버 문구 — 그 가산이 지금 단서가 아니라 다른 값으로 계산됐을 때."""

    rule: str
    slot: str
    value: str
    note: str


@dataclass
class UserExplanation:
    summary: str
    clue_checks: List[ClueCheck] = field(default_factory=list)
    adjustment_notes: List[AdjustmentNote] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 공용 도구
# ---------------------------------------------------------------------------


def _norm(text: Any) -> str:
    return "".join(str(text or "").split()).lower()


def _latest_answer(answers: Sequence[Any], slot: str) -> Optional[str]:
    """그 슬롯의 **유효한 최신** 답. 건너뛴 답·빈 값은 정정이 아니므로 무시하고 더 앞의 답을 본다."""
    for answer in reversed(list(answers or ())):
        if getattr(answer, "slot", None) != slot or getattr(answer, "skipped", False):
            continue
        value = str(getattr(answer, "value", "") or "").strip()
        if value:
            return value
    return None


def _release_year(release_date: Any) -> Optional[int]:
    found = re.match(r"\s*((?:19|20)\d{2})", str(release_date or ""))
    return int(found.group(1)) if found else None


def _adj_basis(adj: ScoreAdjustment) -> RankingBasis:
    return RankingBasis(
        kind=BASIS_KIND_ADJUSTMENT,
        code=adj.rule,
        label=RULE_LABELS.get(adj.rule, adj.rule),
        delta=adj.delta,
    )


def _label_from_basis(status: str, basis: Sequence[RankingBasis]) -> str:
    if not basis:
        return BASIS_NONE
    return BASIS_PENALIZED if status == STATUS_MISMATCH else BASIS_SCORED


def _label_for_search_value(basis: Sequence[RankingBasis]) -> str:
    """값이 충돌한 줄의 순위 기여 문구. 줄의 상태가 미확인이라 기록의 부호로 가른다 — 감점이면 감점이라고 쓴다."""
    if not basis:
        return BASIS_NONE
    if any(b.delta is not None and b.delta < 0 for b in basis):
        return BASIS_PENALIZED_SEARCH_VALUE
    return BASIS_SCORED_SEARCH_VALUE


@dataclass
class _Context:
    """한 곡의 설명을 만드는 동안만 쓰는 메모 — 슬롯별 **현재 단서 값**과 그 출처.

    상세 기록의 가산 줄이 지금 단서와 같은 값으로 계산된 것인지 가릴 때 쓴다(`_adjustment_notes`).
    """

    current: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)    # slot → [(값, 출처)]

    def note_current(self, slot: str, value: str, origin: str) -> None:
        self.current.setdefault(slot, []).append((value, origin))


# ---------------------------------------------------------------------------
# 성별
# ---------------------------------------------------------------------------


def _check_gender(
    query: str,
    search_analysis: Any,
    answers: Sequence[Any],
    track: Any,
    explain: SongExplain,
    ctx: _Context,
) -> Optional[ClueCheck]:
    used = str(getattr(search_analysis, "vocal_gender", "") or "")
    used = used if used in _GENDERS else ""
    conflict = ""
    clue = _latest_answer(answers, "vocal_gender")
    origin = ORIGIN_ANSWER
    if clue not in _GENDERS:
        rule = known_slot_values(query).get("vocal_gender")
        if rule and used and rule != used:
            # 원문 규칙값과 검색이 쓴 값이 다르다 — 어느 쪽이 사용자 뜻인지 모른다. 단정하지 않는다.
            # 규칙값은 사용자가 쓴 글자가 아니라 **규칙이 읽은 값**이다("남자 가수 … 여자 버전" → 혼성).
            clue, origin, conflict = used, ORIGIN_ANALYSIS, rule
        elif rule:
            clue, origin = rule, ORIGIN_QUERY
        else:
            return None
    ctx.note_current(SLOT_GENDER, clue, origin)

    actual = str(getattr(track, "vocal_gender", "") or "").strip()
    if conflict:
        status = STATUS_UNVERIFIED
        observed = (
            f"등록된 보컬 정보: {actual if actual in _GENDERS else '없음'} · 원문에서 읽은 값({conflict})과 "
            f"검색이 해석한 값({used})이 달라 일치 여부를 단정하지 않음"
        )
    elif actual not in _GENDERS:
        status, observed = STATUS_UNVERIFIED, "등록된 보컬 정보 없음"
    elif actual == clue:
        status, observed = STATUS_MATCH, f"등록된 보컬 정보: {actual}"
    elif actual == "혼성":
        # 말한 성별의 보컬이 들어 있을 수 있다 — 단일 성별의 완전 일치와는 구분한다.
        status, observed = STATUS_PARTIAL, f"등록된 보컬 정보: 혼성(남녀 함께) — {clue} 보컬이 포함될 수 있음"
    elif clue == "혼성":
        status, observed = STATUS_MISMATCH, f"등록된 보컬 정보: {actual}(혼성으로 등록되지 않음)"
    else:
        status, observed = STATUS_MISMATCH, f"등록된 보컬 정보: {actual}"

    basis: List[RankingBasis] = []
    for adj in explain.adjustments:
        if adj.rule in _GENDER_RULES:
            # 검색이 비교한 값(기록의 value, 없으면 검색용 분석의 값)이 **지금 단서와 같을 때만** 연결한다.
            # 정정 전 값으로 더해진 가산은 현재 단서의 근거가 아니다(상세 줄의 표시는 `_adjustment_notes`).
            compared = adj.value or used
            if compared == clue and (conflict or _GENDER_RULE_STATUS[adj.rule] == status):
                basis.append(_adj_basis(adj))
        elif adj.rule == "answer_match" and adj.slot == "vocal_gender":
            if adj.value == clue and status == STATUS_MATCH:
                basis.append(_adj_basis(adj))

    basis_label = _label_for_search_value(basis) if conflict else _label_from_basis(status, basis)

    return ClueCheck(
        slot=SLOT_GENDER,
        clue=f"{clue} 보컬(검색이 해석한 값)" if conflict else f"{clue} 보컬",
        clue_origin=origin,
        observed=observed,
        status=status,
        status_label=STATUS_LABELS[status],
        source="track.vocal_gender",
        ranking_basis=basis,
        basis_label=basis_label,
        detail_kind=KIND_SLOT_CONFLICT if conflict else "",
    )


# ---------------------------------------------------------------------------
# 장르
# ---------------------------------------------------------------------------


def _genre_family(expression: Any) -> frozenset:
    """장르 표현이 가리키는 **등록 장르 표기의 집합**. 모르는 표현이면 빈 집합.

    표기가 달라도 같은 장르면 같은 집합이 된다 — "인디"·"인디음악", "일렉트로닉"·"EDM"·"일렉트로니카",
    "힙합"·"랩/힙합", "락"·"록/메탈". 붙여 쓴 복합 장르는 둘 다 가리킨다 — "록발라드" → 록/메탈 + 발라드.
    다른 표기는 원문 판정과 같은 표(`known_slots.genre_aliases`)에서 읽는다.

    **장르를 담는 값에만 쓴다**(분석의 genre · 답변 · 원문에서 뽑은 장르 낱말 · 곡의 등록 장르). 낱말이 들어 있는지를
    글자로 보므로 일반 문장에 쓰면 "기록"의 록을 읽는다 — 원문 전체의 판정은 `known_genre_values`가 한다.
    """
    hay = _norm(expression)
    if not hay:
        return frozenset()
    family = set()
    for word, tokens in _GENRE_CORPUS.items():
        if _norm(word) in hay or any(_norm(token) in hay for token in tokens):   # 사용자 낱말 · 등록 표기
            family.update(tokens)
        # 등록 표기의 한쪽 낱말 그대로인 값 — "랩"(랩/힙합), "Soul"(R&B/Soul). 글자 전체가 같을 때만 본다
        family.update(token for token in tokens if hay in {_norm(part) for part in token.split("/")})
    for alias, canon in genre_aliases().items():
        if _norm(alias) in hay:
            family.update(_GENRE_CORPUS.get(canon, ()))
    return frozenset(family)


def _genre_consistent(clue: str, compared: str) -> bool:
    """두 장르 표현이 같은 장르를 가리키는가. 표기가 아니라 계열로 본다."""
    a, b = _norm(clue), _norm(compared)
    if not a or not b:
        return False
    family_a, family_b = _genre_family(clue), _genre_family(compared)
    if family_a and family_b:
        return bool(family_a & family_b)
    return a in b or b in a      # 표에 없는 장르 — 글자로만 견준다


def _genre_in(clue: str, registered: str) -> bool:
    """등록 장르(예: "발라드, 국내드라마")가 이 단서의 장르를 담고 있는가."""
    return bool(_norm(registered)) and _genre_consistent(clue, registered)


def _stated_genres(query: str) -> List[str]:
    """원문의 장르 진술 전부. 같은 등록 표기를 가리키는 표현(록·메탈)은 한 단서로 묶는다 — "록·메탈"."""
    grouped: Dict[Any, List[str]] = {}
    for value in known_genre_values(query):
        grouped.setdefault(_genre_family(value) or value, []).append(value)
    return ["·".join(values) for values in grouped.values()]


def _genre_row(
    clue: str,
    origin: str,
    conflict: str,
    used: str,
    registered: str,
    explain: SongExplain,
    ctx: _Context,
) -> ClueCheck:
    parts = clue.split("·") if origin == ORIGIN_QUERY else [clue]
    if conflict:
        status = STATUS_UNVERIFIED
        observed = (
            f"등록 장르: {registered or '정보 없음'} · 원문에서 읽은 값({conflict})과 검색이 해석한 값({used})이 "
            "달라 일치 여부를 단정하지 않음"
        )
    elif not registered:
        status, observed = STATUS_UNVERIFIED, "등록 장르 정보 없음"
    elif any(_genre_in(part, registered) for part in parts):
        status, observed = STATUS_MATCH, f"등록 장르: {registered}"
    else:
        # 멜론 세분류와 사용자 체감(인디·상위/하위 분류·복수 장르)은 다른 축이다. 문자열이 다르다고 모순이 아니다.
        status, observed = STATUS_UNVERIFIED, f"등록 장르는 {registered}이며 {clue} 여부를 확인할 자료는 없음"

    basis: List[RankingBasis] = []
    for adj in explain.adjustments:
        if adj.rule == "genre_match":
            compared = adj.value or used
            if any(_genre_consistent(part, compared) for part in parts) and (conflict or status == STATUS_MATCH):
                basis.append(_adj_basis(adj))
        elif adj.rule == "answer_match" and adj.slot == "genre":
            if adj.value == clue and status == STATUS_MATCH:
                basis.append(_adj_basis(adj))

    basis_label = _label_for_search_value(basis) if conflict else _label_from_basis(status, basis)

    return ClueCheck(
        slot=SLOT_GENRE,
        clue=f"{clue}(검색이 해석한 값)" if conflict else clue,
        clue_origin=origin,
        observed=observed,
        status=status,
        status_label=STATUS_LABELS[status],
        source="track.genre",
        ranking_basis=basis,
        basis_label=basis_label,
        detail_kind=KIND_SLOT_CONFLICT if conflict else "",
    )


def _check_genre(
    query: str,
    search_analysis: Any,
    answers: Sequence[Any],
    track: Any,
    explain: SongExplain,
    ctx: _Context,
) -> List[ClueCheck]:
    """장르 단서 행. 사용자가 장르를 둘 말했으면("R&B 발라드") 진술마다 한 줄이다 — 첫 값만 보면 검색이 고른
    둘째 값과 어긋나 보인다. 유효한 최신 답변이 있으면 그 답 하나가 현재 단서다."""
    used = str(getattr(search_analysis, "genre", "") or "").strip()
    registered = str(getattr(track, "genre", "") or "").strip()

    answer = _latest_answer(answers, "genre")
    if answer:
        ctx.note_current(SLOT_GENRE, answer, ORIGIN_ANSWER)
        return [_genre_row(answer, ORIGIN_ANSWER, "", used, registered, explain, ctx)]

    stated = _stated_genres(query)
    if not stated:
        return []
    if _genre_family(used) and not any(_genre_consistent(part, used) for clue in stated for part in clue.split("·")):
        # 검색이 쓴 값이 **아는 장르**인데 원문의 어느 장르 진술과도 다르다 — 어느 쪽이 사용자 뜻인지 모른다.
        # 검색값의 계열을 모르면("OST"·"K-POP") 충돌이 아니다 — 사용자가 말한 장르를 그대로 단서로 둔다.
        ctx.note_current(SLOT_GENRE, used, ORIGIN_ANALYSIS)
        return [_genre_row(used, ORIGIN_ANALYSIS, "·".join(stated), used, registered, explain, ctx)]
    rows: List[ClueCheck] = []
    for clue in stated:
        for part in clue.split("·"):
            ctx.note_current(SLOT_GENRE, part, ORIGIN_QUERY)
        rows.append(_genre_row(clue, ORIGIN_QUERY, "", used, registered, explain, ctx))
    return rows


def _adjustment_notes(explain: SongExplain, ctx: _Context) -> List[AdjustmentNote]:
    """상세 기록의 성별·장르 가산 줄에 붙일 표시 — 그 줄이 **사용자가 지금 말한 단서 그대로**가 아닐 때.

    - 재질문에서 남성→여성으로 정정하면 남성 기준 가산이 기록에 남는다 → "정정 전 값(남성) 기준".
      앞선 답변으로 계산된 가산이면 → "이전 답(남성) 기준".
    - 원문에서 읽은 값과 검색이 해석한 값이 달라 표가 단정하지 않은 줄 → "검색이 해석한 값(여성) 기준".
      표가 일치 여부를 말하지 않았는데 상세가 "말한 성별과 일치"로만 읽히면 반대로 말하는 셈이다.
    - 검색이 사용자가 말하지 않은 값("OST")으로 계산한 가산 → "검색이 해석한 값(OST) 기준".

    기록은 지우지 않는다(실제 계산이다). 현재 단서와 같은 값으로 계산된 줄에는 붙지 않는다 — 장르를 둘
    말했을 때 한 줄에만 연결된 가산도 여기에 해당한다. 비교값이 기록에 없으면 무엇을 기준으로 했는지 모르므로
    붙이지 않고, 지금 단서가 없으면(분석만 채운 값) 견줄 것이 없으므로 붙이지 않는다.
    """
    notes: List[AdjustmentNote] = []
    seen = set()
    for adj in explain.adjustments:
        if not adj.value:
            continue        # 무엇과 비교했는지 기록이 없으면 다르다고 말할 수 없다
        if adj.rule in _GENDER_RULES:
            slot = SLOT_GENDER
        elif adj.rule == "genre_match":
            slot = SLOT_GENRE
        elif adj.rule == "answer_match" and adj.slot in (SLOT_GENDER, SLOT_GENRE):
            slot = adj.slot
        else:
            continue
        current = ctx.current.get(slot) or []
        if not current:
            continue        # 지금 단서가 없으면 견줄 것이 없다
        if slot == SLOT_GENDER:
            same = any(value == adj.value for value, _ in current)
        else:
            same = any(_genre_consistent(value, adj.value) for value, _ in current)
        interpreted = any(origin == ORIGIN_ANALYSIS for _, origin in current)   # 값이 충돌해 검색값을 보인 줄
        if same and not interpreted:
            continue
        if same or not any(origin == ORIGIN_ANSWER for _, origin in current):
            note = f"검색이 해석한 값({adj.value}) 기준"
        elif adj.rule == "answer_match":
            note = f"이전 답({adj.value}) 기준"
        else:
            note = f"정정 전 값({adj.value}) 기준"
        key = (adj.rule, adj.slot, adj.value)
        if key not in seen:
            seen.add(key)
            notes.append(AdjustmentNote(adj.rule, adj.slot, adj.value, note))
    return notes


# ---------------------------------------------------------------------------
# 가사
# ---------------------------------------------------------------------------


def _lyric_ranking(explain: SongExplain) -> Tuple[List[RankingBasis], str]:
    """가사 표면 일치가 **순위를 만들었는가** — `schemas/explain.evidence_kind`와 같은 판정에 기록을 붙인다."""
    basis: List[RankingBasis] = []
    scored = False
    for path in explain.live_paths():
        if path.path == "lyrics_surface" and path.delta != 0.0:
            scored = True
            basis.append(RankingBasis(
                kind=BASIS_KIND_PATH, code=path.path, label=path.label, delta=path.delta,
            ))
    protected = False
    for rule in explain.order_rules:
        if rule.rule in LYRIC_ORDER_RULES:
            protected = True
            basis.append(RankingBasis(
                kind=BASIS_KIND_ORDER_RULE, code=rule.rule, label=RULE_LABELS.get(rule.rule, rule.rule),
            ))
    if scored and protected:
        return basis, BASIS_SCORED_AND_PROTECTED
    if scored:
        return basis, BASIS_SCORED
    if protected:
        return basis, BASIS_PROTECTED
    return basis, BASIS_SUPPLEMENTARY


def _lyric_match_row(clue_text: str, origin: str, match: Dict[str, Any], quote: Any, explain: SongExplain) -> ClueCheck:
    match_type = str(match.get("match_type") or "")
    phrase = str(match.get("matched_phrase") or "")
    quoted = f"“{quote.quote}”" if quote is not None and getattr(quote, "quote", "") else ""

    if match_type == "exact":
        status, kind = STATUS_MATCH, KIND_LYRIC_EXACT
        observed = f"가사에서 확인: {quoted}" if quoted else "가사와 표기 정규화 후 일치(원문 인용은 확보하지 못함)"
    elif match_type == "phonetic":
        # 음차는 **추정 표기**다. 기억한 구절이 그대로 있다고 말하지 않는다.
        status, kind = STATUS_PARTIAL, KIND_LYRIC_PHONETIC
        observed = f"들리는 대로 기억한 표기를 ‘{phrase}’로 추정해 가사와 표기 정규화 후 일치" if phrase \
            else "음차 추정 표기가 가사와 표기 정규화 후 일치"
        if quoted:
            observed += f" · 가사: {quoted}"
    elif match_type == "fuzzy":
        status, kind = STATUS_PARTIAL, KIND_LYRIC_FUZZY
        observed = "가사와 근사 일치(일부 글자 겹침 · 원문 인용 없음)"
    else:
        status, kind = STATUS_PARTIAL, KIND_LYRIC_FUZZY
        observed = "가사와 표기 정규화 후 일치"

    basis, basis_label = _lyric_ranking(explain)
    source = (
        f"{quote.source}@{quote.data_version}"
        if quote is not None and getattr(quote, "source", "") else "explain.lyric_match"
    )
    return ClueCheck(
        slot=SLOT_LYRICS, clue=clue_text, clue_origin=origin, observed=observed,
        status=status, status_label=STATUS_LABELS[status], source=source,
        ranking_basis=basis, basis_label=basis_label, detail_kind=kind,
    )


def _lyric_adjustment_row(clue_text: str, origin: str, adj: ScoreAdjustment) -> ClueCheck:
    """full_lyrics 경로가 아니라 발췌·요약(highlight/summary)에서 맞은 가산. 요약은 가사가 아니다."""
    phonetic = adj.rule.startswith("lyric_phonetic_in_")
    where = adj.rule.rsplit("_in_", 1)[-1]
    if where == "highlight":
        status = STATUS_PARTIAL if phonetic else STATUS_MATCH
        observed = ("음차 추정 표기가 가사 발췌와 표기 정규화 후 일치" if phonetic
                    else "가사 발췌와 표기 정규화 후 일치(원문 전체 인용은 없음)")
        source = "track.lyrics_highlight"
        kind = KIND_LYRIC_HIGHLIGHT_PHONETIC if phonetic else KIND_LYRIC_HIGHLIGHT
    elif where == "summary":
        status = STATUS_PARTIAL
        observed = ("음차 추정 표기가 가사 요약과 표기 정규화 후 일치(원문 확인 아님)" if phonetic
                    else "가사 요약과 표기 정규화 후 일치(원문 확인 아님)")
        source = "track.lyrics_summary"
        kind = KIND_LYRIC_SUMMARY_PHONETIC if phonetic else KIND_LYRIC_SUMMARY
    else:
        status = STATUS_PARTIAL
        observed = ("음차 추정 표기가 가사 발췌·요약 경계에서 표기 정규화 후 일치(원문 확인 아님)" if phonetic
                    else "가사 발췌·요약 경계에서 표기 정규화 후 일치(원문 확인 아님)")
        source = "track.lyrics_highlight+lyrics_summary"
        kind = KIND_LYRIC_COMBINED_PHONETIC if phonetic else KIND_LYRIC_COMBINED
    return ClueCheck(
        slot=SLOT_LYRICS, clue=clue_text, clue_origin=origin, observed=observed,
        status=status, status_label=STATUS_LABELS[status], source=source,
        ranking_basis=[_adj_basis(adj)], basis_label=BASIS_SCORED, detail_kind=kind,
    )


def _lyric_unverified_row(clue: Any, explain: SongExplain) -> ClueCheck:
    text = str(getattr(clue, "text", "") or "")
    if getattr(clue, "kind", "") == "semantic":
        # 뜻으로 기억한 가사 — 구절 검출을 주장하지 않는다. 의미 검색이 평가했다는 범위까지만.
        text_path = next((p for p in explain.live_paths() if p.path == "text_hybrid"), None)
        observed = "뜻으로 기억한 가사라 구절 일치는 확인하지 않음"
        basis: List[RankingBasis] = []
        basis_label = BASIS_NONE
        if text_path is not None:
            observed += f" · 텍스트 의미 검색이 이 곡을 {text_path.rank}위로 평가"
            basis = [RankingBasis(kind=BASIS_KIND_PATH, code=text_path.path, label=text_path.label, delta=text_path.delta)]
            basis_label = BASIS_SEMANTIC
        return ClueCheck(
            slot=SLOT_LYRICS, clue=text, clue_origin=ORIGIN_ANALYSIS, observed=observed,
            status=STATUS_UNVERIFIED, status_label=STATUS_LABELS[STATUS_UNVERIFIED],
            source="analysis.lyric_clues", ranking_basis=basis, basis_label=basis_label,
            detail_kind=KIND_LYRIC_SEMANTIC,
        )
    if not explain.lyric_match and any(
        p.path == "lyrics_surface" and p.delta != 0.0 for p in explain.live_paths()
    ):
        # 가사 구절 경로의 기여는 있는데 어느 구절이 맞았는지 기록이 없다. "확인하지 못함"도 "확인됨"도 말할 수 없다.
        return ClueCheck(
            slot=SLOT_LYRICS, clue=text, clue_origin=ORIGIN_ANALYSIS,
            observed="가사 구절 일치 경로에 기여가 기록됐으나 어느 구절인지는 기록에 없음",
            status=STATUS_UNVERIFIED, status_label=STATUS_LABELS[STATUS_UNVERIFIED],
            source="analysis.lyric_clues", detail_kind=KIND_LYRIC_UNATTRIBUTED,
        )
    return ClueCheck(
        slot=SLOT_LYRICS, clue=text, clue_origin=ORIGIN_ANALYSIS,
        observed="이 곡 가사에서 그 구절의 일치를 확인하지 못함",
        status=STATUS_UNVERIFIED, status_label=STATUS_LABELS[STATUS_UNVERIFIED],
        source="analysis.lyric_clues", detail_kind=KIND_LYRIC_UNMATCHED,
    )


def _check_lyrics(search_analysis: Any, explain: SongExplain, evidence: Sequence[Any]) -> List[ClueCheck]:
    rows: List[ClueCheck] = []
    clues = list(getattr(search_analysis, "lyric_clues", []) or [])
    match = dict(explain.lyric_match or {})
    matched_text = str(match.get("clue_text") or "")
    quote = next((item for item in evidence or () if getattr(item, "song_id", None) == explain.song_id), None)

    lyric_adjs = [adj for adj in explain.adjustments
                  if adj.rule.startswith("lyric_clue_in_") or adj.rule.startswith("lyric_phonetic_in_")]
    surface_clues = [c for c in clues if getattr(c, "kind", "") in ("verbatim", "partial", "phonetic")]

    def _adjustment_for(text: str) -> Optional[ScoreAdjustment]:
        for adj in lyric_adjs:
            if adj.value and _norm(adj.value) == _norm(text):
                return adj
        # value가 없는 옛 기록 — 표면 단서가 하나뿐일 때만 그 단서의 것으로 본다. 둘 이상이면 어느 단서인지 모른다.
        if len(surface_clues) == 1 and lyric_adjs and not any(a.value for a in lyric_adjs):
            return lyric_adjs[0]
        return None

    matched_row_done = False
    for clue in clues:
        text = str(getattr(clue, "text", "") or "")
        if not text:
            continue
        if match and matched_text and _norm(text) == _norm(matched_text):
            rows.append(_lyric_match_row(text, ORIGIN_ANALYSIS, match, quote, explain))
            matched_row_done = True
            continue
        adj = _adjustment_for(text) if getattr(clue, "kind", "") != "semantic" else None
        if adj is not None:
            rows.append(_lyric_adjustment_row(text, ORIGIN_ANALYSIS, adj))
            continue
        rows.append(_lyric_unverified_row(clue, explain))

    if match and matched_text and not matched_row_done:
        # 분석의 단서 목록과 기록의 구절이 어긋나도(변형 표기 등) 기록된 일치는 보인다.
        rows.append(_lyric_match_row(matched_text, ORIGIN_ANALYSIS, match, quote, explain))

    keyword_adj = next((adj for adj in explain.adjustments if adj.rule == "lyric_keyword_tokens"), None)
    keywords = [k for k in (getattr(search_analysis, "lyric_keywords", []) or []) if k]
    if keyword_adj is not None and keywords:
        rows.append(ClueCheck(
            slot=SLOT_LYRICS, clue="가사 키워드: " + ", ".join(keywords), clue_origin=ORIGIN_ANALYSIS,
            observed="가사 발췌·요약에서 키워드 일부 겹침" + (f"({keyword_adj.detail})" if keyword_adj.detail else ""),
            status=STATUS_PARTIAL, status_label=STATUS_LABELS[STATUS_PARTIAL],
            source="track.lyrics_highlight+lyrics_summary",
            ranking_basis=[_adj_basis(keyword_adj)], basis_label=BASIS_SCORED, detail_kind=KIND_LYRIC_KEYWORDS,
        ))
    return rows


# ---------------------------------------------------------------------------
# 발매 시기
# ---------------------------------------------------------------------------


def _check_era(analysis: Any, search_analysis: Any, track: Any, explain: SongExplain) -> Optional[ClueCheck]:
    year = _release_year(getattr(track, "release_date", None))
    stage = getattr(search_analysis, "life_stage", None)
    stage_text = str(getattr(stage, "text", "") or "") if getattr(search_analysis, "has_life_stage", False) else ""

    if not getattr(search_analysis, "has_release_era", False):
        if not stage_text:
            return None
        # 들었던 시기만 있고 연도로 바꾸지 못했다 — 없는 것은 곡 자료가 아니라 사용자의 기준 연도다.
        observed = (f"발매 {year}년 · 들었던 시기를 연도로 바꾸려면 출생 연도가 필요함" if year is not None
                    else "발매일 정보 없음 · 들었던 시기를 연도로 바꾸려면 출생 연도가 필요함")
        return ClueCheck(
            slot=SLOT_ERA, clue=f"{stage_text}에 듣던 노래", clue_origin=ORIGIN_ANALYSIS, observed=observed,
            status=STATUS_UNVERIFIED, status_label=STATUS_LABELS[STATUS_UNVERIFIED],
            source="track.release_date", detail_kind=KIND_ERA_NEEDS_BIRTH_YEAR,
        )

    era = search_analysis.release_era
    start, end = int(era.start_year), int(era.end_year)
    # 응답의 분석(원본)에는 시기가 없는데 검색용 분석에는 있다 → 출생 연도로 유도한 추정 창이다.
    estimated = not getattr(analysis, "has_release_era", False)
    if estimated:
        clue = f"{stage_text or '그 시절'}에 듣던 노래(출생 연도로 추정한 발매 시기 {start}~{end}년)"
        origin = ORIGIN_BIRTH_YEAR
        window_name = "추정 기간"
    else:
        clue = f"{start}년 발매" if start == end else f"{start}~{end}년 무렵 발매"
        origin = ORIGIN_ANALYSIS
        window_name = "기억한 기간"

    if year is None:
        status, observed, kind = STATUS_UNVERIFIED, "발매일 정보 없음", KIND_ERA_NO_DATE
    elif start <= year <= end:
        status, observed, kind = STATUS_MATCH, f"발매 {year}년 · {window_name} 안", KIND_ERA_IN_WINDOW
    else:
        distance = start - year if year < start else year - end
        if distance <= _ERA_NEAR_YEARS:
            status, observed, kind = STATUS_PARTIAL, f"발매 {year}년 · {window_name} 밖 {distance}년 차이(근접)", KIND_ERA_NEAR
        elif estimated and year < start:
            # 듣던 시기보다 먼저 나온 곡은 그 시절에 들을 수 있었다 — 모순이 아니다. 나중에 나온 곡만 모순이다.
            status, kind = STATUS_UNVERIFIED, KIND_ERA_EARLIER_THAN_ESTIMATE
            observed = f"발매 {year}년 · 추정 기간보다 {distance}년 먼저 발매(그 시절에 들었을 수 있음)"
        else:
            status, observed, kind = STATUS_MISMATCH, f"발매 {year}년 · {window_name} 밖 {distance}년 차이", KIND_ERA_LATER

    basis: List[RankingBasis] = []
    if status in (STATUS_MATCH, STATUS_PARTIAL):
        window = f"{start}~{end}"
        for adj in explain.adjustments:
            if adj.rule == "release_era" and (not adj.value or adj.value == window):
                basis.append(_adj_basis(adj))
    if not basis:
        basis_label = BASIS_NONE
    elif status == STATUS_PARTIAL:
        basis_label = BASIS_NEAR     # 범위 밖 근접 가산 — 범위 안의 일치로 쓰지 않는다
    else:
        basis_label = BASIS_SCORED

    return ClueCheck(
        slot=SLOT_ERA, clue=clue, clue_origin=origin, observed=observed,
        status=status, status_label=STATUS_LABELS[status], source="track.release_date",
        ranking_basis=basis, basis_label=basis_label, detail_kind=kind,
    )


# ---------------------------------------------------------------------------
# MV 장면
# ---------------------------------------------------------------------------


def _check_mv(query: str, rerank_skip_reason: str) -> Optional[ClueCheck]:
    phrases = mv_scene_phrases(query)
    if not phrases:
        return None
    basis: List[RankingBasis] = []
    basis_label = BASIS_NONE
    if rerank_skip_reason == RERANK_SKIP_MV_SCENE_UNVERIFIABLE:
        basis.append(RankingBasis(
            kind=BASIS_KIND_RERANK_SKIP, code=rerank_skip_reason,
            label="장면을 판단할 자료가 없어 리랭킹 모델을 부르지 않음",
        ))
        basis_label = BASIS_RERANK_SKIPPED
    return ClueCheck(
        slot=SLOT_MV,
        clue=" / ".join(phrases),
        clue_origin=ORIGIN_QUERY,
        # 자료 없음은 장면이 없다는 뜻이 아니다. 가사·상상 이미지·표지는 MV 장면의 자료가 아니다.
        observed="해당 MV 장면을 확인할 자료 없음(자료 없음은 장면이 없다는 뜻이 아님)",
        status=STATUS_UNVERIFIED,
        status_label=STATUS_LABELS[STATUS_UNVERIFIED],
        source="rerank_policy.MV_SCENE_EVIDENCE_FIELDS (없음)",
        ranking_basis=basis,
        basis_label=basis_label,
    )


# ---------------------------------------------------------------------------
# 요약 문장 — 최대 2문장: ① 확인된 것(또는 제한된 설명) ② 확인하지 못한 것
# ---------------------------------------------------------------------------

Clause = Tuple[str, str]   # (연결형 "…했고", 종결형 "…했어요")


def _short_name(check: ClueCheck) -> str:
    if check.slot == SLOT_GENDER:
        return check.clue                 # "남성 보컬"
    if check.slot == SLOT_GENRE:
        return check.clue                 # "인디"
    if check.slot == SLOT_ERA:
        return "추정한 시기" if check.clue_origin == ORIGIN_BIRTH_YEAR else "발매 시기"
    if check.slot == SLOT_LYRICS:
        return "기억한 구절"
    return check.clue


def _unverified_topic(check: ClueCheck) -> Optional[str]:
    """미확인 단서를 요약에서 부를 말(조사 없이). None이면 요약에 올리지 않는다(표에만 남는다)."""
    if check.detail_kind in (KIND_SLOT_CONFLICT, KIND_ERA_NEEDS_BIRTH_YEAR, KIND_ERA_EARLIER_THAN_ESTIMATE):
        return None                       # 없는 것은 곡 자료가 아니다 — "자료가 없다"고 말하지 않는다
    if check.slot == SLOT_MV:
        return "말씀하신 뮤직비디오 장면"
    if check.slot == SLOT_GENRE:
        return f"{check.clue} 여부"
    if check.slot == SLOT_GENDER:
        return "보컬 성별"
    if check.slot == SLOT_ERA:
        return "발매 시기"
    return None


def _join_names(checks: Sequence[ClueCheck]) -> str:
    return "·".join(dict.fromkeys(_short_name(c) for c in checks))


def _topic_particle(text: str) -> str:
    """주제 조사 은/는 — 마지막 글자의 받침으로 고른다. 한글이 아니면 '은(는)'."""
    if not text:
        return "은(는)"
    code = ord(text[-1]) - 0xAC00
    if 0 <= code < 11172:
        return "은" if code % 28 else "는"
    return "은(는)"


# 가사 행의 세부 종류 → 요약 문장 머리(반영됐을 때, 쓰이지 않았을 때). **강한 근거부터** 적는다 — 요약은 이 순서로
# 첫 줄을 고른다. "가사에서 확인"은 원문 경로 일치와 발췌 일치까지만이다. 요약(lyrics_summary)은 가사가 아니라
# 요약문이고, 음차는 추정 표기라 "기억하신 구절이"가 아니라 "비슷한 표기가"다.
_LYRIC_HEADS: Dict[str, Tuple[str, str]] = {
    KIND_LYRIC_EXACT: ("기억하신 구절이 가사에서 확인되어", "기억하신 구절이 가사에서 확인됐지만"),
    KIND_LYRIC_HIGHLIGHT: ("기억하신 구절이 가사 발췌에서 확인되어", "기억하신 구절이 가사 발췌에서 확인됐지만"),
    KIND_LYRIC_PHONETIC: ("기억하신 구절과 비슷한 표기가 가사에서 확인되어", "기억하신 구절과 비슷한 표기가 가사에서 확인됐지만"),
    KIND_LYRIC_HIGHLIGHT_PHONETIC: (
        "기억하신 구절과 비슷한 표기가 가사 발췌에서 확인되어", "기억하신 구절과 비슷한 표기가 가사 발췌에서 확인됐지만"),
    KIND_LYRIC_FUZZY: ("기억하신 구절과 가사가 근사하게 겹쳐", "기억하신 구절과 가사가 근사하게 겹치지만"),
    KIND_LYRIC_COMBINED: ("기억하신 구절이 가사 발췌·요약 문구와 겹쳐", "기억하신 구절이 가사 발췌·요약 문구와 겹치지만"),
    KIND_LYRIC_COMBINED_PHONETIC: (
        "기억하신 구절과 비슷한 표기가 가사 발췌·요약 문구와 겹쳐", "기억하신 구절과 비슷한 표기가 가사 발췌·요약 문구와 겹치지만"),
    KIND_LYRIC_SUMMARY: ("기억하신 구절이 가사 요약 문구와 겹쳐", "기억하신 구절이 가사 요약 문구와 겹치지만"),
    KIND_LYRIC_SUMMARY_PHONETIC: (
        "기억하신 구절과 비슷한 표기가 가사 요약 문구와 겹쳐", "기억하신 구절과 비슷한 표기가 가사 요약 문구와 겹치지만"),
}
_LYRIC_HIT_ORDER: Tuple[str, ...] = tuple(_LYRIC_HEADS)


def _strongest_lyric_hit(checks: Sequence[ClueCheck]) -> Optional[ClueCheck]:
    """가사 행 중 요약이 말할 한 줄 — 가장 강한 근거. 나머지는 표에 남는다."""
    hits = [c for c in checks if c.slot == SLOT_LYRICS and c.detail_kind in _LYRIC_HEADS]
    if not hits:
        return None
    return min(hits, key=lambda c: _LYRIC_HIT_ORDER.index(c.detail_kind))


def _lyric_clause(row: ClueCheck) -> Clause:
    head, unused = _LYRIC_HEADS[row.detail_kind]
    kinds = {b.kind for b in row.ranking_basis}
    scored = BASIS_KIND_PATH in kinds or BASIS_KIND_ADJUSTMENT in kinds
    protected = BASIS_KIND_ORDER_RULE in kinds
    if scored and protected:
        return (f"{head} 검색 점수에 반영되고 먼저 배치됐고", f"{head} 검색 점수에 반영되고 먼저 배치됐어요")
    if scored:
        return (f"{head} 검색 점수에 반영됐고", f"{head} 검색 점수에 반영됐어요")
    if protected:
        return (f"{head} 먼저 배치됐고", f"{head} 먼저 배치됐어요")
    return (f"{unused} 이 순위 계산에는 쓰이지 않았고", f"{unused} 이 순위 계산에는 쓰이지 않았어요")


def _positive_clauses(checks: Sequence[ClueCheck], *, facts_only: bool = False) -> List[Clause]:
    """확인된 것. `facts_only`면 곡 자료와 맞는다는 사실만 말하고 "검색 점수에 반영"이라는 말을 쓰지 않는다
    (제보로 들어온 곡 — 호출부가 순위 기록을 뗀 행을 넘긴다)."""
    clauses: List[Clause] = []
    lyric_hit = _strongest_lyric_hit(checks)
    if lyric_hit is not None:
        clauses.append(_lyric_clause(lyric_hit))
    elif not facts_only and any(c.detail_kind == KIND_LYRIC_UNATTRIBUTED for c in checks):
        clauses.append(("가사 구절 일치 기록이 검색 점수에 반영됐지만 어느 구절인지는 기록에 없고",
                        "가사 구절 일치 기록이 검색 점수에 반영됐지만 어느 구절인지는 기록에 없어요"))

    attrs = [c for c in checks if c.slot in (SLOT_GENDER, SLOT_GENRE, SLOT_ERA)]
    matched = [c for c in attrs if c.status == STATUS_MATCH]
    scored, plain = [c for c in matched if c.ranking_basis], [c for c in matched if not c.ranking_basis]
    partial = [c for c in attrs if c.status == STATUS_PARTIAL]
    mismatch = [c for c in attrs if c.status == STATUS_MISMATCH]
    if scored:
        n = _join_names(scored)
        clauses.append((f"{n} 단서는 곡 정보와 일치해 검색 점수에 반영됐고", f"{n} 단서는 곡 정보와 일치해 검색 점수에 반영됐어요"))
    if plain:
        n = _join_names(plain)
        clauses.append((f"{n} 단서는 곡 정보와 일치하고", f"{n} 단서는 곡 정보와 일치해요"))
    if partial:
        n = _join_names(partial)
        clauses.append((f"{n} 단서는 부분적으로만 일치하고", f"{n} 단서는 부분적으로만 일치해요"))
    if mismatch:
        n = _join_names(mismatch)
        clauses.append((f"{n} 단서는 등록 정보와 다르고", f"{n} 단서는 등록 정보와 달라요"))

    if lyric_hit is None and any(c.detail_kind == KIND_LYRIC_KEYWORDS for c in checks):
        # 키워드 겹침은 구절 확인이 아니다 — 따로, 겹침으로만 말한다.
        if facts_only:
            clauses.append(("가사 키워드 일부가 발췌·요약과 겹치고", "가사 키워드 일부가 발췌·요약과 겹쳐요"))
        else:
            clauses.append(("가사 키워드 일부가 발췌·요약과 겹쳐 검색 점수에 반영됐고",
                            "가사 키워드 일부가 발췌·요약과 겹쳐 검색 점수에 반영됐어요"))
    return clauses


def _negative_clauses(checks: Sequence[ClueCheck]) -> List[Clause]:
    clauses: List[Clause] = []
    topics = list(dict.fromkeys(
        t for t in (_unverified_topic(c) for c in checks if c.status == STATUS_UNVERIFIED) if t
    ))
    if topics:
        joined = ", ".join(topics)
        particle = _topic_particle(joined)
        clauses.append((f"{joined}{particle} 확인할 자료가 없고", f"{joined}{particle} 확인할 자료가 없어요"))
    if _strongest_lyric_hit(checks) is None and any(c.detail_kind == KIND_LYRIC_UNMATCHED for c in checks):
        clauses.append(("기억하신 구절은 이 곡 가사에서 확인하지 못했고", "기억하신 구절은 이 곡 가사에서 확인하지 못했어요"))
    return clauses


def _sentence(clauses: Sequence[Clause]) -> str:
    """절을 한 문장으로 — 마지막 절만 종결형이다."""
    if not clauses:
        return ""
    return ", ".join([conj for conj, _ in clauses[:-1]] + [clauses[-1][1]]) + "."


def _subject_particle(text: str) -> str:
    """주격 조사 이/가 — 마지막 글자의 받침으로 고른다. 한글이 아니면 '이(가)'."""
    if not text:
        return "이(가)"
    code = ord(text[-1]) - 0xAC00
    if 0 <= code < 11172:
        return "이" if code % 28 else "가"
    return "이(가)"


def _limited_sentence(explain: SongExplain, checks: Sequence[ClueCheck]) -> str:
    """강한 근거가 없을 때 — 유사도 기록의 범위 안에서만, 한 문장으로."""
    linked_rules = {b.code for c in checks for b in c.ranking_basis if b.kind == BASIS_KIND_ADJUSTMENT}
    other_adjs = sorted(
        (a for a in explain.adjustments if a.delta > 0 and a.rule not in linked_rules
         and a.rule not in _SLOT_RULES and not a.rule.startswith("lyric_")),
        key=lambda a: a.delta, reverse=True,
    )
    if other_adjs:
        top = other_adjs[0]
        label = RULE_LABELS.get(top.rule, top.rule)
        if top.rule in _FACT_RULES:
            return f"‘{label}’ 기록이 검색 점수에 반영됐어요."
        return f"‘{label}’ 기록이 검색 점수에 반영된 후보예요(말씀하신 단서를 직접 확인한 것은 아니에요)."
    strongest = max(explain.live_paths(), key=lambda p: p.delta, default=None)
    if strongest is not None and strongest.path == "lyrics_surface":
        # 가사 구절 경로는 유사도가 아니라 일치다. 어느 구절인지 모를 뿐이다(단서 행이 없는 경우).
        return "가사 구절 일치 기록이 검색 점수에 반영된 후보예요(어느 구절인지는 기록에 없어요)."
    if strongest is not None and strongest.path == "feedback_case":
        # 들어온 곡은 `_summary`가 먼저 처리한다. 여기는 그 표시 없이 기록만 받은 호출자를 위한 자리다.
        return FEEDBACK_SOURCE_SENTENCE
    if strongest is not None:
        name = _PATH_SEARCH_NAMES.get(strongest.path, strongest.label)
        return f"{name}{_subject_particle(name)} 비슷하게 평가한 후보예요(특정 구절이나 장면을 확인한 것은 아니에요)."
    return "일부 단서가 비슷한 후보예요(기록된 근거는 많지 않아요)."


def _summary(checks: Sequence[ClueCheck], explain: SongExplain) -> str:
    """최대 2문장. ① 확인된 것을 한 문장으로(없으면 제한된 설명) ② 확인하지 못한 것을 한 문장으로.

    **이전 사용자 제보로 후보에 들어온 곡은 출처 문장이 먼저 온다**(그 뒤에 위 두 문장, 최대 3문장). 이 곡은 기존
    검색만으로는 후보 밖이었다 — 단서 행이 있든 없든, 다른 경로 기여가 남아 있든 그 사실을 먼저 말한다.
    확인된 것은 곡 자료와 맞는다는 사실로만 적는다. 이 곡의 가산은 진입 판정에만 쓰였고 자리는 후보 끝이라
    "검색 점수에 반영됐다"고 하면 순위를 만든 것처럼 읽힌다(리뷰 FR-07).
    """
    second = _sentence(_negative_clauses(checks))
    if getattr(explain, "feedback_entrant", False):
        facts = _sentence(_positive_clauses([replace(c, ranking_basis=[]) for c in checks], facts_only=True))
        return " ".join(s for s in (FEEDBACK_SOURCE_SENTENCE, facts, second) if s)
    first = _sentence(_positive_clauses(checks)) or _limited_sentence(explain, checks)
    return " ".join(s for s in (first, second) if s)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------


def build_user_explanation(
    *,
    query: str,
    analysis: Any,
    search_analysis: Any,
    answers: Sequence[Any],
    track: Any,
    explain: SongExplain,
    evidence: Sequence[Any] = (),
    rerank_skip_reason: str = "",
) -> UserExplanation:
    """곡 하나의 사용자용 설명. 입력을 바꾸지 않고, 모델·DB를 부르지 않는다.

    - query: 사용자 원문(성별·장르·MV 장면 단서의 증거)
    - analysis: 응답에 실리는 분석(원본). 시기가 여기 없고 search_analysis에만 있으면 출생 연도 추정 창이다
    - search_analysis: 검색에 **실제로 쓴** 분석(출생 연도 창 반영). 가사 단서·시기 창·해석한 성별/장르는 여기서 읽는다
    - answers: 이번 요청의 재질문 답변. 유효한 최신 답이 원문보다 우선한다(건너뛴 답은 정정이 아니다)
    - track / explain / evidence: 결과 곡, 그 곡의 실행 기록, 같은 실행이 되짚은 가사 인용
    - rerank_skip_reason: 요청의 정책 생략 사유(MV 장면 자료 없음 등)
    """
    ctx = _Context()
    checks: List[ClueCheck] = []
    gender = _check_gender(query, search_analysis, answers, track, explain, ctx)
    if gender is not None:
        checks.append(gender)
    checks.extend(_check_genre(query, search_analysis, answers, track, explain, ctx))
    checks.extend(_check_lyrics(search_analysis, explain, evidence))
    era = _check_era(analysis, search_analysis, track, explain)
    if era is not None:
        checks.append(era)
    mv = _check_mv(query, rerank_skip_reason)
    if mv is not None:
        checks.append(mv)
    if getattr(explain, "feedback_entrant", False):
        # 제보로 들어온 곡: 가산·감점·경로 기록과 이어진 행은 "검색 점수에 반영"이 아니라 진입 판정에만 쓰였다고 적는다.
        # 연결된 기록(`ranking_basis`)은 지우지 않는다 — 그 가산이 있었던 것은 사실이고 상세 기록에도 남아 있다.
        scored_kinds = {BASIS_KIND_ADJUSTMENT, BASIS_KIND_PATH}
        checks = [
            replace(c, basis_label=BASIS_ENTRY_ONLY) if any(b.kind in scored_kinds for b in c.ranking_basis) else c
            for c in checks
        ]
    return UserExplanation(
        summary=_summary(checks, explain),
        clue_checks=checks,
        adjustment_notes=_adjustment_notes(explain, ctx),
    )
