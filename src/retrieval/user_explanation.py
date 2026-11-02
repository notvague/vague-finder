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
- **곡 자료(태그·요약)는 생성된 자료다.** 분석의 `korean_tags`·`lyric_semantic_query`·`performance_clues.sound_ensemble`을
  곡의 태그·요약(`_MATERIAL_FIELDS`)과 글자로 대조해 찾은 값을 **그대로 인용**한다 — "곡 자료에 이렇게 적혀 있음"까지만
  말하고 "가사에서 확인"이라고 쓰지 않는다. 찾지 못하면 미확인이다(다름·없음이 아니다). 태그별 가산 기록은 없으므로
  "검색 점수에 반영"이라고 쓰지 않는다 — 편성 단서는 묶음 가산(`performance_clues`) 기록이 있을 때만 그 기록을 잇는다.
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
from src.retrieval.modality_queries import has_explicit_visual_clue, text_outside_artwork
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

# 단서 종류(행의 slot). 성별·장르·가사·시기·MV 장면 + 곡 자료(태그·요약)와 대조하는 분석 단서.
SLOT_GENDER = "vocal_gender"
SLOT_GENRE = "genre"
SLOT_LYRICS = "lyrics"
SLOT_ERA = "release_era"
SLOT_MV = "mv_scene"
SLOT_LYRIC_MEANING = "lyric_meaning"   # 분석의 lyric_semantic_query — 뜻으로 기억한 가사 내용
SLOT_TAG = "tag"                       # 분석의 korean_tags 한 개(고음·술자리 같은 상황·소리 단서)
SLOT_SOUND = "sound"                   # 분석의 performance_clues.sound_ensemble 한 개(밴드사운드·어쿠스틱 …)
SLOT_REGISTERED = "registered"         # 등록 정보로 판정하는 분석 단서(걸그룹·남성솔로·드라마OST·댄스곡 …)
SLOT_NOT_FOUND = "material_not_found"  # 곡 자료에서 찾지 못한 단서를 곡마다 한 줄로 묶은 행
MATERIAL_SLOTS = (SLOT_LYRIC_MEANING, SLOT_TAG, SLOT_SOUND)

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
KIND_MATERIAL_TAG = "material_tag"           # 곡 태그 값과 같은 표현이 적혀 있음
KIND_MATERIAL_TEXT = "material_text"         # 곡 요약·설명 문장에 같은 낱말이 있음(태그 일치는 없음)
KIND_MATERIAL_NEAR = "material_near"         # 더 짧은 곡 태그가 단서 안에 들어 있음(비슷한 표현)
KIND_MATERIAL_NOT_FOUND = "material_not_found"   # 곡 자료에서 찾지 못함 — 미확인
KIND_REGISTERED = "registered_info"          # 태그가 아니라 등록 정보(형태·성별·장르·앨범명)로 판정

MATERIAL_NOT_FOUND_TEXT = "곡 태그·요약에서 같은 표현을 찾지 못함(없다는 뜻은 아님)"

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
# 편성 단서는 묶음으로 한 번 가산된다(`performance_clues`). 이 단서 하나가 얼마를 만들었는지는 기록에 없다.
BASIS_GROUP_SCORED = "검색 점수에 반영(보컬·편성 단서 묶음 기준 · 단서별 몫은 기록에 없음)"

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
    # 작품명은 말하지 않는다 — 추정한 이름을 사실처럼 보이게 하지 않는다
    "work_name_aux": "작품명 추정 보조 검색",
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
# 곡 자료(태그·요약)와 대조하는 분석 단서 — korean_tags · lyric_semantic_query · 편성 단서
# ---------------------------------------------------------------------------

# 대조하는 곡 자료. 모두 생성된 자료다 — 인용할 때 이름을 그대로 붙여 "어디에 적혀 있는가"를 밝힌다.
# (필드, 화면 이름, 목록인가)
_MATERIAL_FIELDS: Tuple[Tuple[str, str, bool], ...] = (
    ("sound_tags", "사운드 태그", True),
    ("place_activity_tags", "장소·활동 태그", True),
    ("time_weather_tags", "시간·날씨 태그", True),
    ("emotion_tags", "감정 태그", True),
    ("mood_tags", "무드 태그", True),
    ("vibe_tags", "느낌 태그", True),
    ("relation_context_tags", "관계·상황 태그", True),
    ("lyrics_summary", "가사 요약", False),
    ("search_style_summary", "분위기 설명", False),
)
_FIELDS_FOR_TAG = tuple(f for f, _, _ in _MATERIAL_FIELDS)
_FIELDS_FOR_SOUND = ("sound_tags", "search_style_summary")
_MEANING_TAG_FIELDS = ("relation_context_tags", "emotion_tags")
_MATERIAL_NAMES = {f: name for f, name, _ in _MATERIAL_FIELDS}
_MATERIAL_IS_LIST = {f: is_list for f, _, is_list in _MATERIAL_FIELDS}

# 한 행에 인용하는 수. 태그는 짧아 셋까지, 문장은 길어 하나까지다 — 나머지는 "외 N건"으로 센다.
_MAX_MATERIAL_QUOTES = 3
_MAX_SENTENCE_QUOTES = 1
_MAX_MEANING_QUOTES = 2
# 문장 인용의 최대 길이(앞뒤를 줄임표로 자른다).
_MAX_QUOTE_CHARS = 48

# 사용자가 그대로 말하지 않은 단서의 표시. 분석이 바꿔 쓰거나 덧붙인 값을 "기억한 단서"로 읽히지 않게 한다.
INTERPRETED_CLUE_SUFFIX = "(검색이 해석한 단서)"
INTERPRETED_MEANING_SUFFIX = "(검색이 해석한 내용)"

# 다른 행이 이미 다루거나 단서가 되지 못하는 낱말 — korean_tags에서 행을 만들지 않는다.
_TAG_SKIP_WORDS = frozenset(_norm(w) for w in (
    "노래", "곡", "가사", "가수", "음악", "멜로디", "뮤비", "뮤직비디오", "앨범", "보컬",
    "남자", "여자", "남성", "여성", "혼성", "남자가수", "여자가수", "남성가수", "여성가수",
    "남자보컬", "여자보컬", "남성보컬", "여성보컬", "남자노래", "여자노래", "남녀",
))
# 대조할 자료가 없는 표지 단서. 곡 태그·요약은 표지를 설명하는 자료가 아니어서 찾아봐야 늘 미확인이다.
_COVER_WORDS = frozenset(_norm(w) for w in (
    "앨범표지", "앨범커버", "앨범아트", "앨범자켓", "앨범재킷", "표지", "커버", "자켓", "재킷", "일러스트", "사진", "그림",
))
# 시기 단서는 시기 행이 다룬다. 태그·요약에서 찾을 값이 아니다.
_ERA_TAG_RE = re.compile(
    r"\d+(?:년대?|년생|살)?|(?:19|20)?\d{2}년대?(?:초반|중반|후반|초|중|말)?.{0,4}"
    r"|옛날|옛날노래|옛노래|예전|예전노래|요즘|요즘노래|최근|최신|최신곡|학창시절|중학교|고등학교|대학교"
)
# 한 글자 단서 뒤에 붙어도 같은 낱말로 보는 조사. "술"은 "술을"과 같지만 "기술"과는 다르다.
_ONE_CHAR_PARTICLES = frozenset("을를이가은는도에과와의로만")
# 한 글자 단서가 가리키는 값. 앞뒤 글자를 넓게 허용하면 "기술"·"예술"·깨진 태그 "술자격"까지 걸리므로 **목록에
# 있는 값만** 인정한다. 태그는 값 전체가 같아야 하고("술자리스태프"는 아니다), 문장은 낱말이 들어 있으면 된다.
_SHORT_CLUE_VALUES: Dict[str, Tuple[str, ...]] = {
    "술": ("술자리", "혼술", "술집", "음주", "소주", "맥주", "포장마차", "포차", "막걸리", "술잔"),
    "비": ("비오는날", "빗소리", "빗길", "장마", "봄비", "겨울비", "소낙비", "소나기"),
    "눈": ("첫눈", "함박눈", "눈오는날"),
    "밤": ("늦은밤", "여름밤", "가을밤", "겨울밤", "봄밤", "한밤중", "심야", "밤길", "밤거리", "밤산책", "밤하늘"),
    "봄": ("봄날", "봄바람", "봄밤", "봄비", "초봄", "늦봄", "봄나들이"),
    "꽃": ("벚꽃", "벚꽃길", "꽃잎", "꽃길"),
}
# 글자는 들어 있지만 뜻이 다른 표현. 단서마다 실제 곡 자료에서 확인한 것만 적는다.
_FALSE_FRIENDS: Dict[str, Tuple[str, ...]] = {
    "고음": ("복고음", "고음비트"),
    "슬픈": ("구슬픈",),
}
# ㅂ 불규칙 활용의 끝 글자. 단서와 곡 자료의 활용형이 다를 때만 쓴다.
_BIEUP_ENDINGS = ("워", "움", "운")
# 단서 안에 들어 있어도 근거가 되지 않는 흔한 태그("허스키보컬"의 "보컬", "살랑거리는"의 "거리", "무한도전"의 "도전").
_NEAR_GENERIC_TAGS = frozenset(_norm(w) for w in (
    "보컬", "노래", "멜로디", "사운드", "음색", "목소리", "리듬", "비트", "감성", "분위기", "거리", "도전",
))
# 뜻 문장에서 낱말 끝을 떼어 줄기를 만든다 — 긴 것부터. 형태소 분석기(모델)는 부르지 않는다.
_ENDINGS = (
    "이라는", "하는데", "했었던", "하는", "했던", "하던", "한다", "하고", "해서", "하며", "에서", "으로", "에게",
    "처럼", "까지", "부터", "라는", "이란", "였던", "었던", "았던", "인",
    "을", "를", "이", "가", "은", "는", "의", "에", "와", "과", "도", "로", "한", "던", "고", "며",
)
# 뜻 문장에서 곡을 가르지 못하는 낱말. 가사 요약의 23%에 나오는 "연인"처럼 흔한 말은 겹쳐도 근거가 아니다.
_MEANING_STOP = frozenset(_norm(w) for w in (
    "내용", "가사", "노래", "이야기", "얘기", "느낌", "그런", "있는", "같은", "어떤", "모습", "사람", "마음",
    "그리고", "하지만", "있다", "없다", "있었", "대한", "관한",
    "연인", "사랑", "상대", "자신", "그대", "남자", "여자",
))
# 등록 정보로 판정하는 형태 단서: (형태 후보, 성별, 아이돌처럼 등록 정보가 말해 주지 않는 부분이 있는가).
_FORM_TAGS: Dict[str, Tuple[Tuple[str, ...], str, bool]] = {
    "솔로": (("솔로",), "", False),
    "남성솔로": (("솔로",), "남성", False),
    "남자솔로": (("솔로",), "남성", False),
    "여성솔로": (("솔로",), "여성", False),
    "여자솔로": (("솔로",), "여성", False),
    "그룹": (("그룹",), "", False),
    "걸그룹": (("그룹", "걸그룹"), "여성", False),
    "여성그룹": (("그룹", "걸그룹"), "여성", False),
    "여자그룹": (("그룹", "걸그룹"), "여성", False),
    "보이그룹": (("그룹",), "남성", False),
    "남성그룹": (("그룹",), "남성", False),
    "남자그룹": (("그룹",), "남성", False),
    "혼성그룹": (("그룹",), "혼성", False),
    "아이돌": (("그룹", "걸그룹"), "", True),
    "남자아이돌": (("그룹",), "남성", True),
    "남성아이돌": (("그룹",), "남성", True),
    "여자아이돌": (("그룹", "걸그룹"), "여성", True),
    "여성아이돌": (("그룹", "걸그룹"), "여성", True),
    "밴드": (("밴드",), "", False),
    "듀오": (("듀오",), "", False),
    "남성듀오": (("듀오",), "남성", False),
    "남자듀오": (("듀오",), "남성", False),
    "여성듀오": (("듀오",), "여성", False),
    "여자듀오": (("듀오",), "여성", False),
    "남녀보컬": ((), "혼성", False),
    "혼성보컬": ((), "혼성", False),
    "남녀듀엣": ((), "혼성", False),
    "혼성듀엣": ((), "혼성", False),
}
# OST 단서 → 그 작품 종류를 뜻하는 등록 장르. 앨범·제목에 OST 표기만 있으면 작품 종류는 모르므로 부분 일치다.
_OST_TAGS: Dict[str, Tuple[str, ...]] = {
    "ost": ("국내드라마", "국외드라마", "국내영화", "국외영화", "애니메이션/웹툰", "게임", "국내뮤지컬"),
    "드라마ost": ("국내드라마", "국외드라마"),
    "영화ost": ("국내영화", "국외영화"),
    "애니ost": ("애니메이션/웹툰",),
    "애니메이션ost": ("애니메이션/웹툰",),
    "게임ost": ("게임",),
}
_GENRE_WORD_SUFFIXES = ("곡", "음악", "노래", "풍")
# 성별을 품은 태그("남성발라드"·"여자가수"). "남자친구"처럼 성별 낱말로 시작할 뿐인 태그는 읽지 않는다.
_GENDER_PREFIXES = (("남성", "남성"), ("남자", "남성"), ("여성", "여성"), ("여자", "여성"), ("남녀", "혼성"), ("혼성", "혼성"))
_GENDERED_TAIL_RE = re.compile(r"보컬|솔로|그룹|듀오|듀엣|밴드|아이돌|가수|노래|랩|래퍼|발라드|댄스|힙합|록|목소리")
# 장르 낱말 뒤에 붙어도 장르를 말하는 것으로 보는 꼬리("힙합그룹"·"발라드가수"). "일렉기타"·"재즈바"·"소울메이트"는 아니다.
_GENRE_PERSON_TAIL_RE = re.compile(r"보컬|가수|그룹|듀오|밴드|아이돌|래퍼")
_ANSWER_FORMS = frozenset({"솔로", "그룹", "듀오", "밴드"})


@dataclass
class _MaterialHit:
    field: str
    quote: str          # 곡 자료에 적힌 값 그대로(문장은 앞뒤를 줄일 수 있다)
    strength: str       # "tag" 태그에 같은 표현 · "text" 문장에 같은 낱말 · "near" 더 짧은 태그가 단서 안에


def _strip_ending(word: str) -> str:
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 2:
            return word[: -len(ending)]
    return word


def _tag_terms(tag: str) -> List[str]:
    """한 korean_tag로 찾을 표현 — 태그 그대로, 끝을 뗀 줄기(다를 때)."""
    norm = _norm(tag)
    terms = [norm]
    stem = _strip_ending(norm)
    if stem != norm and len(stem) >= 2:
        terms.append(stem)
    return terms


def _meaning_terms(text: str) -> List[str]:
    """뜻 문장의 내용 낱말 줄기. 두 글자 미만·흔한 말은 뺀다."""
    out: List[str] = []
    for word in re.findall(r"[가-힣A-Za-z0-9]+", str(text or "")):
        stem = _strip_ending(_norm(word))
        if len(stem) >= 2 and stem not in _MEANING_STOP and stem not in out:
            out.append(stem)
    return out


def _term_variants(term: str) -> List[str]:
    """ㅂ 불규칙 활용만 잇는다 — "그리워"·"그리움"·"그리운", "귀여운"·"귀여움". 어근을 넓게 잘라 "그리고"까지 걸리지 않게 한다."""
    if len(term) >= 3 and term[-1] in _BIEUP_ENDINGS:
        return [term, *(term[:-1] + ending for ending in _BIEUP_ENDINGS if ending != term[-1])]
    return [term]


def _one_char_in(term: str, text: str) -> bool:
    """한 글자 단서는 낱말 그대로(조사 하나까지)일 때만 맞다고 본다."""
    for word in re.findall(r"[가-힣A-Za-z0-9]+", text):
        word = word.lower()
        if word == term or (len(word) == 2 and word[0] == term and word[1] in _ONE_CHAR_PARTICLES):
            return True
    return False


def _without_false_friends(term: str, text: str) -> str:
    """단서와 글자만 같고 뜻이 다른 표현을 지운다. 같은 문장의 진짜 일치("복고음악과 시원한 고음")는 남는다."""
    for friend in _FALSE_FRIENDS.get(term, ()):
        text = text.replace(friend, " ")
    return text


def _term_in_tag(term: str, value: str) -> Optional[str]:
    """곡 태그 하나와 견준다. 'tag'(단서가 태그 안에) · 'near'(두 글자 이상의 태그가 단서 안에) · None."""
    v = _norm(value)
    if not term or not v:
        return None
    if len(term) == 1:
        return "tag" if _one_char_in(term, value) else None
    if term in _without_false_friends(term, v):
        return "tag"
    if len(v) >= 2 and v in term and v not in _NEAR_GENERIC_TAGS:
        return "near"
    # 활용 끝만 다른 경우("그리워"·"그리움")는 정해진 활용형만 인정한다. 앞글자만 같은 태그까지 잇으면
    # "드라마"가 "드라이브"에, "술자리"가 깨진 태그 "술자격"에 걸린다.
    if any(v.startswith(variant) for variant in _term_variants(term)[1:]):
        return "near"
    return None


def _term_in_text(term: str, text: str) -> bool:
    """문장에 낱말이 있는가. 두 글자 낱말은 띄어쓰기를 건너 맞추지 않는다 — "이 별이"는 "이별"이 아니다."""
    if not term or not text:
        return False
    if len(term) == 1:
        return _one_char_in(term, text)
    if len(term) == 2:
        return term in _without_false_friends(term, str(text).lower())
    return term in _without_false_friends(term, _norm(text))


def _sentences_of(text: Any) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。])\s+|\n+", str(text or "")) if part.strip()]


def _clip_quote(sentence: str, term: str) -> str:
    """문장 인용을 줄인다. 앞뒤만 자르고 글자를 바꾸지 않는다. 낱말 중간에서 끊지 않는다 — "떠난 연인"이 "…난 연인"이
    되면 다른 말로 읽힌다. 낱말이 원문 그대로 보이면 그 둘레를 남긴다."""
    if len(sentence) <= _MAX_QUOTE_CHARS:
        return sentence
    at = sentence.find(term) if term else -1
    if at < 0:
        at, term = 0, ""
    half = (_MAX_QUOTE_CHARS - len(term)) // 2
    start = max(0, at - half)
    end = min(len(sentence), start + _MAX_QUOTE_CHARS)
    start = max(0, end - _MAX_QUOTE_CHARS)
    if start > 0 and not sentence[start - 1].isspace():
        # 앞쪽이 낱말 중간이면 다음 낱말부터. 찾는 낱말을 넘어가면 그 낱말이 든 어절의 시작까지 물러난다.
        next_space = sentence.find(" ", start, at + 1)
        start = next_space + 1 if next_space >= 0 else sentence.rfind(" ", 0, at) + 1
    if end < len(sentence) and not sentence[end].isspace():
        prev_space = sentence.rfind(" ", at + len(term), end)
        if prev_space > 0:
            end = prev_space
        else:
            next_space = sentence.find(" ", end)
            end = next_space if next_space >= 0 else len(sentence)
    return ("…" if start > 0 else "") + sentence[start:end].strip() + ("…" if end < len(sentence) else "")


def _find_material(
    track: Any, terms: Sequence[str], fields: Sequence[str], exact_values: Sequence[str] = (),
) -> List[_MaterialHit]:
    """terms 가운데 하나라도 곡 자료에 적혀 있는 곳. 값마다 한 번만, 필드 순서대로.

    exact_values는 짧은 단서의 허용 목록이다. 태그는 값 전체가 같을 때만, 문장은 낱말이 있을 때 인정한다.
    """
    hits: List[_MaterialHit] = []
    seen = set()
    exact = {_norm(v) for v in exact_values}
    for field_name in fields:
        raw = getattr(track, field_name, None)
        if _MATERIAL_IS_LIST[field_name]:
            for value in list(raw or ()):
                value = str(value or "").strip()
                strength = "tag" if _norm(value) in exact else None
                for term in terms:
                    if strength == "tag":
                        break
                    found = _term_in_tag(term, value)
                    if found == "tag":
                        strength = "tag"
                    else:
                        strength = strength or found
                if strength and (field_name, value) not in seen:
                    seen.add((field_name, value))
                    hits.append(_MaterialHit(field_name, value, strength))
        else:
            for sentence in _sentences_of(raw):
                term = next((t for t in [*terms, *exact_values] if _term_in_text(_norm(t), sentence)), None)
                if term is not None and (field_name, sentence) not in seen:
                    seen.add((field_name, sentence))
                    hits.append(_MaterialHit(field_name, _clip_quote(sentence, term), "text"))
    return hits


def _quote_line(hits: Sequence[_MaterialHit], total: int) -> str:
    # 같은 곳에 적힌 값은 이름을 한 번만 쓴다 — "장소·활동 태그 ‘술집’·‘혼술’"
    by_field: Dict[str, List[str]] = {}
    for hit in hits:
        by_field.setdefault(hit.field, []).append(f"‘{hit.quote}’")
    shown = [f"{_MATERIAL_NAMES[field_name]} {'·'.join(quotes)}" for field_name, quotes in by_field.items()]
    more = f" 외 {total - len(hits)}건" if total > len(hits) else ""
    return "곡 자료에 이렇게 적혀 있음: " + " · ".join(shown) + more


def _material_observed(hits: Sequence[_MaterialHit]) -> str:
    """가장 분명한 근거만 인용한다. 태그에 같은 표현이 있으면 긴 문장을 덧붙이지 않아 표가 길어지지 않는다."""
    tags = [h for h in hits if h.strength == "tag"]
    if tags:
        return _quote_line(tags[:_MAX_MATERIAL_QUOTES], len(tags))
    texts = [h for h in hits if h.strength == "text"]
    if texts:
        return _quote_line(texts[:_MAX_SENTENCE_QUOTES], len(texts))
    return _quote_line(hits[:_MAX_MATERIAL_QUOTES], len(hits))


def _material_status(hits: Sequence[_MaterialHit]) -> Tuple[str, str]:
    """태그에 같은 표현 → 일치. 문장 속 낱말·더 짧은 태그 → 부분 일치(문맥·부정은 보지 않는다). 없음 → 미확인."""
    if any(h.strength == "tag" for h in hits):
        return STATUS_MATCH, KIND_MATERIAL_TAG
    if any(h.strength == "text" for h in hits):
        return STATUS_PARTIAL, KIND_MATERIAL_TEXT
    if hits:
        return STATUS_PARTIAL, KIND_MATERIAL_NEAR
    return STATUS_UNVERIFIED, KIND_MATERIAL_NOT_FOUND


def _said_in_query(tag: str, query: str) -> bool:
    """사용자가 이 표현을 글자 그대로(활용 끝 제외) 말했는가. 한 글자는 낱말로만 본다 — "비슷한"의 "비"가 아니다."""
    terms = _tag_terms(tag)
    if len(terms[0]) == 1:
        return _one_char_in(terms[0], str(query or ""))
    hay = _norm(query)
    return any(term in hay for term in terms)


def _material_row(
    slot: str, clue: str, track: Any, terms: Sequence[str], fields: Sequence[str], exact_values: Sequence[str] = (),
) -> ClueCheck:
    hits = _find_material(track, terms, fields, exact_values)
    status, kind = _material_status(hits)
    used_fields = list(dict.fromkeys(h.field for h in hits)) or list(fields)
    return ClueCheck(
        slot=slot, clue=clue, clue_origin=ORIGIN_ANALYSIS,
        observed=_material_observed(hits) if hits else MATERIAL_NOT_FOUND_TEXT,
        status=status, status_label=STATUS_LABELS[status],
        source="+".join(f"track.{f}" for f in used_fields),
        detail_kind=kind,
    )


def _meaning_row(meaning: str, query: str, track: Any) -> Optional[ClueCheck]:
    """뜻으로 기억한 가사 내용 한 줄. 내용 낱말이 **둘 이상** 겹칠 때만 부분 일치다(낱말이 하나뿐인 문장은 그 하나).

    낱말 하나만 봐서는 곡을 가르지 못한다(전체 곡의 66%가 부분 일치였다). 문맥·부정은 보지 않으므로 일치라고 하지
    않는다. 태그는 낱말을 세는 데만 쓰고, 인용은 가사 요약 문장을 우선한다 — 태그는 태그 행이 이미 보인다.
    """
    terms = _meaning_terms(meaning)
    if not terms:
        return None
    clue = meaning if _norm(meaning) in _norm(query) else f"{meaning}{INTERPRETED_MEANING_SUFFIX}"
    variants = {term: [_norm(v) for v in _term_variants(term)] for term in terms}
    found: set = set()
    tag_hits: List[_MaterialHit] = []
    for field_name in _MEANING_TAG_FIELDS:
        for value in list(getattr(track, field_name, None) or ()):
            value = str(value or "").strip()
            matched = [t for t in terms if any(_term_in_tag(v, value) for v in variants[t])]
            if matched:
                found.update(matched)
                tag_hits.append(_MaterialHit(field_name, value, "near"))
    scored: List[Tuple[int, int, str]] = []
    for order, sentence in enumerate(_sentences_of(getattr(track, "lyrics_summary", ""))):
        matched = [t for t in terms if any(_term_in_text(v, sentence) for v in variants[t])]
        if matched:
            found.update(matched)
            scored.append((-len(matched), order, sentence))
    if len(found) < min(2, len(terms)):
        return ClueCheck(
            slot=SLOT_LYRIC_MEANING, clue=clue, clue_origin=ORIGIN_ANALYSIS,
            observed="곡 태그·요약에서 이 내용과 충분히 겹치는 표현을 찾지 못함(없다는 뜻은 아님)",
            status=STATUS_UNVERIFIED, status_label=STATUS_LABELS[STATUS_UNVERIFIED],
            source="track.lyrics_summary", detail_kind=KIND_MATERIAL_NOT_FOUND,
        )
    scored.sort()
    quotes = [
        _MaterialHit("lyrics_summary", _clip_quote(sentence, next(
            (v for t in terms for v in variants[t] if v in sentence), "")), "text")
        for _, _, sentence in scored
    ]
    hits = quotes or tag_hits
    return ClueCheck(
        slot=SLOT_LYRIC_MEANING, clue=clue, clue_origin=ORIGIN_ANALYSIS,
        observed=_quote_line(hits[:_MAX_MEANING_QUOTES], len(hits)),
        status=STATUS_PARTIAL, status_label=STATUS_LABELS[STATUS_PARTIAL],
        source="+".join(dict.fromkeys(f"track.{h.field}" for h in hits)),
        detail_kind=KIND_MATERIAL_TEXT if quotes else KIND_MATERIAL_NEAR,
    )


def _registered_form(track: Any) -> Tuple[List[str], str]:
    """곡의 등록 형태·성별. 표기 그대로 돌려준다 — 화면에 그대로 인용한다."""
    forms = [str(v).strip() for v in (getattr(track, "artist_types", None) or ()) if str(v).strip()]
    forms = [v for v in forms if v.lower() != "unknown"]
    gender = str(getattr(track, "vocal_gender", "") or "").strip()
    return forms, gender if gender in _GENDERS else ""


def _form_row(tag: str, clue: str, search_analysis: Any, track: Any, explain: SongExplain) -> ClueCheck:
    """형태 단서(걸그룹·남성솔로·밴드 …)는 태그가 아니라 등록 정보로 판정한다. 다르다고 단정하지 않고 값을 보인다."""
    want_forms, want_gender, partial_only = _FORM_TAGS[_norm(tag)]
    forms, gender = _registered_form(track)
    registered = " · ".join([*forms, *([gender] if gender else [])])
    form_ok = not want_forms or any(f in forms for f in want_forms)
    gender_ok = not want_gender or gender == want_gender
    if not registered:
        status, observed = STATUS_UNVERIFIED, "등록된 형태·보컬 정보 없음"
    elif form_ok and gender_ok:
        status = STATUS_PARTIAL if partial_only else STATUS_MATCH
        observed = f"등록 정보: {registered}" + ("(아이돌 여부는 자료에 없음)" if partial_only else "")
    else:
        status, observed = STATUS_UNVERIFIED, f"등록 정보: {registered}({tag} 여부를 단정하지 않음)"
    basis: List[RankingBasis] = []
    if status != STATUS_UNVERIFIED:
        said = {str(v) for v in (getattr(getattr(search_analysis, "artist_type", None), "values", None) or ())}
        adj = next((a for a in explain.adjustments if a.rule == "artist_type"), None)
        if adj is not None and said & set(want_forms):
            basis = [_adj_basis(adj)]
    return ClueCheck(
        slot=SLOT_REGISTERED, clue=clue, clue_origin=ORIGIN_ANALYSIS, observed=observed,
        status=status, status_label=STATUS_LABELS[status], source="track.artist_types+track.vocal_gender",
        ranking_basis=basis, basis_label=BASIS_SCORED if basis else BASIS_NONE, detail_kind=KIND_REGISTERED,
    )


def _ost_row(tag: str, clue: str, track: Any) -> ClueCheck:
    genres = str(getattr(track, "genre", "") or "").strip()
    wanted = _OST_TAGS[_norm(tag)]
    album = str(getattr(track, "album", "") or "").strip()
    title = str(getattr(track, "title", "") or "").strip()
    marked = next((name for name in (album, title) if "ost" in name.lower()), "")
    if any(g in genres for g in wanted):
        status, observed = STATUS_MATCH, f"등록 장르: {genres}"
    elif marked:
        status, observed = STATUS_PARTIAL, f"등록 정보에 OST 표기: ‘{marked}’(어떤 작품인지는 자료에 없음)"
    else:
        status = STATUS_UNVERIFIED
        observed = f"등록 장르는 {genres or '정보 없음'}이며 {tag} 여부를 확인할 자료는 없음"
    return ClueCheck(
        slot=SLOT_REGISTERED, clue=clue, clue_origin=ORIGIN_ANALYSIS, observed=observed,
        status=status, status_label=STATUS_LABELS[status], source="track.genre+track.album+track.title",
        basis_label=BASIS_NONE, detail_kind=KIND_REGISTERED,
    )


def _genre_word(norm_tag: str) -> str:
    """"댄스곡"·"록음악"처럼 장르 낱말에 말꼬리가 붙은 태그의 장르 낱말. 아니면 빈 문자열.

    `_genre_family`는 글자 포함으로 본다("기록"의 "록")는 주의가 있어, 여기서는 표에 있는 낱말과 **전체가 같을 때만** 본다.
    """
    known = _known_genre_words()
    for suffix in _GENRE_WORD_SUFFIXES:
        if norm_tag.endswith(suffix) and norm_tag[: -len(suffix)] in known:
            return norm_tag[: -len(suffix)]
    return ""


def _known_genre_words() -> set:
    return {_norm(word) for word in _GENRE_CORPUS} | {_norm(alias) for alias in genre_aliases()}


def _tag_gender(norm_tag: str) -> str:
    """태그가 품은 성별. 없으면 빈 문자열."""
    form = _FORM_TAGS.get(norm_tag)
    if form is not None:
        return form[1]
    for prefix, gender in _GENDER_PREFIXES:
        if norm_tag.startswith(prefix) and _GENDERED_TAIL_RE.fullmatch(norm_tag[len(prefix):]):
            return gender
    return ""


def _tag_genre_word(norm_tag: str) -> str:
    """태그가 장르를 말할 때 그 낱말 — "댄스곡"의 댄스, "남성발라드"의 발라드, "힙합그룹"의 힙합.

    앞글자만 같은 태그("일렉기타"의 일렉, "재즈바"의 재즈, "인디언"의 인디)는 장르가 아니다. 끝이 장르 낱말이거나
    사람을 가리키는 꼬리가 붙을 때만 읽고, 두 글자 이상만 본다("기록"의 "록"이 아니다).
    """
    word = _genre_word(norm_tag)
    if word:
        return word
    for known in sorted(_known_genre_words(), key=len, reverse=True):
        if len(known) < 2 or known == norm_tag:
            continue
        if norm_tag.endswith(known):
            return known
        if norm_tag.startswith(known) and _GENRE_PERSON_TAIL_RE.fullmatch(norm_tag[len(known):]):
            return known
    return ""


def _stale_after_answers(norm_tag: str, answers: Sequence[Any]) -> bool:
    """재질문 답변으로 정정된 값을 품은 태그인가. 정정 전 값은 지금 단서가 아니므로 근거로 보이지 않는다.

    분석의 korean_tags는 원문 기준이라 답변을 반영하지 않는다. 그대로 두면 댄스→발라드로 정정해도 "댄스곡 단서는 곡
    정보와 일치해요"가 남는다. 정정 전 값으로 계산된 가산은 상세 기록에 남고 `_adjustment_notes`가 표시한다.
    """
    gender = _latest_answer(answers, "vocal_gender")
    if gender in _GENDERS:
        tag_gender = _tag_gender(norm_tag)
        if tag_gender and tag_gender != gender:
            return True
    genre = _latest_answer(answers, "genre")
    if genre:
        word = _tag_genre_word(norm_tag)
        if word and not _genre_consistent(word, genre):
            return True
    form = _latest_answer(answers, "type")
    entry = _FORM_TAGS.get(norm_tag)
    return bool(form in _ANSWER_FORMS and entry and entry[0] and form not in entry[0])


def _genre_tag_row(word: str, tag: str, clue: str, search_analysis: Any, track: Any, explain: SongExplain) -> ClueCheck:
    registered = str(getattr(track, "genre", "") or "").strip()
    if not registered:
        status, observed = STATUS_UNVERIFIED, "등록 장르 정보 없음"
    elif _genre_in(word, registered):
        status, observed = STATUS_MATCH, f"등록 장르: {registered}"
    else:
        status, observed = STATUS_UNVERIFIED, f"등록 장르는 {registered}이며 {tag} 여부를 확인할 자료는 없음"
    used = str(getattr(search_analysis, "genre", "") or "")
    basis = [
        _adj_basis(a) for a in explain.adjustments
        if a.rule == "genre_match" and status == STATUS_MATCH and _genre_consistent(word, a.value or used)
    ]
    return ClueCheck(
        slot=SLOT_REGISTERED, clue=clue, clue_origin=ORIGIN_ANALYSIS, observed=observed,
        status=status, status_label=STATUS_LABELS[status], source="track.genre",
        ranking_basis=basis, basis_label=BASIS_SCORED if basis else BASIS_NONE, detail_kind=KIND_REGISTERED,
    )


def _covered_words(query: str, search_analysis: Any) -> set:
    """다른 행이 이미 다루거나 대조할 자료가 없는 낱말 — 장르·성별·가사 구절/키워드·가수·제목·표지."""
    covered = set(_TAG_SKIP_WORDS) | set(_COVER_WORDS)
    for word, tokens in _GENRE_CORPUS.items():
        covered.add(_norm(word))
        for token in tokens:
            covered.add(_norm(token))
            covered.update(_norm(part) for part in token.split("/"))
    covered.update(_norm(alias) for alias in genre_aliases())
    covered.update(_norm(v) for v in known_genre_values(query))
    for name in ("genre", "artist_name", "song_title"):
        covered.add(_norm(getattr(search_analysis, name, "")))
    covered.update(_norm(k) for k in (getattr(search_analysis, "lyric_keywords", []) or []))
    covered.update(_norm(getattr(c, "text", "")) for c in (getattr(search_analysis, "lyric_clues", []) or []))
    covered.discard("")
    return covered


def _check_material(
    query: str,
    search_analysis: Any,
    track: Any,
    explain: SongExplain,
    existing: Sequence[ClueCheck] = (),
    answers: Sequence[Any] = (),
) -> List[ClueCheck]:
    """분석이 뽑은 상황·소리·가사 내용 단서를 곡 자료와 대조한 행.

    - 등록 정보로 판정할 수 있는 단서(형태·OST·장르 변형)는 등록 정보로 본다 — 태그에 없다고 미확인으로 두면 걸그룹 곡에
      "걸그룹을 찾지 못함"이라고 말하게 된다.
    - 표지·시기 단서는 태그·요약으로 판정할 수 없으니 행을 만들지 않는다. 표지 묘사 질의에서는 노래·소리를 말하는 절에
      글자 그대로 있는 태그만 쓴다 — "표지에 밤하늘에 별"의 별을 가사 요약의 별로 확인했다고 하면 다른 자료로 근거를
      만드는 셈이다. 어느 절에서 왔는지 모르는 태그(원문에 그대로 없는 값)도 쓰지 않는다.
    - 재질문 답변으로 정정된 값을 품은 태그는 쓰지 않는다(`_stale_after_answers`).
    - 못 찾은 단서는 곡마다 한 줄로 묶는다. 사용자가 말한 것은 숨기지 않고(숨기면 확인으로 읽힌다), 말하지 않고 분석이
      덧붙인 것은 빼는 대신 찾았을 때 "검색이 해석한 단서"로 표시한다.
    - 순위 기여: korean_tags·가사 뜻은 텍스트 검색 질의에 들어가지만 **태그별 기록은 없다** — "순위 반영 기록 없음".
      편성 단서는 묶음 가산(`performance_clues`) 기록이 있고 이 곡 자료와 맞을 때만 그 기록을 잇는다.
    """
    rows: List[ClueCheck] = []
    missing: List[str] = []

    meaning = str(getattr(search_analysis, "lyric_semantic_query", "") or "").strip()
    if meaning:
        row = _meaning_row(meaning, query, track)
        if row is not None:
            rows.append(row)

    def label(tag: str) -> Tuple[str, bool]:
        said = _said_in_query(tag, query)
        return (tag if said else f"{tag}{INTERPRETED_CLUE_SUFFIX}"), said

    covered = _covered_words(query, search_analysis)
    is_visual = bool(getattr(search_analysis, "has_visual_clue", False)) or has_explicit_visual_clue(query)
    outside_artwork = text_outside_artwork(query) if is_visual else None
    seen = set()
    tag_rows: Dict[str, ClueCheck] = {}
    for tag in getattr(search_analysis, "korean_tags", []) or []:
        tag = str(tag or "").strip()
        norm = _norm(tag)
        if not norm or norm in covered or norm in seen or _ERA_TAG_RE.fullmatch(norm):
            continue
        if outside_artwork is not None and not _said_in_query(tag, outside_artwork):
            continue
        if _stale_after_answers(norm, answers):
            continue
        seen.add(norm)
        clue, said = label(tag)
        if norm in _FORM_TAGS:
            rows.append(_form_row(tag, clue, search_analysis, track, explain))
            continue
        if norm in _OST_TAGS:
            rows.append(_ost_row(tag, clue, track))
            continue
        word = _genre_word(norm)
        if word:
            has_row = any(c.slot == SLOT_GENRE and _genre_consistent(word, c.clue) for c in existing)
            if not has_row:
                rows.append(_genre_tag_row(word, tag, clue, search_analysis, track, explain))
            continue
        row = _material_row(
            SLOT_TAG, clue, track, _tag_terms(tag), _FIELDS_FOR_TAG, _SHORT_CLUE_VALUES.get(norm, ()),
        )
        if row.status != STATUS_UNVERIFIED:
            rows.append(row)
            tag_rows[norm] = row
        elif said:
            missing.append(tag)

    clue_obj = getattr(search_analysis, "performance_clues", None)
    group_adj = next((a for a in explain.adjustments if a.rule == "performance_clues"), None)
    for sound in list(getattr(clue_obj, "sound_ensemble", None) or ()):
        sound = str(sound or "").strip()
        norm = _norm(sound)
        # 모델이 편성 칸에 장르를 넣기도 한다("댄스"·"발라드") — 장르 행과 겹치면 같은 단서가 두 번 나온다
        if not norm or norm in covered or _genre_word(norm):
            continue
        # 표지에 그려진 악기가 편성 단서로 읽힌 경우("표지에 남자가 있고 통기타를 들고 있었어")도 근거로 쓰지 않는다
        if outside_artwork is not None and not _said_in_query(sound, outside_artwork):
            continue
        if norm in seen:
            # 분석이 같은 값을 태그와 편성 양쪽에 넣은 경우(프롬프트 예시에도 있다). 행은 하나지만 편성 가산 기록은 잇는다.
            duplicate = tag_rows.get(norm)
            if duplicate is not None and group_adj is not None and _find_material(track, _tag_terms(sound), _FIELDS_FOR_SOUND):
                duplicate.ranking_basis = [_adj_basis(group_adj)]
                duplicate.basis_label = BASIS_GROUP_SCORED
            continue
        seen.add(norm)
        clue, said = label(sound)
        row = _material_row(SLOT_SOUND, clue, track, _tag_terms(sound), _FIELDS_FOR_SOUND)
        if row.status != STATUS_UNVERIFIED:
            if group_adj is not None:
                row.ranking_basis = [_adj_basis(group_adj)]
                row.basis_label = BASIS_GROUP_SCORED
            rows.append(row)
        elif said:
            missing.append(sound)

    if missing:
        rows.append(ClueCheck(
            slot=SLOT_NOT_FOUND, clue=", ".join(missing), clue_origin=ORIGIN_ANALYSIS,
            observed=MATERIAL_NOT_FOUND_TEXT,
            status=STATUS_UNVERIFIED, status_label=STATUS_LABELS[STATUS_UNVERIFIED],
            source="+".join(f"track.{f}" for f in _FIELDS_FOR_TAG), detail_kind=KIND_MATERIAL_NOT_FOUND,
        ))
    return rows


def _merge_semantic_lyric_rows(checks: List[ClueCheck], meaning_text: str) -> List[ClueCheck]:
    """분석이 같은 문장을 뜻 가사 단서(lyric_clues의 semantic)와 lyric_semantic_query 양쪽에 넣으면 한 줄로 합친다.

    합친 줄은 곡 자료 대조 결과에 의미 검색의 기록(텍스트 의미 검색 N위 · 유사도로 반영)을 그대로 잇는다.
    """
    meaning = next((c for c in checks if c.slot == SLOT_LYRIC_MEANING), None)
    if meaning is None:
        return checks
    twins = [c for c in checks if c.slot == SLOT_LYRICS and c.detail_kind == KIND_LYRIC_SEMANTIC
             and _norm(c.clue) == _norm(meaning_text)]
    if not twins:
        return checks
    twin = twins[0]
    note = twin.observed.split(" · ", 1)[1] if " · " in twin.observed else ""
    if note:
        meaning.observed += f" · {note}"
    if twin.ranking_basis:
        meaning.ranking_basis = list(twin.ranking_basis)
        meaning.basis_label = twin.basis_label
    return [c for c in checks if c is not twin]


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
    if check.slot == SLOT_LYRIC_MEANING:
        return "가사 내용"
    return check.clue.removesuffix(INTERPRETED_CLUE_SUFFIX)


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


def _positive_clauses(
    checks: Sequence[ClueCheck], *, facts_only: bool = False, hit_counts: Optional[Dict[Tuple[str, str], int]] = None,
) -> List[Clause]:
    """확인된 것. `facts_only`면 곡 자료와 맞는다는 사실만 말하고 "검색 점수에 반영"이라는 말을 쓰지 않는다
    (제보로 들어온 곡 — 호출부가 순위 기록을 뗀 행을 넘긴다)."""
    clauses: List[Clause] = []
    lyric_hit = _strongest_lyric_hit(checks)
    if lyric_hit is not None:
        clauses.append(_lyric_clause(lyric_hit))
    elif not facts_only and any(c.detail_kind == KIND_LYRIC_UNATTRIBUTED for c in checks):
        clauses.append(("가사 구절 일치 기록이 검색 점수에 반영됐지만 어느 구절인지는 기록에 없고",
                        "가사 구절 일치 기록이 검색 점수에 반영됐지만 어느 구절인지는 기록에 없어요"))

    # 이름은 셋까지만 적으므로 순서가 곧 내용이다. 결과 곡 중 **적은 곡만 가진 단서**를 먼저 부른다 — 모든 곡에 있는
    # 이별·그리움이 칸을 차지하면 곡을 가르는 고음·술이 밀린다. 같으면 일치(태그)를 겹침(문장)보다 먼저 부른다.
    counts = hit_counts or {}
    material = sorted(
        (c for c in checks if c.slot in MATERIAL_SLOTS and c.status in (STATUS_MATCH, STATUS_PARTIAL)),
        key=lambda c: (counts.get((c.slot, c.clue), 0), c.status != STATUS_MATCH),
    )
    if material:
        # 곡마다 다른 점이 여기서 갈린다 — 성별·장르 같은 공통 단서보다 앞에 둔다. 생성된 곡 자료와 글자가 겹친다는
        # 사실까지만 말한다. 단서별 가산 기록은 없으므로 "반영"이라고 쓰지 않는다(편성 묶음 기록은 표에 남는다).
        names = list(dict.fromkeys(_short_name(c) for c in material))
        n = "·".join(names[:_MAX_MATERIAL_QUOTES]) + (" 등" if len(names) > _MAX_MATERIAL_QUOTES else "")
        clauses.append((f"{n} 단서는 곡 자료(태그·요약)에 적힌 표현과 겹치고",
                        f"{n} 단서는 곡 자료(태그·요약)에 적힌 표현과 겹쳐요"))

    attrs = [c for c in checks if c.slot in (SLOT_GENDER, SLOT_GENRE, SLOT_ERA, SLOT_REGISTERED)]
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


def _summary(
    checks: Sequence[ClueCheck], explain: SongExplain, *, hit_counts: Optional[Dict[Tuple[str, str], int]] = None,
) -> str:
    """최대 2문장. ① 확인된 것을 한 문장으로(없으면 제한된 설명) ② 확인하지 못한 것을 한 문장으로.

    **이전 사용자 제보로 후보에 들어온 곡은 출처 문장이 먼저 온다**(그 뒤에 위 두 문장, 최대 3문장). 이 곡은 기존
    검색만으로는 후보 밖이었다 — 단서 행이 있든 없든, 다른 경로 기여가 남아 있든 그 사실을 먼저 말한다.
    확인된 것은 곡 자료와 맞는다는 사실로만 적는다. 이 곡의 가산은 진입 판정에만 쓰였고 자리는 후보 끝이라
    "검색 점수에 반영됐다"고 하면 순위를 만든 것처럼 읽힌다(리뷰 FR-07).
    """
    second = _sentence(_negative_clauses(checks))
    if getattr(explain, "feedback_entrant", False):
        facts = _sentence(_positive_clauses(
            [replace(c, ranking_basis=[]) for c in checks], facts_only=True, hit_counts=hit_counts,
        ))
        return " ".join(s for s in (FEEDBACK_SOURCE_SENTENCE, facts, second) if s)
    first = _sentence(_positive_clauses(checks, hit_counts=hit_counts)) or _limited_sentence(explain, checks)
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
    - 곡 자료 대조 행(`_check_material`): search_analysis의 korean_tags·lyric_semantic_query·편성 단서를 track의
      태그·요약과 글자로 견준다. 문장은 템플릿이고 찾은 값을 그대로 인용한다
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
    checks.extend(_check_material(query, search_analysis, track, explain, checks, answers))
    checks = _merge_semantic_lyric_rows(checks, str(getattr(search_analysis, "lyric_semantic_query", "") or "").strip())
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


def distinguish_summaries(
    items: Sequence[Tuple[UserExplanation, SongExplain]],
) -> List[UserExplanation]:
    """한 응답의 결과 곡을 함께 보고 요약 문장을 다시 쓴다. 곡마다 다른 단서가 요약의 이름 칸에 먼저 오게 한다.

    `build_user_explanation`은 곡 하나만 보므로 어느 단서가 곡을 가르는지 모른다. 단서 표·상태·순위 기록은 그대로이고
    요약의 순서만 바뀐다. 모델·DB를 부르지 않는다. 곡이 둘 미만이면 비교할 것이 없어 그대로 돌려준다.
    """
    if len(items) < 2:
        return [ux for ux, _ in items]
    hit_counts: Dict[Tuple[str, str], int] = {}
    for ux, _ in items:
        for key in {(c.slot, c.clue) for c in ux.clue_checks if c.status in (STATUS_MATCH, STATUS_PARTIAL)}:
            hit_counts[key] = hit_counts.get(key, 0) + 1
    return [replace(ux, summary=_summary(ux.clue_checks, explain, hit_counts=hit_counts)) for ux, explain in items]
