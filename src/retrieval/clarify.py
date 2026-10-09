"""
재질문 선택 (단계 4).

무엇을 물을지 고르고, 받은 답변을 분석에 병합한다.

## 왜 이렇게 설계했는가 — 전부 단계 3 측정에서 나왔다

`results_clarify_v01` (dev 53건 / 개입 9건, Hit@10):

    reject_only           3/9    ← 질문 없이 거절만. 넘어야 할 기준선
    oracle:vocal_gender   7/9
    oracle:genre          5/9
    oracle:release_era    4/9
    oracle:type           1/9    ← 맞게 답해도 Reject-only보다 나쁘다
    oracle:best           8/9    ← 슬롯을 완벽히 고르는 천장

**1. 물을 수 있는 슬롯은 `vocal_gender`와 `genre`뿐이다.**

`type`은 맞게 답해도 해롭다. `search_router.py`의 `use_deep_audio_fusion`
조건에 `not has_artist_type_clue`가 있어서, 답변으로 `artist_type`이 채워지면
deep audio fusion 경로가 통째로 꺼진다. q115가 그 경로에 의존하는 질의였고
4위에서 Top-10 밖으로 떨어졌다.

`release_era`는 틀리게 답하면 정답이 후보 풀에서 사라진다. 오답 시 후보@30
이탈률이 `type` 4/9, `release_era` 3/9인 반면 `vocal_gender`와 `genre`는
0/9다. 후보 풀에서 사라지면 이후 어떤 거절·질문으로도 회수할 수 없으므로,
"틀려도 안전한" 슬롯만 쓴다.

**2. 승격 시뮬레이션은 만들지 않았다.**

계획서는 각 후보를 가상 정답으로 놓고 승격을 시뮬레이션하는 `score_question`을
설계했다. 그런데 실측 헤드룸이 **1건(7→8)** 이다. 항상 성별만 물어도 7/9이고
완벽한 선택이 8/9다. 그 1건을 위해 후보마다 재검색을 시뮬레이션하는 비용은
정당화되지 않는다. 대신 "이 슬롯이 남은 후보를 실제로 가르는가"만 본다.

**3. 가르지 못하면 묻지 않는다.**

남은 후보가 전부 남성이면 성별을 물어도 아무것도 좁혀지지 않는다.
q203이 그런 경우다 — 성별을 정확히 답해도 순위가 그대로였고(후보 14위 유지),
장르를 답했을 때만 5위로 올라왔다.
`pick_question()`은 이럴 때 `None`을 돌려주고, 그러면 Reject-only로 동작한다.

## v02 측정에서 다시 고친 것 (`results_clarify_v02`)

첫 구현은 점수(1 - Σp² × 슬롯 가중치)로 슬롯을 골랐고 **5/9에 그쳤다.**
항상 성별만 묻는 것(7/9)보다 못했다. 원인 둘을 고쳤다.

**"이미 말한 슬롯은 묻지 않는다"가 3건을 날렸다.** c701·c705·q101 전부 질의에
성별이 들어 있어 건너뛰었는데, 셋 다 그 성별이 **틀렸다**. 사람이 잘못 기억하는
것이 이 프로젝트의 전제이므로 이미 말한 슬롯도 다시 확인한다.
자세한 근거는 `stated_in_query()` 참조.

**점수 방식이 장르를 3/4번 골랐다.** 장르는 값이 4~6종이라 2~3종인 성별보다
점수가 높게 나오지만 실제 회수 효과는 반대다. 점수 비교를 버리고
`ALLOWED_SLOTS` 순서대로 "가를 수 있는 첫 슬롯"을 고른다.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Dict, Iterable, Optional, Sequence

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer, ClarifyOption, ClarifyQuestion, MatchingTrack

logger = logging.getLogger(__name__)

# 사용자가 직접 고른 값이라 모델 추론보다 높게 잡는다.
ANSWER_CONFIDENCE = 0.8

# 물어도 되는 슬롯을 **우선순위 순서대로** 적는다. 단계 3에서 "틀려도 후보 풀을
# 지키는" 것만 남겼고, 순서는 실측 회수율이다 — 성별 7/9, 장르 5/9.
#
# 점수로 고르지 않고 순서로 고르는 이유: v02 측정에서 점수 방식(1 - Σp² × 가중치)이
# 장르를 3/4번 골라 5/9에 그쳤다. 장르는 값이 4~6종이라 2~3종인 성별보다 점수가
# 높게 나오는데, 실제 회수 효과는 반대다. 표본 9건에서 순서 규칙이 점수 조율보다
# 안전하다. 장르가 이기는 것은 성별이 후보를 아예 가르지 못할 때뿐이었고(q203),
# 그 경우는 성별이 skip 조건에 걸려 자연히 장르로 넘어간다.
ALLOWED_SLOTS: Sequence[str] = ("vocal_gender", "genre")

QUESTION_TEXT: Dict[str, str] = {
    "vocal_gender": "부른 사람 목소리는 어느 쪽이었나요?",
    "genre": "어떤 장르에 가까웠나요?",
}

# 남은 후보 중 이 비율 이상이 값을 갖고 있어야 묻는다. 메타데이터가 비어 있는
# 후보가 많으면 답변을 받아도 대부분을 판별할 수 없다.
MIN_COVERAGE = 0.6

# 이보다 덜 갈리면 묻지 않는다. 0.15는 대략 90:10보다 치우친 분포를 걸러낸다.
MIN_DISCRIMINATION = 0.15

# 코퍼스의 artist type은 자유 문자열이라 4종으로 접어야 한다.
# search_router에서 옮겨 왔다 — 답변 일치 판정과 검색 부스팅이 같은 기준을 써야 한다.
ARTIST_TYPE_ALIASES = {
    "솔로": "솔로",
    "솔로가수": "솔로",
    "그룹": "그룹",
    "아이돌그룹": "그룹",
    "혼성그룹": "그룹",
    "듀오": "듀오",
    "듀엣": "듀오",
    "밴드": "밴드",
    "인디밴드": "밴드",
}


def canonical_artist_types(values: Iterable[str]) -> set:
    """자유 문자열 type 목록을 솔로/그룹/듀오/밴드로 정규화한다."""
    normalized: set = set()
    for value in values:
        compact = "".join(str(value).split())
        for alias, canonical in ARTIST_TYPE_ALIASES.items():
            if alias in compact:
                normalized.add(canonical)
    return normalized


_VOCAL_GENDERS = ("남성", "여성", "혼성")
_ARTIST_TYPES = ("솔로", "그룹", "듀오", "밴드")
_DECADE_RE = re.compile(r"(19|20)\d{2}")


# ---------------------------------------------------------------------------
# 후보에서 슬롯 값 읽기
# ---------------------------------------------------------------------------

def slot_value(track: MatchingTrack, slot: str) -> Optional[str]:
    """후보 한 곡의 슬롯 값. 값이 없거나 선택지로 쓸 수 없으면 None."""
    if slot == "vocal_gender":
        value = (track.vocal_gender or "").strip()
        return value if value in _VOCAL_GENDERS else None
    if slot == "genre":
        # 코퍼스 장르는 "발라드, 국내드라마"처럼 쉼표로 이어 붙는 경우가 있다.
        # '랩/힙합'은 슬래시가 장르명의 일부라 쉼표로만 나눈다.
        first = (track.genre or "").split(",")[0].strip()
        return first or None
    return None


def value_counts(candidates: Iterable[MatchingTrack], slot: str) -> Counter:
    counts: Counter = Counter()
    for track in candidates:
        value = slot_value(track, slot)
        if value:
            counts[value] += 1
    return counts


def discrimination(counts: Counter) -> float:
    """정답이 어느 값이든, 답변 하나로 남은 후보가 얼마나 줄어드는지의 기댓값.

    1 - Σ p² 다. 전부 같은 값이면 0(아무것도 못 가른다), 고르게 갈릴수록 1에
    가깝다. 엔트로피 대신 이걸 쓰는 이유는 값 종류가 많다는 이유만으로 점수가
    올라가는 것을 피하기 위해서다 — 계획서도 "엔트로피만으로는 장르만 묻게
    된다"고 지적했고, 실측에서도 장르는 성별보다 약했다.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return 1.0 - sum((n / total) ** 2 for n in counts.values())


def coverage(counts: Counter, candidate_count: int) -> float:
    if candidate_count <= 0:
        return 0.0
    return sum(counts.values()) / candidate_count


# ---------------------------------------------------------------------------
# 질문 선택
# ---------------------------------------------------------------------------

def stated_in_query(analysis: QueryAnalysis, slot: str) -> bool:
    """사용자가 최초 질의에서 이미 말한 슬롯인지.

    **이걸 이유로 건너뛰지 않는다.** 계획서는 "이미 명시한 슬롯"을 질문하지 않을
    조건 맨 앞에 뒀고 당연해 보이지만, v02 측정에서 그 조건이 9건 중 3건을
    통째로 날렸다(c701·c705·q101 — 전부 성별을 물었으면 회수됐다).

    사람이 잘못 기억하는 것이 이 프로젝트의 전제다. 이미 말한 속성을 다시
    확인하는 것이 바로 그 오기억을 잡는 방법이고, 비대칭이 유리하다 —
    맞게 답했으면 같은 값이 다시 들어와 아무것도 바뀌지 않고, 틀렸을 때만 크게
    이득이다. 화면에서는 "혹시 남자 목소리가 맞나요?" 같은 재확인으로 보이면 된다.

    판정 자체는 남겨 둔다. 질문 문구를 확인형으로 바꾸는 데 쓸 수 있다.
    """
    if slot == "vocal_gender":
        return analysis.vocal_gender is not None
    if slot == "genre":
        return bool((analysis.genre or "").strip())
    return False


def can_ask(candidates: Sequence[MatchingTrack], slot: str) -> bool:
    """이 슬롯이 남은 후보를 실제로 가르는가. 못 가르면 물어도 소용이 없다."""
    counts = value_counts(candidates, slot)
    if len(counts) < 2:
        return False
    if coverage(counts, len(candidates)) < MIN_COVERAGE:
        return False
    return discrimination(counts) >= MIN_DISCRIMINATION


def build_question(slot: str, candidates: Sequence[MatchingTrack]) -> ClarifyQuestion:
    counts = value_counts(candidates, slot)
    options = [
        ClarifyOption(value=value, count=count)
        for value, count in counts.most_common()
    ]
    breakdown = " / ".join(f"{o.value} {o.count}" for o in options)
    return ClarifyQuestion(
        slot=slot,
        question=QUESTION_TEXT[slot],
        options=options,
        reason=f"남은 후보 {len(candidates)}곡이 {breakdown}으로 갈려서",
    )


def pick_question(
    analysis: QueryAnalysis,
    candidates: Sequence[MatchingTrack],
    asked_slots: Sequence[str] = (),
) -> Optional[ClarifyQuestion]:
    """남은 후보를 보고 물을 슬롯을 고른다. 물을 게 없으면 None.

    candidates는 **사용자에게 보여준 곡을 뺀 나머지**여야 한다. 이미 보여준
    곡으로 질문을 만들면 사용자가 아니라고 한 것들을 기준으로 묻게 된다.

    None을 돌려주는 것은 실패가 아니라 정상 동작이다. 호출부는 질문 없이
    Reject-only로 진행하면 된다.
    """
    if not candidates:
        return None

    asked = set(asked_slots)
    for slot in ALLOWED_SLOTS:          # 우선순위 순서 — 앞엣것부터 본다
        if slot in asked:
            continue
        if can_ask(candidates, slot):
            return build_question(slot, candidates)
    return None


# ---------------------------------------------------------------------------
# 답변 반영 — 재검색이 아니라 후보 재정렬
# ---------------------------------------------------------------------------

# 답변 일치 보너스. search_router의 명시 부스팅과 같은 boost_unit 배수를 쓴다.
# 불일치 페널티는 두지 않는다 — 사람의 기억은 틀리므로 반대쪽 후보를 깎으면
# 정답을 잃는다.
#
# 값은 dev 개입 9건에서 ×1.5 / ×3 / ×5 / ×8을 재서 골랐다(2026-09-16).
#
#     배수      ×1.5    ×3     ×5     ×8
#     rule:oracle   6/9   8/9    8/9    8/9
#     rule:noisy    1/9   1/9    1/9    1/9
#
# ×3에서 포화한다. ×1.5는 c701·c705를 놓쳤는데, 둘 다 사용자가 성별을 잘못
# 기억해 후보 풀이 통째로 반대쪽으로 채워진 질의였다(정답이 21위·17위).
# 세 칸 올리는 데 그쳐 Top-10에 못 들어왔다.
#
# ×5·×8도 같은 결과지만 더 작은 값을 택했다. 보너스가 작을수록 기존 검색·
# 리랭킹 신호를 덜 덮어쓰고, 평가 세트에 없는 케이스에서 과하게 개입할 위험이
# 줄어든다. 기존 명시 부스팅 체계(0.25~5, 제목 일치가 최대 ×5) 안에도 자연스럽다.
ANSWER_MATCH_MULTIPLIER = 3.0


def _norm(text: str) -> str:
    return "".join((text or "").split()).lower()


def _parse_decade(value: str) -> Optional[int]:
    """'2010년대' -> 2010. 연대로 해석되지 않으면 None."""
    match = _DECADE_RE.search(value)
    if not match:
        return None
    year = int(match.group(0))
    return year - (year % 10)


def answer_matches(track: MatchingTrack, answer: ClarifyAnswer) -> bool:
    """이 후보가 사용자의 답변과 맞는가.

    `pick_question`이 묻는 것은 vocal_gender·genre뿐이지만, 네 슬롯을 모두
    처리한다. 하네스가 슬롯별 효과를 비교해야 하고, 옛 클라이언트가 우리가
    묻지 않는 슬롯으로 답을 보낼 수도 있다.

    회귀 이력: type·release_era를 빠뜨렸더니 하네스의 해당 정책 네 개가
    53/53건에서 reject_only와 완전히 같은 값을 냈다. "답변이 효과 없음"이
    아니라 "답변이 아예 반영되지 않음"이었는데, 그 수치를 슬롯별 효과의
    근거로 읽을 뻔했다.
    """
    if answer.skipped:
        return False
    value = answer.value.strip()
    if not value:
        return False

    if answer.slot == "genre":
        # 멜론 세분류가 "발라드, 국내드라마"처럼 붙어 오므로 부분 일치로 본다.
        return bool(track.genre) and _norm(value) in _norm(track.genre)

    if answer.slot in ("type", "artist_type"):
        return value in canonical_artist_types(track.artist_types)

    if answer.slot == "release_era":
        want = _parse_decade(value)
        actual = _parse_decade(track.release_date or "")
        return want is not None and want == actual

    return slot_value(track, answer.slot) == value


def apply_answer_bonus(
    scored: Sequence[tuple],
    tracks: Dict[str, MatchingTrack],
    answers: Sequence[ClarifyAnswer],
    boost_unit: float,
    multiplier: Optional[float] = None,
    recorder=None,
) -> list:
    """답변을 **후보 풀 내부 재정렬**로만 반영한다. 재검색하지 않는다.

    왜 재검색이 아닌가 — 측정에서 답변을 질의에 넣으면 **맞는 답변도 순위를
    해치는** 것이 확인됐다. `analysis.genre`는 `_search_performance_clues`의
    sparse 항에 들어가므로, 답변이 그 보조 경로의 검색 결과를 통째로 바꾸고
    RRF 퓨전이 흔들린다. test split의 q200이 그 사례다.

        q200  reject_only   후보 4위  → 최종 4위
              장르 정답 반영 후보 15위 → Top-10 밖   (후보 풀은 그대로, 순위만 추락)

    반면 규칙이 회수한 8건은 **전부 정답이 이미 reject_only 후보 풀 안에**
    있었다(후보 11~27위). 재정렬만으로 닿을 수 있는 자리다. 그래서 재정렬이
    엄격히 우월하다 — 회수는 유지하고 손해만 없앤다.
    계획서 단계 5의 "후보 풀 내부의 메타데이터 호환도 재정렬만"이 이 뜻이었다.

    multiplier는 보정용이다. 기본값(모듈 상수)이 운영 값이고, 하네스가
    `--answer-multiplier`로 바꿔 가며 재서 고른다.
    """
    if not answers:
        return list(scored)

    weight = ANSWER_MATCH_MULTIPLIER if multiplier is None else multiplier
    adjusted = []
    for song_id, score in scored:
        track = tracks.get(song_id)
        if track is not None:
            for answer in answers:
                if answer_matches(track, answer):
                    score += boost_unit * weight
                    # 기록만 남긴다 — 점수 계산은 위 한 줄이 전부다.
                    if recorder is not None:
                        recorder.adjust(
                            song_id,
                            "answer_match",
                            boost_unit * weight,
                            f"{answer.slot}={answer.value}",
                        )
        adjusted.append((song_id, score))

    adjusted.sort(key=lambda item: item[1], reverse=True)
    return adjusted


def analysis_with_answers(
    analysis: QueryAnalysis,
    answers: Sequence[ClarifyAnswer],
) -> QueryAnalysis:
    """답변을 반영한 분석 사본. **검색이 아니라 순위 비교에만 쓴다.**

    답변을 검색 질의에 넣으면 맞는 답변도 순위를 해친다(q200). 그래서 후보
    검색은 원래 분석으로 하고 답변은 `apply_answer_bonus`로만 반영한다.

    그런데 검색 이후 단계에도 질의의 속성을 다시 보는 곳이 있다 —
    `exact_lyric_constraint_score`가 가사 exact 후보를 성별·장르로 다시
    묶는다. 거기서 원래 분석을 쓰면 **사용자가 정정한 답이 무시된다.**
    "남성"이라고 잘못 기억했다가 "여성"으로 고쳐도 남성 그룹이 먼저 온다.
    보너스로 후보 1위가 된 여성 곡이 최종 Top-10에서 통째로 밀려났다.

    그 자리에는 이 함수가 돌려주는 정정된 사본을 넘긴다. 원본은 바꾸지 않는다.
    """
    if not answers:
        return analysis
    merged = analysis.model_copy(deep=True)
    for answer in answers:
        merged = merge_answer(merged, answer)
    return merged


# ---------------------------------------------------------------------------
# 답변 병합 (구 방식 — 재검색 경로에서는 더 이상 쓰지 않는다)
# ---------------------------------------------------------------------------

def answer_confirms_analysis(analysis: QueryAnalysis, answer: ClarifyAnswer) -> bool:
    """답이 원래 질의 분석과 **같은 값**인지 — 새 정보가 아니라 확인만 해 준 답인지.

    병합해도 분석의 해당 슬롯 값이 바뀌지 않으면 확인용 답이다(확신도는 보지 않는다).
    스킵·빈 값·모르는 슬롯은 정보가 없으므로 False다.
    """
    if answer.skipped or not answer.value.strip():
        return False
    value = answer.value.strip()
    slot = answer.slot
    # merge_answer가 무시하는 값(목록 밖 성별·유형, 연대로 못 읽는 값)은 분석을 안 바꾸지만 확인용이 아니라
    # 정보 없음이다 — 리뷰. 그대로 두면 "같은 값"으로 분류돼 빠진다.
    if slot == "vocal_gender" and value not in _VOCAL_GENDERS:
        return False
    if slot in ("type", "artist_type") and value not in _ARTIST_TYPES:
        return False
    if slot == "release_era" and _parse_decade(value) is None:
        return False
    if slot not in ("vocal_gender", "genre", "type", "artist_type", "release_era"):
        return False  # merge_answer를 거치면 "알 수 없는 슬롯" 경고가 찍힌다 — 정보 없음으로 바로 처리
    merged = merge_answer(analysis.model_copy(deep=True), answer)
    if slot == "vocal_gender":
        return bool(analysis.vocal_gender) and merged.vocal_gender == analysis.vocal_gender
    if slot == "genre":
        return bool(analysis.genre) and merged.genre == analysis.genre
    if slot in ("type", "artist_type"):
        return bool(analysis.artist_type.values) and list(merged.artist_type.values) == list(analysis.artist_type.values)
    if slot == "release_era":
        return (
            analysis.release_era.start_year is not None
            and merged.release_era.start_year == analysis.release_era.start_year
            and merged.release_era.end_year == analysis.release_era.end_year
        )
    return False


def answers_for_reranker(
    analysis: QueryAnalysis,
    answers: Sequence[ClarifyAnswer],
) -> list[ClarifyAnswer]:
    """LLM 리랭커 프롬프트에 넘길 답변. 실험 스위치 `GEMINI_RERANK_CORRECTIONS`.

    - all(기본): 전부 넘긴다 — v09 재질문 측정까지의 동작
    - new_only: 원래 분석과 같은 값을 확인해 준 답은 뺀다. listwise가 확인용 답을 "more reliable than
      the original query"로 받고 일반 속성을 과하게 따라 정답을 Top-10 밖으로 보낸 m402(남성·발라드)가 근거
      (results_clarify_v09). 답변 보너스·가사 exact 묶음은 이 함수를 거치지 않는다 — 거기서는 전부 쓴다.
    """
    if reranker_corrections_mode() != "new_only":
        return list(answers)
    return [a for a in answers if not answer_confirms_analysis(analysis, a)]


def reranker_corrections_mode() -> str:
    """`GEMINI_RERANK_CORRECTIONS`의 실제 적용값 — 측정 runinfo에도 이 함수로 적는다."""
    mode = os.getenv("GEMINI_RERANK_CORRECTIONS", "all").strip().lower()
    return "new_only" if mode == "new_only" else "all"


def merge_answer(analysis: QueryAnalysis, answer: ClarifyAnswer) -> QueryAnalysis:
    """사용자 답변을 분석 객체의 해당 슬롯에 채운다.

    ⚠️ **재검색 경로에서는 더 이상 쓰지 않는다.** 답변이 분석에 들어가면
    `analysis.genre`가 보조 검색 경로의 sparse 질의로 흘러가 맞는 답변도
    순위를 해친다(q200: 후보 4위 → 15위). 지금은 `apply_answer_bonus`로
    후보 재정렬에만 반영한다.

    남겨 둔 이유는 둘이다. 하네스가 구 방식과 새 방식을 나란히 재려면 필요하고,
    옛 클라이언트가 우리가 묻지 않는 슬롯(type/release_era)으로 답을 보낼 수
    있다. 부스팅 신호일 뿐 필터가 아니므로 반대쪽 후보를 제거하지는 않는다.

    주의: artist_type과 release_era는 문자열 필드가 아니라 구조화 객체다.
    (ArtistTypeClue.values / ReleaseEra.start_year·end_year)
    setattr로 문자열을 넣으면 검증 에러가 난다.
    """
    if answer.skipped:
        return analysis

    value = answer.value.strip()
    if not value:
        return analysis

    slot = answer.slot

    if slot == "vocal_gender":
        if value not in _VOCAL_GENDERS:
            logger.warning("[clarify] 알 수 없는 보컬 성별 무시: %r", value)
            return analysis
        analysis.vocal_gender = value

    elif slot == "genre":
        analysis.genre = value

    elif slot in ("type", "artist_type"):
        # 프론트가 'type'으로 보낼 수 있어 둘 다 받는다.
        if value not in _ARTIST_TYPES:
            logger.warning("[clarify] 알 수 없는 아티스트 유형 무시: %r", value)
            return analysis
        analysis.artist_type.values = [value]
        analysis.artist_type.confidence = ANSWER_CONFIDENCE

    elif slot == "release_era":
        start = _parse_decade(value)
        if start is None:
            logger.warning("[clarify] 연대로 해석할 수 없는 값 무시: %r", value)
            return analysis
        analysis.release_era.start_year = start
        analysis.release_era.end_year = start + 9
        analysis.release_era.confidence = ANSWER_CONFIDENCE

    else:
        # 모르는 슬롯을 조용히 무시하면 원인 추적이 어려워진다.
        logger.warning("[clarify] 알 수 없는 슬롯 무시: %s", slot)

    return analysis
