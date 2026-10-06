"""
retrieval/explain.py — 검색 실행 기록

## 무엇인가

"왜 이 곡이 1위인가"에 답하기 위한 **실행 기록**이다. 모델이 내부적으로 어떤 단어를
보고 관련도를 높게 줬는지는 여기서 알 수 없다. 대신 **실제 검색·정렬 과정에서 확인된
것**만 남긴다 — 어느 경로에서 몇 위로 들어왔고, 어떤 보정이 얼마나 적용됐고,
리랭킹 전후 순위가 어떻게 바뀌었는지.

그래서 이것으로 만들 수 있는 문장은 "모델이 이 곡을 이렇게 해석했다"가 아니라
"이 곡은 이런 근거로 이 자리에 왔다"다. 그 구분을 지키는 것이 이 모듈의 목적이다.

## 왜 LLM이 아닌가

곡 텍스트를 LLM에 주고 "왜 어울리나"를 물으면 **순위가 틀렸을 때도 그럴듯한 답이
나온다.** 설명이 랭커의 실제 동작과 무관해지면, 설명 기능이 오히려 잘못된 순위를
정당화한다. 기록에서 출발하면 그 위험이 없다 — 기록에 없는 것은 말하지 않는다.

## 불변 조건

**기록을 켜도 순위와 점수가 달라지지 않아야 한다.** 그래서 기록기는 쓰기 전용이고,
파이프라인은 기록기의 값을 읽어 판단하지 않는다. 기본값은 아무것도 하지 않는
`NULL_RECORDER`라 호출부에 `if` 분기를 넣지 않아도 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# 경로 이름 — 화면·로그에 그대로 쓰이므로 여기서만 정의한다.
#
# text_hybrid가 dense와 sparse로 갈라지지 않는 이유: Qdrant 하이브리드 검색이
# 두 점수를 합쳐서 돌려주므로(`vector_db/qdrant_backend.py`) 라우터는 이미 합쳐진
# 하나의 텍스트 경로를 받는다. 분리하려면 Qdrant 어댑터까지 계측해야 한다.
PATH_LABELS: Dict[str, str] = {
    "text_hybrid": "텍스트(의미+키워드)",
    "image": "앨범 이미지",
    "audio": "소리",
    "context": "배경 지식",
    "lyrics_surface": "가사 구절 일치",
    "title": "제목 구조",
    "title_presence": "제목 표기",
    "title_meaning": "제목 의미",
    "balanced_semantic": "의미 균형 보조",
    "performance": "보컬·편성 단서",
    "performance_metadata": "편성 메타데이터",
    "metadata": "메타데이터 단서",
}


# 가사 비교는 **표기 정규화 뒤에** 한다.
#
# `normalize_lyric_surface()`는 NFKC 정규화 + 소문자 변환 뒤 영숫자·한글만 남긴다 —
# 대소문자·공백·개행·구두점이 모두 지워지고 문자 순서만 남는다. 그래서
# `"I FOUND\nTHE WAY!"`는 `"I found the way"`를 **문자열로 포함하지 않지만**
# 정규화하면 둘 다 `"ifoundtheway"`가 되어 일치로 판정된다.
#
# 따라서 "가사 원문에 그대로 있음"이라고 하면 사실보다 강하다. 어간·활용은 바꾸지
# 않으므로 "의미가 비슷함"보다는 강하다 — 그 사이를 정확히 적어야 한다.
LYRIC_NORMALIZATION_NOTE = (
    "대소문자·공백·구두점을 지우고 문자 순서만 비교한다"
    "(NFKC + 소문자 + 영숫자·한글만). 어간과 활용은 바꾸지 않는다"
)

# 가사 표면 검색의 일치 유형 → 사람이 읽는 말.
#
# **점수의 의미가 유형마다 다르다**(`lyrics_exact_search.py`). exact/phonetic은
# 정규화 후 부분문자열로 발견된 경우이고, 그때 점수는 문자 유사도가 아니라
# **분석 모델이 그 단서에 매긴 확신도**다. fuzzy만 유사도가 섞인다
# (유사도 × 확신도). 전부 "일치도"라고 적으면 확신도를 문자 유사도로 읽게 된다.
LYRIC_MATCH_LABELS: Dict[str, str] = {
    "exact": "기억한 구절이 가사와 표기 정규화 후 일치",
    "phonetic": "음차 추정 표기가 가사와 표기 정규화 후 일치",
    "fuzzy": "가사와 근사 일치",
}

# 그 점수를 무엇이라 불러야 하는가.
LYRIC_SCORE_LABELS: Dict[str, str] = {
    "exact": "단서 확신도",
    "phonetic": "단서 확신도",
    "fuzzy": "유사도×확신도",
}


# 가사 일치를 **순서로** 반영하는 규칙들.
#
# 점수로 반영되는 경로 기여(`lyrics_surface`)와 나란히 봐야 "이 가사가 순위를
# 만들었나"를 판단할 수 있다. 부스트를 꺼 둔 대조 실험(`LYRIC_BOOST_SCALE=0`)에서는
# 기여가 0이면서 보호만 걸리는 경우가 있고, 그때도 순위는 가사가 만든 것이다.
#
# `search_router.lyric_priority_rule()`이 내는 값과 같아야 한다(시험으로 고정).
LYRIC_ORDER_RULES = frozenset({"lyrics_exact_priority", "lyrics_phonetic_priority"})


# 리랭커 한 번의 실행 결과. 백엔드가 직접 보고해야 한다.
#
# 호출 횟수로는 알 수 없다 — Gemini listwise는 재시도가 전부 실패해도 예외를 던지지 않고
# **기존 순서를 그대로 돌려준다**. 호출은 있었지만 리랭킹은 없었던 것이다.
RERANK_APPLIED = "applied"   # 모델이 돌아 순서를 정했다
RERANK_FAILED = "failed"     # 호출했으나 실패해 입력 순서로 폴백했다
RERANK_SKIPPED = "skipped"   # 모델을 부를 조건이 아니었다(후보 없음, 꺼짐)
RERANK_UNKNOWN = "unknown"   # 상태를 보고하지 않는 백엔드. 단정하지 않는다


@dataclass
class ScoreMix:
    """리랭킹 **최종 점수를 무엇으로 합성했는가.**

    리랭커 점수가 곧 최종 점수가 아니다. 두 백엔드 모두 기존 검색 점수를 섞는다.
    그래서 "리랭커가 0.998을 줬다"만으로 "리랭커가 이 순서를 정했다"고 말하면
    틀린다 — 실제로 CE 0.998인 곡이 1위가 아닌 경우가 있었다.

    무엇을 섞는지가 백엔드마다 다르므로 그것까지 남긴다.
      Cross-Encoder : weight × CE 점수         + (1-weight) × min-max(검색 점수)
      Gemini        : weight × (순위·관련도)   + (1-weight) × 순위 점수(검색)
    """

    backend: str                        # 합성식을 만든 백엔드
    weight: float                       # 리랭커 쪽에 실린 **최종** 가중치
    configured_weight: float            # 설정값. confidence로 깎이기 전
    confidence: float = 1.0             # 가중치를 깎은 비율. 1.0이면 깎지 않았다
    rerank_component: float = 0.0       # 합성에 들어간 리랭커 쪽 값
    retrieval_component: float = 0.0    # 합성에 들어간 검색 쪽 값
    final: float = 0.0                  # weight×rerank + (1-weight)×retrieval

    # 리랭커 쪽 값이 정규화된 값인지 원점수인지. Cross-Encoder는 spread_ref가
    # 꺼져 있으면 CE 원점수를 그대로 섞는다 — "정규화된 점수"라고 하면 거짓이다.
    rerank_normalized: bool = True

    strategy: str = ""                  # 정렬 전략(전체 재정렬 / 저신뢰 상위 N개만)

    # **이 곡의 순서가 합성 점수로 결정됐는가.**
    # 저신뢰 전략에서는 상위 N개만 다시 정렬하고 그 아래는 검색 순서를 그대로 둔다.
    # 점수는 합성됐지만 순서에는 쓰이지 않았으므로 구분해야 한다.
    reordered: bool = True

    detail: str = ""


@dataclass
class RerankRun:
    """리랭커 실행 한 번의 결과.

    `judged_ids`가 따로 있는 이유: 백엔드가 **넘겨받은 후보를 전부 평가하지는
    않는다.** Gemini listwise는 `max_candidates`까지만 보고 나머지는 그대로 뒤에
    붙인다. 전달한 곡 전체에 "리랭킹됨"을 붙이면, 모델이 본 적 없는 곡에 대해
    리랭킹을 말하게 된다.
    """

    # 트랙 타입을 명시하지 않는다 — MatchingTrack을 import하면 순환이 생기고
    # (schemas → … → explain), 이 모듈은 트랙의 구조를 쓰지 않는다.
    tracks: List[Any] = field(default_factory=list)
    status: str = RERANK_SKIPPED
    judged_ids: List[str] = field(default_factory=list)

    # 곡별 최종 점수 합성식. 백엔드만 아는 값이라 백엔드가 보고해야 한다.
    mixes: Dict[str, ScoreMix] = field(default_factory=dict)

    # 모델이 스스로 쓴 이유 문장. **검증된 기록과 섞지 않는다**(아래 SongExplain 참고).
    model_notes: Dict[str, List[str]] = field(default_factory=dict)

    # 모델이 낸 순서 **뒤에** 규칙이 위치를 또 바꾼 것. (song_id, rule, detail)
    #
    # Gemini 경로는 모델 순서에 rescue/보호 규칙을 연달아 적용한다. 그 이동까지
    # 모델에 귀속하면 "리랭킹이 12위에서 3위로 올림"처럼 모델이 하지 않은 일을
    # 말하게 된다.
    order_notes: List[Tuple[str, str, str]] = field(default_factory=list)

    # 규칙이 개입하기 **전**, 리랭킹 단계 자체가 만든 순서.
    #
    # 비어 있으면 `tracks`의 순서가 그대로 리랭킹 단계의 결과라는 뜻이다.
    # order_notes가 있는 백엔드는 반드시 이것을 채워야 한다 — 그러지 않으면
    # 규칙이 옮긴 위치를 모델의 정렬로 기록하게 된다.
    model_order_ids: List[str] = field(default_factory=list)


# 요청 전체의 재정렬 단계. 실행 결과들에서 파생한다.
REORDER_NOT_ATTEMPTED = "not_attempted"      # 게이트를 통과하지 못해 시도 자체가 없음
REORDER_MODEL_APPLIED = "model_applied"      # 한 번이라도 모델이 돌았다
REORDER_MODEL_FAILED = "model_failed"        # 실행은 했으나 성공이 0회
REORDER_MODEL_SKIPPED = "model_skipped"      # 실행은 했으나 전부 건너뜀
REORDER_MODEL_UNKNOWN = "model_unknown"      # 상태를 모르는 백엔드
REORDER_ORDER_RULES_ONLY = "order_rules_only"  # 모델 실행 0회, 규칙만 순서를 바꿈

# 단계 → 사람이 읽는 말. 라벨은 이 모듈에서만 정의한다(PATH_LABELS와 같은 이유).
#
# **이 라벨은 리랭킹 단계에 무슨 일이 있었는지만 말한다.** 최종 순서가 무엇인지는
# 말하지 않는다 — 요청 단위로는 알 수 없기 때문이다. 모델이 실패해도 가사 보호
# 규칙은 그대로 적용되므로 "실패해서 검색 순서를 사용"은 거짓이 될 수 있다.
# 실제로 그 조합에서 곡별 설명("최종 순위 2→1위")과 정면으로 모순됐다.
# 최종 배치는 곡별 기록이 말한다.
REORDER_STAGE_LABELS: Dict[str, str] = {
    REORDER_NOT_ATTEMPTED: "리랭킹을 시도하지 않음",
    REORDER_MODEL_APPLIED: "리랭킹 적용",
    REORDER_MODEL_FAILED: "리랭킹 실패",
    REORDER_MODEL_SKIPPED: "리랭킹 건너뜀(조건 미충족)",
    REORDER_MODEL_UNKNOWN: "리랭킹 결과 확인 불가",
    REORDER_ORDER_RULES_ONLY: "모델 미실행, 순서 규칙만 적용",
}


@dataclass
class PathContribution:
    """한 경로가 이 곡을 얼마나 밀어 올렸는지."""

    path: str
    rank: int          # 그 경로 안에서의 순위 (1-based)
    delta: float       # 융합 점수에 **실제로** 더해진 양
    detail: str = ""   # "원문 그대로 일치 ×8.0, 단서 확신도 0.92" 처럼 근거를 남긴다

    # RRF 절단에서 탈락해 **최종 점수에 반영되지 않은** 기여.
    # 탈락한 곡이 가사 경로로 다시 들어오면 점수는 0에서 다시 시작하므로,
    # 이 기여를 합계에 넣으면 설명이 실제 점수보다 커진다.
    dropped: bool = False

    @property
    def label(self) -> str:
        return PATH_LABELS.get(self.path, self.path)


@dataclass
class ScoreAdjustment:
    """점수를 더하거나 뺀 규칙. delta가 음수면 감점이다."""

    rule: str
    delta: float
    detail: str = ""


@dataclass
class OrderRule:
    """점수와 무관하게 **순서만** 바꾼 규칙.

    가사 exact 그룹 우선 배치가 그 예다. 점수를 보면 이유가 보이지 않으므로
    따로 남겨야 한다.
    """

    rule: str
    detail: str = ""


@dataclass
class SongExplain:
    song_id: str
    paths: List[PathContribution] = field(default_factory=list)
    adjustments: List[ScoreAdjustment] = field(default_factory=list)
    order_rules: List[OrderRule] = field(default_factory=list)

    fused_score: Optional[float] = None       # 경로 융합까지의 점수
    retrieval_score: Optional[float] = None   # 보정까지 반영한 리랭킹 직전 점수

    # 요청 전체 기준 순위. **두 원인이 섞여 있다** — 모델의 정렬과 보호 배치로 인한
    # 밀림이 같은 숫자에 들어간다. 그래서 아래 두 값을 따로 둔다.
    rank_before_rerank: Optional[int] = None
    rank_after_rerank: Optional[int] = None

    # 모델이 **자기가 본 묶음 안에서** 이 곡을 어디로 옮겼는가.
    # 이것만이 모델에 귀속할 수 있는 변화다.
    group_rank_before: Optional[int] = None
    group_rank_after: Optional[int] = None

    # 이 곡 앞에 배치된 **가사 보호 곡의 수.**
    #
    # 하락 폭이 아니다. 보호 곡이 애초에 1위였다면 2위 곡은 2위 그대로다.
    # 그래서 "밀려남"이라고 단정하지 않고 "보호 곡 뒤에 배치"라는 사실만 남긴다.
    # 실제 하락은 rank_before/after가 따로 말해 준다.
    behind_protected: int = 0

    rerank_score: Optional[float] = None

    # 가사 표면 일치가 무엇으로 성립했는지 — 계측용.
    #
    # 점수만으로는 부스트가 어떤 단서에 과도하게 걸리는지 알 수 없다. 짧은 조각이
    # 여러 곡에 걸려도 점수는 단서 확신도 그대로다(q211: "위험하다" 4글자가 2곡에
    # 걸려 ×8.0을 받았다). 구절 길이와 코퍼스 일치 곡 수가 그것을 드러낸다.
    #
    # 트랙 타입을 쓰지 않도록 평범한 dict로 담는다(이 모듈은 src를 import하지 않는다).
    lyric_match: Optional[Dict[str, Any]] = None

    # 가사 부스트 **직전·직후**의 융합 순위. 부스트가 실제로 몇 칸을 올렸는가.
    lyric_boost_rank_before: Optional[int] = None
    lyric_boost_rank_after: Optional[int] = None

    # 최종 점수를 무엇으로 합성했는가. rerank_score 하나만으로는 순서를 설명할 수
    # 없다 — 리랭커 점수가 가장 높은 곡이 1위가 아닌 경우가 실제로 있었다.
    score_mix: Optional[ScoreMix] = None

    # 모델이 스스로 쓴 이유 문장(Gemini listwise의 reason).
    #
    # **검증된 기록이 아니다.** 여러 pass의 응답을 평균해 순서를 만들고 그 뒤에
    # rescue 규칙이 순위를 또 바꾸므로, 이 문장이 최종 순위의 이유라고 보장할 수
    # 없다. 그래서 확인된 근거와 같은 목록에 섞지 않고 따로 담는다 —
    # render_ko()는 이것을 쓰지 않으며, 노출 단계에서 "모델이 말한 이유(미검증)"로
    # 라벨을 붙여 따로 보여준다.
    model_notes: List[str] = field(default_factory=list)

    # **이 곡의 최종 순위에 리랭커 결과가 반영됐는가.**
    #
    # "모델이 성공했다"와는 다른 질문이다. 두 가지 이유로 어긋난다.
    #   - 가사 보호 곡이 단독 그룹이면 모델 호출을 건너뛴다
    #   - Gemini는 max_candidates까지만 평가하고 뒷부분은 그대로 붙인다
    #   - 한 그룹이 성공한 뒤 다음 그룹에서 예외가 나면 **전체가 폴백**한다
    reorder_applied: bool = False

    # 리랭커가 스스로 만든 문장(Gemini listwise의 reason)은 **여기에 두지 않는다.**
    # 보존 자체는 할 가치가 있지만, 검증된 실행 기록과 섞이면 안 된다 — Gemini는
    # 여러 응답을 합치고 그 뒤에 규칙이 순위를 또 바꾸므로 한 응답의 문장이 최종
    # 순위의 이유라고 보장할 수 없다. 별도 필드로 분리해 넣는 것은 노출 단계의 일이다.

    def path_total(self) -> float:
        """**최종 점수에 반영된** 경로 기여 합. 탈락한 기여는 빼야 fused_score와 맞는다."""
        return sum(p.delta for p in self.paths if not p.dropped)

    def dropped_paths(self) -> List[PathContribution]:
        return [p for p in self.paths if p.dropped]

    def live_paths(self) -> List[PathContribution]:
        return [p for p in self.paths if not p.dropped]

    def adjustment_total(self) -> float:
        return sum(a.delta for a in self.adjustments)


@dataclass
class SearchExplain:
    """한 요청의 실행 기록."""

    query: str = ""
    modality_weights: Dict[str, float] = field(default_factory=dict)

    # **가중치가 0이 아니어도 그 경로가 돌았다는 뜻은 아니다.** 경로가 예외로
    # 죽으면 라우터는 빈 결과로 대체하고 계속 간다. 그것을 남기지 않으면
    # "가중치 0.20인데 기여가 한 줄도 없다"가 되어, 보는 사람은 그 경로가
    # 돌았지만 이 곡을 못 올린 것으로 읽는다. 경로 코드 → 사유.
    failed_paths: Dict[str, str] = field(default_factory=dict)

    # 질의 분석이 Gemini가 아니라 규칙 폴백이었는가. 쿼터 초과·타임아웃·키 미설정에서
    # 그렇게 되는데, 검색은 그대로 진행되므로 화면만 보면 구분되지 않는다.
    analysis_fallback: bool = False

    reranker: str = "none"           # 주입된 리랭커 클래스명 또는 "none"
    rerank_error: str = ""           # 라우터가 예외를 잡았으면 사유
    # 리랭커를 부른 횟수와 그 결과. 호출 수 ≠ 성공 수다.
    rerank_runs: List[str] = field(default_factory=list)
    songs: Dict[str, SongExplain] = field(default_factory=dict)

    # 게이트를 통과해 재정렬을 시도했는가. 통과하지 못하면 실행 자체가 없다.
    reorder_attempted: bool = False
    # 재정렬 경로가 끝까지 가서 **그 결과가 최종 순서가 됐는가.**
    # 중간에 예외가 나면 앞 그룹이 성공했더라도 전체가 검색 순서로 돌아간다.
    reorder_committed: bool = False
    # 규칙이 순서를 바꿨는가(가사 보호 배치). 모델 실패와 **동시에** 참일 수 있다.
    order_rules_applied: bool = False

    @property
    def rerank_runs_applied(self) -> int:
        return sum(1 for status in self.rerank_runs if status == RERANK_APPLIED)

    @property
    def rerank_applied(self) -> bool:
        """모델이 한 번이라도 성공했는가. 호출 여부와는 다른 질문이다."""
        return self.rerank_runs_applied > 0

    @property
    def reorder_stage(self) -> str:
        """실행 결과에서 파생한다. 따로 두면 기록과 어긋날 수 있다.

        **커밋 여부를 먼저 본다.** 실행이 성공했는지가 아니라 그 결과가 최종
        순서가 됐는지가 설명의 기준이다 — 앞 그룹이 성공하고 뒤에서 예외가 나면
        전체가 검색 순서로 돌아가므로, 성공 이력이 있어도 반영된 것은 없다.
        """
        if not self.reorder_attempted:
            return REORDER_NOT_ATTEMPTED
        if not self.reorder_committed:
            return REORDER_MODEL_FAILED
        if self.rerank_applied:
            return REORDER_MODEL_APPLIED
        if not self.rerank_runs:
            return (
                REORDER_ORDER_RULES_ONLY
                if self.order_rules_applied
                else REORDER_NOT_ATTEMPTED
            )
        if any(status == RERANK_FAILED for status in self.rerank_runs):
            return REORDER_MODEL_FAILED
        if any(status == RERANK_UNKNOWN for status in self.rerank_runs):
            return REORDER_MODEL_UNKNOWN
        if self.order_rules_applied:
            return REORDER_ORDER_RULES_ONLY
        return REORDER_MODEL_SKIPPED

    def get(self, song_id: str) -> SongExplain:
        found = self.songs.get(song_id)
        if found is None:
            found = SongExplain(song_id=song_id)
            self.songs[song_id] = found
        return found


class ExplainRecorder:
    """쓰기 전용 기록기.

    파이프라인은 이 객체에 값을 넣기만 한다. 여기서 읽어 판단하는 코드가 생기면
    "기록을 켜면 순위가 바뀐다"가 되므로 넣지 않는다.
    """

    enabled = True

    def __init__(self, query: str = "") -> None:
        self.record = SearchExplain(query=query)

    # --- 요청 단위 ---
    def set_weights(self, text: float, image: float, audio: float) -> None:
        self.record.modality_weights = {"text": text, "image": image, "audio": audio}

    def note_path_failed(self, path: str, reason: str) -> None:
        """이 경로가 예외로 죽어 빈 결과로 대체됐다.

        기여가 **없는 것**과 경로가 **죽은 것**은 다른 사실이다. 둘 다 화면에서는
        "그 경로 줄이 없다"로 보이므로, 죽었다는 것은 따로 적어야 말할 수 있다.
        """
        self.record.failed_paths[path] = reason

    def set_analysis_fallback(self, fallback: bool) -> None:
        self.record.analysis_fallback = bool(fallback)

    def set_reranker(self, name: str) -> None:
        self.record.reranker = name

    def set_reorder_attempted(self, attempted: bool) -> None:
        self.record.reorder_attempted = attempted

    def note_order_rules_applied(self) -> None:
        self.record.order_rules_applied = True

    def note_rerank_run(self, status: str, judged_ids: Iterable[str]) -> None:
        """리랭커 실행 한 번을 결과와 함께 기록한다.

        judged_ids는 백엔드가 **실제로 평가한 곡**이다. 넘겨준 후보 전체가 아니다 —
        Gemini는 max_candidates까지만 본다. 모델이 못 본 곡에 리랭킹을 말하면
        거짓이 되므로 평가한 곡만 표시한다.

        이 표시는 아직 **잠정**이다. 뒤에서 예외가 나 전체가 폴백하면
        revoke_reorder()로 취소된다.
        """
        self.record.rerank_runs.append(status)
        if status != RERANK_APPLIED:
            return
        for song_id in judged_ids:
            self.record.get(song_id).reorder_applied = True

    def commit_reorder(self) -> None:
        """재정렬 결과가 최종 순서가 됐다."""
        self.record.reorder_committed = True

    def revoke_reorder(self) -> None:
        """전체가 검색 순서로 폴백했다. 앞 그룹의 성공 표시도 취소한다.

        합성식과 모델 문장도 지운다. 합성 점수는 최종 순서에 쓰이지 않았고,
        모델의 문장은 버려진 순서를 설명하는 것이므로 남기면 둘 다 거짓이 된다.
        """
        self.record.reorder_committed = False
        for explain in self.record.songs.values():
            explain.reorder_applied = False
            explain.order_rules.clear()
            explain.behind_protected = 0
            explain.group_rank_before = None
            explain.group_rank_after = None
            explain.score_mix = None
            explain.model_notes.clear()

    def set_rerank_error(self, message: str) -> None:
        self.record.rerank_error = message

    def mark_paths_dropped(self, song_ids) -> None:
        """이 곡들의 지금까지 기록된 경로 기여를 '최종 점수 미반영'으로 표시한다."""
        for song_id in song_ids:
            for contribution in self.record.get(song_id).paths:
                contribution.dropped = True

    # --- 곡 단위 ---
    def path(
        self,
        song_id: str,
        path: str,
        rank: int,
        delta: float,
        detail: str = "",
    ) -> None:
        self.record.get(song_id).paths.append(
            PathContribution(path=path, rank=rank, delta=float(delta), detail=detail)
        )

    def adjust(self, song_id: str, rule: str, delta: float, detail: str = "") -> None:
        if not delta:
            return
        self.record.get(song_id).adjustments.append(
            ScoreAdjustment(rule=rule, delta=float(delta), detail=detail)
        )

    def order(self, song_id: str, rule: str, detail: str = "") -> None:
        self.record.get(song_id).order_rules.append(OrderRule(rule=rule, detail=detail))

    def set_fused(self, song_id: str, score: float) -> None:
        self.record.get(song_id).fused_score = float(score)

    def set_retrieval(self, song_id: str, score: float) -> None:
        self.record.get(song_id).retrieval_score = float(score)

    def set_rank_before(self, ranked_ids: List[str]) -> None:
        for rank, song_id in enumerate(ranked_ids, start=1):
            self.record.get(song_id).rank_before_rerank = rank

    def set_rank_after(self, ranked_ids: List[str]) -> None:
        for rank, song_id in enumerate(ranked_ids, start=1):
            self.record.get(song_id).rank_after_rerank = rank

    def set_group_ranks(self, before_ids, after_ids, only: Iterable[str]) -> None:
        """모델이 본 묶음 안에서의 전후 순위. `only`(평가한 곡)에만 남긴다.

        평가하지 않은 뒷부분에 순위를 붙이면 모델이 정렬한 것처럼 보인다.
        """
        allowed = set(only)
        for rank, song_id in enumerate(before_ids, start=1):
            if song_id in allowed:
                self.record.get(song_id).group_rank_before = rank
        for rank, song_id in enumerate(after_ids, start=1):
            if song_id in allowed:
                self.record.get(song_id).group_rank_after = rank

    def note_behind_protected(self, song_ids, count: int) -> None:
        """이 곡들 앞에 배치된 보호 곡 수를 남긴다.

        하락 폭이 아니다 — 보호 곡이 원래 앞에 있었으면 순위는 그대로다.
        """
        if count <= 0:
            return
        for song_id in song_ids:
            self.record.get(song_id).behind_protected = count

    def set_rerank_score(self, song_id: str, score: Optional[float]) -> None:
        if score is not None:
            self.record.get(song_id).rerank_score = float(score)

    def note_lyric_match(self, song_id: str, detail: Optional[Dict[str, Any]]) -> None:
        """가사 표면 일치의 근거. 계측용이며 랭킹 계산에 쓰이지 않는다."""
        if detail:
            self.record.get(song_id).lyric_match = dict(detail)

    def note_lyric_boost_ranks(self, before, after) -> None:
        """가사 부스트 직전·직후의 융합 순위. before에 없던 곡은 None으로 남는다."""
        for song_id, rank in after.items():
            explain = self.record.get(song_id)
            explain.lyric_boost_rank_before = before.get(song_id)
            explain.lyric_boost_rank_after = rank

    def set_score_mix(self, song_id: str, mix: ScoreMix) -> None:
        """최종 점수 합성식. 백엔드가 보고한 것을 그대로 담는다."""
        self.record.get(song_id).score_mix = mix

    def note_model_reason(self, song_id: str, texts: Iterable[str]) -> None:
        """모델이 스스로 쓴 이유 문장. 검증된 근거와 **섞지 않는다.**"""
        explain = self.record.get(song_id)
        for text in texts:
            cleaned = " ".join(str(text).split())
            if cleaned and cleaned not in explain.model_notes:
                explain.model_notes.append(cleaned)


class _NullRecorder(ExplainRecorder):
    """아무것도 기록하지 않는 기본값.

    호출부가 `rec is not None` 분기를 두지 않아도 되게 하려고 null object로 둔다.
    분기가 없으면 "기록을 켰을 때만 지나가는 코드"가 생기지 않는다.
    """

    enabled = False

    def __init__(self) -> None:  # noqa: D107 - record를 만들지 않는다
        self.record = SearchExplain()

    def set_weights(self, text: float, image: float, audio: float) -> None:
        return

    def note_path_failed(self, path: str, reason: str) -> None:
        return

    def set_analysis_fallback(self, fallback: bool) -> None:
        return

    def set_reranker(self, name: str) -> None:
        return

    def set_reorder_attempted(self, attempted: bool) -> None:
        return

    def note_order_rules_applied(self) -> None:
        return

    def note_rerank_run(self, status, judged_ids) -> None:
        return

    def commit_reorder(self) -> None:
        return

    def revoke_reorder(self) -> None:
        return

    def set_rerank_error(self, message: str) -> None:
        return

    def mark_paths_dropped(self, song_ids) -> None:
        return

    def path(self, song_id, path, rank, delta, detail="") -> None:
        return

    def adjust(self, song_id, rule, delta, detail="") -> None:
        return

    def order(self, song_id, rule, detail="") -> None:
        return

    def set_fused(self, song_id, score) -> None:
        return

    def set_retrieval(self, song_id, score) -> None:
        return

    def set_rank_before(self, ranked_ids) -> None:
        return

    def set_rank_after(self, ranked_ids) -> None:
        return

    def set_group_ranks(self, before_ids, after_ids, only) -> None:
        return

    def note_behind_protected(self, song_ids, count) -> None:
        return

    def set_rerank_score(self, song_id, score) -> None:
        return

    def note_lyric_match(self, song_id, detail) -> None:
        return

    def note_lyric_boost_ranks(self, before, after) -> None:
        return

    def set_score_mix(self, song_id, mix) -> None:
        return

    def note_model_reason(self, song_id, texts) -> None:
        return


NULL_RECORDER = _NullRecorder()


# ---------------------------------------------------------------------------
# 문장화 — 기록에 있는 것만 말한다
# ---------------------------------------------------------------------------

# 규칙 이름 → 사람이 읽는 말. 기록에 남은 rule 문자열은 전부 여기 있어야 한다.
RULE_LABELS: Dict[str, str] = {
    # 가사 단서는 **어느 필드에서 맞았는지**를 구분해야 한다. 이 규칙이 보는 것은
    # full_lyrics가 아니라 lyrics_highlight(발췌)와 lyrics_summary(요약)를 합친
    # 텍스트다. 요약에만 있는 문구를 "가사에 있음"이라고 하면 사실이 아니다.
    # 원문 전체 일치는 별도 경로(lyrics_surface)가 담당한다.
    #
    # 이 비교도 `normalize_lyric_surface()`를 거친다(LYRIC_NORMALIZATION_NOTE).
    # "발췌에 있음"이라고 하면 원문 그대로 있었다는 뜻이 되므로 쓰지 않는다.
    "lyric_clue_in_highlight": "기억한 구절이 가사 발췌와 표기 정규화 후 일치",
    "lyric_clue_in_summary": "기억한 구절이 가사 요약과 표기 정규화 후 일치",
    "lyric_clue_in_combined": "기억한 구절이 가사 발췌·요약 경계에서 표기 정규화 후 일치",
    "lyric_phonetic_in_highlight": "음차 추정 표기가 가사 발췌와 표기 정규화 후 일치",
    "lyric_phonetic_in_summary": "음차 추정 표기가 가사 요약과 표기 정규화 후 일치",
    "lyric_phonetic_in_combined": "음차 추정 표기가 가사 발췌·요약 경계에서 표기 정규화 후 일치",
    "lyric_keyword_tokens": "가사 발췌·요약에서 키워드 겹침(표기 정규화 후)",
    "title_exact": "제목 완전일치",
    "title_constraints": "제목 구조 일치",
    "title_hanja_presence": "제목의 한자 표기 일치",
    "title_meaning": "제목 의미 단서 일치",
    "artist_match": "가수 일치",
    "vocal_gender_match": "말한 성별과 일치",
    "vocal_gender_partial": "혼성이라 부분 일치",
    "vocal_gender_mismatch": "말한 성별과 다름",
    "release_era": "발매 시기 일치",
    "artist_type": "솔로·그룹 형태 일치",
    "performance_clues": "보컬 역할·편성 단서 일치",
    "genre_match": "장르 일치",
    "answer_match": "재질문 답변과 일치",
    # 보호 배치는 일치 유형마다 규칙이 다르다. 하나로 묶으면 음차 추정 일치에
    # "가사가 그대로 일치"가 붙어 두 번 틀린 말이 된다(정규화 후 비교 + 음차 추정).
    "lyrics_exact_priority": "기억한 구절이 가사와 일치해 먼저 배치",
    "lyrics_phonetic_priority": "음차 추정 표기의 가사 일치를 보호해 먼저 배치",
    "lyrics_behind_protected": "가사 일치 보호 곡 뒤에 배치",
    # 모델이 낸 순서를 **그 뒤에** 규칙이 다시 바꾼 것. 모델에 귀속하면 거짓이 된다.
    "gemini_lyric_anchor_protect": "가사 표면일치 1위를 모델 순서보다 앞에 고정",
    "gemini_title_shape_rescue": "제목 구조가 완전일치해 상위 구간에 보존",
    "gemini_title_shape_demoted": "제목 구조 일치 곡에 자리를 내주어 뒤로 이동",
    "gemini_rare_fact_rescue": "외부 검증된 희소 단서라 상위로 끌어올림",
}


def _fmt(delta: float) -> str:
    sign = "+" if delta >= 0 else "−"
    return f"{sign}{abs(delta):.4f}"


def _mix_note(mix: ScoreMix) -> str:
    """최종 점수를 무엇으로 합성했는지 한 문장으로.

    이 문장이 필요한 이유: 리랭커 점수가 가장 높은 곡이 1위가 아닌 경우가 있다.
    합성 비율을 밝히지 않으면 "리랭커가 순서를 정했다"는 잘못된 인상을 준다.
    """
    share = f"리랭커 {mix.weight:.0%} + 검색 {1.0 - mix.weight:.0%}"
    if mix.confidence < 1.0:
        share += f"(신뢰도 {mix.confidence:.2f}로 {mix.configured_weight:.0%}에서 축소)"
    return f"최종 점수는 {share} 합성"


def render_ko(
    explain: SongExplain,
    *,
    max_items: int = 3,
    reorder_stage: str = REORDER_NOT_ATTEMPTED,
) -> str:
    """기록만으로 짧은 한국어 설명을 만든다.

    LLM을 쓰지 않는다. 기록에 없는 주장은 만들 수 없으므로 환각이 생기지 않는다.
    대신 문장이 기계적이다 — 그것이 이 단계의 의도다.

    reorder_stage가 필요한 이유: 순위가 바뀌었다는 사실만으로는 **모델이 돌았는지**
    알 수 없다. 리랭킹을 끈 경우와 실패한 경우에도 전후 순위는 기록되므로, 상태를
    받지 않으면 "리랭킹이 N위 유지"처럼 하지 않은 일을 말하게 된다.

    `explain.model_notes`는 **쓰지 않는다.** 모델이 스스로 쓴 문장은 검증된 기록이
    아니고, 여기에 섞으면 확인된 근거와 구별할 수 없게 된다. 노출 단계에서 별도
    라벨로 보여줄 몫이다.
    """
    parts: List[str] = []

    # 순서를 바꾼 규칙이 있으면 그것이 가장 중요한 이유다.
    for rule in explain.order_rules:
        label = RULE_LABELS.get(rule.rule, rule.rule)
        parts.append(f"{label}{f' ({rule.detail})' if rule.detail else ''}")

    # 최종 점수에 반영된 기여만 근거로 쓴다.
    strongest_paths = sorted(explain.live_paths(), key=lambda p: p.delta, reverse=True)
    for p in strongest_paths[:max_items]:
        detail = f", {p.detail}" if p.detail else ""
        parts.append(f"{p.label} 경로 {p.rank}위{detail} ({_fmt(p.delta)})")

    strongest_adj = sorted(explain.adjustments, key=lambda a: abs(a.delta), reverse=True)
    for a in strongest_adj[:max_items]:
        label = RULE_LABELS.get(a.rule, a.rule)
        detail = f", {a.detail}" if a.detail else ""
        parts.append(f"{label}{detail} ({_fmt(a.delta)})")

    before, after = explain.rank_before_rerank, explain.rank_after_rerank

    # 재정렬은 **원인이 둘 이상 섞인다.** 모델이 정렬한 것과, 가사 보호 곡이
    # 앞에 배치되어 밀려난 것이 같은 순위 변화에 들어간다. 그래서 하나를 골라
    # 말하지 않고, 확인된 사실을 각각 적는다.
    #
    #   (1) 최종 순위가 어떻게 바뀌었나 — 원인을 붙이지 않는다
    #   (2) 모델이 자기 묶음 안에서 무엇을 했나
    #   (3) 규칙이 무엇을 했나
    notes: List[str] = []

    group_before, group_after = explain.group_rank_before, explain.group_rank_after

    # **점수를 냈다와 자리를 정했다는 다른 일이다.**
    #
    # 저신뢰 전략은 상위 N개만 다시 정렬하고 그 아래는 검색 순서를 그대로 둔다.
    # 그 곡들도 합성 점수는 받으므로 순위 변화를 리랭커에 귀속하기 쉽지만,
    # 실제로 그 자리를 정한 것은 검색 순서다. 운영 설정(RERANKER_SPREAD_REF=0.02)에서
    # 실제로 일어나는 일이라 — 어떤 질의의 6~10위가 전부 여기에 해당했다 —
    # 구분하지 않으면 매번 거짓을 말한다.
    order_from_model = explain.score_mix is None or explain.score_mix.reordered

    same_as_group = (
        before is not None
        and after is not None
        and before == group_before
        and after == group_after
    )

    def _moved(b: int, a: int) -> str:
        return f"{b}위에서 {a}위로 " + ("올림" if a < b else "내림")

    if explain.reorder_applied and same_as_group and order_from_model:
        # 모델이 본 묶음이 후보 전체와 같다 — 순위 변화를 모델에 귀속할 수 있다.
        notes.append(
            f"리랭킹이 {_moved(before, after)}"
            if before != after
            else f"리랭킹이 {after}위 유지"
        )
    else:
        rank_changed = (
            before is not None and after is not None and before != after
        )
        if rank_changed:
            notes.append(f"최종 순위 {before}→{after}위")

        if explain.reorder_applied and not order_from_model:
            # 점수는 합성에 들어갔지만 이 곡의 자리는 리랭커가 정하지 않았다.
            notes.append(
                "리랭커 점수는 합성됐으나 순서는 검색 순서 유지"
                f"({explain.score_mix.strategy})"
            )
        elif explain.reorder_applied:
            if group_before is None or group_after is None:
                # 평가는 됐지만 반환 목록에 없다. 어디로 갔는지는 알 수 없으므로
                # "유지"라고 하면 안 된다 — 정답이 결과 밖으로 밀린 이유를
                # 분석할 때 정확히 이 문장이 거짓이 된다.
                notes.append("평가됐으나 반환 결과 밖")
            elif group_before != group_after:
                notes.append(f"리랭킹은 묶음 안에서 {_moved(group_before, group_after)}")
            else:
                notes.append("리랭킹 적용(묶음 안 순서 유지)")
        elif reorder_stage == REORDER_MODEL_FAILED:
            # 실패했는데 순위도 그대로면 "검색 순서 유지"가 사실이다. 순위가
            # 바뀌었다면 원인이 따로 있으므로 그렇게 말하면 안 된다.
            notes.append(
                "리랭킹은 실패"
                if rank_changed or explain.behind_protected
                else "리랭킹이 실패해 검색 순서 유지"
            )
        elif reorder_stage == REORDER_MODEL_UNKNOWN:
            notes.append("리랭킹 결과를 확인할 수 없음")
        elif explain.order_rules:
            # 모델은 돌았지만 이 곡은 넘기지 않았다(단독 보호 그룹).
            notes.append("순서 규칙으로 배치(리랭커에 넘기지 않음)")

    if explain.behind_protected:
        label = RULE_LABELS["lyrics_behind_protected"]
        notes.append(f"{label} {explain.behind_protected}곡")

    # 합성식은 마지막에 둔다 — 앞의 재정렬 문장을 한정하는 각주다.
    if explain.score_mix is not None and explain.reorder_applied:
        notes.append(_mix_note(explain.score_mix))

    # "근거가 없다"는 기록이 비었을 때만 할 수 있는 말이다. 점수 근거가 없어도
    # 재정렬 기록이 있으면 그것이 이 곡이 이 자리에 온 이유다 — 여기서 버리면
    # 리랭킹만으로 올라온 곡을 설명할 수 없게 된다.
    dropped_only = bool(explain.dropped_paths()) and not parts
    if not parts and not notes and not dropped_only:
        return "기록된 근거가 없습니다."

    if not parts:
        head = ["1차 융합에서 탈락한 기여만 기록됨"] if dropped_only else []
        return " · ".join([*head, *notes])

    sentence = " · ".join(parts)

    # 1차 융합에서 탈락했다가 다시 들어온 것 자체가 설명이다.
    if explain.dropped_paths():
        sentence += " (1차 융합에서 탈락 후 재합류)"

    if notes:
        sentence += " → " + " · ".join(notes)

    return sentence
