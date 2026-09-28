"""응답으로 내보내는 검색 실행 기록.

`src/retrieval/explain.py`의 dataclass는 파이프라인이 쓰는 내부 표현이다. 그것을
그대로 응답 모델로 쓰지 않는 이유는 둘이다.

1. **말할 수 있는 것과 없는 것의 경계를 여기서 고정한다.** 모델이 스스로 쓴 문장은
   검증된 근거와 같은 목록에 넣지 않고, 라벨을 붙여 따로 내보낸다.
2. 내부 표현은 계측 편의를 위해 바뀔 수 있다. 화면이 그 구조에 직접 붙으면
   계측을 고칠 때마다 화면이 깨진다.

여기서 만드는 문장은 모두 기록에서 나온다. LLM을 쓰지 않으므로 기록에 없는 주장은
만들 수 없다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.retrieval.explain import (
    LYRIC_MATCH_LABELS,
    LYRIC_ORDER_RULES,
    PATH_LABELS,
    REORDER_NOT_ATTEMPTED,
    REORDER_STAGE_LABELS,
    RULE_LABELS,
    PathContribution,
    ScoreAdjustment,
    ScoreMix,
    SearchExplain,
    SongExplain,
    render_ko,
)

# 모델의 문장에 붙는 경고. 화면이 이 문구를 직접 쓰게 해서, "검증된 근거"와 섞여
# 표시되는 경로를 만들지 않는다.
MODEL_NOTES_CAPTION = (
    "모델이 스스로 말한 이유입니다. 실행 기록으로 검증되지 않았고, "
    "최종 순위의 근거와 다를 수 있습니다"
)

# 질의 분석이 규칙 폴백이었을 때 화면이 쓸 문구. 화면이 직접 짓지 않게 서버가 준다.
#
# "실패"라고만 하면 검색이 안 됐다는 뜻으로 읽힌다. 실제로는 분석만 규칙으로
# 내려앉고 검색은 그대로 돌았으므로, 무엇이 대신 쓰였는지까지 말한다.
ANALYSIS_FALLBACK_LABEL = "질의 분석 실패 — 규칙 기반 분석으로 검색"

# 근거가 **순위를 만든 것**인지, 순위가 정해진 뒤 덧붙인 설명인지.
#
# 이 구분이 없으면 뒤에 찾아낸 문장을 "1위가 된 이유"라고 말하게 된다. 가사
# 일치는 기록되지만 그 기여가 최종 점수에 남지 않는 경우가 있다(1차 절단에서
# 탈락). 그때 인용은 사실이지만 순위의 근거는 아니다.
EVIDENCE_RANKING = "ranking"
EVIDENCE_SUPPLEMENTARY = "supplementary"
EVIDENCE_KIND_LABELS: Dict[str, str] = {
    EVIDENCE_RANKING: "이 순위를 만든 근거",
    EVIDENCE_SUPPLEMENTARY: "순위에 쓰이지 않은 보충 설명",
}


class PathContributionOut(BaseModel):
    """한 검색 경로가 이 곡을 얼마나 밀어 올렸는지."""

    path: str = Field(..., description="경로 코드")
    label: str = Field(..., description="경로 이름 (화면용)")
    rank: int = Field(..., description="그 경로 안에서의 순위")
    delta: float = Field(..., description="융합 점수에 실제로 더해진 양")
    detail: str = Field(default="", description="가중치·배수 같은 부가 정보")
    dropped: bool = Field(
        default=False,
        description=(
            "1차 융합 절단에서 탈락해 **최종 점수에 반영되지 않은** 기여. "
            "합계에 넣으면 설명이 실제 점수보다 커진다"
        ),
    )

    @classmethod
    def of(cls, item: PathContribution) -> "PathContributionOut":
        return cls(
            path=item.path,
            label=item.label,
            rank=item.rank,
            delta=item.delta,
            detail=item.detail,
            dropped=item.dropped,
        )


class ScoreAdjustmentOut(BaseModel):
    """점수를 더하거나 뺀 규칙. delta가 음수면 감점이다."""

    rule: str = Field(..., description="규칙 코드")
    label: str = Field(..., description="규칙 이름 (화면용)")
    delta: float
    detail: str = ""

    @classmethod
    def of(cls, item: ScoreAdjustment) -> "ScoreAdjustmentOut":
        return cls(
            rule=item.rule,
            label=RULE_LABELS.get(item.rule, item.rule),
            delta=item.delta,
            detail=item.detail,
        )


class OrderRuleOut(BaseModel):
    """점수와 무관하게 **순서만** 바꾼 규칙."""

    rule: str
    label: str
    detail: str = ""


class ScoreMixOut(BaseModel):
    """리랭킹 최종 점수를 무엇으로 합성했는가.

    이것이 없으면 "리랭커 점수가 가장 높은 곡이 1위"라는 잘못된 인상을 준다 —
    실제로는 검색 점수가 함께 섞이고, CE 최고점 곡이 1위가 아닌 경우가 있었다.
    """

    backend: str
    weight: float = Field(..., description="리랭커 쪽에 실린 최종 가중치")
    configured_weight: float = Field(..., description="설정값. 신뢰도로 깎이기 전")
    confidence: float = Field(..., description="가중치를 깎은 비율. 1.0이면 깎지 않음")
    rerank_component: float
    retrieval_component: float
    final: float
    rerank_normalized: bool = Field(
        ...,
        description="리랭커 쪽 값이 정규화된 값인지. False면 원점수를 그대로 섞었다",
    )
    strategy: str = Field(default="", description="정렬 전략")
    reordered: bool = Field(
        default=True,
        description=(
            "이 곡의 **순서**가 합성 점수로 결정됐는가. False면 점수는 합성됐지만 "
            "순서는 검색 순서를 그대로 유지했다"
        ),
    )
    detail: str = ""

    @classmethod
    def of(cls, mix: ScoreMix) -> "ScoreMixOut":
        return cls(
            backend=mix.backend,
            weight=mix.weight,
            configured_weight=mix.configured_weight,
            confidence=mix.confidence,
            rerank_component=mix.rerank_component,
            retrieval_component=mix.retrieval_component,
            final=mix.final,
            rerank_normalized=mix.rerank_normalized,
            strategy=mix.strategy,
            reordered=mix.reordered,
            detail=mix.detail,
        )


class LyricEvidenceOut(BaseModel):
    """가사 원문 인용 하나.

    `lyric_match.matched_phrase`는 정규화 표기라 인용문이 될 수 없다. 이것은 그
    일치를 원문으로 되짚어 **글자 그대로** 떠 온 구간이다. 되짚지 못하면 이
    목록은 비어 있다 — 빈 자리를 그럴듯한 문장으로 채우지 않는다.
    """

    song_id: str = Field(..., description="이 인용이 속한 곡. 다른 곡 문장이 섞이지 않게 함께 싣는다")
    source: str = Field(..., description="어느 저장소의 어느 필드인가")
    quote: str = Field(..., description="원문 그대로의 구간")
    line: str = Field(..., description="그 구간이 걸친 줄 전체")
    char_start: int = Field(..., description="원문에서의 시작 위치")
    char_end: int = Field(..., description="원문에서의 끝 위치(제외)")
    data_version: str = Field(
        ...,
        description=(
            "이 곡 가사의 판(sha256 앞 16자). 스냅샷이 바뀌면 같은 위치가 다른 "
            "문장을 가리키므로 인용에는 판이 함께 붙어야 한다"
        ),
    )
    matched_phrase: str = Field(..., description="무엇을 되짚었는가(정규화 표기)")
    kind: str = Field(
        ...,
        description=(
            f"{EVIDENCE_RANKING}이면 이 근거가 최종 점수에 반영됐다. "
            f"{EVIDENCE_SUPPLEMENTARY}면 사실이지만 순위를 만들지 않았다"
        ),
    )
    kind_label: str = Field(..., description="kind의 화면용 이름")


class TrackExplain(BaseModel):
    """곡 한 개가 이 자리에 온 근거."""

    song_id: str
    summary: str = Field(
        ...,
        description="기록만으로 만든 한 문장. LLM을 쓰지 않으므로 기록에 없는 말은 없다",
    )

    paths: List[PathContributionOut] = Field(
        default_factory=list,
        description="최종 점수에 반영된 경로 기여 (기여 큰 순)",
    )
    dropped_paths: List[PathContributionOut] = Field(
        default_factory=list,
        description="1차 융합에서 탈락해 최종 점수에 반영되지 않은 기여",
    )
    adjustments: List[ScoreAdjustmentOut] = Field(
        default_factory=list,
        description="점수를 더하거나 뺀 규칙 (|증감| 큰 순)",
    )
    order_rules: List[OrderRuleOut] = Field(
        default_factory=list,
        description="점수와 무관하게 순서만 바꾼 규칙",
    )

    path_total: float = Field(..., description="반영된 경로 기여 합. fused_score와 맞아야 한다")
    adjustment_total: float = 0.0
    fused_score: Optional[float] = None
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None

    rank_before_rerank: Optional[int] = None
    rank_after_rerank: Optional[int] = None
    group_rank_before: Optional[int] = Field(
        default=None,
        description="모델이 본 묶음 안에서의 순위(전). 모델에 귀속할 수 있는 유일한 변화다",
    )
    group_rank_after: Optional[int] = Field(
        default=None,
        description="묶음 안에서의 순위(후). 규칙이 개입하기 전 리랭킹 단계의 순서다",
    )
    behind_protected: int = Field(
        default=0,
        description=(
            "이 곡 앞에 배치된 가사 보호 곡 수. **하락 폭이 아니다** — "
            "보호 곡이 원래 앞에 있었으면 순위는 그대로다"
        ),
    )

    reorder_applied: bool = Field(
        default=False,
        description="이 곡의 최종 순위에 리랭커 결과가 반영됐는가",
    )
    reorder_stage: str = Field(default=REORDER_NOT_ATTEMPTED, description="요청 전체의 재정렬 단계")
    reorder_stage_label: str = Field(default="", description="단계 이름 (화면용)")

    lyric_match: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "가사 표면 일치의 근거 — 일치 구절·단서 종류·정규화 길이·"
            "코퍼스에서 겹치는 곡 수. **계측용이며 랭킹에 쓰이지 않는다**"
        ),
    )
    lyric_match_label: str = Field(
        default="",
        description=(
            "lyric_match.match_type의 화면용 이름. 화면이 이 문구를 직접 짓지 않게 "
            "서버가 내보낸다 — '가사 원문에 그대로 있다'처럼 실제보다 강한 말이 "
            "생기는 자리였다"
        ),
    )
    lyric_boost_rank_before: Optional[int] = Field(
        default=None, description="가사 부스트 직전 융합 순위"
    )
    lyric_boost_rank_after: Optional[int] = Field(
        default=None, description="가사 부스트 직후 융합 순위"
    )

    score_mix: Optional[ScoreMixOut] = None

    model_notes: List[str] = Field(
        default_factory=list,
        description=(
            "모델이 스스로 쓴 이유 문장. **검증된 기록이 아니다** — "
            "summary에는 들어가지 않는다"
        ),
    )
    model_notes_caption: str = Field(
        default=MODEL_NOTES_CAPTION,
        description="model_notes를 보여줄 때 함께 표시할 경고 문구",
    )

    evidence: List[LyricEvidenceOut] = Field(
        default_factory=list,
        description=(
            "주장을 뒷받침하는 원문 인용. 되짚지 못했으면 비어 있다 — "
            "없는 근거를 지어내지 않는다"
        ),
    )


class FailedPathOut(BaseModel):
    """예외로 죽어 빈 결과로 대체된 경로.

    `modality_weights`에 값이 남아 있어도 이 목록에 있으면 그 경로의 기여는 없다.
    둘을 함께 보지 않으면 "가중치는 0.20인데 기여가 한 줄도 없다"를 잘못 읽는다.
    """

    path: str = Field(..., description="경로 코드")
    label: str = Field(..., description="화면용 문구")
    reason: str = Field(..., description="예외 종류와 메시지")


class SearchExplainOut(BaseModel):
    """요청 한 번의 실행 기록 요약. 곡별 근거는 각 결과의 explain에 붙는다."""

    query: str = ""
    modality_weights: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "이 요청에 적용된 경로 가중치. 0이면 그 경로는 실행되지 않았다. "
            "**0이 아니라고 그 경로가 끝까지 돌았다는 뜻은 아니다** — "
            "failed_paths를 함께 봐야 한다"
        ),
    )
    failed_paths: List[FailedPathOut] = Field(
        default_factory=list,
        description="예외로 죽어 빈 결과로 대체된 경로. 비어 있으면 모두 정상이었다",
    )
    analysis_fallback: bool = Field(
        default=False,
        description=(
            "질의 분석이 Gemini가 아니라 규칙 폴백이었는가. 검색은 그대로 "
            "진행되므로 결과만 봐서는 구분되지 않는다"
        ),
    )
    analysis_fallback_label: str = Field(
        default=ANALYSIS_FALLBACK_LABEL,
        description="analysis_fallback이 참일 때 화면이 쓸 문구",
    )
    reranker: str = "none"
    reorder_stage: str = REORDER_NOT_ATTEMPTED
    reorder_stage_label: str = ""
    rerank_error: str = Field(
        default="",
        description="라우터가 예외를 잡았으면 그 사유. 비어 있으면 예외는 없었다",
    )
    rerank_calls: int = Field(default=0, description="리랭커를 부른 횟수")
    rerank_calls_applied: int = Field(
        default=0,
        description="그중 모델이 실제로 순서를 정한 횟수. 호출 수 ≠ 성공 수다",
    )

    @classmethod
    def of(cls, record: SearchExplain) -> "SearchExplainOut":
        stage = record.reorder_stage
        return cls(
            query=record.query,
            modality_weights=dict(record.modality_weights),
            failed_paths=[
                FailedPathOut(
                    path=path,
                    label=f"{PATH_LABELS.get(path, path)} 경로 실패",
                    reason=reason,
                )
                for path, reason in record.failed_paths.items()
            ],
            analysis_fallback=record.analysis_fallback,
            reranker=record.reranker,
            reorder_stage=stage,
            reorder_stage_label=REORDER_STAGE_LABELS.get(stage, stage),
            rerank_error=record.rerank_error,
            rerank_calls=len(record.rerank_runs),
            rerank_calls_applied=record.rerank_runs_applied,
        )


def evidence_kind(explain: SongExplain) -> str:
    """이 곡의 가사 근거가 **순위를 만들었는가.**

    둘 중 하나면 랭킹 근거다.

    1. 최종 점수에 남은 `lyrics_surface` 기여가 **0이 아니다**
    2. 가사 보호 규칙이 이 곡의 **자리를 바꿨다**

    **경로가 기록됐다는 것만으로는 부족하다.** `_fuse_add`는 더한 양이 0이어도
    기여를 남긴다 — 부스트를 꺼 둔 대조 실험(`LYRIC_BOOST_SCALE=0`)이 그렇다.
    그 경우 가사는 후보 유입에만 쓰였고 순위는 다른 것이 만들었으므로,
    인용을 "이 순위를 만든 근거"라고 하면 하지 않은 일을 말하게 된다.

    1차 절단에서 탈락한 기여(`dropped`)도 점수에 없으므로 보충 설명이다 —
    있었던 일이지만 이 순위의 이유는 아니다.
    """
    scored = any(
        path.path == "lyrics_surface" and path.delta != 0.0
        for path in explain.live_paths()
    )
    protected = any(rule.rule in LYRIC_ORDER_RULES for rule in explain.order_rules)
    return EVIDENCE_RANKING if (scored or protected) else EVIDENCE_SUPPLEMENTARY


def to_track_explain(
    explain: SongExplain,
    *,
    reorder_stage: str = REORDER_NOT_ATTEMPTED,
    max_items: int = 3,
    evidence: Optional[List[Any]] = None,
) -> TrackExplain:
    """곡 하나의 기록을 응답 모델로 옮긴다.

    reorder_stage는 요청 전체에서 파생한 값이라 밖에서 받는다. 이것 없이는
    "리랭킹이 N위 유지"처럼 하지 않은 일을 말하게 된다 — 리랭킹을 끈 경우와
    실패한 경우에도 전후 순위는 기록되기 때문이다.
    """
    return TrackExplain(
        song_id=explain.song_id,
        summary=render_ko(explain, max_items=max_items, reorder_stage=reorder_stage),
        paths=[
            PathContributionOut.of(item)
            for item in sorted(explain.live_paths(), key=lambda p: p.delta, reverse=True)
        ],
        dropped_paths=[PathContributionOut.of(item) for item in explain.dropped_paths()],
        adjustments=[
            ScoreAdjustmentOut.of(item)
            for item in sorted(
                explain.adjustments, key=lambda a: abs(a.delta), reverse=True
            )
        ],
        order_rules=[
            OrderRuleOut(
                rule=rule.rule,
                label=RULE_LABELS.get(rule.rule, rule.rule),
                detail=rule.detail,
            )
            for rule in explain.order_rules
        ],
        path_total=explain.path_total(),
        adjustment_total=explain.adjustment_total(),
        fused_score=explain.fused_score,
        retrieval_score=explain.retrieval_score,
        rerank_score=explain.rerank_score,
        rank_before_rerank=explain.rank_before_rerank,
        rank_after_rerank=explain.rank_after_rerank,
        group_rank_before=explain.group_rank_before,
        group_rank_after=explain.group_rank_after,
        behind_protected=explain.behind_protected,
        reorder_applied=explain.reorder_applied,
        reorder_stage=reorder_stage,
        reorder_stage_label=REORDER_STAGE_LABELS.get(reorder_stage, reorder_stage),
        lyric_match=dict(explain.lyric_match) if explain.lyric_match else None,
        lyric_match_label=LYRIC_MATCH_LABELS.get(
            (explain.lyric_match or {}).get("match_type", ""), ""
        ),
        lyric_boost_rank_before=explain.lyric_boost_rank_before,
        lyric_boost_rank_after=explain.lyric_boost_rank_after,
        score_mix=(
            ScoreMixOut.of(explain.score_mix) if explain.score_mix is not None else None
        ),
        model_notes=list(explain.model_notes),
        evidence=[
            LyricEvidenceOut(
                song_id=item.song_id,
                source=item.source,
                quote=item.quote,
                line=item.line,
                char_start=item.char_start,
                char_end=item.char_end,
                data_version=item.data_version,
                matched_phrase=item.matched_phrase,
                kind=kind,
                kind_label=EVIDENCE_KIND_LABELS[kind],
            )
            for kind in [evidence_kind(explain)]
            for item in (evidence or [])
            # 다른 곡의 문장이 섞이는 사고를 **여기서 한 번 더** 막는다.
            # 값이 맞더라도 이 검사가 있으면 나중에 배선이 어긋날 때 드러난다.
            if item.song_id == explain.song_id
        ],
    )
