"""
Eval set Pydantic schemas.

EvalQuery
    하나의 검색 평가 케이스. 한국어 질의 + 정답(positives) + 오답(negatives) +
    평가 시나리오 구분(tier) + 모달리티 포커스(modality_focus).
    - positives: top-10 안에 반드시 등장해야 하는 song_id (multi-label)
    - allowed: 원래 타깃은 아니지만 질의 설명에 맞아 팀이 정답으로 인정한 song_id.
      엄격 지표는 positives만, 확장 지표는 positives + allowed로 잰다 (이중 정답 구조)
    - negatives: top-10에 절대 등장하면 안 되는 song_id (함정)
    - negative_reason: 왜 함정인지 사람이 읽을 수 있는 메모
    - tier: vague | baseline | low_signal | misinformation
    - modality_focus: text | audio | image | multimodal
    - split: dev | test (튜닝은 dev, 최종 보고는 test)
    - label_status: 정답 라벨이 확정됐는지. tier와 직교한다.
    - query_set: 질의 출처. 통합 후에도 세트별 분리 집계를 가능하게 한다.

EvalSet
    EvalQuery 목록 컨테이너. version/created_at 메타데이터 포함.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

QueryCategory = Literal["mood", "artist", "lyrics", "scene", "genre", "mixed"]

# 평가 시나리오 구분
#   vague          : 모호한 자연어 검색 (Vague-Finder 핵심 가치 -- 주력 KPI)
#   baseline       : 단순 아티스트/장르 룩업 (사니티 체크, KPI에서 분리 측정)
#   low_signal     : 단서 부족 -- graceful handling 검증 (정답 없음)
#   misinformation : 사용자가 잘못된 정보 포함 -- 강건성 검증
QueryTier = Literal["vague", "baseline", "low_signal", "misinformation"]

# 어느 모달리티를 *주로* 검증하는 질의인지
#   text       : 가사/태그/메타 기반 (대부분)
#   audio      : 악기/사운드 텍스처 -- CLAP 임베딩 단독 효과 측정
#   image      : 앨범 커버 시각 속성 -- SigLIP2 임베딩 단독 효과 측정
#   multimodal : 둘 이상 모달리티 신호 동시 필요
ModalityFocus = Literal["text", "audio", "image", "multimodal"]

# 홀드아웃 구분. dev에서 튜닝하고 test로만 최종 보고한다.
# 필수 필드인 이유: 기본값을 두면 신규 질의가 조용히 dev로 쌓여 홀드아웃이
# 다시 0개가 된다 (v0.4에서 실제로 일어난 일).
QuerySplit = Literal["dev", "test"]

# 정답 라벨 상태. tier(시나리오 구분)와 직교한다.
#   labeled       : positives 확정. 정확도 집계 대상.
#   pending_cover : 앨범 표지 실물 확인 대기. 라벨이 부분적이거나 비어 있다.
#                   라벨 없이 집계에 넣으면 q118처럼 잘못된 정답으로 원인 추적이
#                   어려워지므로 정확도 집계에서 제외한다.
#   no_target     : 정답이 없는 것이 설계 의도. 개방형 추천 질의와 단서 극빈
#                   질의가 여기 해당하며, 시스템이 뻗지 않는지만 확인한다.
#   unreachable   : 정답은 알지만 현 코퍼스·검색 구성에서 후보 풀에 들어오지
#                   못한다. 난도 조절에 실패한 질의로, 집계에 넣으면 Hit가
#                   0으로 깔려 다른 질의의 개선을 가린다. 지우지 않고 남기는
#                   이유는 왜 실패했는지가 다음 사람에게 필요하기 때문이다.
#                   ⚠️ 코퍼스가 커지면 다시 판정해야 한다 — 순위는 코퍼스 상대적이다.
LabelStatus = Literal["labeled", "pending_cover", "no_target", "unreachable"]

# 질의 출처 세트. 통합 파일에서도 세트별로 분리 집계하기 위한 것으로,
# v0.5 기준선 비교는 query_set="v04" 부분집합으로만 수행한다.
#   v04         : 2026-08 3인 작성 60개 (v0.5 공식 기준선의 대상)
#   modality_v1 : 2026-08-24 모달리티 분리 효과 + 나무위키 사전 기준선 14개
#   clarify_v1  : 2026-09 재질문 측정용 신규 질의 (단계 2)
QuerySet = Literal["v04", "modality_v1", "clarify_v1"]

# 질의가 어떤 단서에 의존하는지. modality_v1의 type을 옮긴 것으로,
# 나무위키 크롤링 전후 비교처럼 단서 유형별 효과를 볼 때 사용한다.
ClueType = Literal[
    "external_context",  # 고유명사·외부맥락 (드라마/예능/애니 등)
    "audio_only",        # 청각 단서만
    "image_only",        # 시각 단서만
    "mixed_av",          # 이미지 + 오디오 혼합
    "sound_texture",     # 악기 편성·음색 묘사
    "title_structure",   # 제목의 글자 수/문자 종류/의미 기억
    "era_group",         # 시대 + 아티스트 유형 조합
    "composition",       # 곡 구성 (듀엣 역할 분담, 도입부 등)
    "lyric_fragment",    # 가사 조각 (정확/부분/음차)
]


class EvalQuery(BaseModel):
    query_id: str = Field(..., description="예: 'q001'")
    query: str = Field(..., description="사용자가 입력하는 자연어 질의(한국어)")
    category: QueryCategory = Field(..., description="질의 카테고리")
    tier: QueryTier = Field(
        default="vague",
        description="평가 시나리오 구분. baseline/low_signal/misinformation은 주력 KPI에서 분리 집계",
    )
    modality_focus: ModalityFocus = Field(
        default="text",
        description="이 질의가 주로 검증하는 모달리티. audio/image 질의로 CLAP/SigLIP2 단독 효과 측정",
    )
    split: QuerySplit = Field(
        ...,
        description="dev면 튜닝용, test면 홀드아웃. 기본값 없음 — 신규 질의가 조용히 dev로 쌓이는 것을 막는다",
    )
    query_set: QuerySet = Field(
        ...,
        description="질의 출처 세트. 기준선 비교는 이 값으로 부분집합을 잘라 수행한다",
    )
    label_status: LabelStatus = Field(
        default="labeled",
        description="정답 라벨 상태. labeled가 아니면 정확도 집계에서 제외",
    )
    clue_type: Optional[ClueType] = Field(
        default=None,
        description="질의가 의존하는 단서 유형 (선택). 단서 유형별 효과 분석에 사용",
    )
    positives: List[str] = Field(
        default_factory=list,
        description="top-10 안에 반드시 등장해야 하는 song_id (label_status=labeled일 때 1개 이상 필수)",
    )
    allowed: List[str] = Field(
        default_factory=list,
        description="허용 정답 song_id. 원래 타깃이 아니지만 질의 설명에 맞아 팀이 인정한 곡 — 확장 지표에서만 정답으로 센다",
    )
    negatives: List[str] = Field(
        default_factory=list,
        description="top-10에 절대 등장하면 안 되는 song_id (False Positive 방지)",
    )
    negative_reason: Optional[str] = Field(
        default=None, description="negatives가 왜 함정인지 사람이 읽을 수 있는 메모"
    )
    notes: Optional[str] = Field(default=None, description="작성자 메모 (선택)")

    @property
    def is_scorable(self) -> bool:
        """정확도 지표(Hit/MRR/nDCG) 집계에 넣어도 되는 질의인지.

        기준은 "정답이 있고, 그 정답에 도달할 수 있는가"다. pending_cover라도
        오디오·메타데이터 축이 교차검증된 부분 라벨이 있으면 집계에 넣는다
        (m401/m402가 그런 경우다). 정답이 없는 질의를 넣으면 Hit가 0으로
        깔려 다른 질의의 개선을 가린다.
        """
        return bool(self.positives) and self.label_status not in ("no_target", "unreachable")

    @property
    def has_verified_label(self) -> bool:
        """정답 라벨이 전부 확정됐는지. pending_cover는 시각 축이 미확정이다."""
        return self.label_status == "labeled"

    @model_validator(mode="after")
    def _validate(self) -> "EvalQuery":
        # 1) positives 와 negatives 가 겹치면 정답=오답 모순
        overlap = set(self.positives) & set(self.negatives)
        if overlap:
            raise ValueError(
                f"[{self.query_id}] positives/negatives 겹침: {sorted(overlap)}"
            )
        # 1-1) 허용 정답은 원래 타깃·함정과 겹칠 수 없다
        overlap = set(self.allowed) & (set(self.positives) | set(self.negatives))
        if overlap:
            raise ValueError(
                f"[{self.query_id}] allowed가 positives/negatives와 겹침: {sorted(overlap)}"
            )
        # 2) negatives 가 있으면 negative_reason 필수
        if self.negatives and not self.negative_reason:
            raise ValueError(
                f"[{self.query_id}] negatives 있을 땐 negative_reason 필수"
            )
        # 3) 라벨이 확정된 질의는 positives 1개 이상 필수.
        #    비워야 한다면 label_status를 pending_cover / no_target으로 명시할 것.
        if self.label_status == "labeled" and not self.positives:
            raise ValueError(
                f"[{self.query_id}] positives 필수 "
                f"(비우려면 label_status=pending_cover 또는 no_target)"
            )
        # 4) 정답이 없는 것이 설계 의도인데 positives가 있으면 둘 중 하나가 틀렸다
        if self.label_status == "no_target" and self.positives:
            raise ValueError(
                f"[{self.query_id}] label_status=no_target인데 positives가 있음"
            )
        return self


class EvalSet(BaseModel):
    version: str = Field(..., description="시맨틱 버전 (예: '0.1.0')")
    created_at: str = Field(..., description="ISO8601 날짜 (예: '2026-05-24')")
    description: Optional[str] = None
    queries: List[EvalQuery] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvalSet":
        ids = [q.query_id for q in self.queries]
        dup = {x for x in ids if ids.count(x) > 1}
        if dup:
            raise ValueError(f"query_id 중복: {sorted(dup)}")
        return self

    def by_category(self, category: QueryCategory) -> List[EvalQuery]:
        return [q for q in self.queries if q.category == category]

    def by_tier(self, tier: QueryTier) -> List[EvalQuery]:
        return [q for q in self.queries if q.tier == tier]

    def by_modality_focus(self, focus: ModalityFocus) -> List[EvalQuery]:
        return [q for q in self.queries if q.modality_focus == focus]

    def by_split(self, split: QuerySplit) -> List[EvalQuery]:
        return [q for q in self.queries if q.split == split]

    def by_query_set(self, query_set: QuerySet) -> List[EvalQuery]:
        return [q for q in self.queries if q.query_set == query_set]

    def scorable(self) -> List[EvalQuery]:
        """정확도 집계 대상만. pending_cover / no_target은 빠진다."""
        return [q for q in self.queries if q.is_scorable]

    def category_distribution(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(q.category for q in self.queries))

    def tier_distribution(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(q.tier for q in self.queries))

    def modality_distribution(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(q.modality_focus for q in self.queries))

    def split_distribution(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(q.split for q in self.queries))

    def query_set_distribution(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(q.query_set for q in self.queries))
