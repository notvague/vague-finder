from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.backend.schemas.explain import SearchExplainOut, TrackExplain
from src.backend.schemas.query import QueryAnalysis


# 거절·질문 한도. 라우트가 종료 판정에 같은 값을 써야 해서 상수로 노출한다.
# 숫자가 두 곳에 흩어지면 스키마는 거부하는데 라우트는 질문을 계속 내주는
# 상태가 된다 — 실제로 그렇게 422가 났다.
MAX_REJECTED_IDS = 20
MAX_ASKED_SLOTS = 2
MAX_TURNS = 2


class ClarifyAnswer(BaseModel):
    """1턴에서 물은 슬롯에 대한 사용자 응답."""

    slot: str = Field(..., min_length=1, description="답변 대상 슬롯 (예: vocal_gender)")
    value: str = Field(default="", description="사용자가 고른 값. skipped면 빈 문자열")
    skipped: bool = Field(
        default=False,
        description="'잘 모르겠어요' 선택 여부. True면 슬롯을 채우지 않는다",
    )

    @model_validator(mode="after")
    def clear_value_when_skipped(self) -> "ClarifyAnswer":
        # 스킵인데 값이 실려 오면 병합 단계에서 잘못된 슬롯을 채우게 된다.
        if self.skipped:
            self.value = ""
        return self


class ClarifyOption(BaseModel):
    """재질문 선택지. count는 '왜 이걸 묻는지' 근거로 화면에 쓸 수 있다."""

    value: str = Field(..., description="후보들의 실제 값에서 뽑은 선택지")
    count: int = Field(default=0, ge=0, description="이 값을 가진 후보 수")


class ClarifyQuestion(BaseModel):
    """규칙 기반으로 고른 재질문 한 개."""

    slot: str = Field(..., min_length=1, description="물어볼 슬롯")
    question: str = Field(..., min_length=1, description="사용자에게 보일 질문 문구")
    options: List[ClarifyOption] = Field(default_factory=list)
    reason: str = Field(
        default="",
        description="이 슬롯을 고른 근거 (예: '후보 10곡이 남성 6 / 여성 4로 갈려서')",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="자연어 텍스트 형태의 검색 질의어")
    top_k: int = Field(default=10, ge=1, le=50, description="반환할 검색 결과의 개수")
    use_rerank: bool = Field(default=True, description="Cross-Encoder 리랭킹 사용 여부")
    candidate_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="리랭킹 전에 유지할 후보 수. 생략하면 max(top_k*3, 30)",
    )
    explain: bool = Field(
        default=False,
        description=(
            "선정 근거를 함께 받는다. 결과 곡마다 explain이 채워지고 응답에 실행 "
            "기록 요약이 붙는다. 순위와 점수는 달라지지 않는다 — 기록기는 쓰기 "
            "전용이고 랭킹 계산이 그것을 읽지 않는다"
        ),
    )

    # --- 재질문(2턴) ---
    # 세션 저장소를 두지 않고 QueryAnalysis를 클라이언트가 왕복시킨다.
    # 서버는 stateless로 남는다.
    prior_analysis: Optional[QueryAnalysis] = Field(
        default=None,
        description="1턴 응답의 analysis를 그대로 되돌려준다. 있으면 질의를 재분석하지 않는다",
    )
    answers: List[ClarifyAnswer] = Field(
        default_factory=list,
        max_length=2,
        description=(
            "지금까지의 재질문 응답 누적. 2턴 누적 호환도 계산에 답변 이력이 "
            "모두 필요해서 단수 answer가 아니라 리스트다"
        ),
    )
    asked_slots: List[str] = Field(
        default_factory=list,
        max_length=2,
        description="직전 응답의 asked_slots를 그대로 되돌려준다. 같은 슬롯 재질문 방지",
    )
    previous_candidate_ids: List[str] = Field(
        default_factory=list,
        max_length=30,
        description="직전 응답의 candidate_ids를 그대로 되돌려준다",
    )
    rejected_ids: List[str] = Field(
        default_factory=list,
        max_length=MAX_REJECTED_IDS,
        description=(
            "'이 중에는 없어요'로 거절한 곡 id 누적. 후보 풀에서 제외된다. "
            "하드 제외는 이 필드뿐이다 — 답변 불일치 후보는 지우지 않는다"
        ),
    )
    turn: int = Field(
        default=1,
        ge=1,
        description="직전 응답의 turn을 그대로 되돌려준다. 서버가 +1해서 내려보낸다",
    )

    @model_validator(mode="after")
    def validate_candidate_k(self) -> "SearchRequest":
        if self.candidate_k is not None and self.candidate_k < self.top_k:
            raise ValueError("candidate_k는 top_k보다 크거나 같아야 합니다.")
        return self

    @model_validator(mode="after")
    def validate_clarify_turn(self) -> "SearchRequest":
        # answers만 오면 어떤 분석에 병합할지 알 수 없다. 조용히 무시하면
        # 사용자는 답했는데 반영이 안 된 것처럼 보이므로 명시적으로 막는다.
        # rejected_ids는 분석 없이도 제외만 하면 되므로 막지 않는다(라우터가 경고).
        if self.answers and self.prior_analysis is None:
            raise ValueError(
                "answers를 보내려면 prior_analysis도 함께 보내야 합니다."
            )
        return self

    @model_validator(mode="after")
    def validate_unique_rejections(self) -> "SearchRequest":
        # 같은 id가 쌓이면 max_length=20에 금방 걸려 실제 거절 곡이 잘린다.
        if len(set(self.rejected_ids)) != len(self.rejected_ids):
            raise ValueError("rejected_ids에 중복된 곡 id가 있습니다.")
        return self


class LyricSurfaceMatch(BaseModel):
    """가사 표면 일치가 **무엇으로 어떻게** 성립했는지.

    부스트가 어떤 단서에 과도하게 걸리는지 보려면 점수만으로는 부족하다.
    짧은 조각이 여러 곡에 걸려도 점수는 단서 확신도 그대로이기 때문이다
    (q211: "위험하다" 네 글자가 다른 곡에 걸려 ×8.0을 받았다).

    **계측용이다. 랭킹에 쓰이지 않는다.**
    """

    clue_text: str = Field(..., description="사용자가 기억한 원래 구절")
    clue_kind: str = Field(
        ...,
        description="단서 종류(verbatim/partial/phonetic). **일치 유형과 다르다**",
    )
    matched_phrase: str = Field(
        ..., description="실제로 일치한 정규화 표기. 원문이 아니라 변형일 수 있다"
    )
    is_variant: bool = Field(
        default=False, description="원문이 아니라 음차 변형 후보로 맞았는가"
    )
    normalized_length: int = Field(
        ..., description="정규화 후 길이. 짧을수록 우연히 겹칠 확률이 높다"
    )
    match_type: str = Field(
        ...,
        description=(
            "검색이 판정한 일치 유형(exact/fuzzy/phonetic). partial 단서가 "
            "exact로 판정되는 경우가 있어 clue_kind와 따로 남긴다"
        ),
    )
    confidence: float = Field(..., description="분석 모델이 이 단서에 매긴 확신도")
    corpus_match_count: int = Field(
        ...,
        description=(
            "**후보 절단 전 전체 가사 코퍼스**에서 이 구절이 맞는 곡 수. "
            "1이면 그 곡에만 있는 구절이고, 크면 흔한 조각이다"
        ),
    )


class MatchingTrack(BaseModel):
    id: str = Field(..., description="곡의 고유 ID (Melon ID 등)")
    score: float = Field(..., description="최종 검색 점수")
    retrieval_score: Optional[float] = Field(
        default=None,
        description="리랭킹 직전 RRF + 규칙 기반 부스팅 점수",
    )
    rerank_score: Optional[float] = Field(
        default=None,
        description="Cross-Encoder가 계산한 0~1 관련도 점수",
    )

    # Metadata
    title: str = Field(..., description="곡 제목")
    artist: Optional[str] = Field(None, description="아티스트명")
    album: Optional[str] = Field(None, description="앨범 제목")
    release_date: Optional[str] = None
    genre: Optional[str] = None
    type: Optional[str] = Field(None, description="아티스트 구성/형태 (솔로, 그룹, 듀엣 등)")
    vocal_gender: Optional[str] = None
    artist_types: List[str] = Field(default_factory=list)

    # Semantic Analysis & Vibe
    search_style_summary: Optional[str] = Field(None, description="곡 분위기 요약")
    mood_tags: List[str] = Field(default_factory=list)
    time_weather_tags: List[str] = Field(default_factory=list)
    place_activity_tags: List[str] = Field(default_factory=list)
    emotion_tags: List[str] = Field(default_factory=list)
    vibe_tags: List[str] = Field(default_factory=list)
    relation_context_tags: List[str] = Field(default_factory=list)
    color_tags: List[str] = Field(default_factory=list)
    sound_tags: List[str] = Field(default_factory=list)
    melon_playlist_tags: List[str] = Field(default_factory=list)
    visual_imagery: List[str] = Field(default_factory=list)

    # Lyrics Data
    lyrics_highlight: Optional[str] = None
    lyrics_summary: Optional[str] = None
    lyric_match_type: Optional[
        Literal["exact", "fuzzy", "phonetic"]
    ] = Field(
        default=None,
        description="full_lyrics 표면 검색에서 확인된 가사 일치 유형",
    )
    lyric_match_detail: Optional[LyricSurfaceMatch] = Field(
        default=None,
        description="가사 표면 일치의 근거. 계측용이며 랭킹에 쓰이지 않는다",
    )
    lyric_match_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "가사 표면 검색 점수. **유형마다 의미가 다르다** — exact/phonetic은 "
            "표기 정규화(대소문자·공백·구두점 제거) 후 부분문자열로 발견된 경우라 "
            "분석 단서의 확신도이고, fuzzy는 문자 유사도 × 확신도다"
        ),
    )

    # Community & Popularity
    sentiment_summary: Optional[str] = None
    fans_tags: List[str] = Field(default_factory=list)
    major_emotion: Optional[str] = None
    fame: Optional[str] = None

    # Links
    melon_url: Optional[str] = None
    youtube_url: Optional[str] = None
    cover_url: Optional[str] = None

    # --- 선정 근거 ---
    # 요청에 explain=true를 준 경우에만 채워진다. None은 "근거가 없다"가 아니라
    # "기록을 켜지 않았다"는 뜻이다.
    explain: Optional[TrackExplain] = Field(
        default=None,
        description="이 곡이 이 자리에 온 근거. 요청에 explain=true를 주면 채워진다",
    )

    @field_validator("artist", "genre", "type", "vocal_gender", mode="before")
    @classmethod
    def coerce_list_to_str(cls, value):
        if isinstance(value, list):
            return ", ".join(str(x) for x in value) if value else None
        return value



class SearchResponse(BaseModel):
    results: List[MatchingTrack] = Field(
        default_factory=list,
        description="검색된 상위 K개의 음악 리스트",
    )
    total_found: int = Field(default=0, description="발견된 곡의 수")

    # --- 재질문(2턴) ---
    # analysis를 Optional로 둔 이유: 필수로 만들면 기존 호출부와 테스트가
    # 전부 깨진다. 라우터는 항상 채워서 내려보낸다.
    analysis: Optional[QueryAnalysis] = Field(
        default=None,
        description="이번 턴에 사용된 질의 분석. 클라이언트가 보관했다가 다음 턴에 되돌려준다",
    )
    clarify: Optional[ClarifyQuestion] = Field(
        default=None,
        description="되물을 것이 있으면 채워진다. None이면 대화 종료",
    )
    asked_slots: List[str] = Field(
        default_factory=list,
        description="지금까지 물어본 슬롯 누적. 다음 요청에 그대로 되돌려준다",
    )
    candidate_ids: List[str] = Field(
        default_factory=list,
        description=(
            "이번 턴 후보 풀의 id (results보다 넓다). 다음 요청에 "
            "previous_candidate_ids로 되돌려준다"
        ),
    )
    rejected_ids: List[str] = Field(
        default_factory=list,
        description="지금까지 거절된 곡 id 누적. 다음 요청에 그대로 되돌려준다",
    )
    turn: int = Field(default=1, ge=1, description="현재 턴 번호")

    explain: Optional[SearchExplainOut] = Field(
        default=None,
        description=(
            "이번 턴 실행 기록 요약(경로 가중치·리랭커·재정렬 단계). "
            "요청에 explain=true를 준 경우에만 채워진다"
        ),
    )
