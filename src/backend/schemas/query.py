from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class ModalityWeights(BaseModel):
    text: float = Field(default=0.6, ge=0.0, le=1.0, description="KoE5 + BM25 텍스트 경로 가중치")
    image: float = Field(default=0.2, ge=0.0, le=1.0, description="SigLIP2 이미지 경로 가중치")
    audio: float = Field(default=0.2, ge=0.0, le=1.0, description="CLAP 오디오 경로 가중치")

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        # Gemini가 합이 정확히 1.0이 아닌 값을 반환할 수 있으므로
        # 필드 유효성 검사 전에 정규화한다.
        if isinstance(data, dict):
            t = float(data.get("text", 0.6))
            i = float(data.get("image", 0.2))
            a = float(data.get("audio", 0.2))
            total = t + i + a
            if total > 0 and abs(total - 1.0) > 0.001:
                data = dict(data)
                data["text"] = round(t / total, 4)
                data["image"] = round(i / total, 4)
                data["audio"] = round(a / total, 4)
        return data


class TitleConstraints(BaseModel):
    script: Optional[
        Literal["latin", "hangul", "hanja", "numeric", "mixed"]
    ] = Field(
        default=None,
        description=(
            "사용자가 기억하는 제목의 문자 종류. "
            "영문=latin, 한글=hangul, 한자=hanja, 숫자=numeric."
        ),
    )

    char_count: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="공백/문장부호를 제외한 제목 글자 수 단서.",
    )

    word_count: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="제목 단어 수 단서.",
    )

    contains_number: Optional[bool] = Field(
        default=None,
        description="제목에 숫자가 포함되었다는 명시적 기억.",
    )

    repeated_char: Optional[bool] = Field(
        default=None,
        description="TT처럼 동일 문자가 반복되는 제목인지.",
    )

    repeated_word: Optional[bool] = Field(
        default=None,
        description="High High처럼 같은 단어가 반복되는 제목인지.",
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="해당 제목 구조 기억의 신뢰도.",
    )


class LyricClue(BaseModel):
    """사용자가 제공한 가사 관련 단서 한 개.

    verbatim/partial/phonetic은 full_lyrics 표면 검색에 사용하고 semantic은
    의미 기반 dense 검색에만 사용한다.
    """

    text: str = Field(..., min_length=1, max_length=300)
    kind: Literal["verbatim", "partial", "phonetic", "semantic"]
    variants: List[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "phonetic 단서가 실제 가사에서 표기됐을 가능성이 있는 원어 후보. "
            "사용자가 적은 text는 그대로 보존하고 후보 표기만 여기에 둔다."
        ),
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["rule", "model", "fallback"] = "model"


class ReleaseEra(BaseModel):
    start_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="사용자가 기억하는 발매 시기 시작 연도",
    )
    end_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="사용자가 기억하는 발매 시기 종료 연도",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="발매 시기 기억의 신뢰도",
    )


class ArtistTypeClue(BaseModel):
    values: List[
        Literal["솔로", "그룹", "듀오", "밴드"]
    ] = Field(default_factory=list)
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="가수 구성 기억의 신뢰도",
    )


class TitleMeaningClue(BaseModel):
    """제목의 철자/길이가 아니라 사용자가 기억하는 제목의 의미·유형.

    구조 단서인 ``TitleConstraints``와 분리한다. 예를 들어 "외국 사람
    이름 같은 제목"은 실제 제목 문자열도, 알파벳 제목이라는 확정 기억도
    아니므로 기존 ``script`` 슬롯에 넣으면 안 된다.
    """

    kind: Optional[
        Literal[
            "foreign_person_name",
            "person_name",
            "place_name",
            "sentence",
            "question",
            "onomatopoeia",
            "object_name",
            "other",
        ]
    ] = None
    text: str = Field(
        default="",
        max_length=200,
        description="사용자가 말한 제목 의미/유형 단서의 짧은 표현.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PerformanceClues(BaseModel):
    """곡 안에서 실제로 들리는 보컬 역할과 편성 단서.

    credited artist의 고정 정체성인 ``artist_type``과 분리한다. 예를 들어
    그룹의 솔로 보컬 파트, 솔로 가수의 피처링 듀엣, 밴드 사운드 편곡은
    아티스트 타입을 바꾸지 않는다.
    """

    vocal_count: Optional[
        Literal["solo", "duet", "multiple", "choir"]
    ] = Field(
        default=None,
        description="곡에서 들리는 보컬 인원/구성. 아티스트 구성과는 별개.",
    )
    vocal_roles: List[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "곡 내 역할 단서. 예: 남성노래, 여성노래, 남성랩, 여성랩, "
            "피처링보컬, 나레이션, 코러스, 합창."
        ),
    )
    sound_ensemble: List[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "들리는 편성/프로덕션 단서. 예: 밴드사운드, 어쿠스틱, "
            "오케스트라, 전자음악, 라이브, 아카펠라."
        ),
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ContextClue(BaseModel):
    """검색에 사용할 외부 사실 단서 한 개. 정답 곡명은 추측하지 않는다."""

    target: str = Field(
        default="",
        max_length=120,
        description="사용자가 언급한 작품·인물·행사 등의 대상. 불명확하면 빈 문자열.",
    )
    relation: str = Field(
        ...,
        min_length=1,
        max_length=160,
        description="그 대상과 곡의 관계(삽입곡, 제작 비화, 커버 무대 등).",
    )
    search_query: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="사용자가 제공한 배경 사실만으로 만든 한국어 검색 문장.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="사용자가 기억한 외부 사실 단서의 신뢰도. 전체 분석 신뢰도와 별개.",
    )


class QueryAnalysis(BaseModel):
    original_query: str = Field(..., description="사용자가 입력한 원본 한국어 질의")
    intent_type: Literal[
        "mood", "place", "time", "genre", "artist", "lyrics", "mixed"
    ] = Field(
        ...,
        description=(
            "질의 의도 분류. lyrics는 '가사에 X 단어' 류 키워드 정확매칭 의도 "
            "(BM25 가중치 ↑, dense ↓). 베이스라인 측정에서 Gemini가 'lyric' 반환으로 "
            "validation 실패하던 케이스 흡수 (2026-05-25)."
        ),
    )
    korean_tags: List[str] = Field(
        default_factory=list,
        description="BM25 보강용 한국어 명사 키워드 (띄어쓰기 없는 단어 단위)",
    )
    lyric_keywords: List[str] = Field(
        default_factory=list,
        description="질의에서 '가사에 ~단어'처럼 명시된 정확 매칭 키워드. BM25 sparse 가중 부스팅용.",
    )
    lyric_clues: List[LyricClue] = Field(
        default_factory=list,
        description=(
            "가사 단서 구조. 실제로 기억한 연속 구절은 verbatim/partial, "
            "들리는 대로 적은 소리는 phonetic, 줄거리식 설명은 semantic으로 구분한다."
        ),
    )
    lyric_semantic_query: str = Field(
        default="",
        description="가사 내용을 직접 인용하지 않고 의미만 설명한 경우의 간결한 검색 문장.",
    )
    song_title: str = Field(
        default="",
        description="질의에 명시된 곡 제목 (원문 공백 유지). 없으면 빈 문자열.",
    )
    title_constraints: TitleConstraints = Field(
    default_factory=TitleConstraints,
    description=(
        "사용자가 직접 제목을 모르지만 제목의 형태를 기억하는 경우의 구조적 단서."
        ),
    )
    artist_name: str = Field(
        default="",
        description="질의에 명시된 아티스트명 (원문 공백 유지). 없으면 빈 문자열.",
    )
    artist_name_alt: List[str] = Field(
        default_factory=list,
        description=(
            "아티스트명의 대체 표기 — 메타데이터에 저장될 수 있는 모든 형태. "
            "예: 뉴진스 → ['NewJeans','New Jeans']. BM25 sparse 매칭에서 한국어↔영문 "
            "토큰화 mismatch 흡수용 (2026-05-25 q004 baseline R=0 진단 결과)."
        ),
    )
    vocal_gender: Optional[Literal["남성", "여성", "혼성"]] = Field(
        default=None,
        description="질의에 '남자/여자 노래'처럼 명시된 보컬 성별. 없거나 불명확하면 null.",
    )
    genre: str = Field(
        default="",
        description="질의에 명시된 장르명 (예: '발라드', '랩/힙합'). 없으면 빈 문자열.",
    )
    image_english_query: str = Field(
        ...,
        max_length=600,
        description=(
            "SigLIP2 텍스트→이미지 임베딩 전용 영어 질의. 앨범 아트에서 "
            "실제로 볼 수 있는 색상·피사체·구도·재질/질감·사진/일러스트 "
            "스타일·타이포그래피만 포함한다. 사용자가 앨범/표지/커버라는 명사를 "
            "생략해도 정적인 아트워크를 구체적으로 묘사했다면 포함한다."
        ),
    )
    audio_english_query: str = Field(
        ...,
        max_length=600,
        description=(
            "CLAP 텍스트→오디오 임베딩 전용 영어 질의. 실제 파형에서 들을 수 있는 "
            "보컬 음색/역할·악기·템포·리듬·장르·프로덕션·청각적 분위기와 "
            "악기의 전후 관계·등장 구간·점층 변화·의도적인 악기 부재를 포함한다. "
            "명시적 청각 단서가 없으면 빈 문자열."
        ),
    )
    has_visual_clue: bool = Field(
        default=False,
        description=(
            "질의에 앨범 아트 시각 단서(색/그림/사진/질감/구도/타이포그래피 등)가 "
            "있으면 True. 앨범/표지/커버라는 명사를 생략한 구체적인 정적 아트워크 "
            "묘사도 포함한다. 단순 무드, 청취 중 떠오른 심상, 뮤직비디오·애니메이션·"
            "무대 장면은 제외한다. False일 때 SearchRouter가 SigLIP2(image) 경로를 "
            "0%로 차단하고 text/audio로 재정규화한다"
        ),
    )
    modality_weights: ModalityWeights = Field(default_factory=ModalityWeights)
    text_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="텍스트 경로 내 dense(KoE5) vs sparse(BM25) 비중. 1.0=dense 전용, 0.0=sparse 전용.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Gemini 분석 신뢰도")
    release_era: ReleaseEra = Field(
        default_factory=ReleaseEra,
        description="사용자가 기억하는 발매 연도 범위",
    )
    artist_type: ArtistTypeClue = Field(
        default_factory=ArtistTypeClue,
        description="credited artist 자체의 솔로/그룹/듀오/밴드 정체성 단서",
    )
    title_meaning_clue: TitleMeaningClue = Field(
        default_factory=TitleMeaningClue,
        description="제목의 문자 구조가 아닌 의미·유형에 대한 기억.",
    )
    performance_clues: PerformanceClues = Field(
        default_factory=PerformanceClues,
        description="곡 안에서 들리는 보컬 역할·인원·사운드 편성 단서",
    )
    context_clues: List[ContextClue] = Field(
        default_factory=list,
        max_length=4,
        description="외부 배경 사실 단서 목록. Text/Image/Audio 가중치와 독립적으로 검색한다.",
    )

    @model_validator(mode="after")
    def enforce_modality_query_gates(self) -> "QueryAnalysis":
        """빈/비활성 전용 질의가 퓨전 가중치를 차지하지 않도록 보정한다.

        교차 모달 문장 내용의 검증은 QueryAnalyzer의 safeguard가 담당하고,
        이 스키마 계층은 어떤 생성 경로로 만들어진 객체라도 빈 문자열을 실제
        검색 경로로 보내지 않는 마지막 불변식을 제공한다.
        """
        self.image_english_query = " ".join(
            self.image_english_query.split()
        )
        self.audio_english_query = " ".join(
            self.audio_english_query.split()
        )

        if not self.has_visual_clue:
            self.image_english_query = ""
        elif not self.image_english_query:
            self.has_visual_clue = False

        text_weight = self.modality_weights.text
        image_weight = (
            self.modality_weights.image
            if self.image_english_query
            else 0.0
        )
        audio_weight = (
            self.modality_weights.audio
            if self.audio_english_query
            else 0.0
        )
        total = text_weight + image_weight + audio_weight
        if total <= 0:
            text_weight, image_weight, audio_weight = 1.0, 0.0, 0.0
        else:
            text_weight /= total
            image_weight /= total
            audio_weight /= total
        self.modality_weights = ModalityWeights(
            text=round(text_weight, 6),
            image=round(image_weight, 6),
            audio=round(audio_weight, 6),
        )
        return self

    @property
    def has_lexical_lyric_clue(self) -> bool:
        return bool(self.lyric_keywords) or any(
            clue.kind in {"verbatim", "partial", "phonetic"}
            for clue in self.lyric_clues
        )

    @property
    def has_exact_lyric_clue(self) -> bool:
        """실제 철자를 안다고 볼 수 있는 가사 단서인지 여부."""
        return bool(self.lyric_keywords) or any(
            clue.kind in {"verbatim", "partial"}
            for clue in self.lyric_clues
        )

    @property
    def has_phonetic_lyric_clue(self) -> bool:
        return any(clue.kind == "phonetic" for clue in self.lyric_clues)

    @property
    def has_any_lyric_clue(self) -> bool:
        return bool(
            self.has_lexical_lyric_clue
            or self.lyric_semantic_query.strip()
            or any(clue.kind == "semantic" for clue in self.lyric_clues)
        )

    @property
    def has_context_clue(self) -> bool:
        return any(clue.confidence > 0 for clue in self.context_clues)

    @property
    def has_release_era(self) -> bool:
        return (
            self.release_era.start_year is not None
            and self.release_era.end_year is not None
            and self.release_era.confidence > 0
        )


    @property
    def has_artist_type_clue(self) -> bool:
        return bool(
            self.artist_type.values
            and self.artist_type.confidence > 0
        )


    @property
    def has_metadata_clue(self) -> bool:
        return (
            self.has_release_era
            or self.has_artist_type_clue
        )

    @property
    def has_performance_clue(self) -> bool:
        clue = self.performance_clues
        return bool(
            clue.confidence > 0
            and (
                clue.vocal_count
                or clue.vocal_roles
                or clue.sound_ensemble
            )
        )

    @property
    def has_composite_performance_clue(self) -> bool:
        """후보군 확장을 정당화할 만큼 구체적인 공연/사운드 단서인지.

        단순히 "여자가 부른다" 같은 한 축만으로 별도 후보 경로를 켜면
        대부분의 질의 순위가 변한다. 서로 다른 두 축 이상이 있거나,
        휘파람/비프음처럼 희소한 효과음이 명시된 경우에만 확장한다.
        """
        if not self.has_performance_clue:
            return False
        clue = self.performance_clues
        rare_sounds = {
            "휘파람",
            "비프음",
            "벨소리",
            "박수",
            "핑거스냅",
            "사이렌",
            "전화음",
            "자연음",
            "인도풍",
            "중동풍",
            "국악",
        }
        rare_sound_match = bool(
            rare_sounds & set(clue.sound_ensemble)
        )
        structured_vocal_roles = bool(
            clue.vocal_count and clue.vocal_roles
        )
        vocal_and_sound = bool(
            clue.vocal_roles and clue.sound_ensemble
        )
        count_and_sound = bool(
            clue.vocal_count and clue.sound_ensemble
        )
        multiple_specific_roles = len(clue.vocal_roles) >= 2
        return bool(
            rare_sound_match
            or structured_vocal_roles
            or vocal_and_sound
            or count_and_sound
            or multiple_specific_roles
        )

    @property
    def has_title_meaning_clue(self) -> bool:
        clue = self.title_meaning_clue
        return bool(clue.kind and clue.text.strip() and clue.confidence > 0)

    @property
    def needs_balanced_semantic_recall(self) -> bool:
        """기존 dense 중심 검색을 유지하면서 0.6 hybrid 보조 경로가 필요한지."""
        explicit_genre = bool(
            self.genre.strip()
            or any(
                token in self.original_query.lower()
                for token in (
                    "힙합",
                    "랩",
                    "발라드",
                    "록 음악",
                    "록 느낌",
                    "록밴드",
                    "락 음악",
                    "락 느낌",
                    "락밴드",
                    "재즈",
                    "알앤비",
                    "r&b",
                    "댄스",
                    "트로트",
                    "메탈",
                    "포크",
                    "클래식",
                )
            )
        )
        return bool(
            self.lyric_semantic_query.strip()
            and not self.has_lexical_lyric_clue
            and self.text_alpha > 0.60
            and self.has_release_era
            and self.has_artist_type_clue
            and self.performance_clues.sound_ensemble
            and explicit_genre
        )
