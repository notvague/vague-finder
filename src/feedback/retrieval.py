"""승인 사례로 검색 후보를 보충할 곡을 고른다. 라우터(`search_router`)에 주입돼 두 단계로 불린다.

1. `find_cases` — 기존 검색 경로와 **나란히** 돈다. 질의를 임베딩해 유사 사례를 읽고(미만료·같은 임베딩 판·
   문턱 이상), 그 사례가 가리키는 곡의 **현재** 정보를 곡 색인에서 읽는다. 곡이 없으면 삭제된 곡이다.
2. `validate` — 기존 경로가 끝난 뒤 **후보에 합치기 직전**에 돈다. 사례의 Mongo 최신 문서를 한 번에 읽어
   활성 제출·승인·제출 판·검토 판·미만료를 그 시각 기준으로 다시 본다.

**상태 확인 시점은 2단계의 Mongo 조회 한 번이다.** 그 뒤(융합·리랭킹 중)에 철회·수정·승인 취소·만료가
일어나면 진행 중이던 그 요청에는 반영되지 않고, 다음 검색부터 그 사례가 빠진다. Qdrant 포인트가 아직
지워지지 않았어도, 늦게 끝난 색인 작업이 예전 판을 다시 올렸어도 2단계에서 걸러진다.

Mongo를 읽지 못하면 **사례를 하나도 쓰지 않는다** — Qdrant에 남은 값만 믿고 보충하지 않는다.

여기서 고르는 것은 "후보에 넣을 곡"까지다. 최종 순위는 기존 리랭커가 곡 정보만 보고 정한다.
사용자 사유와 제출 이력은 Mongo에서 읽지도 않는다. 재사용 문장은 "비어 있지 않은가"를 보는 데만 쓰고
이 모듈 밖으로 내보내지 않는다 — 돌려주는 것은 곡 정보와 사례의 판 번호·유사도뿐이다.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer, MatchingTrack
from src.feedback.case_index import CaseHit, CaseIndexBusy, FeedbackCaseIndex
from src.feedback.review import CaseVersion, clean_text, reusable_case, utcnow
from src.feedback.settings import EMBEDDING_VERSION, RetrievalSettings, retrieval_enabled
from src.retrieval.search_service import SearchService
from src.vector_db.settings import TEXT_HYBRID_INDEX_NAME

logger = logging.getLogger(__name__)

# 사례 문장에 덧붙이는 재질문 답. 출생 연도(birth_year)는 사용자 개인 값이라 넣지 않는다.
_ANSWER_LABELS = {
    "vocal_gender": "가수 성별", "genre": "장르", "type": "가수 형태", "artist_type": "가수 형태",
    "release_era": "발매 시기",
}

# 조회 서비스는 유효한 사례가 있으면 valid_cases까지만 말한다. 그 뒤의 세 상태는 **최종 후보 절단 뒤**
# 라우터가 정한다 — 실제로 들어왔는지를 보고. valid_cases는 검색 결과의 최종 상태로 남지 않는다.
STATUS_VALID = "valid_cases"          # 최신 상태 확인을 통과한 사례가 있다(아직 후보에 넣기 전)
STATUS_APPLIED = "applied"            # 사례가 가리킨 곡이 후보에 새로 들어왔다
STATUS_NOT_ENTERED = "not_entered"    # 후보 밖 곡을 가리킨 사례가 있었지만 들어온 곡이 없다(절단·상한·자리 없음)
STATUS_ALREADY = "already_in_candidates"  # 사례가 가리킨 곡이 모두 이미 후보였다(바꾼 것 없음)
STATUS_NO_CASE = "no_case"            # 문턱을 넘는 사례가 없었다
STATUS_NO_VALID_CASE = "no_valid_case"  # 사례는 있었지만 최신 상태 확인에서 모두 빠졌다
STATUS_TIMEOUT = "timeout"            # 시간 예산 안에 끝나지 않았다
STATUS_FAILED = "failed"              # 조회·확인 실패(Mongo·Qdrant·임베딩)

# 사례가 빠진 사유. 평가 기록에 건수로 남는다.
DROP_MISSING = "missing"              # Mongo에 원본이 없다(TTL로 지워졌거나 포인트만 남았다)
DROP_INVALID = "not_reusable"         # 철회·미승인·만료·판 불일치
DROP_REJECTED = "rejected_song"       # 재질문에서 사용자가 거절한 곡
DROP_SONG_GONE = "song_missing"       # 곡 색인에 없는 곡(삭제)
DROP_DUPLICATE = "same_song"          # 같은 곡의 더 가까운 사례가 이미 있다
# 유효한 사례 가운데 후보에 더하지 않은 것(already_candidate·over_limit·no_gain)은 라우터가 센다 —
# 이름은 `explain.FeedbackCaseExplain.outcome`과 같다.
DROP_OVER_LIMIT = "over_limit"


def build_case_query(analysis: QueryAnalysis, answers: Optional[Sequence[ClarifyAnswer]] = None) -> str:
    """사례 조회용 질의문. 검색이 실제로 쓴 분석과 **지금 유효한 재질문 답**으로 만든다.

    - 본문은 `analysis.search_text` — 텍스트 경로의 dense 질의와 같은 문장이다(생애 단계 표현은 빠져 있다).
    - 가사의 의미 단서가 있으면 텍스트 경로와 같은 모양으로 붙인다.
    - 재질문 답(성별·장르·형태·시기)은 정정이므로 붙인다. 건너뛴 답·빈 값은 넣지 않는다.
    - **출생 연도는 넣지 않는다.** 사용자 개인 값이고, 검색에는 시기 창으로만 반영된다.
    """
    parts = [analysis.search_text]
    if analysis.lyric_semantic_query and not analysis.has_exact_lyric_clue:
        parts.append(f"가사 의미: {analysis.lyric_semantic_query}")
    for answer in answers or ():
        label = _ANSWER_LABELS.get(answer.slot)
        value = clean_text(answer.value)
        if label and value and not answer.skipped:
            parts.append(f"{label}: {value}")
    return "\n".join(part for part in parts if part and part.strip())


@dataclass
class CaseLookup:
    """1단계 결과. 아직 Mongo로 확인하지 않은 사례와, 그 사례가 가리키는 곡의 현재 정보."""

    hits: List[CaseHit] = field(default_factory=list)
    tracks: Dict[str, MatchingTrack] = field(default_factory=dict)
    settings: RetrievalSettings = field(default_factory=RetrievalSettings)


@dataclass(frozen=True)
class SupplementItem:
    track: MatchingTrack
    case: CaseVersion
    similarity: float
    rank: int              # 유효 사례 안에서의 유사도 순위(1부터)


@dataclass
class FeedbackSupplement:
    status: str = STATUS_NO_CASE
    items: List[SupplementItem] = field(default_factory=list)
    scanned: int = 0                                   # 문턱을 넘어 확인 대상이 된 사례 수
    dropped: Dict[str, int] = field(default_factory=dict)
    settings: RetrievalSettings = field(default_factory=RetrievalSettings)


class FeedbackRetriever:
    def __init__(
        self,
        case_index: FeedbackCaseIndex,
        store: Callable[[], Any],
        embedder: Callable[[], Any],
        vector_client: Any,
        *,
        settings: Optional[Callable[[], RetrievalSettings]] = None,
        clock: Callable[[], datetime] = utcnow,
    ):
        """store·embedder는 **부를 때 만드는 함수**다. 스위치가 꺼져 있으면 Mongo 연결도 모델도 만들지 않는다.

        vector_client도 만들 때는 만지지 않는다 — 곡 색인 핸들은 처음 조회할 때 연다.
        """
        self._case_index = case_index
        self._store = store
        self._embedder = embedder
        self._vector_client = vector_client
        self._song_index: Any = None           # 곡 텍스트 색인(`fetch`만 쓴다)
        self._settings = settings or RetrievalSettings.from_env
        self._clock = clock

    @property
    def enabled(self) -> bool:
        return retrieval_enabled()

    def settings(self) -> RetrievalSettings:
        return self._settings()

    # --- 1단계 ---------------------------------------------------------------
    def find_cases(self, query_text: str, *, cancel: Optional[threading.Event] = None,
                   settings: Optional[RetrievalSettings] = None) -> CaseLookup:
        settings = settings or self.settings()
        lookup = CaseLookup(settings=settings)
        text = clean_text(query_text)
        # 사례 컬렉션이 아직 없으면(색인 작업이 한 번도 돌지 않았다) 임베딩도 하지 않는다.
        if not text or not self._case_index.exists():
            return lookup
        started = time.monotonic()
        # 텍스트 경로와 같은 규약: 질의에는 E5 `query: ` 접두사, L2 정규화.
        vector = self._embedder().embed_passages([f"query: {text}"], add_e5_prefix=False, normalize=True)[0]
        if cancel is not None and cancel.is_set():
            return lookup
        remaining = max(0.01, settings.budget_seconds - (time.monotonic() - started))
        lookup.hits = self._case_index.search(
            vector, top_k=settings.top_cases, now_ts=self._clock().timestamp(),
            embedding_version=EMBEDDING_VERSION, min_similarity=settings.min_similarity, lock_timeout=remaining,
        )
        if not lookup.hits or (cancel is not None and cancel.is_set()):
            return lookup
        song_ids = list(dict.fromkeys(hit.song_id for hit in lookup.hits))
        if self._song_index is None:
            self._song_index = self._vector_client.Index(TEXT_HYBRID_INDEX_NAME)
        matches = self._song_index.fetch(song_ids)["matches"]
        lookup.tracks = {str(match["id"]): SearchService.track_from_match(match) for match in matches}
        return lookup

    # --- 2단계 ---------------------------------------------------------------
    def validate(self, lookup: CaseLookup, *, excluded_ids: Optional[Iterable[str]] = None,
                 budget_seconds: Optional[float] = None) -> FeedbackSupplement:
        settings = lookup.settings
        supplement = FeedbackSupplement(scanned=len(lookup.hits), settings=settings)
        if not lookup.hits:
            return supplement
        excluded: Set[str] = {str(song_id) for song_id in (excluded_ids or ())}
        documents = self._store().fetch_cases(
            [hit.feedback_id for hit in lookup.hits],
            budget_seconds=settings.budget_seconds if budget_seconds is None else budget_seconds,
        )
        # 만료·판 일치는 **Mongo를 읽은 뒤의 시각**으로 본다. 조회가 오래 걸린 사이 만료된 사례를 쓰지 않는다.
        now = self._clock()
        now_ts = now.timestamp()

        def drop(reason: str) -> None:
            supplement.dropped[reason] = supplement.dropped.get(reason, 0) + 1

        seen_songs: Set[str] = set()
        # 유사도 내림차순(동점은 사례 ID 순). 곡마다 **가장 가까운 사례 하나**만 남긴다 —
        # 같은 곡의 제출이 여러 건이어도 점수가 쌓이지 않는다.
        for hit in sorted(lookup.hits, key=lambda hit: (-hit.similarity, hit.feedback_id)):
            document = documents.get(hit.feedback_id)
            if document is None:
                drop(DROP_MISSING)
                continue
            case = reusable_case(document, now)
            if (case is None or not case.matches_payload(hit.payload())
                    or hit.embedding_version != EMBEDDING_VERSION or hit.expires_at_ts <= now_ts
                    or hit.similarity < settings.min_similarity):
                drop(DROP_INVALID)
                continue
            if case.song_id in excluded:
                drop(DROP_REJECTED)
                continue
            track = lookup.tracks.get(case.song_id)
            if track is None:
                drop(DROP_SONG_GONE)
                continue
            if case.song_id in seen_songs:
                drop(DROP_DUPLICATE)
                continue
            # 곡 수 상한(max_songs)은 여기서 자르지 않는다. 이미 후보인 곡을 뺀 뒤에 라우터가 적용한다 —
            # 여기서 자르면 상위 사례가 모두 기존 후보일 때 후보 밖 곡이 상한에 밀려 사라진다.
            seen_songs.add(case.song_id)
            supplement.items.append(SupplementItem(
                track=track, case=case, similarity=hit.similarity, rank=len(supplement.items) + 1))
        # 유효한 사례가 있으면 여기서는 "후보에 올릴 곡이 있다"까지만 말한다. 들어왔는지는 라우터가 정한다.
        supplement.status = STATUS_VALID if supplement.items else STATUS_NO_VALID_CASE
        return supplement


__all__ = [
    "CaseIndexBusy", "CaseLookup", "FeedbackRetriever", "FeedbackSupplement", "SupplementItem",
    "build_case_query",
]
