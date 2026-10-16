"""전체 가사를 이용한 표면 구절 exact/fuzzy 후보 검색.

텍스트 경로의 형태소 BM25는 일반 키워드 검색용으로 유지하고, 사용자가 실제로 기억한
가사 구절은 MongoDB 원본 ``full_lyrics``에서 문자 순서를 보존해 찾는다. 곡 수가
수천 개 수준인 현재 카탈로그에서는 전체 문서를 메모리에 캐시하는 편이 단순하고
결정적이다.

**판(스냅샷)의 수명(2026-10-16, v58).** 세 경우를 구분한다.

1. **최초 적재** — 쓸 판이 없다. 요청이 Mongo 전체를 읽고 기다린다(블로킹, 예열이
   보통 이 비용을 먼저 낸다). 실패하면 예외가 그대로 올라간다 — 가사 경로를 빈
   결과로 조용히 생략하지 않는다.
2. **TTL 만료(판이 있다)** — 요청은 **지금 판으로 바로 검색**하고, 같은 서비스에서
   **백그라운드 갱신 하나**만 시작한다. 새 판은 새 객체에서 끝까지 만든 뒤 한 번에
   교체한다. 갱신이 실패하면 지금 판을 유지하고 `refresh_retry_seconds` 동안 다시
   시작하지 않는다. 그동안은 **오래된 자료**를 낼 수 있다(§RUN_INFO 참고).
3. **명시적 `refresh()`** — 무효화다. 판을 버리고 세대를 올린다. 다음 검색은 1처럼
   블로킹 적재를 하고, 진행 중이던 갱신(세대가 다름)은 결과를 **게시하지 않는다**.

`cache_ttl_seconds=0`은 예전과 같이 "캐시 없음"이다 — 매 검색이 동기로 다시 읽는다.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.backend.schemas.query import LyricClue
from src.retrieval import timing
from src.backend.schemas.search import LyricSurfaceMatch, MatchingTrack
from src.retrieval.lyrics_query import (
    char_ngrams,
    normalize_lyric_surface,
    normalize_with_origins,
)

logger = logging.getLogger(__name__)

# 가사가 들어 있는 컬렉션. 인용의 출처 문자열이 여기서 나오므로 한 곳에서만 적는다.
COLLECTION = "songs"
LYRICS_FIELD = f"{COLLECTION}.lyrics_data.full_lyrics"

# 백그라운드 갱신이 실패한 뒤 다시 시도하기까지의 기본 대기(초). 요청마다 재시도하면
# Mongo가 아픈 동안 가사 요청 수만큼 전체 읽기가 몰린다.
DEFAULT_REFRESH_RETRY_SECONDS = 60.0

# 가사 읽기 **전용** Mongo 클라이언트의 제한. 공용 클라이언트(`src/common/mongodb.py`)에는
# socket 제한이 없어 Mongo가 멎으면 읽기가 영원히 돌아오지 않는다 — 그러면 갱신 스레드도,
# 그것을 기다리는 서버 종료도 멈춘다. 다른 Mongo 작업에 영향을 주지 않도록 여기만 따로 연다.
# socket 제한은 **한 왕복**(getMore 한 배치) 기준이라 배치 크기를 함께 고정해 왕복 하나의
# 크기를 예측 가능하게 둔다(v56 실측 5,143곡 350초 ≈ 68ms/곡 → 200곡 ≈ 14초 < 60초).
DEFAULT_MONGO_SOCKET_TIMEOUT_SECONDS = 60.0
DEFAULT_MONGO_BATCH_SIZE = 200
# 종료가 갱신 스레드를 기다리는 여유. socket 제한이 걸리면 스레드는 곧 끝나므로
# "socket 제한 + 여유"면 충분하다. 넘기면 제한이 작동하지 않은 것이라 False로 알린다.
DEFAULT_CLOSE_WAIT_MARGIN_SECONDS = 15.0


def _make_lyrics_mongo_client(socket_timeout_seconds: float) -> Any:
    """가사 읽기 전용 MongoClient. 공용 연결 설정(URI·DB)은 같고 socket 제한만 더한다."""
    from pymongo import MongoClient

    from src.common.mongodb import MONGO_URI

    return MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=int(max(1.0, socket_timeout_seconds) * 1000),
        appname="vaguefinder-lyrics",
    )


@dataclass(frozen=True)
class _LyricsDocument:
    track: MatchingTrack
    normalized_lyrics: str
    trigrams: frozenset[str]
    # 인용하려면 원문이 있어야 한다. 정규화본은 대소문자·띄어쓰기·줄바꿈이
    # 지워진 문자열이라 그대로는 인용문이 못 된다.
    full_lyrics: str = ""
    # 이 곡 가사의 판. 스냅샷이 바뀌면 같은 위치가 다른 문장을 가리키므로,
    # 인용에는 어느 판에서 뽑았는지가 함께 붙어야 나중에 되짚을 수 있다.
    lyrics_sha16: str = ""


@dataclass(frozen=True)
class LyricsSnapshot:
    """어느 한 시점의 가사 전체. **한 번 만들면 바뀌지 않는다.**

    검색과 인용이 같은 판을 봐야 한다. 서비스가 들고 있는 "지금 판"을 그때그때
    읽으면, 검색이 끝난 뒤 TTL이 만료되거나 다른 요청이 갱신했을 때 인용이
    **다른 판**을 가리킨다 — 실제로 판 해시와 구절 위치가 둘 다 달라졌다
    (2026-09-24 재현). 그래서 검색이 쓴 판을 요청에 그대로 들려 보낸다.
    """

    documents: Tuple["_LyricsDocument", ...]
    by_id: Mapping[str, "_LyricsDocument"]
    loaded_at: float
    # 어느 세대의 자료원 읽기로 만든 판인가. `refresh()`가 세대를 올리면 그 전에
    # 시작한 읽기의 결과는 게시되지 않는다(무효화 이전 자료가 되살아나지 않게).
    generation: int = 0

    def get(self, song_id: str) -> Optional["_LyricsDocument"]:
        return self.by_id.get(str(song_id))


@dataclass(frozen=True)
class LyricExcerpt:
    """가사 인용 하나. **원문에서 글자 단위로 떠 온 구간이다.**

    `matched_phrase`는 정규화 표기라 인용문이 될 수 없다. 이 구조는 그 일치가
    원문의 어느 자리였는지를 되짚어, 그 자리의 **원문 그대로**를 담는다.

    `line`은 그 구간이 걸친 줄 전체다 — 구절만 떼어 놓으면 앞뒤가 잘려 무슨
    말인지 알 수 없다.
    """

    song_id: str
    source: str            # 어느 저장소의 어느 필드인가
    quote: str             # 원문 그대로의 구간
    line: str              # 그 구간이 걸친 줄 전체
    char_start: int        # 원문에서의 시작 위치
    char_end: int          # 원문에서의 끝 위치(제외)
    data_version: str      # 이 곡 가사의 판
    matched_phrase: str    # 무엇을 되짚었는가(정규화 표기)


class LyricsServiceClosed(RuntimeError):
    """닫힌 서비스에 판이 없는데 읽기를 요구했다. 종료 중에 Mongo를 새로 읽지 않는다."""


class _RefreshCancelled(Exception):
    """갱신이 도중에 취소됐다(종료 또는 `refresh()`로 세대가 바뀜). 실패가 아니다."""


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _track_from_document(document: Mapping[str, Any]) -> MatchingTrack:
    metadata = document.get("metadata") or {}
    lyrics = document.get("lyrics_data") or {}
    semantic = document.get("semantic_analysis") or {}
    community = document.get("community_feedback") or {}
    links = document.get("links") or {}
    popularity = community.get("popularity") or {}
    song_id = str(
        document.get("song_id")
        or document.get("id")
        or document.get("_id")
        or ""
    )

    return MatchingTrack(
        id=song_id,
        score=0.0,
        title=str(metadata.get("title") or document.get("title") or "Unknown"),
        artist=metadata.get("artist") or document.get("artist"),
        album=metadata.get("album"),
        release_date=metadata.get("release_date"),
        genre=metadata.get("genre"),
        vocal_gender=metadata.get("vocal_gender"),
        artist_types=_as_list(metadata.get("type")),
        search_style_summary=semantic.get("search_style_summary"),
        mood_tags=_as_list(semantic.get("mood_tags")),
        time_weather_tags=_as_list(semantic.get("time_weather_tags")),
        place_activity_tags=_as_list(semantic.get("place_activity_tags")),
        emotion_tags=_as_list(semantic.get("emotion_tags")),
        vibe_tags=_as_list(semantic.get("vibe_tags")),
        relation_context_tags=_as_list(semantic.get("relation_context_tags")),
        color_tags=_as_list(semantic.get("color_tags")),
        sound_tags=_as_list(semantic.get("sound_tags")),
        melon_playlist_tags=_as_list(semantic.get("melon_playlist_tags")),
        visual_imagery=_as_list(semantic.get("visual_imagery")),
        lyrics_highlight=lyrics.get("lyrics_highlight"),
        lyrics_summary=lyrics.get("lyrics_summary"),
        sentiment_summary=community.get("sentiment_summary"),
        fans_tags=_as_list(community.get("fans_tags")),
        major_emotion=community.get("major_emotion"),
        fame=popularity.get("fame"),
        melon_url=links.get("melon_url"),
        youtube_url=links.get("youtube_url"),
        cover_url=links.get("cover_url"),
    )


def _fuzzy_surface_score(
    phrase: str,
    lyrics: str,
    lyrics_trigrams: frozenset[str],
) -> float:
    """짧은 오기억을 허용하되 긴 가사 전체의 산발적 토큰 일치는 억제한다."""
    if not phrase or not lyrics or len(phrase) < 5:
        return 0.0

    query_trigrams = char_ngrams(phrase)
    if not query_trigrams:
        return 0.0

    coverage = len(query_trigrams & lyrics_trigrams) / len(query_trigrams)
    if coverage < 0.55:
        return 0.0

    longest = SequenceMatcher(None, phrase, lyrics, autojunk=False).find_longest_match(
        0,
        len(phrase),
        0,
        len(lyrics),
    )
    contiguous = longest.size / len(phrase)
    return 0.65 * coverage + 0.35 * contiguous


def _clue_surfaces(clue: LyricClue) -> List[str]:
    """원문 단서와 원어 후보를 순서 보존·중복 제거해 반환한다."""
    values = [clue.text, *clue.variants]
    surfaces: List[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_lyric_surface(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            surfaces.append(normalized)
    return surfaces


def _line_around(text: str, start: int, end: int) -> str:
    """구간이 걸친 줄 전체. 구절만 떼면 앞뒤가 잘려 무슨 말인지 알 수 없다."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    return text[line_start:line_end].strip()


def _error_label(exc: BaseException) -> str:
    """상태·로그용 오류 요약 — **예외 형식 이름만**(원인 사슬 최대 3단).

    `str(exc)`는 싣지 않는다. pymongo 오류 메시지에는 호스트·포트·계정이 섞일 수 있고
    이 값은 `/health`에 그대로 나가기 때문이다.
    """
    names: List[str] = []
    current: Optional[BaseException] = exc
    while current is not None and len(names) < 3:
        names.append(type(current).__name__)
        current = current.__cause__
    return " <- ".join(names)


class LyricsExactSearchService:
    """MongoDB full_lyrics snapshot을 사용하는 exact/fuzzy 검색 서비스.

    잠금이 둘이다. `_state_lock`은 판·세대·갱신 상태를 바꾸는 **짧은** 구간만 지키고,
    Mongo를 읽는 동안에는 절대 잡지 않는다. `_load_lock`은 **최초(블로킹) 적재**만
    직렬화한다 — 판이 없을 때 요청 둘이 각자 전체를 읽지 않게. TTL 만료 갱신은 어느
    잠금도 쥐지 않은 백그라운드 스레드에서 돈다.

    `documents_source`는 자료원을 주입하는 자리다(테스트·검증 서버). 없으면 **가사 전용**
    Mongo 클라이언트(`mongo_socket_timeout_seconds` 한 왕복 제한 · `mongo_batch_size`)로 읽는다 —
    공용 클라이언트는 socket 제한이 없어 Mongo가 멎으면 읽기와 종료가 함께 멈춘다.
    `clock`은 monotonic 시계를 바꿔 꽂는 자리다(가짜 시계로 TTL·재시도 간격을 시험).
    """

    def __init__(
        self,
        documents: Optional[Iterable[Mapping[str, Any]]] = None,
        fuzzy_threshold: float = 0.78,
        cache_ttl_seconds: float = 300.0,
        *,
        documents_source: Optional[Callable[[], Iterable[Mapping[str, Any]]]] = None,
        refresh_retry_seconds: float = DEFAULT_REFRESH_RETRY_SECONDS,
        clock: Optional[Callable[[], float]] = None,
        mongo_socket_timeout_seconds: float = DEFAULT_MONGO_SOCKET_TIMEOUT_SECONDS,
        mongo_batch_size: int = DEFAULT_MONGO_BATCH_SIZE,
        close_wait_margin_seconds: float = DEFAULT_CLOSE_WAIT_MARGIN_SECONDS,
    ) -> None:
        self._seed_documents = list(documents) if documents is not None else None
        self._documents_source = documents_source
        self._fuzzy_threshold = fuzzy_threshold
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._refresh_retry_seconds = max(0.0, float(refresh_retry_seconds))
        self._clock: Callable[[], float] = clock or time.monotonic
        self._mongo_socket_timeout_seconds = max(1.0, float(mongo_socket_timeout_seconds))
        self._mongo_batch_size = max(1, int(mongo_batch_size))
        self._close_wait_seconds = self._mongo_socket_timeout_seconds + max(
            0.0, float(close_wait_margin_seconds)
        )
        self._mongo_client: Any = None

        self._snapshot: Optional[LyricsSnapshot] = None
        self._generation = 0
        self._closed = False
        self._state_lock = threading.Lock()
        # 최초(블로킹) 적재가 돌고 있는지 — close()가 백그라운드 갱신과 함께 기다린다.
        self._state_cond = threading.Condition(self._state_lock)
        self._blocking_loads = 0
        self._load_lock = threading.Lock()

        # 백그라운드 갱신 상태. 전부 `_state_lock` 아래에서만 바꾼다.
        self._refreshing = False
        self._refresh_thread: Optional[threading.Thread] = None
        self._refresh_started_at: Optional[float] = None
        self._refresh_failed_at: Optional[float] = None
        self._refresh_error: Optional[str] = None
        self._refresh_successes = 0
        self._refresh_failures = 0
        self._refresh_cancelled = 0
        self._refresh_discarded = 0
        self._last_success_wall: Optional[float] = None

    # ------------------------------------------------------------------
    # 공개 수명 주기
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """무효화 — 다음 검색에서 다시 읽게 한다. **이미 나간 판은 그대로 산다** —
        그 판으로 검색한 요청은 끝까지 같은 판에서 인용한다.

        TTL 만료와 다르다. 만료는 지금 판을 계속 쓰면서 뒤에서 갱신하지만, 무효화는
        지금 판을 **버린다**(다음 검색이 블로킹으로 읽는다). 진행 중인 백그라운드
        갱신은 세대가 달라져 결과를 게시하지 못한다 — 무효화 이전에 읽은 자료가
        되살아나지 않는다. 실패 대기도 지운다(사용자가 다시 읽으라고 한 것이다).
        """
        with self._state_lock:
            self._generation += 1
            self._snapshot = None
            self._refresh_failed_at = None
            self._refresh_error = None

    def close(self, timeout: Optional[float] = None) -> bool:
        """종료. 새 갱신을 막고, 돌고 있는 갱신이 끝나기를 기다린 뒤 전용 Mongo 클라이언트를 닫는다.

        세대를 올려 두므로 돌던 갱신은 다음 문서에서 멈추고(취소), 혹시 끝까지 읽었어도
        게시하지 않는다. 이미 나간 판은 들고 있는 요청이 계속 쓴다. 닫힌 뒤의
        `search()`는 **마지막 판이 있으면 그 판으로** 답하고 새로 읽지는 않는다
        (판이 없으면 `LyricsServiceClosed`).

        **대기는 유한하다.** Mongo 읽기는 전용 클라이언트의 socket 제한(`mongo_socket_timeout_seconds`,
        한 배치 기준) 안에 반환되거나 실패하므로, 기본 대기는 "socket 제한 + 여유"다. 그 안에
        스레드가 끝나지 않으면 **False** — 제한이 작동하지 않았다는 신호이며 "취소 완료"가 아니다
        (주입한 자료원처럼 제한이 없는 읽기가 그렇다).

        timeout: 기다릴 시간. None이면 `close_wait_seconds`.
        """
        if timeout is None:
            timeout = self._close_wait_seconds
        with self._state_lock:
            self._closed = True
            self._generation += 1
            thread = self._refresh_thread
        deadline = time.monotonic() + timeout
        finished = True
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                logger.warning(
                    "[lyrics] 가사 판 갱신이 %.0f초 안에 멈추지 않았다 — 자료원 읽기가 아직 돌고 있다(socket 제한 %.0f초)",
                    timeout,
                    self._mongo_socket_timeout_seconds,
                )
                finished = False
        # 진행 중인 최초(블로킹) 적재도 같은 한도 안에서 기다린다. 그 적재는 다음 문서에서
        # 취소되고(`LyricsServiceClosed`), 끝나기 전에 한도를 넘기면 False다.
        with self._state_cond:
            while self._blocking_loads > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._state_cond.wait(remaining):
                    if self._blocking_loads > 0:
                        logger.warning(
                            "[lyrics] 최초 적재 %d건이 %.0f초 안에 끝나지 않았다 — 자료원 읽기가 아직 돌고 있다",
                            self._blocking_loads,
                            timeout,
                        )
                        finished = False
                    break
        self._close_mongo_client()
        return finished

    @property
    def close_wait_seconds(self) -> float:
        return self._close_wait_seconds

    @property
    def closed(self) -> bool:
        return self._closed

    def status(self) -> Dict[str, Any]:
        """운영 확인용 상태. 건수·시각·상태뿐이다 — 가사 내용은 싣지 않는다."""
        now = self._clock()
        with self._state_lock:
            snapshot = self._snapshot
            age = None if snapshot is None else round(now - snapshot.loaded_at, 1)
            if snapshot is None:
                state = "empty"
            elif self._cache_ttl_seconds <= 0:
                state = "no_cache"
            elif now - snapshot.loaded_at < self._cache_ttl_seconds:
                state = "fresh"
            else:
                state = "stale"
            failed_age = (
                None if self._refresh_failed_at is None else round(now - self._refresh_failed_at, 1)
            )
            retry_in = None
            if failed_age is not None:
                retry_in = max(0.0, round(self._refresh_retry_seconds - failed_age, 1))
            return {
                "state": state,
                "closed": self._closed,
                "documents": 0 if snapshot is None else len(snapshot.documents),
                "generation": self._generation,
                "snapshot_generation": None if snapshot is None else snapshot.generation,
                "age_seconds": age,
                "ttl_seconds": self._cache_ttl_seconds,
                "last_success_epoch": self._last_success_wall,
                "refreshing": self._refreshing,
                "initial_loads_running": self._blocking_loads,
                "refresh_running_seconds": (
                    None if not self._refreshing or self._refresh_started_at is None
                    else round(now - self._refresh_started_at, 1)
                ),
                "refresh_successes": self._refresh_successes,
                "refresh_failures": self._refresh_failures,
                "refresh_cancelled": self._refresh_cancelled,
                "refresh_discarded": self._refresh_discarded,
                "last_refresh_error": self._refresh_error,
                "last_failure_age_seconds": failed_age,
                "retry_seconds": self._refresh_retry_seconds,
                "retry_in_seconds": retry_in,
                "mongo_socket_timeout_seconds": self._mongo_socket_timeout_seconds,
                "mongo_batch_size": self._mongo_batch_size,
                "close_wait_seconds": self._close_wait_seconds,
            }

    # ------------------------------------------------------------------
    # 자료원
    # ------------------------------------------------------------------

    def _source_documents(self) -> Iterable[Mapping[str, Any]]:
        if self._documents_source is not None:
            return self._documents_source()
        if self._seed_documents is not None:
            return self._seed_documents

        # pymongo 의존성과 DB 연결은 실제 가사 검색이 실행될 때까지 지연한다.
        # 공용 클라이언트가 아니라 **가사 전용**(socket 제한 있음) 클라이언트로 읽는다.
        from src.common.mongodb import DB_NAME

        client = self._get_mongo_client()
        return client[DB_NAME][COLLECTION].find(
            {},
            {
                "song_id": 1,
                "title": 1,
                "artist": 1,
                "metadata": 1,
                "lyrics_data.full_lyrics": 1,
                "lyrics_data.lyrics_highlight": 1,
                "lyrics_data.lyrics_summary": 1,
                "semantic_analysis": 1,
                "community_feedback": 1,
                "links": 1,
            },
            batch_size=self._mongo_batch_size,
        )

    def _get_mongo_client(self) -> Any:
        with self._state_lock:
            if self._closed:
                # 종료 뒤 재개된 적재가 클라이언트를 새로 열어 남기지 않게 — 같은 잠금 안에서 거부한다.
                raise LyricsServiceClosed("가사 서비스가 닫혀 Mongo 클라이언트를 만들지 않는다")
            if self._mongo_client is None:
                self._mongo_client = _make_lyrics_mongo_client(self._mongo_socket_timeout_seconds)
            return self._mongo_client

    def _close_mongo_client(self) -> None:
        with self._state_lock:
            client, self._mongo_client = self._mongo_client, None
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - 종료 경로
                logger.debug("[lyrics] 전용 Mongo 클라이언트 닫기 실패", exc_info=False)

    def _build_snapshot(
        self, generation: int, should_cancel: Callable[[], bool]
    ) -> LyricsSnapshot:
        """자료원을 끝까지 읽어 **새 객체**로 판을 만든다. 어느 잠금도 잡지 않는다.

        문서마다 `should_cancel()`을 보고 멈춘다(종료·세대 변경). pymongo cursor는
        한 배치 단위로 받으므로 취소도 그 단위다. 끝나든 멈추든 cursor는 닫는다.
        도중에 예외가 나면 만들던 목록은 버려진다 — 부분 판은 게시되지 않는다.
        """
        loaded: List[_LyricsDocument] = []
        source = self._source_documents()
        try:
            for raw in source:
                if should_cancel():
                    raise _RefreshCancelled()
                lyrics_data = raw.get("lyrics_data") or {}
                full_lyrics = str(lyrics_data.get("full_lyrics") or "")
                normalized = normalize_lyric_surface(full_lyrics)
                if not normalized:
                    continue
                track = _track_from_document(raw)
                if not track.id:
                    continue
                loaded.append(
                    _LyricsDocument(
                        track=track,
                        normalized_lyrics=normalized,
                        trigrams=char_ngrams(normalized),
                        full_lyrics=full_lyrics,
                        lyrics_sha16=hashlib.sha256(
                            full_lyrics.encode("utf-8")
                        ).hexdigest()[:16],
                    )
                )
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - 정리 실패는 결과에 영향이 없다
                    logger.debug("[lyrics] 자료원 cursor 닫기 실패 — %s", _error_label(exc))

        # 인용은 곡 하나를 바로 찾아야 한다. 결과 열 곡 때문에 전곡을
        # 다시 훑으면 근거를 펼칠 때마다 전체 순회가 한 번씩 더 돈다.
        return LyricsSnapshot(
            documents=tuple(loaded),
            by_id={document.track.id: document for document in loaded},
            loaded_at=self._clock(),
            generation=generation,
        )

    # ------------------------------------------------------------------
    # 판 얻기
    # ------------------------------------------------------------------

    def snapshot(self) -> LyricsSnapshot:
        """지금 판. 없으면 새로 읽는다(**블로킹**). 오래됐으면 지금 판을 주고 뒤에서 갱신한다.

        `search()`가 쓴 판을 인용까지 들고 가려면 호출부가 판을 손에 쥐어야 한다.
        그래서 목록이 아니라 판 자체를 돌려준다.
        """
        return self._load()

    def _is_fresh_locked(self, snapshot: LyricsSnapshot, now: float) -> bool:
        return now - snapshot.loaded_at < self._cache_ttl_seconds

    def _load(self) -> LyricsSnapshot:
        if self._cache_ttl_seconds <= 0:
            # 캐시 없음(기존 경계값) — 매 검색이 동기로 다시 읽는다. 백그라운드 없음.
            return self._load_blocking(replace=True)

        with self._state_lock:
            current = self._snapshot
            if current is not None:
                now = self._clock()
                if not self._is_fresh_locked(current, now):
                    # 만료 — 요청은 기다리지 않는다. 지금 판으로 검색하고 뒤에서 한 번만 갱신.
                    self._start_refresh_locked(now)
                return current
            if self._closed:
                raise LyricsServiceClosed("가사 서비스가 닫혔고 쓸 판이 없다")
        return self._load_blocking(replace=False)

    def _load_blocking(self, *, replace: bool) -> LyricsSnapshot:
        """최초 적재(또는 캐시 없음). 자료원을 끝까지 읽고 기다린다.

        replace=False면 기다리는 사이 다른 요청이 판을 올렸을 때 그 판을 쓴다(한 번만 읽기).
        읽는 동안 `refresh()`가 세대를 올렸으면 그 결과는 게시하지 않고 이 요청에만 준다 —
        요청이 본 자료는 일관되지만, 무효화 뒤의 다음 검색은 다시 읽는다.
        """
        with self._load_lock:
            with self._state_lock:
                if not replace and self._snapshot is not None:
                    current = self._snapshot
                    now = self._clock()
                    if not self._is_fresh_locked(current, now):
                        self._start_refresh_locked(now)
                    return current
                if self._closed:
                    raise LyricsServiceClosed("가사 서비스가 닫혔고 쓸 판이 없다")
                # 종료 확인·세대 확보·적재 등록은 **같은 잠금 구간**이어야 한다. 사이에 잠금이
                # 풀리면 close()가 "진행 중 적재 0건"으로 True를 돌려준 뒤 이 적재가 시작된다.
                # 종료가 먼저면 여기서 거부되고, 등록이 먼저면 close()가 이 적재를 기다린다.
                generation = self._generation
                self._blocking_loads += 1

            started = time.perf_counter()
            try:
                snapshot = self._build_snapshot(generation, should_cancel=lambda: self._closed)
            except _RefreshCancelled:
                with self._state_lock:
                    self._refresh_cancelled += 1
                raise LyricsServiceClosed("가사 서비스가 닫혀 최초 적재를 중단했다") from None
            finally:
                with self._state_lock:
                    self._blocking_loads -= 1
                    self._state_cond.notify_all()

            with self._state_lock:
                if self._generation == generation and not self._closed:
                    self._publish_locked(snapshot)
                    published = True
                else:
                    self._refresh_discarded += 1
                    published = False
            logger.info(
                "[lyrics] 가사 판 적재 %s — %d곡, %.1f초",
                "완료" if published else "완료(무효화돼 게시 안 함)",
                len(snapshot.documents),
                time.perf_counter() - started,
            )
            return snapshot

    def _publish_locked(self, snapshot: LyricsSnapshot) -> None:
        self._snapshot = snapshot
        self._last_success_wall = time.time()

    # ------------------------------------------------------------------
    # 백그라운드 갱신
    # ------------------------------------------------------------------

    def _start_refresh_locked(self, now: float) -> bool:
        """`_state_lock`을 쥔 채 부른다. 조건이 맞으면 갱신 스레드 하나를 띄운다.

        안 띄우는 경우: 닫혔다 · 이미 돌고 있다 · 지난 실패 뒤 재시도 간격이 안 지났다.
        요청마다 스레드를 만드는 것이 아니라 갱신 한 번에 하나다(TTL 주기당 최대 하나).
        """
        if self._closed or self._refreshing:
            return False
        if (
            self._refresh_failed_at is not None
            and now - self._refresh_failed_at < self._refresh_retry_seconds
        ):
            return False
        generation = self._generation
        self._refreshing = True
        self._refresh_started_at = now
        thread = threading.Thread(
            target=self._refresh_worker,
            args=(generation,),
            name="lyrics_refresh",
            daemon=True,
        )
        self._refresh_thread = thread
        thread.start()
        logger.info("[lyrics] 가사 판 TTL 만료 — 지금 판으로 검색하며 백그라운드 갱신 시작(세대 %d)", generation)
        return True

    def _refresh_worker(self, generation: int) -> None:
        started = time.perf_counter()
        try:
            snapshot = self._build_snapshot(
                generation,
                should_cancel=lambda: self._closed or self._generation != generation,
            )
        except _RefreshCancelled:
            with self._state_lock:
                self._refreshing = False
                self._refresh_cancelled += 1
            logger.info("[lyrics] 가사 판 갱신 취소(세대 %d) — %.1f초", generation, time.perf_counter() - started)
            return
        except Exception as exc:  # noqa: BLE001 - 실패를 기록하고 지금 판을 유지한다
            label = _error_label(exc)  # 형식 이름만 — 메시지(호스트·계정이 섞일 수 있음)는 남기지 않는다
            with self._state_lock:
                self._refreshing = False
                if self._closed or self._generation != generation:
                    # 무효화·종료 뒤에 늦게 실패한 **옛 세대** 읽기다. 새 판의 재시도 대기를
                    # 걸면 안 된다 — 성공 결과를 폐기하는 것과 같은 기준으로 폐기한다.
                    self._refresh_discarded += 1
                    stale = True
                else:
                    self._refresh_failures += 1
                    self._refresh_failed_at = self._clock()
                    self._refresh_error = label
                    stale = False
            if stale:
                logger.info("[lyrics] 옛 세대(%d) 갱신이 늦게 실패 — 무시(%s)", generation, label)
                return
            logger.warning(
                "[lyrics] 가사 판 갱신 실패(지금 판 유지, %.0f초 뒤 재시도 가능) — %s",
                self._refresh_retry_seconds,
                label,
            )
            return

        with self._state_lock:
            self._refreshing = False
            if self._closed or self._generation != generation:
                self._refresh_discarded += 1
                published = False
            else:
                self._publish_locked(snapshot)
                self._refresh_successes += 1
                self._refresh_failed_at = None
                self._refresh_error = None
                published = True
        logger.info(
            "[lyrics] 가사 판 갱신 %s — %d곡, %.1f초",
            "완료" if published else "완료(세대가 바뀌어 게시 안 함)",
            len(snapshot.documents),
            time.perf_counter() - started,
        )

    # ------------------------------------------------------------------
    # 인용 · 검색
    # ------------------------------------------------------------------

    def excerpt(
        self,
        song_id: str,
        matched_phrase: str,
        snapshot: Optional[LyricsSnapshot],
    ) -> Optional[LyricExcerpt]:
        """정규화된 일치 구절을 **그 판의 원문 구간으로** 되짚는다.

        **판을 받아서 쓴다. 여기서 새로 읽지 않는다.** 예전에는 `_load()`를 불렀는데,
        검색이 끝난 뒤 TTL이 만료되거나 다른 요청이 갱신하면 **검색에 쓴 가사와
        다른 판을 인용**했다(판 해시·구절 위치가 둘 다 달라지는 것을 재현했다).
        게다가 그 적재는 Mongo 왕복(측정 7.0초)이라, 비동기 라우트에서 동기로
        부르면 이벤트 루프까지 막는다.

        그래서 검색이 쓴 판을 그대로 받는다. 판이 없으면 **인용하지 않는다** —
        원래 판을 확보하지 못한 인용은 근거가 아니다.

        **못 되짚으면 None이다.** 네 가지 경우가 있다.

        1. 검색이 쓴 판이 없다 — 가사 경로가 돌지 않은 요청이다
        2. 그 곡이 그 판에 없다 — 다른 곡의 문장을 붙이는 사고를 막는다
        3. 정규화본에 그 구절이 없다 — fuzzy 일치가 이렇다. 3-gram 겹침으로
           뽑힌 것이라 **이어지는 한 구간이 원문에 없을 수 있다**
        4. 되짚은 구간을 다시 정규화했더니 원래 구절과 다르다 —
           `normalize_with_origins`의 글자 단위 정규화가 전체 정규화와 갈린
           경우다. 드물지만, 틀린 인용을 내느니 내지 않는다
        """
        phrase = normalize_lyric_surface(matched_phrase)
        if not phrase or not song_id or snapshot is None:
            return None
        document = snapshot.get(song_id)
        if document is None:
            return None

        normalized, origins = normalize_with_origins(document.full_lyrics)
        start = normalized.find(phrase)
        if start < 0:
            return None

        # origins는 정규화 글자 → 원문 글자다. 끝 글자의 원문 위치에 1을 더해야
        # 그 글자까지 포함된다(파이썬 구간은 끝을 제외한다).
        char_start = origins[start]
        char_end = origins[start + len(phrase) - 1] + 1
        quote = document.full_lyrics[char_start:char_end]

        # 되짚은 구간이 정말 그 구절인가. 이 확인이 없으면 위치가 한 칸만 밀려도
        # 다른 문장을 "원문 그대로"라고 내보내게 된다.
        if normalize_lyric_surface(quote) != phrase:
            return None

        return LyricExcerpt(
            song_id=str(song_id),
            source=LYRICS_FIELD,
            quote=quote,
            line=_line_around(document.full_lyrics, char_start, char_end),
            char_start=char_start,
            char_end=char_end,
            data_version=document.lyrics_sha16,
            matched_phrase=phrase,
        )

    def search(
        self,
        clues: Sequence[LyricClue],
        top_k: Optional[int] = 100,
        snapshot_out: Optional[List[LyricsSnapshot]] = None,
    ) -> List[MatchingTrack]:
        searchable = [
            clue
            for clue in clues
            if clue.kind in {"verbatim", "partial", "phonetic"}
            and _clue_surfaces(clue)
        ]
        if not searchable:
            return []


        results: List[MatchingTrack] = []
        # 구절별로 **전체 코퍼스에서** 몇 곡에 걸렸는지 센다. 후보를 자르기 전의
        # 수여야 "흔한 조각인가"를 말할 수 있다. 같은 순회에서 세므로 추가 비용이 없다.
        phrase_hits: Dict[tuple, int] = {}
        pending: List[tuple] = []

        # 스냅샷 적재와 훑기를 나눈다. 전곡을 Mongo에서 새로 읽는 것과, 이미 읽어 둔
        # 것을 구절로 훑는 것은 고치는 방법이 다르다(TTL 대 알고리즘). 판이 있으면
        # 이 구간은 잠금 한 번(마이크로초)이다 — 만료돼도 여기서 기다리지 않는다.
        with timing.step("lyrics.load"):
            snapshot = self._load()
        # 이 검색이 **무슨 판을 봤는지**를 호출부에 들려 보낸다. 인용은 이 판에서만
        # 뜬다(`candidate_ids_out`과 같은 꼴로, 반환 타입을 바꾸지 않기 위한 통로다).
        if snapshot_out is not None:
            snapshot_out.append(snapshot)
        documents = snapshot.documents
        scan_started = time.perf_counter()

        for document in documents:
            best_score = 0.0
            best_type: Optional[str] = None
            best_key: Optional[tuple] = None

            for clue_index, clue in enumerate(searchable):
                for phrase_index, phrase in enumerate(_clue_surfaces(clue)):
                    if phrase in document.normalized_lyrics:
                        match_type = (
                            "phonetic" if clue.kind == "phonetic" else "exact"
                        )
                        score = float(clue.confidence)
                    else:
                        raw_fuzzy_score = _fuzzy_surface_score(
                            phrase,
                            document.normalized_lyrics,
                            document.trigrams,
                        )
                        if raw_fuzzy_score < self._fuzzy_threshold:
                            continue
                        score = raw_fuzzy_score * float(clue.confidence)
                        match_type = "fuzzy"

                    key = (clue_index, phrase_index)
                    phrase_hits[key] = phrase_hits.get(key, 0) + 1

                    if score > best_score:
                        best_score = score
                        best_type = match_type
                        best_key = key

            if best_type is None:
                continue

            pending.append((document, best_score, best_type, best_key))

        timing.mark("lyrics.scan", scan_started, songs=len(documents))

        for document, best_score, best_type, best_key in pending:
            clue_index, phrase_index = best_key
            clue = searchable[clue_index]
            phrase = _clue_surfaces(clue)[phrase_index]
            results.append(
                document.track.model_copy(
                    update={
                        "score": best_score,
                        "lyric_match_type": best_type,
                        "lyric_match_score": best_score,
                        # 계측용. 랭킹 계산에는 쓰이지 않는다.
                        "lyric_match_detail": LyricSurfaceMatch(
                            clue_text=clue.text,
                            clue_kind=clue.kind,
                            matched_phrase=phrase,
                            is_variant=phrase_index > 0,
                            normalized_length=len(phrase),
                            match_type=best_type,
                            confidence=float(clue.confidence),
                            corpus_match_count=phrase_hits.get(best_key, 0),
                        ),
                    }
                )
            )

        results.sort(
            key=lambda track: (
                track.lyric_match_type == "exact",
                track.lyric_match_type == "phonetic",
                float(track.lyric_match_score or 0.0),
            ),
            reverse=True,
        )
        return results[:top_k] if top_k is not None else results
