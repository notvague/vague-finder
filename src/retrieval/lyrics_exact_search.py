"""전체 가사를 이용한 표면 구절 exact/fuzzy 후보 검색.

Pinecone의 형태소 BM25는 일반 키워드 검색용으로 유지하고, 사용자가 실제로 기억한
가사 구절은 MongoDB 원본 ``full_lyrics``에서 문자 순서를 보존해 찾는다. 곡 수가
수천 개 수준인 현재 카탈로그에서는 전체 문서를 메모리에 캐시하는 편이 단순하고
결정적이다.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.backend.schemas.query import LyricClue
from src.retrieval import timing
from src.backend.schemas.search import LyricSurfaceMatch, MatchingTrack
from src.retrieval.lyrics_query import (
    char_ngrams,
    normalize_lyric_surface,
    normalize_with_origins,
)

# 가사가 들어 있는 컬렉션. 인용의 출처 문자열이 여기서 나오므로 한 곳에서만 적는다.
COLLECTION = "songs"
LYRICS_FIELD = f"{COLLECTION}.lyrics_data.full_lyrics"


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


class LyricsExactSearchService:
    """MongoDB full_lyrics snapshot을 사용하는 exact/fuzzy 검색 서비스."""

    def __init__(
        self,
        documents: Optional[Iterable[Mapping[str, Any]]] = None,
        fuzzy_threshold: float = 0.78,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._seed_documents = list(documents) if documents is not None else None
        self._fuzzy_threshold = fuzzy_threshold
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._snapshot: Optional[LyricsSnapshot] = None
        self._load_lock = threading.Lock()

    def refresh(self) -> None:
        """다음 검색에서 다시 읽게 한다. **이미 나간 판은 그대로 산다** —
        그 판으로 검색한 요청은 끝까지 같은 판에서 인용한다."""
        with self._load_lock:
            self._snapshot = None

    def _source_documents(self) -> Iterable[Mapping[str, Any]]:
        if self._seed_documents is not None:
            return self._seed_documents

        # pymongo 의존성과 DB 연결은 실제 가사 검색이 실행될 때까지 지연한다.
        from src.common.mongodb import get_collection

        return get_collection(COLLECTION).find(
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
        )

    def snapshot(self) -> LyricsSnapshot:
        """지금 판. 없거나 오래됐으면 새로 읽는다. **블로킹이다.**

        `search()`가 쓴 판을 인용까지 들고 가려면 호출부가 판을 손에 쥐어야 한다.
        그래서 목록이 아니라 판 자체를 돌려준다.
        """
        return self._load()

    def _load(self) -> LyricsSnapshot:
        now = time.monotonic()
        current = self._snapshot
        if current is not None and now - current.loaded_at < self._cache_ttl_seconds:
            return current

        with self._load_lock:
            now = time.monotonic()
            current = self._snapshot
            if current is not None and now - current.loaded_at < self._cache_ttl_seconds:
                return current

            loaded: List[_LyricsDocument] = []
            for raw in self._source_documents():
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

            # 인용은 곡 하나를 바로 찾아야 한다. 결과 열 곡 때문에 905곡을
            # 다시 훑으면 근거를 펼칠 때마다 전체 순회가 한 번씩 더 돈다.
            self._snapshot = LyricsSnapshot(
                documents=tuple(loaded),
                by_id={document.track.id: document for document in loaded},
                loaded_at=time.monotonic(),
            )
            return self._snapshot

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

        # 스냅샷 적재와 훑기를 나눈다. 905곡을 Mongo에서 새로 읽는 것과, 이미 읽어 둔
        # 것을 구절로 훑는 것은 고치는 방법이 다르다(TTL 대 알고리즘).
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
