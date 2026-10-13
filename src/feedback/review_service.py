"""검토 화면이 쓰는 조회·갱신. Mongo만 읽고 쓴다 — **Qdrant를 열지 않는다.**

로컬 Qdrant 폴더는 한 프로세스만 열 수 있고 그 주인은 검색 서버다. 검토 화면은 별도 프로세스라
검토 결과를 Mongo에 적기만 하고, 색인은 검색 서버의 작업 루프(`worker`)가 그 기록을 보고 처리한다.
현재 곡 정보도 Qdrant가 아니라 Mongo `songs`에서 읽는다.

목록 응답에는 원질의·사유 본문을 싣지 않는다. 상세를 열 때만 한 건의 원문을 보낸다.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.feedback.review import (
    REVIEW_STATE_LABELS,
    REVIEW_STATES,
    as_utc,
    effective_review_state,
    is_expired,
    review_version,
    utcnow,
)
from src.feedback.settings import EMBEDDING_VERSION, MAX_REUSABLE_TEXT, MAX_REVIEW_NOTE
from src.feedback.store import TTL_READY, FeedbackStore, ReviewError

logger = logging.getLogger(__name__)

LIST_STATES = (*REVIEW_STATES, "withdrawn", "all")
MAX_PAGE_SIZE = 50
# 상세에 싣는 검색 분석 요약. 분석 전체(영문 질의·가중치)는 검토에 필요 없다.
_ANALYSIS_KEYS = ("intent_type", "song_title", "artist_name", "vocal_gender", "genre")


def _iso(value: Any) -> Optional[str]:
    moment = as_utc(value)
    return moment.isoformat() if moment else None


def _remaining_seconds(document: Mapping[str, Any], now: datetime) -> int:
    expires = as_utc(document.get("expires_at"))
    return max(0, int((expires - now).total_seconds())) if expires else 0


class MongoSongCatalog:
    """Mongo `songs`에서 현재 곡 정보를 읽는다(검색 서버가 가사를 읽는 곳과 같은 컬렉션)."""

    def __init__(self, collection: Any):
        self._collection = collection

    def lookup(self, song_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        ids = [str(song_id) for song_id in dict.fromkeys(song_ids) if song_id]
        if not ids:
            return {}
        found: Dict[str, Dict[str, Any]] = {}
        cursor = self._collection.find(
            {"$or": [{"_id": {"$in": ids}}, {"song_id": {"$in": ids}}]},
            {"song_id": 1, "title": 1, "artist": 1, "metadata": 1},
        )
        for document in cursor:
            metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
            song_id = str(document.get("song_id") or document.get("_id"))
            found[song_id] = {
                "title": document.get("title") or metadata.get("title"),
                "artist": document.get("artist") or metadata.get("artist"),
                "album": metadata.get("album"),
                "release_date": metadata.get("release_date"),
                "genre": metadata.get("genre"),
            }
        return found


class ReviewService:
    def __init__(self, store: FeedbackStore, catalog: Any, reviewer: str = "local"):
        self.store = store
        self.catalog = catalog
        self.reviewer = reviewer

    # ------------------------------------------------------------------
    def session(self) -> Dict[str, Any]:
        return {
            # 로컬 설정에서 읽은 이름이다. 로그인한 신원이 아니다 — 화면도 그렇게 적는다.
            "reviewer": self.reviewer,
            "reviewer_verified": False,
            "states": [{"value": state, "label": REVIEW_STATE_LABELS[state]} for state in REVIEW_STATES],
            "limits": {"reusable_text": MAX_REUSABLE_TEXT, "note": MAX_REVIEW_NOTE},
        }

    def list(self, state: str, page: int, page_size: int) -> Dict[str, Any]:
        if state not in LIST_STATES:
            raise ReviewError(422, "invalid_state", "알 수 없는 상태 필터입니다.")
        page, page_size = max(1, page), max(1, min(MAX_PAGE_SIZE, page_size))
        now = utcnow()
        documents, total = self.store.list_for_review(state, page=page, page_size=page_size, now=now)
        runs = self.store.runs_by_id([doc["_id"] for doc in documents], {"shown": 1})
        return {
            "state": state, "page": page, "page_size": page_size, "total": total,
            "items": [self._summary(doc, runs.get(doc["_id"]), now) for doc in documents],
        }

    def detail(self, search_id: str) -> Dict[str, Any]:
        now = utcnow()
        found = self.store.get_for_review(search_id, now)
        if found is None:
            raise ReviewError(404, "not_found", "피드백을 찾을 수 없습니다. 보관 기간이 지나 삭제됐을 수 있습니다.")
        return self._detail(*found, now)

    def review(self, search_id: str, *, state: str, reusable_text: str, note: str,
               expected_revision: int, expected_review_version: int) -> Dict[str, Any]:
        """검토 결과 저장. 충돌하면 `ReviewError.current`를 **화면이 그릴 수 있는 상세**로 바꿔 다시 던진다."""
        try:
            self.store.apply_review(
                search_id, state=state, reusable_text=reusable_text, note=note,
                expected_revision=expected_revision, expected_review_version=expected_review_version,
                reviewer=self.reviewer,
            )
        except ReviewError as exc:
            if exc.current is not None:
                exc.current = self._conflict_detail(search_id)
            raise
        return self.detail(search_id)

    def status(self) -> Dict[str, Any]:
        """자동 삭제 준비 상태와 색인 작업의 마지막 주기. 원문·연결 정보는 싣지 않는다."""
        now = utcnow()
        ttl = self.store.ttl_status()
        report: Dict[str, Any] = {
            "ttl": ttl, "ttl_ready": all(value == TTL_READY for value in ttl.values()),
            "expired_remaining": None, "index": None, "worker": None, "embedding_version": EMBEDDING_VERSION,
        }
        try:
            report["expired_remaining"] = self.store.count_expired(now)
            report["index"] = self.store.count_index_work(EMBEDDING_VERSION, now)
            worker = self.store.read_ops_status()
            if worker:
                report["worker"] = {
                    key: (_iso(value) if isinstance(value, datetime) else value)
                    for key, value in worker.items() if key != "_id"
                }
        except Exception as exc:
            logger.warning("review status unavailable: %s", type(exc).__name__)
            report["error"] = "storage_unavailable"
        return report

    # ------------------------------------------------------------------
    def _conflict_detail(self, search_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.detail(search_id)
        except ReviewError:
            return None

    @staticmethod
    def _shown(run: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        shown = (run or {}).get("shown")
        return [item for item in shown if isinstance(item, dict)] if isinstance(shown, list) else []

    def _summary(self, document: Mapping[str, Any], run: Optional[Mapping[str, Any]], now: datetime) -> Dict[str, Any]:
        song_id = document.get("selected_song_id")
        shown = next((item for item in self._shown(run) if str(item.get("id")) == str(song_id)), None)
        state = effective_review_state(document)
        index = document.get("index") if isinstance(document.get("index"), dict) else None
        return {
            "search_id": document["_id"],
            "user_status": document.get("status"),
            "review_state": state,
            "review_state_label": REVIEW_STATE_LABELS[state],
            "revision": document.get("revision"),
            "review_version": review_version(document),
            "selected_song": None if not song_id else {
                "id": str(song_id),
                # 검색 당시 화면에 보인 제목·순위. 현재 곡 정보는 상세에서 따로 읽는다.
                "title": (shown or {}).get("title"), "artist": (shown or {}).get("artist"),
                "rank": (shown or {}).get("rank"),
            },
            "has_reason": bool(document.get("reason")),
            "created_at": _iso(document.get("created_at")),
            "updated_at": _iso(document.get("updated_at")),
            "expires_at": _iso(document.get("expires_at")),
            "remaining_seconds": _remaining_seconds(document, now),
            "index_state": index.get("state") if index else None,
        }

    def _detail(self, document: Mapping[str, Any], run: Optional[Mapping[str, Any]], now: datetime) -> Dict[str, Any]:
        summary = self._summary(document, run, now)
        run = run or {}
        analysis = run.get("analysis") if isinstance(run.get("analysis"), dict) else {}
        era = analysis.get("release_era") if isinstance(analysis.get("release_era"), dict) else {}
        rejected = [str(song_id) for song_id in (run.get("rejected_ids") or [])]
        song_id = document.get("selected_song_id")

        current_song: Optional[Dict[str, Any]] = None
        rejected_songs: List[Dict[str, Any]] = [{"id": rid, "title": None, "artist": None} for rid in rejected]
        try:
            songs = self.catalog.lookup([*([str(song_id)] if song_id else []), *rejected])
        except Exception as exc:
            # 곡 정보를 못 읽어도 검토는 할 수 있어야 한다. 못 읽었다는 사실만 화면에 알린다.
            logger.warning("review song lookup unavailable: %s", type(exc).__name__)
            current_song = {"id": str(song_id), "found": None} if song_id else None
        else:
            if song_id:
                info = songs.get(str(song_id))
                current_song = {"id": str(song_id), "found": info is not None, **(info or {})}
            rejected_songs = [{"id": rid, **{k: (songs.get(rid) or {}).get(k) for k in ("title", "artist")}} for rid in rejected]

        review = document.get("review") if isinstance(document.get("review"), dict) else None
        state = summary["review_state"]
        reviewable = document.get("status") == "pending" and not is_expired(document, now)
        return {
            **summary,
            "query": run.get("query"),
            "run_found": bool(run),
            "context": {
                "turn": run.get("turn"),
                "answers": [a for a in (run.get("answers") or []) if isinstance(a, dict)],
                "asked_slots": list(run.get("asked_slots") or []),
                "rejected": rejected_songs,
                "release_era": {key: era.get(key) for key in ("start_year", "end_year")} if era else None,
                "analysis": {key: analysis.get(key) for key in _ANALYSIS_KEYS if analysis.get(key)},
            },
            "shown": [{key: item.get(key) for key in ("id", "rank", "title", "artist")} for item in self._shown(run)],
            "reason": document.get("reason") or "",
            "current_song": current_song,
            # 지금 유효한 검토만 싣는다. 사용자 수정으로 무효가 된 예전 검토는 대기로 읽힌다.
            "review": None if review is None or state != review.get("state") else {
                "state": state, "note": review.get("note") or "", "reviewed_by": review.get("reviewed_by"),
                "reviewed_at": _iso(review.get("reviewed_at")), "version": review.get("version"),
                "source_revision": review.get("source_revision"),
            },
            "reusable_text": document.get("reusable_text") or "",
            "review_history": [
                {**{key: event.get(key) for key in ("state", "version", "source_revision", "reviewed_by")},
                 "reviewed_at": _iso(event.get("reviewed_at"))}
                for event in (document.get("review_history") or []) if isinstance(event, dict)
            ],
            "user_revisions": len(document.get("history") or []),
            "reviewable": reviewable,
        }
