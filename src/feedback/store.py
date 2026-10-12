"""Mongo 피드백 저장. 검색 기록은 불변, 선택 변경은 문서 하나의 CAS로 처리한다.

기존 가사 Mongo 연결과 분리해 짧은 시간 예산을 적용한다. 일반 검색 로그에 원문·토큰을 남기지 않는다.
수집만 하며 검색 경로나 곡 메타데이터를 수정하지 않는다.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone

from pymongo import MongoClient, ReturnDocument, timeout
from pymongo.errors import DuplicateKeyError

from src.backend.schemas.feedback import FeedbackReceipt, FeedbackRequest, FeedbackResponse

RETENTION_DAYS = 30
MAX_REVISIONS = 20
DB_BUDGET_SECONDS = 0.75


class FeedbackError(Exception):
    def __init__(self, status: int, message: str, current=None):
        self.status, self.message, self.current = status, message, current
        super().__init__(message)


def collection_enabled() -> bool:
    return os.getenv("FEEDBACK_COLLECTION_ENABLED", "0").lower() in {"1", "true", "yes"}


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _response(document) -> FeedbackResponse:
    return FeedbackResponse(
        search_id=document["_id"], revision=document["revision"],
        status=document["status"], selected_song_id=document["selected_song_id"],
        reason=document["reason"],
    )


class FeedbackStore:
    def __init__(self, database, client=None):
        self.runs = database["search_runs"]
        self.feedback = database["search_feedback"]
        self.client = client
        self._indexed = False
        self._index_lock = threading.Lock()

    def ensure_indexes(self):
        if self._indexed:
            return
        # 첫 검색이 경합해도 색인은 한 번만 만든다. 잠금 대기도 짧게 제한한다.
        if not self._index_lock.acquire(timeout=0.05):
            raise TimeoutError("feedback index initialization busy")
        try:
            if not self._indexed:
                self.runs.create_index("expires_at", expireAfterSeconds=0)
                self.feedback.create_index("expires_at", expireAfterSeconds=0)
                self._indexed = True
        finally:
            self._index_lock.release()

    def record_search(self, snapshot: dict, parent=None) -> FeedbackReceipt:
        with timeout(DB_BUDGET_SECONDS):
            self.ensure_indexes()
            now = datetime.now(timezone.utc)
            receipt = FeedbackReceipt(search_id=uuid.uuid4().hex, token=secrets.token_urlsafe(32))
            previous = None
            if parent:
                previous = self.runs.find_one({"_id": parent.search_id, "expires_at": {"$gt": now}})
                if (previous is None or previous["query"] != snapshot["query"]
                        or not hmac.compare_digest(previous["token_hash"], _digest(parent.token))):
                    previous = None  # 잘못된 연결이 검색이나 새 피드백 수집을 막지는 않는다.
            self.runs.insert_one({
                **snapshot, "_id": receipt.search_id, "token_hash": _digest(receipt.token),
                "conversation_id": previous["conversation_id"] if previous else receipt.search_id,
                "parent_search_id": previous["_id"] if previous else None,
                "created_at": now, "expires_at": now + timedelta(days=RETENTION_DAYS),
            })
            return receipt

    def submit(self, request: FeedbackRequest) -> FeedbackResponse:
        with timeout(DB_BUDGET_SECONDS):
            now = datetime.now(timezone.utc)
            run = self.runs.find_one({"_id": request.search_id, "expires_at": {"$gt": now}})
            if run is None or not hmac.compare_digest(run["token_hash"], _digest(request.token)):
                raise FeedbackError(404, "이 검색의 제출 권한이 없거나 보관 기간이 지났습니다.")
            if request.action == "select" and request.selected_song_id not in run["shown_ids"]:
                raise FeedbackError(422, "이 검색에서 표시한 곡만 선택할 수 있습니다.")
            payload = {"action": request.action, "selected_song_id": request.selected_song_id, "reason": request.reason}
            event = {**payload, "submission_id": request.submission_id, "created_at": now}
            current = self.feedback.find_one({"_id": request.search_id})
            if current is not None:
                replay = self._check_revision(current, request, payload)
                if replay:
                    return replay
                if request.action == "withdraw" and current["status"] == "withdrawn":
                    return _response(current)
            elif request.expected_revision != 0 or request.action == "withdraw":
                raise FeedbackError(409, "먼저 정답을 제출해 주세요.")

            revision = request.expected_revision + 1
            # 수정 한도를 다 써도 마지막 철회 한 번은 허용한다.
            if revision > MAX_REVISIONS and request.action != "withdraw":
                raise FeedbackError(409, "이 검색의 수정 횟수를 모두 사용했습니다.", _response(current))
            selection = {
                "revision": revision, "status": "withdrawn" if request.action == "withdraw" else "pending",
                "selected_song_id": request.selected_song_id, "reason": request.reason, "updated_at": now,
                # 수정 때 이전 검토·색인 결과를 재사용하지 않도록 항상 초기화한다.
                "review": None, "reusable_text": None,
            }
            event["revision"] = revision
            if current is None:
                document = {
                    "_id": request.search_id, "search_id": request.search_id, "source": "shown_result",
                    "created_at": now, "expires_at": run["expires_at"], "history": [event], **selection,
                }
                try:
                    self.feedback.insert_one(document)
                    return _response(document)
                except DuplicateKeyError:
                    pass  # 같은 검색의 첫 제출이 경합했다. 아래에서 재시도인지 충돌인지 확인한다.
            else:
                updated = self.feedback.find_one_and_update(
                    {"_id": request.search_id, "revision": request.expected_revision},
                    {"$set": selection, "$push": {"history": event}},
                    return_document=ReturnDocument.AFTER,
                )
                if updated is not None:
                    return _response(updated)
            current = self.feedback.find_one({"_id": request.search_id})
            if current is not None:
                replay = self._check_revision(current, request, payload)
                if replay:
                    return replay
            raise FeedbackError(409, "다른 제출이 먼저 저장됐습니다. 확인 후 다시 제출해 주세요.",
                                _response(current) if current else None)

    @staticmethod
    def _check_revision(current, request, payload):
        for event in current["history"]:
            if event["submission_id"] == request.submission_id:
                if any(event[key] != value for key, value in payload.items()):
                    raise FeedbackError(409, "같은 제출 번호로 내용을 바꿀 수 없습니다.", _response(current))
                # 응답 유실 뒤의 재시도. 이후 수정이 있었다면 현재 상태를 돌려준다.
                return _response(current)
        if current["revision"] != request.expected_revision:
            raise FeedbackError(409, "저장된 선택이 바뀌었습니다. 확인 후 다시 제출해 주세요.", _response(current))
        return None


_store = None
_store_lock = threading.Lock()


def get_feedback_store() -> FeedbackStore:
    global _store
    with _store_lock:
        if _store is None:
            from src.common.mongodb import DB_NAME, MONGO_URI

            client = MongoClient(
                MONGO_URI, connect=False, tz_aware=True, serverSelectionTimeoutMS=500,
                connectTimeoutMS=500, socketTimeoutMS=500, timeoutMS=750,
            )
            _store = FeedbackStore(client[DB_NAME], client)
        return _store


def close_feedback_store():
    global _store
    with _store_lock:
        if _store is not None and _store.client is not None:
            _store.client.close()
        _store = None
