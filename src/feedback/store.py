"""Mongo 피드백 저장. 검색 기록은 불변, 선택 변경은 문서 하나의 CAS로 처리한다.

기존 가사 Mongo 연결과 분리해 짧은 시간 예산을 적용한다. 일반 검색 로그에 원문·토큰을 남기지 않는다.
곡 메타데이터는 수정하지 않는다.

`search_feedback` 문서 하나에 세 주체가 쓴다. 모두 **같은 문서의 조건 갱신**이라 서로를 덮어쓰지 않는다.

- 사용자 제출(`submit`) — `revision` 일치 조건. 선택이 바뀌면 `review`·`reusable_text`·`index`를 비운다.
- 운영자 검토(`apply_review`) — `revision`·`review_version` 일치 + 활성 제출 + 미만료 조건.
- 색인 작업(`mark_indexed` 등) — `revision`·`review_version` 일치 + 승인 상태 조건.

사용자 수정이 먼저 들어가면 검토 조건이 깨져 예전 선택을 승인할 수 없고, 검토가 먼저 들어가면
그 뒤의 사용자 수정이 승인을 비운다. 색인 작업의 완료 표시는 그 사이 판이 바뀌었으면 실패한다.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

from typing import Any, Dict, Iterable, List, Mapping, Optional

from pymongo import MongoClient, ReturnDocument, timeout
from pymongo.errors import DuplicateKeyError, OperationFailure

from src.backend.schemas.feedback import FeedbackReceipt, FeedbackRequest, FeedbackResponse
from src.feedback.review import (
    REVIEW_APPROVED,
    REVIEW_DECISIONS,
    REVIEW_PENDING,
    REVIEW_STATES,
    CaseVersion,
    clean_text,
    effective_review_state,
    is_expired,
    review_version,
    utcnow,
)
from src.feedback.settings import MAX_REUSABLE_TEXT, MAX_REVIEW_NOTE, index_worker_enabled

logger = logging.getLogger(__name__)

RETENTION_DAYS = 30
MAX_REVISIONS = 20
DB_BUDGET_SECONDS = 0.75
# 기동할 때 TTL 색인을 만드는 데 주는 시간. 요청 경로 예산으로는 큰 컬렉션의 색인을 못 만든다.
INDEX_BUDGET_SECONDS = 10.0
# 검토 화면과 색인 작업의 DB 호출 한 번에 주는 시간. 검색 응답을 붙잡는 자리가 아니라 넉넉히 둔다.
OPS_BUDGET_SECONDS = 5.0
# 문서에 남기는 검토 결정 이력 수(상태·판·시각만, 문장은 남기지 않는다). 문서와 함께 만료된다.
REVIEW_HISTORY_LIMIT = 20
# TTL 색인을 다시 확인하는 간격. 색인이 사라졌으면 이때 다시 만들고, `/health`의 보고도 이 간격 안에서 따라온다.
TTL_RECHECK_SECONDS = 600.0
OPS_STATUS_COLLECTION = "feedback_ops_status"
OPS_STATUS_ID = "index_worker"

# TTL 색인 상태. ready만 "자동 삭제가 준비됐다"는 뜻이다.
TTL_READY = "ready"
TTL_MISSING = "missing"                # expires_at 색인이 없다
TTL_NOT_TTL = "not_ttl"                # 색인은 있는데 TTL이 아니다 — 같은 키라 TTL 색인을 새로 못 만든다
TTL_WRONG_SECONDS = "wrong_seconds"    # expireAfterSeconds가 0이 아니다 — 만료 시각보다 늦게 지워진다
TTL_UNAUTHORIZED = "unauthorized"      # 색인 목록을 읽을 권한이 없다
TTL_UNREACHABLE = "unreachable"        # 저장소에 닿지 못했거나 그 밖의 오류
TTL_UNKNOWN = "unknown"                # 아직 확인하지 않았다


class FeedbackError(Exception):
    def __init__(self, status: int, message: str, current=None, *, code=None):
        self.status, self.message, self.current = status, message, current
        self.code = code
        super().__init__(message)


class ReviewError(Exception):
    """검토 갱신 거부. `current`는 충돌했을 때의 최신 문서다(화면이 다시 보여준다)."""

    def __init__(self, status: int, code: str, message: str, current: Optional[dict] = None):
        self.status, self.code, self.message, self.current = status, code, message, current
        super().__init__(message)


def review_filter(state: str, now: datetime) -> Dict[str, Any]:
    """검토 목록의 상태 필터. **만료된 문서는 어느 목록에도 넣지 않는다**(TTL 삭제가 늦어도)."""
    alive: Dict[str, Any] = {"expires_at": {"$gt": now}}
    if state == "all":
        return alive
    if state == "withdrawn":
        return {**alive, "status": "withdrawn"}
    if state == REVIEW_PENDING:
        # review가 없거나(null·누락) 결정이 아닌 값이면 대기다. $nin은 필드가 없는 문서도 맞춘다.
        return {**alive, "status": "pending", "review.state": {"$nin": list(REVIEW_DECISIONS)}}
    if state in REVIEW_DECISIONS:
        return {**alive, "status": "pending", "review.state": state}
    raise ValueError(f"알 수 없는 검토 상태: {state}")


def _case_filter(case: CaseVersion, now: datetime) -> Dict[str, Any]:
    """색인 작업의 조건 갱신 필터 — **작업이 본 그 판이 지금도 유효할 때만** 맞는다."""
    return {
        "_id": case.feedback_id, "revision": case.source_revision, "review_version": case.review_version,
        "status": "pending", "review.state": REVIEW_APPROVED, "expires_at": {"$gt": now},
    }


def _ttl_state(indexes: Mapping[str, Any]) -> str:
    """`index_information()` 결과에서 `expires_at` TTL 색인의 상태를 읽는다."""
    found = TTL_MISSING
    for info in indexes.values():
        keys = [name for name, _ in (info.get("key") or [])]
        if keys != ["expires_at"]:
            continue
        seconds = info.get("expireAfterSeconds")
        if seconds is None:
            found = TTL_NOT_TTL
        elif seconds == 0:
            return TTL_READY
        else:
            found = TTL_WRONG_SECONDS
    return found


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
    def __init__(self, database, client=None, *, ops_budget: float = OPS_BUDGET_SECONDS):
        self.runs = database["search_runs"]
        self.feedback = database["search_feedback"]
        self.client = client
        self.ops_budget = ops_budget
        self._database = database
        # 이 저장소가 TTL 색인을 확인한 결과를 프로세스의 보고(`ttl_report`)에 쓰는가. 서버가 쓰는 저장소만
        # 켠다(`get_feedback_store`·`prepare_indexes`) — 시험이나 도구가 만든 저장소는 보고를 바꾸지 않는다.
        self.reports_ttl = False
        self._indexed = False
        self._verified_at = 0.0     # 색인 생성 명령이 마지막으로 성공한 시각(monotonic)
        self._index_lock = threading.Lock()

    def ensure_indexes(self) -> bool:
        """TTL 색인을 만든다. **만들지 못해도 기록을 막지 않는다** — 됐는지만 돌려준다.

        이 색인은 보관 기간이 지난 문서를 지우는 용도다. 만료 판정은 조회할 때 `expires_at`으로
        직접 하므로 색인이 아직 없어도 기록과 제출은 정확하다. 예전에는 다른 요청이 만드는 동안
        0.05초만 기다리다 예외를 내서, 기동 직후 동시에 들어온 검색이 제출 권한 없이 나갔다.

        다른 요청이 만드는 중이면 기다리지 않고 지나간다. 실패하면 다음 기록에서 다시 시도한다.
        요청 경로에서는 기록 예산(`DB_BUDGET_SECONDS`) 안에서만 돌기 때문에 색인 생성이 그보다
        오래 걸리는 컬렉션은 여기서 끝내 못 만든다 — 기동할 때 넉넉한 예산으로 한 번 만든다
        (`prepare_indexes`).

        **한 번 됐다고 끝까지 믿지 않는다.** `TTL_RECHECK_SECONDS`마다 생성 명령을 다시 보낸다 — 같은 색인이
        있으면 아무 일도 하지 않는 명령이고, 누가 지웠으면 다시 만든다. 그 결과로 `/health`의 TTL 보고도
        고친다: 기동할 때 저장소에 닿지 못해 `unreachable`로 남았더라도 여기서 성공하면 `ready`가 되고,
        준비됐다고 적혀 있었는데 실패하면 `unknown`으로 내린다(색인 작업을 켜지 않은 서버에서 보고가 기동
        시점 값에 굳지 않게 — 리뷰 FR-04).
        """
        if self._indexed and time.monotonic() - self._verified_at < TTL_RECHECK_SECONDS:
            return True
        if not self._index_lock.acquire(blocking=False):
            return self._indexed
        try:
            # 이 옵션으로 생성 명령이 성공했다는 것은 두 컬렉션에 `expires_at` TTL(0초) 색인이 있다는 뜻이다.
            # 같은 키에 다른 옵션의 색인이 있으면 Mongo가 명령을 거부한다.
            self.runs.create_index("expires_at", expireAfterSeconds=0)
            self.feedback.create_index("expires_at", expireAfterSeconds=0)
            self._indexed = True
            self._verified_at = time.monotonic()
            _note_ttl_verified(self, True)
        except Exception as exc:
            self._indexed = False
            _note_ttl_verified(self, False)
            # DB 예외에는 URI·명령 문서가 포함될 수 있어 예외 종류만 남긴다.
            logger.warning("feedback TTL index not ready: %s", type(exc).__name__)
        finally:
            self._index_lock.release()
        return self._indexed

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
                raise FeedbackError(409, "이 검색의 수정 횟수를 모두 사용했습니다.", _response(current),
                                    code="revision_limit")
            selection = {
                "revision": revision, "status": "withdrawn" if request.action == "withdraw" else "pending",
                "selected_song_id": request.selected_song_id, "reason": request.reason, "updated_at": now,
                # 수정 때 이전 검토·색인 결과를 재사용하지 않도록 항상 초기화한다. `review_version`은 그대로 둔다 —
                # 되돌리면 같은 번호가 다른 문장을 가리키게 된다. 남은 Qdrant 포인트는 검색이 걸러내고
                # (제출 판 불일치) 색인 작업이 지운다.
                "review": None, "reusable_text": None, "index": None,
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

    # ------------------------------------------------------------------
    # 검토 — 로컬 검토 화면(`review_app`)만 부른다. 공개 API에는 연결하지 않는다.
    # ------------------------------------------------------------------

    def list_for_review(self, state: str, *, page: int, page_size: int,
                        now: Optional[datetime] = None) -> tuple[List[dict], int]:
        """검토 목록 한 쪽과 전체 건수. 만료가 가까운 것부터 — 기한 안에 볼 수 있게."""
        query = review_filter(state, now or utcnow())
        with timeout(self.ops_budget):
            total = self.feedback.count_documents(query)
            cursor = (
                self.feedback.find(query, {"history": 0, "review_history": 0})
                .sort([("expires_at", 1), ("_id", 1)])
                .skip(max(0, page - 1) * page_size)
                .limit(page_size)
            )
            return list(cursor), total

    def runs_by_id(self, ids: Iterable[str], projection: Optional[dict] = None) -> Dict[str, dict]:
        """검색 기록 묶음. **제출 토큰 해시는 어떤 경우에도 돌려주지 않는다.**"""
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return {}
        with timeout(self.ops_budget):
            found = self.runs.find({"_id": {"$in": wanted}}, projection or {"token_hash": 0})
            return {doc["_id"]: doc for doc in found}

    def get_for_review(self, search_id: str, now: Optional[datetime] = None) -> Optional[tuple[dict, Optional[dict]]]:
        """피드백 문서와 그 검색 기록. 없거나 **만료됐으면 None** — 지워지기 전이라도 내용을 보여주지 않는다."""
        now = now or utcnow()
        with timeout(self.ops_budget):
            feedback = self.feedback.find_one({"_id": search_id})
            if feedback is None or is_expired(feedback, now):
                return None
            return feedback, self.runs.find_one({"_id": search_id}, {"token_hash": 0})

    def apply_review(self, search_id: str, *, state: str, reusable_text: str, note: str,
                     expected_revision: int, expected_review_version: int, reviewer: str,
                     now: Optional[datetime] = None) -> dict:
        """검토 결과를 저장한다. 화면이 본 제출 판·검토 판이 지금도 최신일 때만 반영된다.

        조건은 한 번의 `find_one_and_update`에 모두 건다 — 사용자의 수정·철회와 다른 검토가 그 사이에
        들어왔으면 갱신이 일어나지 않고 `ReviewError(409)`로 최신 문서를 돌려준다. 만료된 사례도 같은
        조건에서 걸러진다. 검토 판은 저장할 때마다 오른다(문장만 고쳐도, 승인을 취소해도).

        승인이면 같은 갱신에서 `index.state="pending"`을 함께 쓴다. 승인 저장과 색인 예약이 한 문서의
        한 번의 쓰기라, 둘 사이에서 죽어 색인이 영영 빠지는 일이 없다.
        """
        now = now or utcnow()
        if state not in REVIEW_STATES:
            raise ReviewError(422, "invalid_state", "검토 결과는 대기·승인·보류·제외 중 하나여야 합니다.")
        text, note = clean_text(reusable_text), clean_text(note)
        if len(text) > MAX_REUSABLE_TEXT:
            raise ReviewError(422, "text_too_long", f"재사용 단서는 {MAX_REUSABLE_TEXT}자까지 적을 수 있습니다.")
        if len(note) > MAX_REVIEW_NOTE:
            raise ReviewError(422, "note_too_long", f"검토 사유는 {MAX_REVIEW_NOTE}자까지 적을 수 있습니다.")
        if state == REVIEW_APPROVED and not text:
            raise ReviewError(422, "text_required", "승인하려면 검색에 재사용할 단서를 적어야 합니다.")
        reviewer = clean_text(reviewer)[:80] or "local"
        version = expected_review_version + 1
        review = {
            "state": state, "source_revision": expected_revision, "version": version,
            "reviewed_by": reviewer, "reviewed_at": now, "note": note,
        }
        index = None
        if state == REVIEW_APPROVED:
            index = {
                "state": "pending", "source_revision": expected_revision, "review_version": version,
                "embedding_version": None, "attempts": 0, "next_attempt_at": None, "error": None,
                "updated_at": now,
            }
        event = {key: review[key] for key in ("state", "version", "source_revision", "reviewed_by", "reviewed_at")}
        query = {
            "_id": search_id, "revision": expected_revision, "status": "pending", "expires_at": {"$gt": now},
            # 검토한 적 없는 문서에는 이 필드가 없다(#47 이전 자료 포함). 0과 없음을 같은 것으로 맞춘다.
            "review_version": expected_review_version if expected_review_version > 0 else {"$in": [0, None]},
        }
        update = {
            "$set": {"review": review, "reusable_text": text or None, "review_version": version, "index": index},
            "$push": {"review_history": {"$each": [event], "$slice": -REVIEW_HISTORY_LIMIT}},
        }
        with timeout(self.ops_budget):
            updated = self.feedback.find_one_and_update(query, update, return_document=ReturnDocument.AFTER)
            if updated is not None:
                return updated
            current = self.feedback.find_one({"_id": search_id})
        if current is None:
            raise ReviewError(404, "not_found", "피드백을 찾을 수 없습니다. 보관 기간이 지나 삭제됐을 수 있습니다.")
        if is_expired(current, now):
            raise ReviewError(410, "expired", "보관 기간이 지난 사례는 검토할 수 없습니다.")
        if current.get("status") != "pending":
            raise ReviewError(409, "withdrawn", "사용자가 철회한 사례는 검토할 수 없습니다.", current)
        if current.get("revision") != expected_revision:
            raise ReviewError(409, "stale_revision", "사용자가 선택을 바꿨습니다. 최신 내용을 확인해 주세요.", current)
        saved = current.get("review")
        if (review_version(current) == version and isinstance(saved, Mapping)
                and all(saved.get(key) == review[key] for key in ("state", "source_revision", "version", "reviewed_by", "note"))
                and (current.get("reusable_text") or None) == (text or None)):
            return current  # 저장은 됐는데 응답을 잃은 뒤의 재시도. 같은 내용이면 한 번만 반영된 것으로 본다.
        raise ReviewError(409, "stale_review", "다른 검토가 먼저 저장됐습니다. 최신 내용을 확인해 주세요.", current)

    # ------------------------------------------------------------------
    # 색인 작업 — 승인 사례를 Qdrant에 올리고 그 결과를 문서에 남긴다.
    # ------------------------------------------------------------------

    _CASE_PROJECTION = {"history": 0, "review_history": 0, "reason": 0}

    def find_index_work(self, embedding_version: str, limit: int, now: Optional[datetime] = None) -> List[dict]:
        """색인해야 할 승인 사례. 대기·재시도 시각이 된 실패·임베딩 판이 다른 완료·색인 기록이 없는 승인."""
        now = now or utcnow()
        query = {
            "status": "pending", "review.state": REVIEW_APPROVED, "expires_at": {"$gt": now},
            "$or": [
                {"index": None},
                {"index.state": "pending"},
                {"index.state": "failed", "index.next_attempt_at": {"$lte": now}},
                {"index.state": "indexed", "index.embedding_version": {"$ne": embedding_version}},
            ],
        }
        with timeout(self.ops_budget):
            cursor = self.feedback.find(query, self._CASE_PROJECTION).sort([("expires_at", 1), ("_id", 1)]).limit(limit)
            return list(cursor)

    def count_index_work(self, embedding_version: str, now: Optional[datetime] = None) -> Dict[str, int]:
        """상태 표시용 건수. 실패는 재시도 시각과 무관하게 센다."""
        now = now or utcnow()
        approved = {"status": "pending", "review.state": REVIEW_APPROVED, "expires_at": {"$gt": now}}
        with timeout(self.ops_budget):
            return {
                "approved": self.feedback.count_documents(approved),
                "indexed": self.feedback.count_documents(
                    {**approved, "index.state": "indexed", "index.embedding_version": embedding_version}),
                "failed": self.feedback.count_documents({**approved, "index.state": "failed"}),
            }

    def fetch_cases(self, ids: Iterable[str], *, budget_seconds: Optional[float] = None) -> Dict[str, dict]:
        """피드백 문서 묶음(사유·이력 제외). 검색과 색인 작업이 최신 상태를 확인할 때 쓴다."""
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return {}
        with timeout(self.ops_budget if budget_seconds is None else budget_seconds):
            return {doc["_id"]: doc for doc in self.feedback.find({"_id": {"$in": wanted}}, self._CASE_PROJECTION)}

    def mark_indexed(self, case: CaseVersion, embedding_version: str, now: Optional[datetime] = None) -> bool:
        """색인 완료 표시. **작업이 올린 판이 지금도 유효한 최신 판일 때만** 반영된다.

        늦게 끝난 예전 판 작업은 여기서 False를 받는다 — 최신 판을 완료로 표시하지 못한다.
        """
        now = now or utcnow()
        index = {
            "state": "indexed", "source_revision": case.source_revision, "review_version": case.review_version,
            "embedding_version": embedding_version, "attempts": 0, "next_attempt_at": None, "error": None,
            "updated_at": now,
        }
        with timeout(self.ops_budget):
            return self.feedback.update_one(_case_filter(case, now), {"$set": {"index": index}}).matched_count == 1

    def mark_index_failed(self, case: CaseVersion, *, error: str, attempts: int, next_attempt_at: datetime,
                          now: Optional[datetime] = None) -> bool:
        """색인 실패 기록과 다음 시도 시각. 판이 바뀌었으면 기록하지 않는다(새 판은 이미 대기다)."""
        now = now or utcnow()
        index = {
            "state": "failed", "source_revision": case.source_revision, "review_version": case.review_version,
            "embedding_version": None, "attempts": attempts, "next_attempt_at": next_attempt_at, "error": error,
            "updated_at": now,
        }
        with timeout(self.ops_budget):
            return self.feedback.update_one(_case_filter(case, now), {"$set": {"index": index}}).matched_count == 1

    def reset_index_pending(self, case: CaseVersion, now: Optional[datetime] = None) -> bool:
        """완료로 표시된 사례를 다시 색인 대기로 돌린다(포인트가 없거나 예전 판으로 덮였을 때의 복구)."""
        now = now or utcnow()
        index = {
            "state": "pending", "source_revision": case.source_revision, "review_version": case.review_version,
            "embedding_version": None, "attempts": 0, "next_attempt_at": None, "error": None, "updated_at": now,
        }
        with timeout(self.ops_budget):
            query = {**_case_filter(case, now), "index.state": "indexed"}
            return self.feedback.update_one(query, {"$set": {"index": index}}).matched_count == 1

    def find_indexed(self, *, after_id: str, limit: int, now: Optional[datetime] = None) -> List[dict]:
        """색인 완료로 표시된 승인 사례를 `_id` 순서로 한 묶음. Qdrant에 실제로 있는지 대조할 때 쓴다."""
        now = now or utcnow()
        query = {
            "_id": {"$gt": after_id}, "status": "pending", "review.state": REVIEW_APPROVED,
            "index.state": "indexed", "expires_at": {"$gt": now},
        }
        with timeout(self.ops_budget):
            return list(self.feedback.find(query, self._CASE_PROJECTION).sort([("_id", 1)]).limit(limit))

    # ------------------------------------------------------------------
    # 만료·운영 상태
    # ------------------------------------------------------------------

    def ttl_status(self) -> Dict[str, str]:
        """두 컬렉션의 `expires_at` TTL 색인 상태. 예외를 내지 않고 상태 코드로 돌려준다."""
        report: Dict[str, str] = {}
        for name, collection in (("search_runs", self.runs), ("search_feedback", self.feedback)):
            try:
                with timeout(self.ops_budget):
                    report[name] = _ttl_state(collection.index_information())
            except OperationFailure as exc:
                report[name] = TTL_UNAUTHORIZED if exc.code == 13 else TTL_UNREACHABLE
            except Exception:
                report[name] = TTL_UNREACHABLE
        return report

    def count_expired(self, now: Optional[datetime] = None) -> Dict[str, int]:
        """만료됐는데 아직 남아 있는 문서 수. TTL 청소가 밀렸거나 색인이 없으면 0이 아니다."""
        query = {"expires_at": {"$lte": now or utcnow()}}
        with timeout(self.ops_budget):
            return {
                "search_runs": self.runs.count_documents(query),
                "search_feedback": self.feedback.count_documents(query),
            }

    def delete_expired(self, now: Optional[datetime] = None) -> Dict[str, int]:
        """만료된 문서를 직접 지운다. **TTL 색인이 준비되지 않았을 때의 대체 수단이다**(`indexer`).

        TTL이 지울 문서와 같은 조건(`expires_at`이 지남)이라 보관 기간을 줄이지 않는다.
        """
        query = {"expires_at": {"$lte": now or utcnow()}}
        with timeout(self.ops_budget):
            return {
                "search_feedback": self.feedback.delete_many(query).deleted_count,
                "search_runs": self.runs.delete_many(query).deleted_count,
            }

    def write_ops_status(self, payload: Mapping[str, Any]) -> None:
        """색인 작업의 마지막 주기 요약. 건수·시각·오류 종류만 — 원문·문장·연결 정보는 넣지 않는다."""
        with timeout(self.ops_budget):
            self._database[OPS_STATUS_COLLECTION].replace_one(
                {"_id": OPS_STATUS_ID}, {"_id": OPS_STATUS_ID, **payload}, upsert=True)

    def read_ops_status(self) -> Optional[dict]:
        with timeout(self.ops_budget):
            return self._database[OPS_STATUS_COLLECTION].find_one({"_id": OPS_STATUS_ID})


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
            _store.reports_ttl = True
        return _store


_TTL_COLLECTIONS = ("search_runs", "search_feedback")
_ttl_report: Dict[str, str] = {}
_ttl_checked_at: Optional[datetime] = None
_ttl_report_lock = threading.Lock()


def ttl_report() -> Dict[str, str]:
    """마지막으로 확인한 TTL 색인 상태(컬렉션 → 상태 코드). 확인한 적 없으면 `unknown`."""
    with _ttl_report_lock:
        return dict(_ttl_report) or {name: TTL_UNKNOWN for name in _TTL_COLLECTIONS}


def ttl_checked_at() -> Optional[datetime]:
    """TTL 상태를 마지막으로 확인한 시각. 보고가 얼마나 낡았는지 알 수 있게 `/health`에 함께 싣는다."""
    with _ttl_report_lock:
        return _ttl_checked_at


def ttl_ready() -> bool:
    return all(state == TTL_READY for state in ttl_report().values())


def _set_ttl_report(report: Mapping[str, str]) -> None:
    global _ttl_checked_at
    with _ttl_report_lock:
        _ttl_report.clear()
        _ttl_report.update(report)
        _ttl_checked_at = utcnow()


def _note_ttl_verified(store: "FeedbackStore", created: bool) -> None:
    """이 프로세스의 저장소가 TTL 색인 생성 명령을 다시 보낸 결과를 보고에 반영한다.

    성공하면 둘 다 `ready`. 실패하면 `ready`라고 적혀 있던 칸만 `unknown`으로 내린다 — 기동할 때 읽은
    구체적인 사유(`not_ttl`·`wrong_seconds`·`unauthorized`)는 지우지 않는다. 서버가 쓰는 저장소
    (`reports_ttl`)가 아니면 보고를 바꾸지 않는다.
    """
    if not store.reports_ttl:
        return
    if created:
        _set_ttl_report({name: TTL_READY for name in _TTL_COLLECTIONS})
        return
    current = ttl_report()
    _set_ttl_report({name: TTL_UNKNOWN if state == TTL_READY else state for name, state in current.items()})


def refresh_ttl_report(store: Optional["FeedbackStore"] = None) -> Dict[str, str]:
    """TTL 색인 상태를 다시 읽어 둔다. 실패는 상태 코드로 남는다."""
    try:
        report = (store or get_feedback_store()).ttl_status()
    except Exception as exc:
        logger.warning("feedback TTL status unavailable: %s", type(exc).__name__)
        report = {name: TTL_UNREACHABLE for name in _TTL_COLLECTIONS}
    _set_ttl_report(report)
    return dict(report)


def prepare_indexes() -> bool:
    """기동할 때 TTL 색인을 만들어 두고, **실제로 준비됐는지 확인해 둔다**(`ttl_report`).

    수집과 색인 작업이 모두 꺼져 있으면 연결하지 않는다. 색인 작업만 켠 자리(수집은 껐지만 쌓인 자료를
    정리하는 중)에서도 만든다 — 만료 정리가 수집 스위치에 묶이지 않게.

    요청 경로의 0.75초 예산과 달리 `INDEX_BUDGET_SECONDS`를 준다. 실패해도 서버는 뜬다 —
    기록할 때마다 다시 시도한다. 돌려주는 값은 "이 프로세스가 색인 생성 명령을 끝냈는가"이고,
    권한 부족·옵션이 다른 기존 색인 같은 사유는 `ttl_report()`가 말한다.
    """
    if not (collection_enabled() or index_worker_enabled()):
        return False
    try:
        with timeout(INDEX_BUDGET_SECONDS):
            store = get_feedback_store()
            store.reports_ttl = True   # 이 서버의 저장소다. 뒤의 재확인 결과가 `/health` 보고를 따라가게 한다
            created = store.ensure_indexes()
    except Exception as exc:
        logger.warning("feedback TTL index not ready: %s", type(exc).__name__)
        return False
    report = refresh_ttl_report(store)
    if any(state != TTL_READY for state in report.values()):
        logger.warning("feedback TTL index is not ready: %s", report)
    return created


def close_feedback_store():
    global _store
    with _store_lock:
        if _store is not None and _store.client is not None:
            _store.client.close()
        _store = None
