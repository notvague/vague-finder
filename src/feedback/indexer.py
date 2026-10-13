"""승인 사례 색인과 만료·무효 사례 정리의 한 주기(`run_once`). 요청 경로 밖에서 돈다(`worker`).

기준은 Mongo다. 한 주기는 다섯 단계이고, 한 단계가 실패해도 나머지는 계속한다.

1. **만료 포인트 삭제** — Qdrant의 `expires_at_ts`로 지운다. Mongo 원본이 TTL로 먼저 사라졌어도 지워진다.
2. **포인트 대조** — 포인트를 한 묶음 읽어 Mongo 최신 문서와 맞춘다. 원본 없음·철회·미승인·예전 판이면
   그 판의 포인트만 지운다. 문서는 유효한데 포인트가 예전 판이면 색인 대기로 되돌린다.
3. **색인** — 색인할 승인 사례를 임베딩해 올린다. 올리기 직전에 Mongo를 다시 읽어 판이 그대로인지 보고,
   올린 뒤 완료 표시도 판 조건으로 건다. 그 사이 판이 바뀌었으면 방금 올린 포인트를 지운다.
4. **완료 대조** — Mongo에 완료로 적힌 사례가 Qdrant에 그 판으로 있는지 본다. 없으면 색인 대기로 되돌린다
   (Qdrant 폴더를 새로 만들었거나, 늦은 작업이 포인트를 덮은 뒤 지워진 경우).
5. **Mongo 만료 문서** — TTL 색인이 준비되지 않았을 때만 만료 문서를 직접 지운다(TTL의 대체 수단).

**Mongo를 읽지 못한 주기에는 포인트를 지우지 않는다.** "문서가 없다"와 "읽지 못했다"를 구분할 수 없으면
유효한 사례를 지우게 된다. 그동안에도 검색은 Mongo 확인에 실패한 사례를 쓰지 않는다(`retrieval`).

재시작 복구에 따로 할 일은 없다. 색인할 것은 문서의 `index` 기록에서, 지울 것은 포인트 대조에서 매 주기
다시 찾는다. 같은 주기를 두 번 돌려도 결과가 같다 — 포인트 ID가 사례마다 정해져 있고 삭제는 판 조건이다.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.feedback.case_index import FeedbackCaseIndex
from src.feedback.review import CaseVersion, clean_text, reusable_case, utcnow
from src.feedback.settings import EMBEDDING_VERSION, WorkerSettings
from src.feedback.store import TTL_READY, FeedbackStore

logger = logging.getLogger(__name__)

# 운영 상태에 남기는 오류 종류. 예외 본문(URI·명령 문서가 들어갈 수 있다)은 남기지 않는다.
ERROR_STORE = "mongo_unavailable"
ERROR_QDRANT = "qdrant_error"
ERROR_EMBEDDING = "embedding_error"

RETRY_BASE_SECONDS = 60.0
RETRY_MAX_SECONDS = 3600.0


def retry_delay(attempts: int) -> float:
    """실패 횟수에 따른 다음 시도까지의 대기. 60초에서 시작해 두 배씩, 한 시간에서 멈춘다."""
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** max(0, min(attempts, 16) - 1)))


@dataclass
class CycleReport:
    indexed: int = 0                 # 올리고 완료로 표시한 사례
    index_failed: int = 0            # 임베딩·적재 실패로 재시도 예약한 사례
    skipped_stale: int = 0           # 작업 도중 판이 바뀌거나 무효가 돼 완료로 표시하지 않은 사례
    deleted_expired_points: int = 0
    deleted_stale_points: int = 0    # 원본 없음·철회·미승인·예전 판 포인트
    repaired: int = 0                # 색인 대기로 되돌린 사례
    mongo_expired_deleted: int = 0   # TTL 대신 직접 지운 만료 문서
    points: Optional[int] = None
    expired_points_remaining: Optional[int] = None       # 만료됐는데 Qdrant에 남은 포인트
    expired_documents_remaining: Optional[int] = None    # 만료됐는데 Mongo에 남은 문서(TTL 청소 대기 포함)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class FeedbackIndexer:
    def __init__(
        self,
        store: FeedbackStore,
        case_index: FeedbackCaseIndex,
        embed: Callable[[Sequence[str]], Any],
        settings: Optional[WorkerSettings] = None,
        clock: Callable[[], datetime] = utcnow,
    ):
        self.store = store
        self.case_index = case_index
        self.embed = embed
        self.settings = settings or WorkerSettings()
        self.clock = clock
        self._scroll_offset: Any = None
        self._verify_cursor = ""
        self._last_success_at: Optional[datetime] = None
        self._collection_ready = False

    # ------------------------------------------------------------------
    def run_once(self, stop: Optional[threading.Event] = None) -> CycleReport:
        report = CycleReport()
        stopped = (lambda: stop is not None and stop.is_set())
        steps: List[Tuple[str, Callable[[CycleReport, Callable[[], bool]], None]]] = [
            ("expired_points", self._delete_expired_points),
            ("reconcile_points", self._reconcile_points),
            ("index", self._index_pending),
            ("verify_indexed", self._verify_indexed),
            ("mongo_expired", self._delete_expired_documents),
        ]
        for name, step in steps:
            if stopped():
                break
            try:
                step(report, stopped)
            except _StepError as exc:
                self._note(report, exc.code)
                logger.warning("feedback index step %s failed: %s (%s)", name, exc.code, exc.cause)
            except Exception as exc:  # 한 단계가 죽어도 다음 단계(특히 만료 정리)는 돌아야 한다
                self._note(report, ERROR_QDRANT)
                # 컬렉션이 사라졌을 수 있다(폴더를 새로 만들었거나 누가 지웠다). 다음 단계에서 다시 확인해 만든다 —
                # 그러면 완료 대조가 빠진 사례를 색인 대기로 되돌린다.
                self._collection_ready = False
                logger.warning("feedback index step %s failed: %s", name, type(exc).__name__)
        if not stopped():
            # 종료 중이면 상태 기록(DB 호출 몇 번)을 건너뛴다 — 서버가 이 스레드를 기다리고 있다.
            self._write_status(report)
        return report

    @staticmethod
    def _note(report: CycleReport, code: str) -> None:
        if code not in report.errors:
            report.errors.append(code)

    def _ensure_collection(self) -> None:
        if not self._collection_ready:
            self.case_index.ensure()
            self._collection_ready = True

    # --- 1. 만료 포인트 ---------------------------------------------------
    def _delete_expired_points(self, report: CycleReport, stopped: Callable[[], bool]) -> None:
        self._ensure_collection()
        report.deleted_expired_points = self.case_index.delete_expired(self.clock().timestamp())

    # --- 2. 포인트 → Mongo 대조 -------------------------------------------
    def _reconcile_points(self, report: CycleReport, stopped: Callable[[], bool]) -> None:
        self._ensure_collection()
        payloads, next_offset = self.case_index.scroll(self._scroll_offset, self.settings.sweep_batch)
        ids = [str(payload["feedback_id"]) for payload in payloads if payload.get("feedback_id")]
        documents = self._store(lambda: self.store.fetch_cases(ids))   # 못 읽으면 이번 주기에는 지우지 않는다
        now = self.clock()
        for payload in payloads:
            feedback_id = payload.get("feedback_id")
            if not feedback_id or stopped():
                continue
            case = reusable_case(documents.get(str(feedback_id)), now)
            if case is not None and case.matches_payload(payload):
                continue
            revision, version = payload.get("source_revision"), payload.get("review_version")
            if isinstance(revision, int) and isinstance(version, int):
                self.case_index.delete_version(str(feedback_id), revision, version)
                report.deleted_stale_points += 1
            if case is not None and self._store(lambda case=case: self.store.reset_index_pending(case, now)):
                report.repaired += 1   # 문서는 유효한데 포인트가 예전 판이었다. 지금 판을 다시 올린다
        self._scroll_offset = next_offset

    # --- 3. 색인 -----------------------------------------------------------
    def _index_pending(self, report: CycleReport, stopped: Callable[[], bool]) -> None:
        now = self.clock()
        work = self._store(lambda: self.store.find_index_work(EMBEDDING_VERSION, self.settings.index_batch, now))
        if not work:
            return
        self._ensure_collection()
        size = self.settings.embed_batch
        for start in range(0, len(work), size):
            if stopped():
                return
            self._index_batch(work[start:start + size], report)

    def _index_batch(self, documents: List[dict], report: CycleReport) -> None:
        now = self.clock()
        batch: List[Tuple[CaseVersion, dict, str]] = []
        for document in documents:
            case = reusable_case(document, now)
            if case is None:
                report.skipped_stale += 1
                continue
            batch.append((case, document, clean_text(document.get("reusable_text"))))
        if not batch:
            return
        try:
            vectors = self.embed([text for _, _, text in batch])
            if len(vectors) != len(batch):
                raise ValueError("임베딩 수가 입력과 다르다")
            vectors = [[float(value) for value in vector] for vector in vectors]
            if not all(math.isfinite(value) for vector in vectors for value in vector):
                raise ValueError("유한하지 않은 임베딩")
        except Exception as exc:
            logger.warning("feedback case embedding failed: %s", type(exc).__name__)
            self._fail(batch, ERROR_EMBEDDING, report)
            return

        # 임베딩하는 동안 사용자가 고치거나 철회했을 수 있다. 올리기 직전에 최신 문서로 다시 본다.
        current = self._store(lambda: self.store.fetch_cases([case.feedback_id for case, _, _ in batch]))
        now = self.clock()
        entries = []
        for (case, document, _), vector in zip(batch, vectors):
            if reusable_case(current.get(case.feedback_id), now) != case:
                report.skipped_stale += 1
                continue
            entries.append((case, vector, current[case.feedback_id]["expires_at"]))
        if not entries:
            return
        try:
            self.case_index.upsert(entries, embedding_version=EMBEDDING_VERSION)
        except Exception as exc:
            logger.warning("feedback case upsert failed: %s", type(exc).__name__)
            wanted = {case for case, _, _ in entries}
            self._fail([item for item in batch if item[0] in wanted], ERROR_QDRANT, report)
            return
        for case, _, _ in entries:
            if self._store(lambda case=case: self.store.mark_indexed(case, EMBEDDING_VERSION, self.clock())):
                report.indexed += 1
                continue
            # 올리는 사이 판이 바뀌었거나 만료·철회됐다. 완료로 표시하지 않고, 방금 올린 이 판의 포인트만 지운다.
            # 새 판이 승인돼 있으면 그 문서는 이미 색인 대기라 다음 주기에 올라간다.
            report.skipped_stale += 1
            try:
                self.case_index.delete_version(case.feedback_id, case.source_revision, case.review_version)
            except Exception as exc:  # 못 지워도 검색이 걸러내고 다음 대조에서 지워진다
                logger.warning("feedback stale point cleanup failed: %s", type(exc).__name__)
                self._note(report, ERROR_QDRANT)

    def _fail(self, batch: Sequence[Tuple[CaseVersion, dict, str]], error: str, report: CycleReport) -> None:
        self._note(report, error)
        now = self.clock()
        for case, document, _ in batch:
            index = document.get("index") if isinstance(document.get("index"), dict) else {}
            same_version = index.get("review_version") == case.review_version and index.get("state") == "failed"
            attempts = (index.get("attempts") or 0) + 1 if same_version else 1
            try:
                self.store.mark_index_failed(
                    case, error=error, attempts=attempts,
                    next_attempt_at=now + timedelta(seconds=retry_delay(attempts)), now=now,
                )
            except Exception as exc:
                # 실패 기록을 못 남겨도 문서는 여전히 색인 대기다 — 다음 주기에 다시 잡힌다.
                logger.warning("feedback index failure not recorded: %s", type(exc).__name__)
                self._note(report, ERROR_STORE)
            report.index_failed += 1

    # --- 4. Mongo(완료) → 포인트 대조 --------------------------------------
    def _verify_indexed(self, report: CycleReport, stopped: Callable[[], bool]) -> None:
        now = self.clock()
        documents = self._store(lambda: self.store.find_indexed(
            after_id=self._verify_cursor, limit=self.settings.sweep_batch, now=now))
        if not documents:
            self._verify_cursor = ""
            return
        self._ensure_collection()
        payloads = self.case_index.retrieve([document["_id"] for document in documents])
        for document in documents:
            case = reusable_case(document, now)
            if case is None:
                continue
            payload = payloads.get(case.feedback_id)
            if payload is not None and case.matches_payload(payload) \
                    and payload.get("embedding_version") == (document.get("index") or {}).get("embedding_version"):
                continue
            if self._store(lambda case=case: self.store.reset_index_pending(case, now)):
                report.repaired += 1
        self._verify_cursor = documents[-1]["_id"] if len(documents) >= self.settings.sweep_batch else ""

    # --- 5. Mongo 만료 문서 -------------------------------------------------
    def _delete_expired_documents(self, report: CycleReport, stopped: Callable[[], bool]) -> None:
        from src.feedback import store as store_module

        ttl = store_module.refresh_ttl_report(self.store)
        if not all(state == TTL_READY for state in ttl.values()):
            # TTL이 지워 주지 않는 자리다. 보관 기간이 지난 문서를 같은 조건으로 직접 지운다.
            deleted = self._store(lambda: self.store.delete_expired(self.clock()))
            report.mongo_expired_deleted = sum(deleted.values())
        remaining = self._store(lambda: self.store.count_expired(self.clock()))
        report.expired_documents_remaining = sum(remaining.values())

    # ------------------------------------------------------------------
    def _store(self, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except Exception as exc:
            raise _StepError(ERROR_STORE, type(exc).__name__) from None

    def _write_status(self, report: CycleReport) -> None:
        """마지막 주기 요약을 Mongo에 남긴다(검토 화면이 읽는다). 건수·시각·오류 종류만."""
        now = self.clock()
        try:
            if self._collection_ready:
                report.points = self.case_index.count()
                report.expired_points_remaining = self.case_index.count_expired(now.timestamp())
        except Exception as exc:
            logger.warning("feedback point count failed: %s", type(exc).__name__)
            self._note(report, ERROR_QDRANT)
        try:
            if self._last_success_at is None:
                previous = self.store.read_ops_status() or {}
                self._last_success_at = previous.get("last_success_at")
            if report.ok:
                self._last_success_at = now
            from src.feedback import store as store_module

            self.store.write_ops_status({
                "last_cycle_at": now, "last_success_at": self._last_success_at,
                "last_error": report.errors[0] if report.errors else None,
                "cycle": {key: value for key, value in asdict(report).items() if key != "errors"},
                "errors": list(report.errors), "ttl": store_module.ttl_report(),
                "embedding_version": EMBEDDING_VERSION,
            })
        except Exception as exc:
            logger.warning("feedback ops status not written: %s", type(exc).__name__)
            self._note(report, ERROR_STORE)


class _StepError(Exception):
    def __init__(self, code: str, cause: str):
        self.code, self.cause = code, cause
        super().__init__(code)
