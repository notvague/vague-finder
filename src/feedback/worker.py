"""색인·정리 작업 루프. **검색 서버 프로세스 안의 스레드 하나**로 돈다.

로컬 Qdrant는 저장 폴더를 한 프로세스만 열 수 있다. 그래서 별도 worker 프로세스를 띄우지 않고, 폴더를
이미 열어 둔 검색 서버가 자기 클라이언트로 색인한다. 검토 화면(다른 프로세스)은 Mongo에 검토 결과만
적고, 이 루프가 주기마다 Mongo를 읽어 처리한다.

- 켜기: `FEEDBACK_INDEX_WORKER_ENABLED=1`. 수집(`FEEDBACK_COLLECTION_ENABLED`)·검색 반영
  (`FEEDBACK_RETRIEVAL_ENABLED`)과 독립이다 — 수집을 꺼도 쌓인 자료의 만료 정리는 계속 돈다.
- 기동하면 **바로 한 주기**를 돌린다. 꺼져 있던 동안 밀린 색인과 만료 정리가 여기서 처리된다.
- 그 뒤 `FEEDBACK_WORKER_INTERVAL_SECONDS`(기본 60초)마다 한 주기.

**종료.** `stop_and_wait()`는 멈춤 신호를 주고 스레드가 **끝날 때까지 기다린다.** 돌고 있는 주기는
지금 하던 묶음(임베딩·적재 한 번)까지 마치고 나온다. 서버는 이것이 돌아온 뒤에 Qdrant와 Mongo를 닫는다 —
asyncio 작업을 취소하는 것으로는 동기 스레드가 멈추지 않아서, 닫힌 클라이언트를 만지게 된다.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence

from src.feedback.review import utcnow
from src.feedback.settings import WorkerSettings, index_worker_enabled

logger = logging.getLogger(__name__)

# 주기를 시작하지도 못했을 때의 오류 코드(저장소·모델 준비 실패). 단계별 코드는 `indexer`에 있다.
ERROR_WORKER = "worker_unavailable"

# 상태: disabled → running → stopped. 주기 결과는 last_* 에 따로 남는다.
_state: Dict[str, Any] = {}
_lock = threading.Lock()
_stop = threading.Event()
_thread: Optional[threading.Thread] = None


def _reset_state() -> None:
    _state.clear()
    _state.update({
        "state": "disabled", "cycles": 0, "last_cycle_at": None, "last_success_at": None,
        "last_error": None, "last_report": None,
    })


_reset_state()


def enabled() -> bool:
    return index_worker_enabled()


def status() -> Dict[str, Any]:
    """공개 `/health`에 실어도 되는 값만 — 건수·시각·오류 종류. 원문·예외 본문·연결 정보는 없다."""
    with _lock:
        snapshot = dict(_state)
    for key in ("last_cycle_at", "last_success_at"):
        if isinstance(snapshot[key], datetime):
            snapshot[key] = snapshot[key].isoformat()
    return snapshot


def _default_indexer() -> Any:
    """검색 서버가 이미 쥔 Qdrant 클라이언트·KoE5 모델·Mongo 저장소를 그대로 쓴다. 새로 열지 않는다.

    돌려주는 것은 `indexer.FeedbackIndexer`다(모듈을 여기서 불러오므로 주석은 Any로 둔다).
    """
    from src.backend.api.dependencies import get_feedback_case_index, get_text_embedder
    from src.feedback.indexer import FeedbackIndexer
    from src.feedback.store import get_feedback_store

    settings = WorkerSettings.from_env()

    def embed(texts: Sequence[str]) -> Any:
        # 곡 텍스트와 같은 규약: E5 `passage: ` 접두사 + L2 정규화. 한 번에 작은 묶음만 넣어
        # 같은 모델을 쓰는 검색 질의 임베딩이 오래 밀리지 않게 한다.
        return get_text_embedder().embed_passages(list(texts), batch_size=settings.embed_batch, normalize=True)

    return FeedbackIndexer(get_feedback_store(), get_feedback_case_index(), embed, settings)


def _loop(build: Callable[[], Any], interval: float) -> None:
    indexer = None
    while not _stop.is_set():
        report = None
        error: Optional[str] = None
        try:
            if indexer is None:
                indexer = build()
            report = indexer.run_once(_stop)
            if report.errors:
                error = report.errors[0]
        except Exception as exc:
            # 저장소·모델을 준비하지 못했다. 로그에는 예외 종류만, 상태에는 고정 코드만 남기고
            # 다음 주기에 다시 시도한다.
            error = ERROR_WORKER
            logger.warning("feedback index worker cycle failed: %s", type(exc).__name__)
        now = utcnow()
        with _lock:
            _state["cycles"] += 1
            _state["last_cycle_at"] = now
            _state["last_error"] = error
            if error is None:
                _state["last_success_at"] = now
            if report is not None:
                _state["last_report"] = {
                    key: getattr(report, key) for key in (
                        "indexed", "index_failed", "skipped_stale", "deleted_expired_points",
                        "deleted_stale_points", "repaired", "mongo_expired_deleted", "points",
                        "expired_points_remaining", "expired_documents_remaining",
                    )
                }
        if _stop.wait(interval):
            break
    with _lock:
        _state["state"] = "stopped"


def start_background(build: Optional[Callable[[], Any]] = None, *, interval: Optional[float] = None) -> bool:
    """루프를 띄운다. 꺼져 있거나 이미 돌고 있으면 아무것도 하지 않는다."""
    global _thread
    if not enabled():
        with _lock:
            if _thread is None or not _thread.is_alive():
                _reset_state()
        return False
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        _stop.clear()
        _reset_state()
        _state["state"] = "running"
        seconds = WorkerSettings.from_env().interval_seconds if interval is None else interval
        _thread = threading.Thread(
            target=_loop, args=(build or _default_indexer, seconds), name="feedback_index_worker", daemon=True,
        )
        _thread.start()
    return True


def stop_and_wait(timeout: Optional[float] = None) -> bool:
    """멈춤 신호를 주고 스레드가 끝나기를 기다린다. 끝났으면(또는 돌고 있지 않았으면) True.

    기본은 **제한 없이 기다린다** — 돌고 있는 묶음이 Qdrant에 쓰는 중일 수 있고, 그 전에 클라이언트를
    닫으면 그 쓰기가 닫힌 저장소를 만진다. 한 묶음은 DB 호출 예산(5초)과 임베딩 한 번으로 묶여 있다.
    """
    global _thread
    _stop.set()
    thread = _thread
    if thread is None:
        return True
    thread.join(timeout)
    if thread.is_alive():
        return False
    _thread = None
    return True
