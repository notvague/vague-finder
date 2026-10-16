"""기동 때 한 번, Context(배경 지식) 색인·별칭이 설정과 맞는지 확인한다.

**왜 필요한가.** 2026-10-16 v56 운영 검증에서 서버 방식 Qdrant로 옮기면서 컬렉션만 옮기고
Context **별칭** 두 개를 옮기지 않았다. 서비스는 그대로 떴고, 검색마다 로그에
`배경 지식 경로 실패 (빈 결과로 대체)` 경고만 남긴 채 Context 없이 돌았다 — 응답만 봐서는
모르니 1차 측정 전체를 Context 빠진 조건으로 재는 줄도 몰랐다. 저수준 검색기
(`ContextQdrantSearch`)는 검색 시점에야 별칭·manifest를 확인하고, 라우터는 경로 하나의
예외를 빈 결과로 바꾸는 것이 **의도된** 계약이다(한 경로가 죽어도 검색은 계속돼야 한다).
그래서 "Context를 쓰도록 설정된 서비스가 Context를 실제로 쓸 수 있는가"는 요청 경로가 아니라
기동 경계에서 한 번 확인한다.

**세 상태를 섞지 않는다.**

- `disabled` — `CONTEXT_WEIGHT=0`. 의도적으로 끈 것이라 Qdrant에 묻지 않는다.
- `verified` — 별칭 2개 · 대상 컬렉션 존재·벡터 설정 · 상태 manifest · 로컬 dense/sparse
  artifact 해시가 발행 세대와 맞다. **기동 시점의 결과**다. 지금 연결이 살아 있다는 보장이 아니다.
- `config_error` — 별칭 누락/일부/잘못된 대상, manifest 누락·손상·불일치, 미발행. 다시 시도해도
  낫지 않는 설정 오류라 **기동을 실패시킨다**(`ContextStartupError`). 운영자가 별칭·manifest·
  namespace를 고쳐야 한다(docs/namuwiki_context_vector_db.md "기동 검사").
- `unavailable` — 연결 불가·시간 초과·벡터 클라이언트 초기화 실패. 설정 오류가 아니라 **확인 불가**다.
  기존 벡터 DB 장애 정책(경고 + 기동 계속, 요청에서 다시 시도)과 같게 하되, 확인됐다고 적지 않는다.

**하지 않는 것.** 모델을 올리거나 내려받지 않고(KoE5 임베더를 만들지 않는다), 포인트·payload·
벡터를 훑지 않고, 별칭을 만들거나 고치거나 재발행하지 않는다. 검색 결과·가중·후보 정책은
건드리지 않는다 — 검사는 읽기만 한다.

로그와 상태에는 자격 증명·접속 URI·원문 예외를 싣지 않는다. 설정 오류는 이 코드가 만든 고정
문구와 사유 코드, 일시 장애는 예외 **형식 이름**만 남긴다.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.retrieval.context_qdrant_search import (
    ContextPublicationError,
    ContextPublicationStatus,
    ContextQdrantSearch,
)
from src.retrieval.context_search_settings import load_context_settings

logger = logging.getLogger(__name__)

NOT_CHECKED = "not_checked"
DISABLED = "disabled"
VERIFIED = "verified"
CONFIG_ERROR = "config_error"
UNAVAILABLE = "unavailable"

# 상태 소비자가 "기동 때 결과"임을 오해하지 않게 늘 함께 싣는다.
_SOURCE = "startup"
_NOTE = "기동 시점의 결과다. 지금 연결 상태를 보장하지 않는다."

_lock = threading.Lock()
_state: Dict[str, Any] = {"state": NOT_CHECKED}


class ContextStartupError(RuntimeError):
    """Context를 쓰도록 설정됐는데 색인·별칭·manifest가 그 설정과 맞지 않는다 — 기동을 멈춘다.

    `reason`은 `ContextPublicationError.reason`(또는 설정값 오류면 `settings_invalid`)을 그대로 든다.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _set(state: str, **fields: Any) -> Dict[str, Any]:
    value = {"state": state, "source": _SOURCE, "checked_at": _now(), "note": _NOTE, **fields}
    with _lock:
        _state.clear()
        _state.update(value)
    return dict(value)


def reset() -> None:
    """테스트·재기동용. 앱이 다시 뜨면 `run()`이 덮어쓴다."""
    with _lock:
        _state.clear()
        _state["state"] = NOT_CHECKED


def status() -> Dict[str, Any]:
    """`/health`용. 저장된 기동 결과를 돌려줄 뿐 — 여기서 Qdrant를 다시 묻지 않는다."""
    with _lock:
        return dict(_state)


def enabled() -> bool:
    """서비스가 Context 경로를 실제로 쓰는 조건 — 라우터의 `_context_weight > 0`과 같다.

    설정값이 잘못되면(`CONTEXT_WEIGHT=abc`) ValueError가 올라온다 — 삼키지 않는다. 호출부(`run`)가
    설정 오류로 분류한다.
    """
    return load_context_settings().weight > 0


def _publication_fields(status_value: ContextPublicationStatus) -> Dict[str, Any]:
    return {
        "dense_alias": status_value.dense_alias,
        "sparse_alias": status_value.sparse_alias,
        "dense_collection": status_value.dense_collection,
        "sparse_collection": status_value.sparse_collection,
        "context_build_id": status_value.build_id,
        "dense_dimension": status_value.dense_dimension,
    }


def run(
    vector_client: Optional[Any],
    *,
    searcher_factory: Any = None,
) -> Dict[str, Any]:
    """기동 검사 한 번. 결과를 저장하고 돌려준다. 설정 오류면 `ContextStartupError`.

    vector_client: `get_vector_client()`가 만든 `QdrantVectorClient`. 그 초기화가 실패했으면 None —
        그때는 묻지 않고 `unavailable`로 남긴다(벡터 DB 장애 정책과 같다).
    searcher_factory: 테스트용. 기본은 임베더 **없이** `ContextQdrantSearch(vector_client)`를 만든다 —
        검사에서 KoE5를 만들지 않기 위해서다. 검색이 쓰는 인스턴스(라우터)와 별개이고 자원을 쥐지 않는다.
    """
    try:
        if not enabled():
            logger.info("[context] CONTEXT_WEIGHT=0 — Context 경로가 꺼져 있어 기동 검사를 건너뛴다")
            return _set(DISABLED, reason="context_weight_zero")
    except ValueError as exc:
        # 설정값 자체가 틀렸다(음수·숫자 아님). 전에는 첫 검색에서 라우터를 만들다 500으로 터졌다.
        _set(CONFIG_ERROR, reason="settings_invalid")
        raise ContextStartupError(
            f"Context 설정값이 잘못됐다 — {exc}", reason="settings_invalid"
        ) from exc

    if vector_client is None:
        logger.warning("[context] 벡터 클라이언트가 없어 Context 발행을 확인하지 못했다(확인 불가)")
        return _set(UNAVAILABLE, reason="vector_client_unavailable")

    try:
        searcher = (
            searcher_factory(vector_client) if searcher_factory is not None
            else ContextQdrantSearch(vector_client)
        )
        verified = searcher.verify_publication()
    except ContextPublicationError as exc:
        _set(CONFIG_ERROR, reason=exc.reason)
        logger.error("[context] Context 발행 설정 오류(%s): %s", exc.reason, exc)
        raise ContextStartupError(
            f"Context를 쓰도록 설정됐는데 색인·별칭이 맞지 않는다({exc.reason}): {exc}",
            reason=exc.reason,
        ) from exc
    except Exception as exc:  # noqa: BLE001 - 연결·시간 초과 등: 설정 오류가 아니라 확인 불가
        kind = type(exc).__name__
        logger.warning("[context] Context 발행을 확인하지 못했다(연결 문제로 추정, 요청에서 다시 시도): %s", kind)
        return _set(UNAVAILABLE, reason="backend_error", error_type=kind)

    logger.info(
        "[context] Context 발행 확인: build=%s dense=%s sparse=%s",
        verified.build_id[:16], verified.dense_collection, verified.sparse_collection,
    )
    return _set(VERIFIED, reason="ok", publication=_publication_fields(verified))
