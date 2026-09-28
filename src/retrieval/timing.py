"""요청 한 번이 어디서 시간을 썼는지 기록한다.

**왜 `ExplainRecorder`와 따로인가.** 기록기는 *근거*를 담는 그릇이고 시간은 근거가
아니다. 섞으면 시간을 재려고 explain을 켜야 하는 상황이 되고, "설명은 순위를 바꾸지
않는다"는 경계도 흐려진다. 이쪽은 `explain=false`에서도 돈다.

**왜 단계별 총합이 아니라 구간(span) 목록인가.** 검색 경로는 겹쳐서 돈다. 스레드풀이
4칸인데 보조 경로까지 최대 10개가 제출되므로 경로 시간을 더하면 실제 경과보다 훨씬
커진다. 시작·끝 시각과 부모를 남겨야 "무엇을 기다리느라 늦었는지"를 볼 수 있다.

**왜 제출(queued)과 시작(started)을 나누는가.** 다섯 번째로 제출된 경로는 앞 작업이
끝나기 전에는 시작조차 못 한다. 그 대기를 실행 시간에 섞으면 멀쩡한 경로를 범인으로
지목하게 된다.

**왜 ContextVar인가.** 경로 안쪽(임베딩 대 DB 조회)까지 재려면 `SearchService`처럼
호출부가 여럿인 함수에도 지점이 필요한데, 시그니처를 다 바꾸면 평가 스크립트까지
번진다. ContextVar는 async 태스크별·스레드별로 갈리므로 동시 요청 둘이 서로의
구간을 주워 담지 않는다(스레드 로컬과 달리 이벤트 루프 안에서도 안전하다).
`run_in_executor`는 컨텍스트를 옮겨주지 않으므로 작업 안에서 다시 묶어 준다.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# (기록기, 부모 구간 id). 묶이지 않은 곳에서는 None이라 아무것도 기록하지 않는다.
_CURRENT: ContextVar[Optional[Tuple["TimingRecorder", int]]] = ContextVar(
    "vaguefinder_timing", default=None
)

_LOG_PATH_ENV = "SEARCH_TIMING_LOG"


@dataclass
class Span:
    """한 구간. 시각은 모두 `perf_counter()` 기준 초."""

    id: int
    name: str
    parent: int
    queued: float
    started: float
    ended: float
    attrs: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def wait_ms(self) -> float:
        """제출부터 실행 시작까지. 스레드풀 대기가 여기에 잡힌다."""
        return (self.started - self.queued) * 1000.0

    @property
    def run_ms(self) -> float:
        return (self.ended - self.started) * 1000.0

    @property
    def total_ms(self) -> float:
        return (self.ended - self.queued) * 1000.0

    def as_dict(self, origin: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "parent": self.parent,
            # 시작 시각을 요청 시작 기준 오프셋으로 남긴다. 겹침을 보려면
            # 길이만으로는 부족하고 "언제부터 언제까지"가 있어야 한다.
            "at_ms": round((self.queued - origin) * 1000.0, 2),
            "wait_ms": round(self.wait_ms, 2),
            "run_ms": round(self.run_ms, 2),
        }
        if self.attrs:
            out["attrs"] = self.attrs
        if self.error:
            out["error"] = self.error
        return out


class TimingRecorder:
    """구간을 모으는 그릇. 스레드 여러 개가 같이 쓴다."""

    enabled = True

    def __init__(self, request_id: str = "", query: str = "") -> None:
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self.query = query
        self.origin = time.perf_counter()
        self.wall_started = time.time()
        self.attrs: Dict[str, Any] = {}
        self._spans: List[Span] = []
        self._next_id = 1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 기록
    # ------------------------------------------------------------------

    def _claim_id(self) -> int:
        with self._lock:
            span_id = self._next_id
            self._next_id += 1
            return span_id

    def add(
        self,
        name: str,
        parent: int,
        queued: float,
        started: float,
        ended: float,
        error: str = "",
        span_id: Optional[int] = None,
        **attrs: Any,
    ) -> int:
        span = Span(
            id=span_id if span_id is not None else self._claim_id(),
            name=name,
            parent=parent,
            queued=queued,
            started=started,
            ended=ended,
            attrs={k: v for k, v in attrs.items() if v is not None},
            error=error,
        )
        with self._lock:
            self._spans.append(span)
        return span.id

    def note(self, **attrs: Any) -> None:
        """요청 전체에 해당하는 값(결과 수, 응답 코드 등)."""
        self.attrs.update({k: v for k, v in attrs.items() if v is not None})

    def set_query(self, query: str) -> None:
        """미들웨어는 본문을 읽지 않는다. 질의는 라우트가 알려 준다.

        본문을 미들웨어에서 읽으면 스트림을 소비해 라우트가 다시 못 읽는다.
        """
        self.query = query

    @contextlib.contextmanager
    def step(self, name: str, parent: Optional[int] = None, **attrs: Any) -> Iterator[None]:
        """이 블록이 도는 동안을 한 구간으로 남긴다.

        블록 안에서 다시 `timing.step()`을 부르면 이 구간의 자식이 된다 —
        경로 안의 임베딩·DB 조회를 그렇게 잡는다.
        """
        span_id = self._claim_id()
        if parent is None:
            current = _CURRENT.get()
            parent = current[1] if current is not None and current[0] is self else 0
        started = time.perf_counter()
        token = _CURRENT.set((self, span_id))
        error = ""
        try:
            yield
        except BaseException as exc:  # 실패한 구간도 시간은 남는다
            error = type(exc).__name__
            raise
        finally:
            _CURRENT.reset(token)
            self.add(
                name,
                parent,
                started,
                started,
                time.perf_counter(),
                error=error,
                span_id=span_id,
                **attrs,
            )

    def mark(self, name: str, since: float, **attrs: Any) -> int:
        """`since`부터 지금까지를 한 구간으로 남긴다.

        `with`로 감쌀 수 없는 자리를 위한 것이다 — 중간에 빠져나가는 return이
        여럿인 긴 구간은 블록으로 묶으려면 수백 줄을 들여쓰기해야 한다.
        """
        current = _CURRENT.get()
        parent = current[1] if current is not None and current[0] is self else 0
        return self.add(name, parent, since, since, time.perf_counter(), **attrs)

    def job(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        parent: Optional[int] = None,
        **attrs: Any,
    ) -> Callable[[], Any]:
        """스레드풀에 넘길 함수를 시간 기록으로 감싼다.

        제출 시각을 **감쌀 때** 붙잡아 두는 것이 요점이다. 작업 안에서 재면 이미
        대기가 끝난 뒤라 풀이 밀렸는지 알 수 없다.
        """
        queued = time.perf_counter()
        span_id = self._claim_id()
        if parent is None:
            current = _CURRENT.get()
            parent = current[1] if current is not None and current[0] is self else 0
        outer_parent = parent

        def runner() -> Any:
            started = time.perf_counter()
            # 작업 스레드에는 컨텍스트가 넘어오지 않는다. 여기서 다시 묶어야
            # 경로 안쪽(임베딩/조회) 구간이 이 작업의 자식으로 붙는다.
            token = _CURRENT.set((self, span_id))
            error = ""
            try:
                return fn(*args)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                _CURRENT.reset(token)
                self.add(
                    name,
                    outer_parent,
                    queued,
                    started,
                    time.perf_counter(),
                    error=error,
                    span_id=span_id,
                    **attrs,
                )

        return runner

    # ------------------------------------------------------------------
    # 읽기
    # ------------------------------------------------------------------

    @property
    def spans(self) -> List[Span]:
        with self._lock:
            return sorted(self._spans, key=lambda s: (s.queued, s.id))

    def total_ms(self) -> float:
        """요청 시작부터 지금까지. 뿌리 구간이 끝나기 전에도 읽을 수 있다."""
        return (time.perf_counter() - self.origin) * 1000.0

    def find(self, name: str) -> Optional[Span]:
        for span in self.spans:
            if span.name == name:
                return span
        return None

    def run_ms_of(self, name: str) -> Optional[float]:
        span = self.find(name)
        return None if span is None else span.run_ms

    def to_dict(self) -> Dict[str, Any]:
        spans = self.spans
        return {
            "request_id": self.request_id,
            "started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime(self.wall_started)
            ),
            "query": self.query,
            "total_ms": round(self.total_ms(), 2),
            **({"attrs": self.attrs} if self.attrs else {}),
            "spans": [span.as_dict(self.origin) for span in spans],
        }

    def summary(self) -> str:
        """로그 한 줄. 전체와 굵직한 단계만 — 나머지는 JSONL에 있다."""
        parts = [f"rid={self.request_id}", f"total={self.total_ms():.0f}ms"]
        for label, name in (
            ("analysis", "analysis"),
            ("search", "search"),
            ("rerank", "search.rerank"),
        ):
            span = self.find(name)
            if span is not None:
                parts.append(f"{label}={span.run_ms:.0f}ms")
        paths = [s for s in self.spans if s.name.startswith("path.")]
        if paths:
            slowest = max(paths, key=lambda s: s.total_ms)
            parts.append(
                f"paths={len(paths)} slowest={slowest.name[5:]}"
                f"({slowest.wait_ms:.0f}+{slowest.run_ms:.0f}ms)"
            )
        if self.attrs:
            parts.extend(f"{k}={v}" for k, v in self.attrs.items())
        return "[timing] " + " ".join(parts)


class _NullTimer:
    """계측을 끈 자리에 들어가는 빈 그릇.

    호출부에 `if timer:` 분기를 만들지 않으려고 같은 모양을 갖춘다. 평가 스크립트와
    테스트는 이것을 쓰므로 계측 코드가 그쪽 경로를 건드리지 않는다.
    """

    enabled = False
    request_id = ""

    def note(self, **attrs: Any) -> None:
        return None

    def set_query(self, query: str) -> None:
        return None

    @contextlib.contextmanager
    def step(self, name: str, parent: Optional[int] = None, **attrs: Any) -> Iterator[None]:
        yield

    def job(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        parent: Optional[int] = None,
        **attrs: Any,
    ) -> Callable[[], Any]:
        def runner() -> Any:
            return fn(*args)

        return runner

    def add(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def mark(self, name: str, since: float, **attrs: Any) -> int:
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {}

    def summary(self) -> str:
        return ""


NULL_TIMER = _NullTimer()


# ----------------------------------------------------------------------
# 묶여 있는 기록기에 기대는 지점 — 시그니처를 못 바꾸는 깊은 함수용
# ----------------------------------------------------------------------


def current() -> Optional[Tuple[TimingRecorder, int]]:
    return _CURRENT.get()


@contextlib.contextmanager
def bound(timer: Any, parent: int = 0) -> Iterator[None]:
    """이 블록 안에서 `timing.step()`이 이 기록기에 붙게 한다."""
    if not getattr(timer, "enabled", False):
        yield
        return
    token = _CURRENT.set((timer, parent))
    try:
        yield
    finally:
        _CURRENT.reset(token)


@contextlib.contextmanager
def step(name: str, **attrs: Any) -> Iterator[None]:
    """지금 묶여 있는 기록기에 구간을 남긴다. 묶인 게 없으면 아무 일도 안 한다."""
    current_binding = _CURRENT.get()
    if current_binding is None:
        yield
        return
    timer, parent = current_binding
    with timer.step(name, parent=parent, **attrs):
        yield


def mark(name: str, since: float, **attrs: Any) -> None:
    """`since`부터 지금까지를 남긴다. `step`의 블록 없는 판."""
    current_binding = _CURRENT.get()
    if current_binding is None:
        return
    timer, parent = current_binding
    timer.add(name, parent, since, since, time.perf_counter(), **attrs)


# ----------------------------------------------------------------------
# 내보내기
# ----------------------------------------------------------------------


def log_path() -> Optional[Path]:
    """`SEARCH_TIMING_LOG`가 가리키는 JSONL 파일. 없으면 파일로 남기지 않는다."""
    raw = os.getenv(_LOG_PATH_ENV, "").strip()
    return Path(raw) if raw else None


def emit(timer: Any) -> None:
    """한 줄 요약은 로그로, 전체 구간은 JSONL로.

    파일 경로를 환경변수로 받는 이유: 측정할 때만 파일이 생겨야 한다. 늘 쌓으면
    개발 중에 아무도 안 보는 파일이 커진다.
    """
    if not getattr(timer, "enabled", False):
        return
    logger.info("%s", timer.summary())
    path = log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(timer.to_dict(), ensure_ascii=False) + "\n")
    except OSError as exc:
        # 측정 기록을 남기지 못하는 것이 요청을 실패시킬 이유는 아니다.
        logger.warning("[timing] 기록 파일을 쓰지 못했다(%s): %s", path, exc)
