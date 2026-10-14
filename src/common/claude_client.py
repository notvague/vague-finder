"""Anthropic Messages API 어댑터. 검색 기본 모델은 Sonnet, adaptive / medium이다.

기존 분석·리랭킹의 프롬프트와 파서를 공유하며 Gemini 호환 호출 면을 제공한다.
리랭킹은 기본으로 structured outputs를 사용하고 분석은 기존 JSON 검증을 유지한다.
CLAUDE_USAGE_LOG / CLAUDE_BUDGET_USD로 사용량 기록과 선택적 예산 제한을 설정한다.
예산은 프로세스 간 예약을 포함하며, 응답 없는 호출은 상한으로 기록한다.
원문은 CLAUDE_RAW_LOG를 명시했을 때만 기록한다. 키는 기록하지 않는다.

thinking·effort는 `VF_CLAUDE_THINKING` · `VF_CLAUDE_EFFORT`로 받는다. `CLAUDE_EFFORT`는 Claude Code가 자기 설정으로
셸에 내보내는 이름이라(2026-10-14 확인, 값 xhigh) 그 셸에서 띄운 서버·측정이 .env와 다른 effort로 돌았다 —
load_dotenv는 이미 있는 변수를 덮지 않는다. 이름이 겹치는 두 칸만 접두어를 붙인다.
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
# 날짜 없는 고정 ID(별칭 아님). https://platform.claude.com/docs/en/about-claude/models/overview — 2026-10-14 확인.
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT_SECONDS = 60.0
_KEY_ENVS = ("CLAUDE_API", "ANTHROPIC_API_KEY")
THINKING_ENV = "VF_CLAUDE_THINKING"
EFFORT_ENV = "VF_CLAUDE_EFFORT"
_THINKING_VALUES = frozenset({"", "adaptive", "disabled"})
_EFFORT_VALUES = frozenset({"", "low", "medium", "high", "xhigh", "max"})
_FENCE_VALUES = frozenset({"", "strip"})
_STRUCTURED_VALUES = frozenset({"", "json_schema", "off"})
_FENCE = re.compile(r"\A\s*```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?```\s*\Z", re.DOTALL)

# 100만 토큰당 정가(USD). https://platform.claude.com/docs/en/about-claude/pricing — 2026-10-14 확인.
# Haiku 5.5는 프롬프트(캐시 읽기·쓰기를 포함한 입력 전체)가 10만 토큰을 넘는 요청에 높은 단가를 매긴다.
# Sonnet 5.5는 1M 전체가 같은 단가다(긴 프롬프트 단가 없음). cache_write는 5분 캐시 단가다.
# 구독 크레딧으로 냈는지와 무관한 정가다. 단가를 모르는 모델은 비용을 None으로 둔다 — 0으로 치지 않는다.
PRICES_PER_MTOK: Dict[str, Dict[str, Any]] = {
    "claude-haiku-5-5": {
        "long_prompt_over_tokens": 100_000,
        "base": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
        "long": {"input": 0.50, "output": 2.50, "cache_write": 0.625, "cache_read": 0.05},
    },
    "claude-sonnet-5-5": {
        "long_prompt_over_tokens": None,
        "base": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.10},
    },
}

# 입력 토큰 상한의 여유. 바이트 단위 BPE 토큰은 한 개가 1바이트 이상을 덮으므로 프롬프트의 토큰 수는 UTF-8 바이트 수를
# 넘지 않는다. 메시지 틀(역할 표지 등)과 thinking·effort 설정이 붙이는 토큰은 이 여유로 덮는다.
# 실측(Haiku 1,108회): 입력 토큰 / 글자 수 최대 0.742 — 바이트 수(한글 3바이트)는 그보다 훨씬 크다.
PROMPT_FRAME_TOKENS = 2048
# 동기 호출이 요청 제한 뒤에도 결과를 못 받으면 이만큼 더 기다렸다가 취소한다(기록을 남길 시간).
_SETTLE_GRACE_SECONDS = 5.0

_raw_lock = threading.Lock()
_ledger_thread_lock = threading.Lock()


class ClaudeError(RuntimeError):
    """Claude 호출이 쓸 수 있는 답을 내지 못했다."""


class ClaudeAPIError(ClaudeError):
    """API가 오류로 답했다(4xx·5xx). 키는 메시지에 넣지 않는다."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(f"Claude API {status} {error_type}: {message[:300]}")
        self.status = status
        self.error_type = error_type

    @property
    def is_billing(self) -> bool:
        """크레딧이 없어 거절됐는가. 재시도해도 같은 답이므로 호출부가 바로 멈춰야 한다."""
        return "credit balance" in str(self).lower() or self.error_type == "billing_error"


class ClaudeOutputError(ClaudeError):
    """200으로 왔지만 답이 아니다 — 거절(refusal) · 출력 한도에서 잘림 · 빈 응답."""


class ClaudeBudgetExceeded(ClaudeError):
    """`CLAUDE_BUDGET_USD`를 넘게 된다. 부르지 않았다."""


def claude_api_key() -> str:
    for name in _KEY_ENVS:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def claude_configured() -> bool:
    return bool(claude_api_key())


def claude_model_name() -> str:
    """`CLAUDE_MODEL_NAME`, 없으면 DEFAULT_MODEL."""
    return os.getenv("CLAUDE_MODEL_NAME", "").strip() or DEFAULT_MODEL


def _choice(env: str, allowed: frozenset) -> str:
    raw = os.getenv(env, "").strip().lower()
    if raw not in allowed:
        # 값은 메시지에 넣지 않는다(gemini_client와 같은 이유 — 이 칸에 키를 잘못 넣을 수 있다).
        raise ValueError(f"{env} 값을 알 수 없다 — {' / '.join(sorted(v for v in allowed if v))} 중 하나거나 비워 둔다")
    return raw


def claude_structured_output() -> str:
    """`CLAUDE_STRUCTURED_OUTPUT` — 비우면 쓰지 않음, json_schema면 JSON을 요청하는 호출부가 스키마를 붙인다."""
    value = _choice("CLAUDE_STRUCTURED_OUTPUT", _STRUCTURED_VALUES)
    return "" if value == "off" else value or "json_schema"


def claude_request_settings() -> Dict[str, Any]:
    """요청에 실제로 들어가는 설정 — 측정 기록용. 비어 있는 칸은 API 기본값을 쓴다는 뜻이다."""
    return {
        "model": claude_model_name(),
        "thinking": _choice(THINKING_ENV, _THINKING_VALUES) or "adaptive",
        "effort": _choice(EFFORT_ENV, _EFFORT_VALUES) or "medium",
        "max_tokens": int(os.getenv("CLAUDE_MAX_TOKENS", "") or DEFAULT_MAX_TOKENS),
        "sampling": "보내지 않음(모델 기본값) — temperature=0 아님",
        "json_mode": "코드 울타리만 벗김(CLAUDE_JSON_FENCE=strip)" if _choice("CLAUDE_JSON_FENCE", _FENCE_VALUES)
                     else "없음(프롬프트 지시만)",
        "structured_output": claude_structured_output() or "없음",
    }


def strip_code_fence(text: str) -> str:
    """답 **전체**가 코드 울타리 한 겹에 싸여 있으면 안쪽만 돌려준다. 아니면 그대로 — 본문 중간의 울타리는 건드리지 않는다."""
    match = _FENCE.match(text)
    return match.group(1) if match else text


def cost_usd(model: str, usage: Mapping[str, Any]) -> Optional[float]:
    """usage로 정가 기준 비용을 낸다. 단가를 모르는 모델이면 None."""
    table = PRICES_PER_MTOK.get(model)
    if table is None:
        return None
    fresh = int(usage.get("input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    read = int(usage.get("cache_read_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    threshold = table.get("long_prompt_over_tokens")
    tier = table["long"] if threshold is not None and fresh + written + read > threshold else table["base"]
    total = (fresh * tier["input"] + written * tier["cache_write"] + read * tier["cache_read"]
             + output * tier["output"])
    return total / 1_000_000


def prompt_token_upper_bound(prompt: str) -> int:
    """프롬프트 입력 토큰의 상한 — UTF-8 바이트 수 + 메시지 틀 여유(`PROMPT_FRAME_TOKENS`). 토크나이저와 무관하다."""
    return len(prompt.encode("utf-8")) + PROMPT_FRAME_TOKENS


def worst_case_cost_usd(model: str, input_tokens_max: int, max_tokens: int) -> Optional[float]:
    """부르기 전의 상한 — 입력은 `input_tokens_max`(보통 `prompt_token_upper_bound`), 출력은 `max_tokens`를 다 쓴다고 본다.

    입력 단가는 그 등급의 입력·캐시 쓰기 중 비싼 쪽(캐시를 쓰지 않지만 과소평가하지 않게), 등급은 입력 상한이
    긴 프롬프트 문턱을 넘으면 높은 등급으로 본다. 단가를 모르면 None.
    """
    table = PRICES_PER_MTOK.get(model)
    if table is None:
        return None
    threshold = table.get("long_prompt_over_tokens")
    tier = table["long"] if threshold is not None and input_tokens_max > threshold else table["base"]
    return (input_tokens_max * max(tier["input"], tier["cache_write"]) + max_tokens * tier["output"]) / 1_000_000


def usage_log_path() -> Optional[Path]:
    raw = os.getenv("CLAUDE_USAGE_LOG", "").strip()
    return Path(raw) if raw else None


def _pending_path(log: Path) -> Path:
    return log.with_name(log.name + ".pending")


def _record_cost(record: Mapping[str, Any]) -> float:
    """기록 한 줄이 예산에서 차지하는 값 — 비용, 모르면 상한. 둘 다 없으면 한도를 지킬 수 없으므로 오류다."""
    cost = record.get("cost_usd")
    if cost is None:
        cost = record.get("worst_case_usd")
    if cost is None:
        raise ValueError("비용도 상한 추정도 없는 호출 기록이 있다 — 0으로 세지 않는다. 기록을 확인한다")
    return float(cost)


def spent_usd(path: Optional[Path] = None) -> float:
    """호출 기록의 누적 비용. 답을 못 받아 실제 과금을 모르는 호출은 상한 추정치로 적혀 있다."""
    path = path or usage_log_path()
    if path is None or not path.exists():
        return 0.0
    total = 0.0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                total += _record_cost(json.loads(line))
    return total


def pending_usd(path: Optional[Path] = None) -> float:
    """예약만 하고 아직 기록으로 바뀌지 않은 호출의 상한 합 — 진행 중이거나, 끝나기 전에 프로세스가 죽은 호출."""
    path = path or usage_log_path()
    if path is None or not _pending_path(path).exists():
        return 0.0
    open_: Dict[str, float] = {}
    with open(_pending_path(path), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("open"):
                open_[entry["id"]] = float(entry["worst_case_usd"])
            else:
                open_.pop(entry["id"], None)
    return sum(open_.values())


class _Tail:
    """append-only 파일을 **이어서** 읽는다. 호출마다 처음부터 읽으면 호출 수에 비례해 느려진다(제곱) —
    예산이 걸린 긴 측정에서 예약 한 번이 수천 줄 파싱이 됐다.

    다른 파일로 바뀌었거나(inode) 짧아졌거나 이어 읽을 자리가 줄 끝이 아니면 처음부터 다시 읽는다 —
    사람이 기록을 지우거나 잘라낸 경우다. 쓰다 만 줄(개행 없음)은 다음에 다시 읽는다.
    """

    def __init__(self) -> None:
        self.ino = -1
        self.offset = 0

    def read(self, path: Path) -> tuple[bool, list]:
        """(처음부터 다시 읽었는가, 새 기록들). `_ledger_locked` 안에서 부른다."""
        if not path.exists():
            restarted = self.ino != -1
            self.ino, self.offset = -1, 0
            return restarted, []
        stat = path.stat()
        restarted = False
        records: list = []
        with open(path, "rb") as handle:
            if stat.st_ino != self.ino or stat.st_size < self.offset or not self._at_line_start(handle):
                self.ino, self.offset, restarted = stat.st_ino, 0, True
            handle.seek(self.offset)
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                self.offset += len(raw)
                line = raw.strip()
                if line:
                    records.append(json.loads(line))
        return restarted, records

    def _at_line_start(self, handle: Any) -> bool:
        if self.offset == 0:
            return True
        handle.seek(self.offset - 1)
        return handle.read(1) == b"\n"


@dataclass
class _LedgerTally:
    log: _Tail = field(default_factory=_Tail)
    pending: _Tail = field(default_factory=_Tail)
    spent: float = 0.0
    open_: Dict[str, float] = field(default_factory=dict)


_tallies: Dict[Path, _LedgerTally] = {}


def _tally(log: Path) -> _LedgerTally:
    """기록·예약 파일의 누적 — 새로 붙은 줄만 읽어 갱신한다. `_ledger_locked` 안에서 부른다.

    잘못된 기록(비용도 상한도 없음)을 만나면 누적을 버리고 예외를 올린다 — 다음 호출도 같은 기록에서 다시 멈춘다.
    읽은 자리를 넘겨 두면 그 기록을 조용히 건너뛰게 된다.
    """
    tally = _tallies.setdefault(log.resolve(), _LedgerTally())
    try:
        restarted, records = tally.log.read(log)
        if restarted:
            tally.spent = 0.0
        for record in records:
            tally.spent += _record_cost(record)
        restarted, entries = tally.pending.read(_pending_path(log))
        if restarted:
            tally.open_.clear()
        for entry in entries:
            if entry.get("open"):
                tally.open_[entry["id"]] = float(entry["worst_case_usd"])
            else:
                tally.open_.pop(entry["id"], None)
    except Exception:
        _tallies.pop(log.resolve(), None)
        raise
    return tally


@contextlib.contextmanager
def _ledger_locked(log: Path) -> Iterator[None]:
    """기록·예약 파일을 고칠 때 잡는 잠금 — 같은 프로세스의 스레드끼리, 다른 프로세스끼리 모두."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_thread_lock, open(log.with_name(log.name + ".lock"), "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _budget_usd() -> float:
    raw = os.getenv("CLAUDE_BUDGET_USD", "").strip()
    return float(raw) if raw else 0.0


def _reserve(model: str, worst: Optional[float]) -> Optional[str]:
    """예산이 걸려 있으면 이번 호출의 상한을 예약한다. 넘으면 부르지 않는다. 예산이 없으면 None."""
    budget = _budget_usd()
    if budget <= 0:
        return None
    log = usage_log_path()
    if log is None:
        raise ValueError("CLAUDE_BUDGET_USD를 쓰려면 CLAUDE_USAGE_LOG(누적 기록)가 있어야 한다")
    if worst is None:
        # 단가를 모르면 비용을 0으로 세게 되어 한도가 지켜지지 않는다 — 부르지 않는다.
        raise ClaudeBudgetExceeded(f"{model}의 단가를 몰라 예산을 지킬 수 없다 — PRICES_PER_MTOK에 공식 단가를 더한다")
    with _ledger_locked(log):
        tally = _tally(log)
        spent, pending = tally.spent, sum(tally.open_.values())
        if spent + pending + worst > budget:
            raise ClaudeBudgetExceeded(
                f"예산 ${budget:.2f} 중 ${spent:.4f}를 썼고 진행 중 상한 ${pending:.4f}, 이번 호출의 상한 추정이 "
                f"${worst:.4f}다 — 부르지 않는다"
            )
        reservation = uuid.uuid4().hex
        with open(_pending_path(log), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": reservation, "open": True, "worst_case_usd": worst}) + "\n")
    return reservation


def _settle(record: Dict[str, Any], reservation: Optional[str]) -> None:
    """호출 기록을 쓰고 예약을 닫는다 — 한 잠금 안에서 해서 그 사이에 누적이 비지 않게 한다."""
    log = usage_log_path()
    if log is None:
        return
    with _ledger_locked(log):
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if reservation is not None:
            with open(_pending_path(log), "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": reservation, "open": False}) + "\n")


def _append_raw(record: Mapping[str, Any], text: Optional[str], prompt: str) -> None:
    """`CLAUDE_RAW_LOG`를 주면 응답 원문을 따로 남긴다 — 형식 위반을 사후에 보려는 것이다. 기본은 남기지 않는다.

    `CLAUDE_RAW_LOG_PROMPT=1`이면 프롬프트도 함께 남긴다(응답이 프롬프트의 후보를 빠짐없이 담았는지 대조할 때).
    답을 못 받은 호출은 text가 None이고 실패 사유(error)가 붙는다 — 어느 입력에서 실패했는지 대조할 수 있게.
    """
    raw = os.getenv("CLAUDE_RAW_LOG", "").strip()
    if not raw:
        return
    line = {key: record.get(key) for key in ("at", "run", "tag", "model", "prompt_head_sha", "request_id")}
    line["text"] = text
    if text is None:
        line["error"] = record.get("error")
    if os.getenv("CLAUDE_RAW_LOG_PROMPT", "").strip() == "1":
        line["prompt"] = prompt
    with _raw_lock:
        Path(raw).parent.mkdir(parents=True, exist_ok=True)
        with open(raw, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


class _LoopThread:
    """요청을 도는 이벤트 루프 하나(데몬 스레드). 연결 풀이 이 루프에 묶여 호출 사이에 재사용된다."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="claude-http", daemon=True).start()

    def submit(self, coro: Any) -> "Any":
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


_loop_thread: Optional[_LoopThread] = None
_loop_thread_lock = threading.Lock()


def _runner() -> _LoopThread:
    global _loop_thread
    if _loop_thread is None:
        with _loop_thread_lock:
            if _loop_thread is None:
                _loop_thread = _LoopThread()
    return _loop_thread


@dataclass
class ClaudeResponse:
    """`google.genai` 응답 중 호출부가 읽는 `.text`와, 측정에 쓰는 usage·시간."""

    text: str
    usage: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: Optional[float] = None


class _Models:
    def __init__(self, owner: "ClaudeGenaiClient"):
        self._owner = owner

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> ClaudeResponse:
        return self._owner.generate(model=model, contents=contents, config=config)


class _AsyncModels:
    def __init__(self, owner: "ClaudeGenaiClient"):
        self._owner = owner

    async def generate_content(self, *, model: str, contents: Any, config: Any = None) -> ClaudeResponse:
        """호출부의 루프에서 기다린다. 호출부가 취소하면(`wait_for` 시간 초과) 진행 중인 요청도 취소된다."""
        future = self._owner._submit(model=model, contents=contents, config=config)
        return await asyncio.wrap_future(future)


class _Aio:
    def __init__(self, owner: "ClaudeGenaiClient"):
        self.models = _AsyncModels(owner)

    async def aclose(self) -> None:
        """연결 풀은 어댑터가 프로세스 동안 들고 있다. 리랭커가 부르더라도 닫지 않는다."""


class ClaudeGenaiClient:
    """`google.genai.Client` 중 이 저장소가 쓰는 면(`models.generate_content` · `aio.models.generate_content`)만 가진 어댑터.

    동기 면도 같은 루프 스레드에서 돈다 — 둘 다 요청 전체 시간을 `asyncio.wait_for`로 끊는다.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tag: str = "",
        timeout_seconds: Optional[float] = None,
        transport: Any = None,
    ):
        self._api_key = (api_key or claude_api_key()).strip()
        if not self._api_key:
            raise ValueError("Claude 설정이 없다 — CLAUDE_API(또는 ANTHROPIC_API_KEY)가 필요하다")
        self._tag = tag
        self._timeout = float(timeout_seconds or os.getenv("CLAUDE_TIMEOUT_SECONDS", "") or DEFAULT_TIMEOUT_SECONDS)
        self._thinking = _choice(THINKING_ENV, _THINKING_VALUES) or "adaptive"
        self._effort = _choice(EFFORT_ENV, _EFFORT_VALUES) or "medium"
        self._json_fence = _choice("CLAUDE_JSON_FENCE", _FENCE_VALUES)
        if self._thinking == "disabled" and self._effort not in {"low", "medium", "high"}:
            raise ValueError(f"{THINKING_ENV}=disabled면 {EFFORT_ENV}를 low / medium / high 중에서 정해야 한다")
        self._max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", "") or DEFAULT_MAX_TOKENS)
        self._transport = transport
        self._http: Any = None          # 루프 스레드에서만 만들고 쓴다
        self.models = _Models(self)
        self.aio = _Aio(self)

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(transport=self._transport) if self._transport else httpx.AsyncClient()
        return self._http

    def close(self) -> None:
        """서버 종료 후 연결 풀을 만든 루프에서 닫는다. 호출 단위 aclose와 별개다."""
        if self._http is None:
            return

        async def close_http() -> None:
            if self._http is not None:
                await self._http.aclose()
                self._http = None

        _runner().submit(close_http()).result(timeout=5.0)

    @staticmethod
    def _timeout_from(config: Any) -> Optional[float]:
        """호출부가 준 제한(`http_options.timeout`, 밀리초). 분석기가 남은 예산을 이렇게 넘긴다."""
        if not isinstance(config, Mapping):
            return None
        options = config.get("http_options")
        if isinstance(options, Mapping) and options.get("timeout"):
            return max(0.001, float(options["timeout"]) / 1000.0)
        return None

    @staticmethod
    def _reject_tools(config: Any) -> None:
        tools = config.get("tools") if isinstance(config, Mapping) else getattr(config, "tools", None)
        if tools:
            raise NotImplementedError("Claude 경로에는 Google Search 도구가 없다 — 도구를 쓰는 설정으로 부를 수 없다")

    def _request_timeout(self, contents: Any, config: Any) -> float:
        """입력을 확인하고 이번 요청의 전체 시간 제한을 정한다 — min(어댑터 제한, 호출부가 준 제한)."""
        if not isinstance(contents, str):
            raise TypeError("Claude 어댑터는 문자열 프롬프트만 받는다")
        self._reject_tools(config)
        return min(self._timeout, self._timeout_from(config) or self._timeout)

    def _submit(self, *, model: str, contents: Any, config: Any) -> Any:
        """루프 스레드에 요청을 넘긴다. 돌려주는 것은 concurrent.futures.Future다."""
        timeout = self._request_timeout(contents, config)
        return _runner().submit(self._agenerate(model=model, contents=contents, config=config, timeout=timeout))

    def generate(self, *, model: str, contents: Any, config: Any = None) -> ClaudeResponse:
        timeout = self._request_timeout(contents, config)
        future = _runner().submit(self._agenerate(model=model, contents=contents, config=config, timeout=timeout))
        try:
            return future.result(timeout=timeout + _SETTLE_GRACE_SECONDS)
        except BaseException:
            if not future.done():
                # 루프 안의 제한이 먼저 끊었어야 하는데 결과가 안 왔다(또는 이 스레드가 중단됐다) — 요청을 취소한다.
                future.cancel()
            raise

    async def _agenerate(self, *, model: str, contents: str, config: Any, timeout: float) -> ClaudeResponse:
        import httpx

        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": contents}],
        }
        if self._thinking:
            body["thinking"] = {"type": self._thinking}
        if self._effort:
            body["output_config"] = {"effort": self._effort}
        schema = config.get("response_json_schema") if isinstance(config, Mapping) else None
        if schema is not None:
            if not isinstance(schema, Mapping):
                raise TypeError("response_json_schema는 JSON Schema dict여야 한다")
            body.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": dict(schema)}

        # 스키마가 붙으면 API가 형식 안내 system 프롬프트를 더한다 — 그 몫은 PROMPT_FRAME_TOKENS 여유와 스키마 바이트로 덮는다.
        input_max = prompt_token_upper_bound(contents) + (len(json.dumps(schema).encode("utf-8")) if schema else 0)
        worst = worst_case_cost_usd(model, input_max, self._max_tokens)
        record: Dict[str, Any] = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run": os.getenv("CLAUDE_USAGE_RUN", "").strip(),
            "tag": self._tag,
            "model": model,
            "thinking": self._thinking or "api_default",
            "effort": self._effort or "api_default",
            "max_tokens": self._max_tokens,
            "output_format": "json_schema" if schema is not None else "text",
            "timeout_s": round(timeout, 3),
            "prompt_chars": len(contents),
            "input_tokens_max": input_max,
            "worst_case_usd": worst,
            # 재시도는 같은 프롬프트 끝에 정정 문장을 붙인다 — 앞부분 지문이 같으면 같은 입력의 재시도다.
            "prompt_head_sha": hashlib.sha256(contents[:4000].encode("utf-8")).hexdigest()[:12],
        }
        reservation = _reserve(model, worst)
        started = time.perf_counter()
        settled = False

        def finish(*, ok: bool, error: str, cost: Optional[float], basis: str) -> None:
            nonlocal settled
            record["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            record["ok"] = ok
            record["error"] = error
            record["cost_usd"] = cost
            record["cost_is_estimate"] = basis == "upper_bound"
            record["cost_basis"] = basis
            _settle(record, reservation)
            settled = True
            if not ok:
                _append_raw(record, None, contents)

        try:
            try:
                response = await asyncio.wait_for(
                    self._client().post(
                        API_URL,
                        headers={
                            "x-api-key": self._api_key,
                            "anthropic-version": API_VERSION,
                            "content-type": "application/json",
                        },
                        json=body,
                        timeout=timeout,
                    ),
                    timeout,
                )
            except (httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ConnectError) as exc:
                # 연결 전에 실패했다 — 요청이 서버에 닿지 않았으므로 과금되지 않는다.
                finish(ok=False, error=type(exc).__name__, cost=0.0, basis="not_sent")
                raise ClaudeError(f"Claude 호출 실패: {type(exc).__name__}") from exc
            except (TimeoutError, httpx.TimeoutException) as exc:
                # 서버가 처리를 마쳤는지 알 수 없다. 과금됐을 수 있으므로 상한 추정치로 센다.
                finish(ok=False, error="timeout", cost=worst, basis="upper_bound")
                raise TimeoutError(f"Claude 호출이 {timeout:.1f}초 안에 끝나지 않았다") from exc
            except httpx.HTTPError as exc:
                # 보낸 뒤 끊겼다(읽기 오류·프로토콜 오류) — 처리됐는지 모른다.
                finish(ok=False, error=type(exc).__name__, cost=worst, basis="upper_bound")
                raise ClaudeError(f"Claude 호출 실패: {type(exc).__name__}") from exc

            record["http_status"] = response.status_code
            record["request_id"] = response.headers.get("request-id", "")
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if response.status_code != 200:
                error = payload.get("error") if isinstance(payload, Mapping) else None
                error = error if isinstance(error, Mapping) else {}
                error_type = str(error.get("type") or "http_error")
                if response.status_code >= 500:
                    finish(ok=False, error=error_type, cost=worst, basis="upper_bound")
                else:
                    finish(ok=False, error=error_type, cost=0.0, basis="rejected_4xx")
                raise ClaudeAPIError(response.status_code, error_type, str(error.get("message") or ""))

            usage = dict(payload.get("usage") or {}) if isinstance(payload, Mapping) else {}
            stop_reason = str(payload.get("stop_reason") or "") if isinstance(payload, Mapping) else ""
            blocks = (payload.get("content") or []) if isinstance(payload, Mapping) else []
            text = "".join(str(b.get("text") or "") for b in blocks if isinstance(b, Mapping) and b.get("type") == "text")
            record.update({
                "stop_reason": stop_reason,
                "usage": usage,
                "output_chars": len(text),
                "thinking_blocks": sum(1 for b in blocks if isinstance(b, Mapping) and b.get("type") == "thinking"),
            })
            if usage:
                cost, basis = cost_usd(model, usage), "usage"
            else:
                cost, basis = worst, "upper_bound"     # 200인데 usage를 못 읽었다 — 과금을 모른다
            if stop_reason in {"refusal", "max_tokens"} or not text.strip():
                reason = stop_reason if stop_reason in {"refusal", "max_tokens"} else "empty_text"
                finish(ok=False, error=reason, cost=cost, basis=basis)
                raise ClaudeOutputError(f"Claude 응답을 쓸 수 없다: {reason}")
            # 형식 기록: 울타리로 감쌌는가(프롬프트만으로 JSON을 지켰는지의 지표)와, 벗겨서 돌려줬는가.
            wants_json = isinstance(config, Mapping) and config.get("response_mime_type") == "application/json"
            record["fenced"] = text.lstrip().startswith("```")
            returned = strip_code_fence(text) if (self._json_fence == "strip" and wants_json) else text
            record["fence_stripped"] = returned != text
            finish(ok=True, error="", cost=cost, basis=basis)
            _append_raw(record, text, contents)
            return ClaudeResponse(
                text=returned,
                usage=usage,
                stop_reason=stop_reason,
                model=str(payload.get("model") or model),
                latency_ms=record["latency_ms"],
                cost_usd=cost,
            )
        except asyncio.CancelledError:
            # 호출부가 시간 예산을 넘겨 끊었다. 서버가 처리했을 수 있으므로 상한으로 센다.
            if not settled:
                finish(ok=False, error="cancelled", cost=worst, basis="upper_bound")
            raise
        finally:
            if not settled:
                finish(ok=False, error="aborted", cost=worst, basis="upper_bound")


def make_claude_genai_client(*, api_key: Optional[str] = None, tag: str = "",
                           timeout_seconds: Optional[float] = None) -> ClaudeGenaiClient:
    """환경의 키로 어댑터를 만든다. 키가 없으면 ValueError — 호출부는 그 전에 `claude_configured`로 거른다."""
    return ClaudeGenaiClient(api_key=api_key, tag=tag, timeout_seconds=timeout_seconds)
