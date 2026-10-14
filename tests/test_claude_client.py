"""Claude 호출 모듈 — 네트워크 없이(httpx MockTransport) 요청 모양·기록·예산·오류 처리를 본다."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List

import httpx
import pytest

from src.common import claude_client as cc

KEY = "test-key-not-real"


def _ok(text: str = '{"ok": true}', *, stop: str = "end_turn", usage: Dict[str, Any] | None = None, blocks=None) -> Dict[str, Any]:
    return {
        "model": "claude-haiku-5-5",
        "stop_reason": stop,
        "content": blocks if blocks is not None else [{"type": "text", "text": text}],
        "usage": usage or {"input_tokens": 1000, "output_tokens": 200},
    }


def _client(monkeypatch, handler: Callable[[httpx.Request], httpx.Response], **env: str) -> cc.ClaudeGenaiClient:
    for name in ("VF_CLAUDE_THINKING", "VF_CLAUDE_EFFORT", "CLAUDE_MAX_TOKENS", "CLAUDE_USAGE_LOG", "CLAUDE_BUDGET_USD",
                 "CLAUDE_USAGE_RUN", "CLAUDE_TIMEOUT_SECONDS", "CLAUDE_MODEL_NAME", "CLAUDE_RAW_LOG", "CLAUDE_JSON_FENCE", "CLAUDE_RAW_LOG_PROMPT",
                 "CLAUDE_STRUCTURED_OUTPUT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return cc.ClaudeGenaiClient(api_key=KEY, tag="t", transport=httpx.MockTransport(handler))


def _capture(seen: List[httpx.Request], payload: Dict[str, Any], status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=payload)
    return handler


def test_request_drops_sampling_and_json_mode(monkeypatch):
    seen: List[httpx.Request] = []
    client = _client(monkeypatch, _capture(seen, _ok()))
    response = client.models.generate_content(
        model="claude-haiku-5-5", contents="질의",
        config={"response_mime_type": "application/json", "temperature": 0.0, "top_p": 1.0},
    )
    assert response.text == '{"ok": true}'
    body = json.loads(seen[0].content)
    assert body == {"model": "claude-haiku-5-5", "max_tokens": cc.DEFAULT_MAX_TOKENS,
                    "messages": [{"role": "user", "content": "질의"}],
                    "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    assert seen[0].headers["x-api-key"] == KEY and seen[0].headers["anthropic-version"] == cc.API_VERSION


def test_thinking_and_effort_are_sent_when_set(monkeypatch):
    seen: List[httpx.Request] = []
    client = _client(monkeypatch, _capture(seen, _ok()), VF_CLAUDE_THINKING="disabled", VF_CLAUDE_EFFORT="medium",
                     CLAUDE_MAX_TOKENS="4096")
    client.models.generate_content(model="m", contents="p")
    body = json.loads(seen[0].content)
    assert body["thinking"] == {"type": "disabled"} and body["output_config"] == {"effort": "medium"}
    assert body["max_tokens"] == 4096


def test_disabled_thinking_defaults_to_medium_and_rejects_xhigh(monkeypatch):
    client = _client(monkeypatch, _capture([], _ok()), VF_CLAUDE_THINKING="disabled")
    assert client._effort == "medium"
    with pytest.raises(ValueError, match="VF_CLAUDE_EFFORT"):
        _client(monkeypatch, _capture([], _ok()), VF_CLAUDE_THINKING="disabled", VF_CLAUDE_EFFORT="xhigh")


@pytest.mark.parametrize("name,value", [("VF_CLAUDE_THINKING", "enabled"), ("VF_CLAUDE_EFFORT", "key-pasted-by-mistake")])
def test_unknown_setting_is_an_error_and_value_is_not_echoed(monkeypatch, name, value):
    with pytest.raises(ValueError) as caught:
        _client(monkeypatch, _capture([], _ok()), **{name: value})
    assert value not in str(caught.value)


def test_text_is_returned_untouched_and_thinking_blocks_are_skipped(monkeypatch):
    fenced = '```json\n{"a": 1}\n```'
    blocks = [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": fenced}]
    client = _client(monkeypatch, _capture([], _ok(blocks=blocks)))
    assert client.models.generate_content(model="m", contents="p").text == fenced


@pytest.mark.parametrize("payload,reason", [
    (_ok(stop="refusal"), "refusal"),
    (_ok(stop="max_tokens"), "max_tokens"),
    (_ok(text="   "), "empty_text"),
])
def test_unusable_output_raises_and_is_logged_as_failure(monkeypatch, tmp_path, payload, reason):
    log = tmp_path / "usage.jsonl"
    client = _client(monkeypatch, _capture([], payload), CLAUDE_USAGE_LOG=str(log))
    with pytest.raises(cc.ClaudeOutputError, match=reason):
        client.models.generate_content(model="claude-haiku-5-5", contents="p")
    record = json.loads(log.read_text(encoding="utf-8"))
    # 답은 못 썼지만 과금은 됐다 — 비용을 0으로 적지 않는다.
    assert record["ok"] is False and record["error"] == reason and record["cost_usd"] > 0


def test_api_error_carries_status_without_key_and_billing_is_flagged(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    error = {"type": "error", "error": {"type": "invalid_request_error",
                                        "message": "Your credit balance is too low to access the Anthropic API."}}
    client = _client(monkeypatch, _capture([], error, status=400), CLAUDE_USAGE_LOG=str(log))
    with pytest.raises(cc.ClaudeAPIError) as caught:
        client.models.generate_content(model="claude-haiku-5-5", contents="p")
    assert caught.value.status == 400 and caught.value.is_billing and KEY not in str(caught.value)
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["ok"] is False and record["cost_usd"] == 0.0 and KEY not in log.read_text(encoding="utf-8")


def test_overloaded_error_is_not_billing(monkeypatch):
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    client = _client(monkeypatch, _capture([], error, status=529))
    with pytest.raises(cc.ClaudeAPIError) as caught:
        client.models.generate_content(model="m", contents="p")
    assert caught.value.status == 529 and not caught.value.is_billing


def test_timeout_is_logged_with_worst_case_cost(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = _client(monkeypatch, handler, CLAUDE_USAGE_LOG=str(log))
    with pytest.raises(TimeoutError):
        client.models.generate_content(model="claude-haiku-5-5", contents="p" * 1000)
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["error"] == "timeout" and record["cost_is_estimate"] is True and record["cost_basis"] == "upper_bound"
    assert record["cost_usd"] == pytest.approx(
        cc.worst_case_cost_usd("claude-haiku-5-5", 1000 + cc.PROMPT_FRAME_TOKENS, cc.DEFAULT_MAX_TOKENS))


def test_caller_timeout_in_milliseconds_is_respected(monkeypatch):
    seen: List[httpx.Request] = []
    client = _client(monkeypatch, _capture(seen, _ok()), CLAUDE_TIMEOUT_SECONDS="60")
    client.models.generate_content(model="m", contents="p", config={"http_options": {"timeout": 2500}})
    assert seen[0].extensions["timeout"]["read"] == pytest.approx(2.5)


def test_tools_are_rejected_not_ignored(monkeypatch):
    client = _client(monkeypatch, _capture([], _ok()))
    with pytest.raises(NotImplementedError):
        client.models.generate_content(model="m", contents="p", config={"tools": [object()]})
    with pytest.raises(TypeError):
        client.models.generate_content(model="m", contents=["p"])


def test_usage_log_has_no_prompt_or_answer_text(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    client = _client(monkeypatch, _capture([], _ok(text='{"secret_answer": 1}')), CLAUDE_USAGE_LOG=str(log),
                     CLAUDE_USAGE_RUN="r1")
    client.models.generate_content(model="claude-haiku-5-5", contents="비밀 질의 본문")
    raw = log.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert "비밀 질의 본문" not in raw and "secret_answer" not in raw and KEY not in raw
    assert record["run"] == "r1" and record["tag"] == "t" and record["ok"] is True
    assert record["usage"] == {"input_tokens": 1000, "output_tokens": 200}
    assert record["cost_usd"] == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1_000_000)


def test_budget_blocks_before_calling(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    # 비용을 모르는 기록은 상한(worst_case_usd)으로 센다 — 0으로 세지 않는다.
    log.write_text(json.dumps({"cost_usd": 0.5}) + "\n" + json.dumps({"cost_usd": None, "worst_case_usd": 0.499}) + "\n",
                   encoding="utf-8")
    seen: List[httpx.Request] = []
    client = _client(monkeypatch, _capture(seen, _ok()), CLAUDE_USAGE_LOG=str(log), CLAUDE_BUDGET_USD="1.00")
    with pytest.raises(cc.ClaudeBudgetExceeded):
        client.models.generate_content(model="claude-haiku-5-5", contents="p" * 5000)
    assert seen == []                                   # 부르지 않았다
    assert cc.spent_usd(log) == pytest.approx(0.999)


def test_record_without_cost_or_upper_bound_stops_the_budget(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    log.write_text(json.dumps({"cost_usd": None}) + "\n", encoding="utf-8")
    seen: List[httpx.Request] = []
    client = _client(monkeypatch, _capture(seen, _ok()), CLAUDE_USAGE_LOG=str(log), CLAUDE_BUDGET_USD="100")
    with pytest.raises(ValueError, match="0으로 세지 않는다"):
        client.models.generate_content(model="claude-haiku-5-5", contents="p")
    assert seen == []


def test_budget_without_ledger_is_an_error(monkeypatch):
    client = _client(monkeypatch, _capture([], _ok()), CLAUDE_BUDGET_USD="1.00")
    with pytest.raises(ValueError, match="CLAUDE_USAGE_LOG"):
        client.models.generate_content(model="m", contents="p")


def test_cost_uses_long_prompt_tier_and_unknown_model_is_none():
    base = cc.cost_usd("claude-haiku-5-5", {"input_tokens": 100_000, "output_tokens": 1000})
    long = cc.cost_usd("claude-haiku-5-5", {"input_tokens": 60_000, "cache_read_input_tokens": 40_001,
                                             "output_tokens": 1000})
    assert base == pytest.approx((100_000 * 0.10 + 1000 * 0.50) / 1e6)
    assert long == pytest.approx((60_000 * 0.50 + 40_001 * 0.05 + 1000 * 2.50) / 1e6)
    assert cc.cost_usd("some-other-model", {"input_tokens": 1}) is None


def test_key_lookup_order_and_missing_key(monkeypatch):
    monkeypatch.delenv("CLAUDE_API", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not cc.claude_configured()
    with pytest.raises(ValueError, match="CLAUDE_API"):
        cc.make_claude_genai_client()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "b")
    assert cc.claude_api_key() == "b"
    monkeypatch.setenv("CLAUDE_API", "a")
    assert cc.claude_api_key() == "a"


def test_raw_log_is_opt_in(monkeypatch, tmp_path):
    raw = tmp_path / "raw.jsonl"
    client = _client(monkeypatch, _capture([], _ok(text='{"a": 1}')))
    client.models.generate_content(model="m", contents="p")
    assert not raw.exists()
    client = _client(monkeypatch, _capture([], _ok(text='{"a": 1}')), CLAUDE_RAW_LOG=str(raw))
    monkeypatch.setenv("CLAUDE_RAW_LOG", str(raw))
    client.models.generate_content(model="m", contents="프롬프트 본문")
    line = json.loads(raw.read_text(encoding="utf-8"))
    assert line["text"] == '{"a": 1}' and "프롬프트 본문" not in raw.read_text(encoding="utf-8")
    # 프롬프트는 따로 켰을 때만 남는다.
    monkeypatch.setenv("CLAUDE_RAW_LOG_PROMPT", "1")
    client.models.generate_content(model="m", contents="프롬프트 본문")
    assert json.loads(raw.read_text(encoding="utf-8").splitlines()[-1])["prompt"] == "프롬프트 본문"


FENCED = '```json\n[{"index": 0}]\n```'


def test_fence_is_kept_by_default_and_recorded(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    client = _client(monkeypatch, _capture([], _ok(text=FENCED)), CLAUDE_USAGE_LOG=str(log))
    response = client.models.generate_content(model="claude-haiku-5-5", contents="p",
                                              config={"response_mime_type": "application/json"})
    assert response.text == FENCED
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["fenced"] is True and record["fence_stripped"] is False


def test_fence_strip_only_when_json_was_requested(monkeypatch, tmp_path):
    log, raw = tmp_path / "usage.jsonl", tmp_path / "raw.jsonl"
    client = _client(monkeypatch, _capture([], _ok(text=FENCED)), CLAUDE_USAGE_LOG=str(log), CLAUDE_JSON_FENCE="strip",
                     CLAUDE_RAW_LOG=str(raw))
    stripped = client.models.generate_content(model="claude-haiku-5-5", contents="p",
                                              config={"response_mime_type": "application/json"})
    assert stripped.text == '[{"index": 0}]'
    untouched = client.models.generate_content(model="claude-haiku-5-5", contents="p")     # JSON 요청이 아니다
    assert untouched.text == FENCED
    first, second = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert first["fenced"] and first["fence_stripped"] and second["fenced"] and not second["fence_stripped"]
    # 원문 기록에는 모델이 쓴 그대로가 남는다.
    assert json.loads(raw.read_text(encoding="utf-8").splitlines()[0])["text"] == FENCED


@pytest.mark.parametrize("text,expected", [
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('  ```\n{"a": 1}\n```  \n', '{"a": 1}'),
    ('{"a": "```코드```"}', '{"a": "```코드```"}'),                      # 본문 중간의 울타리
    ('설명입니다.\n```json\n{"a": 1}\n```', '설명입니다.\n```json\n{"a": 1}\n```'),   # 앞에 말이 붙으면 그대로(형식 위반으로 남는다)
    ('```json\n{"a": 1}', '```json\n{"a": 1}'),                          # 닫히지 않은 울타리
])
def test_strip_code_fence_only_unwraps_a_whole_answer(text, expected):
    assert cc.strip_code_fence(text) == expected


def test_budget_refuses_a_model_without_known_price(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    seen: List[httpx.Request] = []
    client = _client(monkeypatch, _capture(seen, _ok()), CLAUDE_USAGE_LOG=str(log), CLAUDE_BUDGET_USD="5.00")
    with pytest.raises(cc.ClaudeBudgetExceeded, match="단가"):
        client.models.generate_content(model="claude-unknown-model", contents="p")
    assert seen == []                                   # 비용을 0으로 세며 부르지 않는다


def test_sonnet_price_has_no_long_prompt_tier():
    cost = cc.cost_usd("claude-sonnet-5-5", {"input_tokens": 150_000, "output_tokens": 1000,
                                              "cache_read_input_tokens": 10_000})
    assert cost == pytest.approx((150_000 * 2.00 + 10_000 * 0.10 + 1000 * 10.00) / 1e6)


# ---- 상한 추정 · 예약 · 전체 시간 제한 (experiments/sonnet_comparison 전에 보강) ----

def test_worst_case_never_underestimates_the_bill():
    """토큰 수가 바이트 수를 넘지 않는 한(바이트 단위 BPE), 실제 청구액은 상한을 넘지 못한다."""
    prompt = "가사 한 줄 — 흐릿한 기억의 노래 🎵 " * 300 + "English words and JSON {\"id\":\"1\"} " * 100
    upper = cc.prompt_token_upper_bound(prompt)
    assert upper == len(prompt.encode("utf-8")) + cc.PROMPT_FRAME_TOKENS
    for model in cc.PRICES_PER_MTOK:
        worst = cc.worst_case_cost_usd(model, upper, 16000)
        # 가장 나쁜 청구: 입력 토큰 = 바이트 수, 그중 일부가 캐시 쓰기로 잡히고, 출력은 한도를 다 쓴다.
        byte_tokens = len(prompt.encode("utf-8"))
        for written in (0, byte_tokens // 2, byte_tokens):
            billed = cc.cost_usd(model, {"input_tokens": byte_tokens - written, "cache_creation_input_tokens": written,
                                         "output_tokens": 16000})
            assert billed <= worst + 1e-12
    assert cc.worst_case_cost_usd("claude-unknown-model", upper, 16000) is None


def test_worst_case_uses_long_tier_when_the_upper_bound_crosses_it():
    # 상한이 문턱을 넘으면 실제 토큰이 문턱 아래일 수 있어도 높은 등급으로 본다(과소평가하지 않는다).
    worst = cc.worst_case_cost_usd("claude-haiku-5-5", 100_001, 0)
    assert worst == pytest.approx(100_001 * 0.625 / 1e6)


@pytest.mark.parametrize("status,error_type,basis,billed", [
    (529, "overloaded_error", "upper_bound", True),     # 서버 쪽 오류 — 처리 여부를 모른다
    (500, "api_error", "upper_bound", True),
    (429, "rate_limit_error", "rejected_4xx", False),   # 요청을 거절했다
])
def test_error_responses_are_costed_by_what_we_know(monkeypatch, tmp_path, status, error_type, basis, billed):
    log = tmp_path / "usage.jsonl"
    error = {"type": "error", "error": {"type": error_type, "message": "x"}}
    client = _client(monkeypatch, _capture([], error, status=status), CLAUDE_USAGE_LOG=str(log))
    with pytest.raises(cc.ClaudeAPIError):
        client.models.generate_content(model="claude-sonnet-5-5", contents="p")
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["cost_basis"] == basis and (record["cost_usd"] > 0) is billed


@pytest.mark.parametrize("exc,basis", [
    (httpx.ConnectError, "not_sent"),
    (httpx.ReadError, "upper_bound"),                   # 보낸 뒤 끊겼다
    (httpx.RemoteProtocolError, "upper_bound"),
])
def test_transport_errors_are_zero_only_when_nothing_was_sent(monkeypatch, tmp_path, exc, basis):
    log = tmp_path / "usage.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc("x", request=request)

    client = _client(monkeypatch, handler, CLAUDE_USAGE_LOG=str(log))
    with pytest.raises(cc.ClaudeError):
        client.models.generate_content(model="claude-sonnet-5-5", contents="p")
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["cost_basis"] == basis
    assert record["cost_usd"] == (0.0 if basis == "not_sent" else pytest.approx(record["worst_case_usd"]))


def test_ok_status_without_usage_counts_the_upper_bound(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    payload = {"model": "claude-sonnet-5-5", "stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}]}
    client = _client(monkeypatch, _capture([], payload), CLAUDE_USAGE_LOG=str(log))
    client.models.generate_content(model="claude-sonnet-5-5", contents="p")
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["ok"] and record["cost_basis"] == "upper_bound" and record["cost_usd"] == record["worst_case_usd"]


def test_whole_request_time_is_cut_even_when_bytes_keep_arriving(monkeypatch, tmp_path):
    """읽기 제한은 조각 하나를 기다리는 값이다 — 조각이 계속 오면 끝나지 않는다. 전체 시간으로 끊는다."""
    import time as _time

    log = tmp_path / "usage.jsonl"

    async def trickle():
        for _ in range(30):
            await asyncio.sleep(0.1)
            yield b" "

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=trickle())

    client = _client(monkeypatch, handler, CLAUDE_USAGE_LOG=str(log), CLAUDE_TIMEOUT_SECONDS="0.4")
    started = _time.monotonic()
    with pytest.raises(TimeoutError):
        client.models.generate_content(model="claude-sonnet-5-5", contents="p")
    assert _time.monotonic() - started < 1.5            # 3초짜리 응답을 끝까지 기다리지 않았다
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["error"] == "timeout" and record["cost_basis"] == "upper_bound"


def _wait_for_lines(path, count: int, seconds: float = 3.0) -> List[Dict[str, Any]]:
    import time as _time

    deadline = _time.monotonic() + seconds
    while _time.monotonic() < deadline:
        if path.exists():
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) >= count:
                return lines
        _time.sleep(0.02)
    raise AssertionError(f"{path.name}에 기록 {count}줄이 {seconds}초 안에 생기지 않았다")


def test_async_caller_cancellation_cancels_the_request(monkeypatch, tmp_path):
    """리랭커는 aio 호출을 wait_for(남은 예산)로 감싼다. 끊으면 진행 중인 요청이 취소되고 상한으로 기록된다."""
    log = tmp_path / "usage.jsonl"
    cancelled: List[bool] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return httpx.Response(200, json=_ok())

    client = _client(monkeypatch, handler, CLAUDE_USAGE_LOG=str(log), CLAUDE_TIMEOUT_SECONDS="60")

    async def call():
        return await asyncio.wait_for(client.aio.models.generate_content(model="claude-sonnet-5-5", contents="p"), 0.2)

    with pytest.raises(TimeoutError):
        asyncio.run(call())
    record = _wait_for_lines(log, 1)[0]
    assert record["error"] == "cancelled" and record["cost_basis"] == "upper_bound" and cancelled == [True]


def test_reservations_stop_concurrent_calls_from_overshooting(monkeypatch, tmp_path):
    """둘이 동시에 '아직 여유가 있다'고 보고 함께 넘는 일이 없어야 한다 — 진행 중인 호출의 상한을 예약해 둔다."""
    import threading

    log = tmp_path / "usage.jsonl"
    release = threading.Event()
    entered: List[bool] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.append(True)
        while not release.is_set():
            await asyncio.sleep(0.01)
        return httpx.Response(200, json=_ok())

    worst = cc.worst_case_cost_usd("claude-sonnet-5-5", cc.prompt_token_upper_bound("p"), cc.DEFAULT_MAX_TOKENS)
    client = _client(monkeypatch, handler, CLAUDE_USAGE_LOG=str(log), CLAUDE_BUDGET_USD=f"{worst * 1.5:.6f}")
    first: Dict[str, Any] = {}
    thread = threading.Thread(target=lambda: first.setdefault(
        "response", client.models.generate_content(model="claude-sonnet-5-5", contents="p")))
    thread.start()
    import time as _time
    while not entered:
        _time.sleep(0.01)
    assert cc.pending_usd(log) == pytest.approx(worst)
    with pytest.raises(cc.ClaudeBudgetExceeded, match="진행 중"):
        client.models.generate_content(model="claude-sonnet-5-5", contents="p")
    release.set()
    thread.join(5)
    assert first["response"].text == '{"ok": true}' and len(entered) == 1
    assert cc.pending_usd(log) == 0.0 and len(_wait_for_lines(log, 1)) == 1


def test_open_reservation_from_a_dead_process_is_still_counted(monkeypatch, tmp_path):
    """다른 프로세스가 예약하고 끝내지 못한 채 죽었다 — 그 상한은 계속 센다."""
    import subprocess
    import sys
    from pathlib import Path

    log = tmp_path / "usage.jsonl"
    worst = cc.worst_case_cost_usd("claude-sonnet-5-5", cc.prompt_token_upper_bound("p"), cc.DEFAULT_MAX_TOKENS)
    env = {"PATH": "/usr/bin:/bin", "CLAUDE_USAGE_LOG": str(log), "CLAUDE_BUDGET_USD": f"{worst * 1.5:.6f}"}
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", f"from src.common import claude_client as cc; cc._reserve('claude-sonnet-5-5', {worst!r})"],
                   cwd=root, env=env, check=True, timeout=60)
    seen: List[httpx.Request] = []
    client = _client(monkeypatch, _capture(seen, _ok()), CLAUDE_USAGE_LOG=str(log), CLAUDE_BUDGET_USD=f"{worst * 1.5:.6f}")
    with pytest.raises(cc.ClaudeBudgetExceeded):
        client.models.generate_content(model="claude-sonnet-5-5", contents="p")
    assert seen == []


# ---- structured outputs ----

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


def test_schema_in_config_becomes_output_format_next_to_effort(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    seen: List[httpx.Request] = []
    client = _client(monkeypatch, _capture(seen, _ok()), VF_CLAUDE_EFFORT="medium", CLAUDE_USAGE_LOG=str(log))
    client.models.generate_content(model="claude-sonnet-5-5", contents="p",
                                   config={"response_mime_type": "application/json", "response_json_schema": SCHEMA})
    body = json.loads(seen[0].content)
    assert body["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["output_format"] == "json_schema"
    # 스키마 바이트만큼 입력 상한을 늘린다.
    assert record["input_tokens_max"] == cc.prompt_token_upper_bound("p") + len(json.dumps(SCHEMA).encode("utf-8"))


def test_no_schema_means_no_format(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    seen: List[httpx.Request] = []
    client = _client(monkeypatch, _capture(seen, _ok()), CLAUDE_USAGE_LOG=str(log))
    client.models.generate_content(model="claude-sonnet-5-5", contents="p", config={"response_mime_type": "application/json"})
    assert "format" not in json.loads(seen[0].content).get("output_config", {})
    assert json.loads(log.read_text(encoding="utf-8"))["output_format"] == "text"


def test_structured_output_setting_is_validated_without_echo(monkeypatch):
    monkeypatch.setenv("CLAUDE_STRUCTURED_OUTPUT", "json_schema")
    assert cc.claude_structured_output() == "json_schema"
    assert cc.claude_request_settings()["structured_output"] == "json_schema"
    monkeypatch.setenv("CLAUDE_STRUCTURED_OUTPUT", "key-pasted-by-mistake")
    with pytest.raises(ValueError) as caught:
        cc.claude_structured_output()
    assert "key-pasted-by-mistake" not in str(caught.value)
