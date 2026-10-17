"""v63 — Vertex 호출 예산 통제(HTTP 시도 단위 예약·정산·한도).

감싸는 자리: `google.genai._api_client.BaseApiClient._request_once` · `_async_request_once` — SDK가 **실제로 HTTP 요청을
한 번 보내는 자리**다(SDK 재시도는 이 함수를 다시 부르므로 시도마다 한 번씩 잡힌다; aiohttp가 없어 내부 재시도 경로도 없음 —
`v63_runtime.verify_guards`가 확인). 시도마다 장부(JSONL, 실행 사이 공유)에 먼저 예약을 적고 끝나면 정산한다.

- 예약(호출 전): 입력 = 요청 본문 JSON 글자 수를 토큰 수 상한으로 보고(ensure_ascii라 한글 1자 = 6글자 → 과대 추정)
  × 입력 단가, 출력 = 모델 최대 출력 65,536토큰(thinking 포함) × 출력 단가. countTokens는 0원이지만 시도 수에는 센다.
- 정산(응답 뒤): `usageMetadata`의 prompt·candidates·thoughts·toolUsePrompt 토큰. **캐시 토큰도 입력 정가로 센다**(할인 무시, 보수적).
- 실패(예외·취소·시간 초과): 사용량을 모르므로 **예약액을 미확인 비용으로 남긴다**(0으로 지우지 않는다).
- 한도: 확정 + 미확인 + 진행 중 예약 + 이번 예약 > 비용 상한이면 보내지 않는다. 시도 수 ≥ 상한, 마감 시각 이후,
  인증·권한·결제 오류(401·403, 그리고 400 중 billing/permission 문구) 1회, 연속 실패 5회, 누적 실패 20회면 **정지**
  (이후 모든 시도를 보내지 않고 `RunAborted`로 실행을 끊는다).

단가(USD/100만 토큰): gemini-3.1-flash-lite 입력 0.25 · 출력 1.50 — Google 공식 Gemini API 가격표 표준 등급
(https://ai.google.dev/gemini-api/docs/pricing, 2026-10-17 01:2x KST 확인). Vertex AI 가격 페이지는 스크립트로 본문이 읽히지 않았다
(v50과 같음) — 집계 사이트의 Vertex 표기도 같은 값. 최대 출력 65,536은 Vertex 모델 문서(3-1-flash-lite) 확인값.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

PRICES = {"gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50}}
MAX_OUTPUT_TOKENS = 65_536

GUARD: Optional["BudgetGuard"] = None


class BudgetHalt(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


class BudgetGuard:
    def __init__(self, ledger: Path, *, cost_cap_usd: float, attempt_cap: int, deadline_epoch: float,
                 run_tag: str, model: str) -> None:
        if model not in PRICES:
            raise SystemExit(f"단가를 모르는 모델: {model}")
        self.ledger = ledger
        self.cost_cap = float(cost_cap_usd)
        self.attempt_cap = int(attempt_cap)
        self.deadline = float(deadline_epoch)
        self.run_tag = run_tag
        self.model = model
        self.price = PRICES[model]
        self.lock = threading.Lock()
        self.inflight: Dict[str, float] = {}
        self.halted = False
        self.halt_reason = ""
        self.consecutive_failures = 0
        self.local = {"attempts": 0, "settled": 0, "failed": 0, "blocked": 0, "cost": 0.0, "unverified": 0.0}
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.touch(exist_ok=True)
        totals = self.totals()
        if totals["halt"]:
            self._halt(f"장부에 이전 정지 기록: {totals['halt']}", write=False)

    # ------------------------------------------------------------------ 장부
    def _append(self, row: Dict[str, Any]) -> None:
        row = {"ts": _now(), "run": self.run_tag, "pid": os.getpid(), **row}
        with self.ledger.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)

    def totals(self) -> Dict[str, Any]:
        """장부 전체(모든 실행)를 다시 읽어 합한다. 정산·실패가 없는 예약(프로세스가 죽은 경우)은 미확인으로 센다."""
        reserves: Dict[str, Dict[str, Any]] = {}
        closed: Dict[str, Dict[str, Any]] = {}
        halt = ""
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("event")
            if kind == "reserve":
                reserves[row["id"]] = row
            elif kind in {"settle", "fail"}:
                closed[row["id"]] = row
            elif kind == "halt":
                halt = row.get("reason", "")
        settled = sum(r.get("cost_usd", 0.0) for r in closed.values() if r["event"] == "settle")
        unverified = sum(r.get("unverified_usd", 0.0) for r in closed.values() if r["event"] == "fail")
        open_ids = [i for i in reserves if i not in closed]
        open_cost = sum(reserves[i]["reserve_usd"] for i in open_ids)
        return {
            "attempts": len(reserves),
            "settled_calls": sum(1 for r in closed.values() if r["event"] == "settle"),
            "failed_calls": sum(1 for r in closed.values() if r["event"] == "fail"),
            "open_reservations": len(open_ids),
            "settled_usd": round(settled, 6),
            "unverified_usd": round(unverified + open_cost, 6),
            "committed_usd": round(settled + unverified + open_cost, 6),
            "halt": halt,
        }

    def _halt(self, reason: str, write: bool = True) -> None:
        self.halted = True
        self.halt_reason = reason
        if write:
            self._append({"event": "halt", "reason": reason})

    # ------------------------------------------------------------------ 시도 하나
    def reserve(self, url: str, body_chars: int) -> Dict[str, Any]:
        is_count = ":countTokens" in url
        kind = "count_tokens" if is_count else ("generate" if ":generateContent" in url else "other")
        reserve = 0.0 if is_count else (
            body_chars * self.price["input"] + MAX_OUTPUT_TOKENS * self.price["output"]
        ) / 1e6
        with self.lock:
            if self.halted:
                self.local["blocked"] += 1
                raise BudgetHalt(f"정지 상태 — {self.halt_reason}")
            totals = self.totals()
            inflight = sum(self.inflight.values())
            reason = ""
            if totals["attempts"] >= self.attempt_cap:
                reason = f"SDK 요청 시도 상한 {self.attempt_cap}회 도달"
            elif time.time() >= self.deadline:
                reason = "작업 시간 마감 시각 이후"
            elif totals["committed_usd"] + inflight + reserve > self.cost_cap:
                reason = (f"비용 상한 ${self.cost_cap}: 확정·미확인 ${totals['committed_usd']:.4f} + 진행 중 ${inflight:.4f}"
                          f" + 이번 예약 ${reserve:.4f}")
            if reason:
                self.local["blocked"] += 1
                self._halt(reason)
                raise BudgetHalt(reason)
            attempt_id = uuid.uuid4().hex[:12]
            self.inflight[attempt_id] = reserve
            self.local["attempts"] += 1
            self._append({"event": "reserve", "id": attempt_id, "kind": kind, "model": self.model,
                          "body_chars": body_chars, "reserve_usd": round(reserve, 6)})
        return {"id": attempt_id, "kind": kind, "reserve": reserve, "started": time.perf_counter()}

    def settle(self, ticket: Dict[str, Any], body_text: str) -> None:
        usage: Dict[str, Any] = {}
        model_version = None
        try:
            payload = json.loads(body_text) if body_text else {}
            usage = payload.get("usageMetadata") or {}
            model_version = payload.get("modelVersion")
        except Exception:  # noqa: BLE001 - 본문을 못 읽으면 예약액을 미확인으로 남긴다
            payload = None
        latency = round((time.perf_counter() - ticket["started"]) * 1000, 1)
        if ticket["kind"] == "count_tokens":
            cost = 0.0
            known = True
        else:
            known = bool(usage) and usage.get("promptTokenCount") is not None
            prompt = int(usage.get("promptTokenCount") or 0) + int(usage.get("toolUsePromptTokenCount") or 0)
            out = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
            cost = (prompt * self.price["input"] + out * self.price["output"]) / 1e6
        with self.lock:
            self.inflight.pop(ticket["id"], None)
            self.consecutive_failures = 0
            if known:
                self.local["settled"] += 1
                self.local["cost"] += cost
                self._append({"event": "settle", "id": ticket["id"], "kind": ticket["kind"], "latency_ms": latency,
                              "cost_usd": round(cost, 8), "model_version": model_version,
                              "prompt_tokens": usage.get("promptTokenCount"),
                              "candidates_tokens": usage.get("candidatesTokenCount"),
                              "thoughts_tokens": usage.get("thoughtsTokenCount"),
                              "cached_tokens": usage.get("cachedContentTokenCount"),
                              "tool_prompt_tokens": usage.get("toolUsePromptTokenCount"),
                              "total_tokens": usage.get("totalTokenCount")})
            else:
                self.local["unverified"] += ticket["reserve"]
                self._append({"event": "fail", "id": ticket["id"], "kind": ticket["kind"], "latency_ms": latency,
                              "error": "usage_missing", "unverified_usd": round(ticket["reserve"], 6)})

    def fail(self, ticket: Dict[str, Any], exc: BaseException) -> None:
        code = getattr(exc, "code", None)
        status = getattr(exc, "status", None)
        message = str(getattr(exc, "message", "") or "")[:300].lower()
        latency = round((time.perf_counter() - ticket["started"]) * 1000, 1)
        with self.lock:
            self.inflight.pop(ticket["id"], None)
            self.local["failed"] += 1
            self.local["unverified"] += ticket["reserve"]
            self.consecutive_failures += 1
            self._append({"event": "fail", "id": ticket["id"], "kind": ticket["kind"], "latency_ms": latency,
                          "error": type(exc).__name__, "error_code": code, "error_status": status,
                          "unverified_usd": round(ticket["reserve"], 6)})
            totals = self.totals()
            if code in (401, 403) or (code == 400 and any(w in message for w in ("billing", "permission", "credential"))):
                self._halt(f"인증·권한·결제 오류 {code} {status} — 재시도하지 않는다")
            elif self.consecutive_failures >= 5:
                self._halt(f"연속 실패 {self.consecutive_failures}회")
            elif totals["failed_calls"] >= 20:
                self._halt(f"누적 실패 {totals['failed_calls']}회")


def install(ledger: Path, *, cost_cap_usd: float, attempt_cap: int, deadline_epoch: float, run_tag: str,
            model: str) -> BudgetGuard:
    global GUARD
    import google.genai._api_client as api

    guard = BudgetGuard(ledger, cost_cap_usd=cost_cap_usd, attempt_cap=attempt_cap, deadline_epoch=deadline_epoch,
                        run_tag=run_tag, model=model)
    GUARD = guard
    if getattr(api.BaseApiClient, "_v63_budget_installed", False):
        return guard

    orig_once = api.BaseApiClient._request_once
    orig_async_once = api.BaseApiClient._async_request_once

    def _body_chars(http_request: Any) -> int:
        data = getattr(http_request, "data", None)
        if not data:
            return 0
        if isinstance(data, (bytes, str)):
            return len(data)
        return len(json.dumps(data))

    def _text(response: Any) -> str:
        stream = getattr(response, "response_stream", None)
        if isinstance(stream, list) and stream and isinstance(stream[0], str):
            return stream[0]
        return ""

    def request_once(self, http_request, stream=False):
        ticket = GUARD.reserve(str(http_request.url), _body_chars(http_request))
        try:
            response = orig_once(self, http_request, stream)
        except BaseException as exc:
            GUARD.fail(ticket, exc)
            raise
        GUARD.settle(ticket, _text(response))
        return response

    async def async_request_once(self, http_request, stream=False):
        ticket = GUARD.reserve(str(http_request.url), _body_chars(http_request))
        try:
            response = await orig_async_once(self, http_request, stream)
        except BaseException as exc:
            GUARD.fail(ticket, exc)
            raise
        GUARD.settle(ticket, _text(response))
        return response

    api.BaseApiClient._request_once = request_once  # type: ignore[assignment]
    api.BaseApiClient._async_request_once = async_request_once  # type: ignore[assignment]
    api.BaseApiClient._v63_budget_installed = True
    return guard
