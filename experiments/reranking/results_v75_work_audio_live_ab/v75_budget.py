"""v69 — Vertex 호출 예산 통제(v68 `v68_budget.py` 사본 + 아래 v69 변경). v66 규칙: v63 규칙 + **첫 HTTP 시도 기준 마감** + 실행 사이 원자적 예약.

v69 변경(첫 Vertex 시도 전에 고정 · 운영 서비스 재시도 정책은 그대로 — 이 실험 실행기에만 적용):
- **장애 집중 정지**: 장부 전체(두 실행 공유) 기준 ① 연속 504 3회 — 마지막 성공 정산 뒤 504(`code` 504 또는 `DEADLINE_EXCEEDED`)가
  3회. 504가 아닌 실패(재시도 취소 등)는 끊지도 세지도 않는다(v68 B1처럼 504 → 재시도 취소가 번갈아 오면 끊기지 않게)
  ② 최근 10분(600초) 안 504 5회. 둘 중 하나면 장부에 정지를 적어 **두 실행 모두** 더 보내지 않는다.
- 기존 규칙 유지·강화: 인증·권한·결제 첫 오류 정지 · 연속 실패 5회(v68은 프로세스 안에서만 셌다 → 장부 전체, 마지막 성공 정산 뒤)
  · 누적 실패 20회.
- 비용 판단에서 진행 중 예약을 한 번만 센다: v68은 장부의 열린 예약(이 프로세스 진행분 포함) + 프로세스 진행분을 더해 두 번
  셌다(보수 쪽 과대 — v68 A2 정지는 확정·미확인 $3.8903 + 이번 예약 $0.1133 > $4로 진행분 0이라 영향 없음).

감싸는 자리: `google.genai._api_client.BaseApiClient._request_once` · `_async_request_once` — SDK가 **실제로 HTTP 요청을
한 번 보내는 자리**다. SDK 재시도는 이 함수를 다시 부르므로 시도마다 한 번씩 잡힌다(google-genai 2.23.0: 클라이언트에
`retry_options`가 없으면 `stop_after_attempt(1)` — 이 실험의 리랭커·preflight는 넘기지 않는다). countTokens도 시도로 센다.

- 장부: JSONL 하나를 모든 실행(4실행, 스모크 포함)이 공유한다. 예약 판단과 기록은 `ledger.lock` 파일 잠금 안에서 한 번에 한다.
- **마감**: 장부의 첫 시도 예약 때 `anchor`(첫 시도 시각 + 창, 기본 5,400초 = 90분)를 같은 잠금 안에서 적는다. 이후 실행은
  그 값을 읽기만 한다 — 실행마다 다시 세지 않는다. 마감 이후에는 새 시도를 보내지 않는다(진행 중 요청은 서비스의 기존
  요청 제한 20초·공유 예산 30초로 끝난다).
- 예약(보내기 전): 입력 = 요청 본문 JSON 글자 수를 토큰 수 상한으로 보고(ensure_ascii라 한글 1자 = 6글자 → 과대 추정)
  × 입력 단가 + 출력 = 모델 최대 출력 65,536토큰(응답+thinking) × 출력 단가. countTokens는 0원.
- 정산(응답 뒤): `usageMetadata`의 prompt·toolUsePrompt(입력) + candidates·thoughts(출력). 캐시 토큰도 입력 정가로 센다(보수적).
- 실패(예외·취소·시간 초과·사용량 없는 응답): 예약액을 **미확인 비용**으로 남긴다(0으로 지우지 않는다).
- 보내지 않는 조건: 확정 + 미확인 + 진행 중 예약(장부의 닫히지 않은 예약 포함) + 이번 예약 > 비용 상한 · 시도 수 ≥ 상한 · 마감 이후.
- 정지(이후 모든 시도 차단 — `v69_eval_runtime.RunAborted`로 실행을 끊는다): 401·403, 400 중 billing/permission/credential 문구,
  google-auth 인증 오류(RefreshError·DefaultCredentialsError) 1회 · 연속 실패 5회 · 누적 실패 20회 · 위의 보내지 않는 조건.

단가(USD/100만 토큰, Vertex AI 공식 가격표 "Gemini 3" Standard 표, Global 엔드포인트, ≤200K 입력):
gemini-3.1-flash-lite 입력(text·image·video) 0.25 · 출력(response and reasoning) 1.50 —
https://cloud.google.com/vertex-ai/generative-ai/pricing (2026-10-17 18:4x KST 확인, v69에서 2026-10-18 00:20 KST 다시 내려받아
같은 값 확인 — 본문을 내려받아 표에서 직접 읽음).
같은 표의 Priority 등급은 0.45/2.70 — 이 코드는 등급을 지정하지 않는다(Standard PayGo). Non-global은 0.275/1.65(쓰지 않음).
최대 출력 65,536토큰 · 입력 1,048,576토큰은 Vertex 모델 문서(gemini-3.1-flash-lite, GA, 2026-10-16 UTC 갱신) 확인값.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

PRICES = {"gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50}}
MAX_OUTPUT_TOKENS = 65_536
AUTH_EXC_NAMES = ("RefreshError", "DefaultCredentialsError", "TransportError")

GUARD: Optional["BudgetGuard"] = None

# 장애 집중 정지 상수(v69, 첫 시도 전 고정)
STREAK_504 = 3
WINDOW_504_SECONDS = 600
WINDOW_504_COUNT = 5
CONSECUTIVE_FAILURES = 5
CUMULATIVE_FAILURES = 20


def is_504(row: Dict[str, Any]) -> bool:
    return row.get("error_code") == 504 or row.get("error_status") == "DEADLINE_EXCEEDED"


class BudgetHalt(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


class BudgetGuard:
    def __init__(self, ledger: Path, *, cost_cap_usd: float, attempt_cap: int, deadline_window_seconds: float,
                 run_tag: str, model: str, clock=time.time) -> None:
        if model not in PRICES:
            raise SystemExit(f"단가를 모르는 모델: {model}")
        self.ledger = ledger
        self.lockfile = ledger.with_suffix(ledger.suffix + ".lock")
        self.cost_cap = float(cost_cap_usd)
        self.attempt_cap = int(attempt_cap)
        self.window = float(deadline_window_seconds)
        self.run_tag = run_tag
        self.model = model
        self.price = PRICES[model]
        self.clock = clock
        self.lock = threading.Lock()
        self.inflight: Dict[str, float] = {}
        self.halted = False
        self.halt_reason = ""
        self.consecutive_failures = 0
        self.local = {"attempts": 0, "generate": 0, "count_tokens": 0, "other": 0, "settled": 0, "failed": 0,
                      "blocked": 0, "cost": 0.0, "unverified": 0.0}
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.touch(exist_ok=True)
        self.lockfile.touch(exist_ok=True)
        totals = self.totals()
        if totals["halt"]:
            self._halt(f"장부에 이전 정지 기록: {totals['halt']}", write=False)

    # ------------------------------------------------------------------ 장부
    @contextmanager
    def _file_lock(self):
        with self.lockfile.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _append(self, row: Dict[str, Any]) -> None:
        row = {"ts": _now(), "epoch": round(self.clock(), 3), "run": self.run_tag, "pid": os.getpid(), **row}
        with self.ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def totals(self) -> Dict[str, Any]:
        """장부 전체(모든 실행)를 다시 읽어 합한다. 정산·실패가 없는 예약(프로세스가 죽은 경우)은 미확인으로 센다."""
        reserves: Dict[str, Dict[str, Any]] = {}
        closed: Dict[str, Dict[str, Any]] = {}
        halt = ""
        anchor = None
        # v69: 장부 순서대로 마지막 성공 정산 뒤의 연속 실패·연속 504와 504 시각(두 실행 공유)
        streak_any = 0
        streak_504 = 0
        epochs_504: list = []
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("event")
            if kind == "reserve":
                reserves[row["id"]] = row
            elif kind in {"settle", "fail"}:
                closed[row["id"]] = row
                if kind == "settle":
                    streak_any = streak_504 = 0
                else:
                    streak_any += 1
                    if is_504(row):
                        streak_504 += 1
                        epochs_504.append(float(row.get("epoch") or 0.0))
            elif kind == "halt":
                halt = row.get("reason", "")
            elif kind == "anchor" and anchor is None:
                anchor = row
        settled = sum(r.get("cost_usd", 0.0) for r in closed.values() if r["event"] == "settle")
        unverified = sum(r.get("unverified_usd", 0.0) for r in closed.values() if r["event"] == "fail")
        open_ids = [i for i in reserves if i not in closed]
        open_cost = sum(reserves[i]["reserve_usd"] for i in open_ids)
        kinds: Dict[str, int] = {}
        for r in reserves.values():
            kinds[r.get("kind", "?")] = kinds.get(r.get("kind", "?"), 0) + 1
        return {
            "attempts": len(reserves),
            "attempts_by_kind": kinds,
            "settled_calls": sum(1 for r in closed.values() if r["event"] == "settle"),
            "failed_calls": sum(1 for r in closed.values() if r["event"] == "fail"),
            "open_reservations": len(open_ids),
            "settled_usd": round(settled, 6),
            "unverified_usd": round(unverified + open_cost, 6),
            "open_reserve_usd": round(open_cost, 6),
            "committed_usd": round(settled + unverified + open_cost, 6),
            "deadline_epoch": anchor["deadline_epoch"] if anchor else None,
            "first_attempt_epoch": anchor["first_attempt_epoch"] if anchor else None,
            "halt": halt,
            "consecutive_failures_since_success": streak_any,
            "consecutive_504_since_success": streak_504,
            "fail_504_epochs": epochs_504,
        }

    def deadline_passed(self) -> bool:
        deadline = self.totals()["deadline_epoch"]
        return deadline is not None and self.clock() >= deadline

    def _halt(self, reason: str, write: bool = True) -> None:
        self.halted = True
        self.halt_reason = reason
        if write:
            with self._file_lock():
                self._append({"event": "halt", "reason": reason})

    # ------------------------------------------------------------------시도 하나
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
            reason = ""
            with self._file_lock():
                totals = self.totals()
                now = self.clock()
                inflight = sum(self.inflight.values())
                deadline = totals["deadline_epoch"]
                if totals["halt"]:
                    reason = f"장부 정지 기록: {totals['halt']}"
                elif totals["attempts"] >= self.attempt_cap:
                    reason = f"SDK 요청 시도 상한 {self.attempt_cap}회 도달"
                elif deadline is not None and now >= deadline:
                    reason = "신규 호출 마감(첫 시도 + 창) 이후"
                elif totals["committed_usd"] + reserve > self.cost_cap:
                    # committed = 확정 + 미확인 + 장부의 열린 예약(이 프로세스 진행분 포함) — 진행분을 다시 더하지 않는다(v69)
                    reason = (f"비용 상한 ${self.cost_cap}: 확정·미확인·열린 예약 ${totals['committed_usd']:.4f}"
                              f"(그중 이 프로세스 진행 중 ${inflight:.4f}) + 이번 예약 ${reserve:.4f}")
                if reason:
                    self.local["blocked"] += 1
                    self.halted = True
                    self.halt_reason = reason
                    self._append({"event": "halt", "reason": reason, "blocked_kind": kind})
                else:
                    if deadline is None:
                        self._append({"event": "anchor", "first_attempt_epoch": round(now, 3),
                                      "window_seconds": self.window, "deadline_epoch": round(now + self.window, 3)})
                    attempt_id = uuid.uuid4().hex[:12]
                    self.inflight[attempt_id] = reserve
                    self.local["attempts"] += 1
                    self.local[kind] += 1
                    self._append({"event": "reserve", "id": attempt_id, "kind": kind, "model": self.model,
                                  "body_chars": body_chars, "reserve_usd": round(reserve, 6)})
            if reason:
                raise BudgetHalt(reason)
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
            with self._file_lock():
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
                                  "total_tokens": usage.get("totalTokenCount"),
                                  "count_total_tokens": (payload or {}).get("totalTokens") if ticket["kind"] == "count_tokens" else None})
                else:
                    self.local["unverified"] += ticket["reserve"]
                    self._append({"event": "fail", "id": ticket["id"], "kind": ticket["kind"], "latency_ms": latency,
                                  "error": "usage_missing", "unverified_usd": round(ticket["reserve"], 6)})

    def fail(self, ticket: Dict[str, Any], exc: BaseException) -> None:
        code = getattr(exc, "code", None)
        status = getattr(exc, "status", None)
        message = str(getattr(exc, "message", "") or exc)[:300].lower()
        latency = round((time.perf_counter() - ticket["started"]) * 1000, 1)
        with self.lock:
            self.inflight.pop(ticket["id"], None)
            self.local["failed"] += 1
            self.local["unverified"] += ticket["reserve"]
            self.consecutive_failures += 1
            with self._file_lock():
                self._append({"event": "fail", "id": ticket["id"], "kind": ticket["kind"], "latency_ms": latency,
                              "error": type(exc).__name__, "error_code": code, "error_status": status,
                              "unverified_usd": round(ticket["reserve"], 6)})
                totals = self.totals()
            reason = ""
            now = self.clock()
            recent_504 = sum(1 for e in totals["fail_504_epochs"] if e >= now - WINDOW_504_SECONDS)
            if (code in (401, 403) or type(exc).__name__ in AUTH_EXC_NAMES
                    or (code == 400 and any(w in message for w in ("billing", "permission", "credential")))):
                reason = f"인증·권한·결제 오류 {code} {status} {type(exc).__name__} — 재시도하지 않는다"
            elif totals["consecutive_504_since_success"] >= STREAK_504:
                reason = f"장애 집중: 연속 504 {totals['consecutive_504_since_success']}회(마지막 성공 정산 뒤)"
            elif recent_504 >= WINDOW_504_COUNT:
                reason = f"장애 집중: 최근 {WINDOW_504_SECONDS}초 안 504 {recent_504}회"
            elif max(self.consecutive_failures, totals["consecutive_failures_since_success"]) >= CONSECUTIVE_FAILURES:
                reason = f"연속 실패 {max(self.consecutive_failures, totals['consecutive_failures_since_success'])}회"
            elif totals["failed_calls"] >= CUMULATIVE_FAILURES:
                reason = f"누적 실패 {totals['failed_calls']}회"
        if reason:
            self._halt(reason)


def install(ledger: Path, *, cost_cap_usd: float, attempt_cap: int, deadline_window_seconds: float, run_tag: str,
            model: str) -> BudgetGuard:
    global GUARD
    import google.genai._api_client as api

    guard = BudgetGuard(ledger, cost_cap_usd=cost_cap_usd, attempt_cap=attempt_cap,
                        deadline_window_seconds=deadline_window_seconds, run_tag=run_tag, model=model)
    GUARD = guard
    if getattr(api.BaseApiClient, "_v75_budget_installed", False):
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
    api.BaseApiClient._v75_budget_installed = True
    return guard
