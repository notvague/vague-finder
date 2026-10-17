"""v66_budget 자체 검사 — 네트워크 없음, 임시 장부만. 가짜 `_request_once`·`_async_request_once`로
예약·정산·미확인·비용/시도 한도·첫 시도 기준 마감(실행 사이 공유·초기화 없음)·정지 이어받기·인증 오류를 확인한다.

    venv/bin/python selftest_budget.py   # 종료 코드 0 = 전부 통과
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import google.genai._api_client as api  # noqa: E402

calls = []


class FakeError(Exception):
    def __init__(self, code, message="server"):
        super().__init__(f"fake {code}")
        self.code = code
        self.status = "FAKE"
        self.message = message


class RefreshError(Exception):  # google.auth.exceptions.RefreshError와 같은 이름
    pass


def fake_once(self, http_request, stream=False):
    calls.append(http_request.url)
    behaviour = http_request.data.get("_fake")
    if behaviour == "403":
        raise FakeError(403, "permission denied")
    if behaviour == "400billing":
        raise FakeError(400, "billing account disabled")
    if behaviour == "500":
        raise FakeError(500)
    if behaviour == "refresh":
        raise RefreshError("invalid_grant")
    usage = {"promptTokenCount": 10_000, "candidatesTokenCount": 1_000, "thoughtsTokenCount": 500, "totalTokenCount": 11_500}
    if behaviour == "nousage":
        usage = {}
    return SimpleNamespace(response_stream=[json.dumps({"usageMetadata": usage, "modelVersion": "gemini-3.1-flash-lite"})])


async def fake_async_once(self, http_request, stream=False):
    return fake_once(self, http_request, stream)


api.BaseApiClient._request_once = fake_once
api.BaseApiClient._async_request_once = fake_async_once

import v66_budget as vb  # noqa: E402

failures = 0
MODEL = "gemini-3.1-flash-lite"


def check(name, cond):
    global failures
    print(("ok   " if cond else "FAIL ") + name)
    failures += 0 if cond else 1


def req(kind="generate", fake=None, chars=1000):
    url = "https://x/models/gemini-3.1-flash-lite:" + ("countTokens" if kind == "count" else "generateContent")
    return SimpleNamespace(url=url, data={"_fake": fake, "pad": "x" * chars})


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


client = object.__new__(api.BaseApiClient)
send = lambda r: api.BaseApiClient._request_once(client, r)  # noqa: E731

# 1) 정산·미확인·비용 상한
with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.jsonl"
    g = vb.install(ledger, cost_cap_usd=0.5, attempt_cap=50, deadline_window_seconds=5400, run_tag="t", model=MODEL)
    check("설치만으로는 마감 기준점이 생기지 않음", g.totals()["deadline_epoch"] is None)
    send(req("count"))
    t = g.totals()
    check("countTokens 0원·시도 1·종류 count_tokens", t["attempts"] == 1 and t["settled_usd"] == 0
          and t["attempts_by_kind"] == {"count_tokens": 1})
    check("첫 시도에서 기준점 = 첫 시도 + 5400초", t["deadline_epoch"] is not None
          and abs(t["deadline_epoch"] - t["first_attempt_epoch"] - 5400) < 1e-6)
    asyncio.run(api.BaseApiClient._async_request_once(client, req()))
    t = g.totals()
    expected = (10_000 * 0.25 + (1_000 + 500) * 1.5) / 1e6
    check(f"generate 정산(thinking 포함 출력) ${expected}", abs(t["settled_usd"] - expected) < 1e-9)
    send(req(fake="nousage"))
    t = g.totals()
    check("사용량 없는 응답 → 예약액 미확인", t["unverified_usd"] > 0.09)
    try:
        send(req(fake="500"))
    except FakeError:
        pass
    t = g.totals()
    check("500 실패 → 미확인 누적·정지 아님", t["failed_calls"] == 2 and not g.halted)
    blocked = None
    for _ in range(6):
        try:
            send(req(fake="nousage"))
        except vb.BudgetHalt as exc:
            blocked = str(exc)
            break
    check(f"비용 상한에서 보내기 전에 거부({blocked})", blocked is not None and "비용 상한" in blocked and g.halted)
    n = len(calls)
    try:
        send(req())
    except vb.BudgetHalt:
        pass
    check("정지 뒤 HTTP 시도 없음", len(calls) == n)
    check("committed ≤ 상한", g.totals()["committed_usd"] <= 0.5)

# 2) 시도 상한 · 정지 이어받기
with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.jsonl"
    g = vb.install(ledger, cost_cap_usd=10, attempt_cap=3, deadline_window_seconds=5400, run_tag="t2", model=MODEL)
    for _ in range(3):
        send(req())
    try:
        send(req())
        check("시도 상한 거부", False)
    except vb.BudgetHalt as exc:
        check(f"시도 상한 거부({exc})", "시도 상한" in str(exc))
    g2 = vb.install(ledger, cost_cap_usd=10, attempt_cap=100, deadline_window_seconds=5400, run_tag="t3", model=MODEL)
    check("정지 기록을 다음 실행이 이어받음", g2.halted)
    n = len(calls)
    try:
        send(req())
    except vb.BudgetHalt:
        pass
    check("이어받은 정지 상태에서 HTTP 시도 없음", len(calls) == n)

# 3) 첫 시도 기준 마감 — 다음 실행(새 가드)이 다시 세지 않는다
with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.jsonl"
    clock = Clock(1_000_000.0)
    g = vb.install(ledger, cost_cap_usd=10, attempt_cap=100, deadline_window_seconds=5400, run_tag="run1", model=MODEL)
    g.clock = clock
    send(req())
    clock.t += 3000
    g2 = vb.install(ledger, cost_cap_usd=10, attempt_cap=100, deadline_window_seconds=5400, run_tag="run2", model=MODEL)
    g2.clock = clock
    send(req())
    t = g2.totals()
    check("둘째 실행도 첫 시도 기준(마감 = 1,005,400)", t["deadline_epoch"] == 1_005_400.0)
    check("기준점 기록은 하나뿐", sum(1 for l in ledger.read_text().splitlines() if '"anchor"' in l) == 1)
    clock.t = 1_005_400.0
    check("마감 시각 도달 → deadline_passed", g2.deadline_passed())
    n = len(calls)
    try:
        send(req("count"))
        check("마감 뒤 countTokens도 거부", False)
    except vb.BudgetHalt as exc:
        check(f"마감 뒤 countTokens도 거부({exc})", "마감" in str(exc) and len(calls) == n)
    g3 = vb.install(ledger, cost_cap_usd=10, attempt_cap=100, deadline_window_seconds=99999, run_tag="run3", model=MODEL)
    check("창을 크게 준 새 실행도 정지·마감을 되돌리지 못함", g3.halted)
    with ledger.open("a") as fh:
        fh.write(json.dumps({"event": "reserve", "id": "dead", "reserve_usd": 0.1}) + "\n")
    check("닫히지 않은 예약 → 미확인", g3.totals()["unverified_usd"] >= 0.1 and g3.totals()["open_reservations"] == 1)

# 4) 인증·권한·결제 오류는 첫 번째에서 정지
for fake, label in (("403", "403"), ("400billing", "400 결제"), ("refresh", "RefreshError")):
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "ledger.jsonl"
        g = vb.install(ledger, cost_cap_usd=10, attempt_cap=100, deadline_window_seconds=5400, run_tag="t4", model=MODEL)
        try:
            send(req(fake=fake))
        except (FakeError, RefreshError):
            pass
        check(f"{label} 한 번 → 정지", g.halted and "인증" in g.halt_reason)

# 5) 연속 실패 5회 정지 · 누적 실패 20회 정지
with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.jsonl"
    g = vb.install(ledger, cost_cap_usd=100, attempt_cap=1000, deadline_window_seconds=5400, run_tag="t5", model=MODEL)
    for _ in range(5):
        try:
            send(req(fake="500"))
        except (FakeError, vb.BudgetHalt):
            pass
    check("연속 실패 5회 → 정지", g.halted and "연속" in g.halt_reason)
with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.jsonl"
    g = vb.install(ledger, cost_cap_usd=100, attempt_cap=1000, deadline_window_seconds=5400, run_tag="t6", model=MODEL)
    for i in range(40):
        try:
            send(req(fake="500" if i % 2 == 0 else None))
        except (FakeError, vb.BudgetHalt):
            pass
        if g.halted:
            break
    check(f"누적 실패 20회 → 정지({g.halt_reason})", g.halted and "누적" in g.halt_reason
          and g.totals()["failed_calls"] == 20)

print("FAILURES", failures)
sys.exit(1 if failures else 0)
