"""v65_budget 자체 검사 — 네트워크 없음. 가짜 `_request_once`·`_async_request_once`로 예약·정산·미확인·한도·정지를 확인한다."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import google.genai._api_client as api  # noqa: E402

calls = []


def fake_once(self, http_request, stream=False):
    calls.append(http_request.url)
    behaviour = http_request.data.get("_fake")
    if behaviour == "403":
        raise SimpleNamespace_error(403)
    if behaviour == "500":
        raise SimpleNamespace_error(500)
    usage = {"promptTokenCount": 10_000, "candidatesTokenCount": 1_000, "thoughtsTokenCount": 0, "totalTokenCount": 11_000}
    if behaviour == "nousage":
        usage = {}
    return SimpleNamespace(response_stream=[json.dumps({"usageMetadata": usage, "modelVersion": "gemini-3.1-flash-lite"})])


async def fake_async_once(self, http_request, stream=False):
    return fake_once(self, http_request, stream)


class SimpleNamespace_error(Exception):  # noqa: N801
    def __init__(self, code):
        super().__init__(f"fake {code}")
        self.code = code
        self.status = "FAKE"
        self.message = "permission denied" if code == 403 else "server"


api.BaseApiClient._request_once = fake_once
api.BaseApiClient._async_request_once = fake_async_once

import v65_budget as vb  # noqa: E402

failures = 0


def check(name, cond):
    global failures
    print(("ok   " if cond else "FAIL ") + name)
    failures += 0 if cond else 1


def req(kind="generate", fake=None, chars=1000):
    url = "https://x/models/gemini-3.1-flash-lite:" + ("countTokens" if kind == "count" else "generateContent")
    return SimpleNamespace(url=url, data={"_fake": fake, "pad": "x" * chars})


with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.jsonl"
    g = vb.install(ledger, cost_cap_usd=0.5, attempt_cap=8, deadline_epoch=time.time() + 3600, run_tag="t", model="gemini-3.1-flash-lite")
    client = object.__new__(api.BaseApiClient)
    api.BaseApiClient._request_once(client, req("count"))
    t = g.totals()
    check("countTokens 0원·시도 1", t["attempts"] == 1 and t["settled_usd"] == 0)
    asyncio.run(api.BaseApiClient._async_request_once(client, req()))
    t = g.totals()
    expected = (10_000 * 0.25 + 1_000 * 1.5) / 1e6
    check(f"generate 정산 ${expected}", abs(t["settled_usd"] - expected) < 1e-9)
    api.BaseApiClient._request_once(client, req(fake="nousage"))
    t = g.totals()
    check("사용량 없는 응답 → 예약액 미확인", t["unverified_usd"] > 0.09)
    try:
        api.BaseApiClient._request_once(client, req(fake="500"))
    except SimpleNamespace_error:
        pass
    t = g.totals()
    check("500 실패 → 미확인 누적·정지 아님", t["failed_calls"] == 2 and not g.halted)
    # 비용 상한 0.5: 확정+미확인 ≈ 0.2 → 예약 0.098씩 → 몇 번 뒤 거부
    blocked = None
    for i in range(6):
        try:
            api.BaseApiClient._request_once(client, req(fake="nousage"))
        except vb.BudgetHalt as exc:
            blocked = str(exc)
            break
    check(f"비용 상한에서 보내기 전에 거부({blocked})", blocked is not None and "비용 상한" in blocked and g.halted)
    n = len(calls)
    try:
        api.BaseApiClient._request_once(client, req())
    except vb.BudgetHalt:
        pass
    check("정지 뒤 HTTP 시도 없음", len(calls) == n)
    t = g.totals()
    check("committed ≤ 상한", t["committed_usd"] <= 0.5)

with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.jsonl"
    g = vb.install(ledger, cost_cap_usd=10, attempt_cap=3, deadline_epoch=time.time() + 3600, run_tag="t2", model="gemini-3.1-flash-lite")
    client = object.__new__(api.BaseApiClient)
    for _ in range(3):
        api.BaseApiClient._request_once(client, req())
    try:
        api.BaseApiClient._request_once(client, req())
        check("시도 상한 거부", False)
    except vb.BudgetHalt as exc:
        check(f"시도 상한 거부({exc})", "시도 상한" in str(exc))
    # 같은 장부로 새 프로세스(가드)를 만들어도 정지 상태를 이어받는다
    g2 = vb.install(ledger, cost_cap_usd=10, attempt_cap=100, deadline_epoch=time.time() + 3600, run_tag="t3", model="gemini-3.1-flash-lite")
    check("정지 기록을 다음 실행이 이어받음", g2.halted)

with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.jsonl"
    g = vb.install(ledger, cost_cap_usd=10, attempt_cap=100, deadline_epoch=time.time() + 3600, run_tag="t4", model="gemini-3.1-flash-lite")
    client = object.__new__(api.BaseApiClient)
    try:
        api.BaseApiClient._request_once(client, req(fake="403"))
    except SimpleNamespace_error:
        pass
    check("403 한 번 → 정지", g.halted and "인증" in g.halt_reason)

with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.jsonl"
    g = vb.install(ledger, cost_cap_usd=10, attempt_cap=100, deadline_epoch=time.time() - 1, run_tag="t5", model="gemini-3.1-flash-lite")
    client = object.__new__(api.BaseApiClient)
    try:
        api.BaseApiClient._request_once(client, req())
        check("마감 뒤 거부", False)
    except vb.BudgetHalt as exc:
        check(f"마감 뒤 거부({exc})", "마감" in str(exc))
    # 열린 예약(프로세스가 죽은 경우)은 미확인으로 센다
    with ledger.open("a") as fh:
        fh.write(json.dumps({"event": "reserve", "id": "dead", "reserve_usd": 0.1}) + "\n")
    check("닫히지 않은 예약 → 미확인", g.totals()["unverified_usd"] >= 0.1)

print("FAILURES", failures)
sys.exit(1 if failures else 0)
