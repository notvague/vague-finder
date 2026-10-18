"""run_live · v71_budget 연결 자체 검사 — 네트워크·자격 증명 없음. 실제 `google.genai.Client`를 만들고 SDK가 HTTP를 보내는
자리(`_request_once`)만 가짜로 바꾼다. 그래서 운영 어댑터(`VertexWorkNameResolver`) → SDK → 예산 장치 → (가짜)전송까지 실제
호출과 같은 길을 지난다. 질의는 합성 문장뿐이다.

    venv/bin/python selftest_live.py   # 종료 코드 0 = 전부 통과
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]

import httpx  # noqa: E402
import google.genai._api_client as api  # noqa: E402
from google.genai import errors  # noqa: E402

SCRIPT: List[Any] = []
SEEN: List[Any] = []
OK_TEXT = json.dumps({"decision": "candidates", "works": [
    {"name": "가상병원", "aliases": [], "media_type": "drama", "evidence": "의사들이 밴드 하는 드라마"}]}, ensure_ascii=False)


def _body(text: str) -> str:
    return json.dumps({"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
                       "usageMetadata": {"promptTokenCount": 400, "candidatesTokenCount": 40, "totalTokenCount": 440},
                       "modelVersion": "gemini-3.1-flash-lite"})


def fake_once(self, http_request, stream=False):
    SEEN.append({"url": str(http_request.url), "timeout": http_request.timeout})
    if ":countTokens" in str(http_request.url):
        return api.HttpResponse({}, [json.dumps({"totalTokens": 400})])
    item = SCRIPT.pop(0) if SCRIPT else OK_TEXT
    if isinstance(item, BaseException):
        raise item
    return api.HttpResponse({}, [_body(item)])


api.BaseApiClient._request_once = fake_once  # v71_budget.install보다 먼저 — 예산 장치가 이 가짜를 감싼다

import v71_budget  # noqa: E402
import run_live  # noqa: E402
from src.retrieval import work_name_resolution as wnr  # noqa: E402
from google.auth.credentials import AnonymousCredentials  # noqa: E402
from google import genai  # noqa: E402

Q = [{"set": "s", "qid": f"s{i}", "query": f"합성 질의 {i} — 의사들이 밴드 하는 드라마에 나온 노래"} for i in range(3)]
failures = 0


def check(name: str, cond: bool) -> None:
    global failures
    print(("ok   " if cond else "FAIL ") + name)
    failures += 0 if cond else 1


def server_error(code: int, status: str) -> errors.APIError:
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": status.lower(), "status": status}})


def fresh(ledger: Path, *, cost=0.25, attempts=60, window=1200.0, tag="t") -> Any:
    v71_budget.GUARD = None
    return v71_budget.install(ledger, cost_cap_usd=cost, attempt_cap=attempts, deadline_window_seconds=window,
                              run_tag=tag, model="gemini-3.1-flash-lite")


def client() -> run_live.RecordingClient:
    return run_live.RecordingClient(genai.Client(vertexai=True, project="selftest", location="global",
                                                 credentials=AnonymousCredentials()))


def rep(ledger: Path, script: List[Any], items=Q, **kw) -> dict:
    SCRIPT[:] = list(script)
    SEEN.clear()
    guard = fresh(ledger, **kw)
    out = run_live.run_rep(items, rep=1, client=client(), guard=guard, timeout_seconds=4.0)
    out["guard"] = guard
    out["totals"] = guard.totals()
    out["seen"] = list(SEEN)
    return out


def replay(entries: dict, query: str) -> Any:
    d = Path(tempfile.mkdtemp())
    meta = {"prompt_version": wnr.PROMPT_VERSION, "prompt_sha16": wnr.versions()["prompt_sha16"], "model": "gemini-3.1-flash-lite"}
    (d / "c.json").write_text(json.dumps({"meta": meta, "entries": entries}, ensure_ascii=False), encoding="utf-8")
    return wnr.CachedWorkNameResolver(d / "c.json", expect_model="gemini-3.1-flash-lite").resolve(query, timeout_seconds=4)


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    # 1. 정상 — 시도 수(countTokens 1 + 생성 3), 원문 보존, 캐시 재생이 실시간 파싱과 같음, 요청별 시간 제한 4초
    a = rep(tmp / "l1.jsonl", [OK_TEXT] * 3)
    check("정상: 장부 시도 4(count 1 + generate 3)", a["totals"]["attempts"] == 4
          and a["totals"]["attempts_by_kind"] == {"count_tokens": 1, "generate": 3})
    check("정상: 캐시 3건 ok·원문 보존", len(a["entries"]) == 3 and all(e["text"] == OK_TEXT for e in a["entries"].values()))
    live_names = [w["name"] for w in a["rows"][0]["works"]]
    check("정상: 재생 결과 = 실시간 파싱", [w.name for w in replay(a["entries"], Q[0]["query"]).works] == live_names == ["가상병원"])
    check("정상: SDK 요청 시간 제한 4초", all(s["timeout"] == 4.0 for s in a["seen"] if ":generate" in s["url"]))
    check("정상: 요청 설정 = 운영 어댑터(temperature 0 · max 512 · json)",
          a["rows"][0]["config"].get("max_output_tokens") == 512 and a["rows"][0]["config"].get("temperature") == 0.0
          and a["rows"][0]["config"].get("response_mime_type") == "application/json")
    check("정상: 정산 비용 = 사용량", abs(a["totals"]["settled_usd"] - 3 * (400 * 0.25 + 40 * 1.5) / 1e6) < 1e-6
          and a["totals"]["unverified_usd"] == 0)
    # 2. 두 번째 반복은 첫 반복 캐시를 읽지 않는다 — 같은 장부로 다시 돌리면 생성 시도가 다시 3회, 요청 지문 같음
    b = rep(tmp / "l1.jsonl", [OK_TEXT] * 3)
    check("반복 2: 생성 시도 3회 더(캐시 재사용 없음)", b["totals"]["attempts_by_kind"]["generate"] == 6)
    check("반복 2: 요청 지문 같음", [r["request_sha16"] for r in a["rows"]] == [r["request_sha16"] for r in b["rows"]])
    bodies = {}
    for line in (tmp / "l1.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "reserve" and row["kind"] == "generate":
            bodies.setdefault(row["body_sha16"], 0)
            bodies[row["body_sha16"]] += 1
    check("반복 2: 장부 본문 지문이 질의마다 2번씩", sorted(bodies.values()) == [2, 2, 2])
    # 3. 시간 초과 — WorkNameRuntime과 같은 문자열, 예약은 미확인으로 남음, 재생도 같은 상태
    c = rep(tmp / "l3.jsonl", [httpx.ReadTimeout("t"), OK_TEXT, OK_TEXT])
    check("시간 초과: 캐시 error:ReadTimeout", c["entries"][wnr.query_key(Q[0]["query"])] == {"status": "failed", "error": "error:ReadTimeout"})
    check("시간 초과: 미확인 비용 = 예약액(0으로 지우지 않음)", c["totals"]["unverified_usd"] > 0.09)
    r = replay(c["entries"], Q[0]["query"])
    check("시간 초과: 재생 상태 failed·error:ReadTimeout", r.status == "failed" and r.error == "error:ReadTimeout")
    # 서비스 런타임이 같은 예외에서 같은 상태를 만드는가(가짜 해석기로 예외만 재현)
    class Boom:
        name = "model:x"

        def resolve(self, q, *, timeout_seconds):
            raise httpx.ReadTimeout("t")
    from src.retrieval.work_mentions import WorkMentionStore
    st = WorkMentionStore.load(ROOT / "artifacts/work_names/work_mentions_v1.json.gz")
    rec, _ = wnr.WorkNameRuntime(Boom(), st).run(
        SimpleAnalysis(Q[0]["query"]), wnr.GateDecision(True, "t"), lambda *a: [])
    check("시간 초과: 서비스 런타임 상태 = failed:error:ReadTimeout", rec.status == "failed:error:ReadTimeout")
    # 4. 형식 오류 — 원문은 ok로 캐시, 재생 때 같은 파서가 failed
    d = rep(tmp / "l4.jsonl", ["이건 JSON이 아님", OK_TEXT, OK_TEXT])
    r = replay(d["entries"], Q[0]["query"])
    check("형식 오류: 원문 보존·재생 failed", d["entries"][wnr.query_key(Q[0]["query"])]["text"] == "이건 JSON이 아님"
          and r.status == "failed" and d["rows"][0]["parsed_status"] == "failed" and r.error == d["rows"][0]["parsed_error"])
    # 5. 권한 오류 — 첫 오류 정지, 남은 질의 not_run(캐시에 없음 → 재생 cache_miss), 같은 장부의 다음 실행은 0회
    e = rep(tmp / "l5.jsonl", [server_error(403, "PERMISSION_DENIED"), OK_TEXT, OK_TEXT])
    check("권한: 정지·남은 2건 not_run", e["stopped"] != "" and [r["status"] for r in e["rows"]] == ["failed", "not_run", "not_run"])
    check("권한: not_run은 캐시에 없음 → 재생 cache_miss", len(e["entries"]) == 1 and replay(e["entries"], Q[1]["query"]).error == "cache_miss")
    f = rep(tmp / "l5.jsonl", [OK_TEXT] * 3)
    check("권한: 같은 장부의 다음 반복은 시도 0", f["totals"]["attempts"] == e["totals"]["attempts"] and all(r["status"] == "not_run" for r in f["rows"]))
    # 6. 연속 504 3회 정지(질의 4개)
    q4 = Q + [{"set": "s", "qid": "s3", "query": "합성 질의 3 — 경연 프로그램 노래"}]
    # (규칙 자체를 보려고 상한 $1 — $0.25에서는 실패 2회의 미확인 예약 때문에 세 번째 시도가 비용 상한에 먼저 막힌다, 아래 9b)
    g = rep(tmp / "l6.jsonl", [server_error(504, "DEADLINE_EXCEEDED")] * 3 + [OK_TEXT], items=q4, cost=1.0)
    check("504×3: 정지·4번째 not_run", "504" in g["guard"].halt_reason and g["rows"][3]["status"] == "not_run")
    # 7. 연속 실패 3회(504 아님)도 정지 — v70 규칙
    h = rep(tmp / "l7.jsonl", [server_error(500, "INTERNAL")] * 3 + [OK_TEXT], items=q4, cost=1.0)
    check("연속 실패 3회 정지", "연속 실패 3" in h["guard"].halt_reason and h["rows"][3]["status"] == "not_run")
    # 8. 시도 상한은 countTokens를 센다 — 상한 3: count 1 + 생성 2, 세 번째 질의는 보내지 않음
    i = rep(tmp / "l8.jsonl", [OK_TEXT] * 3, attempts=3)
    check("시도 상한(countTokens 포함)", i["totals"]["attempts"] == 3 and i["rows"][2]["status"] == "not_run")
    # 9. 비용 상한 — 실패 1회(예약 ≈ $0.099 미확인) 뒤 상한 $0.15에서 다음 예약이 막힘
    j = rep(tmp / "l9.jsonl", [server_error(500, "INTERNAL"), OK_TEXT, OK_TEXT], cost=0.15)
    check("비용 상한: 미확인 예약이 상한에 남아 다음 시도 차단", j["rows"][1]["status"] == "not_run" and "비용 상한" in j["guard"].halt_reason)
    # 9b. 승인 상한 $0.25 그대로: 실패 2회 뒤 세 번째 시도는 비용 상한으로 막힌다(사전 계산 — RUN_INFO §3)
    m = rep(tmp / "l9b.jsonl", [server_error(504, "DEADLINE_EXCEEDED")] * 2 + [OK_TEXT], cost=0.25)
    check("$0.25: 실패 2회 뒤 세 번째 차단(비용 상한)", [r["status"] for r in m["rows"]] == ["failed", "failed", "not_run"]
          and "비용 상한" in m["guard"].halt_reason)
    # 10. 마감 — 첫 시도 + 창 이후 새 시도 없음
    guard = fresh(tmp / "l10.jsonl", window=1200)
    t0 = guard.clock()
    guard.clock = lambda: t0 + 5000
    SCRIPT[:] = [OK_TEXT] * 3
    with open(tmp / "l10.jsonl", "a") as fh:
        fh.write(json.dumps({"event": "anchor", "first_attempt_epoch": t0, "window_seconds": 1200, "deadline_epoch": t0 + 1200}) + "\n")
    k = run_live.run_rep(Q, rep=1, client=client(), guard=guard, timeout_seconds=4.0)
    check("마감 이후 시도 없음", all(r["status"] == "not_run" for r in k["rows"]) and "마감" in guard.halt_reason)
    # 11. SDK 재시도도 HTTP 시도마다 센다(운영은 재시도 없음 — 계수 장치가 재시도를 놓치지 않는지만 확인)
    SCRIPT[:] = [server_error(503, "UNAVAILABLE"), server_error(503, "UNAVAILABLE"), OK_TEXT]
    guard = fresh(tmp / "l11.jsonl", cost=1.0)  # 실패 2회 미확인 예약(≈$0.197) + 세 번째 예약이 $0.25를 넘으므로
    cl = genai.Client(vertexai=True, project="selftest", location="global", credentials=AnonymousCredentials())
    cl.models.generate_content(model="gemini-3.1-flash-lite", contents="x", config={"http_options": {
        "retry_options": {"attempts": 3, "initial_delay": 0.01, "max_delay": 0.02}}})
    check("SDK 재시도 3회 = 장부 시도 3", guard.totals()["attempts"] == 3)
    # 12. 누출 검사 — 프롬프트 틀에 정답 id·v67 이름이 들어가면 잡는다
    mat = {"ids": {"1234567"}, "names": ["가상드라마"]}
    clean = run_live.check_no_answer_leak(Q, wnr.build_resolver_prompt, mat)
    dirty = run_live.check_no_answer_leak(Q, lambda q: wnr.build_resolver_prompt(q) + " 1234567 가상드라마", mat)
    check("누출 검사: 깨끗한 입력 통과·주입 검출", not clean["problems"] and len(dirty["problems"]) == 6)
    # 13. 승인 변수 없음·출력 있음·설정 오류는 시도 전에 멈춘다(하위 프로세스)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    p = subprocess.run([sys.executable, str(HERE / "run_live.py"), "--rep", "1", "--ledger", str(tmp / "x.jsonl")],
                       capture_output=True, text=True, env=env)
    check("승인 변수 없으면 시작 안 함", p.returncode != 0 and "V71_LIVE_APPROVED" in p.stderr and not (tmp / "x.jsonl").exists())
    p = subprocess.run([sys.executable, str(HERE / "run_live.py"), "--rep", "1", "--ledger", str(HERE / "ledger.jsonl")],
                       capture_output=True, text=True, env={**env, "V71_LIVE_APPROVED": "1"})
    check("장부가 결과 폴더 안이면 시작 안 함", p.returncode != 0 and not (HERE / "ledger.jsonl").exists())
    if not (HERE / "live").exists():
        p = subprocess.run([sys.executable, str(HERE / "run_live.py"), "--rep", "1", "--ledger", str(tmp / "y.jsonl")],
                           capture_output=True, text=True, env={**env, "V71_LIVE_APPROVED": "1"})
        check("GCP_PROJECT_ID 없으면 시도 전 설정 오류", p.returncode != 0 and "GCP_PROJECT_ID" in p.stderr
              and not (tmp / "y.jsonl").exists() and not (HERE / "live").exists())
    print(f"{'전부 통과' if not failures else f'실패 {failures}'}")
    return 1 if failures else 0


class SimpleAnalysis:
    def __init__(self, q: str) -> None:
        self.original_query = q
        self.context_clues: list = []


if __name__ == "__main__":
    sys.exit(main())
