"""v71 실제 작품명 해석 — 판정(gate_v2) 통과 dev 22질의를 **한 반복** 해석한다. 반복마다 따로 실행하고 같은 장부를 쓴다.

- 해석기는 운영 어댑터 `VertexWorkNameResolver` 그대로다(요청 설정·`max_output_tokens`·요청별 `http_options` 시간 제한이
  서비스와 같다). 클라이언트는 운영과 같은 `make_genai_client(purpose="retrieval")`이고, 응답 원문·사용량을 남기려고
  얇게 감싼다(요청 인자는 바꾸지 않는다). 시간 제한은 운영 기본 4초(`resolver_timeout_seconds()`).
- 예외는 `WorkNameRuntime.run`과 같은 문자열로 캐시에 남긴다(`timeout` · `error:<형식>`) — 오프라인 재생이 서비스가 같은
  응답을 받았을 때의 상태를 그대로 만든다. 형식 오류 응답은 원문 그대로 캐시에 두고 재생 때 같은 파서가 거른다.
- 예산: `v71_budget`(v69 장치 사본)이 SDK가 HTTP 요청을 보내는 자리(`_request_once`)마다 예약·정산한다. countTokens·재시도도
  시도로 센다. 반복 둘이 **같은 장부**를 쓰므로 한도·마감(첫 시도 + 20분)·정지가 공유된다.
- 해석기 입력은 질의 원문뿐이다. 정답(relevant_ids)·정답 작품명은 읽지 않는다 — 누출 검사(`check_no_answer_leak`)만 따로 읽는다.
- 이 실행은 캐시를 **읽지 않는다**(두 번째 반복이 첫 반복 응답을 재사용하지 않는다). 출력 파일이 있으면 시작하지 않는다.

    V71_LIVE_APPROVED=1 venv/bin/python with_env.py GCP_PROJECT_ID,GCP_LOCATION,GOOGLE_APPLICATION_CREDENTIALS -- \\
        venv/bin/python run_live.py --rep 1 --ledger <결과 폴더 밖>/ledger.jsonl
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in list(sys.path):
    if p in ("", ".", str(HERE)):
        sys.path.remove(p)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(HERE))

MODEL = "gemini-3.1-flash-lite"
COST_CAP_USD = 0.25
ATTEMPT_CAP = 60
WINDOW_SECONDS = 1200  # 첫 외부 시도부터 20분
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")
# v70 offline_v3/gate_abstain의 판정 통과 22질의(같은 순서) — 다르면 자료나 판정이 바뀐 것이다
EXPECTED_QIDS = ("q105", "m103", "c602", "c603", "c607", "c701", "c705", "c707", "n010", "n031", "n033", "n038", "n043",
                 "n045", "n096", "t024", "t036", "t037", "t041", "t043", "t046", "t098")
EXPECTED_VERSIONS = {"gate": "gate_v2", "validation": "validate_v3", "prompt": "resolver_prompt_v2",
                     "prompt_sha16": "31494c3f69f77481", "store_store_sha256_16": "bfd522ebe9da7590"}
V67_ORACLE = Path("/private/tmp/vague-finder-v67-candidate-recall/experiments/reranking/"
                  "results_v67_candidate_recall_diagnosis/oracle_work_name.py")
REMOVE_ENV = ("CLAUDE_EFFORT", "CLAUDE_API", "GEMINI_API_KEY", "GOOGLE_API_KEY", "WORK_NAME_RESOLUTION_ENABLED",
              "WORK_NAME_RESOLVER", "WORK_NAME_AUX_WEIGHT", "WORK_NAME_RESOLVER_TIMEOUT_SECONDS", "VF_CLAUDE_EFFORT")
FIXED_ENV = {"GEMINI_MODEL_NAME": MODEL, "GEMINI_RETRIEVAL_BACKEND": "vertex"}


def load_inputs() -> List[Dict[str, Any]]:
    """판정 통과 질의 — (set, qid, 질의 원문, 분석). relevant_ids는 담지 않는다."""
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.evaluate_search_accuracy import read_queries
    from src.retrieval.work_name_resolution import gate_indirect_work_reference

    items = []
    for s in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{s}_dev_vertex.json")
        for row in read_queries(ROOT / f"experiments/reranking/eval_queries_{s}.csv", "dev"):
            qid, query = row["query_id"].strip(), row["query"].strip()
            analysis = cache.get(qid, query)
            if gate_indirect_work_reference(analysis).triggered:
                items.append({"set": s, "qid": qid, "query": query})
    return items


def input_fingerprint(items: List[Dict[str, Any]]) -> str:
    from src.retrieval.work_name_resolution import query_key

    return hashlib.sha256(json.dumps([[i["qid"], query_key(i["query"])] for i in items]).encode()).hexdigest()[:16]


def _answer_material() -> Dict[str, Any]:
    from src.retrieval.evaluate_search_accuracy import parse_relevant_ids

    ids: set = set()
    for s in SETS:
        with open(ROOT / f"experiments/reranking/eval_queries_{s}.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids |= parse_relevant_ids(row.get("relevant_ids") or "")
    names: List[str] = []
    if V67_ORACLE.exists():
        for node in ast.parse(V67_ORACLE.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "WORK_NAMES" for t in node.targets):
                names = list(ast.literal_eval(node.value).values())
    return {"ids": ids, "names": names}


def check_no_answer_leak(items: List[Dict[str, Any]], prompt_fn: Any, material: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """해석기 입력(프롬프트)에 정답 곡 id·v67 작품명이 질의 원문 밖에서 들어가지 않았는가. 프롬프트 = 고정 틀 + 질의."""
    material = material or _answer_material()
    problems = []
    for item in items:
        prompt = prompt_fn(item["query"])
        outside = prompt.replace(json.dumps(item["query"], ensure_ascii=False), "")
        if prompt == outside:
            problems.append((item["qid"], "query_not_embedded_as_json"))
        for rid in material["ids"]:
            if len(rid) >= 5 and rid in outside:
                problems.append((item["qid"], "relevant_id_in_prompt"))
        for name in material["names"]:
            if name and name in outside:
                problems.append((item["qid"], "v67_name_in_prompt_template"))
    return {"checked": len(items), "relevant_ids_known": len(material["ids"]), "v67_names_known": len(material["names"]),
            "problems": problems}


class RecordingModels:
    """`client.models` 대리 — 인자는 그대로 넘기고 마지막 요청 인자·응답을 남긴다."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_kwargs: Dict[str, Any] = {}
        self.last_response: Any = None

    def generate_content(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        self.last_response = None
        self.last_response = self._inner.generate_content(**kwargs)
        return self.last_response

    def count_tokens(self, **kwargs: Any) -> Any:
        return self._inner.count_tokens(**kwargs)


class RecordingClient:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.models = RecordingModels(inner.models)


def request_fingerprint(kwargs: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def _response_fields(response: Any) -> Dict[str, Any]:
    if response is None:
        return {}
    usage = getattr(response, "usage_metadata", None)
    cands = getattr(response, "candidates", None) or []
    finish = str(getattr(cands[0], "finish_reason", "") or "") if cands else ""
    return {
        "text": getattr(response, "text", None) or "",
        "finish_reason": finish,
        "model_version": getattr(response, "model_version", None),
        "usage": {"prompt": getattr(usage, "prompt_token_count", None),
                  "candidates": getattr(usage, "candidates_token_count", None),
                  "thoughts": getattr(usage, "thoughts_token_count", None),
                  "total": getattr(usage, "total_token_count", None)},
    }


def run_rep(items: List[Dict[str, Any]], *, rep: int, client: RecordingClient, guard: Any, timeout_seconds: float,
            model: str = MODEL, preflight: bool = True) -> Dict[str, Any]:
    """한 반복. 반환: 캐시 entries · 원기록 rows · 요약. 정지되면 남은 질의는 `not_run`(캐시에 넣지 않음 → 재생 때 cache_miss)."""
    from src.retrieval import work_name_resolution as wnr
    from v71_budget import BudgetHalt

    resolver = wnr.VertexWorkNameResolver(model=model, client_factory=lambda: client)
    entries: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    preflight_info: Dict[str, Any] = {"done": False}
    stopped = ""
    if preflight:
        try:
            counted = client.models.count_tokens(model=model, contents=wnr.build_resolver_prompt(items[0]["query"]))
            preflight_info = {"done": True, "total_tokens": getattr(counted, "total_tokens", None)}
        except BudgetHalt as exc:
            stopped = f"halt_before_preflight: {exc}"
        except Exception as exc:  # noqa: BLE001 - 무료 확인 호출 실패면 생성 호출을 하지 않는다
            preflight_info = {"done": False, "error": type(exc).__name__, "code": getattr(exc, "code", None)}
            stopped = f"preflight_failed: {type(exc).__name__}"
    for item in items:
        key = wnr.query_key(item["query"])
        row: Dict[str, Any] = {"rep": rep, "set": item["set"], "qid": item["qid"], "query_key16": key[:16]}
        if stopped or guard.halted:
            stopped = stopped or f"halt: {guard.halt_reason}"
            row["status"] = "not_run"
            rows.append(row)
            continue
        started = time.perf_counter()
        error = None
        output = None
        try:
            output = resolver.resolve(item["query"], timeout_seconds=timeout_seconds)
        except BudgetHalt as exc:
            # 보내기 전에 막혔다 — 호출 없음, 캐시에 넣지 않는다
            stopped = f"halt: {exc}"
            row["status"] = "not_run"
            rows.append(row)
            continue
        except TimeoutError as exc:
            error = ("timeout", exc)
        except Exception as exc:  # noqa: BLE001 - WorkNameRuntime.run과 같은 사상
            error = (f"error:{type(exc).__name__}", exc)
        latency = round((time.perf_counter() - started) * 1000, 1)
        row["request_sha16"] = request_fingerprint(client.models.last_kwargs) if client.models.last_kwargs else ""
        row["config"] = {k: v for k, v in (client.models.last_kwargs.get("config") or {}).items()}
        row["latency_ms"] = latency
        row.update(_response_fields(client.models.last_response))
        if error is not None:
            exc = error[1]
            row.update(status="failed", runtime_error=error[0], error_type=type(exc).__name__,
                       error_code=getattr(exc, "code", None), error_status=getattr(exc, "status", None))
            entries[key] = {"status": "failed", "error": error[0]}
        else:
            row.update(status="ok", parsed_status=output.status, parsed_error=output.error, raw_sha16=output.raw_sha16,
                       truncated=output.truncated,
                       works=[{"name": w.name, "aliases": list(w.aliases), "media_type": w.media_type, "evidence": w.evidence}
                              for w in output.works])
            entries[key] = {"status": "ok", "text": row.get("text", ""), "usage": row.get("usage", {})}
        rows.append(row)
    if not stopped and guard.halted:
        stopped = f"halt: {guard.halt_reason}"
    return {"entries": entries, "rows": rows, "stopped": stopped, "preflight": preflight_info}


def _transport_counter() -> Dict[str, int]:
    """SDK 아래 실제 전송(httpx·requests)을 호스트 종류별로 센다 — 검증용(한도는 v71_budget이 SDK 시도로 건다)."""
    import httpx
    import requests

    counts = {"vertex": 0, "oauth": 0, "other": 0}

    def kind(host: str) -> str:
        host = host or ""
        if host.endswith("aiplatform.googleapis.com"):
            return "vertex"
        if host.startswith("oauth2.") or host.endswith("accounts.google.com") or "iamcredentials" in host:
            return "oauth"
        return "other"

    orig_httpx = httpx.Client.send
    orig_requests = requests.Session.send

    def httpx_send(self, request, *a, **k):  # type: ignore[no-untyped-def]
        counts[kind(request.url.host)] += 1
        return orig_httpx(self, request, *a, **k)

    def requests_send(self, request, *a, **k):  # type: ignore[no-untyped-def]
        from urllib.parse import urlparse

        counts[kind(urlparse(request.url).hostname or "")] += 1
        return orig_requests(self, request, *a, **k)

    httpx.Client.send = httpx_send  # type: ignore[assignment]
    requests.Session.send = requests_send  # type: ignore[assignment]
    return counts


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, choices=(1, 2), required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    args = ap.parse_args()
    out_dir = HERE / f"live/rep{args.rep}"
    cache_path = HERE / f"resolution_cache/rep{args.rep}.json"
    if os.environ.get("V71_LIVE_APPROVED") != "1":
        raise SystemExit("실제 호출은 V71_LIVE_APPROVED=1(사용자 승인)이 있어야 한다")
    if out_dir.exists() or cache_path.exists():
        raise SystemExit(f"{out_dir} 또는 {cache_path}가 이미 있다 — 덮어쓰지 않는다")
    if HERE in args.ledger.resolve().parents or args.ledger.resolve().parent == HERE:
        raise SystemExit("장부는 결과 폴더 밖에 둔다")
    for key in REMOVE_ENV:
        os.environ.pop(key, None)
    os.environ.update(FIXED_ENV)
    if (os.environ.get("GCP_LOCATION") or "global") != "global":
        raise SystemExit("GCP_LOCATION이 global이 아니다 — 가격 확인한 경로와 다르다")

    from src.common import gemini_client as gc
    from src.retrieval import work_name_resolution as wnr
    from src.retrieval.work_mentions import WorkMentionStore
    import v71_budget

    store = WorkMentionStore.load(ROOT / "artifacts/work_names/work_mentions_v1.json.gz")
    versions = wnr.versions(store)
    for k, v in EXPECTED_VERSIONS.items():
        if versions.get(k) != v:
            raise SystemExit(f"버전 불일치 {k}: {versions.get(k)} != {v}")
    timeout_seconds = wnr.resolver_timeout_seconds()
    if timeout_seconds != 4.0 or wnr.MAX_OUTPUT_TOKENS != 512:
        raise SystemExit("운영 기본 시간 제한·출력 상한이 v70과 다르다")
    items = load_inputs()
    if tuple(i["qid"] for i in items) != EXPECTED_QIDS:
        raise SystemExit(f"판정 통과 질의가 v70과 다르다: {[i['qid'] for i in items]}")
    leak = check_no_answer_leak(items, wnr.build_resolver_prompt)
    if leak["problems"] or leak["v67_names_known"] == 0:
        raise SystemExit(f"입력 누출 검사 실패: {leak}")
    route = gc.require_vertex(purpose="retrieval")  # 설정 오류면 시도 전에 멈춘다
    gc.vertex_credentials()                          # 파일 읽기뿐(토큰 요청은 첫 시도에서)

    transport = _transport_counter()
    guard = v71_budget.install(args.ledger, cost_cap_usd=COST_CAP_USD, attempt_cap=ATTEMPT_CAP,
                               deadline_window_seconds=WINDOW_SECONDS, run_tag=f"v71_rep{args.rep}", model=MODEL)
    before = guard.totals()
    client = RecordingClient(gc.make_genai_client(purpose="retrieval"))
    started = time.time()
    result = run_rep(items, rep=args.rep, client=client, guard=guard, timeout_seconds=timeout_seconds)
    elapsed = round(time.time() - started, 1)
    after = guard.totals()

    out_dir.mkdir(parents=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"prompt_version": wnr.PROMPT_VERSION, "prompt_sha16": versions["prompt_sha16"], "model": MODEL,
            "gate_version": wnr.GATE_VERSION, "rep": args.rep, "created_at": datetime.now(timezone.utc).isoformat(),
            "queries_triggered": len(items), "entries": len(result["entries"]), "stopped": result["stopped"]}
    cache_path.write_text(json.dumps({"meta": meta, "entries": result["entries"]}, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    with open(out_dir / "raw.jsonl", "w", encoding="utf-8") as f:
        for row in result["rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    status_counts: Dict[str, int] = {}
    for row in result["rows"]:
        label = row["status"] if row["status"] != "ok" else f"ok:{row.get('parsed_status')}"
        status_counts[label] = status_counts.get(label, 0) + 1
    import google.genai

    conditions = {
        "rep": args.rep, "model": MODEL, "route": route, "sdk": f"google-genai {google.genai.__version__}",
        "timeout_seconds": timeout_seconds, "max_output_tokens": wnr.MAX_OUTPUT_TOKENS, "versions": versions,
        "store": store.fingerprint(), "input_fingerprint16": input_fingerprint(items), "qids": [i["qid"] for i in items],
        "leak_check": {k: v for k, v in leak.items() if k != "problems"} | {"problems": len(leak["problems"])},
        "preflight": result["preflight"], "stopped": result["stopped"], "elapsed_seconds": elapsed,
        "status_counts": status_counts, "guard_local": guard.local, "ledger_before": before, "ledger_after": after,
        "transport_requests": transport, "caps": {"cost_usd": COST_CAP_USD, "attempts": ATTEMPT_CAP,
                                                  "window_seconds": WINDOW_SECONDS},
        "cache_sha16": hashlib.sha256(cache_path.read_bytes()).hexdigest()[:16],
        "src_head": _git("rev-parse", "HEAD").strip(),
        "src_diff_sha16": hashlib.sha256(_git("diff", "HEAD", "--", "src").encode()).hexdigest()[:16],
    }
    (out_dir / "run_conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=1, default=str) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({"rep": args.rep, "status_counts": status_counts, "stopped": result["stopped"],
                      "attempts": after["attempts"], "committed_usd": after["committed_usd"],
                      "settled_usd": after["settled_usd"], "unverified_usd": after["unverified_usd"],
                      "transport": transport, "elapsed_s": elapsed}, ensure_ascii=False))
    return 0 if not result["stopped"] else 3


if __name__ == "__main__":
    sys.exit(main())
