"""작품명 해석 캐시 만들기 — 판정(gate_v1)을 통과한 질의에만 해석기를 1회씩 부른다. **v70에서는 가짜 클라이언트로만 실행했다.**

실제 호출(`--live`)은 다음을 모두 요구한다(하나라도 없으면 시작하지 않는다):
- 환경 `V70_LIVE_APPROVED=1`(사용자 승인 뒤에만 사람이 넣는다) · `--max-calls` · `--max-usd` · 결과 폴더 **밖**의 `--ledger` 경로
- 장부(JSONL)에 시도마다 예약액(입력 상한 + 출력 상한 `MAX_OUTPUT_TOKENS`)과 결과를 적는다. 실패한 시도의 예약액은 미확인으로 남긴다.
- 정지 규칙: 인증·권한·결제(401·403·PERMISSION_DENIED 등) 첫 오류 · 연속 실패 3회 · 504/DEADLINE 3회 · 상한 도달. 정지하면 다시 시작하지 않는다.
- SDK 재시도 없음(google-genai는 retry_options가 없으면 1회). 질의 원문은 캐시에 쓰지 않는다(키 = sha256).

가짜 실행(`--dry-run`): 네트워크·자격 증명 없이 가짜 클라이언트가 abstain을 돌려준다 — 배선·장부·정지 규칙 확인용.

    venv/bin/python build_resolution_cache.py --dry-run --out <캐시.json> --ledger <장부.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in list(sys.path):
    if p in ("", ".", str(HERE)):
        sys.path.remove(p)
sys.path.insert(0, str(ROOT))

PRICE_INPUT_PER_TOKEN = 0.25e-6   # gemini-3.1-flash-lite Standard Global — v69에서 2026-10-18 00:20 KST에 공식 표로 확인. 실행 전 다시 확인한다
PRICE_OUTPUT_PER_TOKEN = 1.50e-6
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")
AUTH_MARKERS = ("401", "403", "PERMISSION_DENIED", "UNAUTHENTICATED", "BILLING", "billing")
DEADLINE_MARKERS = ("504", "DEADLINE_EXCEEDED")


class StopRun(RuntimeError):
    pass


def reserve_usd(prompt: str) -> float:
    """입력 토큰 상한을 글자 수로 잡는다(한글·영문 모두 글자당 1토큰 이하) + 출력 상한."""
    from src.retrieval.work_name_resolution import MAX_OUTPUT_TOKENS

    return len(prompt) * PRICE_INPUT_PER_TOKEN + MAX_OUTPUT_TOKENS * PRICE_OUTPUT_PER_TOKEN


class Ledger:
    def __init__(self, path: Path, *, max_calls: int, max_usd: float) -> None:
        self.path = path
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.rows: List[Dict[str, Any]] = []
        if path.exists():
            self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(r.get("kind") == "halt" for r in self.rows):
            raise StopRun("장부에 정지 기록이 있다 — 다시 시작하지 않는다")

    def _write(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def committed_usd(self) -> float:
        return sum(float(r.get("settled_usd", r.get("reserved_usd", 0.0))) for r in self.rows if r.get("kind") == "attempt")

    def attempts(self) -> int:
        return sum(1 for r in self.rows if r.get("kind") == "attempt")

    def check(self, reserve: float) -> None:
        if self.attempts() + 1 > self.max_calls:
            self.halt("max_calls")
        if self.committed_usd() + reserve > self.max_usd:
            self.halt("max_usd")

    def record(self, **row: Any) -> None:
        self._write({"kind": "attempt", "ts": datetime.now(timezone.utc).isoformat(), **row})
        fails = []
        for r in reversed([r for r in self.rows if r.get("kind") == "attempt"]):
            if r.get("ok"):
                break
            fails.append(r)
        if fails and any(m in str(fails[0].get("error", "")) for m in AUTH_MARKERS):
            self.halt("auth_or_billing_error")
        if sum(1 for r in fails if any(m in str(r.get("error", "")) for m in DEADLINE_MARKERS)) >= 3:
            self.halt("deadline_504x3")
        if len(fails) >= 3:
            self.halt("consecutive_failures_3")

    def halt(self, reason: str) -> None:
        self._write({"kind": "halt", "reason": reason, "ts": datetime.now(timezone.utc).isoformat()})
        raise StopRun(reason)


def triggered_queries() -> List[Dict[str, str]]:
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.evaluate_search_accuracy import read_queries
    from src.retrieval.work_name_resolution import gate_indirect_work_reference

    out = []
    for s in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{s}_dev_vertex.json")
        for row in read_queries(ROOT / f"experiments/reranking/eval_queries_{s}.csv", "dev"):
            qid, query = row["query_id"].strip(), row["query"].strip()
            if gate_indirect_work_reference(cache.get(qid, query)).triggered:
                out.append({"qid": qid, "query": query})
    return out


def _fake_factory(script: Optional[List[Any]] = None) -> Callable[[int], Any]:
    """가짜 클라이언트: script 항목이 예외면 던지고, 문자열이면 그 텍스트를 돌려준다. 없으면 abstain."""
    queue = list(script or [])

    class _Models:
        def generate_content(self, *, model, contents, config):
            item = queue.pop(0) if queue else json.dumps({"decision": "abstain", "works": []})
            if isinstance(item, BaseException):
                raise item
            return SimpleNamespace(text=item, usage_metadata=SimpleNamespace(
                prompt_token_count=len(contents) // 3, candidates_token_count=20, thoughts_token_count=None))

    return lambda timeout_ms: SimpleNamespace(models=_Models())


def build(queries: List[Dict[str, str]], *, out: Path, ledger: Ledger, client_factory: Callable[[int], Any],
          model: str, timeout_seconds: float = 4.0) -> Dict[str, Any]:
    from src.retrieval import work_name_resolution as wnr

    entries: Dict[str, Dict[str, Any]] = {}
    stopped = ""
    # 클라이언트는 한 번만 만든다(연결 재사용, 서비스 어댑터 `VertexWorkNameResolver`와 같다 — 리뷰)
    client = client_factory(int(timeout_seconds * 1000))
    for item in queries:
        prompt = wnr.build_resolver_prompt(item["query"])
        reserve = reserve_usd(prompt)
        try:
            ledger.check(reserve)
        except StopRun as exc:
            stopped = str(exc)
            break
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model, contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.0, "top_p": 1.0,
                        "max_output_tokens": wnr.MAX_OUTPUT_TOKENS},
            )
        except Exception as exc:  # noqa: BLE001 - 장부에 남기고 정지 규칙을 본다
            entries[wnr.query_key(item["query"])] = {"status": "failed", "error": type(exc).__name__}
            try:
                ledger.record(qid=item["qid"], ok=False, error=f"{type(exc).__name__}: {exc}"[:300],
                              reserved_usd=round(reserve, 6), latency_ms=round((time.perf_counter() - started) * 1000, 1))
            except StopRun as stop:
                stopped = str(stop)
                break
            continue
        usage = getattr(response, "usage_metadata", None)
        pin = int(getattr(usage, "prompt_token_count", 0) or 0)
        pout = int(getattr(usage, "candidates_token_count", 0) or 0) + int(getattr(usage, "thoughts_token_count", 0) or 0)
        settled = pin * PRICE_INPUT_PER_TOKEN + pout * PRICE_OUTPUT_PER_TOKEN
        text = getattr(response, "text", None) or ""
        entries[wnr.query_key(item["query"])] = {"status": "ok", "text": text,
                                                 "usage": {"input_tokens": pin, "output_tokens_incl_thinking": pout}}
        ledger.record(qid=item["qid"], ok=True, reserved_usd=round(reserve, 6), settled_usd=round(settled, 6),
                      latency_ms=round((time.perf_counter() - started) * 1000, 1))
    meta = {"prompt_version": wnr.PROMPT_VERSION, "prompt_sha16": wnr.versions()["prompt_sha16"], "model": model,
            "gate_version": wnr.GATE_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "queries_triggered": len(queries), "entries": len(entries), "stopped": stopped}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "entries": entries}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=0.0)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out}가 이미 있다 — 덮어쓰지 않는다")
    model = "gemini-3.1-flash-lite"
    if args.live:
        if os.environ.get("V70_LIVE_APPROVED") != "1":
            raise SystemExit("실제 호출은 V70_LIVE_APPROVED=1(사용자 승인)이 있어야 한다")
        if args.max_calls < 1 or args.max_usd <= 0:
            raise SystemExit("--max-calls · --max-usd가 필요하다")
        if HERE in args.ledger.resolve().parents:
            raise SystemExit("장부는 결과 폴더 밖에 둔다(Git 제외 · 덮어쓰기 방지)")
        from src.common.gemini_client import make_genai_client

        factory = lambda timeout_ms: make_genai_client(http_options={"timeout": timeout_ms}, purpose="retrieval")  # noqa: E731
        max_calls, max_usd = args.max_calls, args.max_usd
    else:
        factory, max_calls, max_usd = _fake_factory(), 1000, 1.0
    ledger = Ledger(args.ledger, max_calls=max_calls, max_usd=max_usd)
    meta = build(triggered_queries(), out=args.out, ledger=ledger, client_factory=factory, model=model)
    print(json.dumps({k: meta[k] for k in ("queries_triggered", "entries", "stopped")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
