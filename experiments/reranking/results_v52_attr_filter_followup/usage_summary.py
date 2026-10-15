"""usage_*.jsonl(usage_tap.py 기록)을 모아 호출 수·실패·재시도·토큰·지연·추정 비용을 낸다. 모델 호출 없음.

    venv/bin/python <이 파일> [usage_*.jsonl ...]     # 인자가 없으면 이 폴더의 usage_*.jsonl 전부

단가(USD / 100만 토큰)는 Google 공식 Gemini API 가격표(https://ai.google.dev/gemini-api/docs/pricing, 2026-10-15 확인)의
표준 등급이다 — gemini-3.1-flash-lite 입력 0.25 · 출력 1.50(thinking 포함) · 캐시 입력 0.025. Vertex AI 가격 페이지는 스크립트로
읽히지 않아(JS 렌더링) 같은 값인지 눈으로 재확인해야 한다. 캐시 할인은 `cached_tokens`가 있을 때만 적용하고, 무료 countTokens는 비용 0.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PRICE = {  # USD per 1M tokens: (input, output incl. thinking, cached input)
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
}


def pct(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def cost_of(row: dict) -> float:
    price = PRICE.get(str(row.get("model", "")).split("/")[-1])
    if price is None or row.get("kind") == "count_tokens" or "error" in row:
        return 0.0
    cached = row.get("cached_tokens") or 0
    prompt = (row.get("prompt_tokens") or 0) - cached
    out = (row.get("candidates_tokens") or 0) + (row.get("thoughts_tokens") or 0)
    return (prompt * price[0] + cached * price[2] + out * price[1]) / 1e6


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gen = [r for r in rows if r["kind"] != "count_tokens"]
    ok = [r for r in gen if "error" not in r]
    return {
        "file": path.name,
        "calls": len(gen),
        "probe_calls": sum(1 for r in rows if r["kind"] == "count_tokens"),
        "errors": Counter(f"{r.get('error')}:{r.get('error_code', '')}" for r in gen if "error" in r),
        "finish": Counter(str(r.get("finish_reason")) for r in ok),
        "model_versions": Counter(str(r.get("model_version")) for r in ok),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in ok),
        "output_tokens": sum((r.get("candidates_tokens") or 0) for r in ok),
        "thoughts_tokens": sum((r.get("thoughts_tokens") or 0) for r in ok),
        "cached_tokens": sum((r.get("cached_tokens") or 0) for r in ok),
        "latency_median_s": statistics.median(r["latency_ms"] for r in ok) / 1000 if ok else 0.0,
        "latency_p95_s": pct([r["latency_ms"] for r in ok], 0.95) / 1000,
        "latency_max_s": max((r["latency_ms"] for r in ok), default=0.0) / 1000,
        "cost_usd": sum(cost_of(r) for r in rows),
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(HERE.glob("usage_*.jsonl"))
    total = defaultdict(float)
    print("| 기록 | 호출 | 확인(무료) | 실패 | 입력 토큰 | 출력 토큰 | thoughts | 지연 중앙값 | p95 | 최대 | 추정 비용 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            print(f"| {path.name} | 0 | 0 | — | 0 | 0 | 0 | — | — | — | $0 |")
            continue
        s = summarize(path)
        err = ", ".join(f"{k}×{v}" for k, v in s["errors"].items()) or "0"
        print(f"| {s['file']} | {s['calls']} | {s['probe_calls']} | {err} | {s['prompt_tokens']:,} | {s['output_tokens']:,} | "
              f"{s['thoughts_tokens']:,} | {s['latency_median_s']:.2f}초 | {s['latency_p95_s']:.2f}초 | {s['latency_max_s']:.2f}초 | ${s['cost_usd']:.3f} |")
        for key in ("calls", "probe_calls", "prompt_tokens", "output_tokens", "thoughts_tokens", "cost_usd"):
            total[key] += s[key]
        total["errors"] += sum(s["errors"].values())
        if len(s["model_versions"]) > 1 or len(s["finish"]) > 1:
            print(f"    model_version {dict(s['model_versions'])} · finish_reason {dict(s['finish'])}")
    print(f"\n합계: 호출 {int(total['calls'])} · 확인 {int(total['probe_calls'])} · 실패 {int(total['errors'])} · "
          f"입력 {int(total['prompt_tokens']):,} · 출력 {int(total['output_tokens']):,} · thoughts {int(total['thoughts_tokens']):,} · "
          f"추정 ${total['cost_usd']:.3f}")


if __name__ == "__main__":
    main()
