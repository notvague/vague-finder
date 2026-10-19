"""v73 3차 소규모 확인 판정 — 실행 전에 고정한 기준(2026-10-18, 결과를 본 뒤 바꾸지 않는다).

성공률과 지연만 본다. 정답률·순위는 읽지 않는다. 모두 통과해야 본 실행(run_live_attempt3.sh)을 시작한다.

  S1 인증·권한·결제 오류 0
  S2 정지 사유는 시간 창 만료(신규 호출 마감)뿐 — 장애 집중·연속 실패·비용·시도 상한 정지는 불합격
  S3 실패한 HTTP 시도 2회 이하, 연속 실패 없음
  S4 리랭킹 최종 실패(검색 순서 폴백) 0
  S5 첫·마지막 정산 generate 사이 15분(900초) 이상 — 1·2차는 시작 7~12분 뒤 지연이 시작됐다
  S6 지연: 마지막 5분 정산 generate 중앙값 ≤ 처음 5분 중앙값 × 1.5, 10초 넘은 정산 generate 3회 이하,
     20초 넘은 정산 generate 1회 이하. 정상 구간(1차 A1 163회)의 최대는 6.2초였다. 중앙값만 보면 2차처럼 일부만 느려진
     구간(8.6~17.1초 5회)을 놓쳐서 10초 조건을 더했다(1·2차 장부로 확인, 실행 전 고정).

    venv/bin/python smoke_check_t60.py --smoke-dir attempt3/smoke   # 통과 0 · 불합격 1 · 자료 없음 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

WINDOW_HALT = "신규 호출 마감(첫 시도 + 창) 이후"
AUTH_CODES = {401, 403}
AUTH_STATUSES = {"PERMISSION_DENIED", "UNAUTHENTICATED", "BILLING_DISABLED"}


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def check(smoke_dir: Path) -> Dict[str, Any]:
    ledger = smoke_dir / "budget/ledger.jsonl"
    if not ledger.exists():
        return {"pass": False, "error": "장부 없음"}
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    closed = sorted((r for r in rows if r["event"] in ("settle", "fail")), key=lambda r: r["epoch"])
    fails = [r for r in closed if r["event"] == "fail"]
    gen_ok = [r for r in closed if r["event"] == "settle" and r.get("kind") == "generate"]
    halts = [r["reason"] for r in rows if r["event"] == "halt"]
    anchor = next((r for r in rows if r["event"] == "anchor"), None)

    auth = [r for r in fails if r.get("error_code") in AUTH_CODES or r.get("error_status") in AUTH_STATUSES]
    consecutive = max((sum(1 for _ in g) for k, g in __import__("itertools").groupby(closed, key=lambda r: r["event"])
                       if k == "fail"), default=0)
    fallbacks = sum(p.read_text(errors="replace").count("최종 실패") for p in (smoke_dir / "logs").glob("smoke_live_*.log"))
    span = (gen_ok[-1]["epoch"] - gen_ok[0]["epoch"]) if len(gen_ok) >= 2 else 0.0
    first5 = [r["latency_ms"] for r in gen_ok if r["epoch"] - gen_ok[0]["epoch"] <= 300] if gen_ok else []
    last5 = [r["latency_ms"] for r in gen_ok if gen_ok[-1]["epoch"] - r["epoch"] <= 300] if gen_ok else []
    over20 = sum(1 for r in gen_ok if r["latency_ms"] > 20000)
    over10 = sum(1 for r in gen_ok if r["latency_ms"] > 10000)
    buckets: Dict[int, List[float]] = {}
    for r in gen_ok:
        buckets.setdefault(int((r["epoch"] - gen_ok[0]["epoch"]) // 300), []).append(r["latency_ms"])

    crit = {
        "S1_auth_errors_0": len(auth) == 0,
        "S2_only_window_halt": all(h == WINDOW_HALT for h in halts),
        "S3_fails_le_2_no_consecutive": len(fails) <= 2 and consecutive <= 1,
        "S4_rerank_fallbacks_0": fallbacks == 0,
        "S5_span_ge_900s": span >= 900,
        "S6_latency_stable": bool(first5 and last5 and _median(last5) <= 1.5 * _median(first5)
                                   and over10 <= 3 and over20 <= 1),
    }
    return {
        "pass": all(crit.values()),
        "criteria": crit,
        "observed": {
            "attempts": sum(1 for r in rows if r["event"] == "reserve"),
            "settled_generate": len(gen_ok),
            "failed_attempts": len(fails),
            "fail_detail": [(r["ts"][11:19], r.get("error_code") or r.get("error"), round(r["latency_ms"] / 1000, 1))
                            for r in fails],
            "max_consecutive_fails": consecutive,
            "auth_errors": len(auth),
            "halts": halts,
            "rerank_fallbacks": fallbacks,
            "first_attempt": anchor["ts"] if anchor else None,
            "span_seconds": round(span, 1),
            "median_first5_ms": round(_median(first5)) if first5 else None,
            "median_last5_ms": round(_median(last5)) if last5 else None,
            "generate_over_10s": over10,
            "generate_over_20s": over20,
            "max_generate_ms": round(max(r["latency_ms"] for r in gen_ok)) if gen_ok else None,
            "per_5min": {f"{k * 5}-{k * 5 + 5}분": {"n": len(v), "median_ms": round(_median(v)),
                                                   "max_ms": round(max(v))} for k, v in sorted(buckets.items())},
            "settled_usd": round(sum(r.get("cost_usd", 0) for r in closed if r["event"] == "settle"), 6),
            "unverified_usd": round(sum(r.get("unverified_usd", 0) for r in fails), 6),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-dir", type=Path, required=True)
    args = ap.parse_args()
    out = check(args.smoke_dir)
    (args.smoke_dir / "smoke_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if "error" in out:
        return 2
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
