"""v66 예산 장부 요약 — 실행별·전체 SDK HTTP 시도(generate·countTokens·그 밖)·실패·재시도·토큰·정산·미확인·열린 예약·정지.
모델·DB 호출 없음. 장부 원본(`budget/ledger.jsonl`)은 Git 제외, 이 요약(`budget_summary.json`)만 추적한다.

재시도 = (리랭커 그룹 안 두 번째 이후 시도, 하네스 기록) + (SDK 단 재시도 = 장부의 generate 시도 − 하네스가 센 generate 호출).

    venv/bin/python budget_summary.py [--ledger budget/ledger.jsonl] [--live-root live] [--out budget_summary.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRICE = {"input": 0.25, "output": 1.50}


def harness_counts(run_dir: Path) -> dict:
    out = {"generate_calls": 0, "retry_attempts": 0, "generate_errors": 0, "groups": 0, "groups_without_call": 0}
    for f in sorted(run_dir.glob("*_dev/v66_queries.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            gens = row.get("generates", [])
            out["generate_calls"] += len(gens)
            out["retry_attempts"] += sum(1 for g in gens if g.get("attempt", 0) > 0)
            out["generate_errors"] += sum(1 for g in gens if g.get("error"))
            n = len(row.get("rerank_runs", []))
            out["groups"] += n
            called = {g.get("group") for g in gens}
            out["groups_without_call"] += sum(1 for i in range(n) if i not in called)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(HERE / "budget/ledger.jsonl"))
    parser.add_argument("--live-root", default=str(HERE / "live"))
    parser.add_argument("--out", default=str(HERE / "budget_summary.json"))
    args = parser.parse_args()
    rows = [json.loads(l) for l in Path(args.ledger).read_text(encoding="utf-8").splitlines() if l.strip()]
    reserves = {r["id"]: r for r in rows if r["event"] == "reserve"}
    closed = {r["id"]: r for r in rows if r["event"] in {"settle", "fail"}}
    anchor = next((r for r in rows if r["event"] == "anchor"), None)
    halts = [r for r in rows if r["event"] == "halt"]
    per = defaultdict(lambda: defaultdict(float))
    lat = defaultdict(list)
    for rid, r in reserves.items():
        run = r["run"]
        p = per[run]
        p["attempts"] += 1
        p[f"attempts_{r['kind']}"] += 1
        p["reserve_usd_sum"] += r["reserve_usd"]
        c = closed.get(rid)
        if c is None:
            p["open_reservations"] += 1
            p["open_reserve_usd"] += r["reserve_usd"]
            continue
        if c["event"] == "settle":
            p["settled"] += 1
            p["cost_usd"] += c.get("cost_usd", 0.0)
            if r["kind"] == "generate":
                p["prompt_tokens"] += c.get("prompt_tokens") or 0
                p["candidates_tokens"] += c.get("candidates_tokens") or 0
                p["thoughts_tokens"] += c.get("thoughts_tokens") or 0
                p["cached_tokens"] += c.get("cached_tokens") or 0
                p["tool_prompt_tokens"] += c.get("tool_prompt_tokens") or 0
                lat[run].append(c.get("latency_ms") or 0)
        else:
            p["failed"] += 1
            p["unverified_usd"] += c.get("unverified_usd", 0.0)
            p[f"fail_{c.get('error')}_{c.get('error_code')}"] += 1
    result = {"ledger_rows": len(rows), "first_attempt_epoch": anchor["first_attempt_epoch"] if anchor else None,
              "deadline_epoch": anchor["deadline_epoch"] if anchor else None,
              "first_attempt_ts": next((r["ts"] for r in rows if r["event"] == "anchor"), None),
              "last_event_ts": rows[-1]["ts"] if rows else None, "halts": [{"run": h["run"], "ts": h["ts"], "reason": h["reason"]} for h in halts],
              "prices_usd_per_1m": PRICE, "runs": {}, "total": {}}
    total = defaultdict(float)
    for run, p in per.items():
        entry = {k: (round(v, 6) if isinstance(v, float) and not v.is_integer() else int(v)) for k, v in sorted(p.items())}
        entry["cost_usd"] = round(p["cost_usd"], 6)
        entry["unverified_usd"] = round(p["unverified_usd"], 6)
        entry["output_tokens_incl_thinking"] = int(p["candidates_tokens"] + p["thoughts_tokens"])
        if lat[run]:
            entry["generate_latency_ms_median"] = round(statistics.median(lat[run]), 1)
            entry["generate_latency_ms_p95"] = round(sorted(lat[run])[int(0.95 * (len(lat[run]) - 1))], 1)
        run_dir = Path(args.live_root) / run
        if run_dir.exists():
            h = harness_counts(run_dir)
            entry["harness"] = h
            entry["sdk_level_retries"] = int(p.get("attempts_generate", 0)) - h["generate_calls"]
            entry["reranker_retry_attempts"] = h["retry_attempts"]
        result["runs"][run] = entry
        for k, v in p.items():
            total[k] += v
    result["total"] = {k: (round(v, 6) if not float(v).is_integer() else int(v)) for k, v in sorted(total.items())}
    result["total"]["cost_usd"] = round(total["cost_usd"], 6)
    result["total"]["unverified_usd"] = round(total["unverified_usd"], 6)
    result["total"]["committed_usd"] = round(total["cost_usd"] + total["unverified_usd"] + total["open_reserve_usd"], 6)
    result["total"]["output_tokens_incl_thinking"] = int(total["candidates_tokens"] + total["thoughts_tokens"])
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    t = result["total"]
    print(f"시도 {t.get('attempts', 0)} (generate {t.get('attempts_generate', 0)} · countTokens {t.get('attempts_count_tokens', 0)}) · "
          f"실패 {t.get('failed', 0)} · 정산 ${t['cost_usd']:.4f} · 미확인 ${t['unverified_usd']:.4f} · 확정+미확인+열림 ${t['committed_usd']:.4f} · "
          f"정지 {len(halts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
