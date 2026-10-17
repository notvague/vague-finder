"""§5-4·§5-5 집계 — `fallback_requests.jsonl` + 서버 구간(리랭킹 단계 시간). 계약(RUN_INFO §3-1)과 대조한다.

판정은 기대 계약과 관측을 나란히 적는다. 모르는 값(구간 기록 없음 등)은 통과로 치지 않는다.
출력: fallback_summary.json · fallback_table.csv(경량 표, Git 추적)

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/analyze_fallback.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOG = HERE / "logs" / "main"
GOLD = {"t014": "609293", "t049": "3170749", "q112": None, "n011": None}
FAIL_MODES = {"exception", "invalid_json", "foreign_ids", "empty_ranking", "wrong_types", "slow"}
APPLY_MODES = {"normal", "invalid_then_normal", "partial_ids", "slow_within"}


def load_gold():
    out = {}
    for name in ("v06", "v09", "v10"):
        for row in csv.DictReader((ROOT /f"experiments/reranking/eval_queries_{name}.csv").open(encoding="utf-8")):
            out[row["query_id"]] = set(filter(None, row["relevant_ids"].replace(";", "|").replace(",", "|").split("|")))
    return out


def subsequence_in_order(items, reference):
    pos = {sid: i for i, sid in enumerate(reference)}
    idx = [pos[s] for s in items if s in pos]
    return idx == sorted(idx)


def main() -> int:
    gold = load_gold()
    timing = {}
    for arm in ("A", "D"):
        for line in (LOG / f"timing_{arm}.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            timing[rec["request_id"]] = rec
    rows = [json.loads(line) for line in (HERE / "fallback_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    table, checks = [], []
    first_normal = {}
    for r in rows:
        rec = timing.get(r.get("request_id"))
        rerank_ms = None
        if rec:
            spans = [s for s in rec["spans"] if s["name"] == "search.rerank"]
            rerank_ms = round(spans[0]["run_ms"], 1) if spans else None
        cands = r.get("candidate_ids") or []
        res = r.get("result_ids") or []
        protected = [t["id"] for t in r.get("tracks", []) if any((rule or "").startswith(("gemini_lyric", "lyric")) for rule in t.get("rules", []))]
        group_status = [t.get("rerank_group_status", "") for t in r.get("tracks", [])]
        g = gold.get(r["query_id"], set())
        rank = next((i for i, sid in enumerate(res, start=1) if sid in g), None)
        cand_rank = next((i for i, sid in enumerate(cands, start=1) if sid in g), None)
        row = {
            "arm": r["arm"], "query_id": r["query_id"], "kind": r["kind"], "step": r["step"], "mode": r["mode"], "status": r["status"],
            "client_ms": r["client_ms"], "server_ms": round(rec["total_ms"], 1) if rec else None, "rerank_stage_ms": rerank_ms,
            "stage": r.get("reorder_stage", ""), "note": bool(r.get("reorder_stage_note")), "groups": r.get("rerank_calls"),
            "applied": r.get("rerank_calls_applied"), "failed": r.get("rerank_groups_failed"),
            "budget_exhausted": r.get("rerank_groups_budget_exhausted"), "model_calls": r.get("rerank_model_calls"),
            "fake_calls": r.get("fake_calls"), "fake_outcomes": "|".join(r.get("fake_outcomes") or []),
            "n_results": r.get("n_results"), "dups": (r.get("n_results") or 0) - (r.get("n_unique") or 0),
            "in_candidates": r.get("results_in_candidates"), "result_eq_candidate_prefix": res == cands[: len(res)],
            "nonprotected_in_candidate_order": subsequence_in_order([s for s in res if s not in protected], cands),
            "protected_count_top15": len(protected), "group_status_set": "|".join(sorted(set(group_status))),
            "gold_rank": rank, "gold_candidate_rank": cand_rank,
            "rejected_in_results": any(x in res for x in r.get("rejected_ids") or []),
            "rejected_in_candidates": any(x in cands for x in r.get("rejected_ids") or []),
            "turn2_candidates_differ_from_turn1": (cands != r.get("turn1_candidate_ids")) if r.get("turn1_candidate_ids") else "",
            "skip_reason": r.get("rerank_skip_reason", ""), "rerank_error": r.get("rerank_error", ""),
            "result_top5": "|".join(res[:5]),
        }
        table.append(row)
        key = (r["arm"], r["query_id"])
        if r["mode"] in ("normal", "turn1_normal") and r["mode"] == "normal" and key not in first_normal:
            first_normal[key] = r
        # 계약 판정
        problems = []
        if r["status"] != 200:
            problems.append("status")
        if r["mode"] != "turn1_normal":
            if row["dups"] or not row["in_candidates"] or row["n_results"] != 15:
                problems.append("shape")
            if row["model_calls"] is not None and row["model_calls"] != row["fake_calls"]:
                problems.append("model_calls!=fake_calls")
            if r["kind"] == "policy_skip":
                if row["stage"] != "policy_skipped" or row["fake_calls"] != 0:
                    problems.append("policy_skip_contract")
            elif r["mode"] in FAIL_MODES:
                if row["stage"] != "model_failed":
                    problems.append(f"stage={row['stage']}")
                if r["kind"] != "lyric_protected" and not row["result_eq_candidate_prefix"]:
                    problems.append("fallback_not_candidate_order")
                if r["kind"] == "lyric_protected" and not row["nonprotected_in_candidate_order"]:
                    problems.append("fallback_nonprotected_order")
                if row["rerank_error"]:
                    problems.append("router_exception")
            elif r["mode"] == "fail_group_first":
                if row["stage"] != "model_partial" or row["failed"] != 1 or not row["note"]:
                    problems.append("partial_contract")
            elif r["mode"] in APPLY_MODES:
                if row["stage"] != "model_applied":
                    problems.append(f"stage={row['stage']}")
            if r["kind"].startswith("turn2") and (row["rejected_in_results"] or row["rejected_in_candidates"]):
                problems.append("rejected_leak")
        checks.append({"arm": r["arm"], "query_id": r["query_id"], "step": r["step"], "mode": r["mode"], "problems": problems})
    # 상태 누출: 마지막 normal == 첫 normal
    leaks = []
    for key, first in first_normal.items():
        lasts = [r for r in rows if (r["arm"], r["query_id"]) == key and r["mode"] == "normal"]
        last = lasts[-1]
        if last is first:
            continue
        same = (last.get("result_ids") == first.get("result_ids") and last.get("reorder_stage") == first.get("reorder_stage")
                and last.get("rerank_calls") == first.get("rerank_calls") and last.get("rerank_model_calls") == first.get("rerank_model_calls")
                and last.get("candidate_ids") == first.get("candidate_ids"))
        leaks.append({"arm": key[0], "query_id": key[1], "last_normal_equals_first": same})
    fails = [c for c in checks if c["problems"]]
    summary = {
        "requests": len(rows), "contract_violations": fails, "leak_checks": leaks,
        "fallback_gold": [{k: t[k] for k in ("arm", "query_id", "mode", "gold_rank", "gold_candidate_rank", "stage")} for t in table
                          if t["query_id"] in ("t014", "t049") and t["mode"] in ("normal", "exception", "slow")],
        "slow": [{k: t[k] for k in ("arm", "query_id", "client_ms", "server_ms", "rerank_stage_ms", "groups", "failed", "budget_exhausted",
                                    "model_calls", "fake_outcomes")} for t in table if t["mode"] == "slow"],
        "status_after": {},
    }
    after = json.loads((HERE / "fallback_status_after.json").read_text(encoding="utf-8"))
    summary["status_after"] = {arm: {"network_attempts": len(s["network_attempts"]), "blocked_client_calls": s["blocked_client_calls"],
                                     "analysis_misses": s["analysis_misses"], "fake_calls": s["fake_calls"]} for arm, s in after.items()}
    (HERE / "fallback_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with (HERE / "fallback_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
