"""§5 항목 1·1b·2 집계 — `sweep_requests.jsonl` + 서버 구간 기록 + v63 offline B 후보 + 사전 분류(`query_categories.csv`).

모르는 값은 정상으로 치지 않는다: 기대한 행이 없거나 상태가 200이 아니면 그대로 세고, 구간 기록이 없는 요청도 따로 센다.
출력: `sweep_summary.json`(요약) · `sweep_per_query.csv`(질의별 경량 표, Git 추적).

실행: venv/bin/python experiments/reranking/results_v64_service_readiness/analyze_sweep.py
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
LOG = HERE / "logs" / "main"
V63_OFFLINE = Path(os.getenv(
    "V64_V63_OFFLINE_DIR",
    "/private/tmp/vague-finder-hybrid-lyrics-validation/experiments/reranking/results_v63_hybrid_lyrics_validation/offline/offline_B_rep1",
))
LYRIC_QUERY_IDS = ("q209", "q211", "q213", "q219", "n007", "n009", "n013", "n017", "n039", "n064",
                   "t003", "t016", "t018", "t025", "t032", "t060", "t063", "t105", "t110")
PROTECT_RULES_PREFIX = ("gemini_lyric", "lyric")


def load_timing(arm: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    path = LOG / f"timing_{arm}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["request_id"]] = rec
    return out


def load_v63() -> Dict[str, Dict[str, Any]]:
    out = {}
    for set_dir in ("v06_dev", "v09_dev", "v10_dev"):
        for line in (V63_OFFLINE / set_dir / "v63_queries.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            out[rec["query_id"]] = rec
    return out


def main() -> int:
    rows = [json.loads(line) for line in (HERE / "sweep_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    timing = {arm: load_timing(arm) for arm in ("A", "D")}
    cats = {r["query_id"]: r for r in csv.DictReader((HERE / "query_categories.csv").open(encoding="utf-8"))}
    v63 = load_v63()
    by = {(r["arm"], r["kind"], r["query_id"]): r for r in rows}
    summary: Dict[str, Any] = {"requests": len(rows)}

    status = collections.Counter((r["arm"], r["kind"], str(r["status"])) for r in rows)
    summary["status"] = {f"{a}|{k}|{s}": n for (a, k, s), n in sorted(status.items())}
    ok = [r for r in rows if r["status"] == 200]
    summary["shape"] = {
        "n_results_counter": dict(collections.Counter(r["n_results"] for r in ok)),
        "duplicates": sum(1 for r in ok if r["n_unique"] != r["n_results"]),
        "candidate_duplicates": sum(1 for r in ok if r["n_candidates_unique"] != r["n_candidates"]),
        "results_outside_candidates": sum(1 for r in ok if not r["results_in_candidates"]),
        "n_candidates_counter": dict(collections.Counter(r["n_candidates"] for r in ok)),
        "failed_paths_nonempty": [(r["arm"], r["query_id"], r["failed_paths"]) for r in ok if r["failed_paths"]],
        "analysis_fallback_true": sum(1 for r in ok if r.get("analysis_fallback")),
        "rerank_error_nonempty": sum(1 for r in ok if r.get("rerank_error")),
    }
    stages = collections.Counter((r["arm"], r["kind"], r["reorder_stage"], r["rerank_skip_reason"]) for r in ok)
    summary["stages"] = {f"{a}|{k}|{s}|{p}": n for (a, k, s, p), n in sorted(stages.items())}
    # 실제 SDK 호출 수 = 가짜 클라이언트가 기록한 호출 수인가
    mismatch_calls = [(r["arm"], r["query_id"], r["rerank_model_calls"], r["fake_calls"]) for r in ok
                      if r["rerank_model_calls"] is not None and r["rerank_model_calls"] != r["fake_calls"]]
    summary["model_calls_vs_fake"] = {"checked": sum(1 for r in ok if r["rerank_model_calls"] is not None),
                                      "null": sum(1 for r in ok if r["rerank_model_calls"] is None), "mismatch": mismatch_calls}
    summary["policy_skipped_fake_calls"] = sum(r["fake_calls"] for r in ok if r["reorder_stage"] == "policy_skipped")
    summary["fake_calls_total"] = {arm: sum(r["fake_calls"] for r in rows if r["arm"] == arm) for arm in ("A", "D")}

    # 항목 1 — 가사 19질의
    lyric: Dict[str, Any] = {}
    for qid in LYRIC_QUERY_IDS:
        entry = {}
        for arm in ("A", "D"):
            r = by.get((arm, "dev", qid))
            if r is None:
                entry[arm] = "MISSING"
                continue
            calls = r.get("lyrics_calls") or []
            searched = [c for c in calls if any(k in ("verbatim", "partial", "phonetic") for k in c.get("clue_kinds", []))]
            matched = [t for t in r.get("tracks", []) if t.get("lyric_match_type")]
            protected_rules = sorted({rule for t in r.get("tracks", []) for rule in t.get("rules", []) if rule and rule.startswith(PROTECT_RULES_PREFIX)})
            entry[arm] = {
                "status": r["status"], "lyrics_searches": len(searched), "lyrics_ok": all(c.get("ok") for c in searched) if searched else None,
                "lyrics_errors": [c.get("error") for c in searched if not c.get("ok")],
                "lyrics_results": [c.get("n_results") for c in searched],
                "lyrics_failed_path": "lyrics_surface" in (r.get("failed_paths") or []) or any("lyric" in p for p in (r.get("failed_paths") or [])),
                "rerank_groups": r["rerank_calls"], "fake_calls": r["fake_calls"], "fake_sizes": r["fake_sizes"],
                "stage": r["reorder_stage"], "results_with_lyric_match": len(matched),
                "results_with_lyric_detail": sum(1 for t in matched if t.get("has_lyric_detail")),
                "group_status_values": sorted({t.get("rerank_group_status", "") for t in r.get("tracks", [])}),
                "protect_rules": protected_rules,
                "v63_protected": len((v63.get(qid, {}).get("protection") or {}).get("protected_ids") or []),
                "v63_rerank_runs": len(v63.get(qid, {}).get("rerank_runs") or []),
                "v63_lyrics_results": [l.get("results") for l in v63.get(qid, {}).get("lyrics") or []],
            }
        lyric[qid] = entry
    zero = [q for q, e in lyric.items() if isinstance(e.get("A"), dict) and e["A"]["lyrics_results"] and all(n == 0 for n in e["A"]["lyrics_results"])]
    multi_group = {arm: [q for q, e in lyric.items() if isinstance(e.get(arm), dict) and (e[arm]["rerank_groups"] or 0) > 1] for arm in ("A", "D")}
    summary["lyric19"] = {
        "searched_all": {arm: all(isinstance(e.get(arm), dict) and e[arm]["lyrics_searches"] >= 1 for e in lyric.values()) for arm in ("A", "D")},
        "errors": {arm: sum(len(e[arm]["lyrics_errors"]) for e in lyric.values() if isinstance(e.get(arm), dict)) for arm in ("A", "D")},
        "lyrics_failed_path": {arm: [q for q, e in lyric.items() if isinstance(e.get(arm), dict) and e[arm]["lyrics_failed_path"]] for arm in ("A", "D")},
        "zero_result_queries_A": zero,
        "multi_group_queries": multi_group,
        "multi_group_matches_v63_runs": {arm: sum(1 for q, e in lyric.items() if isinstance(e.get(arm), dict) and e[arm]["rerank_groups"] == e[arm]["v63_rerank_runs"]) for arm in ("A", "D")},
        "v63_protected_queries": sum(1 for e in lyric.values() if isinstance(e.get("A"), dict) and e["A"]["v63_protected"] > 0),
        "per_query": lyric,
    }
    summary["lyrics_calls_nonlyric_with_search"] = sum(
        1 for r in ok if r["kind"] == "dev" and r["query_id"] not in LYRIC_QUERY_IDS
        and any(any(k in ("verbatim", "partial", "phonetic") for k in c.get("clue_kinds", [])) for c in r.get("lyrics_calls") or []))

    # 항목 1b — A의 HTTP 후보 30 ↔ v63 offline B 후보(같은 조건)
    diff_1b = []
    same_1b = 0
    for qid, rec in v63.items():
        r = by.get(("A", "dev", qid))
        v63_ids = [c["id"] for c in rec.get("candidates") or []]
        if r is None or r["status"] != 200:
            diff_1b.append((qid, "missing_or_error"))
            continue
        if r["candidate_ids"] == v63_ids:
            same_1b += 1
        else:
            first = next((i for i, (x, y) in enumerate(zip(r["candidate_ids"], v63_ids)) if x != y), min(len(r["candidate_ids"]), len(v63_ids)))
            diff_1b.append((qid, f"first_diff_at={first + 1} len={len(r['candidate_ids'])}/{len(v63_ids)} same_set={set(r['candidate_ids']) == set(v63_ids)}"))
    summary["http_vs_v63_offline_B"] = {"queries": len(v63), "identical_order": same_1b, "diffs": diff_1b}

    # 항목 2 — 보조 경로 열림/닫힘 ↔ D의 path.attr_filter 구간, 닫힌 질의 A=D
    attr_rows = []
    for r in ok:
        rec = timing[r["arm"]].get(r["request_id"])
        spans = rec["spans"] if rec else None
        names = [s["name"] for s in spans] if spans else []
        attr = [s for s in (spans or []) if s["name"] == "path.attr_filter"]
        counts = [s for s in (spans or []) if s["name"] == "qdrant.count"]
        attr_rows.append({"arm": r["arm"], "kind": r["kind"], "query_id": r["query_id"], "timing": bool(rec),
                          "attr_span": bool(attr), "filtered_counts": sum(1 for s in counts if (s.get("attrs") or {}).get("filtered")),
                          "unfiltered_counts": sum(1 for s in counts if not (s.get("attrs") or {}).get("filtered")),
                          "paths": sorted({n for n in names if n.startswith("path.")})})
    missing_timing = [(a["arm"], a["query_id"]) for a in attr_rows if not a["timing"]]
    mismatch_open = []
    for a in attr_rows:
        if a["kind"] != "dev":
            continue
        cat = cats[a["query_id"]]["category"]
        expected = a["arm"] == "D" and cat != "closed"
        if a["attr_span"] != expected:
            mismatch_open.append((a["arm"], a["query_id"], cat, a["attr_span"]))
    closed_diff = []
    open_diff = 0
    for qid, cat in ((q, c["category"]) for q, c in cats.items()):
        ra, rd = by.get(("A", "dev", qid)), by.get(("D", "dev", qid))
        if not ra or not rd or ra["status"] != 200 or rd["status"] != 200:
            continue
        same = ra["candidate_ids"] == rd["candidate_ids"]
        if cat == "closed" and not same:
            closed_diff.append(qid)
        if cat != "closed" and not same:
            open_diff += 1
    synth = [a for a in attr_rows if a["kind"] == "synthetic_empty"]
    summary["attr_path"] = {
        "missing_timing": missing_timing, "open_span_mismatch": mismatch_open,
        "D_attr_span_dev": sum(1 for a in attr_rows if a["arm"] == "D" and a["kind"] == "dev" and a["attr_span"]),
        "A_attr_span": sum(1 for a in attr_rows if a["arm"] == "A" and a["attr_span"]),
        "categories_dev": dict(collections.Counter(c["category"] for c in cats.values())),
        "closed_A_ne_D": closed_diff, "open_A_ne_D": open_diff,
        "synthetic": {arm: {"attr_span": sum(1 for a in synth if a["arm"] == arm and a["attr_span"]), "n": sum(1 for a in synth if a["arm"] == arm),
                            "filtered_counts": sum(a["filtered_counts"] for a in synth if a["arm"] == arm)} for arm in ("A", "D")},
        "synthetic_candidates": {arm: [by[(arm, "synthetic_empty", s["query_id"])]["n_candidates"] for s in synth if s["arm"] == arm] for arm in ("A", "D")},
        "D_filtered_counts_per_open_request": dict(collections.Counter(a["filtered_counts"] for a in attr_rows if a["arm"] == "D" and a["attr_span"])),
        "unfiltered_counts_per_request": dict(collections.Counter(a["unfiltered_counts"] for a in attr_rows)),
    }
    miss = [r for r in rows if r["kind"] == "cache_miss"]
    summary["cache_miss"] = [{"arm": r["arm"], "status": r["status"], "fake_calls": r["fake_calls"], "body": (r.get("error_body") or "")[:120]} for r in miss]
    after = json.loads((HERE / "sweep_status_after.json").read_text(encoding="utf-8"))
    summary["status_after"] = {arm: {"network_attempts": len(s["network_attempts"]), "blocked_client_calls": s["blocked_client_calls"],
                                     "analysis_misses": s["analysis_misses"], "lyrics_reads": s["lyrics_source"]["reads"],
                                     "fake_calls": s["fake_calls"]} for arm, s in after.items()}
    (HERE / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    with (HERE / "sweep_per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["arm", "kind", "query_id", "status", "n_results", "n_candidates", "reorder_stage", "rerank_skip_reason", "rerank_calls",
                  "rerank_model_calls", "fake_calls", "failed_paths", "lyrics_searches", "attr_span", "category", "candidate_top5", "result_top5"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        attr_by = {(a["arm"], a["kind"], a["query_id"]): a for a in attr_rows}
        for r in rows:
            a = attr_by.get((r["arm"], r["kind"], r["query_id"]), {})
            writer.writerow({
                "arm": r["arm"], "kind": r["kind"], "query_id": r["query_id"], "status": r["status"], "n_results": r.get("n_results", ""),
                "n_candidates": r.get("n_candidates", ""), "reorder_stage": r.get("reorder_stage", ""), "rerank_skip_reason": r.get("rerank_skip_reason", ""),
                "rerank_calls": r.get("rerank_calls", ""), "rerank_model_calls": r.get("rerank_model_calls", ""), "fake_calls": r.get("fake_calls", ""),
                "failed_paths": "|".join(r.get("failed_paths") or []),
                "lyrics_searches": sum(1 for c in r.get("lyrics_calls") or [] if any(k in ("verbatim", "partial", "phonetic") for k in c.get("clue_kinds", []))),
                "attr_span": a.get("attr_span", ""), "category": cats.get(r["query_id"], {}).get("category", r["kind"]),
                "candidate_top5": "|".join((r.get("candidate_ids") or [])[:5]), "result_top5": "|".join((r.get("result_ids") or [])[:5]),
            })
    printable = {k: v for k, v in summary.items() if k not in ("lyric19",)}
    printable["lyric19"] = {k: v for k, v in summary["lyric19"].items() if k != "per_query"}
    print(json.dumps(printable, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
