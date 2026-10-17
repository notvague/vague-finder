"""v65 §4-⑥ HTTP 비교 — HTTP 실행끼리(로컬↔서버·반복)와 같은 코드·같은 백엔드의 평가기 실행(offline)을 대조한다.

    venv/bin/python compare_http.py  → compare_http.json

항목: 응답 200 · 후보 30 ID·순서 · 결과 15 · 가사 종류 · 리랭킹 단계 · 정책 생략 · 가짜 SDK 호출의 프롬프트 sha16 목록.
평가기(top_k 30)와 서비스(top_k 15)는 리랭커 호출 조건이 달라 프롬프트 sha가 같을 필요가 없다 — 후보만 같은 조건으로 비교하고,
프롬프트는 HTTP끼리 비교한다. 평가기와의 프롬프트 차이는 원인(묶음 수·top_k)만 센다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
SETS = ("v06", "v09", "v10")


def load_http(run: str) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for line in (HERE / "http" / f"{run}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[r["query_id"]] = r
    return rows


def load_eval(run: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for s in SETS:
        for line in (HERE / "offline" / run / f"{s}_dev" / "v65_queries.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["query_id"]] = r
    return out


def view(r: Dict[str, Any]) -> Dict[str, Any]:
    return {"cand": r.get("candidate_ids"), "results": r.get("result_ids"), "lyric": r.get("lyric_types"),
            "stage": r.get("reorder_stage"), "skip": r.get("rerank_skip_reason"),
            "prompts": [c["sha"] for c in r.get("fake_calls", [])], "groups": [c["ids"] for c in r.get("fake_calls", [])]}


def compare_http(a: str, b: str) -> Dict[str, Any]:
    ra, rb = load_http(a), load_http(b)
    fields = ("cand", "results", "lyric", "stage", "skip", "prompts", "groups")
    diff = {f: [q for q in ra if view(ra[q])[f] != view(rb[q])[f]] for f in fields}
    return {"pair": f"{a} vs {b}", "queries": len(ra), "identical_all": sum(1 for q in ra if all(view(ra[q])[f] == view(rb[q])[f] for f in fields)),
            "diff_counts": {f: len(v) for f, v in diff.items()}, "diff_queries": {f: v for f, v in diff.items() if v},
            "cand_set_diff": [q for q in ra if set(ra[q].get("candidate_ids") or []) != set(rb[q].get("candidate_ids") or [])]}


def compare_with_eval(http_run: str, eval_run: str) -> Dict[str, Any]:
    h, e = load_http(http_run), load_eval(eval_run)
    cand_diff = [q for q in h if (h[q].get("candidate_ids") or []) != [c["id"] for c in e[q]["candidates"]]]
    groups_h = {q: len(h[q].get("fake_calls", [])) for q in h}
    groups_e = {q: len(e[q].get("generates", [])) for q in e}
    prompt_same = [q for q in h if [c["sha"] for c in h[q].get("fake_calls", [])] == [g["contents_sha16"] for g in e[q].get("generates", [])]]
    first_group_same_ids = [q for q in h if h[q].get("fake_calls") and e[q].get("rerank_runs")
                            and h[q]["fake_calls"][0]["ids"].split("|") == e[q]["rerank_runs"][0]["input_ids"][:len(h[q]["fake_calls"][0]["ids"].split("|"))]]
    return {"pair": f"HTTP {http_run} vs 평가기 {eval_run}", "queries": len(h),
            "candidates_identical": len(h) - len(cand_diff), "candidate_diff_queries": cand_diff,
            "model_call_count_diff_queries": {q: [groups_h[q], groups_e[q]] for q in h if groups_h[q] != groups_e[q]},
            "prompt_sha_list_identical": len(prompt_same),
            "first_group_candidate_ids_identical": len(first_group_same_ids),
            "status_200": sum(1 for r in h.values() if r["status"] == 200)}


def main() -> int:
    out = {"http_pairs": [], "http_vs_eval": [], "runs": {}}
    runs = sorted(p.stem for p in (HERE / "http").glob("*.jsonl"))
    for run in runs:
        rows = load_http(run)
        st = json.loads((HERE / "http" / f"{run}_status.json").read_text(encoding="utf-8"))
        out["runs"][run] = {
            "status_200": sum(1 for r in rows.values() if r["status"] == 200), "queries": len(rows),
            "fake_calls": sum(len(r.get("fake_calls", [])) for r in rows.values()),
            "policy_skipped": sum(1 for r in rows.values() if r.get("rerank_skip_reason")),
            "failed_paths": sum(1 for r in rows.values() if r.get("failed_paths")),
            "candidates_30": sum(1 for r in rows.values() if len(r.get("candidate_ids") or []) == 30),
            "network_attempts": len(st["after"]["network_attempts"]), "blocked_client_calls": st["after"]["blocked_client_calls"],
            "code": st["after"].get("code"), "backend_file": st["after"].get("backend_file"), "seconds": st["seconds"],
            "lyrics_requests_with_calls": sum(1 for r in rows.values() if r.get("lyrics_calls")),
            "lyrics_errors": sum(1 for r in rows.values() for c in r.get("lyrics_calls") or [] if not c.get("ok")),
        }
    pairs = [("after_local_1", "after_server_1"), ("after_local_2", "after_server_2"), ("after_local_1", "after_local_2"),
             ("after_server_1", "after_server_2"), ("before_local_1", "before_server_1")]
    for a, b in pairs:
        if a in runs and b in runs:
            out["http_pairs"].append(compare_http(a, b))
    for h, e in (("after_local_1", "after_local_rep1"), ("after_server_1", "after_server_rep1"),
                 ("before_local_1", "before_local_rep1"), ("before_server_1", "before_server_rep1")):
        if h in runs:
            out["http_vs_eval"].append(compare_with_eval(h, e))
    (HERE / "compare_http.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out["runs"].items():
        print(k, json.dumps(v, ensure_ascii=False)[:300])
    for p in out["http_pairs"]:
        print(p["pair"], p["identical_all"], p["diff_counts"], "set", len(p["cand_set_diff"]), p["diff_queries"].get("cand"))
    for p in out["http_vs_eval"]:
        print(p["pair"], "cand", p["candidates_identical"], "calls diff", p["model_call_count_diff_queries"], "prompt same", p["prompt_sha_list_identical"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
