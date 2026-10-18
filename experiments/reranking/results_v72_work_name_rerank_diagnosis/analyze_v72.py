"""v72 실제 리랭킹 부분집합 진단 집계 — 모델·DB 호출 없음(저장된 실행 기록만 읽는다).

입력: `live/live_{off,work}_server_rep{1,2}`(완료 표시 DONE이 있는 실행만 "완료"로 센다) · 부분집합 파일 · dev 질의 세트(정답 id).
규칙은 RUN_INFO §4(실행 전 고정)를 그대로 코드로 옮겼다:
  - 분모 = 부분집합 13건 전체(실패·부분·예산 소진 포함, 평가기의 폴백 순서로 채점). 보조 = 네 실행 모두 applied인 질의만.
  - 질의별 Hit@10 분류: 안정 이득 · 안정 손실 · 불안정(같은 구성 두 반복이 다름) · 같음. 하나라도 applied가 아니면 "실패 영향" 표시.
  - 변동: A1↔A2 · B1↔B2 Hit@10 뒤집힘(실패 관련 / 둘 다 applied로 나눔)·최종 10위 목록 차이·응답 sha 차이.
  - 전체 A/B 권고 기준 1~6 판정.
산출: analysis/live_summary.json · analysis/per_query.csv(질의 원문·작품명 없음)

    venv/bin/python analyze_v72.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")
RUNS = {"A1": "live_off_server_rep1", "B1": "live_work_server_rep1", "B2": "live_work_server_rep2", "A2": "live_off_server_rep2"}
NEW_CANDIDATES = ("n033", "n043", "n096")  # v71: B에서 새로 후보 30에 든 정답
TITLE_EQUALS_WORK = "n096"
NAME_IN_QUERY = ("q105", "n031")


def relevant_ids() -> Dict[str, set]:
    out = {}
    for s in SETS:
        with (ROOT / f"experiments/reranking/eval_queries_{s}.csv").open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["split"].strip() == "dev":
                    out[r["query_id"].strip()] = {p.strip() for p in (r["relevant_ids"] or "").split("|") if p.strip()}
    return out


def first_rank(ids: List[str], gold: set) -> Optional[int]:
    for i, sid in enumerate(ids, start=1):
        if sid in gold:
            return i
    return None


def query_status(row: Dict[str, Any]) -> str:
    prot = row.get("protection") or {}
    if prot.get("policy_skip"):
        return "policy_skipped"
    statuses = (row.get("final") or {}).get("statuses") or []
    runs = [r.get("status") for r in row.get("rerank_runs", [])]
    vals = sorted(set(statuses) | set(runs)) or ["(none)"]
    return vals[0] if len(vals) == 1 else "+".join(vals)


def load(name: str, subset: List[str], gold: Dict[str, set]) -> Dict[str, Any]:
    d = HERE / "live" / name
    out: Dict[str, Any] = {"name": name, "exists": d.exists(), "done": (d / "DONE").exists(), "queries": {}}
    if not d.exists():
        return out
    out["conditions_status"] = json.loads((d / "run_conditions.json").read_text(encoding="utf-8")).get("status")
    for s in SETS:
        qf = d / f"{s}_dev/v72_queries.jsonl"
        ef = d / f"{s}_dev/search_eval_dev_explain.jsonl"
        if not qf.exists():
            continue
        work = {}
        if ef.exists():
            for line in ef.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                work[r["query_id"]] = (r.get("work_resolution") or {}).get("status", "(none)")
        for line in qf.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            q = row["query_id"]
            final = (row.get("final") or {}).get("ids") or []
            cands = [c["id"] for c in row.get("candidates") or []]
            gens = row.get("generates", [])
            out["queries"][q] = {
                "set": s, "final_rank": first_rank(final, gold[q]), "cand_rank": first_rank(cands, gold[q]),
                "top10": final[:10], "status": query_status(row), "work_status": work.get(q, "(none)"),
                "generate_calls": len(gens), "generate_errors": [g.get("error") for g in gens if g.get("error")],
                "retries": sum(1 for g in gens if g.get("attempt", 0) > 0),
                "response_sha": [g.get("response_text_sha16") for g in gens if not g.get("error")],
                "prompt_sha": [g.get("contents_sha16") for g in gens],
                "rerank_ms": sum(r.get("ms", 0.0) for r in row.get("rerank_runs", [])),
                "search_ms": row.get("search_ms"),
            }
    out["complete_subset"] = out["done"] and sorted(out["queries"]) == sorted(subset)
    return out


def metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ranks = [r["final_rank"] for r in rows]
    hit = lambda k: sum(1 for x in ranks if x is not None and x <= k)  # noqa: E731
    mrr = sum(1.0 / x for x in ranks if x is not None and x <= 10) / n if n else 0.0
    return {"n": n, "hit1": hit(1), "hit5": hit(5), "hit10": hit(10), "mrr10": round(mrr, 4),
            "hit10_rate": round(hit(10) / n, 4) if n else None}


def hit10(r: Optional[Dict[str, Any]]) -> Optional[bool]:
    if r is None:
        return None
    return r["final_rank"] is not None and r["final_rank"] <= 10


def compare(x: Dict[str, Any], y: Dict[str, Any], qids: List[str]) -> Dict[str, Any]:
    flips, flips_fail, flips_ok, top10_diff, resp_diff, both_applied = [], [], [], [], [], 0
    for q in qids:
        rx, ry = x["queries"].get(q), y["queries"].get(q)
        if rx is None or ry is None:
            continue
        ok = rx["status"] == "applied" and ry["status"] == "applied"
        both_applied += ok
        if hit10(rx) != hit10(ry):
            flips.append(q)
            (flips_ok if ok else flips_fail).append(q)
        if rx["top10"] != ry["top10"]:
            top10_diff.append(q)
        if rx["response_sha"] != ry["response_sha"]:
            resp_diff.append(q)
    return {"pair": f"{x['name']} vs {y['name']}", "compared": sum(1 for q in qids if q in x["queries"] and q in y["queries"]),
            "both_applied": both_applied, "hit10_flips": flips, "hit10_flips_failure_related": flips_fail,
            "hit10_flips_both_applied": flips_ok, "top10_list_diff": top10_diff, "response_sha_diff": resp_diff,
            "rank_changes": {q: [x["queries"][q]["final_rank"], y["queries"][q]["final_rank"]] for q in qids
                             if q in x["queries"] and q in y["queries"]
                             and x["queries"][q]["final_rank"] != y["queries"][q]["final_rank"]}}


def classify(q: str, runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    r = {k: runs[k]["queries"].get(q) for k in RUNS}
    h = {k: hit10(v) for k, v in r.items()}
    if any(v is None for v in h.values()):
        cls = "incomplete"
    elif h["A1"] != h["A2"] or h["B1"] != h["B2"]:
        cls = "unstable"
    elif not h["A1"] and h["B1"]:
        cls = "stable_gain"
    elif h["A1"] and not h["B1"]:
        cls = "stable_loss"
    else:
        cls = "same_hit" if h["A1"] else "same_miss"
    statuses = {k: (v["status"] if v else None) for k, v in r.items()}
    return {"class": cls, "hit10": h, "statuses": statuses,
            "failure_touched": any(s != "applied" for s in statuses.values() if s is not None),
            "final_rank": {k: (v["final_rank"] if v else None) for k, v in r.items()},
            "cand_rank": {"A": r["A1"]["cand_rank"] if r["A1"] else None, "B": r["B1"]["cand_rank"] if r["B1"] else None}}


def decide(per: Dict[str, Dict[str, Any]], var: Dict[str, Any]) -> Dict[str, Any]:
    gains = sorted(q for q, v in per.items() if v["class"] == "stable_gain")
    losses = sorted(q for q, v in per.items() if v["class"] == "stable_loss")
    flips = len(var["A1_A2"]["hit10_flips"]) + len(var["B1_B2"]["hit10_flips"])
    touched = sorted(q for q, v in per.items() if v["failure_touched"])
    incomplete = sorted(q for q, v in per.items() if v["class"] == "incomplete")
    c1 = len(gains) - len(losses) >= 1
    c2 = len(losses) == 0
    c3 = len([q for q in gains if q != TITLE_EQUALS_WORK]) >= 1
    c4 = flips < len(gains) if gains else False
    # 5: 실패 영향 질의를 반대로 셌을 때 1~4 결론이 바뀌는가(이득·손실 판정에 걸린 실패 영향 질의가 있으면 바뀔 수 있다)
    swing = [q for q in touched if per[q]["class"] in ("stable_gain", "stable_loss", "unstable")]
    c5 = not swing and not incomplete
    criteria = {"1_net_stable_gain_ge_1": c1, "2_no_stable_loss": c2, "3_not_only_n096": c3,
                "4_repeat_flips_lt_stable_gains": c4, "5_failures_do_not_swing": c5,
                "6_human_labels": "미측정(라벨 0/22) — 결론에 명시"}
    if all([c1, c2, c3, c4, c5]):
        verdict = "full_ab_plan_recommended"
    elif not gains and c5:
        # 안정 이득이 없다(손실이 있으면 손실도 함께 적는다) — 효과 없음 또는 손해로 정리한다
        verdict = "not_recommended_no_stable_gain" + ("_with_losses" if losses else "")
    else:
        verdict = "hold"
    return {"stable_gains": gains, "stable_losses": losses, "repeat_hit10_flips": flips, "failure_touched": touched,
            "failure_swing_queries": swing, "incomplete": incomplete, "criteria": criteria, "verdict": verdict}


def main() -> int:
    subset_doc = json.loads((HERE / "subset/rerank_input_changed.json").read_text(encoding="utf-8"))
    subset = sorted(q for ids in subset_doc["query_ids_by_set"].values() for q in ids)
    gold = relevant_ids()
    runs = {k: load(v, subset, gold) for k, v in RUNS.items()}
    complete = {k: r.get("complete_subset", False) for k, r in runs.items()}
    summary: Dict[str, Any] = {"subset_count": len(subset), "runs_complete": complete, "runs": {}}
    for k, r in runs.items():
        rows = [r["queries"][q] for q in subset if q in r["queries"]]
        st: Dict[str, int] = {}
        for x in rows:
            st[x["status"]] = st.get(x["status"], 0) + 1
        ws: Dict[str, int] = {}
        for x in rows:
            ws[x["work_status"]] = ws.get(x["work_status"], 0) + 1
        summary["runs"][k] = {"name": r["name"], "done": r["done"], "conditions_status": r.get("conditions_status"),
                              "queries_recorded": len(rows), "metrics_subset_denominator": metrics(rows) if complete[k] else None,
                              "rerank_status": st, "work_status": ws,
                              "generate_calls": sum(x["generate_calls"] for x in rows),
                              "retries": sum(x["retries"] for x in rows),
                              "generate_errors": sum(len(x["generate_errors"]) for x in rows),
                              "rerank_ms_median": sorted(x["rerank_ms"] for x in rows)[len(rows) // 2] if rows else None,
                              "pre_rerank_top10_in_candidates": sum(1 for x in rows if x["cand_rank"] is not None and x["cand_rank"] <= 10),
                              "candidate_recall30": sum(1 for x in rows if x["cand_rank"] is not None)}
    if all(complete.values()):
        applied_all = [q for q in subset if all(runs[k]["queries"][q]["status"] == "applied" for k in RUNS)]
        summary["applied_all_four"] = applied_all
        summary["metrics_applied_all_four"] = {k: metrics([runs[k]["queries"][q] for q in applied_all]) for k in RUNS}
        var = {"A1_A2": compare(runs["A1"], runs["A2"], subset), "B1_B2": compare(runs["B1"], runs["B2"], subset),
               "A1_B1": compare(runs["A1"], runs["B1"], subset), "A2_B2": compare(runs["A2"], runs["B2"], subset)}
        summary["comparisons"] = var
        per = {q: classify(q, runs) for q in subset}
        summary["classification_counts"] = {}
        for v in per.values():
            summary["classification_counts"][v["class"]] = summary["classification_counts"].get(v["class"], 0) + 1
        summary["decision"] = decide(per, var)
        summary["cases"] = {
            "new_candidates": {q: per[q] for q in NEW_CANDIDATES},
            "existing_candidate_misses_A": sorted(q for q, v in per.items() if v["cand_rank"]["A"] is not None
                                                   and not v["hit10"]["A1"] and not v["hit10"]["A2"]),
            "existing_success_A": sorted(q for q, v in per.items() if v["hit10"]["A1"] and v["hit10"]["A2"]),
            "name_in_query": {q: per.get(q, "not in subset (input identical)") for q in NAME_IN_QUERY},
        }
    else:
        per = {}
        summary["note"] = "완료되지 않은 실행이 있어 비교·분류·판정을 하지 않았다(부분 실행을 이어 붙이지 않음)"
    out = HERE / "analysis"
    out.mkdir(exist_ok=True)
    (out / "live_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cols = ["set", "qid", "flags", "A_cand_rank", "B_cand_rank", "A1_rank", "B1_rank", "B2_rank", "A2_rank",
            "A1_status", "B1_status", "B2_status", "A2_status", "B1_work_status", "class", "failure_touched",
            "A1_A2_top10_same", "B1_B2_top10_same", "A1_A2_response_same", "B1_B2_response_same"]
    with (out / "per_query.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for q in subset:
            g = {k: runs[k]["queries"].get(q) for k in RUNS}
            flags = []
            if q in NEW_CANDIDATES:
                flags.append("new_candidate")
            if q == TITLE_EQUALS_WORK:
                flags.append("work_name_equals_song_title")
            if q in NAME_IN_QUERY:
                flags.append("name_in_query")
            same = lambda a, b, f: (g[a][f] == g[b][f]) if g[a] and g[b] else ""  # noqa: E731
            w.writerow({"set": (g["A1"] or g["B1"] or {}).get("set", ""), "qid": q, "flags": ";".join(flags),
                        "A_cand_rank": g["A1"]["cand_rank"] if g["A1"] else "", "B_cand_rank": g["B1"]["cand_rank"] if g["B1"] else "",
                        **{f"{k}_rank": (g[k]["final_rank"] if g[k] and g[k]["final_rank"] is not None else ("" if g[k] is None else ">30"))
                           for k in RUNS},
                        **{f"{k}_status": (g[k]["status"] if g[k] else "") for k in RUNS},
                        "B1_work_status": g["B1"]["work_status"] if g["B1"] else "",
                        "class": per[q]["class"] if per else "", "failure_touched": per[q]["failure_touched"] if per else "",
                        "A1_A2_top10_same": same("A1", "A2", "top10"), "B1_B2_top10_same": same("B1", "B2", "top10"),
                        "A1_A2_response_same": same("A1", "A2", "response_sha"), "B1_B2_response_same": same("B1", "B2", "response_sha")})
    print(json.dumps({k: summary[k] for k in summary if k in ("runs_complete", "classification_counts", "decision")},
                     ensure_ascii=False, indent=1))
    for k, v in summary["runs"].items():
        print(k, v["metrics_subset_denominator"], v["rerank_status"], v["work_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
