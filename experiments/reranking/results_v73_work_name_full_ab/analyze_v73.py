"""v73 dev 176 전체 실제 리랭킹 A/B 집계 — 모델·DB 호출 없음(저장된 실행 기록만 읽는다). v72 `analyze_v72.py`를 176 분모로 넓힌 판.

입력: `live/live_{off,work}_server_rep{1,2}`(DONE이 있고 176건이 모두 기록된 실행만 "완료") · `analysis/precheck_v73.json`(입력 변경 13건) ·
      dev 질의 세트(정답 id) · v72 `analysis/per_query.csv`(13건 참고 순위, 대조군 아님).
규칙은 RUN_INFO §4(실행 전 고정)를 그대로 코드로 옮겼다:
  - 주 지표: 완료 실행마다 dev 176 분모(실패·부분·예산 소진·정책 생략 포함, 평가기의 폴백 순서로 채점). 보조 = 네 실행 모두 applied
    (정책 생략은 네 실행 모두 policy_skipped면 포함)인 질의만.
  - 묶음: 전체 176 · 입력 변경 13 · 입력 동일 163 · 가사 19(v63 `LYRIC_QUERY_IDS`) · 나머지 157 · 세트 v06/v09/v10.
  - 질의별 Hit@10 분류(v72와 같음): 안정 이득 · 안정 손실 · 불안정(같은 구성 두 반복이 다름) · 같음. applied가 아닌 실행이 있으면 "실패 영향".
  - 변동: A1↔A2 · B1↔B2(같은 입력 반복), 입력 동일 163건의 A1↔B1 · A2↔B2(동시 대조군). 뒤집힘은 실패 관련 / 둘 다 정상으로 나눈다.
    응답 sha·최종 10위 목록 차이도 센다.
  - 판정 R1~R5(아래 `decide`). 반복을 독립 표본으로 합치지 않는다 — 짝 비교는 A1↔B1 · A2↔B2를 따로 낸다.
산출: analysis/live_summary.json · analysis/per_query.csv(질의 원문·작품명·모델 원출력 없음)

    venv/bin/python analyze_v73.py
    venv/bin/python analyze_v73.py --live-dir attempt3/live --out-dir attempt3/analysis --ledger attempt3/budget/ledger.jsonl \
        --condition "요청 60초 · 질의 120초를 허용한 평가기 품질(재시도 정정 지시 수정 사본)"   # 3차(§15)
    --ledger를 주면 실행별 정산 generate 지연(중앙값 · p95 · 최대 · 10초/20초 초과 수)을 **지연 보고 항목**으로 함께 낸다
    (품질 판정 R1~R5에는 쓰지 않는다).
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIVE_DIR = HERE / "live"  # main()의 --live-dir로 바꾼다(3차: attempt3/live)
SETS = ("v06", "v09", "v10")
RUNS = {"A1": "live_off_server_rep1", "B1": "live_work_server_rep1", "B2": "live_work_server_rep2", "A2": "live_off_server_rep2"}
NEW_CANDIDATES = ("n033", "n043", "n096")
TITLE_EQUALS_WORK = "n096"
V72_GAINS = ("c707", "n033", "n043", "n096")
APPLIED_SAME = ("n045", "q105", "t024")
LYRIC_QUERY_IDS = (
    "q209", "q211", "q213", "q219", "n007", "n009", "n013", "n017", "n039", "n064",
    "t003", "t016", "t018", "t025", "t032", "t060", "t063", "t105", "t110",
)
TARGETS = {"min_70pct": 124, "goal_80pct": 141}
OK_STATUSES = ("applied", "policy_skipped")


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


def load(name: str, gold: Dict[str, set]) -> Dict[str, Any]:
    d = LIVE_DIR / name
    out: Dict[str, Any] = {"name": name, "exists": d.exists(), "done": (d / "DONE").exists(), "queries": {}, "order": []}
    if not d.exists() or not (d / "run_conditions.json").exists():  # 진행 중(조건 기록 전)인 실행은 없는 것으로 본다
        return out
    out["conditions"] = json.loads((d / "run_conditions.json").read_text(encoding="utf-8"))
    for s in SETS:
        qf = d / f"{s}_dev/v73_queries.jsonl"
        if not qf.exists():
            continue
        for line in qf.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            q = row["query_id"]
            final = (row.get("final") or {}).get("ids") or []
            cands = [c["id"] for c in row.get("candidates") or []]
            gens = row.get("generates", [])
            out["order"].append(q)
            out["queries"][q] = {
                "set": s, "final_rank": first_rank(final, gold[q]), "cand_rank": first_rank(cands, gold[q]),
                "top10": final[:10], "status": query_status(row), "work_status": row.get("work_status") or "(none)",
                "generate_calls": len(gens), "generate_errors": [g.get("error") for g in gens if g.get("error")],
                "retries": sum(1 for g in gens if g.get("attempt", 0) > 0),
                "response_sha": [g.get("response_text_sha16") for g in gens if not g.get("error")],
                "rerank_ms": sum(r.get("ms", 0.0) for r in row.get("rerank_runs", [])),
                "search_ms": row.get("search_ms"),
            }
    out["complete"] = out["done"] and len(out["queries"]) == 176
    return out


def metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ranks = [r["final_rank"] for r in rows]
    hit = lambda k: sum(1 for x in ranks if x is not None and x <= k)  # noqa: E731
    mrr = sum(1.0 / x for x in ranks if x is not None and x <= 10) / n if n else 0.0
    return {"n": n, "hit1": hit(1), "hit5": hit(5), "hit10": hit(10), "mrr10": round(mrr, 4),
            "hit10_rate": round(hit(10) / n, 4) if n else None,
            "candidate_recall30": sum(1 for r in rows if r["cand_rank"] is not None)}


def hit10(r: Optional[Dict[str, Any]]) -> Optional[bool]:
    if r is None:
        return None
    return r["final_rank"] is not None and r["final_rank"] <= 10


def sign_test_two_sided(b: int, c: int) -> Optional[float]:
    """짝 지은 Hit@10의 불일치 쌍(b: x만 맞음, c: y만 맞음)에 대한 정확 이항 양측 검정(McNemar 정확판). 참고용."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


def compare(x: Dict[str, Any], y: Dict[str, Any], qids: List[str]) -> Dict[str, Any]:
    flips, flips_fail, flips_ok, top10_diff, resp_diff, both_ok = [], [], [], [], [], 0
    x_only, y_only = [], []
    for q in qids:
        rx, ry = x["queries"].get(q), y["queries"].get(q)
        if rx is None or ry is None:
            continue
        ok = rx["status"] in OK_STATUSES and ry["status"] in OK_STATUSES
        both_ok += ok
        if hit10(rx) != hit10(ry):
            flips.append(q)
            (flips_ok if ok else flips_fail).append(q)
            (x_only if hit10(rx) else y_only).append(q)
        if rx["top10"] != ry["top10"]:
            top10_diff.append(q)
        if rx["response_sha"] != ry["response_sha"]:
            resp_diff.append(q)
    with_call = [q for q in qids if q in x["queries"] and q in y["queries"]
                 and x["queries"][q]["generate_calls"] and y["queries"][q]["generate_calls"]]
    return {"pair": f"{x['name']} vs {y['name']}", "compared": sum(1 for q in qids if q in x["queries"] and q in y["queries"]),
            "both_ok": both_ok, "hit10_flips": flips, "hit10_flips_failure_related": flips_fail,
            "hit10_flips_both_ok": flips_ok, "x_only_hit10": x_only, "y_only_hit10": y_only,
            "sign_test_p_two_sided": sign_test_two_sided(len(x_only), len(y_only)),
            "top10_list_diff": top10_diff, "queries_with_model_call_both": len(with_call),
            "response_sha_diff": [q for q in resp_diff if q in with_call],
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
            "failure_touched": any(s not in OK_STATUSES for s in statuses.values() if s is not None),
            "final_rank": {k: (v["final_rank"] if v else None) for k, v in r.items()},
            "cand_rank": {"A": r["A1"]["cand_rank"] if r["A1"] else None, "B": r["B1"]["cand_rank"] if r["B1"] else None}}


def decide(per: Dict[str, Dict[str, Any]], changed: List[str], same: List[str], m: Dict[str, Any]) -> Dict[str, Any]:
    """RUN_INFO §4-3(실행 전 고정)."""
    def of(qs, cls):
        return sorted(q for q in qs if per[q]["class"] == cls)
    gains, losses = of(changed, "stable_gain"), of(changed, "stable_loss")
    ctrl_pattern = of(same, "stable_gain") + of(same, "stable_loss")
    ctrl_rate = len(ctrl_pattern) / len(same) if same else 0.0
    touched = sorted(q for q in changed if per[q]["failure_touched"]
                     and per[q]["class"] in ("stable_gain", "stable_loss", "unstable"))
    r1 = len(gains) >= 1 and not losses
    r2 = len(gains) > ctrl_rate * len(changed)
    r3 = len([q for q in gains if q != TITLE_EQUALS_WORK]) >= 1
    r4 = not touched and not of(changed, "incomplete")
    r5 = m["B1"]["hit10"] >= m["A1"]["hit10"] and m["B2"]["hit10"] >= m["A2"]["hit10"]
    criteria = {"R1_changed13_stable_gain_ge1_and_no_stable_loss": r1,
                "R2_gains_exceed_control_pattern_rate": r2,
                "R3_not_only_n096": r3,
                "R4_failures_do_not_swing_changed13": r4,
                "R5_full176_hit10_B_ge_A_in_both_pairs": r5}
    if all(criteria.values()):
        verdict = "reproduced_on_full_dev"
    elif not gains and r4:
        verdict = "not_reproduced_no_stable_gain" + ("_with_losses" if losses else "")
    else:
        verdict = "hold"
    return {"stable_gains_changed13": gains, "stable_losses_changed13": losses,
            "unstable_changed13": of(changed, "unstable"),
            "control163_stable_pattern_queries": ctrl_pattern, "control163_stable_pattern_rate": round(ctrl_rate, 4),
            "control_expected_false_pattern_in_13": round(ctrl_rate * len(changed), 3),
            "stable_losses_control163": of(same, "stable_loss"), "stable_gains_control163": of(same, "stable_gain"),
            "unstable_control163": of(same, "unstable"),
            "failure_swing_changed13": touched, "criteria": criteria, "verdict": verdict,
            "labels": "사람 작품 라벨 미반영(이 세션에 전달된 결과 없음) — 해석 정확도 미측정"}


def v72_reference() -> Dict[str, Dict[str, str]]:
    path = Path("/private/tmp/vague-finder-v72-work-name-rerank/experiments/reranking/results_v72_work_name_rerank_diagnosis/analysis/per_query.csv")
    alt = ROOT / "experiments/reranking/results_v72_work_name_rerank_diagnosis/analysis/per_query.csv"
    p = alt if alt.exists() else path
    with p.open(encoding="utf-8") as fh:
        return {r["qid"]: r for r in csv.DictReader(fh)}


def latency_report(ledger: Path) -> Dict[str, Any]:
    """실행별 정산 generate 지연(지연 보고 항목, 품질 판정과 무관). 실패 시도는 따로 센다."""
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: Dict[str, Any] = {}
    for k, name in RUNS.items():
        ok = sorted(r["latency_ms"] for r in rows if r["event"] == "settle" and r.get("kind") == "generate" and r.get("run") == name)
        fails = [r for r in rows if r["event"] == "fail" and r.get("run") == name]
        if not ok and not fails:
            continue
        out[k] = {"settled_generate": len(ok), "failed_attempts": len(fails),
                  "fail_kinds": sorted({str(r.get("error_code") or r.get("error")) for r in fails}),
                  "median_ms": round(statistics.median(ok)) if ok else None,
                  "p95_ms": round(ok[max(0, math.ceil(0.95 * len(ok)) - 1)]) if ok else None,
                  "max_ms": round(ok[-1]) if ok else None,
                  "over_10s": sum(1 for x in ok if x > 10000), "over_20s": sum(1 for x in ok if x > 20000)}
    out["halts"] = [r["reason"] for r in rows if r["event"] == "halt"]
    return out


def main() -> int:
    import argparse

    global LIVE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", type=Path, default=HERE / "live")
    ap.add_argument("--out-dir", type=Path, default=HERE / "analysis")
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--condition", default="")
    args = ap.parse_args()
    LIVE_DIR = args.live_dir if args.live_dir.is_absolute() else HERE / args.live_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = json.loads((HERE / "analysis/precheck_v73.json").read_text(encoding="utf-8"))
    changed = sorted(pre["input_changed"])
    gold = relevant_ids()
    runs = {k: load(v, gold) for k, v in RUNS.items()}
    complete = {k: r.get("complete", False) for k, r in runs.items()}
    order = next((r["order"] for r in runs.values() if r.get("complete")), [])
    same = sorted(q for q in order if q not in changed)
    groups = {"all176": order, "changed13": changed, "same163": same,
              "lyrics19": [q for q in order if q in LYRIC_QUERY_IDS], "other157": [q for q in order if q not in LYRIC_QUERY_IDS],
              **{s: [q for q in order if runs["A1"]["queries"].get(q, {}).get("set") == s] for s in SETS}}
    summary: Dict[str, Any] = {"evaluation_condition": args.condition or "기본(호출 20초 · 질의 30초, main 코드)",
                               "live_dir": str(LIVE_DIR.relative_to(HERE)), "runs_complete": complete, "group_sizes": {g: len(v) for g, v in groups.items()}, "runs": {}}
    for k, r in runs.items():
        rows = [r["queries"][q] for q in order if q in r["queries"]]
        st: Dict[str, int] = {}
        ws: Dict[str, int] = {}
        for x in rows:
            st[x["status"]] = st.get(x["status"], 0) + 1
            ws[x["work_status"]] = ws.get(x["work_status"], 0) + 1
        rer = [x["rerank_ms"] for x in rows if x["generate_calls"]]
        cond = r.get("conditions") or {}
        summary["runs"][k] = {
            "name": r["name"], "done": r["done"], "complete": complete[k], "conditions_status": cond.get("status"),
            "error": cond.get("error"), "queries_recorded": len(rows),
            "metrics": {g: metrics([r["queries"][q] for q in qs if q in r["queries"]]) for g, qs in groups.items()} if complete[k] else None,
            "rerank_status": st, "work_status": ws,
            "generate_calls": sum(x["generate_calls"] for x in rows), "retries": sum(x["retries"] for x in rows),
            "generate_errors": sum(len(x["generate_errors"]) for x in rows),
            "rerank_ms_median_called": round(statistics.median(rer), 1) if rer else None,
            "rerank_ms_p95_called": round(sorted(rer)[int(0.95 * (len(rer) - 1))], 1) if rer else None,
            "search_ms_median": round(statistics.median([x["search_ms"] for x in rows if x["search_ms"] is not None]), 1) if rows else None,
            "warmup": cond.get("warmup"), "total_seconds": cond.get("total_seconds"),
        }
    if all(complete.values()):
        m = {k: summary["runs"][k]["metrics"]["all176"] for k in RUNS}
        summary["targets"] = {k: {t: {"target": v, "hit10": m[k]["hit10"], "remaining": max(0, v - m[k]["hit10"])}
                                  for t, v in TARGETS.items()} for k in RUNS}
        ok_all = [q for q in order if all(runs[k]["queries"][q]["status"] in OK_STATUSES for k in RUNS)]
        summary["ok_all_four_count"] = len(ok_all)
        summary["metrics_ok_all_four"] = {k: metrics([runs[k]["queries"][q] for q in ok_all]) for k in RUNS}
        cmp = {}
        for name, (x, y) in {"A1_A2": ("A1", "A2"), "B1_B2": ("B1", "B2"), "A1_B1": ("A1", "B1"), "A2_B2": ("A2", "B2")}.items():
            cmp[name] = {g: compare(runs[x], runs[y], groups[g]) for g in ("all176", "changed13", "same163", "lyrics19")}
        summary["comparisons"] = cmp
        per = {q: classify(q, runs) for q in order}
        counts: Dict[str, Dict[str, int]] = {}
        for g in ("all176", "changed13", "same163", "lyrics19"):
            counts[g] = {}
            for q in groups[g]:
                counts[g][per[q]["class"]] = counts[g].get(per[q]["class"], 0) + 1
        summary["classification_counts"] = counts
        summary["decision"] = decide(per, changed, same, m)
        # n096 기여 분리: n096을 뺀 175건 Hit@10
        summary["without_n096"] = {k: metrics([runs[k]["queries"][q] for q in order if q != TITLE_EQUALS_WORK]) for k in RUNS}
        v72 = v72_reference()
        summary["cases"] = {
            "v72_gains": {q: {**per[q], "v72_B_rank": v72.get(q, {}).get("B1_rank"), "v72_A_rank": v72.get(q, {}).get("A1_rank")}
                          for q in V72_GAINS},
            "new_candidates_in_top10_B": {q: [per[q]["hit10"]["B1"], per[q]["hit10"]["B2"]] for q in NEW_CANDIDATES},
            "changed13_vs_v72_rank": {q: {"v72": [v72.get(q, {}).get(c) for c in ("A1_rank", "B1_rank", "B2_rank", "A2_rank")],
                                          "v73": [per[q]["final_rank"][c] for c in ("A1", "B1", "B2", "A2")]} for q in changed},
            "applied_but_input_identical": {q: per[q] for q in APPLIED_SAME},
            "failure_touched_all": sorted(q for q in order if per[q]["failure_touched"]),
        }
        # 실패가 지표에 준 영향: 실행별로 정상이 아닌 질의의 Hit@10과, 같은 질의의 다른 실행 결과
        fail_effect = {}
        for k in RUNS:
            bad = [q for q in order if runs[k]["queries"][q]["status"] not in OK_STATUSES]
            fail_effect[k] = {"queries": bad, "hit10_in_this_run": sum(1 for q in bad if hit10(runs[k]["queries"][q])),
                              "hit10_in_other_runs_same_config": sum(
                                  1 for q in bad if hit10(runs[{"A1": "A2", "A2": "A1", "B1": "B2", "B2": "B1"}[k]]["queries"][q]))}
        summary["failure_effect"] = fail_effect
    else:
        per = {}
        summary["note"] = "완료되지 않은 실행이 있어 비교·분류·판정을 하지 않았다(부분 실행을 이어 붙이지 않음)"
    if args.ledger:
        summary["latency_report"] = latency_report(args.ledger if args.ledger.is_absolute() else HERE / args.ledger)
    out = out_dir
    (out / "live_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cols = ["set", "qid", "group", "flags", "A_cand_rank", "B_cand_rank", "A1_rank", "B1_rank", "B2_rank", "A2_rank",
            "A1_status", "B1_status", "B2_status", "A2_status", "B_work_status", "class", "failure_touched",
            "A1_A2_top10_same", "B1_B2_top10_same", "A1_B1_top10_same", "A2_B2_top10_same",
            "A1_A2_response_same", "B1_B2_response_same", "A1_B1_response_same", "A2_B2_response_same"]
    with (out / "per_query.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for q in order:
            g = {k: runs[k]["queries"].get(q) for k in RUNS}
            flags = [f for f, cond in (("new_candidate", q in NEW_CANDIDATES), ("work_name_equals_song_title", q == TITLE_EQUALS_WORK),
                                       ("lyrics19", q in LYRIC_QUERY_IDS), ("applied_input_identical", q in APPLIED_SAME)) if cond]
            same_f = lambda a, b, f: (g[a][f] == g[b][f]) if g[a] and g[b] else ""  # noqa: E731
            rank = lambda k: (g[k]["final_rank"] if g[k] and g[k]["final_rank"] is not None else ("" if g[k] is None else ">30"))  # noqa: E731
            w.writerow({"set": (g["A1"] or g["B1"] or {}).get("set", ""), "qid": q,
                        "group": "changed13" if q in changed else "same163", "flags": ";".join(flags),
                        "A_cand_rank": (g["A1"]["cand_rank"] or ">30") if g["A1"] else "",
                        "B_cand_rank": (g["B1"]["cand_rank"] or ">30") if g["B1"] else "",
                        **{f"{k}_rank": rank(k) for k in RUNS},
                        **{f"{k}_status": (g[k]["status"] if g[k] else "") for k in RUNS},
                        "B_work_status": g["B1"]["work_status"] if g["B1"] else "",
                        "class": per[q]["class"] if per else "", "failure_touched": per[q]["failure_touched"] if per else "",
                        **{f"{a}_{b}_top10_same": same_f(a, b, "top10") for a, b in (("A1", "A2"), ("B1", "B2"), ("A1", "B1"), ("A2", "B2"))},
                        **{f"{a}_{b}_response_same": same_f(a, b, "response_sha") for a, b in (("A1", "A2"), ("B1", "B2"), ("A1", "B1"), ("A2", "B2"))}})
    print(json.dumps({k: summary.get(k) for k in ("runs_complete", "classification_counts", "decision", "targets")},
                     ensure_ascii=False, indent=1))
    for k, v in summary["runs"].items():
        print(k, (v["metrics"] or {}).get("all176"), v["rerank_status"], v["work_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
