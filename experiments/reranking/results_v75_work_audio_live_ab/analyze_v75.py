"""v75 dev 176 전체 실제 리랭킹 A/B 집계 — 작품명 해석 ON 고정, A = 소리 경로 ON(work_on), B = 소리 경로 생략(work_skip).
모델·DB 호출 없음(저장된 실행 기록만 읽는다). v73 `analyze_v73.py`(PR #88 병합) 사본을 v75 묶음·판정으로 바꾼 판.

입력: `live/live_work_{on,skip}_server_rep{1,2}`(DONE이 있고 176건이 모두 기록된 실행만 "완료") · `analysis/precheck_v75.json`
      (입력 묶음·기존 성공·관심 사례 — 실행 전 고정) · dev 질의 세트(정답 id).
규칙은 RUN_INFO §4(실행 전 고정)를 그대로 코드로 옮겼다:
  - 주 지표: 완료 실행마다 Hit@10, dev 176 분모(실패·부분·예산 소진·정책 생략 포함, 평가기의 폴백 순서로 채점).
    보조: Hit@1 · Hit@5 · MRR@10 · 후보 회수@30 · 리랭킹 상태. 보조 2 = 네 실행 모두 정상인 질의만의 값.
  - 묶음: 전체 176 · 프롬프트 변화 109 · 점수만 변화 5 · 입력 동일 62 · 기존 성공(v73 B) 121 중 프롬프트 변화 72 ·
    가사 19(v63 `LYRIC_QUERY_IDS`) · 나머지 157 · 세트 v06/v09/v10.
  - 질의별 Hit@10 분류: 안정 이득(A 두 번 다 10위 밖, B 두 번 다 10위 안) · 안정 손실(반대) · 불안정(같은 구성 두 반복이 다름) · 같음.
    정상(applied·policy_skipped)이 아닌 실행이 하나라도 있으면 "실패 영향" 표시(주 지표에는 그대로 넣는다).
  - 변동: A1↔A2 · B1↔B2(같은 입력 반복), 입력 동일 62건의 A1↔B1 · A2↔B2(동시 대조군 — 전송 프롬프트가 같다).
  - 짝 비교는 A1↔B1 · A2↔B2를 따로 낸다. 반복을 352·704개 독립 질의처럼 합치지 않는다. 부호 검정 p는 참고만.
  - 판정 S1~S4(아래 `decide`).
산출: analysis/live_summary.json · analysis/per_query.csv(질의 원문·곡 제목·모델 원출력 없음)

    venv/bin/python analyze_v75.py --ledger budget/ledger.jsonl
    --ledger를 주면 실행별 정산 generate 지연(중앙값 · p95 · 최대 · 10초/20초 초과 수)을 **지연 보고 항목**으로 함께 낸다(판정에 쓰지 않음).
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
RUNS = {"A1": "live_work_on_server_rep1", "B1": "live_work_skip_server_rep1", "B2": "live_work_skip_server_rep2",
        "A2": "live_work_on_server_rep2"}
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
    if not out["done"]:  # 미완료 실행은 지표·비교에 쓰지 않는다(부분 실행을 이어 붙이지 않음)
        out["complete"] = False
    for s in SETS:
        qf = d / f"{s}_dev/v75_queries.jsonl"
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


def decide(per: Dict[str, Dict[str, Any]], changed: List[str], same: List[str], m: Dict[str, Any],
           fail_rate: Dict[str, float]) -> Dict[str, Any]:
    """RUN_INFO §4-3(실행 전 고정). changed = 전송 프롬프트 변화 109, same = 입력 동일 62(동시 대조군)."""
    def of(qs, cls):
        return sorted(q for q in qs if per[q]["class"] == cls)
    gains, losses = of(changed, "stable_gain"), of(changed, "stable_loss")
    ctrl_pattern = of(same, "stable_gain") + of(same, "stable_loss")
    ctrl_rate = len(ctrl_pattern) / len(same) if same else 0.0
    touched = sorted(q for q in per if per[q]["failure_touched"]
                     and per[q]["class"] in ("stable_gain", "stable_loss", "unstable"))
    s1 = m["B1"]["hit10"] > m["A1"]["hit10"] and m["B2"]["hit10"] > m["A2"]["hit10"]
    s2 = len(gains) > len(losses) and (len(gains) - len(losses)) > ctrl_rate * len(changed)
    s3 = m["B1"]["mrr10"] >= m["A1"]["mrr10"] and m["B2"]["mrr10"] >= m["A2"]["mrr10"]
    s4 = not touched and fail_rate["B1"] <= fail_rate["A1"] and fail_rate["B2"] <= fail_rate["A2"]
    criteria = {"S1_hit10_B_gt_A_in_both_pairs": s1,
                "S2_stable_gains_exceed_losses_beyond_control_rate": s2,
                "S3_mrr10_B_ge_A_in_both_pairs": s3,
                "S4_failures_do_not_swing_and_B_failure_rate_not_higher": s4}
    if all(criteria.values()):
        verdict = "B_better_on_dev_under_this_condition"
    elif m["B1"]["hit10"] < m["A1"]["hit10"] and m["B2"]["hit10"] < m["A2"]["hit10"]:
        verdict = "B_worse_on_dev_under_this_condition"
    else:
        verdict = "hold_mixed"
    return {"stable_gains_prompt_changed": gains, "stable_losses_prompt_changed": losses,
            "unstable_prompt_changed": of(changed, "unstable"),
            "control62_stable_pattern_queries": ctrl_pattern, "control62_stable_pattern_rate": round(ctrl_rate, 4),
            "control_expected_false_pattern_in_changed": round(ctrl_rate * len(changed), 3),
            "unstable_control62": of(same, "unstable"),
            "failure_swing_any": touched, "criteria": criteria, "verdict": verdict,
            "scope": "dev 176(반복 사용 개발 자료) · 고정 해석 캐시 · 예열된 평가기 · 호출 60초/질의 120초. 운영 채택·독립 검증 아님"}


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
    ap.add_argument("--condition", default="고정 해석 캐시(v71 반복 1) · 예열된 평가기 · 호출 60초 · 질의 120초(평가 전용) · main 81965f6")
    args = ap.parse_args()
    LIVE_DIR = args.live_dir if args.live_dir.is_absolute() else HERE / args.live_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = json.loads((HERE / "analysis/precheck_v75.json").read_text(encoding="utf-8"))
    if not pre.get("pass"):
        raise SystemExit("사전 검증이 통과하지 않은 기록이다")
    ids = pre["input_group_ids"]
    changed, score_only, same = sorted(ids["prompt_changed"]), sorted(ids["score_only"]), sorted(ids["input_same"])
    base_hits = set(pre["base_hits_v73_B"])
    watch = pre["watch_cases(실행 전 고정)"]
    gold = relevant_ids()
    runs = {k: load(v, gold) for k, v in RUNS.items()}
    complete = {k: r.get("complete", False) for k, r in runs.items()}
    order = next((r["order"] for r in runs.values() if r.get("complete")), [])
    groups = {"all176": order, "prompt_changed109": changed, "score_only5": score_only, "input_same62": same,
              "base_hits121": [q for q in order if q in base_hits],
              "base_hits_prompt_changed72": [q for q in changed if q in base_hits],
              "base_fails55": [q for q in order if q not in base_hits],
              "lyrics19": [q for q in order if q in LYRIC_QUERY_IDS], "other157": [q for q in order if q not in LYRIC_QUERY_IDS],
              **{s: [q for q in order if runs["A1"]["queries"].get(q, {}).get("set") == s] for s in SETS}}
    summary: Dict[str, Any] = {"evaluation_condition": args.condition,
                               "live_dir": str(LIVE_DIR.relative_to(HERE)) if LIVE_DIR.is_relative_to(HERE) else "(결과 폴더 밖 — 자체 검사)", "runs_complete": complete,
                               "group_sizes": {g: len(v) for g, v in groups.items()}, "runs": {}}
    for k, r in runs.items():
        rows = [r["queries"][q] for q in order if q in r["queries"]] if order else list(r["queries"].values())
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
            "not_ok_queries": sum(1 for x in rows if x["status"] not in OK_STATUSES),
            "generate_calls": sum(x["generate_calls"] for x in rows), "retries": sum(x["retries"] for x in rows),
            "generate_errors": sum(len(x["generate_errors"]) for x in rows),
            "rerank_ms_median_called": round(statistics.median(rer), 1) if rer else None,
            "rerank_ms_p95_called": round(sorted(rer)[int(0.95 * (len(rer) - 1))], 1) if rer else None,
            "search_ms_median": round(statistics.median([x["search_ms"] for x in rows if x["search_ms"] is not None]), 1) if rows else None,
            "warmup": cond.get("warmup"), "total_seconds": cond.get("total_seconds"),
        }
    per: Dict[str, Dict[str, Any]] = {}
    if all(complete.values()):
        m = {k: summary["runs"][k]["metrics"]["all176"] for k in RUNS}
        summary["targets"] = {k: {t: {"target": v, "hit10": m[k]["hit10"], "remaining": max(0, v - m[k]["hit10"]),
                                      "reached": m[k]["hit10"] >= v} for t, v in TARGETS.items()} for k in RUNS}
        ok_all = [q for q in order if all(runs[k]["queries"][q]["status"] in OK_STATUSES for k in RUNS)]
        summary["ok_all_four_count"] = len(ok_all)
        summary["metrics_ok_all_four"] = {k: metrics([runs[k]["queries"][q] for q in ok_all]) for k in RUNS}
        cmp = {}
        for name, (x, y) in {"A1_A2": ("A1", "A2"), "B1_B2": ("B1", "B2"), "A1_B1": ("A1", "B1"), "A2_B2": ("A2", "B2")}.items():
            cmp[name] = {g: compare(runs[x], runs[y], groups[g])
                         for g in ("all176", "prompt_changed109", "score_only5", "input_same62", "base_hits121", "lyrics19")}
        summary["comparisons"] = cmp
        per = {q: classify(q, runs) for q in order}
        counts: Dict[str, Dict[str, int]] = {}
        for g in ("all176", "prompt_changed109", "score_only5", "input_same62", "base_hits121", "base_fails55", "lyrics19"):
            counts[g] = {}
            for q in groups[g]:
                counts[g][per[q]["class"]] = counts[g].get(per[q]["class"], 0) + 1
        summary["classification_counts"] = counts
        fail_rate = {k: summary["runs"][k]["not_ok_queries"] / 176 for k in RUNS}
        summary["decision"] = decide(per, changed, same, m, fail_rate)
        summary["cases"] = {name: {q: per[q] for q in qs if q in per} for name, qs in watch.items()}
        summary["cases"]["lyrics19"] = {q: per[q] for q in groups["lyrics19"]}
        summary["stable_gains_all"] = sorted(q for q in order if per[q]["class"] == "stable_gain")
        summary["stable_losses_all"] = sorted(q for q in order if per[q]["class"] == "stable_loss")
        summary["unstable_all"] = sorted(q for q in order if per[q]["class"] == "unstable")
        summary["failure_touched_all"] = sorted(q for q in order if per[q]["failure_touched"])
        fail_effect = {}
        for k in RUNS:
            bad = [q for q in order if runs[k]["queries"][q]["status"] not in OK_STATUSES]
            fail_effect[k] = {"queries": bad, "hit10_in_this_run": sum(1 for q in bad if hit10(runs[k]["queries"][q])),
                              "hit10_in_other_runs_same_config": sum(
                                  1 for q in bad if hit10(runs[{"A1": "A2", "A2": "A1", "B1": "B2", "B2": "B1"}[k]]["queries"][q]))}
        summary["failure_effect"] = fail_effect
    else:
        summary["note"] = "완료되지 않은 실행이 있어 비교·분류·판정을 하지 않았다(부분 실행을 이어 붙이지 않음)"
    if args.ledger:
        summary["latency_report"] = latency_report(args.ledger if args.ledger.is_absolute() else HERE / args.ledger)
    (out_dir / "live_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cols = ["set", "qid", "group", "base_hit_v73B", "flags", "A_cand_rank", "B_cand_rank", "A1_rank", "B1_rank", "B2_rank", "A2_rank",
            "A1_status", "B1_status", "B2_status", "A2_status", "work_status", "class", "failure_touched",
            "A1_A2_top10_same", "B1_B2_top10_same", "A1_B1_top10_same", "A2_B2_top10_same",
            "A1_A2_response_same", "B1_B2_response_same", "A1_B1_response_same", "A2_B2_response_same"]
    if order:
        with (out_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for q in order:
                g = {k: runs[k]["queries"].get(q) for k in RUNS}
                flags = [name for name, qs in watch.items() if q in qs] + (["lyrics19"] if q in LYRIC_QUERY_IDS else [])
                same_f = lambda a, b, f: (g[a][f] == g[b][f]) if g[a] and g[b] else ""  # noqa: E731
                rank = lambda k: (g[k]["final_rank"] if g[k] and g[k]["final_rank"] is not None else ("" if g[k] is None else ">30"))  # noqa: E731
                grp = "prompt_changed109" if q in changed else ("score_only5" if q in score_only else "input_same62")
                w.writerow({"set": (g["A1"] or g["B1"] or {}).get("set", ""), "qid": q, "group": grp,
                            "base_hit_v73B": int(q in base_hits), "flags": ";".join(flags),
                            "A_cand_rank": (g["A1"]["cand_rank"] or ">30") if g["A1"] else "",
                            "B_cand_rank": (g["B1"]["cand_rank"] or ">30") if g["B1"] else "",
                            **{f"{k}_rank": rank(k) for k in RUNS},
                            **{f"{k}_status": (g[k]["status"] if g[k] else "") for k in RUNS},
                            "work_status": g["A1"]["work_status"] if g["A1"] else "",
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
