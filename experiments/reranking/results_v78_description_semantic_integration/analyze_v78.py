"""v78 오프라인 후보 A/B 집계 — 모델·DB 호출 없음. 검색이 끝난 기록만 읽는다(정답 라벨은 여기서 처음 붙인다).

입력: 네 실행(`runs/sem_{off,on}_server_rep{1,2}`)의 `v78_queries.jsonl`·`v78_qdrant_calls.jsonl` · v76 질의별 표
(`results_v76_description_evidence_recall/analysis/per_query.csv`의 `v75A_hit10` — 실제 Hit@10 기준 기존 성공 121) ·
평가 질의 CSV(정답 id) · 분석 캐시(search_text — 임베딩 입력 지문 대조용) · v77 R1 순위(`--v77-rankings`, 실행 중 보조 경로 결과 대조).

가짜 리랭커의 최종 순서로 계산한 Hit·MRR은 만들지 않는다. 후보 회수@30(엄격 정답 하나라도)·정답 첫 후보 순위만 센다.

    <venv>/bin/python analyze_v78.py --v77-rankings <v77>/runs/R1/rankings.jsonl.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from compare_inputs import first_prompts  # noqa: E402

SETS = ("v06", "v09", "v10")
EXPECTED = {"v06": 57, "v09": 59, "v10": 60}
RUNS = {"A1": "sem_off_server_rep1", "B1": "sem_on_server_rep1", "B2": "sem_on_server_rep2", "A2": "sem_off_server_rep2"}
REACHED_V77 = ("n040", "c607", "n011", "t031", "c602", "n072", "m402")
# v77 설계 때 정답 곡 자료를 본 질의(v77 RUN_INFO) · v76 R18 — 표시만 한다
DESIGN_SEEN_V77 = {"q115", "q117", "m302", "m303", "m403", "c602", "c603", "t038", "n046", "n064", "t009", "t060", "t073",
                   "t075", "t079", "t080", "t092", "t103", "n011", "t100", "q110"}
V76_PER_QUERY = ROOT / "experiments/reranking/results_v76_description_evidence_recall/analysis/per_query.csv"
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_rows(run: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    run_dir = HERE / "runs" / run
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"완료되지 않은 실행: {run}")
    for s in SETS:
        rows = [json.loads(line) for line in (run_dir / f"{s}_dev" / "v78_queries.jsonl").open(encoding="utf-8")]
        rows = [r for r in rows if r["query_id"] != "_none"]
        if len(rows) != EXPECTED[s]:
            raise SystemExit(f"{run} {s}: 기록 {len(rows)}건 ≠ {EXPECTED[s]}")
        calls = [json.loads(line) for line in (run_dir / f"{s}_dev" / "v78_qdrant_calls.jsonl").open(encoding="utf-8")]
        by_q: Dict[str, int] = Counter(c["query_id"] for c in calls)
        for r in rows:
            r["set"] = s
            r["n_qdrant"] = by_q.get(r["query_id"], 0)
            out[r["query_id"]] = r
    return out


def gold() -> Dict[str, set]:
    out: Dict[str, set] = {}
    for s in SETS:
        with (ROOT / f"experiments/reranking/eval_queries_{s}.csv").open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["split"].strip() == "dev":
                    out[r["query_id"].strip()] = {x.strip() for x in r["relevant_ids"].split("|") if x.strip()}
    return out


def search_texts() -> Dict[str, str]:
    sys.path.insert(0, str(ROOT))
    from src.retrieval.analysis_cache import load_cache

    out = {}
    for s in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{s}_dev_vertex.json")
        for qid in cache.entries:
            out[qid] = cache.get(qid, cache.entries[qid]["query"]).search_text
    return out


def explain_rows(run_key: str) -> Dict[str, Dict[str, Any]]:
    """평가기 explain에서 질의별 작품명 보조 기록만(곡 id). 질의 원문 필드는 읽지 않는다."""
    out: Dict[str, Dict[str, Any]] = {}
    for s in SETS:
        path = HERE / "runs" / RUNS[run_key] / f"{s}_dev" / "search_eval_dev_explain.jsonl"
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            out[r["query_id"]] = {"work_resolution": r.get("work_resolution")}
    return out


def first_rank(ids: List[str], g: set) -> Optional[int]:
    for i, sid in enumerate(ids, start=1):
        if sid in g:
            return i
    return None


def med(xs: List[float]) -> Optional[float]:
    return round(statistics.median(xs), 3) if xs else None


def p90(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(len(s) * 0.9))], 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v77-rankings", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "analysis")
    args = ap.parse_args()
    runs = {k: load_rows(v) for k, v in RUNS.items()}
    G = gold()
    texts = search_texts()
    with V76_PER_QUERY.open(encoding="utf-8") as fh:
        hit10 = {r["qid"]: r["v75A_hit10"] == "True" for r in csv.DictReader(fh)}
    if sum(hit10.values()) != 121 or len(hit10) != 176:
        raise SystemExit("v75 A 성공 기준이 121/176이 아니다")
    v77: Dict[str, Any] = {}
    with gzip.open(args.v77_rankings, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                v77[r["qid"]] = r

    A, B = runs["A1"], runs["B1"]
    per: List[Dict[str, Any]] = []
    entered_occ: List[str] = []
    displaced_occ: List[str] = []
    group_top10: Counter = Counter()
    group_entered: Counter = Counter()
    group_rep: Dict[str, str] = {}
    aux_in_A_moves: List[int] = []
    final_new_occ: List[str] = []
    final_out_occ: List[str] = []
    final_new_from_aux = final_new_gold = 0
    sem_entered_not_final: List[str] = []
    sem_displaced_by_work: List[str] = []
    explain_b = explain_rows("B1")
    in_situ_same = in_situ_total = 0
    in_situ_mismatch: List[str] = []
    for qid in sorted(A, key=lambda q: (A[q]["set"], q)):
        a, b, g = A[qid], B[qid], G[qid]
        a_ids = [c["id"] for c in a["candidates"]]
        b_ids = [c["id"] for c in b["candidates"]]
        a_rank, b_rank = first_rank(a_ids, g), first_rank(b_ids, g)
        sem = b.get("sem") or {}
        status = b.get("sem_status") or "(none)"
        aux = sem.get("aux_ids") or []
        entered = sem.get("entered_ids") or []
        displaced = sem.get("displaced_ids") or []
        entered_occ += entered
        displaced_occ += displaced
        groups = sem.get("groups") or []
        for gr in groups:
            group_top10[gr["group"]] += 1
            if gr["representative"]:
                group_rep[gr["group"]] = gr["representative"]
                if gr["representative"] in entered:
                    group_entered[gr["group"]] += 1
        # 최종 후보 30 수준(작품명 보조·보정·절단까지 지난 뒤)의 진입·이탈
        new_final = [sid for sid in b_ids if sid not in a_ids]
        out_final = [sid for sid in a_ids if sid not in b_ids]
        final_new_occ += new_final
        final_out_occ += out_final
        final_new_from_aux += sum(1 for sid in new_final if sid in aux)
        final_new_gold += sum(1 for sid in new_final if sid in g)
        sem_entered_not_final += [f"{qid}:{sid}" for sid in entered if sid not in b_ids]
        work_disp = set(((explain_b.get(qid) or {}).get("work_resolution") or {}).get("displaced_ids") or [])
        sem_displaced_by_work += [f"{qid}:{sid}" for sid in entered if sid in work_disp]
        # 이미 A 후보였던 보조 곡의 순위 이동(B 후보 안에 남은 것만)
        for sid in aux:
            if sid in a_ids and sid in b_ids:
                aux_in_A_moves.append(a_ids.index(sid) - b_ids.index(sid))
        # 실행 중 보조 경로 결과 ↔ v77 R1(위 10묶음 · 근거 청크 · 점수 6자리 · 대표곡·규칙)
        if status == "applied":
            in_situ_total += 1
            ref = v77[qid]
            mine = [[x["group"], x["score"], x["chunk"]] for x in groups]
            reps = [[x["group"], x["representative"] or None, x["rule"]] for x in groups]
            if mine == [list(x) for x in ref["S"][:10]] and reps == [list(x) for x in ref["S_reps"]]:
                in_situ_same += 1
            else:
                in_situ_mismatch.append(qid)
        q_sha = sha16("query: " + texts[qid])
        a_emb = [s for e in a.get("koe5_embeds", []) for s in e["sha16"]]
        b_emb = [s for e in b.get("koe5_embeds", []) for s in e["sha16"]]
        a_ms = statistics.mean([runs["A1"][qid]["search_ms"], runs["A2"][qid]["search_ms"]])
        b_ms = statistics.mean([runs["B1"][qid]["search_ms"], runs["B2"][qid]["search_ms"]])
        a_prompts, b_prompts = first_prompts(a), first_prompts(b)
        per.append({
            "set": a["set"], "qid": qid, "design_seen_v77": qid in DESIGN_SEEN_V77, "v77_reached": qid in REACHED_V77,
            "v75A_hit10": hit10[qid], "sem_status": status, "vector_source": sem.get("vector_source") or "",
            "aux_len": len(aux), "aux_new": sum(1 for s in aux if s not in a_ids),
            "entered": "|".join(entered), "displaced": "|".join(displaced),
            "gold_in_aux": any(s in g for s in aux),
            "gold_aux_rank": first_rank(aux, g),
            "gold_entered": any(s in g for s in entered), "gold_displaced": any(s in g for s in displaced),
            "A_cand_rank": a_rank, "B_cand_rank": b_rank,
            "recall": {(True, True): "both", (True, False): "lost", (False, True): "gained", (False, False): "neither"}[
                (a_rank is not None, b_rank is not None)],
            "rank_move": (a_rank - b_rank) if (a_rank and b_rank) else None,
            "cand_set_changed": set(a_ids) != set(b_ids), "cand_order_changed": a_ids != b_ids,
            "cand_scores_changed": [c["score"] for c in a["candidates"]] != [c["score"] for c in b["candidates"]],
            "prompt_changed": a_prompts != b_prompts,
            "A_search_ms_mean": round(a_ms, 3), "B_search_ms_mean": round(b_ms, 3), "delta_ms": round(b_ms - a_ms, 3),
            "sem_wait_ms": sem.get("wait_ms"), "sem_embed_ms": sem.get("embed_ms"), "sem_search_ms": sem.get("search_ms"),
            "sem_fetch_ms": sem.get("fetch_ms"),
            "koe5_calls_A": len(a_emb), "koe5_calls_B": len(b_emb),
            "query_vec_embeds_A": a_emb.count(q_sha), "query_vec_embeds_B": b_emb.count(q_sha),
            "qdrant_calls_A": a["n_qdrant"], "qdrant_calls_B": b["n_qdrant"],
        })

    by = {r["qid"]: r for r in per}
    applied = [r for r in per if r["sem_status"] == "applied"]
    succ = [r for r in per if r["v75A_hit10"]]
    gold_ids = set().union(*G.values())
    entered_distinct = set(entered_occ)
    displaced_distinct = set(displaced_occ)
    recall_a = sum(1 for r in per if r["A_cand_rank"])
    recall_b = sum(1 for r in per if r["B_cand_rank"])
    gained = [r["qid"] for r in per if r["recall"] == "gained"]
    lost = [r["qid"] for r in per if r["recall"] == "lost"]
    succ_dropped = [r["qid"] for r in succ if r["A_cand_rank"] and not r["B_cand_rank"]]
    succ_down = {r["qid"]: [r["A_cand_rank"], r["B_cand_rank"]] for r in succ if (r["rank_move"] or 0) < 0}
    succ_up = {r["qid"]: [r["A_cand_rank"], r["B_cand_rank"]] for r in succ if (r["rank_move"] or 0) > 0}
    all_down = {r["qid"]: [r["A_cand_rank"], r["B_cand_rank"]] for r in per if (r["rank_move"] or 0) < 0}
    all_up = {r["qid"]: [r["A_cand_rank"], r["B_cand_rank"]] for r in per if (r["rank_move"] or 0) > 0}
    gold_aux_not_final = [r["qid"] for r in per if r["gold_in_aux"] and not r["B_cand_rank"]]
    deltas_applied = [r["delta_ms"] for r in applied]
    deltas_all = [r["delta_ms"] for r in per]
    deltas_skip = [r["delta_ms"] for r in per if r["sem_status"] != "applied"]
    shared = [r for r in applied if r["vector_source"] == "shared_text_path"]
    own = [r for r in applied if r["vector_source"] == "own_embed"]
    tracked = {q: {k: by[q][k] for k in ("sem_status", "vector_source", "gold_aux_rank", "A_cand_rank", "B_cand_rank",
                                         "gold_entered", "design_seen_v77")} for q in REACHED_V77}
    top_groups = group_top10.most_common(10)
    conds = {k: json.loads((HERE / "runs" / v / "run_conditions.json").read_text(encoding="utf-8")) for k, v in RUNS.items()}
    # 하네스의 raw_outside_logical은 세트 사이에 초기화하지 않아 누적값이다 — 마지막 세트(v10) 값이 실행 전체 합이다
    raw_out = {k: c["sets_done"][SETS[-1]].get("raw_outside_logical", {}) for k, c in conds.items()}

    summary = {
        "recall_at_30": {"A": recall_a, "B": recall_b},
        "gained": gained, "lost": lost,
        "v77_reached_tracked": tracked,
        "v77_reached_entered_candidates": [q for q in REACHED_V77 if by[q]["recall"] == "gained"],
        "v75A_successes": len(succ),
        "v75A_success_dropped_from_candidates": succ_dropped,
        "v75A_success_rank_down": succ_down, "v75A_success_rank_up": succ_up,
        "all_rank_down": all_down, "all_rank_up": all_up,
        "candidate_changes": {k: sum(1 for r in per if r[k]) for k in ("cand_set_changed", "cand_order_changed",
                                                                      "cand_scores_changed", "prompt_changed")},
        "candidate_changes_v75A_success": {k: sum(1 for r in succ if r[k]) for k in ("cand_set_changed", "cand_order_changed",
                                                                                    "prompt_changed")},
        "status_counts": dict(Counter(r["sem_status"] for r in per)),
        "vector_source_counts": dict(Counter(r["vector_source"] or "(none)" for r in per)),
        "aux": {
            "applied_queries": len(applied),
            "aux_len_counts": dict(Counter(r["aux_len"] for r in applied)),
            "aux_new_per_query_median": med([r["aux_new"] for r in applied]),
            "gold_in_aux_queries": [r["qid"] for r in per if r["gold_in_aux"]],
            "gold_in_aux_but_not_in_final_30": gold_aux_not_final,
            "entered": {"occurrences": len(entered_occ), "distinct_songs": len(entered_distinct),
                        "queries": sum(1 for r in per if r["entered"]),
                        "per_query_median_applied": med([len(r["entered"].split("|")) if r["entered"] else 0 for r in applied]),
                        "per_query_max": max((len(r["entered"].split("|")) if r["entered"] else 0) for r in per),
                        "gold_occurrences": sum(1 for r in per for s in r["entered"].split("|") if s and s in G[r["qid"]]),
                        "gold_queries": [r["qid"] for r in per if r["gold_entered"]]},
            "displaced": {"occurrences": len(displaced_occ), "distinct_songs": len(displaced_distinct),
                          "gold_queries": [r["qid"] for r in per if r["gold_displaced"]],
                          "gold_queries_v75A_success": [r["qid"] for r in succ if r["gold_displaced"]]},
            "final_30_vs_A": {"new_occurrences": len(final_new_occ), "new_distinct": len(set(final_new_occ)),
                              "new_from_aux_list": final_new_from_aux, "new_gold_occurrences": final_new_gold,
                              "out_occurrences": len(final_out_occ), "out_distinct": len(set(final_out_occ)),
                              "out_gold_queries": [r["qid"] for r in per if r["A_cand_rank"] and not r["B_cand_rank"]]},
            "sem_entered_but_not_in_final_30": {"occurrences": len(sem_entered_not_final),
                                                "displaced_by_work_name_aux": len(sem_displaced_by_work),
                                                "gold": [x for x in sem_entered_not_final if x.split(":")[1] in G[x.split(":")[0]]]},
            "already_in_A_candidates": {"occurrences": len(aux_in_A_moves),
                                        "moved_up": sum(1 for m in aux_in_A_moves if m > 0),
                                        "unchanged": sum(1 for m in aux_in_A_moves if m == 0),
                                        "moved_down": sum(1 for m in aux_in_A_moves if m < 0),
                                        "move_median": med(aux_in_A_moves)},
        },
        "hubs": {"distinct_groups_in_top10": len(group_top10),
                 "top10_groups": [{"group": gid, "queries_in_top10": n, "queries_rep_entered": group_entered.get(gid, 0)}
                                  for gid, n in top_groups],
                 "top5_groups_query_union": len({r["qid"] for r in per for gr in (B[r["qid"]].get("sem") or {}).get("groups", [])
                                                 if gr["group"] in {gid for gid, _ in top_groups[:5]}}),
                 "entered_occurrences_from_groups_seen_in_10plus_queries": sum(
                     n for gid, n in group_entered.items() if group_top10[gid] >= 10)},
        "in_situ_vs_v77": {"applied": in_situ_total, "same": in_situ_same, "mismatch": in_situ_mismatch},
        "cost": {
            "search_ms_median": {"A": med([r["A_search_ms_mean"] for r in per]), "B": med([r["B_search_ms_mean"] for r in per])},
            "paired_delta_ms": {"all_median": med(deltas_all), "applied_median": med(deltas_applied),
                                "applied_p90": p90(deltas_applied), "skipped_median": med(deltas_skip),
                                "shared_vector_median": med([r["delta_ms"] for r in shared]),
                                "own_embed_median": med([r["delta_ms"] for r in own])},
            "sem_components_ms_median": {"wait_shared": med([r["sem_wait_ms"] for r in shared]),
                                         "embed_own": med([r["sem_embed_ms"] for r in own]),
                                         "search": med([r["sem_search_ms"] for r in applied]),
                                         "fetch": med([r["sem_fetch_ms"] for r in applied if r["sem_fetch_ms"] is not None])},
            "koe5_calls_total": {"A": sum(r["koe5_calls_A"] for r in per), "B": sum(r["koe5_calls_B"] for r in per)},
            "query_vector_embeds_total": {"A": sum(r["query_vec_embeds_A"] for r in per),
                                          "B": sum(r["query_vec_embeds_B"] for r in per)},
            "extra_query_vector_embeds_by_source": {
                "shared_text_path": dict(Counter(r["query_vec_embeds_B"] - r["query_vec_embeds_A"] for r in shared)),
                "own_embed": dict(Counter(r["query_vec_embeds_B"] - r["query_vec_embeds_A"] for r in own)),
                "skipped": dict(Counter(r["koe5_calls_B"] - r["koe5_calls_A"] for r in per if r["sem_status"] != "applied"))},
            "extra_koe5_calls_total": sum(r["koe5_calls_B"] - r["koe5_calls_A"] for r in per),
            "qdrant_logical_calls": {"A": sum(r["qdrant_calls_A"] for r in per), "B": sum(r["qdrant_calls_B"] for r in per)},
            "qdrant_raw_outside_logical_run_total": raw_out,
            "run_seconds": {k: c["total_seconds"] for k, c in conds.items()},
            "cold_before_first_query_seconds": {k: c["cold_before_first_query_seconds"] for k, c in conds.items()},
            "warmup": {k: c["warmup"] for k, c in conds.items()},
            "ru_maxrss_mb": {k: round(c["ru_maxrss_bytes"] / 1e6, 1) for k, c in conds.items()},
        },
    }
    g = summary
    g["gates"] = {
        "G1_recall_net_gain_ge_1": recall_b - recall_a >= 1,
        "G2_no_v75A_success_dropped": not succ_dropped,
        "G3_no_path_error_and_repeats_identical": None,  # compare/ 결과와 실행 조건으로 채운다(아래)
        "G4_applied_paired_delta_median_le_50ms": (med(deltas_applied) or 0) <= 50,
    }
    comp = {n: json.loads((HERE / "compare" / f"{n}.json").read_text(encoding="utf-8")) for n in ("A1_vs_A2", "B1_vs_B2")}
    failures = sum(len(c["sets_done"][s]["path_failures"]) for c in conds.values() for s in SETS)
    sem_failed = sum(1 for k in ("B1", "B2") for r in runs[k].values() if str(r.get("sem_status", "")).startswith("failed"))
    g["gates"]["G3_no_path_error_and_repeats_identical"] = (
        failures == 0 and sem_failed == 0 and all(c["input_identical"] == 176 for c in comp.values()))
    g["gates"]["all"] = all(g["gates"][k] for k in ("G1_recall_net_gain_ge_1", "G2_no_v75A_success_dropped",
                                                    "G3_no_path_error_and_repeats_identical",
                                                    "G4_applied_paired_delta_median_le_50ms"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("recall_at_30", "gained", "lost", "v77_reached_entered_candidates",
                                               "v75A_success_dropped_from_candidates", "candidate_changes",
                                               "candidate_changes_v75A_success", "gates")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
