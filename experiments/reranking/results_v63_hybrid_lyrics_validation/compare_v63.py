"""v63 비교 — 완료 표식·건수·질의 집합·공통 조건을 먼저 검증하고, 실행 쌍마다 질의별 차이와 지표를 낸다. 모델·DB 호출 없음.

    venv/bin/python compare_v63.py --runs <run_dir> <run_dir> ... --pairs A1:B1,A2:B2,A1:A2,B1:B2 --out <폴더> [--overwrite]

run 이름은 run_conditions.json의 `arm`+`rep`(예: A1)로 부른다. 같은 mode끼리만 비교한다.
종료 코드: 0 정상 · 3 검증 실패(완료 표식 없음·부분 실행·건수/질의 집합 불일치·공통 조건 불일치) · 4 반복 간 입력 변동(offline
A/A·B/B에서 후보·보호·프롬프트 지문이 다름) · 2 출력 덮어쓰기 거부.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v63_runtime import EXPECTED_ROWS, LYRIC_QUERY_IDS, SETS, TRACE_QUERY_IDS  # noqa: E402

COMMON_KEYS = ("lyrics_snapshot", "index_fingerprint", "point_count")


class ValidationError(RuntimeError):
    pass


def pct(values: List[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def load_run(run_dir: Path) -> Dict[str, Any]:
    done = run_dir / "DONE"
    if not done.exists():
        raise ValidationError(f"{run_dir.name}: DONE 없음(미완료·실패 실행)")
    if json.loads(done.read_text(encoding="utf-8")).get("partial"):
        raise ValidationError(f"{run_dir.name}: 부분 실행")
    cond = json.loads((run_dir / "run_conditions.json").read_text(encoding="utf-8"))
    if cond.get("status") != "complete":
        raise ValidationError(f"{run_dir.name}: status={cond.get('status')}")
    queries: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for set_name in SETS:
        d = run_dir / f"{set_name}_dev"
        with (d / "search_eval_dev_detail.csv").open(encoding="utf-8-sig") as fh:
            detail = list(csv.DictReader(fh))
        if len(detail) != EXPECTED_ROWS[set_name]:
            raise ValidationError(f"{run_dir.name}/{set_name}: detail {len(detail)}건 ≠ {EXPECTED_ROWS[set_name]}")
        rec = {}
        with (d / "v63_queries.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                rec[row["query_id"]] = row
        calls: Dict[str, List[Dict[str, Any]]] = {}
        with (d / "v63_hybrid_calls.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                c = json.loads(line)
                calls.setdefault(c["query_id"], []).append(c)
        for row in detail:
            qid = row["query_id"]
            if qid not in rec:
                raise ValidationError(f"{run_dir.name}: {qid} 기록 없음")
            queries[qid] = {"set": set_name, "detail": row, "rec": rec[qid], "calls": calls.get(qid, [])}
            order.append(qid)
    if len(order) != sum(EXPECTED_ROWS.values()) or len(set(order)) != len(order):
        raise ValidationError(f"{run_dir.name}: 질의 {len(order)}건(중복 포함 여부 확인)")
    name = f"{cond['arm']}{cond['rep']}"
    return {"name": name, "dir": run_dir, "cond": cond, "queries": queries, "order": order}


def _int(value: str):
    return int(value) if str(value).strip() not in {"", "None"} else None


def per_query(q: Dict[str, Any]) -> Dict[str, Any]:
    d, r = q["detail"], q["rec"]
    cands = r.get("candidates") or []
    runs = r.get("rerank_runs") or []
    prot = r.get("protection") or {}
    final = r.get("final") or {}
    return {
        "cand_ids": [c["id"] for c in cands],
        "cand_scores": [c["score"] for c in cands],
        "cand_lyric": [c["lyric"] for c in cands],
        "lyrics_results": [(e.get("results"), [(t["id"], t["type"], t["score"]) for t in e.get("top", [])]) for e in r.get("lyrics", [])],
        "protected": prot.get("protected_ids", []),
        "tiers": prot.get("tiers", {}),
        "policy_skip": prot.get("policy_skip", ""),
        "groups": [(g["input_ids"], g["top_k"]) for g in runs],
        "group_scores": [g["input_scores"] for g in runs],
        "prompt_shas": [g.get("contents_sha16") for g in r.get("generates", [])],
        "mixes": [g.get("mixes_sha16") for g in runs],
        "order_notes": [g.get("order_notes") for g in runs],
        "group_status": [g.get("status") for g in runs],
        "final_ids": final.get("ids", []),
        "cand_rank": _int(d.get("candidate_rank@30", "")),
        "base_rank": _int(d.get("baseline_rank", "")),
        "rerank_rank": _int(d.get("rerank_rank", "")),
        "outcome": d.get("rerank_outcome", ""),
        "statuses": d.get("rerank_run_statuses", ""),
        "retrieval_ms": float(d.get("retrieval_ms") or 0.0),
        "rerank_ms": float(d.get("rerank_only_ms") or 0.0),
        "hybrid_calls": len(q["calls"]),
        "hybrid_ref_bad": sum(1 for c in q["calls"] if not (c["ref_same_set"] and not c["ref_score_mismatch"]
                                                         and not c["ref_order_differs_beyond_ties"])),
        "hybrid_ref_missing": sum(len(c["ref_missing"]) for c in q["calls"]),
        "hybrid_ref_mismatch": sum(len(c["ref_score_mismatch"]) for c in q["calls"]),
    }


def metrics(run: Dict[str, Any], ids: List[str]) -> Dict[str, Any]:
    rows = [per_query(run["queries"][q]) for q in ids]
    n = len(rows)

    def hit(key: str, k: int) -> int:
        return sum(1 for r in rows if r[key] is not None and r[key] <= k)

    def mrr(key: str) -> float:
        return round(sum((1.0 / r[key]) if r[key] is not None and r[key] <= 10 else 0.0 for r in rows) / n, 4) if n else 0.0

    outcomes: Dict[str, int] = {}
    group_status: Dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        for s in r["group_status"]:
            group_status[s] = group_status.get(s, 0) + 1
    total_ms = [r["retrieval_ms"] + r["rerank_ms"] for r in rows]
    return {
        "n": n,
        "candidate_recall@30": hit("cand_rank", 30),
        "baseline_hit@1": hit("base_rank", 1), "baseline_hit@5": hit("base_rank", 5), "baseline_hit@10": hit("base_rank", 10),
        "baseline_mrr@10": mrr("base_rank"),
        "hit@1": hit("rerank_rank", 1), "hit@5": hit("rerank_rank", 5), "hit@10": hit("rerank_rank", 10),
        "mrr@10": mrr("rerank_rank"),
        "outcomes": dict(sorted(outcomes.items())),
        "group_status": dict(sorted(group_status.items())),
        "rerank_groups": sum(len(r["groups"]) for r in rows),
        "retrieval_ms_median": round(statistics.median([r["retrieval_ms"] for r in rows]), 1) if n else None,
        "retrieval_ms_p95": round(pct([r["retrieval_ms"] for r in rows], 0.95), 1) if n else None,
        "rerank_ms_median": round(statistics.median([r["rerank_ms"] for r in rows]), 1) if n else None,
        "rerank_ms_p95": round(pct([r["rerank_ms"] for r in rows], 0.95), 1) if n else None,
        "total_ms_median": round(statistics.median(total_ms), 1) if n else None,
        "total_ms_p95": round(pct(total_ms, 0.95), 1) if n else None,
        "hybrid_calls": sum(r["hybrid_calls"] for r in rows),
        "hybrid_calls_not_matching_reference": sum(r["hybrid_ref_bad"] for r in rows),
        "hybrid_ref_missing_songs": sum(r["hybrid_ref_missing"] for r in rows),
        "hybrid_ref_score_mismatch_songs": sum(r["hybrid_ref_mismatch"] for r in rows),
        "queries_with_lyrics_search": sum(1 for r in rows if r["lyrics_results"]),
        "queries_with_protected": sum(1 for r in rows if r["protected"]),
    }


FIELDS = ("cand_ids", "cand_scores", "cand_lyric", "lyrics_results", "protected", "tiers", "policy_skip",
          "groups", "group_scores", "prompt_shas", "mixes", "order_notes", "final_ids")
INPUT_FIELDS = ("cand_ids", "cand_scores", "cand_lyric", "lyrics_results", "protected", "tiers", "policy_skip",
                "groups", "group_scores", "prompt_shas")


def compare_pair(a: Dict[str, Any], b: Dict[str, Any], order: List[str]) -> Dict[str, Any]:
    rows = []
    diff_counts = {f: 0 for f in FIELDS}
    gains, losses = [], []
    for qid in order:
        pa, pb = per_query(a["queries"][qid]), per_query(b["queries"][qid])
        diffs = [f for f in FIELDS if pa[f] != pb[f]]
        for f in diffs:
            diff_counts[f] += 1
        ra = pa["rerank_rank"] if pa["rerank_rank"] is not None else 99
        rb = pb["rerank_rank"] if pb["rerank_rank"] is not None else 99
        h10a, h10b = ra <= 10, rb <= 10
        if h10b and not h10a:
            gains.append(qid)
        elif h10a and not h10b:
            losses.append(qid)
        rows.append({
            "query_id": qid, "set": a["queries"][qid]["set"], "lyric_query": int(qid in LYRIC_QUERY_IDS),
            "diff_fields": "|".join(diffs),
            f"{a['name']}_cand_rank": pa["cand_rank"] or "", f"{b['name']}_cand_rank": pb["cand_rank"] or "",
            f"{a['name']}_base_rank": pa["base_rank"] or "", f"{b['name']}_base_rank": pb["base_rank"] or "",
            f"{a['name']}_rank": pa["rerank_rank"] or "", f"{b['name']}_rank": pb["rerank_rank"] or "",
            f"{a['name']}_outcome": pa["outcome"], f"{b['name']}_outcome": pb["outcome"],
            f"{a['name']}_groups": len(pa["groups"]), f"{b['name']}_groups": len(pb["groups"]),
            f"{a['name']}_protected": len(pa["protected"]), f"{b['name']}_protected": len(pb["protected"]),
            "first_cand_diff_pos": next((i + 1 for i, (x, y) in enumerate(zip(pa["cand_ids"], pb["cand_ids"])) if x != y), ""),
            "max_cand_score_delta": round(max((abs(x - y) for x, y in zip(pa["cand_scores"], pb["cand_scores"])), default=0.0), 9)
            if pa["cand_ids"] == pb["cand_ids"] else "",
            "hit10_change": "gain" if qid in gains else ("loss" if qid in losses else ""),
            "rank_change_positive_is_better": ra - rb if (ra != 99 or rb != 99) else 0,
        })
    return {
        "pair": f"{a['name']}:{b['name']}",
        "queries_with_any_diff": sum(1 for r in rows if r["diff_fields"]),
        "queries_with_input_diff": sum(1 for r in rows if set(r["diff_fields"].split("|")) & set(INPUT_FIELDS) - {""}),
        "field_diff_counts": diff_counts,
        "final_rank_diff_queries": [r["query_id"] for r in rows
                                    if r[f"{a['name']}_rank"] != r[f"{b['name']}_rank"]],
        "hit10_gains_b_over_a": gains, "hit10_losses_b_vs_a": losses,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-repeat-identical", action="store_true",
                        help="A/A·B/B 쌍에서 입력 필드(후보·보호·그룹·프롬프트)가 다르면 exit 4 (offline 검증용)")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"출력 폴더에 이미 결과가 있다: {out}", file=sys.stderr)
        return 2
    try:
        runs = {r["name"]: r for r in (load_run(Path(p)) for p in args.runs)}
        modes = {r["cond"]["mode"] for r in runs.values()}
        if len(modes) != 1:
            raise ValidationError(f"mode가 섞였다: {modes}")
        base = next(iter(runs.values()))
        for r in runs.values():
            for key in COMMON_KEYS:
                ka = json.dumps(base["cond"].get(key), sort_keys=True, default=str)
                kb = json.dumps(r["cond"].get(key), sort_keys=True, default=str)
                if key == "index_fingerprint":
                    ka = json.dumps({k: v for k, v in base["cond"][key].items() if k != "build_seconds"}, sort_keys=True)
                    kb = json.dumps({k: v for k, v in r["cond"][key].items() if k != "build_seconds"}, sort_keys=True)
                if ka != kb:
                    raise ValidationError(f"{r['name']}: 공통 조건 {key}가 {base['name']}와 다르다")
            env_a = {k: v for k, v in base["cond"]["env"].items()}
            env_b = {k: v for k, v in r["cond"]["env"].items()}
            if env_a != env_b:
                raise ValidationError(f"{r['name']}: 환경 설정이 {base['name']}와 다르다")
            if r["order"] != base["order"]:
                raise ValidationError(f"{r['name']}: 질의 순서·집합이 다르다")
        pairs = []
        for spec in args.pairs.split(","):
            x, y = spec.split(":")
            if x not in runs or y not in runs:
                raise ValidationError(f"쌍 {spec}: 실행이 없다({sorted(runs)})")
            pairs.append((runs[x], runs[y]))
    except ValidationError as exc:
        print(f"검증 실패: {exc}", file=sys.stderr)
        return 3

    out.mkdir(parents=True, exist_ok=True)
    order = base["order"]
    lyric = [q for q in order if q in LYRIC_QUERY_IDS]
    non_lyric = [q for q in order if q not in LYRIC_QUERY_IDS]
    summary: Dict[str, Any] = {"mode": base["cond"]["mode"], "runs": {}, "pairs": {}}
    for name, run in sorted(runs.items()):
        summary["runs"][name] = {
            "dir": str(run["dir"]),
            "all_176": metrics(run, order),
            "lyric_19": metrics(run, lyric),
            "non_lyric_157": metrics(run, non_lyric),
            "cold_before_first_query_seconds": run["cond"].get("cold_before_first_query_seconds"),
            "qdrant_open_seconds": run["cond"].get("qdrant_open_seconds"),
            "total_seconds": run["cond"].get("total_seconds"),
            "lyrics_snapshot_builds": [b for s in run["cond"]["sets_done"].values() for b in s.get("lyrics_snapshot_builds", [])],
            "budget_local": run["cond"].get("budget_local"),
        }
    repeat_problem = []
    for a, b in pairs:
        result = compare_pair(a, b, order)
        rows = result.pop("rows")
        key = result["pair"]
        summary["pairs"][key] = result
        with (out / f"pair_{a['name']}_{b['name']}.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if a["cond"]["arm"] == b["cond"]["arm"] and result["queries_with_input_diff"]:
            repeat_problem.append(key)
    # 상세 추적 4질의
    trace = {}
    for qid in TRACE_QUERY_IDS:
        trace[qid] = {name: {k: v for k, v in per_query(run["queries"][qid]).items()
                             if k in ("cand_ids", "cand_scores", "cand_lyric", "protected", "tiers", "groups",
                                      "prompt_shas", "group_status", "final_ids", "cand_rank", "base_rank",
                                      "rerank_rank", "outcome", "hybrid_calls", "hybrid_ref_bad",
                                      "hybrid_ref_missing", "hybrid_ref_mismatch")}
                      for name, run in sorted(runs.items())}
        trace[qid]["hybrid_calls"] = {name: [{k: c[k] for k in ("seq", "top_k", "filtered", "ref_identical_ids",
                                                               "ref_missing", "ref_extra", "ref_score_mismatch",
                                                               "ref_order_differs_beyond_ties")}
                                              for c in run["queries"][qid]["calls"]]
                                      for name, run in sorted(runs.items())}
    (out / "trace_4queries.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    summary["repeat_input_variation"] = repeat_problem
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, r in summary["runs"].items():
        m = r["all_176"]
        print(f"{name}: cand {m['candidate_recall@30']} base H1/5/10 {m['baseline_hit@1']}/{m['baseline_hit@5']}/{m['baseline_hit@10']}"
              f" MRR {m['baseline_mrr@10']} | rerank H1/5/10 {m['hit@1']}/{m['hit@5']}/{m['hit@10']} MRR {m['mrr@10']}"
              f" | outcomes {m['outcomes']} | ref-bad {m['hybrid_calls_not_matching_reference']}/{m['hybrid_calls']}")
    for key, p in summary["pairs"].items():
        print(f"{key}: any-diff {p['queries_with_any_diff']} input-diff {p['queries_with_input_diff']} "
              f"fields {{{', '.join(f'{k}:{v}' for k, v in p['field_diff_counts'].items() if v)}}} "
              f"rank-diff {len(p['final_rank_diff_queries'])} gains {p['hit10_gains_b_over_a']} losses {p['hit10_losses_b_vs_a']}")
    if args.require_repeat_identical and repeat_problem:
        print(f"반복 간 입력 변동: {repeat_problem}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
