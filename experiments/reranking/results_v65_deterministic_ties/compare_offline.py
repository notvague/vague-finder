"""v65 평가기 실행 비교 — 후보 30·가사·보호·리랭커 입력·프롬프트 sha·Qdrant 경로별 반환을 실행 쌍마다 대조한다(모델·DB 호출 없음).

    venv/bin/python compare_offline.py [--root offline] [--out compare_offline.json]

비교 항목(질의 176건마다):
  cand_ids      후보 30 ID·순서(평가기 top_k 30)        cand_scores  후보 융합 점수(같은 ID 위치의 최대 절대 차이)
  lyrics        가사 검색 결과 id·종류·점수               protection   보호 곡·tier·정책 생략 사유
  rerank_input  리랭커 그룹별 입력 id·점수·가사 종류      prompts      리랭커 프롬프트 sha16 목록(그룹·패스 순)
  final         최종 순서(가짜 결정적 응답 — 품질 아님)
경로별: 같은 질의의 Qdrant 논리 조회를 (종류, 컬렉션, 질의 안 순번)으로 맞춰 반환 id 목록이 다른 조회를 센다.
리랭킹 전 정답 위치(엄격 정답)도 함께 계산한다 — 품질 판단에 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")
FIELDS = ("cand_ids", "lyrics", "protection", "rerank_input", "prompts", "final")


def load_gold() -> Dict[str, set]:
    gold = {}
    for name in SETS:
        with (ROOT / f"experiments/reranking/eval_queries_{name}.csv").open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row["split"].strip() == "dev":
                    raw = row["relevant_ids"].replace(";", "|").replace(",", "|")
                    gold[row["query_id"].strip()] = {x.strip() for x in raw.split("|") if x.strip()}
    return gold


def load_run(run_dir: Path) -> Dict[str, Any]:
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"완료되지 않은 실행: {run_dir}")
    cond = json.loads((run_dir / "run_conditions.json").read_text(encoding="utf-8"))
    queries: Dict[str, Dict[str, Any]] = {}
    calls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in SETS:
        d = run_dir / f"{s}_dev"
        for line in (d / "v65_queries.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            queries[row["query_id"]] = row
        for line in (d / "v65_qdrant_calls.jsonl").read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            calls[c["query_id"]].append(c)
    return {"name": run_dir.name, "cond": cond, "queries": queries, "calls": calls}


def view(row: Dict[str, Any]) -> Dict[str, Any]:
    cands = row.get("candidates") or []
    return {
        "cand_ids": [c["id"] for c in cands],
        "cand_scores": [c["score"] for c in cands],
        "lyrics": [[(t["id"], t["type"], t["score"]) for t in e.get("top", [])] for e in row.get("lyrics", [])],
        "protection": row.get("protection"),
        "rerank_input": [(r["input_ids"], r["input_scores"], r["input_lyric"], r["top_k"]) for r in row.get("rerank_runs", [])],
        "prompts": [g["contents_sha16"] for g in row.get("generates", [])],
        "final": (row.get("final") or {}).get("ids"),
    }


def path_signature(calls: List[Dict[str, Any]]) -> Dict[str, List[tuple]]:
    """경로(종류:컬렉션)별 논리 조회 결과의 **다중집합**. 경로 스레드가 끝나는 순서는 실행마다 달라서 순번으로 짝짓지 않는다."""
    out: Dict[str, List[tuple]] = defaultdict(list)
    for c in calls:
        key = f"{c['kind']}:{c['collection'].replace('vaguefinder-', '')}"
        out[key].append((c.get("top_k"), bool(c.get("filtered") or c.get("media_filtered")), tuple(c["result_ids"])))
    return {k: sorted(v) for k, v in out.items()}


def rank_of(ids: List[str], gold: set) -> Optional[int]:
    for i, sid in enumerate(ids, start=1):
        if sid in gold:
            return i
    return None


def metrics(run: Dict[str, Any], gold: Dict[str, set]) -> Dict[str, Any]:
    ranks = {q: rank_of(view(r)["cand_ids"], gold.get(q, set())) for q, r in run["queries"].items()}
    n = len(ranks)
    return {
        "queries": n,
        "recall30": sum(1 for r in ranks.values() if r),
        "hit1": sum(1 for r in ranks.values() if r == 1),
        "hit10": sum(1 for r in ranks.values() if r and r <= 10),
        "mrr": round(sum(1 / r for r in ranks.values() if r) / n, 4) if n else 0.0,
        "_ranks": ranks,
    }


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    qids = sorted(set(a["queries"]) | set(b["queries"]))
    diff: Dict[str, List[str]] = {f: [] for f in FIELDS}
    cand_set_diff, max_score_delta, path_diff = [], 0.0, Counter()
    path_diff_queries: Dict[str, List[str]] = defaultdict(list)
    per_query = {}
    for q in qids:
        ra, rb = a["queries"].get(q), b["queries"].get(q)
        if ra is None or rb is None:
            for f in FIELDS:
                diff[f].append(q)
            continue
        va, vb = view(ra), view(rb)
        fields = [f for f in FIELDS if va[f] != vb[f]]
        for f in fields:
            diff[f].append(q)
        if set(va["cand_ids"]) != set(vb["cand_ids"]):
            cand_set_diff.append(q)
        if va["cand_ids"] == vb["cand_ids"]:
            for x, y in zip(va["cand_scores"], vb["cand_scores"]):
                max_score_delta = max(max_score_delta, abs(x - y))
        pa, pb = path_signature(a["calls"].get(q, [])), path_signature(b["calls"].get(q, []))
        paths = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
        kinds = paths
        for k in kinds:
            path_diff[k] += 1
            path_diff_queries[k].append(q)
        if fields or paths:
            per_query[q] = {"fields": fields, "paths": paths}
    return {
        "pair": f"{a['name']} vs {b['name']}", "queries": len(qids),
        "identical_all_fields": len(qids) - len({q for f in FIELDS for q in diff[f]}),
        "diff_counts": {f: len(v) for f, v in diff.items()},
        "cand_set_diff": len(cand_set_diff), "cand_set_diff_queries": cand_set_diff,
        "cand_order_diff_queries": diff["cand_ids"],
        "max_cand_score_delta_same_ids": max_score_delta,
        "path_diff_queries_by_path": dict(sorted(path_diff.items())),
        "path_diff_query_ids": {k: v for k, v in sorted(path_diff_queries.items())},
        "per_query": per_query,
    }


def classes(run: Dict[str, Any]) -> Dict[str, int]:
    out: Counter = Counter()
    for calls in run["calls"].values():
        for c in calls:
            out[f"{c['kind']}:{c['collection'].replace('vaguefinder-', '')}:{c.get('class')}"] += 1
    return dict(sorted(out.items()))


def boundary_summary(run: Dict[str, Any]) -> Dict[str, Any]:
    """경계 동점(기준에서 k번째 점수의 동점이 k 밖까지 이어짐)이 있는 논리 조회와 그 처리."""
    out: Dict[str, Counter] = defaultdict(Counter)
    for calls in run["calls"].values():
        for c in calls:
            key = f"{c['kind']}:{c['collection'].replace('vaguefinder-', '')}"
            out[key]["calls"] += 1
            if c.get("boundary_tie_beyond"):
                out[key]["boundary_tie_calls"] += 1
                out[key][f"boundary_tie_class:{c.get('class')}"] += 1
            if c.get("exact_tie_adjacent_pairs"):
                out[key]["calls_with_exact_ties_in_topk"] += 1
            if c.get("near_tie_adjacent_pairs"):
                out[key]["calls_with_near_ties_in_topk"] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def cost_summary(run: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for calls in run["calls"].values():
        for c in calls:
            key = f"{c['kind']}:{c['collection'].replace('vaguefinder-', '')}"
            agg = out[key]
            agg["logical"] += 1
            for r in c.get("raw", []):
                agg[f"raw_{r['op']}"] += 1
                if r["op"] == "query_points":
                    agg["points_returned"] += r["returned"]
                    if r.get("with_payload") not in (False, None):
                        agg["points_with_payload"] += r["returned"]
                if r["op"] == "retrieve":
                    agg["retrieve_ids"] += r["ids"]
            q_raw = sum(1 for r in c.get("raw", []) if r["op"] == "query_points")
            if q_raw > (2 if c.get("hybrid") else 1):
                agg["expanded_logical"] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(HERE / "offline"))
    parser.add_argument("--out", default=str(HERE / "compare_offline.json"))
    parser.add_argument("--pairs", default="")
    args = parser.parse_args()
    root = Path(args.root)
    runs = {p.name: load_run(p) for p in sorted(root.iterdir()) if p.is_dir() and (p / "DONE").exists()}
    gold = load_gold()
    default_pairs = []
    for code in ("before", "after"):
        default_pairs += [(f"{code}_local_rep1", f"{code}_server_rep1"), (f"{code}_local_rep2", f"{code}_server_rep2"),
                          (f"{code}_local_rep1", f"{code}_local_rep2"), (f"{code}_server_rep1", f"{code}_server_rep2"),
                          (f"{code}_local_rep1", f"{code}_server_rep2"), (f"{code}_local_rep2", f"{code}_server_rep1")]
    default_pairs += [("before_local_rep1", "after_local_rep1"), ("before_server_rep1", "after_server_rep1")]
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p] or default_pairs
    result: Dict[str, Any] = {"runs": {}, "pairs": []}
    for name, run in runs.items():
        m = metrics(run, gold)
        result["runs"][name] = {
            "code_identity": run["cond"].get("code_identity"), "mode": run["cond"]["mode"],
            "network_attempts": run["cond"].get("network_attempts"), "total_seconds": run["cond"].get("total_seconds"),
            "pre_rerank_metrics": {k: v for k, v in m.items() if not k.startswith("_")},
            "qdrant_classes": classes(run), "boundary": boundary_summary(run), "cost": cost_summary(run),
        }
    for a, b in pairs:
        if a in runs and b in runs:
            cmp = compare(runs[a], runs[b])
            ma, mb = metrics(runs[a], gold)["_ranks"], metrics(runs[b], gold)["_ranks"]
            cmp["gold_rank_changes"] = [{"query_id": q, "a": ma.get(q), "b": mb.get(q)} for q in sorted(ma) if ma.get(q) != mb.get(q)]
            result["pairs"].append(cmp)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, r in result["runs"].items():
        print(name, json.dumps(r["pre_rerank_metrics"]), "net", r["network_attempts"])
    for p in result["pairs"]:
        print(p["pair"], "identical", p["identical_all_fields"], json.dumps(p["diff_counts"]), "set", p["cand_set_diff"],
              "paths", json.dumps(p["path_diff_queries_by_path"]), "gold", len(p["gold_rank_changes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
