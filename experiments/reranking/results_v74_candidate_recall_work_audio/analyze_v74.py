"""v74 작품명 × 소리 4구성 비교(리랭킹 전) — 후보 회수@30 · 얻음/잃음 · 정답 후보 순위 · 후보·입력 변화 · 기존 성공 노출 · 경로 호출 · 지연.

모델 호출 없음. **가짜 리랭커의 최종 순서는 읽지 않는다.** v73 3차 실제 실행(B1 = 작품명 ON·소리 ON)의 최종 순위는 기준 실패·성공 분류에만
쓰고, 바뀐 후보에 붙여 새 품질로 계산하지 않는다(입력이 기준과 같은 질의만 "v73 실제 결과가 그대로 유효"로 적는다).

입력:
- `--runs K=폴더,...`: v74 실행(DONE 필수). `--base K`: 비교 기준(work_on 반복 1).
- `--compare K=compare_inputs 결과 json,...`: 기준 ↔ 구성 입력 대조(`compare_inputs.py --a <기준> --b <구성>`). 리랭커 입력 변화의
  정의는 compare_inputs의 것(후보 id·순서·점수 · 가사 · 보호 · 그룹 입력 · 전송 프롬프트)이다.
- `--failures diagnosis/baseline_failures.csv`: v73 B1·B2 기준 성공/실패.

    venv/bin/python analyze_v74.py --runs W1=runs/work_on_server_rep1,... --base W1 --compare O1=compare/....json,... \
        --failures diagnosis/baseline_failures.csv --out-dir analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")
EXPECTED = {"v06": 57, "v09": 59, "v10": 60}
# 관심 사례(설명용 — 코드 분기·규칙에 쓰지 않는다)
WATCH = {"work_gain": ["n033", "n043", "n096"], "overlap_gain": ["c707"], "audio_skip_past_loss": ["q215"],
         "audio_skip_past_gain": ["t058", "t077", "t078", "t100", "t111"]}


def gold() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for s in SETS:
        with open(ROOT / f"experiments/reranking/eval_queries_{s}.csv", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.DictReader(fh) if r["split"].strip() == "dev"]
        if len(rows) != EXPECTED[s]:
            raise SystemExit(f"{s}: dev {len(rows)}")
        for r in rows:
            qid = r["query_id"].strip()
            if qid in out:
                raise SystemExit(f"질의 ID 중복 {qid}")
            rel = [x.strip() for x in r["relevant_ids"].split("|") if x.strip()]
            if not rel:
                raise SystemExit(f"{qid}: 정답 없음")
            out[qid] = {"set": s, "relevant": rel}
    return out


def load(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"미완료 {run_dir}")
    out: Dict[str, Dict[str, Any]] = {}
    for s in SETS:
        for line in open(run_dir / f"{s}_dev" / "v74_queries.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["query_id"] in out:
                raise SystemExit(f"{run_dir}: 질의 중복 {r['query_id']}")
            out[r["query_id"]] = r
    cond = json.loads((run_dir / "run_conditions.json").read_text(encoding="utf-8"))
    out["__conditions__"] = cond  # type: ignore[assignment]
    return out


def qdrant_calls(run_dir: Path) -> Dict[str, Any]:
    """곡 컬렉션 논리 조회(하네스 기록) — 컬렉션별 호출 수·요청 top_k 합. 질의별 귀속은 qdrant_seqs로 한다."""
    by_coll: Dict[str, Dict[str, int]] = {}
    for s in SETS:
        for line in open(run_dir / f"{s}_dev" / "v74_qdrant_calls.jsonl", encoding="utf-8"):
            r = json.loads(line)
            key = f"{r.get('kind', '')}:{r.get('collection', '')}"
            ent = by_coll.setdefault(key, {"calls": 0, "top_k_sum": 0})
            ent["calls"] += 1
            ent["top_k_sum"] += int(r.get("top_k") or 0)
    return by_coll


def first_rank(ids: List[str], relevant: List[str]) -> Optional[int]:
    rel = set(relevant)
    return next((i for i, x in enumerate(ids, 1) if x in rel), None)


def pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    def path_of(v: str) -> Path:
        return Path(v) if Path(v).is_absolute() else HERE / v

    run_dirs = {k: path_of(v) for k, v in (x.split("=") for x in args.runs.split(","))}
    runs = {k: load(d) for k, d in run_dirs.items()}
    conds = {k: r.pop("__conditions__") for k, r in runs.items()}
    cmp_files = {k: json.loads(path_of(v).read_text(encoding="utf-8"))
                 for k, v in (x.split("=") for x in args.compare.split(","))}
    g = gold()
    for k, r in runs.items():
        if set(r) != set(g):
            raise SystemExit(f"{k}: 질의 집합이 정답 파일과 다르다")
    for k, c in cmp_files.items():
        if c["queries"] != len(g) or c["missing_either"]:
            raise SystemExit(f"{k}: 입력 대조 질의 수 {c['queries']} · 누락 {len(c['missing_either'])}")
        if not (c["b"].rstrip("/").endswith(run_dirs[k].name) and c["a"].rstrip("/").endswith(run_dirs[args.base].name)):
            raise SystemExit(f"{k}: 입력 대조 파일의 a/b가 기준·구성 실행과 다르다 ({c['a']} → {c['b']})")
    with open(path_of(args.failures), encoding="utf-8-sig") as fh:
        fail = {r["query_id"]: r for r in csv.DictReader(fh)}
    if set(fail) != set(g):
        raise SystemExit("기준 실패 표의 질의 집합이 다르다")
    base_hits = {q for q, r in fail.items() if r["kind"] == "hit"}
    base = runs[args.base]

    rows = []
    for qid, info in g.items():
        b = base[qid]
        bids = [c["id"] for c in b["candidates"]]
        brank = first_rank(bids, info["relevant"])
        if str(brank or "") != fail[qid]["pool_rank"]:
            raise SystemExit(f"{qid}: 기준 실행 후보 순위 {brank} ≠ 실패 표 {fail[qid]['pool_rank']}")
        out: Dict[str, Any] = {"query_id": qid, "set": info["set"], "relevant_count": len(info["relevant"]),
                               "baseline_kind": fail[qid]["kind"], "v73_B1_final_rank": fail[qid]["B1_rank"],
                               "work_status": b.get("work_status", ""),
                               "watch": "|".join(t for t, qs in WATCH.items() if qid in qs)}
        for k, r in runs.items():
            row = r[qid]
            ids = [c["id"] for c in row["candidates"]]
            out[f"cand_{k}"] = first_rank(ids, info["relevant"]) or ""
            out[f"audio_q_{k}"] = row["audio_queries"]
            out[f"clap_{k}"] = len(row["clap_embed"])
            out[f"qdrant_calls_{k}"] = len(row["qdrant_seqs"])
            out[f"search_ms_{k}"] = row["search_ms"]
            if k in cmp_files:
                c = cmp_files[k]
                out[f"cand_set_changed_{k}"] = int(set(ids) != set(bids))
                out[f"cand_order_or_score_changed_{k}"] = int(
                    ids != bids or [x["score"] for x in row["candidates"]] != [x["score"] for x in b["candidates"]])
                out[f"input_changed_{k}"] = int(qid in set(c["input_diff_queries"]))
                # 실제 전송 프롬프트 기준(내부 입력 변화 중 점수만 바뀐 질의는 프롬프트가 같다)
                out[f"prompt_changed_{k}"] = int(qid in set(c["diff_queries"].get("prompts", [])))
                out[f"new_ids_{k}"] = len(set(ids) - set(bids))
                out[f"policy_skip_{k}"] = (row["protection"] or {}).get("policy_skip", "")
        rows.append(out)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "per_query_configs.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary: Dict[str, Any] = {"queries": len(rows), "base": args.base, "base_hits_v73_B": len(base_hits), "configs": {}}
    for k in runs:
        ms = [r[f"search_ms_{k}"] for r in rows]
        ent: Dict[str, Any] = {
            "run": run_dirs[k].name, "code": conds[k].get("code"),
            "candidate_recall@30": sum(1 for r in rows if r[f"cand_{k}"] != ""),
            "by_set": {s: sum(1 for r in rows if r["set"] == s and r[f"cand_{k}"] != "") for s in SETS},
            "queries_with_audio_query": sum(1 for r in rows if r[f"audio_q_{k}"] > 0),
            "audio_queries_total": sum(r[f"audio_q_{k}"] for r in rows),
            "queries_with_clap_embed": sum(1 for r in rows if r[f"clap_{k}"] > 0),
            "qdrant_logical_calls": sum(r[f"qdrant_calls_{k}"] for r in rows),
            "qdrant_by_collection": qdrant_calls(run_dirs[k]),
            "search_ms_median": round(statistics.median(ms), 1), "search_ms_p95": round(pct(ms, 0.95), 1),
            "search_ms_mean": round(statistics.mean(ms), 1),
            "work_status_counts": {},
        }
        for r in runs[k].values():
            st = r.get("work_status") or "(none)"
            ent["work_status_counts"][st] = ent["work_status_counts"].get(st, 0) + 1
        if k in cmp_files:
            gained = [r["query_id"] for r in rows if r[f"cand_{k}"] != "" and r[f"cand_{args.base}"] == ""]
            lost = [r["query_id"] for r in rows if r[f"cand_{k}"] == "" and r[f"cand_{args.base}"] != ""]
            both = [r for r in rows if r[f"cand_{k}"] != "" and r[f"cand_{args.base}"] != ""]
            up = [(r["query_id"], r[f"cand_{args.base}"], r[f"cand_{k}"]) for r in both if r[f"cand_{k}"] < r[f"cand_{args.base}"]]
            down = [(r["query_id"], r[f"cand_{args.base}"], r[f"cand_{k}"]) for r in both if r[f"cand_{k}"] > r[f"cand_{args.base}"]]
            changed = [r["query_id"] for r in rows if r[f"input_changed_{k}"]]
            ent.update({
                "vs_base_gained": gained, "vs_base_lost": lost,
                "rank_up": len(up), "rank_down": len(down), "rank_same": len(both) - len(up) - len(down),
                "rank_down_detail": down, "rank_up_detail": up,
                "cand_set_changed_queries": sum(r[f"cand_set_changed_{k}"] for r in rows),
                "cand_order_or_score_changed_queries": sum(r[f"cand_order_or_score_changed_{k}"] for r in rows),
                "rerank_input_changed_queries": len(changed),
                "rerank_input_same_queries": len(rows) - len(changed),
                "base_hits_with_input_changed": sorted(q for q in changed if q in base_hits),
                "base_hits_with_input_changed_n": sum(1 for q in changed if q in base_hits),
                "base_hits_with_input_same_n": len(base_hits) - sum(1 for q in changed if q in base_hits),
                "prompt_changed_queries": sum(r[f"prompt_changed_{k}"] for r in rows),
                "base_hits_with_prompt_changed_n": sum(1 for r in rows if r[f"prompt_changed_{k}"] and r["query_id"] in base_hits),
                "input_changed_prompt_same(점수 등 내부 입력만)": sorted(r["query_id"] for r in rows
                                                              if r[f"input_changed_{k}"] and not r[f"prompt_changed_{k}"]),
                "base_hits_lost_from_pool": sorted(q for q in lost if q in base_hits),
                "base_hits_rank_down": [d for d in down if d[0] in base_hits],
                "compare_diff_counts": cmp_files[k]["diff_counts"],
                "groups_not_sent": cmp_files[k]["groups_not_sent"],
            })
        summary["configs"][k] = ent
    summary["watch"] = {qid: {k: rows_by[f"cand_{k}"] for k in runs}
                        for qid, rows_by in ((r["query_id"], r) for r in rows)
                        if any(qid in qs for qs in WATCH.values())}
    (args.out_dir / "summary_configs.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                                       encoding="utf-8")
    for k, e in summary["configs"].items():
        extra = ""
        if k in cmp_files:
            extra = (f" · 얻음 {e['vs_base_gained']} 잃음 {e['vs_base_lost']} · 순위 ↑{e['rank_up']} ↓{e['rank_down']}"
                     f" · 내부 입력 변화 {e['rerank_input_changed_queries']} (기존 성공 {e['base_hits_with_input_changed_n']})"
                     f" · 프롬프트 변화 {e['prompt_changed_queries']} (기존 성공 {e['base_hits_with_prompt_changed_n']})")
        print(f"{k} {e['code']}: 회수 {e['candidate_recall@30']} {e['by_set']}{extra}")


if __name__ == "__main__":
    main()
