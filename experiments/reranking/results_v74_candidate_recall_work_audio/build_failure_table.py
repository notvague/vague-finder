"""v74 기준 실패 표 — v73 3차 실제 실행(작품명 ON = B1·B2) 저장 기록에서 최종 top-10 실패를 다시 세고 후보 안/밖으로 나눈다. 모델·DB 호출 없음.

- 정답: 평가 질의 CSV의 `relevant_ids` 전체(정답 여럿이면 하나라도 top-10·후보 30에 있으면 성공) — 기존 평가 계약.
- top-10: detail의 `rerank_top_ids`(실제 리랭킹 뒤 순서)로 다시 계산하고 평가기 `rerank_hit@10`과 176/176 같은지 본다.
- 후보 30: v74 `work_on` 실행 기록(`v74_queries.jsonl`의 후보 id)으로 다시 계산하고, v73 detail의 `candidate_recall@30`과 같은지 본다
  (v74 work_on = v73 B 입력 176/176 같음이 `compare/`에서 확인된 뒤에만 의미가 있다).
- B1과 B2의 질의별 성공·실패가 다르면 그 질의를 따로 적는다(v73에서는 0이었다).

산출 `diagnosis/baseline_failures.csv`(질의 id·집합·정답 수·후보 순위·B1/B2 최종 순위·구분, 원문 없음) · `diagnosis/baseline_failures.json`.

    venv/bin/python build_failure_table.py --v73 <v73 결과 폴더> --run runs/work_on_server_rep1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")


def relevant_of(set_name: str) -> dict:
    out = {}
    with (ROOT / f"experiments/reranking/eval_queries_{set_name}.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["split"].strip() == "dev":
                out[r["query_id"].strip()] = [x.strip() for x in r["relevant_ids"].split("|") if x.strip()]
    return out


def first_rank(ids: list, relevant: list):
    for i, sid in enumerate(ids, start=1):
        if sid in relevant:
            return i
    return None


def read_detail(run_dir: Path, set_name: str) -> dict:
    with (run_dir / f"{set_name}_dev/search_eval_dev_detail.csv").open(encoding="utf-8-sig") as fh:
        return {r["query_id"].strip(): r for r in csv.DictReader(fh)}


def read_candidates(run_dir: Path, set_name: str) -> dict:
    out = {}
    for line in (run_dir / f"{set_name}_dev/v74_queries.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["query_id"]] = [str(c["id"]) for c in rec["candidates"]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v73", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="v74 work_on 실행 폴더(후보 30)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "diagnosis")
    args = ap.parse_args()
    lives = {"B1": args.v73 / "attempt3/live/live_work_server_rep1", "B2": args.v73 / "attempt3/live/live_work_server_rep2"}
    rows, problems = [], []
    for set_name in SETS:
        rel = relevant_of(set_name)
        details = {k: read_detail(v, set_name) for k, v in lives.items()}
        cands = read_candidates(args.run, set_name)
        if set(cands) != set(rel) or any(set(d) != set(rel) for d in details.values()):
            problems.append(f"{set_name}: 질의 목록 불일치")
            continue
        for qid, relevant in rel.items():
            ranks = {}
            for k, d in details.items():
                top = [x for x in d[qid]["rerank_top_ids"].split("|") if x][:10]
                ranks[k] = first_rank(top, relevant)
                hit_eval = d[qid]["rerank_hit@10"].strip()
                if str(int(ranks[k] is not None)) != str(int(float(hit_eval or 0))):
                    problems.append(f"{qid}/{k}: 다시 계산한 hit@10 ≠ 평가기 {hit_eval}")
            cand_rank = first_rank(cands[qid][:30], relevant)
            ev = details["B1"][qid]["candidate_recall@30"].strip()
            if str(int(cand_rank is not None)) != str(int(float(ev or 0))):
                problems.append(f"{qid}: 후보 회수 v74 {cand_rank} ≠ v73 평가기 {ev}")
            hit = {k: v is not None for k, v in ranks.items()}
            if hit["B1"] != hit["B2"]:
                problems.append(f"{qid}: B1·B2 성공 다름")
            if hit["B1"]:
                kind = "hit"
            elif cand_rank is None:
                kind = "out_of_pool"
            else:
                kind = "in_pool_fail"
            rows.append({"set": set_name, "query_id": qid, "relevant_count": len(relevant),
                         "pool_rank": cand_rank or "", "B1_rank": ranks["B1"] or "", "B2_rank": ranks["B2"] or "",
                         "kind": kind})
    counts = {k: sum(1 for r in rows if r["kind"] == k) for k in ("hit", "out_of_pool", "in_pool_fail")}
    summary = {"queries": len(rows), "counts": counts, "pool_recall@30": sum(1 for r in rows if r["pool_rank"]),
               "problems": problems,
               "out_of_pool": [r["query_id"] for r in rows if r["kind"] == "out_of_pool"],
               "in_pool_fail": [r["query_id"] for r in rows if r["kind"] == "in_pool_fail"],
               "multi_relevant_failures": {r["query_id"]: r["relevant_count"] for r in rows
                                           if r["kind"] != "hit" and r["relevant_count"] > 1},
               "source": {k: str(v) for k, v in lives.items()}, "candidates_run": str(args.run)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "baseline_failures.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / "baseline_failures.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                                          encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("queries", "counts", "pool_recall@30", "problems")}, ensure_ascii=False))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
