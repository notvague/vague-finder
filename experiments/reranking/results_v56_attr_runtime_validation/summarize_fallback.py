"""§6 요약 — `fallback_results.csv`를 사례·모드별 표(markdown)로 낸다. 모델·DB 없음.

실행: venv/bin/python experiments/reranking/results_v56_attr_runtime_validation/summarize_fallback.py > fallback_summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_timing(arm: str):
    out = {}
    path = HERE / f"timing_{arm}.jsonl"
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        spans = {s["name"]: s for s in rec.get("spans", [])}
        out[rec["request_id"]] = {
            "total_ms": rec.get("total_ms"),
            "rerank_ms": round(sum(s.get("run_ms", 0.0) for n, s in spans.items() if n == "search.rerank"), 1) if "search.rerank" in spans else "",
            "rerank_shape": (spans.get("search.rerank", {}).get("attrs") or {}).get("shape", ""),
            "lyrics_ms": round(spans["path.lyrics"].get("run_ms", 0.0), 1) if "path.lyrics" in spans else "",
            "search_ms": round(spans["search"].get("run_ms", 0.0), 1) if "search" in spans else "",
        }
    return out


def main() -> int:
    global HERE
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE, help="폴백 CSV와 timing JSONL 폴더")
    HERE = parser.parse_args().input_dir
    rows = list(csv.DictReader((HERE / "fallback_results.csv").open(encoding="utf-8")))
    timing = {"A": load_timing("A"), "D": load_timing("D")}
    print("| arm | 질의 | 사례 | 단계 | 모드 | 응답(ms) | 서버 rerank(ms) · shape | 가사 경로(ms) | 상태 | stage | calls/applied | 가짜 호출[결과] | 정답 순위(후보 top10 순위) | 결과 수/중복/후보 안 | top-3 | v55 리랭킹 전 top-10 일치 | 오류 |")
    print("|---|---|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|")
    first_normal = {}
    for r in rows:
        t = timing[r["arm"]].get(r["request_id"], {})
        ids = (r.get("result_ids") or "").split("|") if r.get("result_ids") else []
        key = (r["arm"], r["query_id"])
        if r["mode"] == "normal" and key not in first_normal:
            first_normal[key] = ids
        same_as_first_normal = "" if r["mode"] != "normal" or key not in first_normal else ("=" if ids == first_normal[key] else "≠")
        rel = r.get("relevant_rank", "")
        relc = r.get("relevant_candidate_rank_top10", "")
        print("| {arm} | {qid} | {kind} | {step} | {mode} | {ms} | {rr} {shape} | {ly} | {st} | {stage} | {calls}/{applied} | {fc}[{fo}] | {rel}{relc} | {n}/{dup}/{inc} | {top3} | {v55} | {err} |".format(
            arm=r["arm"], qid=r["query_id"], kind=r["kind"], step=r["step"], mode=r["mode"] + (same_as_first_normal and f" ({same_as_first_normal}1st normal)"),
            ms=r["client_ms"], rr=t.get("rerank_ms", ""), shape=t.get("rerank_shape", ""), ly=t.get("lyrics_ms", ""),
            st=r["status"], stage=r.get("reorder_stage", ""), calls=r.get("rerank_calls", ""), applied=r.get("rerank_calls_applied", ""),
            fc=r["fake_calls"], fo=r["fake_outcomes"], rel=rel or "-", relc=f"({relc})" if relc else "",
            n=r.get("n_results", ""), dup=(int(r["n_results"]) - int(r["n_unique"])) if r.get("n_results") else "",
            inc=r.get("all_in_candidates", ""), top3="|".join(ids[:3]),
            v55=("equal" if r.get("v55_baseline_top10_equal") == "True" else ("same-set" if r.get("v55_baseline_top10_same_set") == "True" else ("no" if r.get("v55_baseline_top10_equal") else ""))),
            err=(r.get("rerank_error") or r.get("error") or "")[:60].replace("|", "/"),
        ))
    # 폴백 순서가 normal 과 후보 순서 중 어느 쪽인지
    print()
    print("## 폴백 반환 = 후보 순서(candidate_ids 앞 top_k)인가")
    for r in rows:
        if r["status"] != "200" or r["mode"] in ("normal", "slow_within"):
            continue
        ids = (r.get("result_ids") or "").split("|")
        cands = (r.get("candidate_top10") or "").split("|")
        print(f"- {r['arm']} {r['query_id']} {r['mode']}: 결과 top-10 == 후보 top-10 → {ids[:10] == cands[:10]} (stage={r.get('reorder_stage')}, calls={r.get('rerank_calls')}/{r.get('rerank_calls_applied')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
