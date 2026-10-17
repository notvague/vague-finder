"""소리 경로 구성 비교(리랭킹 전) — 후보 회수@30 · 질의별 정답 후보 순위 · 후보 집합 변화 · 보호·정책 · 소리 실행 · 지연.

모델 호출 없음. 가짜 리랭커의 최종 순서는 읽지 않는다. v66 실제 실행(후1)의 최종 순위는 **A와 입력이 같은 질의에만** 붙인다.

    venv/bin/python analyze_ab.py --runs A1=runs/on_rep1,A2=runs/on_rep2,B1=runs/skip_rep1,... --base A1 \
        --v66-live <v66 live_after_server_rep1> --sound-types sound_types.csv --out-dir analysis

검사: 질의 ID 누락·중복, 세트 건수, 정답 여럿(첫 정답 기준) · 반복 쌍 결정성은 compare_runs.py가 맡는다.
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


def gold() -> Dict[str, Dict[str, Any]]:
    out = {}
    for s in SETS:
        rows = [r for r in csv.DictReader(open(ROOT / f"experiments/reranking/eval_queries_{s}.csv", encoding="utf-8-sig"))
                if r["split"].strip() == "dev"]
        if len(rows) != EXPECTED[s]:
            raise SystemExit(f"{s}: dev {len(rows)}")
        for r in rows:
            qid = r["query_id"].strip()
            if qid in out:
                raise SystemExit(f"질의 ID 중복 {qid}")
            rel = [x.strip() for x in r["relevant_ids"].split("|") if x.strip()]
            if not rel:
                raise SystemExit(f"{qid}: 정답 없음")
            out[qid] = {"set": s, "relevant": rel, "lyric": r.get("query_type", "")}
    return out


def load(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"미완료 {run_dir}")
    out = {}
    for s in SETS:
        for line in open(run_dir / f"{s}_dev" / "v67_queries.jsonl", encoding="utf-8"):
            r = json.loads(line)
            out[r["query_id"]] = r
        for r in csv.DictReader(open(run_dir / f"{s}_dev" / "search_eval_dev_detail.csv", encoding="utf-8-sig")):
            out[r["query_id"]]["detail"] = r
    return out


def first_rank(ids: List[str], relevant: List[str]) -> Optional[int]:
    return next((i for i, x in enumerate(ids, 1) if x in set(relevant)), None)


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--v66-live", type=Path, required=True)
    ap.add_argument("--sound-types", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    runs = {k: load(HERE / v if not Path(v).is_absolute() else Path(v))
            for k, v in (x.split("=") for x in args.runs.split(","))}
    g = gold()
    for k, r in runs.items():
        if set(r) != set(g):
            raise SystemExit(f"{k}: 질의 집합이 정답 파일과 다르다")
    v66 = {}
    for s in SETS:
        for r in csv.DictReader(open(args.v66_live / f"{s}_dev" / "search_eval_dev_detail.csv", encoding="utf-8-sig")):
            v66[r["query_id"]] = r
    stypes = {}
    if args.sound_types and args.sound_types.exists():
        stypes = {r["query_id"]: r["sound_type"] for r in csv.DictReader(open(args.sound_types, encoding="utf-8-sig"))}
    base = runs[args.base]

    rows = []
    for qid, info in g.items():
        b = base[qid]
        bids = [c["id"] for c in b["candidates"]]
        v66_final = v66[qid]["rerank_rank"]
        v66_cand = v66[qid]["candidate_rank@30"]
        out = {"query_id": qid, "set": info["set"], "relevant_count": len(info["relevant"]),
               "lyric_query": int(bool(b.get("lyrics"))),
               "audio_ran_base": int(b["audio_queries"] > 0),
               "sound_type": stypes.get(qid, ""),
               "v66_final_rank": v66_final, "v66_cand_rank": v66_cand}
        # v66 후1 후보 순위와 기준 실행의 후보 순위가 같아야 한다(A=v66 재현 검사의 한 줄)
        if str(first_rank(bids, info["relevant"]) or "") != (v66_cand or ""):
            raise SystemExit(f"{qid}: 기준 실행 후보 순위 ≠ v66 후1")
        for k, r in runs.items():
            row = r[qid]
            ids = [c["id"] for c in row["candidates"]]
            out[f"cand_{k}"] = first_rank(ids, info["relevant"]) or ""
            out[f"pre10_{k}"] = int(bool(first_rank(ids[:10], info["relevant"])))
            out[f"audio_q_{k}"] = row["audio_queries"]
            out[f"search_ms_{k}"] = row["search_ms"]
            if k != args.base:
                out[f"jacc_{k}"] = round(jaccard(bids, ids), 4)
                out[f"new_ids_{k}"] = len(set(ids) - set(bids))
                out[f"same_inputs_{k}"] = int(
                    ids == bids and [c["score"] for c in row["candidates"]] == [c["score"] for c in b["candidates"]]
                    and [(x["input_ids"], x["input_scores"]) for x in row["rerank_runs"]]
                    == [(x["input_ids"], x["input_scores"]) for x in b["rerank_runs"]])
                out[f"policy_skip_{k}"] = (row["protection"] or {}).get("policy_skip", "")
                out[f"protected_same_{k}"] = int((row["protection"] or {}).get("protected_ids") ==
                                                 (b["protection"] or {}).get("protected_ids"))
        out["policy_skip_base"] = (b["protection"] or {}).get("policy_skip", "")
        rows.append(out)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "per_query_ab.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary: Dict[str, Any] = {"queries": len(rows), "base": args.base, "configs": {}}
    for k in runs:
        recall = sum(1 for r in rows if r[f"cand_{k}"] != "")
        pre10 = sum(r[f"pre10_{k}"] for r in rows)
        ms = [r[f"search_ms_{k}"] for r in rows]
        ms_audio = [r[f"search_ms_{k}"] for r in rows if r["audio_ran_base"]]
        ent: Dict[str, Any] = {
            "candidate_recall@30": recall, "pre_rerank_hit@10": pre10,
            "by_set": {s: sum(1 for r in rows if r["set"] == s and r[f"cand_{k}"] != "") for s in SETS},
            "queries_with_audio_query": sum(1 for r in rows if r[f"audio_q_{k}"] > 0),
            "audio_queries_total": sum(r[f"audio_q_{k}"] for r in rows),
            "search_ms_median": round(statistics.median(ms), 1), "search_ms_p95": round(pct(ms, 0.95), 1),
            "search_ms_median_audio_queries": round(statistics.median(ms_audio), 1) if ms_audio else None,
            "search_ms_p95_audio_queries": round(pct(ms_audio, 0.95), 1) if ms_audio else None,
        }
        if k != args.base:
            gained = [r["query_id"] for r in rows if r[f"cand_{k}"] != "" and r[f"cand_{args.base}"] == ""]
            lost = [r["query_id"] for r in rows if r[f"cand_{k}"] == "" and r[f"cand_{args.base}"] != ""]
            up = [r["query_id"] for r in rows if r[f"cand_{k}"] != "" and r[f"cand_{args.base}"] != ""
                  and int(r[f"cand_{k}"]) < int(r[f"cand_{args.base}"])]
            down = [r["query_id"] for r in rows if r[f"cand_{k}"] != "" and r[f"cand_{args.base}"] != ""
                    and int(r[f"cand_{k}"]) > int(r[f"cand_{args.base}"])]
            ent.update({
                "gained": gained, "lost": lost, "net": len(gained) - len(lost),
                "rank_up_in_pool": up, "rank_down_in_pool": down,
                "candidate_set_changed": sum(1 for r in rows if r[f"jacc_{k}"] < 1),
                "inputs_same_as_base": sum(r[f"same_inputs_{k}"] for r in rows),
                "policy_skip_changed": [r["query_id"] for r in rows if r[f"policy_skip_{k}"] != r["policy_skip_base"]],
                "protected_changed": [r["query_id"] for r in rows if not r[f"protected_same_{k}"]],
                # 기존 v66 최종 top-10 정답 중 이 구성에서 후보 순위가 바뀐(=리랭커 입력이 바뀐) 질의 — 실제 평가로만 결말을 안다
                "v66_hits_with_changed_inputs": [r["query_id"] for r in rows
                                                 if r["v66_final_rank"] and int(r["v66_final_rank"]) <= 10
                                                 and not r[f"same_inputs_{k}"]],
            })
            if stypes:
                by: Dict[str, Dict[str, int]] = {}
                for r in rows:
                    t = r["sound_type"] or "(소리 경로 안 돎)"
                    e = by.setdefault(t, {"queries": 0, "gained": 0, "lost": 0, "set_changed": 0})
                    e["queries"] += 1
                    e["gained"] += r["query_id"] in gained
                    e["lost"] += r["query_id"] in lost
                    e["set_changed"] += r[f"jacc_{k}"] < 1
                ent["by_sound_type"] = by
        summary["configs"][k] = ent
    (args.out_dir / "summary_ab.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, e in summary["configs"].items():
        print(k, e["candidate_recall@30"], e["by_set"], "pre10", e["pre_rerank_hit@10"],
              {x: e[x] for x in ("gained", "lost", "net", "candidate_set_changed", "inputs_same_as_base") if x in e})


if __name__ == "__main__":
    main()
