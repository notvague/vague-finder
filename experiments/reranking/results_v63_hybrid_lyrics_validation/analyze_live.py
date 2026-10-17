"""v63 live 분석 — 완료 검증(compare_v63.load_run) 뒤 지표·반복 변동·A/B 차이·원인 분류·지연·입력 동일성. 모델·DB 호출 없음.

    venv/bin/python analyze_live.py --live-root live --offline-root offline --out live_analysis [--overwrite]

원인 분류(질의 단위, 사전 기준 §1-5-3):
  same            네 실행의 최종 정답 순위가 모두 같다
  arm_consistent  A1=A2, B1=B2, A≠B — 두 쌍에서 같은 방향. 오프라인 입력 차이가 있으면 explained, 없으면 unexplained로 표시
  variable        반복 안에서도 순위가 바뀐다(모델 변동). 오프라인 입력 차이 여부를 함께 적는다
종료 코드: 0 · 3 검증 실패 · 2 덮어쓰기 거부.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_v63 import ValidationError, compare_pair, load_run, metrics, pct, per_query  # noqa: E402
from v63_runtime import LYRIC_QUERY_IDS, TRACE_QUERY_IDS  # noqa: E402

INPUT_FIELDS = ("cand_ids", "cand_scores", "cand_lyric", "lyrics_results", "protected", "tiers", "policy_skip",
                "groups", "group_scores")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-root", default=str(HERE / "live"))
    parser.add_argument("--offline-root", default=str(HERE / "offline"))
    parser.add_argument("--out", default=str(HERE / "live_analysis"))
    parser.add_argument("--runs", default="A1,B1,B2,A2")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"출력 폴더에 이미 결과가 있다: {out}", file=sys.stderr)
        return 2
    names = [n for n in args.runs.split(",") if n]
    try:
        live = {}
        for n in names:
            live[n] = load_run(Path(args.live_root) / f"live_{n[0]}_rep{n[1:]}")
            if live[n]["cond"]["mode"] != "live":
                raise ValidationError(f"{n}: live 실행이 아니다")
        offline = {n: load_run(Path(args.offline_root) / f"offline_{n[0]}_rep{n[1:]}") for n in ("A1", "B1")}
        order = offline["A1"]["order"]
        for n, run in live.items():
            if run["order"] != order:
                raise ValidationError(f"{n}: 질의 집합·순서가 오프라인과 다르다")
            for key in ("lyrics_snapshot", "point_count"):
                if json.dumps(run["cond"][key], sort_keys=True) != json.dumps(offline["A1"]["cond"][key], sort_keys=True):
                    raise ValidationError(f"{n}: {key}가 오프라인과 다르다")
            if run["cond"]["index_fingerprint"]["content_sha256_16"] != offline["A1"]["cond"]["index_fingerprint"]["content_sha256_16"]:
                raise ValidationError(f"{n}: 색인 지문이 오프라인과 다르다")
    except ValidationError as exc:
        print(f"검증 실패: {exc}", file=sys.stderr)
        return 3
    out.mkdir(parents=True, exist_ok=True)

    # 오프라인에서 A/B 입력이 달랐던 질의
    off_pair = compare_pair(offline["A1"], offline["B1"], order)
    offline_input_diff = {r["query_id"]: r["diff_fields"] for r in off_pair["rows"]
                          if set(r["diff_fields"].split("|")) & set(INPUT_FIELDS)}

    # live 입력이 같은 arm의 오프라인과 같은가(후보·보호·그룹 + 첫 시도 프롬프트 sha)
    input_identity = {}
    for n, run in live.items():
        off = offline[f"{n[0]}1"]
        diffs = []
        prompt_diffs = []
        for q in order:
            pl, po = per_query(run["queries"][q]), per_query(off["queries"][q])
            if any(pl[f] != po[f] for f in INPUT_FIELDS):
                diffs.append(q)
            # 그룹마다 첫 시도 프롬프트 — 재시도는 정정 지시가 붙어 다르다. 오프라인은 그룹당 1회
            live_first = []
            gens = run["queries"][q]["rec"].get("generates", [])
            runs_ = run["queries"][q]["rec"].get("rerank_runs", [])
            cursor = 0
            for g in runs_:
                calls = int(g.get("model_calls") or 0)
                if calls:
                    live_first.append(gens[cursor]["contents_sha16"] if cursor < len(gens) else None)
                else:
                    live_first.append(None)
                cursor += calls
            off_first = [g["contents_sha16"] for g in off["queries"][q]["rec"].get("generates", [])]
            if [x for x in live_first if x is not None] != off_first[: len([x for x in live_first if x is not None])]:
                prompt_diffs.append(q)
        input_identity[n] = {"input_field_diff_vs_offline": diffs, "first_prompt_sha_diff_vs_offline": prompt_diffs}

    rows = []
    for q in order:
        ranks = {n: per_query(live[n]["queries"][q])["rerank_rank"] for n in names}
        r = {n: (ranks[n] if ranks[n] is not None else 99) for n in names}
        if len(set(r.values())) == 1:
            cls = "same"
        elif all(k in r for k in ("A1", "A2", "B1", "B2")) and r["A1"] == r["A2"] and r["B1"] == r["B2"]:
            cls = "arm_consistent"
        else:
            cls = "variable"
        explained = q in offline_input_diff
        p = {n: per_query(live[n]["queries"][q]) for n in names}
        rows.append({
            "query_id": q, "set": live[names[0]]["queries"][q]["set"], "lyric_query": int(q in LYRIC_QUERY_IDS),
            "cand_rank": p[names[0]]["cand_rank"] or "",
            **{f"{n}_rank": ranks[n] or "" for n in names},
            **{f"{n}_outcome": p[n]["outcome"] for n in names},
            **{f"{n}_groups": p[n]["statuses"] for n in names},
            "class": cls,
            "offline_input_diff": offline_input_diff.get(q, ""),
            "explained_by_input": int(explained) if cls != "same" else "",
        })
    with (out / "per_query.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lyric = [q for q in order if q in LYRIC_QUERY_IDS]
    non_lyric = [q for q in order if q not in LYRIC_QUERY_IDS]
    summary = {"runs": {}, "pairs": {}, "classes": {}, "input_identity": input_identity,
               "offline_input_diff_queries": offline_input_diff}
    for n in names:
        run = live[n]
        first_q = order[0]
        rows_q = [per_query(run["queries"][q]) for q in order]
        warm_total = [x["retrieval_ms"] + x["rerank_ms"] for x in rows_q[1:]]
        warm_ret = [x["retrieval_ms"] for x in rows_q[1:]]
        summary["runs"][n] = {
            "all_176": metrics(run, order), "lyric_19": metrics(run, lyric), "non_lyric_157": metrics(run, non_lyric),
            "cold": {"process_to_first_query_s": run["cond"].get("cold_before_first_query_seconds"),
                     "first_query_retrieval_ms": per_query(run["queries"][first_q])["retrieval_ms"],
                     "qdrant_open_s": run["cond"].get("qdrant_open_seconds"),
                     "lyrics_snapshot_builds": run["cond"]["sets_done"][list(run["cond"]["sets_done"])[-1]].get("lyrics_snapshot_builds")},
            "warm_175": {"retrieval_ms_median": round(statistics.median(warm_ret), 1),
                         "retrieval_ms_p95": round(pct(warm_ret, 0.95), 1),
                         "total_ms_median": round(statistics.median(warm_total), 1),
                         "total_ms_p95": round(pct(warm_total, 0.95), 1)},
            "lyrics_service_status": run["cond"].get("lyrics_service_status"),
            "lyrics_source_reads": run["cond"].get("lyrics_source_reads"),
            "budget_local": run["cond"].get("budget_local"),
            "started_at": run["cond"].get("started_at"), "finished_at": run["cond"].get("finished_at"),
            "total_seconds": run["cond"].get("total_seconds"),
        }
    for a, b in (("A1", "B1"), ("A2", "B2"), ("A1", "A2"), ("B1", "B2"), ("A1", "B2"), ("A2", "B1")):
        if a in live and b in live:
            res = compare_pair(live[a], live[b], order)
            res.pop("rows")
            res.pop("field_diff_counts")
            summary["pairs"][f"{a}:{b}"] = res
    for cls in ("same", "arm_consistent", "variable"):
        qs = [r for r in rows if r["class"] == cls]
        summary["classes"][cls] = {
            "count": len(qs), "lyric": sum(r["lyric_query"] for r in qs),
            "query_ids": [r["query_id"] for r in qs] if cls != "same" else None,
            "explained_by_input": [r["query_id"] for r in qs if r["explained_by_input"] == 1] if cls != "same" else None,
        }
    summary["trace"] = {q: {n: {k: per_query(live[n]["queries"][q])[k] for k in
                                ("cand_rank", "base_rank", "rerank_rank", "outcome", "statuses", "protected", "groups",
                                 "final_ids", "hybrid_ref_bad", "retrieval_ms", "rerank_ms")} for n in names}
                        for q in TRACE_QUERY_IDS}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for n, r in summary["runs"].items():
        m = r["all_176"]
        l = r["lyric_19"]
        print(f"{n}: cand {m['candidate_recall@30']} H1/5/10 {m['hit@1']}/{m['hit@5']}/{m['hit@10']} MRR {m['mrr@10']} "
              f"| lyric19 {l['hit@1']}/{l['hit@5']}/{l['hit@10']} {l['mrr@10']} | outcomes {m['outcomes']} groups {m['group_status']} "
              f"| total med/p95 {m['total_ms_median']}/{m['total_ms_p95']} rerank {m['rerank_ms_median']}/{m['rerank_ms_p95']}")
    for k, p in summary["pairs"].items():
        print(f"{k}: rank-diff {len(p['final_rank_diff_queries'])} {p['final_rank_diff_queries']} gains {p['hit10_gains_b_over_a']} losses {p['hit10_losses_b_vs_a']}")
    print("classes", {k: (v["count"], v["query_ids"], v["explained_by_input"]) for k, v in summary["classes"].items()})
    print("input identity", input_identity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
