"""v77 채점기 — 정답 라벨은 여기서만 읽는다(검색이 끝난 뒤).

    PYTHONPATH=<worktree> venv/bin/python score.py --run R1 [--compare R2]

출력(커밋 대상, 원문 없음): `analysis/per_query.csv` · `analysis/summary.json` · `analysis/repeat_R1_vs_R2.json`.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desc_semantic as ds  # noqa: E402
import retrieve as rv  # noqa: E402  (v76 모듈 경로만 재사용)

DESIGN_SEEN = {"q115", "q117", "m302", "m303", "m403", "c602", "c603", "t038", "n046", "n064", "t009", "t060",
               "t073", "t075", "t079", "t080", "t092", "t103", "n011", "t100", "q110"}
HYP_SEED = {"c602", "c603", "t038"}


def parse_relevant_ids(raw: str) -> set:
    return {p.strip() for p in (raw or "").split("|") if p.strip()}


def load_gold() -> dict:
    gold = {}
    for s in ds.SETS:
        with (ds.ROOT / f"experiments/reranking/eval_queries_{s}.csv").open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row["split"] == "dev":
                    gold[row["query_id"]] = parse_relevant_ids(row["relevant_ids"])
    return gold


def best_group_rank(ranked_groups, gold_groups):
    for i, g in enumerate(ranked_groups, 1):
        if g in gold_groups:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R1")
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()
    run_dir = HERE / "runs" / args.run
    recs = ds.read_jsonl_gz(run_dir / "rankings.jsonl.gz")
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))

    songs, groups, _ = ds.load_corpus()
    chunk_rows = ds.read_jsonl_gz(ds.INDEX_DIR / "chunks.jsonl.gz")
    chunk_text = {(r["group"], r["chunk"]): r["text"] for r in chunk_rows}
    _mod, _store, v76_song_group = rv._load_v76()  # noqa: SLF001
    cands = json.loads((HERE / "inputs/v75A_candidates.json").read_text(encoding="utf-8"))["candidates"]
    hit10 = json.loads((HERE / "inputs/v75A_hit10.json").read_text(encoding="utf-8"))["v75A_hit10"]
    gold = load_gold()  # ← 정답은 여기서 처음 읽는다
    if set(gold) != {r["qid"] for r in recs} or len(recs) != 176:
        raise SystemExit("질의 집합이 dev 176과 다르다")

    rows = []
    for r in recs:
        qid = r["qid"]
        g = gold[qid]
        cand = cands[qid]
        cand_rank = next((i for i, s in enumerate(cand, 1) if s in g), None)
        if hit10[qid]:
            split = "success"
        elif cand_rank is None:
            split = "out_fail"
        else:
            split = "in_fail"
        gold_groups = {songs[s].group for s in g if s in songs and songs[s].group}
        missing_gold = sorted(s for s in g if s not in songs)
        nodesc = [s for s in g if s in songs and songs[s].group is None]
        row = {
            "set": r["set"], "qid": qid, "split": split, "design_seen": int(qid in DESIGN_SEEN),
            "hyp_seed": int(qid in HYP_SEED), "status": r["status"], "exact_lyric": int(bool(r.get("exact_lyric"))),
            "n_gold": len(g), "gold_not_in_corpus": len(missing_gold),
            "gold_without_description": len(nodesc),
            "searchable": int(bool(gold_groups)),
            "v75A_hit10": int(bool(hit10[qid])), "v75A_cand_rank": cand_rank or "",
        }
        if r["status"] != "ok":
            rows.append(row)
            continue
        for path in ("S", "L"):
            ranked = [x[0] for x in r[path]]
            grank = best_group_rank(ranked, gold_groups)
            reps = r[f"{path}_reps"]
            rep_ids = [x[1] for x in reps]
            rep_rank = next((i for i, s in enumerate(rep_ids, 1) if s in g), None)
            expand = [s for gid in ranked[: ds.GROUPS_K] for s in groups[gid].songs]
            new = [s for s in rep_ids if s and s not in cand]
            row.update({
                f"{path}_group_rank": grank or "",
                f"{path}_ranked_groups": len(ranked),
                f"{path}_in_expand10": int(any(s in g for s in expand)),
                f"{path}_expand10_songs": len(expand),
                f"{path}_rep10_rank": rep_rank or "",
                f"{path}_new10": len(new),
                f"{path}_new10_gold": sum(1 for s in new if s in g),
            })
            if path == "S":
                # 정답 묶음(위 100 안 가장 좋은 것)의 근거 성격과 그 묶음 대표곡 규칙
                kind = rule = ""
                gsize = ""
                if grank:
                    gid, _sc, ci = r["S"][grank - 1]
                    gsize = len(groups[gid].songs)
                    text = chunk_text[(gid, ci)]
                    gold_in_group = [s for s in groups[gid].songs if s in g]
                    if gsize == 1:
                        kind = "unique_description"
                    elif any(ds.title_in_chunk(songs[s], text) for s in gold_in_group):
                        kind = "title_in_chunk"
                    else:
                        kind = "album_common"
                    if grank <= ds.GROUPS_K:
                        rule = reps[grank - 1][2]
                        row["S_gold_group_rep_is_gold"] = int(reps[grank - 1][1] in g)
                row.update({"S_gold_group_size": gsize, "S_evidence_kind": kind, "S_gold_group_rep_rule": rule,
                            "S_top1_score": r["S"][0][1] if r["S"] else ""})
        v_ranked = [x[0] for x in r["V"]]
        v_gold_groups = {v76_song_group[s] for s in g if s in v76_song_group}
        row["V_group_rank"] = best_group_rank(v_ranked, v_gold_groups) or ""
        row["V_evidence_songs"] = r["V_evidence_songs"]
        row["encode_ms"] = r["ms"]["encode"]
        row["S_search_ms"] = r["ms"]["S_search"]
        rows.append(row)

    cols = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    (HERE / "analysis").mkdir(exist_ok=True)
    with (HERE / "analysis/per_query.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    def within(rows_, key, k):
        return sum(1 for x in rows_ if isinstance(x.get(key), int) and x[key] <= k)

    splits = {
        "all": rows, "out_fail": [x for x in rows if x["split"] == "out_fail"],
        "success": [x for x in rows if x["split"] == "success"], "in_fail": [x for x in rows if x["split"] == "in_fail"],
        "exact_lyric": [x for x in rows if x["exact_lyric"]],
    }
    summary = {"run": args.run, "counts": {k: len(v) for k, v in splits.items()}, "by_split": {}}
    # 후보 밖 보조 후보: 질의별 출현(_total)과 별도로 고유 곡 수를 센다(같은 곡이 여러 질의에 나오면 출현은 여러 번)
    new_by_q = {p: {r["qid"]: {x[1] for x in r[f"{p}_reps"] if x[1] and x[1] not in cands[r["qid"]]}
                    for r in recs if r["status"] == "ok"} for p in ("S", "L")}
    for name, part in splits.items():
        d = {"searchable": sum(x["searchable"] for x in part),
             "gold_without_description_all": sum(1 for x in part if not x["searchable"])}
        for path in ("S", "L", "V"):
            d[f"{path}_group_at"] = {k: within(part, f"{path}_group_rank", k) for k in ds.CURVE_DEPTHS}
        for path in ("S", "L"):
            d[f"{path}_rep10"] = within(part, f"{path}_rep10_rank", ds.GROUPS_K)
            d[f"{path}_expand10"] = sum(x.get(f"{path}_in_expand10", 0) for x in part)
            d[f"{path}_new10_total"] = sum(x.get(f"{path}_new10", 0) for x in part)
            d[f"{path}_new10_gold"] = sum(x.get(f"{path}_new10_gold", 0) for x in part)
            d[f"{path}_new10_distinct_songs"] = len(set().union(*[new_by_q[path].get(x["qid"], set()) for x in part]))
            ex = [x.get(f"{path}_expand10_songs", 0) for x in part if x["status"] == "ok"]
            d[f"{path}_expand10_songs_median"] = st.median(ex) if ex else None
            d[f"{path}_expand10_songs_max"] = max(ex) if ex else None
        d["S_evidence_kind_at10"] = dict(Counter(x["S_evidence_kind"] for x in part
                                                 if isinstance(x.get("S_group_rank"), int) and x["S_group_rank"] <= 10))
        d["S_gold_group_rep_rule_at10"] = dict(Counter(x["S_gold_group_rep_rule"] for x in part if x.get("S_gold_group_rep_rule")))
        summary["by_split"][name] = d

    out_fail = splits["out_fail"]
    e1_list = [x["qid"] for x in out_fail if isinstance(x.get("S_group_rank"), int) and x["S_group_rank"] <= 10]
    l1_list = [x["qid"] for x in out_fail if isinstance(x.get("L_group_rank"), int) and x["L_group_rank"] <= 10]
    e2_list = [x["qid"] for x in out_fail if isinstance(x.get("S_rep10_rank"), int)]
    summary["out_fail_detail"] = sorted(
        ({"qid": x["qid"], "design_seen": x["design_seen"], "searchable": x["searchable"],
          "S_group_rank": x.get("S_group_rank"), "L_group_rank": x.get("L_group_rank"), "V_group_rank": x.get("V_group_rank"),
          "S_rep10_rank": x.get("S_rep10_rank"), "S_in_expand10": x.get("S_in_expand10"),
          "S_evidence_kind": x.get("S_evidence_kind"), "S_gold_group_size": x.get("S_gold_group_size")}
         for x in out_fail), key=lambda d: (d["S_group_rank"] if isinstance(d["S_group_rank"], int) else 999, d["qid"]))

    # 집중도(S 위 10묶음, 176질의)
    ok = [r for r in recs if r["status"] == "ok"]
    freq = Counter(x[0] for r in ok for x in r["S"][:10])
    top5 = [g for g, _ in freq.most_common(5)]
    rep_freq = Counter(x[1] for r in ok for x in r["S_reps"] if x[1])
    summary["concentration_S_top10"] = {
        "distinct_groups": len(freq), "slots": sum(freq.values()),
        "max_group_frequency": freq.most_common(1)[0][1] if freq else 0,
        "top5_group_frequencies": [freq[g] for g in top5],
        "queries_with_any_top5_group": sum(1 for r in ok if any(x[0] in top5 for x in r["S"][:10])),
        "top5_group_sizes": [len(groups[g].songs) for g in top5],
        "distinct_rep_songs": len(rep_freq), "max_rep_song_frequency": rep_freq.most_common(1)[0][1] if rep_freq else 0,
        "groups_appearing_in_ge_10pct_queries": sum(1 for v in freq.values() if v >= 0.1 * len(ok)),
    }
    tm = meta["timing"]
    summary["cost"] = {"encode_ms_median": tm["encode_ms_median"], "S_search_ms_median": tm["S_search_ms_median"],
                       "S_search_ms_p90": tm["S_search_ms_p90"], "index_matrix_mb": meta["index_matrix_mb"],
                       "rss_index_increase_mb": meta["rss_mb"]["index_increase"], "model_device": meta["model"]["device"]}
    e3_e1 = [q for q in e1_list if q not in DESIGN_SEEN]
    e3_e2 = [q for q in e2_list if q not in DESIGN_SEEN]
    summary["gates"] = {
        "E1": {"S_group_at10_out_fail": e1_list, "L_group_at10_out_fail": l1_list,
               "pass": len(e1_list) >= 3 and len(e1_list) > len(l1_list)},
        "E2": {"S_rep10_out_fail": e2_list, "pass": len(e2_list) >= 3},
        "E3": {"E1_not_design_seen": e3_e1, "E2_not_design_seen": e3_e2,
               "pass": len(e3_e1) >= 2 and len(e3_e2) >= 2},
        "E4": {"S_search_ms_median": tm["S_search_ms_median"], "index_matrix_mb": meta["index_matrix_mb"],
               "pass": tm["S_search_ms_median"] <= 50 and meta["index_matrix_mb"] <= 200},
    }
    if args.compare:
        other = {r["qid"]: r for r in ds.read_jsonl_gz(HERE / "runs" / args.compare / "rankings.jsonl.gz")}
        diff_order = {p: [] for p in ("S", "S_reps", "L", "L_reps", "V")}
        max_score_diff = 0.0
        for r in recs:
            o = other[r["qid"]]
            if r["status"] != "ok":
                continue
            for p in ("S", "L"):
                if [x[0] for x in r[p]] != [x[0] for x in o[p]]:
                    diff_order[p].append(r["qid"])
                for a, b in zip(r[p], o[p]):
                    max_score_diff = max(max_score_diff, abs(a[1] - b[1]))
            for p in ("S_reps", "L_reps", "V"):
                if r[p] != o[p]:
                    diff_order[p].append(r["qid"])
        rep = {"a": args.run, "b": args.compare, "differs": diff_order,
               "identical_all": all(not v for v in diff_order.values()), "max_abs_score_diff": max_score_diff}
        (HERE / f"analysis/repeat_{args.run}_vs_{args.compare}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        summary["repeat"] = rep
    summary["gates"]["E0"] = {"status_counts": meta["status_counts"], "external_calls": meta["external_calls"],
                              "repeat_identical": summary.get("repeat", {}).get("identical_all"),
                              "pass": meta["status_counts"].get("ok") == 176 and meta["external_calls"] == 0
                              and summary.get("repeat", {}).get("identical_all") is True}
    (HERE / "analysis/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"by_split_out_fail": summary["by_split"]["out_fail"], "gates": summary["gates"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
