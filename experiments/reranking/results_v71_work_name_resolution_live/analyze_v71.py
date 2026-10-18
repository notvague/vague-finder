"""v71 집계 — 실제 해석 반복 비교 + 오프라인 후보 비교(A·B1·B2). 모델·DB 호출 없음, 저장된 기록만 읽는다.

추적 산출물(질의 원문·작품명·근거 구절 없음): `analysis/live_summary.json` · `analysis/offline_summary.json` · `analysis/per_query.csv`.
비추적(Git 제외): `analysis_private/model_output_review.csv` — 사람 라벨이 들어온 뒤 대조용(작품명·근거 구절 포함).

    venv/bin/python analyze_v71.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
KNOWN = {"c602", "c603", "c607", "c701", "n033", "n043", "n096", "t043", "c705", "c707"}


def norm(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", s).split()).lower()


def jl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_rank(ids: List[str], rel: List[str]) -> Optional[int]:
    for i, x in enumerate(ids, 1):
        if x in rel:
            return i
    return None


def med(xs: List[float]) -> Optional[float]:
    return round(st.median(xs), 1) if xs else None


def live_summary(raw: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rep, rows in raw.items():
        u = [r.get("usage") or {} for r in rows if r["status"] == "ok"]
        cond = json.loads((HERE / f"live/rep{rep}/run_conditions.json").read_text(encoding="utf-8"))
        out[f"rep{rep}"] = {
            "rows": len(rows),
            "status": {s: sum(1 for r in rows if (r["status"] if r["status"] != "ok" else f"ok:{r['parsed_status']}") == s)
                       for s in sorted({r["status"] if r["status"] != "ok" else f"ok:{r['parsed_status']}" for r in rows})},
            "works_returned": sum(len(r.get("works") or []) for r in rows),
            "queries_with_2plus_works": sum(1 for r in rows if len(r.get("works") or []) >= 2),
            "truncated_over_3": sum(1 for r in rows if r.get("truncated")),
            "finish_reasons": sorted({r.get("finish_reason", "") for r in rows}),
            "thoughts_tokens": sorted({str(x.get("thoughts")) for x in u}),
            "candidates_tokens_max": max((x.get("candidates") or 0) for x in u) if u else None,
            "prompt_tokens_range": [min(x["prompt"] for x in u), max(x["prompt"] for x in u)] if u else None,
            "latency_ms": {"median": med([r["latency_ms"] for r in rows if "latency_ms" in r]),
                           "max": max((r["latency_ms"] for r in rows if "latency_ms" in r), default=None)},
            "model_versions": sorted({str(r.get("model_version")) for r in rows}),
            "ledger_after": {k: cond["ledger_after"][k] for k in ("attempts", "attempts_by_kind", "settled_usd", "unverified_usd",
                                                                  "open_reserve_usd", "committed_usd", "failed_calls", "halt")},
            "transport_requests": cond["transport_requests"], "preflight": cond["preflight"], "stopped": cond["stopped"],
            "elapsed_seconds": cond["elapsed_seconds"], "input_fingerprint16": cond["input_fingerprint16"],
            "cache_sha16": cond["cache_sha16"], "leak_check": cond["leak_check"],
        }
    a, b = raw[1], raw[2]
    names = lambda r: sorted({norm(w["name"]) for w in r.get("works") or []})  # noqa: E731
    out["rep1_vs_rep2"] = {
        "raw_text_identical": sum(x.get("text") == y.get("text") for x, y in zip(a, b)),
        "request_sha_identical": sum(x.get("request_sha16") == y.get("request_sha16") for x, y in zip(a, b)),
        "parsed_status_identical": sum(x.get("parsed_status") == y.get("parsed_status") for x, y in zip(a, b)),
        "normalized_work_names_identical": sum(names(x) == names(y) for x, y in zip(a, b)),
        "of": len(a),
    }
    return out


def offline(raw: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    recs = {c: jl(HERE / f"offline/{c}/records.jsonl") for c in ("off", "live_rep1", "live_rep2")}
    A = recs["off"]
    summary: Dict[str, Any] = {"denominator": len(A)}
    per_query: Dict[str, Dict[str, Any]] = {r["qid"]: {"set": r["set"], "qid": r["qid"], "gate": r["gate"]["triggered"],
                                                        "known_case": r["qid"] in KNOWN} for r in A}
    for r in A:
        per_query[r["qid"]]["A_rank"] = first_rank([c["id"] for c in r["candidates"]], r["relevant"])
    summary["A"] = {"recall_at_30": sum(1 for q in per_query.values() if q["A_rank"]), "search_ms_median_triggered":
                    med([r["search_ms"] for r in A if r["gate"]["triggered"]])}
    for tag, cfg in (("B1", "live_rep1"), ("B2", "live_rep2")):
        B = recs[cfg]
        assert [r["qid"] for r in B] == [r["qid"] for r in A]
        gained, lost, up, down = [], [], [], []
        statuses: Dict[str, int] = {}
        reject: Dict[str, int] = {}
        checks = {"non_triggered_identical": 0, "non_triggered": 0, "not_applied_identical": 0, "not_applied": 0,
                  "applied_size_kept": 0, "applied": 0, "entered_eq_displaced": 0, "lyric_protection_kept": 0,
                  "lyric_protected_queries": 0}
        entered_total = displaced_total = aux_total = aux_outside = gold_displaced = 0
        name_in_query = explicit_rejects = accepted_works = 0
        for ra, rb in zip(A, B):
            q = per_query[ra["qid"]]
            ids_a = [c["id"] for c in ra["candidates"]]
            ids_b = [c["id"] for c in rb["candidates"]]
            wr = rb.get("work_resolution") or {}
            status = wr.get("status", "none") if rb["gate"]["triggered"] else "not_triggered"
            statuses[status] = statuses.get(status, 0) + 1
            same = ra["candidates"] == rb["candidates"]
            if not rb["gate"]["triggered"]:
                checks["non_triggered"] += 1
                checks["non_triggered_identical"] += same
            elif status != "applied":
                checks["not_applied"] += 1
                checks["not_applied_identical"] += same
            else:
                checks["applied"] += 1
                checks["applied_size_kept"] += len(ids_b) == len(ids_a)
                ent, dis = wr.get("entered_ids") or [], wr.get("displaced_ids") or []
                checks["entered_eq_displaced"] += len(ent) == len(dis)
                entered_total += len(ent)
                displaced_total += len(dis)
                gold_displaced += sum(1 for x in dis if x in ra["relevant"])
                aux = wr.get("aux_ids") or []
                aux_total += len(aux)
                outside = [x for x in aux if x not in ids_b]
                aux_outside += len(outside)
                q[f"{tag}_aux_size"] = len(aux)
                q[f"{tag}_aux_outside_30"] = len(outside)
                q[f"{tag}_entered"] = len(ent)
                q[f"{tag}_gold_aux_rank"] = first_rank(aux, ra["relevant"])
            if ra["protected_lyric_ids"]:
                checks["lyric_protected_queries"] += 1
                checks["lyric_protection_kept"] += set(ra["protected_lyric_ids"]) <= set(rb["protected_lyric_ids"])
            works = wr.get("works") or []
            accepted_works += sum(1 for w in works if w.get("accepted"))
            name_in_query += sum(1 for w in works if w.get("name_in_query"))
            for w in works:
                if not w.get("accepted"):
                    reason = str(w.get("reject_reason") or "?")
                    reject[reason] = reject.get(reason, 0) + 1
                    explicit_rejects += reason == "explicit_work_named"
            rank_b = first_rank(ids_b, ra["relevant"])
            q[f"{tag}_status"] = status
            q[f"{tag}_rank"] = rank_b
            q[f"{tag}_works"] = len(works)
            q[f"{tag}_accepted"] = sum(1 for w in works if w.get("accepted"))
            q[f"{tag}_mentions"] = ";".join(str(w.get("mention_songs", "")) for w in works)
            q[f"{tag}_reject"] = ";".join(str(w.get("reject_reason") or "") for w in works if not w.get("accepted"))
            q[f"{tag}_name_in_query"] = sum(1 for w in works if w.get("name_in_query"))
            q[f"{tag}_same_as_A"] = same
            if q["A_rank"] is None and rank_b is not None:
                gained.append({"qid": ra["qid"], "rank": rank_b})
            elif q["A_rank"] is not None and rank_b is None:
                lost.append({"qid": ra["qid"], "A_rank": q["A_rank"]})
            elif q["A_rank"] and rank_b and rank_b != q["A_rank"]:
                (up if rank_b < q["A_rank"] else down).append({"qid": ra["qid"], "A": q["A_rank"], "B": rank_b})
        summary[tag] = {
            "recall_at_30": sum(1 for q in per_query.values() if q.get(f"{tag}_rank")),
            "gained": gained, "lost": lost, "rank_up": up, "rank_down": down,
            "status_counts": statuses, "reject_reasons": reject, "accepted_works": accepted_works,
            "name_in_query_works": name_in_query, "explicit_work_named_rejects": explicit_rejects,
            "entered_total": entered_total, "displaced_total": displaced_total, "gold_displaced": gold_displaced,
            "aux_songs_total": aux_total, "aux_songs_outside_final_30": aux_outside, "checks": checks,
            "search_ms_median_triggered": med([r["search_ms"] for r in B if r["gate"]["triggered"]]),
            "search_ms_median_applied": med([r["search_ms"] for r in B if (r.get("work_resolution") or {}).get("status") == "applied"]),
            "search_ms_median_non_triggered": med([r["search_ms"] for r in B if not r["gate"]["triggered"]]),
        }
    b1, b2 = recs["live_rep1"], recs["live_rep2"]
    summary["B1_vs_B2"] = {
        "candidates_identical": sum(x["candidates"] == y["candidates"] for x, y in zip(b1, b2)),
        "work_resolution_identical": sum(
            {k: v for k, v in (x.get("work_resolution") or {}).items() if not k.endswith("_ms")} ==
            {k: v for k, v in (y.get("work_resolution") or {}).items() if not k.endswith("_ms")} for x, y in zip(b1, b2)),
        "of": len(b1),
    }
    summary["A_search_ms_median_non_triggered"] = med([r["search_ms"] for r in A if not r["gate"]["triggered"]])
    return {"summary": summary, "per_query": per_query}


def private_review(raw: Dict[int, List[Dict[str, Any]]], recs_b1: List[Dict[str, Any]]) -> None:
    """사람 라벨 대조용(Git 제외). 작품명·근거·검증 결과."""
    out = HERE / "analysis_private"
    out.mkdir(exist_ok=True)
    wr_by = {r["qid"]: (r.get("work_resolution") or {}) for r in recs_b1}
    with open(out / "model_output_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "decision", "work_index", "name", "aliases", "media_type", "evidence", "accepted", "reject_reason",
                    "mention_songs", "name_in_query"])
        for row in raw[1]:
            works = (wr_by.get(row["qid"]) or {}).get("works") or []
            if not works:
                w.writerow([row["qid"], row.get("parsed_status"), "", "", "", "", "", "", "", "", ""])
            for i, wk in enumerate(works, 1):
                w.writerow([row["qid"], row.get("parsed_status"), i, wk.get("name"), ";".join(wk.get("aliases") or []),
                            wk.get("media_type"), wk.get("evidence"), wk.get("accepted"), wk.get("reject_reason") or "",
                            wk.get("mention_songs"), wk.get("name_in_query")])


def main() -> int:
    raw = {n: jl(HERE / f"live/rep{n}/raw.jsonl") for n in (1, 2)}
    (HERE / "analysis").mkdir(exist_ok=True)
    live = live_summary(raw)
    (HERE / "analysis/live_summary.json").write_text(json.dumps(live, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    off = offline(raw)
    (HERE / "analysis/offline_summary.json").write_text(json.dumps(off["summary"], ensure_ascii=False, indent=1) + "\n",
                                                        encoding="utf-8")
    rows = list(off["per_query"].values())
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k[:2] not in ("se", "qi", "ga", "kn"), k))
    with open(HERE / "analysis/per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["set", "qid", "gate", "known_case", "A_rank"] +
                           [c for c in cols if c not in ("set", "qid", "gate", "known_case", "A_rank")])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    private_review(raw, jl(HERE / "offline/live_rep1/records.jsonl"))
    print(json.dumps({"live": live["rep1_vs_rep2"], "A": off["summary"]["A"]["recall_at_30"],
                      "B1": off["summary"]["B1"]["recall_at_30"], "B2": off["summary"]["B2"]["recall_at_30"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
