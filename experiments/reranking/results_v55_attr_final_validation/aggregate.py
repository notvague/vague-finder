"""v55 집계 — 봉인 v10 test 42건의 A·B·D 실행을 모델·DB 없이 모아 요약·질의별 비교·정책 사유·지연을 낸다.

    venv/bin/python experiments/reranking/results_v55_attr_final_validation/aggregate.py \
        [--runs A=A_v10_test B=B_v10_test D=D_v10_test] [--out-prefix v10_test]

읽는 것(실행 폴더 안, `search_eval_test_*`): top10 CSV(커밋) · detail CSV(git 제외 — 후보 순위·결과 코드·지연·후보 지문) ·
explain JSONL(git 제외 — 요청 단위 장르 판정 `attr_filter_genre`, 보조 경로 기여) · runinfo JSON(조건 대조).
내는 것: `<prefix>_summary.csv`(설정별 지표) · `<prefix>_per_query.csv`(질의별 A/B/D 후보·리랭킹 전·최종 순위, 사유, 변화) ·
표준 출력(마크다운 표: 지표 · A→D · B→D 이득/손실 · 후보 구성 변화 · 사유별 건수 · 지연 · 리랭킹 결과 · 조건 일치).

판정 규칙(RUN_INFO §2 사전 고정): 리랭킹 결과가 exception·fallback·partial_failure면 그 실행은 "유효 비교"가 아니다 — 여기서는
집계에서 빼지 않고 그대로 센 뒤 표준 출력에 건수와 질의 ID를 적는다(빼는 결정은 보고서에서).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OK_OUTCOMES = {"applied", "policy_skipped", "skipped"}
LOAD_THRESHOLD_S = 5.0  # 준비된 상태의 검색은 p95 1.5초 안 — 이보다 길면 적재(첫 질의·Mongo 가사 판)로 분류


def folder_of(tag: str) -> Path:
    if tag.startswith("/"):
        return Path(tag)
    return (HERE.parent / tag) if "/" in tag else (HERE / tag)


def pct(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    return v[min(len(v) - 1, int(len(v) * p))]


def rank_of(ids: str, relevant: set) -> int | None:
    return next((i for i, sid in enumerate(ids.split("|"), 1) if sid in relevant), None)


def load_run(tag: str, split: str) -> dict:
    folder = folder_of(tag)
    top10 = folder / f"search_eval_{split}_top10.csv"
    detail = folder / f"search_eval_{split}_detail.csv"
    explain = folder / f"search_eval_{split}_explain.jsonl"
    runinfo = folder / f"search_eval_{split}_runinfo.json"
    if not top10.exists():
        raise SystemExit(f"top10 없음: {top10}")
    rows = {}
    for r in csv.DictReader(open(top10, encoding="utf-8-sig")):
        rel = set(filter(None, r["relevant_ids"].split("|")))
        rows[r["query_id"]] = {
            "relevant": rel,
            "final": rank_of(r["rerank_top_ids"], rel),
            "pre": rank_of(r["baseline_top_ids"], rel),
            "final_ids": r["rerank_top_ids"],
            "base_ids": r["baseline_top_ids"],
        }
    if detail.exists():
        for r in csv.DictReader(open(detail, encoding="utf-8-sig")):
            q = r["query_id"]
            if q not in rows:
                continue
            cand = r.get("candidate_rank@30", "")
            rows[q].update(
                {
                    "cand": int(cand) if cand not in ("", "None") else None,
                    "digest": r.get("candidate_digest", ""),
                    "outcome": r.get("rerank_outcome", ""),
                    "policy_skip": r.get("rerank_policy_skip", ""),
                    "calls": int(r.get("rerank_calls") or 0),
                    "retrieval_s": float(r.get("retrieval_ms") or 0) / 1000,
                    "rerank_s": float(r.get("rerank_only_ms") or 0) / 1000,
                    "query_type": r.get("intent_type", ""),
                    "fell_back": r.get("rerank_fell_back", ""),
                    "partial": r.get("rerank_partial_failure", ""),
                }
            )
    if explain.exists():
        for line in open(explain, encoding="utf-8"):
            d = json.loads(line)
            q = d.get("query_id")
            if q not in rows:
                continue
            g = d.get("attr_filter_genre") or {}
            rows[q]["genre_reason"] = g.get("reason", "") if isinstance(g, dict) else ""
            rows[q]["genre_had_condition"] = g.get("had_genre_condition", "") if isinstance(g, dict) else ""
            # v54 candidate_report.py와 같은 규칙: 곡 explain의 paths에 "attr_filter" 경로가 있으면 보조 목록 기여
            songs = d.get("songs") or []
            attr_songs = [
                s for s in songs
                if s.get("explain") and any(p.get("path") == "attr_filter" for p in s["explain"].get("paths", []))
            ]
            rows[q]["attr_songs_n"] = len(attr_songs)
            rows[q]["attr_top30"] = sum(1 for s in attr_songs if s.get("in_candidates"))
            rows[q]["target_in_attr"] = int(any(s.get("relevant") for s in attr_songs))
    info = json.loads(runinfo.read_text(encoding="utf-8")) if runinfo.exists() else {}
    return {"rows": rows, "runinfo": info, "tag": tag}


def metrics(rows: dict) -> dict:
    n = len(rows)
    fin = [r["final"] for r in rows.values()]
    pre = [r["pre"] for r in rows.values()]
    cand = [r.get("cand") for r in rows.values()]
    return {
        "n": n,
        "cand_recall@30": sum(1 for c in cand if c is not None),
        "pre_hit@10": sum(1 for x in pre if x is not None and x <= 10),
        "hit@1": sum(1 for x in fin if x == 1),
        "hit@5": sum(1 for x in fin if x is not None and x <= 5),
        "hit@10": sum(1 for x in fin if x is not None and x <= 10),
        "mrr@10": sum(1 / x for x in fin if x is not None and x <= 10) / n if n else 0.0,
        "outcomes": Counter(r.get("outcome", "?") for r in rows.values()),
        "calls": sum(r.get("calls", 0) for r in rows.values()),
        "invalid": sorted(q for q, r in rows.items() if r.get("outcome") and r["outcome"] not in OK_OUTCOMES),
    }


def fmt_rank(x) -> str:
    return "밖" if x is None else str(x)


def pair_diff(a: dict, b: dict, la: str, lb: str) -> dict:
    better, worse, same = [], [], 0
    in10, out10, up, down = [], [], [], []
    digest_same = digest_diff = 0
    cand_in, cand_out = [], []
    for q in a:
        ra, rb = a[q]["final"], b[q]["final"]
        ka, kb = ra if ra is not None else 99, rb if rb is not None else 99
        if ka == kb:
            same += 1
        elif kb < ka:
            better.append(q)
            (in10 if ra is None else up).append(q)
        else:
            worse.append(q)
            (out10 if rb is None else down).append(q)
        if a[q].get("digest") is not None and b[q].get("digest") is not None:
            if a[q].get("digest") == b[q].get("digest"):
                digest_same += 1
            else:
                digest_diff += 1
        ca, cb = a[q].get("cand"), b[q].get("cand")
        if ca is None and cb is not None:
            cand_in.append(q)
        if ca is not None and cb is None:
            cand_out.append(q)
    return {
        "pair": f"{la}→{lb}", "same": same, "better": better, "worse": worse,
        "in10": in10, "out10": out10, "up": up, "down": down,
        "digest_same": digest_same, "digest_diff": digest_diff,
        "cand_in": cand_in, "cand_out": cand_out,
    }


def latency(rows: dict) -> dict:
    ordered = list(rows.values())
    if not ordered or "retrieval_s" not in ordered[0]:
        return {}
    first = ordered[0]["retrieval_s"]
    loads = [(q, r["retrieval_s"]) for i, (q, r) in enumerate(rows.items()) if i > 0 and r["retrieval_s"] > LOAD_THRESHOLD_S]
    ready = [r["retrieval_s"] for i, r in enumerate(ordered) if i > 0 and r["retrieval_s"] <= LOAD_THRESHOLD_S]
    rerank = [r["rerank_s"] for r in ordered if r.get("outcome") == "applied"]  # 모델을 부른 질의만(skipped·policy_skipped 제외)
    total = [r["retrieval_s"] + r["rerank_s"] for r in ordered]
    return {
        "first_s": first, "loads": loads,
        "ready_med": st.median(ready) if ready else float("nan"), "ready_p95": pct(ready, 0.95), "ready_max": max(ready) if ready else float("nan"), "ready_n": len(ready),
        "rerank_med": st.median(rerank) if rerank else float("nan"), "rerank_p95": pct(rerank, 0.95), "rerank_n": len(rerank),
        "total_med": st.median(total), "total_p95": pct(total, 0.95), "total_sum_min": sum(total) / 60,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["A=A_v10_test", "B=B_v10_test", "D=D_v10_test"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-prefix", default="v10_test")
    args = ap.parse_args()
    runs = {}
    for spec in args.runs:
        label, tag = spec.split("=", 1)
        runs[label] = load_run(tag, args.split)
    labels = list(runs)
    base = runs[labels[0]]["rows"]
    for lb in labels[1:]:
        if set(runs[lb]["rows"]) != set(base):
            raise SystemExit(f"질의 집합이 다름: {labels[0]} vs {lb}")

    # 조건 대조(runinfo)
    print("## 조건 대조 (runinfo)")
    keys = ("corpus", "bm25", "lyrics_source", "gemini", "reranker")
    ref = runs[labels[0]]["runinfo"]
    for lb in labels[1:]:
        other = runs[lb]["runinfo"]
        diffs = [k for k in keys if ref.get(k) != other.get(k)]
        an = ref.get("analysis", {}).get("path") == other.get("analysis", {}).get("path")
        sw = {k: (ref.get("ranking_switches", {}).get(k), other.get("ranking_switches", {}).get(k))
              for k in set(ref.get("ranking_switches", {})) | set(other.get("ranking_switches", {}))
              if ref.get("ranking_switches", {}).get(k) != other.get("ranking_switches", {}).get(k)}
        print(f"- {labels[0]} vs {lb}: 다른 키 {diffs or '없음'} · 같은 분석 캐시 {an} · 스위치 차이 {sw}")
    for lb in labels:
        ri = runs[lb]["runinfo"]
        print(f"- {lb}: split={ri.get('args',{}).get('split')} corpus={ri.get('corpus',{}).get('qdrant_path')}({ri.get('corpus',{}).get('point_count')}) "
              f"bm25={ri.get('bm25',{}).get('sha256_12')} lyrics={ri.get('lyrics_source',{}).get('content_sha256_16')} "
              f"gemini={ri.get('gemini',{}).get('backend')}/{ri.get('gemini',{}).get('location')} cache_drift={ri.get('analysis',{}).get('fingerprint_drift')} "
              f"fallback_ids={ri.get('analysis',{}).get('fallback_query_ids')} policy_eff={ri.get('ranking_switches',{}).get('attr_filter_genre_policy_effective')}")

    # 요약
    print("\n## 지표 (n = 질의 수)")
    print("| 구성 | n | 후보 30 안 | 리랭킹 전 Hit@10 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | 리랭킹 결과 | 리랭킹 실행 수(detail rerank_calls 합, 모델 호출 수는 usage) | 유효하지 않은 결과 질의 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    summary_rows = []
    for lb in labels:
        m = metrics(runs[lb]["rows"])
        outc = " ".join(f"{k}={v}" for k, v in sorted(m["outcomes"].items()))
        print(f"| {lb} | {m['n']} | {m['cand_recall@30']} | {m['pre_hit@10']} | {m['hit@1']} | {m['hit@5']} | {m['hit@10']} | {m['mrr@10']:.3f} | {outc} | {m['calls']} | {', '.join(m['invalid']) or '0'} |")
        summary_rows.append({"config": lb, **{k: v for k, v in m.items() if k not in ("outcomes", "invalid")}, "outcomes": outc, "invalid_queries": "|".join(m["invalid"])})
    with open(HERE / f"{args.out_prefix}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    # 쌍 비교
    print("\n## 쌍 비교 (최종 순위 기준, 밖 = 10위 밖)")
    pairs = []
    if "D" in runs:
        for la in labels:
            if la != "D":
                pairs.append(pair_diff(runs[la]["rows"], runs["D"]["rows"], la, "D"))
    if "A" in runs and "B" in runs:
        pairs.append(pair_diff(runs["A"]["rows"], runs["B"]["rows"], "A", "B"))
    print("| 쌍 | 같음 | 나음 | 나쁨 | 10위 안 든 | 빠진 | 안에서 오름 | 내림 | 후보 지문 같음/다름 | 후보 30 새로 든 | 빠진 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in pairs:
        print(f"| {p['pair']} | {p['same']} | {len(p['better'])} {p['better']} | {len(p['worse'])} {p['worse']} | {len(p['in10'])} {p['in10']} | {len(p['out10'])} {p['out10']} | {len(p['up'])} {p['up']} | {len(p['down'])} {p['down']} | {p['digest_same']}/{p['digest_diff']} | {len(p['cand_in'])} {p['cand_in']} | {len(p['cand_out'])} {p['cand_out']} |")

    # 동일 후보에서의 리랭킹 변동 vs 후보 구성 변화
    if "B" in runs and "D" in runs:
        b, d = runs["B"]["rows"], runs["D"]["rows"]
        same_digest_rank_diff = [q for q in b if b[q].get("digest") == d[q].get("digest") and b[q]["final"] != d[q]["final"]]
        diff_digest_rank_same = [q for q in b if b[q].get("digest") != d[q].get("digest") and b[q]["final"] == d[q]["final"]]
        same_digest_top10_diff = [q for q in b if b[q].get("digest") == d[q].get("digest") and b[q]["final_ids"] != d[q]["final_ids"]]
        print(f"\n- B→D 후보 지문 같은데 정답 순위 다름(리랭킹 변동): {len(same_digest_rank_diff)} {same_digest_rank_diff}")
        print(f"- B→D 후보 지문 같은데 top-10 글자가 다름(비정답 포함 변동): {len(same_digest_top10_diff)} {same_digest_top10_diff}")
        print(f"- B→D 후보 지문 다른데 정답 순위 같음: {len(diff_digest_rank_same)} {diff_digest_rank_same}")

    # 정책 사유
    print("\n## 정책 적용 사유 (explain의 요청 단위 장르 판정 — conditional 실행에만 기록)")
    for lb in labels:
        reasons = Counter(r.get("genre_reason", "") for r in runs[lb]["rows"].values())
        had = Counter((r.get("genre_reason", ""), str(r.get("genre_had_condition", ""))) for r in runs[lb]["rows"].values() if r.get("genre_reason"))
        opened = sum(1 for r in runs[lb]["rows"].values() if r.get("attr_songs_n", 0) > 0)
        tin = sum(1 for r in runs[lb]["rows"].values() if r.get("target_in_attr", 0))
        print(f"- {lb}: 사유 {dict(reasons)} · (사유, 뺄 장르 조건 있었나) {dict(had)} · 보조 경로 기여 있는 질의 {opened} · 정답이 보조 목록에 {tin}")

    # 지연
    print("\n## 지연 (detail의 retrieval_ms · rerank_only_ms; 첫 질의와 5초 초과 검색은 적재로 분류)")
    print("| 실행 | 첫 질의 검색 s | 적재로 분류된 그 밖의 질의 | 준비된 상태 검색 중앙값 / p95 / 최대 (n) | 리랭킹 중앙값 / p95 (n) | 전체 질의당 검색+리랭킹 중앙값 / p95 / 합계 분 |")
    print("|---|---|---|---|---|---|")
    for lb in labels:
        L = latency(runs[lb]["rows"])
        if not L:
            print(f"| {lb} | detail 없음 | | | | |"); continue
        loads = ", ".join(f"{q} {s:.0f}" for q, s in L["loads"]) or "없음"
        print(f"| {lb} | {L['first_s']:.1f} | {loads} | {L['ready_med']:.2f} / {L['ready_p95']:.2f} / {L['ready_max']:.2f} ({L['ready_n']}) | {L['rerank_med']:.2f} / {L['rerank_p95']:.2f} ({L['rerank_n']}) | {L['total_med']:.2f} / {L['total_p95']:.2f} / {L['total_sum_min']:.1f} |")

    # 질의별 CSV
    fields = ["query_id", "query_type"]
    for lb in labels:
        fields += [f"cand_{lb}", f"pre_{lb}", f"final_{lb}", f"outcome_{lb}", f"policy_skip_{lb}", f"reason_{lb}", f"target_in_attr_{lb}"]
    for p in pairs:
        fields.append(f"change_{p['pair']}")
    fields += ["digest_same_BD"] if "B" in runs and "D" in runs else []
    out_rows = []
    for q in base:
        row = {"query_id": q, "query_type": base[q].get("query_type", "")}
        for lb in labels:
            r = runs[lb]["rows"][q]
            row[f"cand_{lb}"] = fmt_rank(r.get("cand")); row[f"pre_{lb}"] = fmt_rank(r.get("pre")); row[f"final_{lb}"] = fmt_rank(r.get("final"))
            row[f"outcome_{lb}"] = r.get("outcome", ""); row[f"policy_skip_{lb}"] = r.get("policy_skip", "")
            row[f"reason_{lb}"] = r.get("genre_reason", ""); row[f"target_in_attr_{lb}"] = r.get("target_in_attr", "")
        for p in pairs:
            la, lb_ = p["pair"].split("→")
            ra, rb = runs[la]["rows"][q]["final"], runs[lb_]["rows"][q]["final"]
            ka, kb = ra if ra is not None else 99, rb if rb is not None else 99
            row[f"change_{p['pair']}"] = "same" if ka == kb else ("better" if kb < ka else "worse")
        if "B" in runs and "D" in runs:
            row["digest_same_BD"] = int(runs["B"]["rows"][q].get("digest") == runs["D"]["rows"][q].get("digest"))
        out_rows.append(row)
    with open(HERE / f"{args.out_prefix}_per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)
    changed = [r for r in out_rows if any(r.get(k) != "same" for k in r if k.startswith("change_"))]
    print(f"\n## 순위가 달라진 질의 {len(changed)}건 (질의 원문은 적지 않는다 — detail CSV 참조)")
    print("| 질의 | 유형 | " + " | ".join(f"후보/전/최종 {lb}" for lb in labels) + " | 사유 D | " + " | ".join(p["pair"] for p in pairs) + " |")
    print("|---|---|" + "---|" * (len(labels) + 1 + len(pairs)))
    for r in changed:
        print(f"| {r['query_id']} | {r['query_type']} | " + " | ".join(f"{r[f'cand_{lb}']}/{r[f'pre_{lb}']}/{r[f'final_{lb}']}" for lb in labels)
              + f" | {r.get('reason_D','')} | " + " | ".join(r[f"change_{p['pair']}"] for p in pairs) + " |")
    print(f"\n저장: {args.out_prefix}_summary.csv · {args.out_prefix}_per_query.csv")


if __name__ == "__main__":
    main()
