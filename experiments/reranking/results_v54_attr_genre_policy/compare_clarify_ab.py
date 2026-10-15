"""재질문 2턴 OFF(A, ATTR_FILTER_WEIGHT=0) vs ON(B, 0.5 · era_plus) 비교 — `--policies flow` 실행용. 모델·DB 없음.

    venv/bin/python experiments/reranking/results_v53_attr_filter_no_genre/compare_clarify_ab.py \
        clarify_attr0_rep1/v06_dev clarify_attr0.5_rep1/v06_dev [--labels A,B]

읽는 것: 두 폴더의 clarify_summary.csv · clarify_detail.csv · clarify_turns.csv · clarify_runinfo.json.
- 전체(57건)와 개입 대상(최초 검색에서 정답이 안 보인 질의)을 따로 낸다. 개입 대상 수가 두 측정 사이에 다를 수 있으므로
  답변 효과는 **양쪽 모두 개입 대상인 같은 질의**에서 따로 센다(results_clarify_v09/compare_clarify.py와 같은 원칙).
- 정책 행이 빠진 질의는 조용히 0으로 세지 않고 멈춘다.
- 리랭킹 상태(rerank_status_counts)에 failed가 있으면 그 측정은 비교에 쓰지 않도록 맨 위에 표시한다.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("experiments/reranking/results_v53_attr_filter_no_genre")
FLOWS = ("flow:reject", "flow:skip", "flow:oracle", "flow:noisy")
COLS = ["n_executed", "n_conversations", "full_hit@1", "full_hit@5", "full_hit@10", "full_mrr@10",
        "int_hit@1", "int_hit@10", "int_candidate_recall@30", "int_found_by_turn2", "n_conv_out_of_candidates"]


def _truthy(v: str) -> bool:
    return v in ("True", "1", "1.0")


def summary(d: Path) -> dict[str, dict]:
    return {r["policy"]: r for r in csv.DictReader(open(d / "clarify_summary.csv", encoding="utf-8-sig"))}


def detail(d: Path) -> list[dict]:
    return list(csv.DictReader(open(d / "clarify_detail.csv", encoding="utf-8-sig")))


def turns(d: Path) -> list[dict]:
    return list(csv.DictReader(open(d / "clarify_turns.csv", encoding="utf-8-sig")))


def hit10_by_query(rows: list[dict]) -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        acc[(r["query_id"], r["policy"])].append(1.0 if _truthy(r["hit10"]) else 0.0)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def rank_path(trows: list[dict], policy: str) -> dict[tuple[str, str], str]:
    """(query, target) → 턴별 '순위(후보 순위)' 경로. 밖이면 '-'."""
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in sorted(trows, key=lambda r: (r["query_id"], r["target_id"], int(r["turn"]))):
        if r["policy"] != policy:
            continue
        rank = r["rank"] or "-"
        cand = r["candidate_rank@30"] or "-"
        tag = f"{rank}({cand})"
        if r["slot"]:
            tag = f"[{r['slot']}={r['answer_kind']}:{r['answer_value'] or '모름'}] " + tag
        out[(r["query_id"], r["target_id"])].append(tag)
    return {k: " → ".join(v) for k, v in out.items()}


def fmt(v: str) -> str:
    if v in ("", None):
        return "-"
    try:
        f = float(v)
    except ValueError:
        return v
    return f"{f:.3f}" if f != int(f) else str(int(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--labels", default="A(끔),B(켬)")
    args = ap.parse_args()
    la, lb = [x.strip() for x in args.labels.split(",", 1)]
    da, db = ROOT / args.base, ROOT / args.new

    for label, d in ((la, da), (lb, db)):
        info = json.loads((d / "clarify_runinfo.json").read_text(encoding="utf-8"))
        c = info["clarify"]
        sw = info.get("ranking_switches", {})
        print(f"{label}: policies={c.get('policies', 'all')} · rerank_status={c.get('rerank_status_counts')} · "
              f"attr_filter_weight={sw.get('attr_filter_weight')} gate={sw.get('attr_filter_gate')} k={sw.get('attr_filter_k')} · "
              f"context_weight={sw.get('context_weight')} · gemini={info.get('reranker', {}).get('gemini', info.get('reranker', {}).get('backend'))} · "
              f"elapsed={c.get('elapsed_seconds')}s")
        failed = {k: v for k, v in (c.get("rerank_status_counts") or {}).items() if "failed" in k or "exception" in k}
        if failed:
            print(f"  !! {label}에 실패 상태가 있다 — 비교에 쓰지 말 것: {failed}")

    sa, sb = summary(da), summary(db)
    print(f"\n## 정책별 집계 (개입 대상 {la} {sa['initial']['n_intervention']} → {lb} {sb['initial']['n_intervention']})")
    print("| 정책 | " + " | ".join(f"{c} {la}→{lb}" for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for pol in sb:
        if pol in sa:
            print(f"| {pol} | " + " | ".join(f"{fmt(sa[pol][c])}→{fmt(sb[pol][c])}" for c in COLS) + " |")

    ra, rb = detail(da), detail(db)
    ha, hb = hit10_by_query(ra), hit10_by_query(rb)
    queries = sorted({q for q, _ in hb})
    for pol in ("initial", *FLOWS):
        missing = [q for q in queries if (q, pol) not in ha or (q, pol) not in hb]
        assert not missing, f"{pol}: 정책 행이 없는 질의 {missing}"
    int_a = {q for q in queries if ha[(q, "initial")] < 1.0}
    int_b = {q for q in queries if hb[(q, "initial")] < 1.0}
    common = sorted(int_a & int_b)
    print(f"\n개입 대상: {la} {len(int_a)} · {lb} {len(int_b)} · 공통 {len(common)} · {la}만 {sorted(int_a - int_b)} · {lb}만 {sorted(int_b - int_a)}")

    print("\n질의 단위 Hit@10 변화 (전체 질의, 목표 곡별 평균):")
    for pol in ("initial", *FLOWS):
        gain = [q for q in queries if ha[(q, pol)] < hb[(q, pol)]]
        loss = [q for q in queries if ha[(q, pol)] > hb[(q, pol)]]
        print(f"  {pol:12s} 이득 {len(gain)} {gain} / 손실 {len(loss)} {loss}")

    print(f"\n같은 질의만 비교 (양쪽 모두 개입 대상인 {len(common)}건, 대화 끝에 목표 곡이 보인 질의 수 — 목표 곡별 평균의 합):")
    print(f"| 정책 | {la} | {lb} |\n|---|---|---|")
    for pol in FLOWS:
        print(f"| {pol} | {sum(ha[(q, pol)] for q in common):.2f} | {sum(hb[(q, pol)] for q in common):.2f} |")
    for label, pol in (("맞는 답의 이득 (oracle − reject)", "flow:oracle"), ("틀린 답의 손해 (noisy − reject)", "flow:noisy"),
                       ("모름의 차이 (skip − reject, 0이어야 한다)", "flow:skip")):
        d_a = sum(ha[(q, pol)] - ha[(q, "flow:reject")] for q in common)
        d_b = sum(hb[(q, pol)] - hb[(q, "flow:reject")] for q in common)
        who_a = [q for q in common if ha[(q, pol)] != ha[(q, "flow:reject")]]
        who_b = [q for q in common if hb[(q, pol)] != hb[(q, "flow:reject")]]
        print(f"| {label} | {d_a:+.2f} {who_a} | {d_b:+.2f} {who_b} |")

    ta, tb = turns(da), turns(db)
    print(f"\n개입 대상 질의별 턴 경로 — '순위(후보 순위)', 답은 [슬롯=종류:값]. {la} | {lb}")
    for pol in FLOWS:
        pa, pb = rank_path(ta, pol), rank_path(tb, pol)
        print(f"\n### {pol}")
        for key in sorted(set(pa) | set(pb)):
            if key[0] not in (int_a | int_b):
                continue
            mark = "" if pa.get(key) == pb.get(key) else "  ← 다름"
            print(f"  {key[0]}/{key[1]}: {pa.get(key, '?')}  |  {pb.get(key, '?')}{mark}")

    # 턴별 후보 집합이 달라진 검색 수 — 보조 경로가 실제로 작동한 검색을 센다
    ca = {(r["query_id"], r["target_id"], r["policy"], r["turn"]): r["candidate_ids"] for r in ta}
    cb = {(r["query_id"], r["target_id"], r["policy"], r["turn"]): r["candidate_ids"] for r in tb}
    shared = sorted(set(ca) & set(cb))
    changed = [k for k in shared if ca[k] != cb[k]]
    by_turn = defaultdict(int)
    for k in changed:
        by_turn[k[3]] += 1
    print(f"\n후보 집합이 다른 검색: {len(changed)}/{len(shared)} (턴별 {dict(by_turn)}) · "
          f"질의 {sorted({k[0] for k in changed})}")


if __name__ == "__main__":
    main()
