"""v53 후보 단계 질의별 보고 — A(끔) · B(장르 포함) · C(장르 제외)의 후보 순위·리랭킹 전 순위를 게이트 부류와 함께 적는다. 모델·DB 없음.

실행 폴더(이 폴더 안 `cand_*` 또는 `rerank_*_repN`)의 detail CSV와 explain JSONL(git 제외)을 읽는다.
  gate_class       gate_analysis.py의 부류(closed_both · open_same · open_relaxed · closed_in_C)
  attr_opened_X    실행 X에서 보조 경로 기여가 실제로 있었는가(explain에 attr_filter 경로를 가진 곡이 있음)
  target_in_attr_X 정답 곡이 보조 목록에 있었는가
  attr_top30_X     후보 30 안에서 보조 목록 기여가 있는 곡 수
  cand_X / pre_X / final_X  후보 순위(30 안) · 리랭킹 전 순위(10 안) · 최종 순위(10 안, 리랭킹 끔이면 pre와 같다)
출력: <out>.csv(곡 ID·순위·플래그만), 표준 출력에 집계(전체 · 부류별 · 질의별 변화).

    venv/bin/python experiments/reranking/results_v53_attr_filter_no_genre/candidate_report.py \
        --runs A=cand_attr0 B=cand_attr0.5_eraplus C=cand_attr0.5_eraplus_nogenre --out candidate_changes.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
SETS = ("v06", "v09", "v10")
TRACK = ("t026", "t008", "t010", "n097", "q101", "q202",  # v52 손실 6
         "q112", "n002", "n040", "t011", "t001", "t016", "n030")  # v51 이득 7


def folder_of(tag: str) -> Path:
    return (HERE.parent / tag) if "/" in tag else (HERE / tag)


def load_run(tag: str) -> dict:
    out = {}
    for s in SETS:
        folder = folder_of(tag) / f"{s}_dev"
        detail = folder / "search_eval_dev_detail.csv"
        if not detail.exists():
            raise SystemExit(f"detail 없음: {detail}")
        explain = {}
        ex_path = folder / "search_eval_dev_explain.jsonl"
        if ex_path.exists():
            for line in open(ex_path, encoding="utf-8"):
                d = json.loads(line)
                explain[d["query_id"]] = d
        for row in csv.DictReader(open(detail, encoding="utf-8-sig")):
            qid = row["query_id"]
            ex = explain.get(qid)
            target = row["relevant_ids"].split("|")[0]
            attr_songs, target_in_attr = [], False
            if ex is not None:
                for sg in ex["songs"]:
                    if sg.get("explain") and any(p["path"] == "attr_filter" for p in sg["explain"]["paths"]):
                        attr_songs.append(sg)
                        if sg["song_id"] == target:
                            target_in_attr = True
            out[(s, qid)] = {
                "cand": int(row["candidate_rank@30"]) if row["candidate_rank@30"] else 0,
                "pre": int(row["baseline_rank"]) if row["baseline_rank"] else 0,
                "final": int(row["rerank_rank"]) if row["rerank_rank"] else 0,
                "digest": row.get("candidate_digest", ""),
                "attr_opened": int(bool(attr_songs)) if ex is not None else "",
                "target_in_attr": int(target_in_attr) if ex is not None else "",
                "attr_top30": sum(1 for sg in attr_songs if sg.get("in_candidates")) if ex is not None else "",
                "outcome": row.get("rerank_outcome", ""),
            }
    return out


def show(r: int) -> str:
    return "밖" if not r else str(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="라벨=실행폴더 (예: A=cand_attr0)")
    ap.add_argument("--out", default="candidate_changes.csv")
    args = ap.parse_args()
    runs = {}
    for item in args.runs:
        label, tag = item.split("=", 1)
        runs[label] = load_run(tag)
    labels = list(runs)
    gate = {r["query_id"]: r for r in csv.DictReader(open(HERE / "gate_classes.csv", encoding="utf-8"))}

    keys = sorted(runs[labels[0]], key=lambda k: (SETS.index(k[0]), k[1]))
    rows = []
    for s, qid in keys:
        row = {"query_id": qid, "set": s, "gate_class": gate.get(qid, {}).get("gate_class", "")}
        for lab in labels:
            v = runs[lab][(s, qid)]
            for f in ("cand", "pre", "final", "attr_opened", "target_in_attr", "attr_top30"):
                row[f"{f}_{lab}"] = v[f]
        rows.append(row)
    out = HERE / args.out
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    def hit(lab, field, k, sel):
        return sum(1 for r in sel if r[f"{field}_{lab}"] and r[f"{field}_{lab}"] <= k)

    def mrr(lab, field, sel):
        return sum(1 / r[f"{field}_{lab}"] for r in sel if r[f"{field}_{lab}"] and r[f"{field}_{lab}"] <= 10) / len(sel)

    def table(sel, title):
        print(f"\n### {title} (n={len(sel)})")
        print("| 실행 | 후보 30 안 | 리랭킹 전 Hit@1 | Hit@5 | Hit@10 | MRR | 최종 Hit@1 | Hit@5 | Hit@10 | MRR | 경로 열림 | 정답∈보조 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lab in labels:
            opened = sum(1 for r in sel if r[f"attr_opened_{lab}"] == 1)
            tin = sum(1 for r in sel if r[f"target_in_attr_{lab}"] == 1)
            print(f"| {lab} | {hit(lab, 'cand', 30, sel)} | {hit(lab, 'pre', 1, sel)} | {hit(lab, 'pre', 5, sel)} | {hit(lab, 'pre', 10, sel)} | {mrr(lab, 'pre', sel):.3f} "
                  f"| {hit(lab, 'final', 1, sel)} | {hit(lab, 'final', 5, sel)} | {hit(lab, 'final', 10, sel)} | {mrr(lab, 'final', sel):.3f} | {opened} | {tin} |")

    table(rows, "dev 전체")
    for s in SETS:
        table([r for r in rows if r["set"] == s], f"세트 {s}")
    for cls in ("open_relaxed", "open_same", "closed_in_C", "closed_both"):
        sel = [r for r in rows if r["gate_class"] == cls]
        if sel:
            table(sel, f"게이트 부류 {cls}")

    # 쌍 비교(인접 라벨 + 첫째→마지막)
    pairs = [(labels[i], labels[i + 1]) for i in range(len(labels) - 1)]
    if len(labels) > 2:
        pairs.append((labels[0], labels[-1]))
    for a, b in pairs:
        same_digest = sum(1 for r in rows if runs[a][(r["set"], r["query_id"])]["digest"] == runs[b][(r["set"], r["query_id"])]["digest"])
        cand_in = [r for r in rows if r[f"cand_{b}"] and not r[f"cand_{a}"]]
        cand_out = [r for r in rows if r[f"cand_{a}"] and not r[f"cand_{b}"]]
        cand_moved = [r for r in rows if r[f"cand_{a}"] and r[f"cand_{b}"] and r[f"cand_{a}"] != r[f"cand_{b}"]]
        pre_in = [r for r in rows if r[f"pre_{b}"] and not r[f"pre_{a}"]]
        pre_out = [r for r in rows if r[f"pre_{a}"] and not r[f"pre_{b}"]]
        fin_in = [r for r in rows if r[f"final_{b}"] and not r[f"final_{a}"]]
        fin_out = [r for r in rows if r[f"final_{a}"] and not r[f"final_{b}"]]
        better = sum(1 for r in rows if (r[f"final_{b}"] or 99) < (r[f"final_{a}"] or 99))
        worse = sum(1 for r in rows if (r[f"final_{b}"] or 99) > (r[f"final_{a}"] or 99))
        print(f"\n== {a} → {b}: 후보 집합 지문 같음 {same_digest}/{len(rows)} · 후보 30 새로 든 {len(cand_in)} · 빠진 {len(cand_out)} · 후보 순위만 바뀐 {len(cand_moved)}"
              f" · 리랭킹 전 10위 안 든 {len(pre_in)} · 빠진 {len(pre_out)} · 최종 10위 안 든 {len(fin_in)} · 빠진 {len(fin_out)} · 최종 순위 나음 {better} : 나쁨 {worse}")
        for label, items in (("후보 든", cand_in), ("후보 빠진", cand_out), ("후보 이동", cand_moved)):
            for r in items:
                print(f"   {label} {r['set']} {r['query_id']} [{r['gate_class']}]: 후보 {show(r[f'cand_{a}'])}→{show(r[f'cand_{b}'])}"
                      f" · 리랭킹 전 {show(r[f'pre_{a}'])}→{show(r[f'pre_{b}'])} · 최종 {show(r[f'final_{a}'])}→{show(r[f'final_{b}'])}"
                      f" · 정답∈보조 {r[f'target_in_attr_{a}']}→{r[f'target_in_attr_{b}']} · 30 안 보조 곡 {r[f'attr_top30_{a}']}→{r[f'attr_top30_{b}']}")

    print("\n### 추적 질의(v52 손실 6 · v51 이득 7)")
    print("| 질의 | 부류 | " + " | ".join(f"후보 {l} · 리랭킹 전 {l} · 최종 {l}" for l in labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for qid in TRACK:
        r = next((r for r in rows if r["query_id"] == qid), None)
        if r is None:
            continue
        cells = " | ".join(f"{show(r[f'cand_{l}'])} · {show(r[f'pre_{l}'])} · {show(r[f'final_{l}'])}" for l in labels)
        print(f"| {qid} | {r['gate_class']} | {cells} |")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
