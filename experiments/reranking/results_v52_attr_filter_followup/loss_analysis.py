"""v50 B(끔) → v51(켬) 질의별 변화 추적 — 손실이 구현 오류인지 후보 구성 변화인지 가른다. 모델·DB 없음.

두 실행의 explain(`*_explain.jsonl`, git 제외 — 두 워크트리에 있다)과 detail을 읽어 dev 176건마다 적는다:
  attr_opened        v51에서 속성 조건 보조 경로가 열렸는가(어느 곡이든 attr_filter 기여가 있음)
  target_in_attr     정답 곡이 보조 목록에 있었는가(정답 explain에 attr_filter 경로)
  target_axes        정답 곡이 받은 속성 가산(vocal_gender_match · genre_match · release_era) — 보조 목록에 없으면 어느 축이 어긋났는지 보인다
  attr_in_top30      v51 후보 30 안에서 보조 목록 기여가 있는 곡 수(= 보조 경로가 후보 끝자리와 경쟁한 규모)
  cand_rank / final_rank  두 실행의 후보 순위·최종 순위
출력: loss_analysis.csv(곡 ID·순위·플래그만), 표준 출력에 요약.

    venv/bin/python experiments/reranking/results_v52_attr_filter_followup/loss_analysis.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
V50 = Path(os.environ.get("V50_DIR", "/private/tmp/vague-finder-v50-context/experiments/reranking/results_v50_vertex_context"))
V51 = Path(os.environ.get("V51_DIR", str(HERE.parent / "results_v51_attr_filter")))
RUNS = {"A": (V50, "rerank_ctx0.5_rep1"), "B": (V51, "rerank_attr0.5_eraplus_rep1")}
SETS = ("v06", "v09", "v10")


def load(root: Path, run: str, s: str):
    ex = {}
    for line in open(root / run / f"{s}_dev" / "search_eval_dev_explain.jsonl", encoding="utf-8"):
        d = json.loads(line)
        ex[d["query_id"]] = d
    de = {r["query_id"]: r for r in csv.DictReader(open(root / run / f"{s}_dev" / "search_eval_dev_detail.csv", encoding="utf-8-sig"))}
    return ex, de


def target_record(d: dict):
    tid = d["relevant_ids"][0]
    for sg in d["songs"]:
        if sg["song_id"] == tid and sg.get("explain"):
            return sg
    return None


def main() -> None:
    rows = []
    for s in SETS:
        exa, dea = load(*RUNS["A"], s)
        exb, deb = load(*RUNS["B"], s)
        for qid in dea:
            a, b = exa[qid], exb[qid]
            ta, tb = target_record(a), target_record(b)
            attr_songs = [sg for sg in b["songs"] if sg.get("explain") and any(p["path"] == "attr_filter" for p in sg["explain"]["paths"])]
            attr_in_top30 = sum(1 for sg in attr_songs if sg.get("in_candidates"))
            axes = ""
            if tb is not None:
                axes = "+".join(sorted({adj.get("rule") or adj.get("label") or "" for adj in tb["explain"].get("adjustments", [])}
                                       & {"vocal_gender_match", "genre_match", "release_era"}))
            ra, rb = dea[qid]["rerank_rank"], deb[qid]["rerank_rank"]
            ca, cb = dea[qid]["candidate_rank@30"], deb[qid]["candidate_rank@30"]
            def _i(v):
                return int(v) if v else 999
            change = "같음" if ra == rb else ("나음" if _i(rb) < _i(ra) else "나쁨")
            rows.append({
                "query_id": qid, "set": s, "target_id": a["relevant_ids"][0],
                "attr_opened": int(bool(attr_songs)),
                "target_in_attr": int(tb is not None and any(p["path"] == "attr_filter" for p in tb["explain"]["paths"])),
                "target_axes": axes,
                "attr_in_top30": attr_in_top30,
                "cand_rank_A": ca, "cand_rank_B": cb, "final_rank_A": ra, "final_rank_B": rb,
                "change": change,
                "era": dea[qid]["release_era"].replace("\n", " ")[:40], "gender": dea[qid]["vocal_gender"], "genre": dea[qid]["genre"],
            })
    out = HERE / "loss_analysis.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    opened = [r for r in rows if r["attr_opened"]]
    print(f"dev {len(rows)}건 · 보조 경로 열림 {len(opened)}건 · 그중 정답이 보조 목록에 있음 {sum(r['target_in_attr'] for r in opened)}건")
    for label in ("나음", "나쁨"):
        sel = [r for r in rows if r["change"] == label]
        print(f"\n{label} {len(sel)}건 (query: 후보 A→B, 최종 A→B, 열림, 정답∈보조, 정답 가산 축, 30 안 보조 곡 수):")
        for r in sel:
            print(f"  {r['query_id']}: {r['cand_rank_A'] or '-'}→{r['cand_rank_B'] or '-'}, {r['final_rank_A'] or '-'}→{r['final_rank_B'] or '-'}, "
                  f"opened={r['attr_opened']}, target_in_attr={r['target_in_attr']}, axes={r['target_axes'] or '-'}, attr_top30={r['attr_in_top30']}")
    # 열렸는데 정답이 보조 목록에 없는 질의 — 어느 축이 어긋났나
    miss = [r for r in opened if not r["target_in_attr"]]
    print(f"\n열렸지만 정답이 보조 목록에 없는 {len(miss)}건 (정답 가산 축 → 빠진 축이 필터에서 어긋난 축):")
    for r in miss:
        print(f"  {r['query_id']}: axes={r['target_axes'] or '-'} 후보 {r['cand_rank_A'] or '-'}→{r['cand_rank_B'] or '-'} 최종 {r['final_rank_A'] or '-'}→{r['final_rank_B'] or '-'} attr_top30={r['attr_in_top30']}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
