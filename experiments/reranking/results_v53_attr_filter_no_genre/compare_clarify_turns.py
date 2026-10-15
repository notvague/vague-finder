"""v53 재질문 2턴 — 두 실행(B 장르 포함 · C 장르 제외)을 질의 × 흐름 × 턴 단위로 맞대어 본다. 모델·DB 없음.

`--query-ids`로 영향 질의만 돈 실행(clarify_sub_*)과 전체 실행 모두 읽는다. 각 폴더의 `clarify_turns.csv`(턴별 순위·후보 순위·
답·후보 집합)와 `clarify_detail.csv`(흐름별 최종 순위·발견 턴)를 읽어
  - 질의 × 흐름마다 턴별 (정답 순위, 후보 30 안 순위, 답 종류·값, 후보 집합 같음?)을 B/C로 나란히
  - 달라진 턴만 모아 적고, 흐름별로 B→C 최종 발견(10위 안) 이득/손실을 센다
  - `clarify_runinfo.json`의 리랭킹 상태(failed가 있으면 비교에 쓰지 말라고 표시)와 스위치를 맨 위에 적는다

    venv/bin/python experiments/reranking/results_v53_attr_filter_no_genre/compare_clarify_turns.py \
        clarify_sub_attr0.5_norerank_rep1/v09_dev clarify_sub_attr0.5_nogenre_norerank_rep1/v09_dev
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
FLOWS = ("initial", "flow:reject", "flow:skip", "flow:oracle", "flow:noisy")


def load(folder: Path):
    turns = defaultdict(list)
    for r in csv.DictReader(open(folder / "clarify_turns.csv", encoding="utf-8-sig")):
        turns[(r["query_id"], r["policy"])].append(r)
    detail = {}
    for r in csv.DictReader(open(folder / "clarify_detail.csv", encoding="utf-8-sig")):
        detail[(r["query_id"], r["policy"])] = r
    info = json.loads((folder / "clarify_runinfo.json").read_text(encoding="utf-8"))
    return turns, detail, info


def rank_str(v: str) -> str:
    return v if v else "밖"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--labels", default="B(장르 포함),C(장르 제외)")
    args = ap.parse_args()
    la, lb = args.labels.split(",")
    ta, da, ia = load(HERE / args.base)
    tb, db, ib = load(HERE / args.new)
    for label, info in ((la, ia), (lb, ib)):
        st = info.get("clarify", {}).get("rerank_status_counts", {})
        sw = info.get("ranking_switches", {})
        flag = " **failed 있음 — 비교에 쓰지 않는다**" if st.get("failed") or st.get("exception") else ""
        print(f"{label}: 리랭킹 상태 {st}{flag} · attr_filter_weight={sw.get('attr_filter_weight')} · use_genre={sw.get('attr_filter_use_genre')}"
              f" · gemini={info.get('gemini')}")

    qids = sorted({q for q, _ in da} | {q for q, _ in db})
    print(f"\n질의 {len(qids)}건: {', '.join(qids)}")
    diffs = []
    for q in qids:
        for pol in FLOWS:
            A, B = ta.get((q, pol), []), tb.get((q, pol), [])
            if not A and not B:
                continue
            if not A or not B:
                # 한쪽만 턴 기록이 있다 = 그쪽만 개입 대상(최초 검색에서 정답이 안 보임). 다른 쪽은 1턴에 찾아 흐름을 돌지 않았다
                who = la if not A else lb
                diffs.append((q, pol, 0, (da.get((q, pol)) or {}).get("rank", ""), (db.get((q, pol)) or {}).get("rank", ""),
                              "", "", f"{who}는 1턴에 발견 — 흐름 안 돔", "", True))
                continue
            for i in range(max(len(A), len(B))):
                a = A[i] if i < len(A) else None
                b = B[i] if i < len(B) else None
                same_cands = (a or {}).get("candidate_ids") == (b or {}).get("candidate_ids")
                ra, rb = (a or {}).get("rank", ""), (b or {}).get("rank", "")
                ca, cb = (a or {}).get("candidate_rank@30", ""), (b or {}).get("candidate_rank@30", "")
                ans_a = f"{(a or {}).get('slot', '')}:{(a or {}).get('answer_kind', '')}={(a or {}).get('answer_value', '')}"
                ans_b = f"{(b or {}).get('slot', '')}:{(b or {}).get('answer_kind', '')}={(b or {}).get('answer_value', '')}"
                if not same_cands or ra != rb or ca != cb or ans_a != ans_b:
                    diffs.append((q, pol, i + 1, ra, rb, ca, cb, ans_a, ans_b, same_cands))
    print(f"\n달라진 턴 {len(diffs)}개 (질의 · 흐름 · 턴: 순위 {la}→{lb} · 후보 순위 → · 답 → · 후보 집합 같음?)")
    for q, pol, t, ra, rb, ca, cb, ans_a, ans_b, same in diffs:
        if t == 0:
            print(f"  {q} {pol}: 최종 순위 {rank_str(ra)}→{rank_str(rb)} · {ans_a}")
            continue
        ans = ans_a if ans_a == ans_b else f"{ans_a} → {ans_b}"
        print(f"  {q} {pol} 턴{t}: 순위 {rank_str(ra)}→{rank_str(rb)} · 후보 {rank_str(ca)}→{rank_str(cb)} · 답 {ans} · 후보 집합 {'같음' if same else '다름'}")

    print(f"\n흐름별 최종(대화 끝) 10위 안 — {la} → {lb}")
    print("| 흐름 | 질의 수 | 10위 안 B | 10위 안 C | 이득 | 손실 | 후보 30 안 B | C |")
    print("|---|---|---|---|---|---|---|---|")
    for pol in FLOWS:
        rows = [(q, da.get((q, pol)), db.get((q, pol))) for q in qids if (q, pol) in da or (q, pol) in db]
        if not rows:
            continue
        hit = lambda r: bool(r and r.get("rank") and int(float(r["rank"])) <= 10)  # noqa: E731
        cand = lambda r: bool(r and r.get("candidate_rank@30"))  # noqa: E731
        gain = [q for q, a, b in rows if hit(b) and not hit(a)]
        loss = [q for q, a, b in rows if hit(a) and not hit(b)]
        print(f"| {pol} | {len(rows)} | {sum(hit(a) for _, a, _ in rows)} | {sum(hit(b) for _, _, b in rows)} | {len(gain)} {gain or ''} | {len(loss)} {loss or ''} "
              f"| {sum(cand(a) for _, a, _ in rows)} | {sum(cand(b) for _, _, b in rows)} |")
    print("\n질의별 최종 순위(흐름: B→C):")
    for q in qids:
        cells = []
        for pol in FLOWS:
            a, b = da.get((q, pol)), db.get((q, pol))
            if a is None and b is None:
                continue
            cells.append(f"{pol.replace('flow:', '')} {rank_str((a or {}).get('rank', ''))}→{rank_str((b or {}).get('rank', ''))}")
        print(f"  {q}: " + " · ".join(cells))


if __name__ == "__main__":
    main()
