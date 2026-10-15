"""v54 재질문 흐름의 턴별 장르 판정 재구성 — 모델·DB 없음. 평가 라벨·정답은 읽지 않는다(턴 CSV의 순위 열은 그대로 옮겨 적기만 한다).

clarify 실행 폴더의 `clarify_turns.csv`(질의 × 흐름 × 턴, 그 턴 직전의 답 slot/answer_kind/answer_value)를 읽어 흐름 안에서 답을
누적하고(서비스가 매 턴 지금까지의 답 전부를 보내는 것과 같다), v49 분석 캐시의 분석으로 B(always)·D(conditional)의 필터와 D의 판정
(`_build_attribute_filter_with_policy`)을 턴마다 다시 계산한다. 그래서 "이 턴에서 B와 D의 필터가 달랐는가 · D가 장르를 유지/제외한 사유"를
모델 호출 없이 적을 수 있다. 실제 실행(B·D 폴더)의 후보 집합 차이와 맞춰 보는 용도다.

    venv/bin/python experiments/reranking/results_v54_attr_genre_policy/clarify_genre_decisions.py \
        clarify_sub_attr0.5_condgenre_rep1/v09_dev --set v09 [--out clarify_genre_decisions_v09.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backend.schemas.search import ClarifyAnswer  # noqa: E402
from src.retrieval import search_router as sr  # noqa: E402
from src.retrieval.analysis_cache import load_cache  # noqa: E402

CACHES = HERE.parent / "results_v49_vertex_baseline"


def filt(analysis, answers, policy: str):
    os.environ["ATTR_FILTER_GATE"] = "era_plus"
    os.environ["ATTR_FILTER_USE_GENRE"] = "1"
    os.environ["ATTR_FILTER_GENRE_POLICY"] = policy
    return sr._build_attribute_filter_with_policy(analysis, answers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--set", required=True, choices=("v06", "v09"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cache = load_cache(CACHES / f"analysis_cache_{args.set}_dev_vertex.json")
    turns = defaultdict(list)
    for r in csv.DictReader(open(HERE / args.folder / "clarify_turns.csv", encoding="utf-8-sig")):
        turns[(r["query_id"], r["policy"], r["target_id"])].append(r)
    rows = []
    for (qid, pol, target), steps in sorted(turns.items()):
        analysis = cache.get(qid, cache.entries[qid]["query"])
        answers: list[ClarifyAnswer] = []
        for st in sorted(steps, key=lambda s: int(s["turn"])):
            if st["slot"]:
                if st["answer_kind"] == "skip":
                    answers.append(ClarifyAnswer(slot=st["slot"], skipped=True))
                else:
                    answers.append(ClarifyAnswer(slot=st["slot"], value=st["answer_value"]))
            fb, _ = filt(analysis, answers, "always")
            fd, dec = filt(analysis, answers, "conditional")
            rows.append({
                "query_id": qid, "policy": pol, "target_id": target, "turn": st["turn"],
                "answers": ";".join(f"{a.slot}={'(모름)' if a.skipped else a.value}" for a in answers),
                "open_B": int(fb is not None), "open_D": int(fd is not None),
                "filters_differ": int(fb != fd), "D_use_genre": int(dec.use_genre), "D_reason": dec.reason,
                "D_had_genre_condition": int(dec.had_genre_condition),
                "rank": st["rank"], "candidate_rank@30": st["candidate_rank@30"],
            })
    out = HERE / (args.out or f"clarify_genre_decisions_{args.set}.csv")
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{args.folder}: 턴 {len(rows)}개 · B/D 필터가 다른 턴 {sum(r['filters_differ'] for r in rows)}")
    print("D 사유별 턴 수(경로 열림):", dict(Counter(r["D_reason"] + ("" if r["D_use_genre"] else "(제외)") for r in rows if r["open_D"])))
    by_q = defaultdict(set)
    for r in rows:
        if r["filters_differ"]:
            by_q[r["query_id"]].add(f"{r['policy'].replace('flow:', '')}턴{r['turn']}")
    print("필터가 다른 질의 → 턴:")
    for q, ts in sorted(by_q.items()):
        print(f"  {q}: {', '.join(sorted(ts))}")
    kept = [(r["query_id"], r["policy"], r["turn"], r["answers"]) for r in rows if r["D_reason"] == "answered_genre"]
    print(f"D가 재질문 장르 답으로 장르를 유지한 턴 {len(kept)}:")
    for k in kept:
        print("  ", k)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
