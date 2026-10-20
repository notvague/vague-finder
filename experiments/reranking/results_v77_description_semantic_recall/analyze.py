"""v77 실패 유형표(원문 없음) — `analysis/per_query.csv`에서 기계적으로 나눈다.

유형(후보 밖 실패 51건, S 기준):
- no_data            정답 곡 모두 소개 없음(검색 불가)
- reached_song       정답 묶음 ≤ 10 · 대표곡 = 정답
- reached_album_only 정답 묶음 ≤ 10 · 대표곡 ≠ 정답
- near               정답 묶음 11~100
- not_reached        100 밖

`MECHANISM`은 도달한 질의를 작성자(모델)가 질의·근거 청크를 읽고 붙인 판단이다. **사람 검증 전**이며 정답 사실로 쓰지 않는다.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

MECHANISM = {
    "c602": ("context_paraphrase", "경연·우승 맥락을 다른 표현(결승·공연 1위)으로 맞춤"),
    "c607": ("context_paraphrase", "옛 동네·추억 드라마 맥락을 다른 표현(80년대·88년 드라마 속 청춘)으로 맞춤. 다시 부름은 근거 청크에 없음"),
    "n011": ("context_paraphrase_plus_year", "데뷔·퍼포먼스를 다른 표현(첫 싱글·과감한 춤)으로, 연도는 글자로 맞춤. L도 6위"),
    "n040": ("same_words", "답가·여자 버전이 글자로 같음. L도 1위 — 의미 검색 고유 이득이 아니라 소개 원문 표현의 이득"),
    "t031": ("artist_name_plus_association", "가수 이름 글자 일치 + 분홍↔딸기 색 연상"),
    "n072": ("album_concept_for_cover", "표지 묘사(우주·행성)를 앨범 콘셉트 소개로 맞춤 — 표지 자체 근거 아님"),
    "m402": ("generic_sound", "현악기·남자·발라드의 일반 소리 묘사 — 같은 표현의 곡이 많아 경계(10위)"),
}


def main() -> int:
    rows = list(csv.DictReader((HERE / "analysis/per_query.csv").open(encoding="utf-8")))
    out = []
    for r in rows:
        if r["split"] != "out_fail":
            continue
        rank = int(r["S_group_rank"]) if r.get("S_group_rank") else None
        if not int(r["searchable"]):
            t = "no_data"
        elif rank and rank <= 10:
            t = "reached_song" if r.get("S_rep10_rank") else "reached_album_only"
        elif rank:
            t = "near"
        else:
            t = "not_reached"
        mech = MECHANISM.get(r["qid"], ("", ""))
        out.append({"set": r["set"], "qid": r["qid"], "type": t, "design_seen": r["design_seen"],
                    "S_group_rank": r.get("S_group_rank", ""), "L_group_rank": r.get("L_group_rank", ""),
                    "V_group_rank": r.get("V_group_rank", ""), "S_rep10_rank": r.get("S_rep10_rank", ""),
                    "S_evidence_kind": r.get("S_evidence_kind", ""), "S_gold_group_size": r.get("S_gold_group_size", ""),
                    "mechanism_model_judgment": mech[0]})
    out.sort(key=lambda d: (["reached_song", "reached_album_only", "near", "not_reached", "no_data"].index(d["type"]),
                            int(d["S_group_rank"]) if d["S_group_rank"] else 999, d["qid"]))
    with (HERE / "analysis/out_fail_types.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    counts = Counter(d["type"] for d in out)
    seen = Counter(d["type"] for d in out if d["design_seen"] == "1")
    res = {"out_fail": len(out), "type_counts": dict(counts), "type_counts_design_seen": dict(seen),
           "mechanism_note": "작성자(모델) 판단 · 사람 검증 전",
           "mechanism": {q: {"label": m[0], "note": m[1]} for q, m in MECHANISM.items()}}
    (HERE / "analysis/out_fail_types.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
