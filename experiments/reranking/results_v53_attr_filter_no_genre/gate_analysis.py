"""v53 게이트 분석 — dev 176건마다 B(장르 포함)와 C(장르 제외) 설정에서 속성 조건 보조 필터가 어떻게 달라지는지. 모델·DB 없음.

v49 Vertex 분석 캐시의 분석 JSON을 그대로 `QueryAnalysis`로 되살려 `_build_attribute_filter`를 두 설정으로 부른다
(`ATTR_FILTER_GATE=era_plus`, `ATTR_FILTER_USE_GENRE=1` → `0`). 질의마다 네 부류 중 하나다:
  closed_both   두 설정 모두 닫힘(시기 창 없음 · 넓음 · 다른 축 없음)
  open_same     두 설정 모두 열리고 필터가 같음(B 필터에 장르 조건이 없었다)
  open_relaxed  두 설정 모두 열리지만 C 필터에서 장르 조건이 빠짐 — **필터 완화 효과**를 볼 질의
  closed_in_C   B에서는 열렸는데 C에서는 닫힘(시기+장르만) — **경로 비활성화 효과**를 볼 질의
출력: gate_classes.csv(질의 ID · 세트 · 부류 · 축 값 · 조건 수), 표준 출력에 집계.

    venv/bin/python experiments/reranking/results_v53_attr_filter_no_genre/gate_analysis.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval import search_router as sr  # noqa: E402
from src.retrieval.analysis_cache import load_cache  # noqa: E402

CACHES = HERE.parent / "results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")


def _filter(analysis, use_genre: bool):
    os.environ["ATTR_FILTER_GATE"] = "era_plus"
    os.environ["ATTR_FILTER_USE_GENRE"] = "1" if use_genre else "0"
    return sr._build_attribute_filter(analysis)


def _has_genre(built) -> bool:
    return bool(built) and any("genre" in cond for cond in built["$and"])


def classify(analysis) -> tuple[str, int, int, bool]:
    b, c = _filter(analysis, True), _filter(analysis, False)
    nb, nc = (len(b["$and"]) if b else 0), (len(c["$and"]) if c else 0)
    if b is None and c is None:
        return "closed_both", nb, nc, False
    if b is not None and c is None:
        return "closed_in_C", nb, nc, _has_genre(b)
    if b == c:
        return "open_same", nb, nc, _has_genre(b)
    return "open_relaxed", nb, nc, _has_genre(b)


def main() -> None:
    rows = []
    for s in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{s}_dev_vertex.json")
        for qid, entry in sorted(cache.entries.items()):
            analysis = cache.get(qid, entry["query"])
            cls, nb, nc, genre_in_b = classify(analysis)
            era = analysis.release_era
            rows.append({
                "query_id": qid, "set": s, "gate_class": cls,
                "conds_B": nb, "conds_C": nc, "genre_cond_in_B": int(genre_in_b),
                "era": f"{era.start_year}-{era.end_year}" if era and era.start_year else "",
                "gender": analysis.vocal_gender or "", "genre": analysis.genre or "",
                "type": "/".join(analysis.artist_type.values) if analysis.has_artist_type_clue else "",
            })
    out = HERE / "gate_classes.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"dev {len(rows)}건")
    for cls in ("closed_both", "open_same", "open_relaxed", "closed_in_C"):
        sel = [r for r in rows if r["gate_class"] == cls]
        by_set = " · ".join(f"{s} {sum(1 for r in sel if r['set'] == s)}" for s in SETS)
        print(f"  {cls:13s} {len(sel):3d}  ({by_set})")
        if cls != "closed_both":
            for r in sel:
                print(f"     {r['query_id']}: era={r['era']} gender={r['gender'] or '-'} genre={r['genre'] or '-'} type={r['type'] or '-'} conds B{r['conds_B']}→C{r['conds_C']}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
