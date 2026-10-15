"""v54 장르 정책 분석 — dev 176건마다 B(always)·C(장르 제외)·D(conditional) 설정에서 속성 조건 보조 필터와 장르 판정이 어떻게 다른지.
모델·DB 없음. 평가 라벨·정답 ID는 읽지 않는다(질의 ID는 행 식별에만 쓴다).

v49 Vertex 분석 캐시의 분석을 `QueryAnalysis`로 되살려 `_build_attribute_filter_with_policy`를 세 설정으로 부른다
(`ATTR_FILTER_GATE=era_plus`, B: USE_GENRE=1·POLICY=always / C: USE_GENRE=0 / D: USE_GENRE=1·POLICY=conditional). 1턴(답 없음) 부류:
  closed_all      세 설정 모두 닫힘
  open_same       B = C = D(필터 같음 — 장르 조건이 없었다)
  D_same_as_B     D = B ≠ C: D가 장르를 유지(성별 없음) — q112 유형. C만 완화
  D_relaxed       D = C ≠ B: D가 장르를 뺐다(성별 조건 있음) — 필터 완화 효과. 사유는 reason_D
  D_other         그 밖(있으면 안 되는 조합 — 있으면 코드 점검)
또 2턴에서 B와 D가 갈릴 수 있는지(`turn2_divergence`)를 서비스가 묻는 슬롯(성별·장르)만 가정해 적는다 — 시기 창이 유효하고
최초 장르가 등록 장르에 맞으며, 성별이 이미 있거나(1턴부터 다름) 성별을 물을 수 있으면(성별 답 뒤 다름) 'possible'. 시기+성별만 있는
질의(n011 유형)는 장르 답을 B·D 모두 조건으로 넣으므로 'same_by_design'. 그 밖은 'none'.
출력: genre_policy_classes.csv, 표준 출력에 집계.

    venv/bin/python experiments/reranking/results_v54_attr_genre_policy/genre_policy_analysis.py
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
from src.retrieval.known_slots import known_slots_from_query  # noqa: E402

CACHES = HERE.parent / "results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")

SETTINGS = {
    "B": {"ATTR_FILTER_USE_GENRE": "1", "ATTR_FILTER_GENRE_POLICY": "always"},
    "C": {"ATTR_FILTER_USE_GENRE": "0", "ATTR_FILTER_GENRE_POLICY": "always"},
    "D": {"ATTR_FILTER_USE_GENRE": "1", "ATTR_FILTER_GENRE_POLICY": "conditional"},
}


def decide(analysis, label: str, answers=()):
    os.environ["ATTR_FILTER_GATE"] = "era_plus"
    os.environ.update(SETTINGS[label])
    return sr._build_attribute_filter_with_policy(analysis, list(answers))


def classify(analysis) -> dict:
    out = {}
    filters = {}
    for label in SETTINGS:
        built, decision = decide(analysis, label)
        filters[label] = built
        out[f"open_{label}"] = int(built is not None)
        out[f"conds_{label}"] = len(built["$and"]) if built else 0
        out[f"reason_{label}"] = decision.reason
    b, c, d = filters["B"], filters["C"], filters["D"]
    if b is None and c is None and d is None:
        cls = "closed_all"
    elif b == c == d:
        cls = "open_same"
    elif d == b and d != c:
        cls = "D_same_as_B"
    elif d == c and d != b:
        cls = "D_relaxed"
    else:
        cls = "D_other"
    out["gate_class"] = cls
    out["had_genre_condition"] = int(decide(analysis, "D")[1].had_genre_condition)
    return out


def turn2_divergence(analysis, qid_known_slots) -> str:
    """서비스 2턴에서 B와 D가 갈릴 수 있는가(성별·장르 질문만 가정). 1턴 D가 B와 이미 다르면 'possible'."""
    era = analysis.release_era
    has_era = bool(analysis.has_release_era and era.start_year is not None and era.end_year is not None
                   and era.end_year >= era.start_year and era.end_year - era.start_year + 1 <= sr._ATTR_FILTER_MAX_ERA_YEARS)
    if not has_era:
        return "none"  # 서비스는 시기를 묻지 않는다(출생 연도 답은 생애 단계 질의에만 — 아래 life_stage 열로 따로 본다)
    genre_registered = bool(sr._registered_genres_for(analysis.genre or ""))
    gender = (analysis.vocal_gender or "").strip() in ("남성", "여성", "혼성")
    if genre_registered and gender:
        return "possible"          # 1턴부터 D가 장르를 뺀다
    if genre_registered and "vocal_gender" not in qid_known_slots:
        return "possible"          # 성별 답이 오면 D가 장르를 뺀다
    if gender and not genre_registered:
        return "same_by_design"    # 장르 답은 B·D 모두 조건(n011 유형)
    return "none"


def main() -> None:
    rows = []
    for s in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{s}_dev_vertex.json")
        for qid, entry in sorted(cache.entries.items()):
            analysis = cache.get(qid, entry["query"])
            era = analysis.release_era
            info = classify(analysis)
            known = known_slots_from_query(analysis)
            rows.append({
                "query_id": qid, "set": s, **info,
                "turn2_divergence": turn2_divergence(analysis, known),
                "life_stage": int(bool(analysis.has_life_stage)),
                "known_slots": "/".join(known),
                "era": f"{era.start_year}-{era.end_year}" if era and era.start_year else "",
                "gender": analysis.vocal_gender or "", "genre": analysis.genre or "",
                "type": "/".join(analysis.artist_type.values) if analysis.has_artist_type_clue else "",
            })
    out = HERE / "genre_policy_classes.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"dev {len(rows)}건")
    for cls in ("closed_all", "open_same", "D_same_as_B", "D_relaxed", "D_other"):
        sel = [r for r in rows if r["gate_class"] == cls]
        by_set = " · ".join(f"{s} {sum(1 for r in sel if r['set'] == s)}" for s in SETS)
        print(f"  {cls:12s} {len(sel):3d}  ({by_set})")
        if cls not in ("closed_all",):
            for r in sel:
                print(f"     {r['query_id']}: era={r['era']} gender={r['gender'] or '-'} genre={r['genre'] or '-'} type={r['type'] or '-'}"
                      f" conds B{r['conds_B']} C{r['conds_C']} D{r['conds_D']} reason_D={r['reason_D']}")
    print("\n1턴 D 사유별(경로가 열린 질의만):")
    opened = [r for r in rows if r["open_D"]]
    for reason in ("answered_genre", "gender_condition", "no_gender"):
        sel = [r for r in opened if r["reason_D"] == reason]
        relaxed = sum(1 for r in sel if r["had_genre_condition"])
        print(f"  {reason:17s} {len(sel):3d}  (그중 뺄 장르 조건이 있던 질의 {relaxed})")
    print("\n2턴에서 B/D가 갈릴 수 있는 질의(v06·v09, 성별·장르 질문 가정):")
    for s in ("v06", "v09"):
        for kind in ("possible", "same_by_design"):
            sel = [r["query_id"] for r in rows if r["set"] == s and r["turn2_divergence"] == kind]
            print(f"  {s} {kind:15s} {len(sel):3d}: {', '.join(sel)}")
    ls = [r["query_id"] for r in rows if r["life_stage"]]
    print(f"\n생애 단계 질의(출생 연도 답으로 시기 창이 생길 수 있음): {len(ls)} {ls}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
