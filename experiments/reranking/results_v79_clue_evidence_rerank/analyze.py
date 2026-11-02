"""실행 폴더들을 모아 RUN_INFO §3의 지표와 판정을 낸다. 모델 호출 없음.

완료되지 않았거나 리랭킹 실패가 섞인 실행은 비교하지 않고 판정을 보류한다(이어 붙이거나 조용히 빼지 않는다).
사용: analyze.py [--runs-dir 폴더] [--out 파일] — 기본 네 실행 off_rep1·hits_rep1·hits_rep2·off_rep2.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parents[2]))
SETS = ("v06", "v09", "v10")
EXPECTED = {"v06": 57, "v09": 59, "v10": 60}
A_RUNS = ("off_rep1", "off_rep2")
B_RUNS = ("hits_rep1", "hits_rep2")
FAILED_OUTCOMES = {"failed", "budget_exhausted", "partial_failed", "model_partial"}


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _num(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_run(runs_dir: Path, run: str) -> Dict[str, Any]:
    """{status, rows, problems}. status = complete | incomplete | failed."""
    rows: Dict[str, Dict[str, Any]] = {}
    problems: List[str] = []
    for name in SETS:
        path = runs_dir / run / name / "search_eval_dev_detail.csv"
        if not path.exists():
            problems.append(f"{name}: detail 없음")
            continue
        got = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        if len(got) != EXPECTED[name]:
            problems.append(f"{name}: {len(got)}/{EXPECTED[name]}건")
        for row in got:
            row["_set"] = name
            rows[row["query_id"]] = row
    status = "incomplete" if problems else "complete"
    bad = [q for q, r in rows.items() if _flag(r.get("rerank_fell_back")) or _flag(r.get("rerank_partial_failure")) or str(r.get("rerank_outcome", "")).strip() in FAILED_OUTCOMES or str(r.get("rerank_error", "")).strip()]
    if bad:
        problems.append(f"리랭킹 실패 {len(bad)}건: {bad[:8]}")
        status = "failed" if status == "complete" else status
    return {"status": status, "rows": rows, "problems": problems}


def metrics(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {"n": len(rows)}
    for key in ("hit@1", "hit@5", "hit@10"):
        out[key] = sum(_flag(r.get(f"rerank_{key}")) for r in rows.values())
    mrr = [_num(r.get("rerank_mrr@10")) or 0.0 for r in rows.values()]
    out["mrr"] = round(sum(mrr) / len(mrr), 4) if mrr else None
    out["candidate_recall"] = sum((_num(r.get("candidate_recall@30")) or 0.0) > 0 for r in rows.values())
    return out


def hit10_ids(rows: Dict[str, Dict[str, Any]]) -> set:
    return {q for q, r in rows.items() if _flag(r.get("rerank_hit@10"))}


def _song_loader(path: Path) -> Dict[str, Any]:
    out = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        m, s, l = r["metadata"], r["semantic_analysis"], r["lyrics_data"]
        tags = ("sound_tags", "place_activity_tags", "time_weather_tags", "emotion_tags", "mood_tags", "vibe_tags", "relation_context_tags")
        out[str(r["song_id"])] = types.SimpleNamespace(
            id=str(r["song_id"]), title=m.get("title"), album=m.get("album"), artist_types=m.get("type") or [],
            genre=", ".join(m.get("genre") or []), vocal_gender=m.get("vocal_gender"),
            lyrics_summary=l.get("lyrics_summary") or "", search_style_summary=s.get("search_style_summary") or "",
            **{k: s.get(k) or [] for k in tags},
        )
    return out


def clue_satisfaction(runs_dir: Path, run: str, songs: Dict[str, Any], analyses: Dict[str, Any]) -> Dict[str, Any]:
    """질의마다 상위 10곡에서 found·partial로 판정된 단서 비율의 평균. 단서가 없는 질의는 뺀다."""
    from src.retrieval.user_explanation import clue_evidence_for_rerank
    per_query: List[float] = []
    top1: List[float] = []
    for name in SETS:
        for row in csv.DictReader((runs_dir / run / name / "search_eval_dev_top10.csv").open(encoding="utf-8-sig")):
            analysis = analyses.get(row["query_id"])
            if analysis is None:
                continue
            ids = [x for x in _top10_ids(row) if x in songs]
            shares = []
            for song_id in ids:
                check = clue_evidence_for_rerank(analysis.original_query, analysis, songs[song_id])
                hit = len(check.get("found", [])) + len(check.get("partial", []))
                total = hit + len(check.get("not_found", []))
                if total:
                    shares.append(hit / total)
            if shares:
                per_query.append(sum(shares) / len(shares))
                top1.append(shares[0])
    return {
        "queries": len(per_query),
        "top10": round(sum(per_query) / len(per_query), 4) if per_query else None,
        "top1": round(sum(top1) / len(top1), 4) if top1 else None,
    }


def _top10_ids(row: Dict[str, Any]) -> List[str]:
    for key in ("rerank_top_ids",):
        value = row.get(key)
        if value:
            return [x for x in str(value).split("|") if x]
    raise KeyError(f"top10 열을 찾지 못함: {list(row)[:12]}")


def judge(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    need = (*A_RUNS, *B_RUNS)
    not_ok = {r: runs[r]["problems"] for r in need if runs.get(r, {}).get("status") != "complete"}
    if not_ok:
        return {"verdict": "보류_유효성", "S0": False, "not_ok": not_ok}
    m = {r: metrics(runs[r]["rows"]) for r in need}
    a_mean = {k: sum(m[r][k] for r in A_RUNS) / 2 for k in ("hit@1", "mrr")}
    s1_all = all(m[b]["hit@10"] >= m[a]["hit@10"] for b in B_RUNS for a in A_RUNS)
    s1_any = any(m[b]["hit@10"] > m[a]["hit@10"] for b in B_RUNS for a in A_RUNS)
    a_all = hit10_ids(runs[A_RUNS[0]]["rows"]) & hit10_ids(runs[A_RUNS[1]]["rows"])
    a_any = hit10_ids(runs[A_RUNS[0]]["rows"]) | hit10_ids(runs[A_RUNS[1]]["rows"])
    b_all = hit10_ids(runs[B_RUNS[0]]["rows"]) & hit10_ids(runs[B_RUNS[1]]["rows"])
    b_any = hit10_ids(runs[B_RUNS[0]]["rows"]) | hit10_ids(runs[B_RUNS[1]]["rows"])
    # 반복마다 갈리는 질의는 효과로 세지 않는다 — 양쪽 B에서 잃고(또는 얻고) 양쪽 A와 달라야 한다
    lost = sorted(a_all - b_any)
    gained = sorted(b_all - a_any)
    s3 = all(m[b]["hit@1"] >= a_mean["hit@1"] and m[b]["mrr"] >= a_mean["mrr"] for b in B_RUNS)
    return {
        "S0": True,
        "S1": s1_all and s1_any,
        "S2": len(lost) < len(gained),
        "S3": s3,
        "lost": lost,
        "gained": gained,
        "metrics": m,
        "same_config_hit10_diff": {
            "A1_vs_A2": sorted(hit10_ids(runs[A_RUNS[0]]["rows"]) ^ hit10_ids(runs[A_RUNS[1]]["rows"])),
            "B1_vs_B2": sorted(hit10_ids(runs[B_RUNS[0]]["rows"]) ^ hit10_ids(runs[B_RUNS[1]]["rows"])),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=str(R / "runs"))
    p.add_argument("--out", default=str(R / "analysis.json"))
    p.add_argument("--songs", default=os.environ.get("V79_SONGS", ""))
    a = p.parse_args()
    runs_dir = Path(a.runs_dir)
    runs = {r: load_run(runs_dir, r) for r in (*A_RUNS, *B_RUNS)}
    result = judge(runs)
    if result.get("S0") and a.songs:
        from src.backend.schemas.query import QueryAnalysis
        songs = _song_loader(Path(a.songs))
        analyses: Dict[str, Any] = {}
        for name in SETS:
            cache = json.load((R.parent / "results_v49_vertex_baseline" / f"analysis_cache_{name}_dev_vertex.json").open(encoding="utf-8"))
            for qid, entry in cache["entries"].items():
                analyses[qid] = QueryAnalysis.model_validate(entry["analysis"])
        sat = {r: clue_satisfaction(runs_dir, r, songs, analyses) for r in (*A_RUNS, *B_RUNS)}
        result["clue_satisfaction"] = sat
        result["S4"] = min(sat[b]["top10"] for b in B_RUNS) > max(sat[a]["top10"] for a in A_RUNS)
        keys = ("S1", "S2", "S3", "S4")
        result["verdict"] = "채택_후보" if all(result[k] for k in keys) else "보류_" + "".join(k for k in keys if not result[k])
    elif result.get("S0"):
        result["verdict"] = "보류_S4_미계산"
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "clue_satisfaction"}, ensure_ascii=False)[:1400])
    if "clue_satisfaction" in result:
        print(json.dumps(result["clue_satisfaction"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
