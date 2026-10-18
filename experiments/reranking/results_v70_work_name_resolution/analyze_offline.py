"""v70 오프라인 결과 요약 — 모델·DB 호출 없음. 원기록(`offline/<구성>/records.jsonl`)만 읽는다.

- 기준 동일성: `off` 후보(id·점수)가 v69 A(on) 오프라인 1회차 후보와 176/176 같은가 · `gate_abstain`이 `off`와 같은가.
- 판정(gate_v1·v2 — 실행 조건의 버전을 적는다): dev 176의 대상 수·사유 분포 · 알려진 진단 사례 18건(v67)의 판정 결과. **dev는 개발에 쓴 자료다.**
- 구성별(off 대비, 리랭킹 전 후보 30): 정답 후보 진입/탈락 · 후보 안 정답 순위 오름/내림 · 들어온/밀려난 곡 · 상태 분포 · 지연.
  "후보 회수 증가"는 리랭킹 뒤 Hit@10 증가가 아니다. 리랭킹 전 10위 안 수는 참고로만 적는다.

    venv/bin/python analyze_offline.py --out analysis/offline_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
V69_A = Path("/private/tmp/vague-finder-v67-candidate-recall/experiments/reranking/"
             "results_v69_audio_skip_repeat/offline/stability/on_server_rep1")
SETS = ("v06", "v09", "v10")
# v67 indirect_reference 꼬리표(후보 밖 14 · 후보 내 4). 재현·진단용 — 독립 평가 자료가 아니다.
KNOWN_OUT = ("c602", "c603", "c607", "c701", "n002", "n016", "n033", "n040", "n043", "n096",
             "t009", "t038", "t040", "t043")
KNOWN_IN = ("q203", "c705", "c707", "t044")
CONFIGS = ("off", "gate_abstain", "oracle", "oracle_nogate", "wrong")


OFFLINE_DIR = HERE / "offline"


def load(config: str) -> Dict[str, Dict[str, Any]]:
    path = OFFLINE_DIR / config / "records.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    out = {r["qid"]: r for r in rows}
    if len(out) != 176:
        raise SystemExit(f"{config}: {len(out)}건")
    return out


def load_v69_a() -> Dict[str, List[tuple]]:
    import ast

    out = {}
    for s in SETS:
        for line in (V69_A / f"{s}_dev" / "v69_queries.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cands = r["candidates"]
            if isinstance(cands, str):
                cands = ast.literal_eval(cands)
            out[r["query_id"]] = [(c["id"], round(float(c["score"]), 9)) for c in cands]
    return out


def best_rank(row: Dict[str, Any]) -> Optional[int]:
    rel = set(row["relevant"])
    return next((i for i, c in enumerate(row["candidates"], 1) if c["id"] in rel), None)


def compare(base: Dict[str, Dict[str, Any]], new: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    gained, lost, up, down, same_pool, displaced_rel = [], [], [], [], 0, []
    entered_total = displaced_total = 0
    for qid, b in base.items():
        n = new[qid]
        br, nr = best_rank(b), best_rank(n)
        if [c["id"] for c in b["candidates"]] == [c["id"] for c in n["candidates"]]:
            same_pool += 1
        if br is None and nr is not None:
            gained.append({"qid": qid, "rank": nr})
        elif br is not None and nr is None:
            lost.append({"qid": qid, "rank_before": br})
        elif br is not None and nr is not None and nr != br:
            (up if nr < br else down).append({"qid": qid, "before": br, "after": nr})
        wr = n.get("work_resolution") or {}
        entered_total += len(wr.get("entered_ids") or [])
        displaced = wr.get("displaced_ids") or []
        displaced_total += len(displaced)
        rel = set(b["relevant"])
        if rel & set(displaced):
            displaced_rel.append(qid)
    recall = lambda d: sum(1 for r in d.values() if best_rank(r) is not None)  # noqa: E731
    pre10 = lambda d: sum(1 for r in d.values() if (best_rank(r) or 99) <= 10)  # noqa: E731
    return {
        "recall_at_30": {"off": recall(base), "config": recall(new)},
        "pre_rerank_rank_le_10_reference_only": {"off": pre10(base), "config": pre10(new)},
        "gained": gained, "lost": lost, "rank_up": up, "rank_down": down,
        "queries_with_identical_pool": same_pool,
        "entered_songs_total": entered_total, "displaced_songs_total": displaced_total,
        "queries_where_a_correct_song_was_displaced": displaced_rel,
    }


def statuses(rows: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(((r.get("work_resolution") or {}).get("status") or "no_record") for r in rows.values()))


def latency(rows: Dict[str, Dict[str, Any]], qids=None) -> Dict[str, float]:
    xs = [r["search_ms"] for q, r in rows.items() if qids is None or q in qids]
    xs_sorted = sorted(xs)
    return {"n": len(xs), "median_ms": round(statistics.median(xs), 1) if xs else 0.0,
            "p90_ms": round(xs_sorted[int(0.9 * (len(xs) - 1))], 1) if xs else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-query-csv", type=Path)
    # v2(리뷰 반영 코드)는 `offline_v2/`에 따로 돌려 v1 결과를 덮지 않는다
    ap.add_argument("--offline-dir", type=Path, default=HERE / "offline")
    args = ap.parse_args()
    global OFFLINE_DIR
    OFFLINE_DIR = args.offline_dir
    runs = {c: load(c) for c in CONFIGS}
    off = runs["off"]
    if not off:
        raise SystemExit("off가 없다")
    summary: Dict[str, Any] = {"note": "리랭킹 전 후보 30 · 외부 모델 0회 · dev 176(개발 자료, 독립 평가 아님)"}
    v69 = load_v69_a()
    mism = [q for q, r in off.items() if [(c["id"], c["score"]) for c in r["candidates"]] != v69[q]]
    summary["off_vs_v69_A_offline_rep1"] = {"identical_queries": 176 - len(mism), "mismatched": mism}
    gate = Counter(r["gate"]["reason"] for r in off.values())
    triggered = sorted(q for q, r in off.items() if r["gate"]["triggered"])
    conditions = OFFLINE_DIR / "off" / "run_conditions.json"
    gate_version = (json.loads(conditions.read_text(encoding="utf-8")).get("versions") or {}).get("gate") \
        if conditions.exists() else None
    summary["gate_dev176"] = {
        "gate_version": gate_version,
        "triggered": len(triggered), "reasons": dict(gate),
        "known_out_of_pool_14": {q: off[q]["gate"]["reason"] for q in KNOWN_OUT},
        "known_in_pool_4": {q: off[q]["gate"]["reason"] for q in KNOWN_IN},
        "triggered_outside_known_18": len(set(triggered) - set(KNOWN_OUT) - set(KNOWN_IN)),
    }
    summary["latency_off"] = latency(off)
    for config in CONFIGS[1:]:
        rows = runs[config]
        if not rows:
            summary[config] = "not_run"
            continue
        block = compare(off, rows)
        block["statuses"] = statuses(rows)
        block["latency"] = latency(rows)
        block["latency_triggered_only"] = {"config": latency(rows, set(triggered)), "off": latency(off, set(triggered))}
        if config == "gate_abstain":
            block["identical_candidates_and_scores_to_off"] = sum(
                1 for q in off if off[q]["candidates"] == rows[q]["candidates"])
        if config in ("oracle", "oracle_nogate"):
            listed = [q for q in rows if (rows[q].get("work_resolution") or {}).get("works")]
            block["oracle_queries_with_works"] = sorted(listed)
            block["per_oracle_query"] = {
                q: {"gate": off[q]["gate"]["reason"], "status": (rows[q].get("work_resolution") or {}).get("status"),
                    "mention_songs": [w.get("mention_songs") for w in (rows[q]["work_resolution"].get("works") or [])],
                    "aux": len(rows[q]["work_resolution"].get("aux_ids") or []),
                    "entered": len(rows[q]["work_resolution"].get("entered_ids") or []),
                    "rank_off": best_rank(off[q]), "rank_config": best_rank(rows[q])}
                for q in sorted(listed)}
        summary[config] = block
    # 반복·배선 확인: oracle 2회차가 1회차와 같은가 · 서비스 공장 경로(cache: 해석기, 가짜 보류 캐시)가 off와 같은가
    def _key(r: Dict[str, Any]) -> tuple:
        w = r.get("work_resolution") or {}
        return (r["candidates"], w.get("status"), w.get("aux_ids"), w.get("entered_ids"), w.get("displaced_ids"))

    rep2, env = load("oracle_rep2"), load("env_cache")
    summary["repeat_and_wiring"] = {
        "oracle_rep1_vs_rep2_identical": sum(_key(runs["oracle"][q]) == _key(rep2[q]) for q in off) if rep2 else "not_run",
        "env_cache_candidates_identical_to_off": sum(env[q]["candidates"] == off[q]["candidates"] for q in off) if env else "not_run",
        "env_cache_statuses": statuses(env) if env else "not_run",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.per_query_csv:
        with open(args.per_query_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "set", "gate", *[f"rank_{c}" for c in CONFIGS],
                        *[f"status_{c}" for c in CONFIGS[1:]], *[f"entered_{c}" for c in CONFIGS[1:]],
                        *[f"displaced_{c}" for c in CONFIGS[1:]]])
            for q, r in off.items():
                w.writerow([q, r["set"], r["gate"]["reason"],
                            *[(best_rank(runs[c][q]) if runs[c] else "") or "" for c in CONFIGS],
                            *[((runs[c][q].get("work_resolution") or {}).get("status") if runs[c] else "") for c in CONFIGS[1:]],
                            *[len((runs[c][q].get("work_resolution") or {}).get("entered_ids") or []) if runs[c] else "" for c in CONFIGS[1:]],
                            *[len((runs[c][q].get("work_resolution") or {}).get("displaced_ids") or []) if runs[c] else "" for c in CONFIGS[1:]]])
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                             if kk in ("recall_at_30", "identical_queries", "triggered",
                                                                       "gained", "lost", "statuses", "reasons")})
                      for k, v in summary.items()}, ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
