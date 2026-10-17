"""(v67 compare_runs.py에 v68 기록 이름만 더함) 두 오프라인 실행의 질의별 입력 대조 — 후보(ID·순서·점수·가사) · 가사 보호 · 정책 생략 · 리랭커 그룹 입력 · 그룹별 첫 프롬프트 sha.

v66 실행(`v66_queries.jsonl`)과 v67 실행(`v67_queries.jsonl`)을 모두 읽는다. 모델 호출 없음.

    venv/bin/python compare_runs.py <실행 폴더 A> <실행 폴더 B> [--out <json>]

검사: 세 세트가 모두 있고 DONE이 있어야 한다 · 질의 ID 중복·누락이 없어야 한다 · 두 실행의 질의 집합이 같아야 한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SETS = ("v06", "v09", "v10")
EXPECTED = {"v06": 57, "v09": 59, "v10": 60}


def load_run(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"미완료 실행: {run_dir}")
    out: Dict[str, Dict[str, Any]] = {}
    for s in SETS:
        d = run_dir / f"{s}_dev"
        # v67·v66·v68 기록을 모두 읽는다(필드 형식은 같다 — v68은 v66 형식에 소리 실행 기록을 더한 것)
        f = next((d / n for n in ("v67_queries.jsonl", "v68_queries.jsonl", "v66_queries.jsonl") if (d / n).exists()), d / "v66_queries.jsonl")
        rows = [json.loads(line) for line in f.open(encoding="utf-8")]
        ids = [r["query_id"] for r in rows]
        if len(ids) != len(set(ids)) or len(ids) != EXPECTED[s] or "_none" in ids:
            raise SystemExit(f"{f}: 질의 수 {len(ids)} · 중복 {len(ids) - len(set(ids))} · _none {'_none' in ids}")
        for r in rows:
            r["set"] = s
            out[r["query_id"]] = r
    return out


def prompts(row: Dict[str, Any]) -> List[str]:
    """그룹마다 첫 호출 프롬프트 sha(v66 기록은 group·attempt가 있고, v67 기록은 그룹당 1회 호출이라 순서 그대로)."""
    gens = row.get("generates") or []
    if gens and "group" in gens[0]:
        return [g["contents_sha16"] for g in gens if g.get("attempt", 0) == 0]
    return [g["contents_sha16"] for g in gens]


def facets(row: Dict[str, Any]) -> Dict[str, Any]:
    cands = row.get("candidates") or []
    prot = row.get("protection") or {}
    return {
        "candidate_ids": [c["id"] for c in cands],
        "candidate_set": sorted(c["id"] for c in cands),
        "candidate_scores": [c["score"] for c in cands],
        "lyrics": [(c["id"], c["lyric"], c["lyric_score"]) for c in cands if c["lyric"]],
        "protected": prot.get("protected_ids"),
        "tiers": prot.get("tiers"),
        "policy_skip": prot.get("policy_skip"),
        "group_inputs": [(g["input_ids"], g["input_scores"], g["input_lyric"]) for g in row.get("rerank_runs") or []],
        "prompts": prompts(row),
    }


def compare(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if set(a) != set(b):
        raise SystemExit(f"질의 집합이 다르다: A만 {sorted(set(a) - set(b))[:5]} · B만 {sorted(set(b) - set(a))[:5]}")
    keys = ("candidate_set", "candidate_ids", "candidate_scores", "lyrics", "protected", "tiers", "policy_skip",
            "group_inputs", "prompts")
    diffs: Dict[str, List[str]] = {k: [] for k in keys}
    all_same = 0
    max_score_delta = 0.0
    for qid in a:
        fa, fb = facets(a[qid]), facets(b[qid])
        same = True
        for k in keys:
            if fa[k] != fb[k]:
                diffs[k].append(qid)
                same = False
        if fa["candidate_ids"] == fb["candidate_ids"]:
            for x, y in zip(fa["candidate_scores"], fb["candidate_scores"]):
                max_score_delta = max(max_score_delta, abs(x - y))
        all_same += same
    return {"queries": len(a), "all_inputs_same": all_same,
            "differs": {k: {"count": len(v), "query_ids": v} for k, v in diffs.items()},
            "max_abs_candidate_score_delta_when_order_same": max_score_delta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = {"a": str(args.run_a), "b": str(args.run_b), **compare(load_run(args.run_a), load_run(args.run_b))}
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(f"{result['all_inputs_same']}/{result['queries']} 같음 · " +
          " · ".join(f"{k} {v['count']}" for k, v in result["differs"].items()), file=sys.stderr)


if __name__ == "__main__":
    main()
