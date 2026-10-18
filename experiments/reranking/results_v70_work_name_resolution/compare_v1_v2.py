"""v70 오프라인 원기록 두 판 비교(기본 v1 `offline/` ↔ v2 `offline_v2/`, `--a`·`--b`로 v2 ↔ v3도) — 모델·DB 호출 없음.

질의 원문·작품명은 쓰지 않는다(질의 id·판정 사유·상태·거절 사유 수·후보 일치 여부만).

    venv/bin/python compare_v1_v2.py --out analysis_v2/v1_vs_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
CONFIGS = ("off", "gate_abstain", "oracle", "oracle_nogate", "wrong", "oracle_rep2", "env_cache")


def load(root: Path, config: str) -> Dict[str, Dict[str, Any]]:
    path = root / config / "records.jsonl"
    if not path.exists():
        return {}
    rows = {r["qid"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    if len(rows) != 176:
        raise SystemExit(f"{path}: {len(rows)}건")
    return rows


def reject_reasons(rows: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    out: Counter = Counter()
    for r in rows.values():
        for w in (r.get("work_resolution") or {}).get("works") or []:
            out["accepted" if w.get("accepted") else (w.get("reject_reason") or "?")] += 1
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    # 기본은 v1(offline/) ↔ v2(offline_v2/). v2 ↔ v3은 --a offline_v2 --b offline_v3
    ap.add_argument("--a", default="offline")
    ap.add_argument("--b", default="offline_v2")
    args = ap.parse_args()
    summary: Dict[str, Any] = {"note": f"a = {args.a}/, b = {args.b}/ (v1 리뷰 반영 전 · v2 반영 후 · v3 리뷰 5 반영). "
                                       "dev 176 · 리랭킹 전 후보 30 · 모델 0회"}
    for config in CONFIGS:
        v1, v2 = load(HERE / args.a, config), load(HERE / args.b, config)
        if not v1 or not v2:
            summary[config] = "not_run"
            continue
        block: Dict[str, Any] = {
            "identical_candidates_and_scores": sum(v1[q]["candidates"] == v2[q]["candidates"] for q in v1),
            "candidate_mismatch_qids": sorted(q for q in v1 if v1[q]["candidates"] != v2[q]["candidates"]),
            "gate_reason_changes": {q: [v1[q]["gate"]["reason"], v2[q]["gate"]["reason"]]
                                    for q in sorted(v1) if v1[q]["gate"]["reason"] != v2[q]["gate"]["reason"]},
            "status_changes": {q: [(v1[q].get("work_resolution") or {}).get("status"),
                                   (v2[q].get("work_resolution") or {}).get("status")]
                               for q in sorted(v1)
                               if (v1[q].get("work_resolution") or {}).get("status")
                               != (v2[q].get("work_resolution") or {}).get("status")},
            "work_verdicts": {"a": reject_reasons(v1), "b": reject_reasons(v2)},
            "name_in_query_b": sum(bool(w.get("name_in_query")) for r in v2.values()
                                   for w in (r.get("work_resolution") or {}).get("works") or []),
        }
        summary[config] = block
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
