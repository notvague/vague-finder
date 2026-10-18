"""v72 부분집합 실행 = 전체 실행 확인 — 같은 구성의 가짜 리랭커 기록에서 부분집합 질의의 입력 지문(compare_inputs.view의 입력 항목 +
최종 순서)과 Qdrant 경로 반환이 같은가. 모델·DB 호출 없음.

    venv/bin/python check_subset_vs_full.py --full <실행 폴더> --subset <실행 폴더> [--out <json>]
종료 코드: 0 같음 · 3 다름
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_inputs as ci  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    full, sub = ci.load_run(args.full), ci.load_run(args.subset)
    qids = sorted(sub["queries"])
    want = sorted(q for s in sub["cond"]["subset_query_ids"].values() for q in s)
    diffs = {}
    for q in qids:
        vf, vs = ci.view(full["queries"][q]), ci.view(sub["queries"][q])
        fields = [f for f in list(ci.INPUT_FIELDS) + ["final"] if vf[f] != vs[f]]
        pf, ps = ci.path_signature(full["calls"].get(q, [])), ci.path_signature(sub["calls"].get(q, []))
        paths = sorted(k for k in set(pf) | set(ps) if pf.get(k) != ps.get(k))
        if fields or paths:
            diffs[q] = {"fields": fields, "paths": paths}
    result = {"full": full["name"], "subset": sub["name"], "subset_sha16": sub["cond"].get("subset_sha16"),
              "queries": len(qids), "expected_queries": len(want), "ids_match": qids == want,
              "identical": len(qids) - len(diffs), "diffs": diffs}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if (result["ids_match"] and not diffs) else 3


if __name__ == "__main__":
    sys.exit(main())
