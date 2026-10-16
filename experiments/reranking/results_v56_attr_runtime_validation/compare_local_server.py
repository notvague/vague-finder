"""§4 보강 — 같은 코드·같은 분석으로 **로컬 파일 모드**(복사본)와 **서버 모드**의 후보 목록(리랭킹 없음)을 대조한다. 모델 호출 없음.

서버로 옮기면서 자료·필터 의미가 바뀌었는지의 직접 확인이다. 라우터 `search()`를 직접 부르고(평가기와 같은 경로)
candidate_ids 30개를 기록한다. `--mode local|server`, `--arm A|D` 로 4번 돌린 뒤 `--compare` 로 대조한다.

실행:
  for m in local server; do for a in A D; do venv/bin/python …/compare_local_server.py --mode $m --arm $a; done; done
  venv/bin/python …/compare_local_server.py --compare
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

QUERY_IDS = ["t014", "t049", "c601", "n018", "t024", "c602", "n019", "t029", "c603", "n020", "t031", "c605",
             "m103", "n002", "t001", "q101", "n006", "t004", "q104", "n010", "t007", "q107"]
ARMS = {
    "A": {"ATTR_FILTER_WEIGHT": "0", "ATTR_FILTER_GENRE_POLICY": "always"},
    "D": {"ATTR_FILTER_WEIGHT": "0.5", "ATTR_FILTER_GENRE_POLICY": "conditional"},
}


def run(mode: str, arm: str) -> None:
    import v56_runtime

    v56_runtime.block_real_models()
    if mode == "server":
        os.environ["QDRANT_URL"] = os.getenv("V56_QDRANT_URL", "http://127.0.0.1:6356")
    else:
        os.environ.pop("QDRANT_URL", None)
        os.environ["QDRANT_PATH"] = "artifacts/qdrant_eval_v56_local"
    os.environ.update({"ATTR_FILTER_GATE": "era_plus", "ATTR_FILTER_K": "30", "ATTR_FILTER_USE_GENRE": "1",
                       "FEEDBACK_COLLECTION_ENABLED": "0", "SEARCH_REFERENCE_YEAR": "2026", "TOKENIZERS_PARALLELISM": "false",
                       **ARMS[arm]})
    import src.backend.api.dependencies as deps

    deps.get_reranker = deps.singleton(v56_runtime.make_fake_reranker)  # type: ignore[assignment]
    from src.backend.schemas.query import QueryAnalysis

    analyzer = v56_runtime.CachedAnalyzer()
    by_id = {qid: a for _q, (qid, a, _f, _p) in analyzer.by_query.items()}
    router = deps.get_search_router()
    out = {"mode": mode, "arm": arm, "qdrant": os.getenv("QDRANT_URL") or os.getenv("QDRANT_PATH"), "candidates": {}}
    for qid in QUERY_IDS:
        analysis = QueryAnalysis(**by_id[qid])
        cands: list = []
        asyncio.run(router.search(analysis, top_k=15, use_rerank=False, candidate_k=30, candidate_ids_out=cands))
        out["candidates"][qid] = cands
        print(mode, arm, qid, len(cands), cands[:5], flush=True)
    router.shutdown()
    try:
        deps.close_vector_client()
    except Exception:
        pass
    tag = os.getenv("V56_TAG", "")
    (HERE / f"local_vs_server_{mode}_{arm}{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def compare() -> None:
    rows = []
    for arm in ARMS:
        local = json.loads((HERE / f"local_vs_server_local_{arm}.json").read_text(encoding="utf-8"))["candidates"]
        server = json.loads((HERE / f"local_vs_server_server_{arm}.json").read_text(encoding="utf-8"))["candidates"]
        for qid in QUERY_IDS:
            a, b = local[qid], server[qid]
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            rows.append({"arm": arm, "query_id": qid, "n_local": len(a), "n_server": len(b), "identical": a == b,
                         "same_set": set(a) == set(b), "first_diff_index": first,
                         "only_local": sorted(set(a) - set(b)), "only_server": sorted(set(b) - set(a))})
            print(f"{arm} {qid:>5} identical={a == b} same_set={set(a) == set(b)} first_diff={first} only_local={sorted(set(a)-set(b))} only_server={sorted(set(b)-set(a))}")
    summary = {"identical": sum(r["identical"] for r in rows), "total": len(rows),
               "same_set_not_order": sum(1 for r in rows if r["same_set"] and not r["identical"]),
               "different_set": sum(1 for r in rows if not r["same_set"]), "rows": rows}
    (HERE / "local_vs_server_compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "server"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--ids", default="", help="쉼표 구분 질의 id — 주면 QUERY_IDS 대신(재현성 반복 확인용)")
    parser.add_argument("--tag", default="", help="출력 파일 이름 접미사")
    args = parser.parse_args()
    if args.ids:
        QUERY_IDS[:] = [x.strip() for x in args.ids.split(",") if x.strip()]
    if args.compare:
        compare()
    else:
        run(args.mode, args.arm)
