"""v61 — 기존 평가 경로(`src.retrieval.evaluate_search_accuracy`, 리랭킹 끔)로 서비스 후보 30개까지 뽑는다. 구성(arm)만 다르다.

  A: 현행 구현(`HYBRID_SCAN_LIMIT` 기본 5000)
  B: 하이브리드 호출을 독립 정확 합산(`exact_hybrid.py`)으로 바꿔 끼움
  C: 현행 구현에 상한만 실제 point 수(`count(exact)`)로 — "실제 대상 수를 반영한 조회" 수정안의 결과 확인용. 기준은 B다.

세 구성은 같은 분석 캐시(v49 Vertex dev) · 같은 색인 복사본(`artifacts/qdrant_v61_audit`) · 같은 설정(`v61_guard.fixed_env`)을
쓴다. 모델·Mongo·운영 Qdrant 호출은 `v61_guard`가 막고, 끝나면 차단 확인 결과를 `conditions.json`에 적는다.

실행(저장소 루트에서):
    venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/run_candidates.py --arm A --set v06 [--query-ids q101,q102] [--rep 1]
산출: <폴더>/cand_<arm>[_rep<n>]/<set>_dev/search_eval_dev_{detail,summary,top10,explain,runinfo} (평가기 산출물 그대로)
      <폴더>/cand_<arm>[_rep<n>]/<set>_dev/conditions.json · calls.jsonl(하이브리드 호출 기록) · calls_vectors.npz(질의 벡터·조회 id, 커밋 제외)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import v61_guard  # noqa: E402

CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "B", "C"], required=True)
    parser.add_argument("--set", choices=["v06", "v09", "v10"], required=True)
    parser.add_argument("--query-ids", default="")
    parser.add_argument("--rep", type=int, default=0, help="반복 번호(0이면 폴더 이름에 붙이지 않는다)")
    parser.add_argument("--out-root", default=str(HERE),
                        help="출력 루트. 수정 코드 검증처럼 기존 기준선(cand_A 등)을 보존해야 할 때는 다른 폴더를 준다")
    parser.add_argument("--overwrite", action="store_true", help="같은 출력 폴더에 이미 평가 산출물이 있어도 덮어쓴다(기본은 중단)")
    args = parser.parse_args()

    applied_env = v61_guard.fixed_env()
    v61_guard.install()
    guard = v61_guard.verify()

    import src.backend.api.dependencies as deps  # noqa: E402
    from src.retrieval import evaluate_search_accuracy as esa  # noqa: E402
    from src.vector_db import qdrant_backend as qb  # noqa: E402
    from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME  # noqa: E402

    from exact_hybrid import ExactHybridReference  # noqa: E402
    from hybrid_capture import HybridCapture  # noqa: E402

    run_name = f"cand_{args.arm}" + (f"_rep{args.rep}" if args.rep else "")
    out_dir = Path(args.out_root) / run_name / f"{args.set}_dev"
    existing = [p for p in ("search_eval_dev_detail.csv", "calls.jsonl", "conditions.json") if (out_dir / p).exists()]
    if existing and not args.overwrite:
        # 저장된 기준선(수정 전 A/B/C 기록)을 수정 코드 검증 실행이 조용히 덮어쓰지 않게 한다
        raise SystemExit(
            f"출력 폴더에 이미 산출물이 있다: {out_dir} ({', '.join(existing)}). "
            "다른 --out-root를 주거나(권장: 기준선 보존), 정말 덮어쓰려면 --overwrite를 준다."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    client = deps.get_vector_client().client
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    point_count = int(client.count(collection, exact=True).count)
    default_limit = qb.HYBRID_SCAN_LIMIT

    reference = None
    reference_build_s = None
    fingerprint = None
    if args.arm == "B":
        t0 = time.perf_counter()
        reference = ExactHybridReference.from_client(client, collection)
        reference_build_s = round(time.perf_counter() - t0, 3)
        fingerprint = reference.fingerprint()
    scan_limit_used = default_limit
    if args.arm == "C":
        qb.HYBRID_SCAN_LIMIT = point_count
        scan_limit_used = point_count
    rss_after_prepare = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    capture = HybridCapture(
        args.arm, reference=reference,
        record_path=out_dir / "calls.jsonl", vectors_path=out_dir / "calls_vectors.npz",
    )
    capture.install()
    rows = esa.read_queries(ROOT / f"experiments/reranking/eval_queries_{args.set}.csv", "dev")
    capture.set_query_ids({row["query"]: row["query_id"].strip() for row in rows})

    eval_args = esa.build_parser().parse_args([
        "--input", f"experiments/reranking/eval_queries_{args.set}.csv",
        "--split", "dev",
        "--analysis-cache", str(CACHES / f"analysis_cache_{args.set}_dev_vertex.json"),
        "--output-dir", str(out_dir),
        *(["--query-ids", args.query_ids] if args.query_ids else []),
    ])
    eval_started = time.perf_counter()
    asyncio.run(esa.evaluate(eval_args))
    eval_s = round(time.perf_counter() - eval_started, 3)
    capture_summary = capture.save()
    rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    guard_after = v61_guard.verify()
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    conditions = {
        "arm": args.arm,
        "set": args.set,
        "rep": args.rep,
        "query_ids": args.query_ids,
        "git_sha": git_sha,
        "collection": collection,
        "point_count": point_count,
        "hybrid_scan_limit_default": default_limit,
        "hybrid_scan_limit_used": scan_limit_used,
        "scan_limit_truncates": (args.arm != "B") and scan_limit_used < point_count,
        "reference_fingerprint": fingerprint,
        "reference_build_seconds": reference_build_s,
        "env": applied_env,
        "guard_before": guard,
        "guard_after": guard_after,
        "capture": capture_summary,
        "eval_seconds": eval_s,
        "total_seconds": round(time.perf_counter() - started, 3),
        "ru_maxrss_bytes": {"before": rss_before, "after_prepare": rss_after_prepare, "end": rss_end},
        "qdrant_meta_mtime": os.path.getmtime(ROOT / applied_env["QDRANT_PATH"] / "meta.json"),
    }
    (out_dir / "conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: conditions[k] for k in ("arm", "set", "point_count", "hybrid_scan_limit_used", "capture", "eval_seconds")},
                     ensure_ascii=False))
    deps.close_vector_client()
    return 0


if __name__ == "__main__":
    sys.exit(main())
