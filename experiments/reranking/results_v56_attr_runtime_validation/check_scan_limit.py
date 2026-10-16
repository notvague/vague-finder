"""§7 관찰 재현 — 하이브리드 전수 합산 상한 `HYBRID_SCAN_LIMIT`(기본 5000) < 코퍼스 5,143점.

dense 조회가 상위 5,000점만 돌아오므로 dense 점수가 가장 낮은 143곡은 하이브리드 합산에서 dense 몫을 잃는다. 어느 143곡이
잘리는지는 동점·부동소수 처리에 따라 로컬 파일 모드와 서버 모드가 다를 수 있어, t049 후보 목록이 v55(로컬)와 달랐던 원인 후보다.
모델 호출 없음(임베딩만). 상한을 5000과 6000으로 바꿔 같은 질의의 텍스트 경로 상위 30을 비교하고, 문제 곡의 dense 순위를 센다.

실행: QDRANT_URL=http://127.0.0.1:6356 venv/bin/python experiments/reranking/results_v56_attr_runtime_validation/check_scan_limit.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import v56_runtime  # noqa: E402

v56_runtime.block_real_models()
os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6356")
os.environ["FEEDBACK_COLLECTION_ENABLED"] = "0"
os.environ["ATTR_FILTER_WEIGHT"] = "0"

import src.backend.api.dependencies as deps  # noqa: E402

deps.get_reranker = deps.singleton(v56_runtime.make_fake_reranker)  # type: ignore[assignment]
from src.vector_db import qdrant_backend as qb  # noqa: E402
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME  # noqa: E402

QUERIES = {"t049": "4020215", "t014": None}


def main() -> int:
    analyzer = v56_runtime.CachedAnalyzer()
    by_id = {qid: (q, a) for q, (qid, a, _f, _p) in analyzer.by_query.items()}
    router = deps.get_search_router()
    service = deps.get_search_service()
    client = deps.get_vector_client().client
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    total = client.count(collection, exact=True).count
    report = {"collection": collection, "points": total, "default_limit": qb.HYBRID_SCAN_LIMIT, "queries": {}}
    for qid, suspect in QUERIES.items():
        from src.backend.schemas.query import QueryAnalysis
        analysis = QueryAnalysis(**by_id[qid][1])
        lists = {}
        for limit in (5000, 6000):
            qb.HYBRID_SCAN_LIMIT = limit
            hits = router._search_text(analysis, 30)
            lists[limit] = [t.id for t in hits]
        qb.HYBRID_SCAN_LIMIT = 5000
        entry = {"top30_5000": lists[5000], "top30_6000": lists[6000], "same": lists[5000] == lists[6000],
                 "first_diff": next((i for i, (a, b) in enumerate(zip(lists[5000], lists[6000])) if a != b), None)}
        if suspect:
            # dense 전체 순위(5,143점 모두)와 sparse 적중 여부
            dense_vec = service.text_embedder.embed_passages([f"query: {analysis.search_text}"], add_e5_prefix=False, normalize=True)[0].tolist()
            pts = client.query_points(collection, query=dense_vec, using=qb.DENSE_VECTOR, limit=total, with_payload=False,
                                      search_params=qb.models.SearchParams(exact=True)).points
            ranks = {str(p.id): i + 1 for i, p in enumerate(pts)}
            entry["suspect"] = suspect
            entry["suspect_dense_rank_of_total"] = ranks.get(suspect)
            entry["suspect_in_top30_5000"] = suspect in lists[5000]
            entry["suspect_in_top30_6000"] = suspect in lists[6000]
            entry["suspect_rank_6000"] = lists[6000].index(suspect) + 1 if suspect in lists[6000] else None
        report["queries"][qid] = entry
        print(qid, {k: v for k, v in entry.items() if not k.startswith("top30")})
    (HERE / "scan_limit_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    router.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
