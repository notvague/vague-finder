"""§4-1 진단 — **같은 질의 벡터**로 로컬 파일 모드와 서버 모드의 컬렉션별 조회 결과를 직접 대조한다(임베딩 비결정성 배제). 모델 호출 없음.

한 프로세스에서 질의 벡터를 한 번 만들고(KoE5 dense · BM25 sparse · SigLIP2 · CLAP) 두 클라이언트(로컬 복사본 · 서버)에 똑같이 던진다.
  text dense 상위 5,000 집합 · sparse 적중 집합 · image/audio 상위 60(서버는 exact=True, 로컬은 전수) · Context dense 상위 100(서비스 코드처럼 search_params 없음 → 서버는 HNSW 근사)
실행: venv/bin/python …/probe_backends.py
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
os.environ.pop("QDRANT_URL", None)
os.environ["QDRANT_PATH"] = "artifacts/qdrant_eval_v56_local"
os.environ["FEEDBACK_COLLECTION_ENABLED"] = "0"

from qdrant_client import QdrantClient, models  # noqa: E402

import src.backend.api.dependencies as deps  # noqa: E402
from src.backend.schemas.query import QueryAnalysis  # noqa: E402
from src.vector_db import qdrant_backend as qb  # noqa: E402
from src.vector_db.settings import (AUDIO_INDEX_NAME, CONTEXT_DENSE_INDEX_NAME, IMAGE_INDEX_NAME, NAMESPACE,  # noqa: E402
                                    TEXT_HYBRID_INDEX_NAME)

QUERY_IDS = ["q101", "t049", "c601", "n010", "m103"]
SERVER_URL = os.getenv("V56_QDRANT_URL", "http://127.0.0.1:6356")


def ids(points):
    return [str(p.id) for p in points]


def main() -> int:
    analyzer = v56_runtime.CachedAnalyzer()
    by_id = {qid: a for _q, (qid, a, _f, _p) in analyzer.by_query.items()}
    text_emb, bm25 = deps.get_text_embedder(), deps.get_bm25_encoder()
    img_emb, aud_emb = deps.get_image_embedder(), deps.get_audio_embedder()
    local = QdrantClient(path="artifacts/qdrant_eval_v56_local")
    server = QdrantClient(url=SERVER_URL)
    coll = lambda name: qb.collection_name(name, NAMESPACE)  # noqa: E731
    ctx_coll = next(c.name for c in server.get_collections().collections if c.name.startswith(CONTEXT_DENSE_INDEX_NAME))
    report = {}
    for qid in QUERY_IDS:
        a = QueryAnalysis(**by_id[qid])
        dense = text_emb.embed_passages([f"query: {a.search_text}"], add_e5_prefix=False, normalize=True)[0].tolist()
        enc = bm25.encode_queries(" ".join(a.korean_tags) or a.search_text)
        enc = enc[0] if isinstance(enc, list) else enc
        sparse = models.SparseVector(indices=[int(i) for i in enc.get("indices", [])], values=[float(v) for v in enc.get("values", [])])
        entry = {}

        def both(collection, query, using, limit, exact_on_server=True, with_filter=None):
            lp = local.query_points(collection, query=query, using=using, limit=limit, with_payload=False, query_filter=with_filter).points
            sp = server.query_points(collection, query=query, using=using, limit=limit, with_payload=False, query_filter=with_filter,
                                     search_params=models.SearchParams(exact=True) if exact_on_server else None).points
            li, si = ids(lp), ids(sp)
            first = next((i for i, (x, y) in enumerate(zip(li, si)) if x != y), None)
            maxdiff = max((abs(float(x.score) - float(y.score)) for x, y in zip(lp, sp) if str(x.id) == str(y.id)), default=None)
            return {"n_local": len(li), "n_server": len(si), "identical_order": li == si, "same_set": set(li) == set(si),
                    "first_diff": first, "only_local": len(set(li) - set(si)), "only_server": len(set(si) - set(li)),
                    "max_abs_score_diff_same_id": maxdiff}

        entry["text_dense_top5000"] = both(coll(TEXT_HYBRID_INDEX_NAME), dense, qb.DENSE_VECTOR, 5000)
        entry["text_dense_top60"] = both(coll(TEXT_HYBRID_INDEX_NAME), dense, qb.DENSE_VECTOR, 60)
        if sparse.indices:
            entry["text_sparse_top5000"] = both(coll(TEXT_HYBRID_INDEX_NAME), sparse, qb.SPARSE_VECTOR, 5000)
        if a.image_english_query.strip():
            iv = img_emb.embed_texts([a.image_english_query], l2_normalize=True)[0].tolist()
            entry["image_top60"] = both(coll(IMAGE_INDEX_NAME), iv, qb.DENSE_VECTOR, 60)
        if a.audio_english_query.strip():
            av = aud_emb.embed_texts([a.audio_english_query], l2_normalize=True)[0].tolist()
            entry["audio_top60_exact"] = both(coll(AUDIO_INDEX_NAME), av, qb.DENSE_VECTOR, 60)
            entry["audio_top60_server_approx"] = both(coll(AUDIO_INDEX_NAME), av, qb.DENSE_VECTOR, 60, exact_on_server=False)
        entry["context_dense_top100_server_approx(as_service)"] = both(ctx_coll, dense, qb.DENSE_VECTOR, 100, exact_on_server=False)
        entry["context_dense_top100_server_exact"] = both(ctx_coll, dense, qb.DENSE_VECTOR, 100, exact_on_server=True)
        report[qid] = entry
        for k, v in entry.items():
            print(f"{qid} {k:>44} identical={v['identical_order']} same_set={v['same_set']} first_diff={v['first_diff']} "
                  f"only_local/server={v['only_local']}/{v['only_server']} max|Δscore|={v['max_abs_score_diff_same_id']}")
    (HERE / "probe_backends.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    local.close(); server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
