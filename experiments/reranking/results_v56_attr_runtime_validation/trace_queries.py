"""§4-1 진단 — 라우터 검색 중 벡터 DB에 나가는 **모든 조회**(QdrantIndex.query/fetch)를 기록해 로컬/서버 모드를 호출 단위로 대조한다. 모델 호출 없음."""
from __future__ import annotations
import asyncio, json, os, sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)): sys.path.insert(0, p)
import v56_runtime; v56_runtime.block_real_models()
mode = sys.argv[1]
if mode == "server": os.environ["QDRANT_URL"] = "http://127.0.0.1:6356"
else: os.environ.pop("QDRANT_URL", None); os.environ["QDRANT_PATH"] = "artifacts/qdrant_eval_v56_local"
os.environ.update({"ATTR_FILTER_WEIGHT": "0", "FEEDBACK_COLLECTION_ENABLED": "0", "SEARCH_REFERENCE_YEAR": "2026", "TOKENIZERS_PARALLELISM": "false"})
from src.vector_db import qdrant_backend as qb
LOG = []
_orig_query, _orig_fetch = qb.QdrantIndex.query, qb.QdrantIndex.fetch
def q(self, **kw):
    res = _orig_query(self, **kw)
    ids = [m["id"] for m in res["matches"]]
    vec = kw.get("vector") or []; sv = kw.get("sparse_vector")
    LOG.append({"op": "query", "index": self._index_name, "top_k": kw.get("top_k"), "filter": json.dumps(kw.get("filter"), ensure_ascii=False, sort_keys=True),
                "hybrid": sv is not None, "vec_sha": hashlib.sha1(json.dumps([round(x, 5) for x in vec]).encode()).hexdigest()[:8],
                "sparse_n": len((sv or {}).get("indices", [])), "n": len(ids), "ids": ids})
    return res
def f(self, ids, namespace=None, fields=None):
    res = _orig_fetch(self, ids, namespace, fields) if namespace is not None else _orig_fetch(self, ids, fields=fields)
    LOG.append({"op": "fetch", "index": self._index_name, "n_in": len(list(ids)), "n": len(res["matches"]), "ids": [m["id"] for m in res["matches"]]})
    return res
qb.QdrantIndex.query, qb.QdrantIndex.fetch = q, f
import src.backend.api.dependencies as deps
deps.get_reranker = deps.singleton(v56_runtime.make_fake_reranker)
from src.backend.schemas.query import QueryAnalysis
an = v56_runtime.CachedAnalyzer(); by_id = {qid: a for _q, (qid, a, _f, _p) in an.by_query.items()}
router = deps.get_search_router()
out = {}
for qid in ["q101", "t049", "c601", "n010"]:
    LOG.clear(); cands = []
    asyncio.run(router.search(QueryAnalysis(**by_id[qid]), top_k=15, use_rerank=False, candidate_k=30, candidate_ids_out=cands))
    # 병렬 호출 순서가 다를 수 있어 (index, filter, top_k, hybrid) 로 정렬
    out[qid] = {"candidates": cands, "calls": sorted(LOG, key=lambda c: (c["op"], c["index"], c.get("filter", ""), str(c.get("top_k")), str(c.get("hybrid")), c.get("vec_sha", "")))}
    print(mode, qid, "calls", len(LOG))
router.shutdown()
(HERE / f"trace_{mode}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
