"""§4-1 진단 — 서비스 코드의 `QdrantIndex.query`(dense+sparse 전수 합산)를 같은 벡터로 로컬/서버에 돌려 상위 100의 순서·점수 차이를 본다. 모델 호출 없음."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)): sys.path.insert(0, p)
import v56_runtime; v56_runtime.block_real_models()
os.environ.pop("QDRANT_URL", None); os.environ["QDRANT_PATH"] = "artifacts/qdrant_eval_v56_local"
from qdrant_client import QdrantClient
import src.backend.api.dependencies as deps
from src.backend.schemas.query import QueryAnalysis
from src.retrieval.search_service import hybrid_score
from src.vector_db import qdrant_backend as qb
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME
an = v56_runtime.CachedAnalyzer(); by_id = {qid: a for _q, (qid, a, _f, _p) in an.by_query.items()}
emb, bm25 = deps.get_text_embedder(), deps.get_bm25_encoder()
local = qb.QdrantIndex(QdrantClient(path="artifacts/qdrant_eval_v56_local"), TEXT_HYBRID_INDEX_NAME, exact=False)
server = qb.QdrantIndex(QdrantClient(url="http://127.0.0.1:6356"), TEXT_HYBRID_INDEX_NAME, exact=True)
report = {}
for qid in ["q101", "t049", "c601", "n010", "m103", "t001"]:
    a = QueryAnalysis(**by_id[qid])
    dense = emb.embed_passages([f"query: {a.search_text}"], add_e5_prefix=False, normalize=True)[0].tolist()
    enc = bm25.encode_queries(" ".join(a.korean_tags) or a.search_text); enc = enc[0] if isinstance(enc, list) else enc
    hd, hs = hybrid_score(dense, enc, a.text_alpha)
    lo = local.query(vector=hd, sparse_vector=hs, top_k=100, include_metadata=False, namespace=NAMESPACE)["matches"]
    se = server.query(vector=hd, sparse_vector=hs, top_k=100, include_metadata=False, namespace=NAMESPACE)["matches"]
    li, si = [m["id"] for m in lo], [m["id"] for m in se]
    ls, ss = {m["id"]: m["score"] for m in lo}, {m["id"]: m["score"] for m in se}
    first = next((i for i, (x, y) in enumerate(zip(li, si)) if x != y), None)
    swaps = [(i, li[i], si[i], round(ls.get(li[i], 0) - ls.get(si[i], 0), 9)) for i in range(min(len(li), len(si))) if li[i] != si[i]][:8]
    maxd = max((abs(ls[k] - ss[k]) for k in ls if k in ss), default=None)
    report[qid] = {"identical": li == si, "same_set": set(li) == set(si), "first_diff": first, "n_positions_differ": sum(1 for x, y in zip(li, si) if x != y),
                   "only_local": sorted(set(li) - set(si)), "only_server": sorted(set(si) - set(li)), "max_abs_score_diff_same_id": maxd,
                   "swaps_(pos, local_id, server_id, local_score_gap)": swaps}
    print(qid, {k: v for k, v in report[qid].items() if k != "swaps_(pos, local_id, server_id, local_score_gap)"}); print("   swaps:", swaps[:4])
(HERE / "probe_hybrid.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
