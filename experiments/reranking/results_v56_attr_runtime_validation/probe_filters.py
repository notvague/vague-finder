"""§4-1 진단 — 라우터가 만드는 payload 필터(제목 구조 · 연주 단서 · 속성 조건)의 **적중 집합**을 로컬/서버에서 직접 대조한다. 모델 호출 없음."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    sys.path.insert(0, p)
import v56_runtime; v56_runtime.block_real_models()
os.environ.update({"ATTR_FILTER_WEIGHT": "0.5", "ATTR_FILTER_GATE": "era_plus", "ATTR_FILTER_K": "30", "ATTR_FILTER_USE_GENRE": "1", "ATTR_FILTER_GENRE_POLICY": "conditional"})
from qdrant_client import QdrantClient
from src.backend.schemas.query import QueryAnalysis
from src.retrieval import search_router as sr
from src.vector_db import qdrant_backend as qb
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME
local = QdrantClient(path="artifacts/qdrant_eval_v56_local"); server = QdrantClient(url="http://127.0.0.1:6356")
coll = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
an = v56_runtime.CachedAnalyzer(); by_id = {qid: a for _q, (qid, a, _f, _p) in an.by_query.items()}
def ids_for(client, flt):
    out, off = set(), None
    while True:
        pts, off = client.scroll(coll, scroll_filter=flt, limit=1000, offset=off, with_payload=False)
        out |= {str(p.id) for p in pts}
        if off is None: return out
builders = {}
for name in dir(sr):
    if name.startswith("_build_") and name.endswith("_filter"): builders[name] = getattr(sr, name)
print("builders:", sorted(builders))
report = {}
for qid in ["q101", "t049", "c601", "n010", "m103", "t001"]:
    a = QueryAnalysis(**by_id[qid]); report[qid] = {}
    for name, fn in sorted(builders.items()):
        try:
            flt = fn(a) if name != "_build_attribute_filter" else fn(a, None)
        except TypeError:
            try: flt = fn(a)
            except Exception as e: flt = f"ERR {e}"
        if not isinstance(flt, dict): continue
        q = qb.translate_filter(flt)
        li, si = ids_for(local, q), ids_for(server, q)
        report[qid][name] = {"filter": flt, "n_local": len(li), "n_server": len(si), "only_local": sorted(li - si)[:5], "only_server": sorted(si - li)[:5]}
        print(qid, name, "local", len(li), "server", len(si), "only_local", len(li - si), "only_server", len(si - li), json.dumps(flt, ensure_ascii=False)[:140])
(HERE / "probe_filters.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
# 불일치 점의 payload 타입 확인
for qid, d in report.items():
    for name, r in d.items():
        for sid in (r["only_local"] + r["only_server"])[:1]:
            pid = qb.point_id(sid)
            lp = local.retrieve(coll, ids=[pid], with_payload=True)[0].payload; sp = server.retrieve(coll, ids=[pid], with_payload=True)[0].payload
            keys = sorted({k for c in r["filter"].get("$and", [r["filter"]]) for k in (c.keys() if isinstance(c, dict) else [])} - {"$or", "$and"})
            print("  mismatch", qid, name, sid, {k: (repr(lp.get(k)), repr(sp.get(k))) for k in keys if k in lp or k in sp})
