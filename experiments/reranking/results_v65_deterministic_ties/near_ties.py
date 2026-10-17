"""v65 §5 남은 차이 진단 — 수정 후에도 로컬·서버 후보 순서가 갈린 질의의 이미지 경로 점수를 **두 백엔드에서 그대로** 읽는다.

같은 질의 벡터(SigLIP2, 저장 분석 캐시의 `image_english_query`)로 로컬 파일 복사본과 전용 서버를 직접 조회해 상위 점수의
원값(백엔드가 돌려준 float)과 float64 독립 기준을 나란히 적는다. 모델·Mongo·외부 호출 없음(임베딩은 로컬 가중치, 오프라인).

    venv/bin/python near_ties.py n071,q205
출력: near_ties.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})

import numpy as np  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from src.retrieval.analysis_cache import load_cache  # noqa: E402

COLL = "vaguefinder-image-768__dev"


def main() -> int:
    wanted = sys.argv[1].split(",")
    analyses = {}
    for s in ("v06", "v09", "v10"):
        cache = load_cache(ROOT / f"experiments/reranking/results_v49_vertex_baseline/analysis_cache_{s}_dev_vertex.json")
        for qid, e in cache.entries.items():
            if qid in wanted:
                analyses[qid] = e["analysis"]
    from src.backend.api.dependencies import get_image_embedder  # noqa: WPS433

    emb = get_image_embedder()
    local = QdrantClient(path="artifacts/qdrant_v65_src")
    server = QdrantClient(url="http://127.0.0.1:6366")
    stored = {}
    offset = None
    while True:
        pts, offset = local.scroll(COLL, limit=1024, offset=offset, with_vectors=True, with_payload=["song_id"])
        for p in pts:
            stored[p.payload["song_id"]] = np.asarray(p.vector["dense"], dtype=np.float64)
        if offset is None:
            break
    out = {}
    for qid in wanted:
        q = emb.embed_texts([analyses[qid]["image_english_query"]], l2_normalize=True)[0].tolist()
        rows = {}
        for name, client in (("local", local), ("server", server)):
            res = client.query_points(COLL, query=q, using="dense", limit=12, with_payload=["song_id"],
                                      **({"search_params": None} if name == "local" else {})).points
            rows[name] = [(p.payload["song_id"], repr(float(p.score))) for p in res]
        qq = np.asarray(q, dtype=np.float64)
        qq /= np.linalg.norm(qq)
        ids = [sid for sid, _ in rows["local"]]
        out[qid] = {
            "query": analyses[qid]["image_english_query"],
            "local": rows["local"], "server": rows["server"],
            "float64_reference": [(sid, repr(float(stored[sid] @ qq))) for sid in ids],
        }
        print(qid, json.dumps(out[qid], ensure_ascii=False)[:1200])
    local.close()
    server.close()
    (HERE / "near_ties.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
