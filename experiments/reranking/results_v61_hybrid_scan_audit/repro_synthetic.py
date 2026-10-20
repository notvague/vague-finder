"""v61 — 합성 자료 재현(메모리 Qdrant · 3차원 · 7곡 · 상한 3). 모델·DB·네트워크 없음.

`tests/vector_db/test_v61_hybrid_scan_audit.py`와 같은 자료를 써서, 상한이 자료 수보다 작을 때 구현 결과와 직접 내적 합산이 어떻게
어긋나는지 표로 찍는다. 실행: venv/bin/python experiments/reranking/results_v61_hybrid_scan_audit/repro_synthetic.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.vector_db import qdrant_backend as qb  # noqa: E402
from src.vector_db.qdrant_backend import QdrantVectorClient  # noqa: E402
from src.vector_db.settings import TEXT_HYBRID_INDEX_NAME  # noqa: E402
from tests.vector_db.test_v61_hybrid_scan_audit import _corpus, _exact, _fill  # noqa: E402


def main() -> int:
    songs = _corpus()
    client = QdrantVectorClient(path=":memory:")
    _fill(client, songs)
    dense_q, sparse_q = [0.5, 0.0, 0.0], {"indices": [7], "values": [0.5]}
    exact = _exact(songs, dense_q, sparse_q)
    out = {"corpus": len(songs), "dense_query": dense_q, "sparse_query": sparse_q, "exact": [], "by_limit": {}}
    print(f"{'id':>3} {'exact':>7} {'dense':>6} {'sparse':>6}")
    for song_id, total, d, s in exact:
        out["exact"].append({"id": song_id, "total": round(total, 6), "dense": round(d, 6), "sparse": round(s, 6)})
        print(f"{song_id:>3} {total:7.3f} {d:6.3f} {s:6.3f}")
    for limit in (3, 5, len(songs)):
        qb.HYBRID_SCAN_LIMIT = limit
        res = client.Index(TEXT_HYBRID_INDEX_NAME).query(vector=dense_q, sparse_vector=sparse_q, top_k=len(songs))
        got = {m["id"]: m["score"] for m in res["matches"]}
        rows = []
        print(f"\n--- HYBRID_SCAN_LIMIT={limit}")
        print(f"{'id':>3} {'impl':>7} {'exact':>7} {'delta':>7} 분류")
        for song_id, total, d, s in exact:
            if song_id not in got:
                label = "양쪽 목록 밖 → 결과에서 사라짐"
                impl = None
            elif abs(got[song_id] - total) > 1e-6:
                label = "dense 몫 누락" if abs(got[song_id] - s) < 1e-6 else ("sparse 몫 누락" if abs(got[song_id] - d) < 1e-6 else "?")
                impl = got[song_id]
            else:
                label = "정확"
                impl = got[song_id]
            rows.append({"id": song_id, "impl": impl, "exact": round(total, 6), "label": label})
            print(f"{song_id:>3} {('%7.3f' % impl) if impl is not None else '   없음':>7} {total:7.3f} {('%7.3f' % (impl - total)) if impl is not None else '':>7} {label}")
        out["by_limit"][str(limit)] = {"result_order": [m["id"] for m in res["matches"]], "rows": rows}
    qb.HYBRID_SCAN_LIMIT = 5000
    client.close()
    (HERE / "repro_synthetic.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
