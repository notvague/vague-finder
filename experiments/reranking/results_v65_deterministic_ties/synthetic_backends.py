"""v65 §4-② 합성 백엔드 — 같은 작은 자료를 로컬 파일 모드·서버 모드에 **정순/역순/섞어서** 넣고, 재적재·서버 재시작 뒤에도
같은 질의가 같은 결과를 내는지 본다. 독립 기준(저장 벡터로 직접 계산 + 계약 키)과도 대조한다. 모델·Mongo 호출 없음.

    V65_CODE=before|after venv/bin/python synthetic_backends.py <phase>
      phase=load     : 서버에 정순(fwd)·역순(rev) 컬렉션 적재 + 로컬 파일 모드 정순/역순 결과 기록 + 서버 결과 기록
      phase=reload   : 서버 rev 컬렉션을 지우고 다른 순서(seed 셔플)로 다시 적재한 뒤 결과 기록
      phase=query    : 적재 없이 결과만 기록(서버 재시작 뒤에 부른다 — 재시작은 셸 스크립트가 docker restart로)
결과: synthetic/<code>_<tag>.json(tag = 둘째 인자, 없으면 phase) — 질의별 id 목록. `synthetic_compare.py`가 대조한다.

서버 컬렉션은 전용 컨테이너(127.0.0.1:6366)에 네임스페이스 `v65syn_*`로만 만든다(실험 색인 dev 컬렉션은 건드리지 않는다).
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = os.environ.get("V65_CODE", "")
if CODE not in {"before", "after"}:
    raise SystemExit("V65_CODE=before|after")
CODE_ROOT = ROOT / "artifacts/src_base_2a6481d" if CODE == "before" else ROOT
for p in reversed([str(CODE_ROOT), str(HERE), str(ROOT)]):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
for key in ("QDRANT_URL", "QDRANT_PATH", "QDRANT_HYBRID_SCAN_LIMIT"):
    os.environ.pop(key, None)

import numpy as np  # noqa: E402
from qdrant_client import models  # noqa: E402

import src.vector_db.qdrant_backend as qb  # noqa: E402
from src.vector_db.qdrant_backend import DENSE_VECTOR, SPARSE_VECTOR, QdrantVectorClient, point_id  # noqa: E402
from src.vector_db.settings import IMAGE_INDEX_NAME, TEXT_HYBRID_INDEX_NAME  # noqa: E402
from qdrant_reference import song_key  # noqa: E402

SERVER = "http://127.0.0.1:6366"
OUT = HERE / "synthetic"
TOPKS = (1, 3, 5, 10, 30)


def corpus(seed: int = 65):
    """곡 120곡: 이미지 벡터는 크기 1~12의 동일 벡터 묶음, 텍스트는 dense·sparse 일부 동일(합산 동점). id 길이 섞음."""
    rng = np.random.default_rng(seed)
    ids, image, text_dense, text_sparse = [], [], [], []
    sizes = [1, 1, 2, 3, 12, 1, 5, 7, 1, 2, 9, 4, 1, 11, 6, 3, 1, 8, 10, 1, 2, 3, 1, 1, 6, 1, 4, 2, 1, 2]
    n = 0
    for g, size in enumerate(sizes):
        vec = rng.normal(size=8)
        vec /= np.linalg.norm(vec)
        tvec = rng.normal(size=6) * 0.3
        sp = {int(i): float(round(v, 3)) for i, v in zip(rng.choice(20, 3, replace=False), rng.uniform(0.1, 2, 3))}
        for _ in range(size):
            n += 1
            sid = str(int(rng.integers(1000, 10 ** int(rng.integers(4, 10)))))
            while sid in ids:
                sid = str(int(sid) + 1)
            ids.append(sid)
            image.append(vec.tolist())
            # 텍스트는 묶음 앞 절반만 같게(합산 동점 일부)
            text_dense.append((tvec if len(ids) % 2 else rng.normal(size=6) * 0.3).tolist())
            text_sparse.append(sp)
    queries = [(rng.normal(size=8)).tolist() for _ in range(12)]
    # 묶음 벡터 자체를 질의로도 써서 동점 묶음이 맨 위·경계에 오게 한다
    queries += [image[i] for i in (0, 10, 40, 70)]
    tqueries = [(rng.normal(size=6)).tolist() for _ in range(8)]
    tsparse = [{"indices": [int(i) for i in rng.choice(20, 2, replace=False)], "values": [1.0, 0.5]} for _ in range(8)]
    return ids, image, text_dense, text_sparse, queries, list(zip(tqueries, tsparse))


def load(client: QdrantVectorClient, ns: str, order) -> None:
    ids, image, td, ts, _, _ = corpus()
    img = client.ensure_collection(IMAGE_INDEX_NAME, dim=8, namespace=ns, recreate=True)
    txt = client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=6, namespace=ns, with_sparse=True, recreate=True)
    for start in range(0, len(order), 17):  # 작은 배치 여러 번 — 세그먼트·삽입 순서가 결과에 영향을 줄 여지를 둔다
        rows = order[start:start + 17]
        client.client.upsert(collection_name=img, wait=True, points=[
            models.PointStruct(id=point_id(ids[i]), vector={DENSE_VECTOR: image[i]}, payload={"song_id": ids[i]}) for i in rows])
        client.client.upsert(collection_name=txt, wait=True, points=[
            models.PointStruct(id=point_id(ids[i]), payload={"song_id": ids[i]},
                               vector={DENSE_VECTOR: td[i], SPARSE_VECTOR: models.SparseVector(
                                   indices=list(ts[i]), values=list(ts[i].values()))}) for i in rows])


def run_queries(client: QdrantVectorClient, ns: str) -> dict:
    _, _, _, _, queries, tq = corpus()
    img, txt = client.Index(IMAGE_INDEX_NAME), client.Index(TEXT_HYBRID_INDEX_NAME)
    out = {}
    for qi, q in enumerate(queries):
        for k in TOPKS:
            out[f"image:{qi}:{k}"] = [m["id"] for m in img.query(vector=q, top_k=k, namespace=ns)["matches"]]
    for qi, (d, s) in enumerate(tq):
        for k in TOPKS:
            out[f"text:{qi}:{k}"] = [m["id"] for m in txt.query(vector=d, sparse_vector=s, top_k=k, namespace=ns)["matches"]]
    return out


def reference() -> dict:
    ids, image, td, ts, queries, tq = corpus()
    m = np.asarray(image, dtype=np.float64)
    out = {}
    for qi, q in enumerate(queries):
        qq = np.asarray(q, dtype=np.float64)
        s = m @ (qq / np.linalg.norm(qq))
        order = sorted(range(len(ids)), key=lambda i: (-float(s[i]), song_key(ids[i])))
        for k in TOPKS:
            out[f"image:{qi}:{k}"] = [ids[i] for i in order[:k]]
    tdm = np.asarray(td, dtype=np.float64)
    for qi, (d, sq) in enumerate(tq):
        dense = tdm @ np.asarray(d, dtype=np.float64)
        terms = dict(zip(sq["indices"], sq["values"]))
        sparse = np.asarray([sum(v * terms.get(i, 0.0) for i, v in row.items()) for row in ts])
        tot = dense + sparse
        order = sorted(range(len(ids)), key=lambda i: (-float(tot[i]), song_key(ids[i])))
        for k in TOPKS:
            out[f"text:{qi}:{k}"] = [ids[i] for i in order[:k]]
    return out


def main() -> int:
    phase = sys.argv[1]
    OUT.mkdir(exist_ok=True)
    ids = corpus()[0]
    fwd = list(range(len(ids)))
    rev = list(reversed(fwd))
    shuffled = list(fwd)
    random.Random(7).shuffle(shuffled)
    result: dict = {"code": CODE, "phase": phase, "backend_file": qb.__file__, "runs": {}}
    server = QdrantVectorClient(url=SERVER)
    try:
        if phase == "load":
            for ns, order in (("v65syn_fwd", fwd), ("v65syn_rev", rev)):
                load(server, ns, order)
            with tempfile.TemporaryDirectory() as tmp:
                for name, order in (("local_fwd", fwd), ("local_rev", rev)):
                    c = QdrantVectorClient(path=str(Path(tmp) / name))
                    try:
                        load(c, "v65syn", order)
                        result["runs"][name] = run_queries(c, "v65syn")
                    finally:
                        c.close()
                    # 같은 파일 저장소를 다시 열어(재시작) 다시 조회
                    c = QdrantVectorClient(path=str(Path(tmp) / name))
                    try:
                        result["runs"][f"{name}_reopen"] = run_queries(c, "v65syn")
                    finally:
                        c.close()
            result["reference"] = reference()
        elif phase == "reload":
            load(server, "v65syn_rev", shuffled)
        result["runs"]["server_fwd"] = run_queries(server, "v65syn_fwd")
        result["runs"]["server_rev"] = run_queries(server, "v65syn_rev")
    finally:
        server.close()
    tag = sys.argv[2] if len(sys.argv) > 2 else phase
    (OUT / f"{CODE}_{tag}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print("ok", CODE, phase, {k: len(v) for k, v in result["runs"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
