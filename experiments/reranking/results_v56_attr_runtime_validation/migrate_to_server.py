"""로컬(파일) Qdrant 색인 복사본 → 전용 서버 Qdrant로 옮기고, 자료가 같은지 확인한다. 모델 호출 없음.

원본 서비스 색인(`artifacts/qdrant`)은 열지 않는다 — 파일 복사본(`artifacts/qdrant_eval_v56_local`, 바이트 동일 확인)을 읽는다.
서버는 이 검증 전용 컨테이너(`vague-finder-qdrant-v56`, 127.0.0.1:6356)다. 운영 서버·컬렉션은 없다(로컬 모드 운영).

확인 항목(§4): 컬렉션 이름·점 수·벡터 설정(크기·거리)·sparse 설정·payload 색인·payload 전량 일치·벡터 표본 일치.
Context 컬렉션의 payload 색인은 서비스 적재기(`src/vector_db/context_qdrant._create_payload_indexes`)가 서버 모드에서
만드는 것과 같은 필드에 만든다 — 로컬 모드에는 없는 차이라 기록한다.

실행: venv/bin/python experiments/reranking/results_v56_attr_runtime_validation/migrate_to_server.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient, models
from qdrant_client.migrate import migrate

SRC_PATH = os.getenv("V56_SRC_PATH", "artifacts/qdrant_eval_v56_local")
URL = os.getenv("V56_QDRANT_URL", "http://127.0.0.1:6356")
OUT = Path(__file__).resolve().parent / "server_collections.json"

# 서비스 적재기가 서버 모드에서 만드는 payload 색인(context_qdrant._create_payload_indexes). 곡 컬렉션 3개에는 없다.
CONTEXT_INDEX_FIELDS = {
    "vaguefinder-context-dense-1024-koe5": ("song_id", "category", "context_build_id"),
    "vaguefinder-context-sparse-bm25": ("song_id", "context_build_id"),
}


def _vec_config(info) -> dict:
    cfg = info.config.params
    dense = {}
    vectors = cfg.vectors
    if isinstance(vectors, dict):
        for name, params in vectors.items():
            dense[name] = {"size": params.size, "distance": str(params.distance.value)}
    elif vectors is not None:
        dense["__default__"] = {"size": vectors.size, "distance": str(vectors.distance.value)}
    sparse = sorted((cfg.sparse_vectors or {}).keys()) if cfg.sparse_vectors else []
    return {"dense": dense, "sparse": sparse}


def _scroll_all(client: QdrantClient, name: str, with_vectors: bool = False):
    offset = None
    out = {}
    while True:
        points, offset = client.scroll(
            collection_name=name, limit=512, offset=offset,
            with_payload=True, with_vectors=with_vectors,
        )
        for p in points:
            out[str(p.id)] = p
        if offset is None:
            return out


def _vec_equal(a, b, tol=1e-6) -> bool:
    if isinstance(a, dict) != isinstance(b, dict):
        return False
    if isinstance(a, dict):
        if set(a) != set(b):
            return False
        return all(_vec_equal(a[k], b[k], tol) for k in a)
    if hasattr(a, "indices"):  # SparseVector
        return list(a.indices) == list(b.indices) and all(
            math.isclose(x, y, abs_tol=tol) for x, y in zip(a.values, b.values)
        ) and len(a.values) == len(b.values)
    return len(a) == len(b) and all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))


def restore_aliases(src, dst, collection_names: list[str]) -> dict:
    """전용 검증 서버에 누락된 별칭만 복원하고 검증한다. 다른 대상을 가리키는 별칭은 덮어쓰지 않는다."""
    source = {a.alias_name: a.collection_name for a in src.get_aliases().aliases}
    before = {a.alias_name: a.collection_name for a in dst.get_aliases().aliases}
    # 컬렉션만 있는 불완전한 복사본을 정상으로 판정하지 않는다.
    for base in CONTEXT_INDEX_FIELDS:
        if not any(alias.startswith(base + "__") for alias in source):
            raise RuntimeError(f"원본 Context 별칭 없음: {base}")
    conflicts = {a: before[a] for a, target in source.items() if a in before and before[a] != target}
    missing_targets = sorted(set(source.values()) - set(collection_names))
    if conflicts or missing_targets:
        raise RuntimeError(f"별칭 복원 중단: conflicts={conflicts}, missing_targets={missing_targets}")
    actions = [models.CreateAliasOperation(create_alias=models.CreateAlias(alias_name=a, collection_name=target))
               for a, target in source.items() if a not in before]
    if actions:
        dst.update_collection_aliases(change_aliases_operations=actions)
    after = {a.alias_name: a.collection_name for a in dst.get_aliases().aliases}
    ok = all(after.get(a) == target for a, target in source.items())
    if not ok:
        raise RuntimeError("서버 별칭 복원 후 검증 실패")
    return {"aliases_source": source, "aliases_server": after, "aliases_ok": ok}


def main() -> int:
    src = QdrantClient(path=SRC_PATH)
    dst = QdrantClient(url=URL)
    server_version = dst.info().version if hasattr(dst, "info") else "?"
    names = sorted(c.name for c in src.get_collections().collections)
    existing = {c.name for c in dst.get_collections().collections}
    report = {
        "source_path": SRC_PATH, "server_url": URL, "server_version": server_version,
        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "collections": {},
    }
    todo = [n for n in names if n not in existing]
    if todo:
        t0 = time.perf_counter()
        migrate(src, dst, collection_names=todo, recreate_on_collision=False, batch_size=256)
        report["migrate_seconds"] = round(time.perf_counter() - t0, 1)
        print(f"migrated {len(todo)} collections in {report['migrate_seconds']}s")
    else:
        print("all collections already on server — verify only")

    rng = random.Random(56)
    ok_all = True
    for name in names:
        s_info, d_info = src.get_collection(name), dst.get_collection(name)
        s_cnt, d_cnt = src.count(name, exact=True).count, dst.count(name, exact=True).count
        s_cfg, d_cfg = _vec_config(s_info), _vec_config(d_info)
        # payload 색인(서버 모드 적재기와 같게)
        base = name.split("__")[0]
        wanted = CONTEXT_INDEX_FIELDS.get(base, ())
        schema = d_info.payload_schema or {}
        for field in wanted:
            if field not in schema:
                dst.create_payload_index(collection_name=name, field_name=field,
                                         field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
        schema = dst.get_collection(name).payload_schema or {}
        payload_idx = {k: str(v.data_type.value) if v.data_type else "?" for k, v in schema.items()}
        # payload 전량 비교
        s_pts, d_pts = _scroll_all(src, name), _scroll_all(dst, name)
        payload_mismatch = [pid for pid in s_pts if pid not in d_pts or s_pts[pid].payload != d_pts[pid].payload]
        extra = [pid for pid in d_pts if pid not in s_pts]
        # 벡터 표본 비교(50점)
        sample = rng.sample(sorted(s_pts), min(50, len(s_pts)))
        s_vec = {str(p.id): p.vector for p in src.retrieve(name, ids=[s_pts[i].id for i in sample], with_vectors=True, with_payload=False)}
        d_vec = {str(p.id): p.vector for p in dst.retrieve(name, ids=[s_pts[i].id for i in sample], with_vectors=True, with_payload=False)}
        vec_mismatch = [pid for pid in sample if not _vec_equal(s_vec.get(pid), d_vec.get(pid))]
        ok = (s_cnt == d_cnt and s_cfg == d_cfg and not payload_mismatch and not extra and not vec_mismatch)
        ok_all &= ok
        report["collections"][name] = {
            "points_source": s_cnt, "points_server": d_cnt, "vectors_source": s_cfg, "vectors_server": d_cfg,
            "payload_indexes_server": payload_idx, "payload_indexes_source": sorted((s_info.payload_schema or {}).keys()),
            "payload_mismatch": len(payload_mismatch), "server_extra_points": len(extra),
            "vector_sample": len(sample), "vector_sample_mismatch": len(vec_mismatch), "ok": ok,
        }
        print(f"{name}: src={s_cnt} dst={d_cnt} cfg_equal={s_cfg == d_cfg} payload_mismatch={len(payload_mismatch)} "
              f"vec_mismatch={len(vec_mismatch)}/{len(sample)} idx={sorted(payload_idx)} ok={ok}")
    report.update(restore_aliases(src, dst, names))
    report["all_ok"] = ok_all and report["aliases_ok"]
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    src.close(); dst.close()
    print("ALL_OK" if ok_all else "MISMATCH", "->", OUT)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
