"""v65 검증 환경 준비 (v64 prepare_server.py 이식) — 색인 파일 복사본 → 전용 Qdrant 서버로 옮기고 자료 정합성을 기록한다. 모델·Mongo 호출 없음.

v56 `migrate_to_server.py`를 이 실험용으로 옮긴 것이다(입력·출력 경로와 확인 항목만 다르다).

- 원본: `artifacts/qdrant_v65_src` — v63 워크트리의 색인 복사본(`qdrant_v63`, 내용 지문 `238299e6c8c17f2a`)을 다시 복사한 것.
  서비스 색인(`artifacts/qdrant`)·v63 복사본은 열지 않는다. 로컬 파일 모드 클라이언트는 이 스크립트 하나만 연다.
- 대상: 전용 컨테이너 `vague-finder-qdrant-v65`(127.0.0.1:6366). 비어 있을 때만 옮긴다(이미 있으면 확인만).
- 확인: 컬렉션·점 수·벡터 설정·payload 전량·벡터 표본 50점·곡 텍스트 컬렉션 내용 지문(v61 `ExactHybridReference`, 양쪽)·
  Context 별칭(원본과 같은 대상, 충돌 시 중단)·Context manifest의 컬렉션 이름·건수와 서버 일치·BM25 파일 sha.
- payload 색인: Context 두 컬렉션에만 서비스 적재기(`context_qdrant._create_payload_indexes`)가 서버 모드에서 만드는 것과 같은 필드.
  곡 컬렉션에는 만들지 않는다(적재기 `qdrant_load`도 만들지 않는다).

실행(워크트리 루트): venv/bin/python experiments/reranking/results_v65_deterministic_ties/prepare_server.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/reranking/results_v61_hybrid_scan_audit"))

from qdrant_client import QdrantClient, models  # noqa: E402
from qdrant_client.migrate import migrate  # noqa: E402
from exact_hybrid import ExactHybridReference  # noqa: E402

SRC_PATH = os.getenv("V65_SRC_PATH", "artifacts/qdrant_v65_src")
URL = os.getenv("V65_QDRANT_URL", "http://127.0.0.1:6366")
OUT = HERE / "server_collections.json"
TEXT = "vaguefinder-text-1024-koe5__dev"
EXPECTED_TEXT_FP = "238299e6c8c17f2a"
CONTEXT_MANIFEST = ROOT / "artifacts/vector_db/context_qdrant/dev/manifest.json"
BM25 = ROOT / "artifacts/bm25_params.json"
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


def _scroll_all(client: QdrantClient, name: str) -> dict:
    offset, out = None, {}
    while True:
        points, offset = client.scroll(collection_name=name, limit=512, offset=offset, with_payload=True, with_vectors=False)
        for p in points:
            out[str(p.id)] = p
        if offset is None:
            return out


def _vec_equal(a, b, tol=1e-6) -> bool:
    if isinstance(a, dict) != isinstance(b, dict):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_vec_equal(a[k], b[k], tol) for k in a)
    if hasattr(a, "indices"):
        return (list(a.indices) == list(b.indices) and len(a.values) == len(b.values)
                and all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a.values, b.values)))
    return a is not None and b is not None and len(a) == len(b) and all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))


def restore_aliases(src, dst, collection_names) -> dict:
    """원본의 별칭을 서버에 그대로 만든다. 다른 대상을 가리키는 별칭이 있으면 덮어쓰지 않고 중단한다."""
    source = {a.alias_name: a.collection_name for a in src.get_aliases().aliases}
    before = {a.alias_name: a.collection_name for a in dst.get_aliases().aliases}
    for base in CONTEXT_INDEX_FIELDS:
        if not any(alias.startswith(base + "__") for alias in source):
            raise RuntimeError(f"원본 Context 별칭 없음: {base}")
    conflicts = {a: before[a] for a, t in source.items() if a in before and before[a] != t}
    missing_targets = sorted(set(source.values()) - set(collection_names))
    if conflicts or missing_targets:
        raise RuntimeError(f"별칭 복원 중단: conflicts={conflicts}, missing_targets={missing_targets}")
    actions = [models.CreateAliasOperation(create_alias=models.CreateAlias(alias_name=a, collection_name=t))
               for a, t in source.items() if a not in before]
    if actions:
        dst.update_collection_aliases(change_aliases_operations=actions)
    after = {a.alias_name: a.collection_name for a in dst.get_aliases().aliases}
    return {"aliases_source": source, "aliases_server_before": before, "aliases_server": after,
            "aliases_ok": all(after.get(a) == t for a, t in source.items())}


def main() -> int:
    src = QdrantClient(path=SRC_PATH)
    dst = QdrantClient(url=URL)
    report = {"source_path": SRC_PATH, "server_url": URL, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "collections": {}}
    try:
        report["server_version"] = dst.info().version
    except Exception as exc:  # noqa: BLE001
        report["server_version"] = f"? ({type(exc).__name__})"
    names = sorted(c.name for c in src.get_collections().collections)
    existing = {c.name for c in dst.get_collections().collections}
    todo = [n for n in names if n not in existing]
    report["migrated"] = todo
    if todo:
        t0 = time.perf_counter()
        migrate(src, dst, collection_names=todo, recreate_on_collision=False, batch_size=256)
        report["migrate_seconds"] = round(time.perf_counter() - t0, 1)
        print(f"migrated {len(todo)} collections in {report['migrate_seconds']}s", flush=True)

    rng = random.Random(65)
    ok_all = True
    for name in names:
        s_info, d_info = src.get_collection(name), dst.get_collection(name)
        s_cnt, d_cnt = src.count(name, exact=True).count, dst.count(name, exact=True).count
        s_cfg, d_cfg = _vec_config(s_info), _vec_config(d_info)
        wanted = CONTEXT_INDEX_FIELDS.get(name.split("__")[0], ())
        schema = d_info.payload_schema or {}
        for field in wanted:
            if field not in schema:
                dst.create_payload_index(collection_name=name, field_name=field,
                                         field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
        schema = dst.get_collection(name).payload_schema or {}
        s_pts, d_pts = _scroll_all(src, name), _scroll_all(dst, name)
        payload_mismatch = [pid for pid in s_pts if pid not in d_pts or s_pts[pid].payload != d_pts[pid].payload]
        extra = [pid for pid in d_pts if pid not in s_pts]
        sample = rng.sample(sorted(s_pts), min(50, len(s_pts)))
        ids = [s_pts[i].id for i in sample]
        s_vec = {str(p.id): p.vector for p in src.retrieve(name, ids=ids, with_vectors=True, with_payload=False)}
        d_vec = {str(p.id): p.vector for p in dst.retrieve(name, ids=ids, with_vectors=True, with_payload=False)}
        vec_mismatch = [pid for pid in sample if not _vec_equal(s_vec.get(pid), d_vec.get(pid))]
        ok = s_cnt == d_cnt and s_cfg == d_cfg and not payload_mismatch and not extra and not vec_mismatch
        ok_all &= ok
        report["collections"][name] = {
            "points_source": s_cnt, "points_server": d_cnt, "vectors_source": s_cfg, "vectors_server": d_cfg,
            "payload_indexes_server": sorted(schema), "payload_mismatch": len(payload_mismatch),
            "server_extra_points": len(extra), "vector_sample": len(sample), "vector_sample_mismatch": len(vec_mismatch), "ok": ok,
        }
        print(f"{name}: src={s_cnt} dst={d_cnt} cfg={s_cfg == d_cfg} payload_mismatch={len(payload_mismatch)} "
              f"vec_mismatch={len(vec_mismatch)}/{len(sample)} idx={sorted(schema)} ok={ok}", flush=True)

    # 곡 텍스트 컬렉션 전체 내용 지문 — 양쪽에서 같은 식으로(v61·v62·v63 복사본과 같은 값이어야 한다)
    fp_src = ExactHybridReference.from_client(src, TEXT).fingerprint()
    fp_dst = ExactHybridReference.from_client(dst, TEXT).fingerprint()
    report["text_fingerprint"] = {"source": fp_src, "server": fp_dst, "expected": EXPECTED_TEXT_FP,
                                  "ok": fp_src["content_sha256_16"] == fp_dst["content_sha256_16"] == EXPECTED_TEXT_FP}
    ok_all &= report["text_fingerprint"]["ok"]

    report.update(restore_aliases(src, dst, names))
    ok_all &= report["aliases_ok"]

    manifest = json.loads(CONTEXT_MANIFEST.read_text(encoding="utf-8"))
    q = manifest["qdrant"]
    aliases = report["aliases_server"]
    ctx = {
        "manifest_status": manifest.get("status"), "source_scope_complete": manifest.get("source_scope_complete"),
        "context_build_id_16": str(manifest.get("context_build_id"))[:16],
        "manifest_content_sha256_16": str(manifest.get("manifest_content_sha256"))[:16],
        "dense_alias_target_ok": aliases.get(q["dense_alias"]) == q["dense_collection"],
        "sparse_alias_target_ok": aliases.get(q["sparse_alias"]) == q["sparse_collection"],
        "dense_count_ok": dst.count(q["dense_collection"], exact=True).count == manifest["dense_record_count"],
        "sparse_count_ok": dst.count(q["sparse_collection"], exact=True).count == manifest["sparse_profile_count"],
    }
    ctx["ok"] = all(v for k, v in ctx.items() if k.endswith("_ok"))
    report["context_manifest"] = ctx
    ok_all &= ctx["ok"]
    report["bm25_sha256_12"] = hashlib.sha256(BM25.read_bytes()).hexdigest()[:12]
    report["bm25_ok"] = report["bm25_sha256_12"] == "bf606b19ab02"
    ok_all &= report["bm25_ok"]
    report["all_ok"] = bool(ok_all)
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    src.close()
    dst.close()
    print("ALL_OK" if ok_all else "MISMATCH", "->", OUT)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
