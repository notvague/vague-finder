"""가사 스냅샷 ↔ 색인 복사본 곡 ID 대조(모델·DB 호출 없음, 로컬 Qdrant 복사본만 연다 — 다른 실행과 동시에 돌리지 않는다).

산출: snapshot_coverage.json (곡 수·교집합·한쪽에만 있는 ID·빈 가사 ID·manifest 요약 — 가사 원문 없음)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v63_runtime as rt  # noqa: E402

os.chdir(rt.ROOT)
rt.fixed_env("offline")
rt.install_guards("offline")
from qdrant_client import QdrantClient  # noqa: E402
from src.vector_db import qdrant_backend as qb  # noqa: E402
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME  # noqa: E402

manifest = rt.snapshot_manifest()
docs = rt.load_snapshot_documents()
snap_ids = {str(d.get("song_id") or "") for d in docs}
nonempty = {str(d.get("song_id") or "") for d in docs if str((d.get("lyrics_data") or {}).get("full_lyrics") or "").strip()}
client = QdrantClient(path=str(rt.ROOT / rt.QDRANT_COPY))
collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
index_ids = set()
offset = None
while True:
    points, offset = client.scroll(collection, limit=1024, offset=offset, with_payload=["song_id"], with_vectors=False)
    index_ids.update(str((p.payload or {}).get("song_id")) for p in points)
    if offset is None:
        break
client.close()
out = {
    "snapshot_documents": len(docs), "snapshot_distinct_ids": len(snap_ids), "snapshot_nonempty_lyrics": len(nonempty),
    "index_points_song_ids": len(index_ids), "intersection": len(snap_ids & index_ids),
    "only_in_snapshot": sorted(snap_ids - index_ids), "only_in_index": sorted(index_ids - snap_ids),
    "index_ids_without_lyrics_in_snapshot": sorted(index_ids - nonempty),
    "manifest": {k: manifest[k] for k in ("started_at", "full_read_started_at", "full_read_finished_at", "finished_at",
                                          "full_read_seconds", "count_documents_before", "count_documents_after",
                                          "light_fingerprint_before", "light_fingerprint_after", "content_from_full_read",
                                          "full_document_sha256_16", "distinct_song_ids", "duplicate_song_ids",
                                          "missing_song_id_docs", "empty_full_lyrics_song_ids",
                                          "consistent_before_full_after", "file", "read", "db_name", "collection")},
}
(HERE / "snapshot_coverage.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in out.items() if k != "manifest"}, ensure_ascii=False))
