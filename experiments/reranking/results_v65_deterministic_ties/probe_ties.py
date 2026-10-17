"""v65 §2 사전 진단 — 색인 복사본의 **저장 벡터 동일 묶음**과 곡 id 형식을 센다(모델·Mongo·서버 호출 없음).

동점은 질의마다 다르지만, 저장 벡터가 바이트까지 같은 점들은 **어떤 질의에서도** 같은 점수를 받는다(정확한 동점).
그 묶음의 크기가 조회 상한(top_k)보다 크면 경계 동점이 반환 목록 정렬만으로는 복구되지 않는다.

- dense: float32 바이트 그대로 묶는다(작은 부동소수점 차이는 다른 벡터 — 반올림하지 않는다).
- sparse: (indices, values) 쌍을 정렬해 묶는다.
- payload `song_id`(곡 컬렉션)·`record_id`/`profile_id`(Context)의 형식(ASCII 숫자열 여부·길이·앞자리 0·중복)과
  point id ↔ `point_id(song_id)` 일치를 센다.

실행(워크트리 루트, 이 프로세스만 복사본을 연다):
  venv/bin/python experiments/reranking/results_v65_deterministic_ties/probe_ties.py
출력: probe_ties.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient  # noqa: E402

from src.vector_db.qdrant_backend import point_id  # noqa: E402

SRC = "artifacts/qdrant_v65_src"


def scroll(client, name):
    offset, out = None, []
    while True:
        pts, offset = client.scroll(collection_name=name, limit=512, offset=offset, with_payload=True, with_vectors=True)
        out.extend(pts)
        if offset is None:
            return out


def id_shape(values):
    vals = [str(v) for v in values]
    ascii_digit = [v for v in vals if v.isascii() and v.isdigit()]
    return {
        "n": len(vals), "distinct": len(set(vals)), "ascii_digit": len(ascii_digit),
        "non_digit_examples": sorted({v for v in vals if not (v.isascii() and v.isdigit())})[:3],
        "leading_zero": sum(1 for v in ascii_digit if len(v) > 1 and v[0] == "0"),
        "lengths": dict(sorted(Counter(len(v) for v in ascii_digit).items())),
        "stripped_differs": sum(1 for v in vals if v != v.strip()),
    }


def groups_report(keys, owners):
    by = defaultdict(list)
    for key, owner in zip(keys, owners):
        by[key].append(owner)
    sizes = sorted((len(v) for v in by.values() if len(v) > 1), reverse=True)
    return {
        "points": len(keys), "distinct_vectors": len(by),
        "points_in_duplicate_groups": sum(sizes), "duplicate_groups": len(sizes),
        "largest_groups": sizes[:10],
        "groups_larger_than_30": sum(1 for s in sizes if s > 30),
    }


def main() -> int:
    client = QdrantClient(path=SRC)
    report = {"source": SRC, "collections": {}}
    for c in sorted(x.name for x in client.get_collections().collections):
        pts = scroll(client, c)
        info = {}
        dense_keys, sparse_keys, owners = [], [], []
        for p in pts:
            vec = p.vector or {}
            payload = p.payload or {}
            owner = str(payload.get("song_id", ""))
            owners.append(owner)
            if "dense" in vec:
                dense_keys.append(hashlib.sha256(np.asarray(vec["dense"], dtype=np.float32).tobytes()).hexdigest())
            if "sparse" in vec:
                sv = vec["sparse"]
                pairs = sorted(zip(sv.indices, sv.values))
                sparse_keys.append(hashlib.sha256(json.dumps(pairs).encode()).hexdigest())
        if dense_keys:
            info["dense"] = groups_report(dense_keys, owners)
            # 같은 dense 벡터 묶음 안에 같은 곡이 여럿(Context 사실) / 다른 곡이 섞였는가
            by = defaultdict(set)
            for k, o in zip(dense_keys, owners):
                by[k].add(o)
            info["dense"]["groups_spanning_multiple_songs"] = sum(1 for v in by.values() if len(v) > 1)
        if sparse_keys:
            info["sparse"] = groups_report(sparse_keys, owners)
        info["song_id"] = id_shape(owners)
        if c.startswith("vaguefinder-context-dense"):
            info["record_id"] = id_shape([(p.payload or {}).get("record_id", "") for p in pts])
            per_song = Counter(owners)
            info["facts_per_song"] = {"songs": len(per_song), "max": max(per_song.values()),
                                      "songs_with_multiple_facts": sum(1 for v in per_song.values() if v > 1)}
            info["point_id_matches_record_id"] = sum(1 for p in pts if p.id == point_id((p.payload or {}).get("record_id", "")))
        elif c.startswith("vaguefinder-context-sparse"):
            info["profile_id"] = id_shape([(p.payload or {}).get("profile_id", "") for p in pts])
            info["point_id_matches_profile_id"] = sum(1 for p in pts if p.id == point_id((p.payload or {}).get("profile_id", "")))
        else:
            info["point_id_matches_song_id"] = sum(1 for p in pts if p.id == point_id(str((p.payload or {}).get("song_id", ""))))
            info["missing_song_id_payload"] = sum(1 for p in pts if not str((p.payload or {}).get("song_id", "")).strip())
        report["collections"][c] = info
        print(c, json.dumps(info, ensure_ascii=False)[:600], flush=True)
    client.close()
    (HERE / "probe_ties.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
