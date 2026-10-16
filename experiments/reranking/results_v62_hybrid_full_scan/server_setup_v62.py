"""v62 — 서버 방식 검증용 **전용** Docker Qdrant를 띄우고 텍스트 컬렉션만 색인 복사본에서 옮겨 내용 지문을 확인한다. 모델 호출 없음.

  - 컨테이너 `vague-finder-qdrant-v62`(qdrant/qdrant:v1.19.1 — 클라이언트와 같은 판), 127.0.0.1:6362(REST) · 6363(gRPC),
    저장소는 이 워크트리의 `artifacts/qdrant_server_v62/storage`(git 제외). **v56 컨테이너(`vague-finder-qdrant-v56`, 6356)는 건드리지 않는다.**
  - 원본 운영 색인(`artifacts/qdrant`)·운영 서버는 열지 않는다. 복사본 `artifacts/qdrant_v61_audit`(v61 복사본의 복사)에서 읽는다.
  - 하이브리드 지연 측정에는 텍스트 컬렉션만 필요하므로 그것만 옮긴다(Context 별칭 등은 이번 범위 밖).
  - 확인: count 일치 · 벡터 설정 일치 · **전체 내용 지문**(`exact_hybrid.ExactHybridReference.fingerprint`, 모든 점의 dense·sparse)이 복사본·v61 기록과 같음
실행: venv/bin/python experiments/reranking/results_v62_hybrid_full_scan/server_setup_v62.py [--stop]   (--stop: 컨테이너만 멈춘다)
출력: server_setup_v62.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v62_common as common  # noqa: E402

APPLIED_ENV, GUARD_BEFORE = common.guard()

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.migrate import migrate  # noqa: E402

from src.vector_db import qdrant_backend as qb  # noqa: E402
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME  # noqa: E402

from exact_hybrid import ExactHybridReference  # noqa: E402

CONTAINER = "vague-finder-qdrant-v62"
IMAGE = "qdrant/qdrant:v1.19.1"
REST, GRPC = 6362, 6363
URL = f"http://127.0.0.1:{REST}"
STORAGE = common.ROOT / "artifacts/qdrant_server_v62/storage"


def _docker(*args: str, check: bool = True) -> str:
    out = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


def _container_state() -> str:
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "absent"


def _wait_ready(timeout_s: float = 60) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{URL}/readyz", timeout=2) as resp:  # noqa: S310 - 루프백 전용 컨테이너
                if resp.status == 200:
                    return time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise RuntimeError("Qdrant 컨테이너가 준비되지 않았다")


def _vec_config(info: Any) -> Dict[str, Any]:
    cfg = info.config.params
    dense = {name: {"size": p.size, "distance": str(p.distance.value)} for name, p in (cfg.vectors or {}).items()} if isinstance(cfg.vectors, dict) else {}
    return {"dense": dense, "sparse": sorted((cfg.sparse_vectors or {}).keys())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if args.stop:
        if _container_state() == "running":
            _docker("stop", CONTAINER)
        print(json.dumps({"container": CONTAINER, "state": _container_state()}))
        return 0

    report: Dict[str, Any] = {"container": CONTAINER, "image": IMAGE, "url": URL, "storage": str(STORAGE.relative_to(common.ROOT)),
                              "untouched": ["vague-finder-qdrant-v56 (6356)", "artifacts/qdrant (운영 색인)", "운영 서버"]}
    state = _container_state()
    report["container_state_before"] = state
    t0 = time.perf_counter()
    if state == "absent":
        STORAGE.mkdir(parents=True, exist_ok=True)
        _docker("run", "-d", "--name", CONTAINER, "-p", f"127.0.0.1:{REST}:6333", "-p", f"127.0.0.1:{GRPC}:6334",
                "-v", f"{STORAGE}:/qdrant/storage", IMAGE)
    elif state != "running":
        _docker("start", CONTAINER)
    report["ready_seconds"] = round(_wait_ready(), 2)
    report["start_seconds"] = round(time.perf_counter() - t0, 2)

    src = QdrantClient(path=os.environ["QDRANT_PATH"])
    dst = QdrantClient(url=URL)
    report["server_version"] = dst.info().version
    collection = qb.collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE)
    existing = {c.name for c in dst.get_collections().collections}
    if collection not in existing:
        t1 = time.perf_counter()
        migrate(src, dst, collection_names=[collection], recreate_on_collision=False, batch_size=256)
        report["migrate_seconds"] = round(time.perf_counter() - t1, 1)
    else:
        report["migrate_seconds"] = None
    report["server_collections"] = sorted(c.name for c in dst.get_collections().collections)

    s_cnt, d_cnt = int(src.count(collection, exact=True).count), int(dst.count(collection, exact=True).count)
    s_cfg, d_cfg = _vec_config(src.get_collection(collection)), _vec_config(dst.get_collection(collection))
    t2 = time.perf_counter()
    src_fp = common.check_fingerprint(ExactHybridReference.from_client(src, collection).fingerprint())
    dst_fp = ExactHybridReference.from_client(dst, collection).fingerprint()
    report["fingerprint_seconds"] = round(time.perf_counter() - t2, 1)
    report["text_collection"] = {"name": collection, "points_source": s_cnt, "points_server": d_cnt, "vectors_source": s_cfg, "vectors_server": d_cfg,
                                 "fingerprint_source": src_fp, "fingerprint_server": dst_fp,
                                 "payload_indexes_server": sorted((dst.get_collection(collection).payload_schema or {}).keys())}
    ok = s_cnt == d_cnt and s_cfg == d_cfg and dst_fp["content_sha256_16"] == src_fp["content_sha256_16"] == common.EXPECTED_FINGERPRINT
    report["all_ok"] = ok
    report["guard"] = {"before": GUARD_BEFORE, "after": common.verify_guard()}
    (common.HERE / "server_setup_v62.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("container", "server_version", "migrate_seconds", "all_ok")}, ensure_ascii=False))
    print("server fingerprint:", dst_fp)
    src.close()
    dst.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
