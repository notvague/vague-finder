"""v60 재현 — Context 별칭·manifest 설정 오류에서 **서비스 기동**이 어떻게 동작하는가(수정 전/후 공용).

합성 Context 자료(tests/context/test_context_qdrant_load 도구, 곡 2개 · 4차원 가짜 임베더)를 임시 폴더의
로컬 Qdrant에 정규 경로(`load_context_qdrant`)로 발행한 뒤, 시나리오마다 저장소를 망가뜨리고 실제
`src.backend.main.app`을 TestClient로 띄운다. 기록하는 것:

- 기동 성공 여부 · `/health.context`(수정 후에만 있음) · 실패 시 예외형과 사유
- 기동 실패 뒤 같은 폴더를 다시 열 수 있는가(로컬 Qdrant 잠금 해제)
- 그 저장소에서 검색기를 직접 불렀을 때의 결과(검색 경로는 수정 전후 같다 — 라우터가 예외를 빈 결과로 바꾼다)
- 정상 저장소의 Context 후보 ID·순서(수정 전후 비교용)

밖으로 나가는 호출은 없다 — Vertex 확인은 대체하고, 피드백 색인·작업 스레드는 끄고, Mongo는 만들지 않는다.

    cd <체크아웃 루트> && PYTHONPATH=. venv/bin/python <이 파일> --out <결과 JSON>
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["SEARCH_WARMUP"] = "0"
os.environ["FEEDBACK_COLLECTION_ENABLED"] = "0"
os.environ["FEEDBACK_INDEX_WORKER_ENABLED"] = "0"
os.environ["FEEDBACK_RETRIEVAL_ENABLED"] = "0"
os.environ.pop("QDRANT_URL", None)
os.environ.pop("GEMINI_RETRIEVAL_BACKEND", None)

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from qdrant_client import models  # noqa: E402

from src.common import gemini_client as gc  # noqa: E402
from src.vector_db.qdrant_backend import QdrantVectorClient  # noqa: E402
from tests.context.test_context_qdrant_load import FakeEmbedder, _hash, _load, _sources  # noqa: E402

NAMESPACE = "test"
DENSE_ALIAS = "vaguefinder-context-dense-1024-koe5__test"
SPARSE_ALIAS = "vaguefinder-context-sparse-bm25__test"
QUERIES = ("테스트 프로그램 배경음악", "프로듀서가 데모를 다시 녹음", "삽입곡")

try:
    context_startup = importlib.import_module("src.backend.context_startup")
except ModuleNotFoundError:
    context_startup = None


def _searcher(pub, client, *, embedder=None):
    from src.retrieval.context_qdrant_search import ContextQdrantSearch

    kwargs = {"text_embedder": embedder} if embedder is not None else {}
    return ContextQdrantSearch(
        client, namespace=NAMESPACE, dense_dir=pub["dense"], sparse_dir=pub["sparse"],
        state_manifest=pub["state"], hash_fn=_hash, expected_dense_dim=4, **kwargs,
    )


def _publish(tmp: Path) -> dict:
    storage = tmp / "qdrant"
    client = QdrantVectorClient(path=str(storage))
    sources = _sources(tmp)
    _load(tmp, sources, client)
    client.close()
    state = tmp / "state.json"
    manifest = json.loads(state.read_text(encoding="utf-8"))
    return {"storage": storage, "dense": sources[1], "sparse": sources[2], "state": state,
            "manifest": manifest}


def _alias_ops(client, delete=(), create=()):
    ops = [models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=a)) for a in delete]
    ops += [models.CreateAliasOperation(create_alias=models.CreateAlias(alias_name=a, collection_name=c))
            for a, c in create]
    client.client.update_collection_aliases(change_aliases_operations=ops)


def _rewrite_state(path: Path, mutate) -> None:
    from src.vector_db.context_qdrant import _manifest_content_digest

    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    data.pop("manifest_content_sha256", None)
    data["manifest_content_sha256"] = _manifest_content_digest(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# --- 시나리오: (이름, 환경, 저장소 변형) ------------------------------------------------------

def _break_aliases_missing(pub):
    c = QdrantVectorClient(path=str(pub["storage"]))
    _alias_ops(c, delete=(DENSE_ALIAS, SPARSE_ALIAS))
    c.close()


def _break_alias_partial(pub):
    c = QdrantVectorClient(path=str(pub["storage"]))
    _alias_ops(c, delete=(SPARSE_ALIAS,))
    c.close()


def _break_wrong_target(pub):
    c = QdrantVectorClient(path=str(pub["storage"]))
    c.client.create_collection(
        "stray__gen", vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.DOT)},
    )
    _alias_ops(c, delete=(DENSE_ALIAS,), create=((DENSE_ALIAS, "stray__gen"),))
    c.close()


def _break_manifest_missing(pub):
    pub["state"].unlink()


def _break_manifest_corrupt(pub):
    pub["state"].write_text("{not json", encoding="utf-8")


def _break_manifest_mismatch(pub):
    _rewrite_state(pub["state"], lambda d: d["qdrant"].__setitem__("dense_vector_name", "vector"))


SCENARIOS = [
    ("ok", {"CONTEXT_WEIGHT": "0.5"}, None),
    ("aliases_missing", {"CONTEXT_WEIGHT": "0.5"}, _break_aliases_missing),
    ("alias_partial", {"CONTEXT_WEIGHT": "0.5"}, _break_alias_partial),
    ("alias_wrong_target", {"CONTEXT_WEIGHT": "0.5"}, _break_wrong_target),
    ("manifest_missing", {"CONTEXT_WEIGHT": "0.5"}, _break_manifest_missing),
    ("manifest_corrupt", {"CONTEXT_WEIGHT": "0.5"}, _break_manifest_corrupt),
    ("manifest_mismatch", {"CONTEXT_WEIGHT": "0.5"}, _break_manifest_mismatch),
    ("disabled_aliases_missing", {"CONTEXT_WEIGHT": "0"}, _break_aliases_missing),
    ("connection_failure", {"CONTEXT_WEIGHT": "0.5", "QDRANT_URL": "http://127.0.0.1:1"}, None),
]


def _boot(pub, env: dict) -> dict:
    """실제 앱 lifespan을 한 번 띄운다. 밖으로 나가는 것은 모두 대체."""
    from fastapi.testclient import TestClient

    from src.backend import main as backend_main
    from src.backend.api import dependencies

    saved = {k: os.environ.get(k) for k in ("CONTEXT_WEIGHT", "QDRANT_URL", "QDRANT_PATH")}
    os.environ["QDRANT_PATH"] = str(pub["storage"])
    for key in ("CONTEXT_WEIGHT", "QDRANT_URL"):
        os.environ.pop(key, None)
    os.environ.update(env)

    patches = []

    def patch(obj, name, value):
        patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    patch(gc, "verify_vertex_setup", lambda *a, **k: {"location": "stub", "models": [], "probe": "stub"})
    patch(backend_main, "prepare_indexes", lambda: False)
    patch(backend_main.feedback_worker, "start_background", lambda *a, **k: False)
    patch(backend_main.feedback_worker, "stop_and_wait", lambda *a, **k: True)
    patch(backend_main, "close_feedback_store", lambda: None)
    if context_startup is not None:
        context_startup.reset()
        patch(context_startup, "ContextQdrantSearch", lambda vector_client: _searcher(pub, vector_client))

    record: dict = {"started": None, "health_context": None, "error": None}
    dependencies.close_vector_client()
    try:
        with TestClient(backend_main.app) as client:
            health = client.get("/health").json()
            record["started"] = True
            record["health_context"] = health.get("context", "<없음 — 수정 전 코드>")
    except Exception as exc:  # noqa: BLE001 - 기동 실패를 기록한다
        record["started"] = False
        record["error"] = {"type": type(exc).__name__, "reason": getattr(exc, "reason", None),
                           "message": str(exc)}
    finally:
        dependencies.close_vector_client()
        for obj, name, value in reversed(patches):
            setattr(obj, name, value)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # 기동 실패 뒤에도 저장 폴더를 다시 열 수 있어야 한다(잠금 해제). 서버 모드 시나리오는 해당 없음.
    if "QDRANT_URL" not in env:
        try:
            QdrantVectorClient(path=str(pub["storage"])).close()
            record["storage_reopenable_after"] = True
        except Exception as exc:  # noqa: BLE001
            record["storage_reopenable_after"] = f"{type(exc).__name__}"
    return record


def _search_probe(pub) -> dict:
    """같은 저장소에서 검색기를 직접 부른 결과 — 라우터는 예외를 빈 결과 + 경고 로그로 바꾼다."""
    client = QdrantVectorClient(path=str(pub["storage"]))
    try:
        searcher = _searcher(pub, client, embedder=FakeEmbedder())
        try:
            hits = searcher.search_fused_songs(QUERIES[0], fact_k=4, sparse_k=2)
            return {"outcome": "results", "song_ids": [h.song_id for h in hits]}
        except Exception as exc:  # noqa: BLE001
            return {"outcome": "exception_to_empty_in_router", "type": type(exc).__name__,
                    "reason": getattr(exc, "reason", None), "message": str(exc)}
    finally:
        client.close()


def _fused_candidates(pub) -> dict:
    client = QdrantVectorClient(path=str(pub["storage"]))
    try:
        searcher = _searcher(pub, client, embedder=FakeEmbedder())
        return {q: [{"song_id": h.song_id, "dense_rank": h.dense_rank, "sparse_rank": h.sparse_rank,
                     "score": round(h.score, 6)}
                    for h in searcher.search_fused_songs(q, fact_k=4, sparse_k=2)] for q in QUERIES}
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    report = {
        "code": {"root": str(ROOT), "commit": commit, "has_context_startup": context_startup is not None},
        "scenarios": {},
        "fused_candidates": None,
    }
    for name, env, breaker in SCENARIOS:
        with tempfile.TemporaryDirectory(prefix=f"v60-{name}-") as tmp:
            pub = _publish(Path(tmp))
            if breaker is not None:
                breaker(pub)
            entry = {"env": env}
            try:
                entry["startup"] = _boot(pub, env)
            except Exception:  # noqa: BLE001
                entry["startup"] = {"harness_error": traceback.format_exc()}
            if "QDRANT_URL" not in env:
                entry["search_probe"] = _search_probe(pub)
            if name == "ok":
                report["fused_candidates"] = _fused_candidates(pub)
            report["scenarios"][name] = entry
            print(f"[{name}] started={entry['startup'].get('started')} "
                  f"context={entry['startup'].get('health_context')} error={entry['startup'].get('error')}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
