"""v65 평가기 하네스 공용 — 코드 판(수정 전/후) · Qdrant 모드(로컬 파일/서버) · 외부 호출 차단 · 가사 스냅샷 · 기록.

v63 `v63_runtime.py`(offline 모드)를 옮긴 것이다. 다른 점:

1. **코드 판**: `V65_CODE=before`이면 `artifacts/src_base_2a6481d`(main `2a6481d`의 `src/` 사본)를, `after`면 워크트리의
   `src/`를 import 경로 맨 앞에 둔다. 이 모듈은 어떤 `src` 모듈보다 먼저 import되어야 한다(`run_offline.py`가 그렇게 한다).
   실제로 어느 파일을 읽었는지는 `code_identity()`가 `src.vector_db.qdrant_backend.__file__`·파일 sha로 남긴다.
2. **모드**: `V65_MODE=local`이면 `QDRANT_PATH=artifacts/qdrant_v65_src`(이 프로세스만 연다), `server`면
   `QDRANT_URL=http://127.0.0.1:6366`(전용 컨테이너 `vague-finder-qdrant-v65`).
3. **Qdrant 기록**: 논리 조회(`QdrantIndex.query` · Context `_dense`/`_sparse`) 하나마다 그 안에서 실제로 나간
   `query_points`·`retrieve`·`count` 호출 수·limit·반환 점 수를 세고, 결과 id 목록을 독립 전체 점수 기준
   (`qdrant_reference.ReferenceStore`)과 대조해 동점·경계 분류를 붙인다. 기록은 값·순서를 바꾸지 않는다.
4. 모델: 기본(`V65_MODEL=fake`)은 v63 offline과 같다 — Gemini 클라이언트 생성을 막고 `GeminiListwiseReranker._generate`를
   **입력 순서 그대로의 정상 JSON**으로 바꾼다. 프롬프트는 실제로 만들어지고 sha256만 남는다. 실제 모델 호출은 없다.
   `V65_MODEL=live`는 실제 Vertex 재측정용으로 **준비만 했고 v65에서 실행하지 않았다**(LIVE_REMEASURE_PLAN.md) — 네트워크·Gemini 차단을
   풀고 `v65_budget`(HTTP 시도 단위 예약·정산·정지)이 대신 막는다. Mongo·Claude 차단은 그대로다.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = os.environ.get("V65_CODE", "")
MODE = os.environ.get("V65_MODE", "")
# 모델: fake(기본, 결정적 가짜 응답·외부 연결 차단) | live(실제 Vertex — **이번 v65에서는 실행하지 않았다**, 준비만).
MODEL = os.environ.get("V65_MODEL", "fake")
if CODE not in {"before", "after"} or MODE not in {"local", "server"} or MODEL not in {"fake", "live"}:
    raise SystemExit("V65_CODE=before|after, V65_MODE=local|server (V65_MODEL=fake|live) 가 필요하다")
BASE_SRC = ROOT / "artifacts/src_base_2a6481d"
CODE_ROOT = BASE_SRC if CODE == "before" else ROOT
V61_DIR = ROOT / "experiments/reranking/results_v61_hybrid_scan_audit"
V63_DIR = ROOT / "experiments/reranking/results_v63_hybrid_lyrics_validation"
# 순서: 코드 판 → 이 폴더 → v63(snapshot_lyrics) → v61(exact_hybrid) → 워크트리(tests 등, src는 앞의 판이 이긴다)
for p in reversed([str(CODE_ROOT), str(HERE), str(V63_DIR), str(V61_DIR), str(ROOT)]):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
if "src" in sys.modules:
    raise SystemExit("src가 이미 import됐다 — v65_eval_runtime을 가장 먼저 import해야 한다")

SNAPSHOT_DIR = Path("/private/tmp/vague-finder-hybrid-lyrics-validation/artifacts/lyrics_snapshot_v63/snapshot")
QDRANT_COPY = "artifacts/qdrant_v65_src"
QDRANT_SERVER = "http://127.0.0.1:6366"
EXPECTED_INDEX_FINGERPRINT = "238299e6c8c17f2a"
EXPECTED_LYRICS = "e0d58f28f1f4a563"
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")

FIXED_ENV: Dict[str, str] = {
    "CONTEXT_WEIGHT": "0.5", "ATTR_FILTER_WEIGHT": "0", "AUDIO_FUSION_SCALE": "1", "SEARCH_REFERENCE_YEAR": "2026",
    "RERANKER_BACKEND": "gemini_listwise", "GEMINI_RERANK_RARE_FACT_VERIFY": "0", "GEMINI_RETRIEVAL_BACKEND": "vertex",
    "GCP_LOCATION": "global", "GEMINI_MODEL_NAME": "gemini-3.1-flash-lite", "GCP_PROJECT_ID": "v65-offline-no-calls",
    "FEEDBACK_COLLECTION_ENABLED": "0", "FEEDBACK_INDEX_WORKER_ENABLED": "0", "FEEDBACK_RETRIEVAL_ENABLED": "0",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
}
REMOVE_ENV = (
    "QDRANT_URL", "QDRANT_PATH", "QDRANT_HYBRID_SCAN_LIMIT", "CLAUDE_EFFORT", "CLAUDE_API", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS", "RERANKER_ENABLED", "GEMINI_RERANK_ENABLED", "GEMINI_RERANK_MODEL_NAME",
    "GEMINI_RERANK_TIMEOUT_SECONDS", "GEMINI_RERANK_BUDGET_SECONDS", "GEMINI_RERANK_MAX_RETRIES", "GEMINI_RERANK_PASSES",
    "GEMINI_RERANK_USE_SEARCH", "LYRICS_CACHE_TTL_SECONDS", "LYRICS_FUZZY_THRESHOLD", "MONGO_URI", "MONGO_DB_NAME",
    "ATTR_FILTER_GATE", "ATTR_FILTER_K", "ATTR_FILTER_USE_GENRE", "ATTR_FILTER_GENRE_POLICY", "VF_CLAUDE_EFFORT",
    "VF_CLAUDE_THINKING",
)


def fixed_env() -> Dict[str, str]:
    keep = {k: os.environ[k] for k in ("GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS") if MODEL == "live" and os.environ.get(k)}
    for key in REMOVE_ENV:
        os.environ.pop(key, None)
    applied = dict(FIXED_ENV)
    if MODEL == "live":
        # 실제 호출: with_env.py가 운영 .env에서 이 두 키만 넘긴다. 없으면 시작하지 않는다(값은 기록하지 않는다)
        if set(keep) != {"GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"}:
            raise SystemExit("live 실행에는 GCP_PROJECT_ID·GOOGLE_APPLICATION_CREDENTIALS가 필요하다(with_env.py)")
        applied.pop("GCP_PROJECT_ID")
        os.environ.update(keep)
    if MODE == "local":
        applied["QDRANT_PATH"] = QDRANT_COPY
    else:
        applied["QDRANT_URL"] = QDRANT_SERVER
    os.environ.update(applied)
    return applied


fixed_env()  # src import 전에 — 모듈 상수가 import 때 환경을 읽는다


class BlockedExternalCall(RuntimeError):
    pass


def _blocked(*_a: Any, **_k: Any) -> Any:
    raise BlockedExternalCall("v65: 외부 클라이언트 생성이 차단됐다")


class _BlockedClient:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise BlockedExternalCall(f"v65: {type(self).__name__} 생성이 차단됐다")


NET_ATTEMPTS: List[str] = []


def install_guards() -> None:
    """Mongo·Claude·가사 Mongo는 항상 막는다. fake면 루프백 밖 네트워크와 Gemini 클라이언트도 막는다(live는 예산 통제로 대신)."""
    _install_always_blocked()
    if MODEL == "fake":
        _install_fake_only_blocks()


def _install_fake_only_blocks() -> None:
    import socket

    orig_connect = socket.socket.connect

    def connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0] if isinstance(address, tuple) else address
            if not (str(host).startswith("127.") or str(host) in {"::1", "localhost"}):
                NET_ATTEMPTS.append(f"{host}:{address[1] if isinstance(address, tuple) else ''}")
                raise OSError(f"v65: 외부 연결 차단 {host}")
        return orig_connect(self, address)

    socket.socket.connect = connect  # type: ignore[assignment]
    orig_gai = socket.getaddrinfo

    def gai(host, *a, **k):  # type: ignore[no-untyped-def]
        h = host.decode() if isinstance(host, bytes) else host
        if h is not None and not (str(h).startswith("127.") or str(h) in {"::1", "localhost"}):
            NET_ATTEMPTS.append(f"dns:{h}")
            raise OSError(f"v65: 외부 이름 조회 차단 {h}")
        return orig_gai(host, *a, **k)

    socket.getaddrinfo = gai  # type: ignore[assignment]
    import src.common.gemini_client as gc
    from google import genai
    import google.genai.client as genai_client_mod

    gc.make_genai_client = _blocked  # type: ignore[assignment]
    gc._load_default_credentials = _blocked  # type: ignore[assignment]
    genai.Client = _BlockedClient  # type: ignore[attr-defined]
    genai_client_mod.Client = _BlockedClient  # type: ignore[attr-defined]
    import src.retrieval.gemini_listwise_reranker as glr

    glr.make_genai_client = _blocked  # type: ignore[assignment]


def _install_always_blocked() -> None:
    import pymongo

    pymongo.MongoClient = _BlockedClient  # type: ignore[attr-defined]
    import src.common.mongodb as mongodb

    mongodb.MongoClient = _BlockedClient  # type: ignore[attr-defined]
    mongodb._connect = _blocked  # type: ignore[assignment]
    import src.retrieval.lyrics_exact_search as les

    les._make_lyrics_mongo_client = _blocked  # type: ignore[assignment]
    try:
        import anthropic

        anthropic.Anthropic = _BlockedClient  # type: ignore[attr-defined]
        anthropic.AsyncAnthropic = _BlockedClient  # type: ignore[attr-defined]
    except ImportError:
        pass


def verify_guards() -> Dict[str, str]:
    out: Dict[str, str] = {}
    import socket
    import pymongo
    import src.common.mongodb as mongodb
    import src.retrieval.lyrics_exact_search as les
    import src.common.gemini_client as gc
    import src.retrieval.gemini_listwise_reranker as glr
    from google import genai

    checks = {
        "pymongo.MongoClient": lambda: pymongo.MongoClient("mongodb://127.0.0.1:1"),
        "src.common.mongodb": lambda: mongodb.get_collection("songs"),
        "lyrics_mongo_client": lambda: les._make_lyrics_mongo_client(1.0),
    }
    if MODEL == "fake":
        checks.update({
            "make_genai_client": lambda: gc.make_genai_client(purpose="retrieval"),
            "reranker.make_genai_client": lambda: glr.make_genai_client(purpose="retrieval"),
            "adc_credentials": lambda: gc._load_default_credentials(),
            "google.genai.Client": lambda: genai.Client(vertexai=True, project="x", location="global"),
        })
    for name, fn in checks.items():
        try:
            fn()
            out[name] = "NOT BLOCKED"
        except BlockedExternalCall:
            out[name] = "blocked"
        except Exception as exc:  # noqa: BLE001
            out[name] = f"NOT BLOCKED ({type(exc).__name__})"
    if MODEL == "fake":
        before = len(NET_ATTEMPTS)
        try:
            socket.create_connection(("192.0.2.1", 9), timeout=0.5)
            out["network"] = "NOT BLOCKED"
        except OSError as exc:
            out["network"] = "blocked" if "v65" in str(exc) else f"NOT BLOCKED ({type(exc).__name__})"
        del NET_ATTEMPTS[before:]
    out["QDRANT_HYBRID_SCAN_LIMIT"] = "unset" if not os.getenv("QDRANT_HYBRID_SCAN_LIMIT") else "SET"
    bad = {k: v for k, v in out.items() if v not in {"blocked", "unset"}}
    if bad:
        raise RuntimeError(f"v65 차단 확인 실패: {bad}")
    return out


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def code_identity() -> Dict[str, Any]:
    import src.vector_db.qdrant_backend as qb
    import src.retrieval.context_qdrant_search as cqs

    files = {}
    for mod in (qb, cqs):
        path = Path(mod.__file__).resolve()
        files[mod.__name__] = {"path": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
                               "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}
    return {"code": CODE, "code_root": str(CODE_ROOT.relative_to(ROOT)) if CODE_ROOT != ROOT else ".", "files": files}


# ---------------------------------------------------------------------- 가사 스냅샷(v63과 같은 방식)

class LyricsSnapshotSource:
    def __init__(self) -> None:
        import gzip
        import pickle

        if not (SNAPSHOT_DIR / "COMPLETE").exists():
            raise RuntimeError(f"가사 스냅샷이 게시되지 않았다: {SNAPSHOT_DIR}")
        self.manifest = json.loads((SNAPSHOT_DIR / "manifest.json").read_text(encoding="utf-8"))
        data = (SNAPSHOT_DIR / self.manifest["file"]["name"]).read_bytes()
        if hashlib.sha256(data).hexdigest()[:16] != self.manifest["file"]["sha256_16"]:
            raise RuntimeError("가사 스냅샷 파일 sha가 manifest와 다르다")
        self.documents = pickle.loads(gzip.decompress(data))
        self.reads = 0
        self.lock = threading.Lock()

    def __call__(self, _service: Any) -> Any:
        with self.lock:
            self.reads += 1
        return iter(list(self.documents))

    def fingerprint(self) -> Dict[str, Any]:
        from snapshot_lyrics import content_digest

        pairs = [(str(d.get("song_id") or ""), str((d.get("lyrics_data") or {}).get("full_lyrics") or ""))
                 for d in self.documents]
        info = {"db": self.manifest.get("db_name"), "collection": "songs", **content_digest(pairs), "source": "v63_snapshot",
                "snapshot_file_sha256_16": self.manifest["file"]["sha256_16"]}
        if info["content_sha256_16"] != EXPECTED_LYRICS:
            raise RuntimeError(f"가사 지문 {info['content_sha256_16']} ≠ {EXPECTED_LYRICS}")
        return info


def install_lyrics_snapshot() -> LyricsSnapshotSource:
    import src.retrieval.lyrics_exact_search as les
    from src.retrieval import evaluate_search_accuracy as esa

    source = LyricsSnapshotSource()
    les.LyricsExactSearchService._source_documents = lambda self: source(self)  # type: ignore[assignment]
    fp = source.fingerprint()
    esa._lyrics_fingerprint = lambda: dict(fp)  # type: ignore[assignment]
    return source


# ---------------------------------------------------------------------- 기록

def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


class Recorder:
    """질의 단위 기록(평가기는 질의를 순차로 돈다). Qdrant 논리 조회마다 원시 호출과 기준 대조를 붙인다."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.lock = threading.Lock()
        self.query_ids: Dict[str, str] = {}
        self.current: Optional[str] = None
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.local = threading.local()
        self.qdrant_calls: List[Dict[str, Any]] = []
        self.offline_generate_calls = 0
        self.raw_outside_logical: Dict[str, int] = {}

    def row(self) -> Dict[str, Any]:
        qid = self.current or "_none"
        return self.rows.setdefault(qid, {"query_id": qid, "lyrics": [], "rerank_runs": [], "generates": [],
                                          "qdrant_seqs": [], "protection": None, "final": None})

    # --- 원시 호출 집계(논리 조회 안이면 그 조회에, 밖이면 따로)
    def _raw(self, name: str, entry: Dict[str, Any]) -> None:
        frames = getattr(self.local, "frames", None)
        if frames:
            frames[-1]["raw"].append({"op": name, **entry})
        else:
            with self.lock:
                self.raw_outside_logical[name] = self.raw_outside_logical.get(name, 0) + 1

    def _begin(self) -> Dict[str, Any]:
        frame = {"raw": []}
        if not hasattr(self.local, "frames"):
            self.local.frames = []
        self.local.frames.append(frame)
        return frame

    def _end(self) -> Dict[str, Any]:
        return self.local.frames.pop()

    def _finish(self, call: Dict[str, Any]) -> None:
        with self.lock:
            call["seq"] = len(self.qdrant_calls) + 1
            call["query_id"] = self.current
            self.qdrant_calls.append(call)
            self.row()["qdrant_seqs"].append(call["seq"])

    def install(self) -> None:
        rec = self
        from qdrant_client import QdrantClient
        from src.retrieval import search_router as sr
        from src.retrieval import evaluate_search_accuracy as esa
        import src.retrieval.lyrics_exact_search as les
        import src.retrieval.gemini_listwise_reranker as glr
        import src.retrieval.context_qdrant_search as cqs
        from src.vector_db import qdrant_backend as qb
        from qdrant_reference import classify

        # --- 원시 Qdrant 호출
        orig_qp, orig_ret, orig_cnt = QdrantClient.query_points, QdrantClient.retrieve, QdrantClient.count

        def query_points(client_self, collection_name, query=None, *a, **kw):
            res = orig_qp(client_self, collection_name, query, *a, **kw)
            frames = getattr(rec.local, "frames", None)
            if frames:
                frames[-1].setdefault("first_query", (query, kw.get("query_filter"), kw.get("using")))
            wp = kw.get("with_payload")
            rec._raw("query_points", {"collection": collection_name.split("__")[0], "limit": kw.get("limit"),
                                      "returned": len(res.points),
                                      "with_payload": wp if isinstance(wp, (bool, type(None))) else list(wp)})
            return res

        def retrieve(client_self, collection_name, ids, *a, **kw):
            res = orig_ret(client_self, collection_name, ids, *a, **kw)
            wp = kw.get("with_payload")
            rec._raw("retrieve", {"collection": collection_name.split("__")[0], "ids": len(list(ids)), "returned": len(res),
                                  "with_payload": wp if isinstance(wp, (bool, type(None))) else list(wp)})
            return res

        def count(client_self, collection_name, *a, **kw):
            res = orig_cnt(client_self, collection_name, *a, **kw)
            rec._raw("count", {"collection": collection_name.split("__")[0], "filtered": kw.get("count_filter") is not None})
            return res

        QdrantClient.query_points = query_points  # type: ignore[assignment]
        QdrantClient.retrieve = retrieve  # type: ignore[assignment]
        QdrantClient.count = count  # type: ignore[assignment]

        # --- 라우터 search: 현재 질의 · 후보 30개
        orig_router_search = sr.SearchRouter.search

        async def router_search(router_self, analysis, *args, **kwargs):
            rec.current = rec.query_ids.get(getattr(analysis, "original_query", None))
            started = time.perf_counter()
            result = await orig_router_search(router_self, analysis, *args, **kwargs)
            row = rec.row()
            row["search_ms"] = round((time.perf_counter() - started) * 1000, 3)
            row["candidates"] = [
                {"id": str(t.id), "score": round(float(t.score or 0.0), 9), "lyric": t.lyric_match_type or "",
                 "lyric_score": round(float(t.lyric_match_score or 0.0), 9) if t.lyric_match_score is not None else None}
                for t in result
            ]
            return result

        sr.SearchRouter.search = router_search  # type: ignore[assignment]

        # --- 곡 컬렉션 논리 조회
        orig_query = qb.QdrantIndex.query

        def query(index_self, *, vector, top_k=10, include_metadata=True, namespace=qb.NAMESPACE, filter=None,
                  sparse_vector=None, **kw):  # noqa: A002
            rec._begin()
            started = time.perf_counter()
            try:
                result = orig_query(index_self, vector=vector, top_k=top_k, include_metadata=include_metadata,
                                    namespace=namespace, filter=filter, sparse_vector=sparse_vector, **kw)
            finally:
                frame = rec._end()
            elapsed = (time.perf_counter() - started) * 1000
            collection = qb.collection_name(index_self._index_name, namespace)
            ref = rec.store.by_name[collection]
            qfilter = qb.translate_filter(filter)
            allowed = ref.allowed(qfilter)
            if sparse_vector is not None:
                ranking = rec.store.hybrid_ranking(vector, sparse_vector, allowed)
            else:
                rows, scores = ref.scores(using="dense", query=vector, allowed=allowed)
                ranking = ref.ranking(rows, scores)
            ids = [str(m["id"]) for m in result["matches"]]
            cls = classify(ids, ranking, int(top_k))
            ref_score = {r[0]: r[1] for r in ranking}
            deltas = [abs(float(m["score"]) - ref_score[str(m["id"])]) for m in result["matches"] if str(m["id"]) in ref_score]
            rec._finish({
                "kind": "index", "collection": collection.split("__")[0], "hybrid": sparse_vector is not None,
                "filtered": bool(filter), "top_k": int(top_k), "elapsed_ms": round(elapsed, 3), "result_ids": ids,
                "max_abs_score_delta": max(deltas) if deltas else 0.0, "candidates_total": len(ranking),
                "raw": frame["raw"], **cls,
            })
            return result

        qb.QdrantIndex.query = query  # type: ignore[assignment]

        # --- Context 논리 조회
        def wrap_context(name: str, kind: str):
            orig = getattr(cqs.ContextQdrantSearch, name)

            def wrapped(search_self, snapshot, query_text, limit, *a, **kw):
                rec._begin()
                started = time.perf_counter()
                try:
                    hits = orig(search_self, snapshot, query_text, limit, *a, **kw)
                finally:
                    frame = rec._end()
                elapsed = (time.perf_counter() - started) * 1000
                collection = snapshot.dense_collection if kind == "fact" else snapshot.sparse_collection
                ref = rec.store.by_name[collection]
                first = frame.get("first_query")
                ids = [h.record_id if kind == "fact" else h.profile_id for h in hits]
                call = {"kind": f"context_{kind}", "collection": collection.split("__")[0], "top_k": int(limit),
                        "elapsed_ms": round(elapsed, 3), "result_ids": ids, "result_song_ids": [h.song_id for h in hits],
                        "raw": frame["raw"], "media_filtered": False}
                if first is not None:
                    qvec, qfilter, using = first
                    call["media_filtered"] = qfilter is not None
                    rows, scores = ref.scores(using=using, query=qvec, allowed=ref.allowed(qfilter))
                    ranking = ref.ranking(rows, scores)
                    call.update(classify(ids, ranking, int(limit)))
                    ref_score = {r[0]: r[1] for r in ranking}
                    deltas = [abs(float(h.score) - ref_score[i]) for h, i in zip(hits, ids) if i in ref_score]
                    call["max_abs_score_delta"] = max(deltas) if deltas else 0.0
                    call["candidates_total"] = len(ranking)
                else:
                    call["class"] = "no_query"  # 조회하지 않음(빈 sparse·빈 작품명 필터)
                rec._finish(call)
                return hits

            setattr(cqs.ContextQdrantSearch, name, wrapped)

        wrap_context("_dense", "fact")
        wrap_context("_sparse", "profile")

        # --- 가사 검색
        orig_lyrics_search = les.LyricsExactSearchService.search

        def lyrics_search(service_self, clues, top_k=100, snapshot_out=None):
            entry: Dict[str, Any] = {"clues": len(clues)}
            try:
                result = orig_lyrics_search(service_self, clues, top_k=top_k, snapshot_out=snapshot_out)
            except BaseException as exc:
                entry["error"] = type(exc).__name__
                rec.row()["lyrics"].append(entry)
                raise
            entry.update({"results": len(result), "top": [
                {"id": str(t.id), "type": t.lyric_match_type, "score": round(float(t.lyric_match_score or 0), 9)}
                for t in result[:30]]})
            rec.row()["lyrics"].append(entry)
            return result

        les.LyricsExactSearchService.search = lyrics_search  # type: ignore[assignment]

        # --- 보호 판정·tier·정책 생략 · 최종 순서
        orig_rpel = esa.rerank_preserving_exact_lyrics

        def rpel(reranker, analysis, candidates, top_k, answers=None, recorder=esa.NULL_RECORDER, errors=None,
                 statuses=None, fallback_candidates=None, skips=None):
            protected_ids = esa.select_protected_lyric_ids(candidates)
            tiers = {str(t.id): round(float(esa.exact_lyric_constraint_score(analysis, t)), 9)
                     for t in candidates if t.id in protected_ids}
            eligible = bool(getattr(reranker, "enabled", True)) and len(candidates) > 1
            skip = esa.rerank_skip_reason(reranker, analysis, candidates) if eligible else ""
            final = orig_rpel(reranker, analysis, candidates, top_k, answers=answers, recorder=recorder, errors=errors,
                              statuses=statuses, fallback_candidates=fallback_candidates, skips=skips)
            row = rec.row()
            row["protection"] = {"protected_ids": [str(t.id) for t in candidates if t.id in protected_ids],
                                 "tiers": tiers, "policy_skip": skip, "eligible": eligible}
            row["final"] = {"ids": [str(t.id) for t in final], "statuses": list(statuses or []),
                            "errors": [e[:200] for e in (errors or [])]}
            return final

        esa.rerank_preserving_exact_lyrics = rpel  # type: ignore[assignment]

        # --- 리랭커 그룹 입력 · 프롬프트 지문 · 결정적 가짜 응답
        orig_run = glr.GeminiListwiseReranker.rerank_run

        def rerank_run(reranker_self, query_text, tracks, top_k, answers=None, *, deadline=None):
            judged = list(tracks[: reranker_self.config.max_candidates])
            rec.local.valid_ids = [str(t.id) for t in judged]
            run = orig_run(reranker_self, query_text, tracks, top_k, answers, deadline=deadline)
            mixes = {sid: _round(dataclasses.asdict(m)) for sid, m in (run.mixes or {}).items()}
            rec.row()["rerank_runs"].append({
                "input_ids": [str(t.id) for t in tracks], "input_scores": [round(float(t.score or 0.0), 9) for t in tracks],
                "input_lyric": [t.lyric_match_type or "" for t in tracks], "top_k": int(top_k), "status": run.status,
                "model_calls": run.model_calls, "output_ids": [str(t.id) for t in run.tracks],
                "mixes_sha16": sha16(json.dumps(mixes, sort_keys=True)),
            })
            return run

        glr.GeminiListwiseReranker.rerank_run = rerank_run  # type: ignore[assignment]

        orig_generate = glr.GeminiListwiseReranker._generate

        def generate_live(reranker_self, *, contents, config, deadline, tally=None):
            entry = {"contents_sha16": sha16(contents), "contents_chars": len(contents),
                     "config_sha16": sha16(json.dumps(config, sort_keys=True, default=str))}
            started = time.perf_counter()
            try:
                response = orig_generate(reranker_self, contents=contents, config=config, deadline=deadline, tally=tally)
            except BaseException as exc:
                entry.update({"error": type(exc).__name__, "ms": round((time.perf_counter() - started) * 1000, 3)})
                rec.row()["generates"].append(entry)
                raise
            entry.update({"ms": round((time.perf_counter() - started) * 1000, 3),
                          "response_text_sha16": sha16(getattr(response, "text", "") or ""),
                          "model_version": getattr(response, "model_version", None)})
            rec.row()["generates"].append(entry)
            budget = sys.modules.get("v65_budget")
            if budget is not None and budget.GUARD is not None and budget.GUARD.halted:
                raise RuntimeError(f"예산 통제 정지: {budget.GUARD.halt_reason}")
            return response

        def generate(reranker_self, *, contents, config, deadline, tally=None):
            if tally is not None:
                tally[0] += 1
            ids = list(getattr(rec.local, "valid_ids", []) or [])
            n = max(1, len(ids))
            payload = {"ranking": [{"id": sid, "relevance": round(1.0 - i / n, 6), "reason": "offline"}
                                   for i, sid in enumerate(ids)], "confidence": 0.9}
            with rec.lock:
                rec.offline_generate_calls += 1
            rec.row()["generates"].append({"contents_sha16": sha16(contents), "contents_chars": len(contents),
                                           "config_sha16": sha16(json.dumps(config, sort_keys=True, default=str))})
            return SimpleNamespace(text=json.dumps(payload))

        glr.GeminiListwiseReranker._generate = generate if MODEL == "fake" else generate_live  # type: ignore[assignment]

    def save(self, out_dir: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "v65_queries.jsonl").open("w", encoding="utf-8") as fh:
            for row in self.rows.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out_dir / "v65_qdrant_calls.jsonl").open("w", encoding="utf-8") as fh:
            for call in self.qdrant_calls:
                fh.write(json.dumps(_round(call), ensure_ascii=False) + "\n")
        classes: Dict[str, int] = {}
        for c in self.qdrant_calls:
            key = f"{c['kind']}:{c['collection'].replace('vaguefinder-', '')}:{c.get('class')}"
            classes[key] = classes.get(key, 0) + 1
        return {
            "queries_recorded": len([q for q in self.rows if q != "_none"]),
            "unattributed_rows": "_none" in self.rows,
            "qdrant_logical_calls": len(self.qdrant_calls),
            "qdrant_classes": dict(sorted(classes.items())),
            "raw_outside_logical": dict(self.raw_outside_logical),
            "lyrics_errors": sum(1 for r in self.rows.values() for e in r["lyrics"] if "error" in e),
            "offline_generate_calls": self.offline_generate_calls,
        }
