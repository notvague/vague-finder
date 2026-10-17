"""v67 하네스 공용 — main 코드 판 사본 · 서버 Qdrant(전용 컨테이너) · 외부 호출 차단 · 가사 스냅샷 · 소리 경로 구성 · 기록.

v66 `v66_eval_runtime.py`를 옮겼다. 다른 점:

1. **코드 판 하나**: `artifacts/src_main_bb7e6f0`(`git archive bb7e6f0 src`). bb7e6f0의 src는 v66 수정 후 기준 7476317과
   같다(`git diff 7476317 bb7e6f0 -- src` 비어 있음) — 사본 트리 지문이 v66 수정 후 `e2cbb47a58b13dd1`과 같아야 시작한다.
2. **모델은 가짜만**(v66 fake와 같은 결정적 정상 JSON). 실제 Vertex 경로·예산 장부는 없앴다. 루프백 밖 네트워크·Gemini·ADC·
   Mongo·Claude를 모두 막는다. 가짜 리랭커의 최종 순서는 **품질 지표가 아니다** — 후보·보호·정책 생략·리랭커 입력만 쓴다.
3. **소리 경로 구성** `V67_AUDIO`(결과를 보기 전에 RUN_INFO §3에 적은 정의):
   - `on`      현행(AUDIO_FUSION_SCALE=1)
   - `skip`    AUDIO_FUSION_SCALE=0 — 라우터의 `use_audio`가 거짓이 되어 CLAP 임베딩·오디오 조회를 하지 않는다(기존 스위치)
   - `weight0` 조회는 현행대로 하고 RRF에 넘기는 소리 가중만 0(텍스트·이미지 가중을 다시 정규화). 소리 목록은 성능 단서 원천·
               메타 캐시에는 그대로 쓰인다 — "가중만 0"과 "경로 생략"의 차이를 보려는 진단 구성(src 변경 없음, 하네스 패치)
   - `named_only` 결과 전에 정한 규칙(`named_sound.py`)이 명시된 소리 사건을 찾은 질의에서만 소리 경로를 돌린다(그 밖은 skip과 같다)
4. 기록: v66과 같은 질의 단위 기록 + CLAP `load`·`embed_texts` 호출 수·시간, 오디오 컬렉션 조회 수를 질의별로 남긴다.
"""
from __future__ import annotations

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
AUDIO = os.environ.get("V67_AUDIO", "")
AUDIO_CONFIGS = ("on", "skip", "weight0", "named_only")
if AUDIO not in AUDIO_CONFIGS:
    raise SystemExit(f"V67_AUDIO={'|'.join(AUDIO_CONFIGS)} 가 필요하다")
COMMIT = "bb7e6f0c3da70d62440a50c80d0ece636138ae9d"
V66_AFTER_TREE_SHA16 = "e2cbb47a58b13dd1"  # v66 RUN_INFO §6 — 수정 후(7476317) 사본 트리 지문
CODE_ROOT = ROOT / "artifacts/src_main_bb7e6f0"
V61_DIR = ROOT / "experiments/reranking/results_v61_hybrid_scan_audit"
V63_DIR = ROOT / "experiments/reranking/results_v63_hybrid_lyrics_validation"
for p in list(sys.path):
    if p in ("", ".", str(ROOT)):
        sys.path.remove(p)
for p in reversed([str(CODE_ROOT), str(HERE), str(V63_DIR), str(V61_DIR)]):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
if "src" in sys.modules:
    raise SystemExit("src가 이미 import됐다 — v67_eval_runtime을 가장 먼저 import해야 한다")

SNAPSHOT_DIR = Path("/private/tmp/vague-finder-hybrid-lyrics-validation/artifacts/lyrics_snapshot_v63/snapshot")
QDRANT_SERVER = "http://127.0.0.1:6370"
CONTAINER = "vague-finder-qdrant-v67"
EXPECTED_INDEX_FINGERPRINT = "238299e6c8c17f2a"
EXPECTED_LYRICS = "e0d58f28f1f4a563"
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")

FIXED_ENV: Dict[str, str] = {
    "CONTEXT_WEIGHT": "0.5", "ATTR_FILTER_WEIGHT": "0", "AUDIO_FUSION_SCALE": "0" if AUDIO == "skip" else "1",
    "SEARCH_REFERENCE_YEAR": "2026",
    "RERANKER_BACKEND": "gemini_listwise", "GEMINI_RERANK_RARE_FACT_VERIFY": "0", "GEMINI_RETRIEVAL_BACKEND": "vertex",
    "GCP_LOCATION": "global", "GEMINI_MODEL_NAME": "gemini-3.1-flash-lite", "GCP_PROJECT_ID": "v67-offline-no-calls",
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
    for key in REMOVE_ENV:
        os.environ.pop(key, None)
    applied = dict(FIXED_ENV)
    applied["QDRANT_URL"] = QDRANT_SERVER
    os.environ.update(applied)
    return applied


fixed_env()  # src import 전에 — 모듈 상수가 import 때 환경을 읽는다


class BlockedExternalCall(RuntimeError):
    pass


def _blocked(*_a: Any, **_k: Any) -> Any:
    raise BlockedExternalCall("v67: 외부 클라이언트 생성이 차단됐다")


class _BlockedClient:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise BlockedExternalCall(f"v67: {type(self).__name__} 생성이 차단됐다")


NET_ATTEMPTS: List[str] = []


def install_guards() -> None:
    """Mongo·Claude·가사 Mongo·Gemini·ADC·루프백 밖 네트워크를 모두 막는다(모델 호출 0회 실행)."""
    import socket

    orig_connect = socket.socket.connect

    def connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0] if isinstance(address, tuple) else address
            if not (str(host).startswith("127.") or str(host) in {"::1", "localhost"}):
                NET_ATTEMPTS.append(f"{host}:{address[1] if isinstance(address, tuple) else ''}")
                raise OSError(f"v67: 외부 연결 차단 {host}")
        return orig_connect(self, address)

    socket.socket.connect = connect  # type: ignore[assignment]
    orig_gai = socket.getaddrinfo

    def gai(host, *a, **k):  # type: ignore[no-untyped-def]
        h = host.decode() if isinstance(host, bytes) else host
        if h is not None and not (str(h).startswith("127.") or str(h) in {"::1", "localhost"}):
            NET_ATTEMPTS.append(f"dns:{h}")
            raise OSError(f"v67: 외부 이름 조회 차단 {h}")
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
    import socket
    import pymongo
    import src.common.mongodb as mongodb
    import src.retrieval.lyrics_exact_search as les
    import src.common.gemini_client as gc
    import src.retrieval.gemini_listwise_reranker as glr
    from google import genai

    out: Dict[str, str] = {}
    checks = {
        "pymongo.MongoClient": lambda: pymongo.MongoClient("mongodb://127.0.0.1:1"),
        "src.common.mongodb": lambda: mongodb.get_collection("songs"),
        "lyrics_mongo_client": lambda: les._make_lyrics_mongo_client(1.0),
        "make_genai_client": lambda: gc.make_genai_client(purpose="retrieval"),
        "reranker.make_genai_client": lambda: glr.make_genai_client(purpose="retrieval"),
        "adc_credentials": lambda: gc._load_default_credentials(),
        "google.genai.Client": lambda: genai.Client(vertexai=True, project="x", location="global"),
    }
    for name, fn in checks.items():
        try:
            fn()
            out[name] = "NOT BLOCKED"
        except BlockedExternalCall:
            out[name] = "blocked"
        except Exception as exc:  # noqa: BLE001
            out[name] = f"NOT BLOCKED ({type(exc).__name__})"
    before = len(NET_ATTEMPTS)
    try:
        socket.create_connection(("192.0.2.1", 9), timeout=0.5)
        out["network"] = "NOT BLOCKED"
    except OSError as exc:
        out["network"] = "blocked" if "v67" in str(exc) else f"NOT BLOCKED ({type(exc).__name__})"
    del NET_ATTEMPTS[before:]
    out["QDRANT_HYBRID_SCAN_LIMIT"] = "unset" if not os.getenv("QDRANT_HYBRID_SCAN_LIMIT") else "SET"
    bad = {k: v for k, v in out.items() if v not in {"blocked", "unset"}}
    if bad:
        raise RuntimeError(f"v67 차단 확인 실패: {bad}")
    return out


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def code_identity() -> Dict[str, Any]:
    """사본 전체(`src/**/*.py`)의 blob 목록이 bb7e6f0 src 트리와 같고, 그 지문이 v66 수정 후 판과 같은지. 실제 import 위치도 본다."""
    import subprocess

    import src.vector_db.qdrant_backend as qb
    import src.retrieval.evaluate_search_accuracy as esa
    import src.retrieval.search_router as sr

    for mod in (qb, esa, sr):
        path = Path(mod.__file__).resolve()
        if not str(path).startswith(str(CODE_ROOT.resolve()) + "/"):
            raise RuntimeError(f"{mod.__name__}가 코드 판 사본 밖에서 import됐다: {path}")
    tree = subprocess.run(["git", "ls-tree", "-r", COMMIT, "src"], capture_output=True, text=True, cwd=ROOT,
                          check=True).stdout.splitlines()
    want = sorted((line.split("\t")[1], line.split()[2]) for line in tree if line.endswith(".py"))
    have = sorted((str(p.relative_to(CODE_ROOT)), _git_blob(p.read_bytes()))
                  for p in (CODE_ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    if want != have:
        raise RuntimeError(f"코드 판 사본이 {COMMIT[:7]}의 src 트리와 다르다")
    tree_sha = hashlib.sha256(json.dumps(want).encode()).hexdigest()[:16]
    if tree_sha != V66_AFTER_TREE_SHA16:
        raise RuntimeError(f"사본 트리 지문 {tree_sha} ≠ v66 수정 후 {V66_AFTER_TREE_SHA16}")
    return {"commit": COMMIT, "code_root": str(CODE_ROOT.relative_to(ROOT)), "src_py_files": len(want),
            "src_py_tree_sha16": tree_sha, "equals_v66_after_tree": True}


# ---------------------------------------------------------------------- 가사 스냅샷(v63·v66과 같은 방식)

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


# ---------------------------------------------------------------------- 소리 경로 구성

def install_audio_config() -> Dict[str, Any]:
    """`weight0`·`named_only`만 하네스에서 라우터를 감싼다. `on`·`skip`은 기존 환경 스위치 그대로(패치 없음)."""
    from src.retrieval import search_router as sr

    info: Dict[str, Any] = {"config": AUDIO, "AUDIO_FUSION_SCALE": os.environ["AUDIO_FUSION_SCALE"], "patched": []}
    if AUDIO == "weight0":
        orig_fuse = sr._rrf_fuse

        def fuse(ranked_lists, weights, top_k, labels=None, recorder=sr.NULL_RECORDER):  # type: ignore[no-untyped-def]
            # 라우터의 유일한 호출부(labels = text_hybrid·image·audio·context)에서만 소리 가중을 0으로 —
            # 라우터가 소리를 끈 경로를 정규화하는 식(w / (w_text + w_image))을 그대로 따른다.
            if labels and len(labels) >= 3 and labels[2] == "audio" and weights[2] > 0:
                s = weights[0] + weights[1]
                head = [weights[0] / s, weights[1] / s] if s > 0 else [1.0, 0.0]
                weights = [*head, 0.0, *weights[3:]]
            return orig_fuse(ranked_lists, weights, top_k, labels=labels, recorder=recorder)

        sr._rrf_fuse = fuse  # type: ignore[assignment]
        info["patched"].append("search_router._rrf_fuse: audio weight → 0 (text·image 재정규화)")
    elif AUDIO == "named_only":
        import named_sound

        orig_search = sr.SearchRouter.search

        async def search(router_self, analysis, *args, **kwargs):  # type: ignore[no-untyped-def]
            decision = named_sound.decide(analysis.original_query, analysis)
            NAMED_DECISIONS[analysis.original_query] = decision
            if decision["run_audio"]:
                return await orig_search(router_self, analysis, *args, **kwargs)
            # 이 질의에서만 소리 경로 생략 — skip과 같은 경로(`_audio_fusion_scale() == 0`)를 탄다
            orig_scale = sr._audio_fusion_scale
            sr._audio_fusion_scale = lambda: 0.0  # type: ignore[assignment]
            try:
                return await orig_search(router_self, analysis, *args, **kwargs)
            finally:
                sr._audio_fusion_scale = orig_scale  # type: ignore[assignment]

        sr.SearchRouter.search = search  # type: ignore[assignment]
        info["patched"].append("SearchRouter.search: named_sound.decide가 거짓이면 그 요청만 _audio_fusion_scale=0")
        info["named_rule_sha16"] = sha16((HERE / "named_sound.py").read_text(encoding="utf-8"))
    return info


NAMED_DECISIONS: Dict[str, Dict[str, Any]] = {}


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
    """질의 단위 기록(평가기는 질의를 순차로 돈다)."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.lock = threading.Lock()
        self.query_ids: Dict[str, str] = {}
        self.current: Optional[str] = None
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.local = threading.local()
        self.qdrant_calls: List[Dict[str, Any]] = []
        self.offline_generate_calls = 0
        self.clap: Dict[str, Any] = {"load_calls": 0, "load_ms": [], "embed_calls": 0}

    def row(self) -> Dict[str, Any]:
        qid = self.current or "_none"
        return self.rows.setdefault(qid, {"query_id": qid, "lyrics": [], "rerank_runs": [], "generates": [],
                                          "qdrant": [], "protection": None, "final": None,
                                          "clap_embed": [], "audio_queries": 0})

    def install(self) -> None:
        rec = self
        from src.retrieval import search_router as sr
        from src.retrieval import evaluate_search_accuracy as esa
        import src.retrieval.lyrics_exact_search as les
        import src.retrieval.gemini_listwise_reranker as glr
        from src.vector_db import qdrant_backend as qb
        from src.embedding.models.audio_clap import CLAPAudioEmbedder

        # --- CLAP: 로드(첫 임베딩 때 지연 로드)와 텍스트 임베딩 호출
        orig_load, orig_embed = CLAPAudioEmbedder.load, CLAPAudioEmbedder.embed_texts

        def load(emb_self):  # type: ignore[no-untyped-def]
            # `model`은 load()를 부르는 property다 — 내부 필드로 본다(재귀 방지)
            if emb_self._model is not None and emb_self._processor is not None:
                return orig_load(emb_self)
            started = time.perf_counter()
            out = orig_load(emb_self)
            with rec.lock:
                rec.clap["load_calls"] += 1
                rec.clap["load_ms"].append(round((time.perf_counter() - started) * 1000, 3))
            return out

        def embed_texts(emb_self, texts, *, l2_normalize=True):  # type: ignore[no-untyped-def]
            started = time.perf_counter()
            out = orig_embed(emb_self, texts, l2_normalize=l2_normalize)
            with rec.lock:
                rec.clap["embed_calls"] += 1
            rec.row()["clap_embed"].append({"texts": len(texts), "ms": round((time.perf_counter() - started) * 1000, 3)})
            return out

        CLAPAudioEmbedder.load = load  # type: ignore[assignment]

        # --- 경로 실패: 라우터는 실패한 경로를 빈 결과로 바꾸고 explain에만 남긴다(평가기 explain 파일에는 안 나간다)
        orig_path_result = sr._path_result

        def path_result(name, result, recorder):  # type: ignore[no-untyped-def]
            if isinstance(result, BaseException):
                rec.row().setdefault("path_failures", []).append(f"{name}: {type(result).__name__}: {str(result)[:160]}")
            return orig_path_result(name, result, recorder)

        sr._path_result = path_result  # type: ignore[assignment]
        CLAPAudioEmbedder.embed_texts = embed_texts  # type: ignore[assignment]

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

        # --- 곡 컬렉션 논리 조회(어느 컬렉션을 몇 번, 몇 개씩)
        orig_query = qb.QdrantIndex.query

        def query(index_self, *, vector, top_k=10, include_metadata=True, namespace=qb.NAMESPACE, filter=None,
                  sparse_vector=None, **kw):  # noqa: A002
            started = time.perf_counter()
            result = orig_query(index_self, vector=vector, top_k=top_k, include_metadata=include_metadata,
                                namespace=namespace, filter=filter, sparse_vector=sparse_vector, **kw)
            collection = qb.collection_name(index_self._index_name, namespace).split("__")[0]
            entry = {"collection": collection, "hybrid": sparse_vector is not None, "filtered": bool(filter),
                     "top_k": int(top_k), "returned": len(result["matches"]),
                     "ms": round((time.perf_counter() - started) * 1000, 3)}
            row = rec.row()
            row["qdrant"].append(entry)
            if "audio" in collection:
                row["audio_queries"] += 1
            with rec.lock:
                rec.qdrant_calls.append({"query_id": rec.current, **entry})
            return result

        qb.QdrantIndex.query = query  # type: ignore[assignment]

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

        # --- 보호 판정·tier·정책 생략 · (가짜) 최종 순서
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
            row["final_fake"] = {"ids": [str(t.id) for t in final], "statuses": list(statuses or []),
                                 "errors": [e[:200] for e in (errors or [])]}
            return final

        esa.rerank_preserving_exact_lyrics = rpel  # type: ignore[assignment]

        # --- 리랭커 그룹 입력 · 프롬프트 지문 · 결정적 가짜 응답(모델 호출 없음)
        orig_run = glr.GeminiListwiseReranker.rerank_run

        def rerank_run(reranker_self, query_text, tracks, top_k, answers=None, *, deadline=None):
            judged = list(tracks[: reranker_self.config.max_candidates])
            rec.local.valid_ids = [str(t.id) for t in judged]
            run = orig_run(reranker_self, query_text, tracks, top_k, answers, deadline=deadline)
            rec.row()["rerank_runs"].append({
                "input_ids": [str(t.id) for t in tracks], "input_scores": [round(float(t.score or 0.0), 9) for t in tracks],
                "input_lyric": [t.lyric_match_type or "" for t in tracks], "top_k": int(top_k), "status": run.status,
                "model_calls": run.model_calls,
            })
            return run

        glr.GeminiListwiseReranker.rerank_run = rerank_run  # type: ignore[assignment]

        def generate(reranker_self, *, contents, config, deadline, tally=None):
            if tally is not None:
                tally[0] += 1
            ids = list(getattr(rec.local, "valid_ids", []) or [])
            n = max(1, len(ids))
            payload = {"ranking": [{"id": sid, "relevance": round(1.0 - i / n, 6), "reason": "offline"}
                                   for i, sid in enumerate(ids)], "confidence": 0.9}
            with rec.lock:
                rec.offline_generate_calls += 1
            rec.row()["generates"].append({"contents_sha16": sha16(contents), "contents_chars": len(contents)})
            return SimpleNamespace(text=json.dumps(payload))

        glr.GeminiListwiseReranker._generate = generate  # type: ignore[assignment]

    def save(self, out_dir: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "v67_queries.jsonl").open("w", encoding="utf-8") as fh:
            for qid, row in self.rows.items():
                q = next((k for k, v in self.query_ids.items() if v == qid), None)
                if q is not None and q in NAMED_DECISIONS:
                    row["named_sound"] = NAMED_DECISIONS[q]
                fh.write(json.dumps(_round(row), ensure_ascii=False) + "\n")
        audio_q = sum(r["audio_queries"] for r in self.rows.values())
        clap_q = sum(1 for r in self.rows.values() if r["clap_embed"])
        return {
            "queries_recorded": len([q for q in self.rows if q != "_none"]),
            "unattributed_rows": "_none" in self.rows,
            "qdrant_logical_calls": len(self.qdrant_calls),
            "audio_collection_queries": audio_q,
            "queries_with_clap_embed": clap_q,
            "lyrics_errors": sum(1 for r in self.rows.values() for e in r["lyrics"] if "error" in e),
            "path_failures": [f"{q}:{f}" for q, r in self.rows.items() for f in r.get("path_failures", [])],
            "offline_generate_calls": self.offline_generate_calls,
        }
