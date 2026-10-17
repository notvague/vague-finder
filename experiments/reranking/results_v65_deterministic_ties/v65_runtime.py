"""v65 HTTP 검증 공용 — v64 `v64_runtime.py`를 옮긴 것(외부 호출 차단 · 저장 분석 캐시 · 가짜 SDK를 꽂은 운영 리랭커 ·
v63 가사 스냅샷 · Qdrant 구간 기록). 달라진 점:

- **코드 판**: `V65_CODE=before`이면 `artifacts/src_base_2a6481d`(main `2a6481d`의 `src/`)를, `after`면 워크트리 `src/`를
  import 경로 맨 앞에 둔다. 이 모듈은 어떤 `src` 모듈보다 먼저 import되어야 한다.
- **가짜 SDK 기록**: 호출마다 프롬프트 본문의 sha256 앞 16자리(`contents_sha16`)와 후보 id를 JSONL에 남긴다(본문은 남기지 않는다).
- 나머지(차단 목록·가사 자료원·분석 캐시·리랭커 주입·관측)는 v64와 같다. 아래는 v64 설명 그대로다.

v64 서버 검증 공용 — 외부 호출 차단 · 저장 분석 캐시 · 가짜 SDK를 꽂은 운영 리랭커 · v63 가사 스냅샷 · Qdrant 구간 기록.

서비스 코드(`src/`)는 바꾸지 않는다. 검증 프로세스 안에서 바꾸는 자리는 아래뿐이고, 모두 **서비스 모듈 import 전에** 건다.

1. 네트워크 — 루프백이 아닌 TCP 연결·이름 조회를 예외로 막고 시도를 모두 기록한다(대상 호스트·포트·첫 호출 위치).
   로컬 Docker Qdrant(127.0.0.1:6364)만 허용된다. 실제 외부 호출이 0회였다는 근거가 이 기록이다.
2. 클라이언트 생성 — `pymongo.MongoClient` · `src.common.mongodb` · 가사 전용 Mongo 클라이언트 · `google.genai.Client` ·
   `gemini_client.make_genai_client`(리랭커 모듈이 import한 이름 포함) · ADC 읽기(`_load_default_credentials`) · `anthropic.*`.
3. 기동 검사 `verify_vertex_setup` — 네트워크 없는 대체(v56과 같다). 실제 검사 함수가 차단 아래에서 외부로 나가지 않는지는
   `probe_guards.py`가 따로 확인한다.
4. 가사 자료원 — `LyricsExactSearchService._source_documents`를 v63 스냅샷 읽기로(v63 `v63_runtime.install_lyrics_snapshot`과
   같은 방식). 판 수명·갱신·검색·보호 규칙은 운영 코드 그대로다.
5. 분석 — 저장 캐시(`CachedAnalyzer`, v56과 같다). 캐시에 없는 질의는 예외 → HTTP 500(규칙 폴백으로 채우지 않는다).
6. 리랭커 — 운영 `GeminiListwiseReranker`에 가짜 SDK 클라이언트(`tests/fake_gemini_client.py`)를 주입. 재시도·파서·공유 마감·
   그룹 상태는 실물이다.
7. 관측 — `QdrantIndex._count`·`_search`·`_payloads`와 가사 `search`를 `timing.step`으로 감싸 요청별 구간(`qdrant.count` 등)을
   `SEARCH_TIMING_LOG`에 남긴다. 값·순서는 바꾸지 않는다(감싸기 비용은 A·D 같은 조건).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V63_DIR = ROOT / "experiments/reranking/results_v63_hybrid_lyrics_validation"
CODE = os.environ.get("V65_CODE", "")
if CODE not in {"before", "after"}:
    raise SystemExit("V65_CODE=before|after 가 필요하다")
CODE_ROOT = ROOT / "artifacts/src_base_2a6481d" if CODE == "before" else ROOT
for p in reversed([str(CODE_ROOT), str(HERE), str(V63_DIR), str(ROOT)]):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
if "src" in sys.modules:
    raise SystemExit("src가 이미 import됐다 — v65_runtime을 가장 먼저 import해야 한다")

SNAPSHOT_DIR = Path(os.getenv(
    "V65_LYRICS_SNAPSHOT_DIR",
    "/private/tmp/vague-finder-hybrid-lyrics-validation/artifacts/lyrics_snapshot_v63/snapshot",
))
EXPECTED_LYRICS_CONTENT = "e0d58f28f1f4a563"
DEFAULT_CACHES = [
    ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v06_dev_vertex.json",
    ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v09_dev_vertex.json",
    ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v10_dev_vertex.json",
    # v55 봉인 test(이미 열림)의 Vertex 분석 — t014·t049 장애 계약 재현에만 쓴다. 새 평가·튜닝에 쓰지 않는다.
    ROOT / "experiments/reranking/results_v55_attr_final_validation/analysis_cache_v10_test_vertex.json",
]

# ---------------------------------------------------------------------- 1. 네트워크

NET_ATTEMPTS: List[Dict[str, Any]] = []
_NET_LOCK = threading.Lock()
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_getaddrinfo = socket.getaddrinfo
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}


class ExternalNetworkBlocked(OSError):
    pass


def _is_loopback(host: Any) -> bool:
    text = str(host or "")
    return text in _LOOPBACK or text.startswith("127.")


def _record_attempt(kind: str, target: str) -> None:
    frames = [f"{Path(f.filename).name}:{f.lineno}" for f in traceback.extract_stack()[-12:-2]]
    with _NET_LOCK:
        NET_ATTEMPTS.append({"kind": kind, "target": target, "at": time.strftime("%H:%M:%S"), "stack": frames})
    log = os.getenv("V65_NET_LOG")
    if log:
        try:
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(NET_ATTEMPTS[-1], ensure_ascii=False) + "\n")
        except OSError:
            pass


def _guard_address(sock: socket.socket, address: Any) -> None:
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return  # 유닉스 소켓 등
    host = address[0] if isinstance(address, tuple) and address else address
    if not _is_loopback(host):
        port = address[1] if isinstance(address, tuple) and len(address) > 1 else ""
        _record_attempt("connect", f"{host}:{port}")
        raise ExternalNetworkBlocked(f"v65: 외부 연결 차단 {host}:{port}")


def _connect(self, address):  # type: ignore[no-untyped-def]
    _guard_address(self, address)
    return _orig_connect(self, address)


def _connect_ex(self, address):  # type: ignore[no-untyped-def]
    _guard_address(self, address)
    return _orig_connect_ex(self, address)


def _getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
    if host is not None and not _is_loopback(host if not isinstance(host, bytes) else host.decode()):
        _record_attempt("dns", str(host))
        raise ExternalNetworkBlocked(f"v65: 외부 이름 조회 차단 {host}")
    return _orig_getaddrinfo(host, *args, **kwargs)


def install_network_guard() -> None:
    socket.socket.connect = _connect  # type: ignore[assignment]
    socket.socket.connect_ex = _connect_ex  # type: ignore[assignment]
    socket.getaddrinfo = _getaddrinfo  # type: ignore[assignment]


# ---------------------------------------------------------------------- 2. 클라이언트 생성 차단

class BlockedExternalCall(RuntimeError):
    """실제 외부 클라이언트 생성이 시도됐다 — 검증 조건 위반이므로 조용히 넘기지 않는다."""


BLOCKED_CALLS: List[str] = []


def _blocked_factory(name: str):
    def _blocked(*_a: Any, **_k: Any) -> Any:
        BLOCKED_CALLS.append(name)
        raise BlockedExternalCall(f"v65: {name} 차단")
    return _blocked


class _BlockedClient:
    _name = "client"

    def __init__(self, *_a: Any, **_k: Any) -> None:
        BLOCKED_CALLS.append(type(self)._name)
        raise BlockedExternalCall(f"v65: {type(self)._name} 생성 차단")


def _blocked_class(name: str) -> type:
    return type(f"Blocked_{name.replace('.', '_')}", (_BlockedClient,), {"_name": name})


_installed = False


def install_guards() -> None:
    """서비스 모듈 import 전에 부른다(코드 판 선택 뒤)."""
    global _installed
    if _installed:
        return
    install_network_guard()
    import pymongo

    pymongo.MongoClient = _blocked_class("pymongo.MongoClient")  # type: ignore[attr-defined]
    import src.common.mongodb as mongodb

    mongodb.MongoClient = _blocked_class("mongodb.MongoClient")  # type: ignore[attr-defined]
    mongodb._connect = _blocked_factory("mongodb._connect")  # type: ignore[assignment]
    import src.retrieval.lyrics_exact_search as les

    les._make_lyrics_mongo_client = _blocked_factory("lyrics_mongo_client")  # type: ignore[assignment]
    import src.common.gemini_client as gc
    from google import genai
    import google.genai.client as genai_client_mod

    gc.make_genai_client = _blocked_factory("make_genai_client")  # type: ignore[assignment]
    gc._load_default_credentials = _blocked_factory("adc_credentials")  # type: ignore[assignment]
    genai.Client = _blocked_class("google.genai.Client")  # type: ignore[attr-defined]
    genai_client_mod.Client = _blocked_class("google.genai.Client")  # type: ignore[attr-defined]
    import src.retrieval.gemini_listwise_reranker as glr

    if hasattr(glr, "make_genai_client"):
        glr.make_genai_client = _blocked_factory("reranker.make_genai_client")  # type: ignore[assignment]
    try:
        import anthropic

        anthropic.Anthropic = _blocked_class("anthropic.Anthropic")  # type: ignore[attr-defined]
        anthropic.AsyncAnthropic = _blocked_class("anthropic.AsyncAnthropic")  # type: ignore[attr-defined]
    except ImportError:
        pass
    _installed = True


def verify_guards() -> Dict[str, str]:
    """차단이 실제로 걸렸는가 — 하나라도 아니면 RuntimeError. 확인 호출은 BLOCKED_CALLS에 남기지 않는다."""
    out: Dict[str, str] = {}
    import pymongo
    import src.common.mongodb as mongodb
    import src.retrieval.lyrics_exact_search as les
    import src.common.gemini_client as gc
    import src.retrieval.gemini_listwise_reranker as glr
    from google import genai

    checks = [
        ("pymongo.MongoClient", lambda: pymongo.MongoClient("mongodb://127.0.0.1:1")),
        ("src.common.mongodb", lambda: mongodb.get_collection("songs")),
        ("lyrics_mongo_client", lambda: les._make_lyrics_mongo_client(1.0)),
        ("make_genai_client", lambda: gc.make_genai_client(purpose="retrieval")),
        ("adc_credentials", lambda: gc._load_default_credentials()),
        ("google.genai.Client", lambda: genai.Client(vertexai=True, project="x", location="global")),
    ]
    if hasattr(glr, "make_genai_client"):
        checks.append(("reranker.make_genai_client", lambda: glr.make_genai_client(purpose="retrieval")))
    before = len(BLOCKED_CALLS)
    for name, fn in checks:
        try:
            fn()
            out[name] = "NOT BLOCKED"
        except BlockedExternalCall:
            out[name] = "blocked"
        except Exception as exc:  # noqa: BLE001 - 다른 예외면 차단이 아니다
            out[name] = f"NOT BLOCKED ({type(exc).__name__})"
    del BLOCKED_CALLS[before:]
    try:
        socket.create_connection(("192.0.2.1", 9), timeout=0.5)  # TEST-NET-1, 원래도 닿지 않는 주소
        out["network"] = "NOT BLOCKED"
    except ExternalNetworkBlocked:
        out["network"] = "blocked"
    except OSError as exc:
        out["network"] = f"NOT BLOCKED ({type(exc).__name__})"
    with _NET_LOCK:  # 확인용 시도는 집계에서 뺀다
        if NET_ATTEMPTS and NET_ATTEMPTS[-1]["target"].startswith("192.0.2.1"):
            NET_ATTEMPTS.pop()
    bad = {k: v for k, v in out.items() if v != "blocked"}
    if bad:
        raise RuntimeError(f"v65 차단 확인 실패: {bad}")
    return out


def fake_verify_vertex_setup(*_args: Any, **_kwargs: Any) -> dict:
    """기동 검사 대체 — 네트워크 없음. 운영 설정 점검을 검증하는 것이 아니다(그 경로는 probe_guards.py)."""
    return {"backend": "blocked(v65)", "location": "none", "model": "fake", "models": ["fake"], "probe": "skipped(v65)"}


# ---------------------------------------------------------------------- 4. 가사 스냅샷

class LyricsSnapshotSource:
    def __init__(self) -> None:
        if not (SNAPSHOT_DIR / "COMPLETE").exists():
            raise RuntimeError(f"가사 스냅샷이 게시되지 않았다: {SNAPSHOT_DIR}")
        self.manifest = json.loads((SNAPSHOT_DIR / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete" or not self.manifest.get("consistent_before_full_after"):
            raise RuntimeError("가사 스냅샷 manifest가 완전하지 않다")
        data = (SNAPSHOT_DIR / self.manifest["file"]["name"]).read_bytes()
        if hashlib.sha256(data).hexdigest()[:16] != self.manifest["file"]["sha256_16"]:
            raise RuntimeError("가사 스냅샷 파일 sha가 manifest와 다르다")
        self.documents = pickle.loads(gzip.decompress(data))
        if len(self.documents) != self.manifest["documents"]:
            raise RuntimeError("가사 스냅샷 문서 수가 manifest와 다르다")
        from snapshot_lyrics import content_digest

        pairs = [(str(d.get("song_id") or ""), str((d.get("lyrics_data") or {}).get("full_lyrics") or "")) for d in self.documents]
        self.content = content_digest(pairs)
        if self.content["content_sha256_16"] != EXPECTED_LYRICS_CONTENT:
            raise RuntimeError(f"가사 내용 지문 {self.content['content_sha256_16']} ≠ {EXPECTED_LYRICS_CONTENT}")
        self.reads = 0
        self.lock = threading.Lock()

    def __call__(self, _service: Any):
        with self.lock:
            self.reads += 1
        return iter(list(self.documents))

    def describe(self) -> Dict[str, Any]:
        return {
            "dir": str(SNAPSHOT_DIR), "documents": len(self.documents), "content_sha256_16": self.content["content_sha256_16"],
            "file_sha256_16": self.manifest["file"]["sha256_16"],
            "full_document_sha256_16": self.manifest.get("full_document_sha256_16"),
            "finished_at": self.manifest.get("finished_at"), "reads": self.reads,
        }


_lyrics: Optional[LyricsSnapshotSource] = None


def install_lyrics_snapshot() -> LyricsSnapshotSource:
    global _lyrics
    import src.retrieval.lyrics_exact_search as les

    _lyrics = LyricsSnapshotSource()
    source = _lyrics
    les.LyricsExactSearchService._source_documents = lambda self: source(self)  # type: ignore[assignment]
    return source


def lyrics_source() -> Optional[LyricsSnapshotSource]:
    return _lyrics


# ---------------------------------------------------------------------- 5. 분석 캐시

class CachedAnalyzer:
    """저장 분석 캐시로만 답한다(v56과 같다). `analyze_async`는 일부러 두지 않는다 — 라우트가 동기 분석기를 스레드로 옮긴다."""

    analysis_budget_seconds = 20.0

    def __init__(self, cache_paths: Optional[List[Path]] = None) -> None:
        from src.retrieval.analysis_cache import load_cache

        self.by_query: Dict[str, Tuple[str, Dict[str, Any], bool, str]] = {}
        self.sources: List[Dict[str, Any]] = []
        for path in cache_paths or DEFAULT_CACHES:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"분석 캐시 없음: {path}")
            cache = load_cache(path)
            for qid, entry in cache.entries.items():
                self.by_query[entry["query"]] = (qid, entry["analysis"], bool(entry.get("fallback")), str(path))
            analyzer = (cache.meta or {}).get("analyzer", {})
            self.sources.append({
                "path": str(path.relative_to(ROOT)), "split": (cache.meta or {}).get("split"), "entries": len(cache.entries),
                "fallback_entries": sum(1 for e in cache.entries.values() if e.get("fallback")),
                "gemini_backend": analyzer.get("gemini_backend"), "prompt_sha": analyzer.get("prompt_sha"),
                "postprocess_sha": analyzer.get("postprocess_sha"),
                "file_sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            })
        self.misses: List[str] = []

    def analyze(self, query: str):
        from src.backend.schemas.query import QueryAnalysis
        from src.retrieval.analysis_cache import AnalysisCacheError

        hit = self.by_query.get(query)
        if hit is None:
            self.misses.append(query)
            raise AnalysisCacheError("v65: 분석 캐시에 없는 질의 — 모델 호출 없이 중단")
        qid, analysis, fallback, _ = hit
        if fallback:
            raise AnalysisCacheError(f"v65: {qid}는 규칙 폴백 분석이라 쓰지 않는다")
        return QueryAnalysis(**analysis)


_analyzer: Optional[CachedAnalyzer] = None


def get_cached_analyzer() -> CachedAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = CachedAnalyzer()
    return _analyzer


# ---------------------------------------------------------------------- 6. 가짜 SDK + 운영 리랭커

_fake_client = None


def get_fake_client():
    global _fake_client
    if _fake_client is None:
        from tests.fake_gemini_client import FakeGeminiClient

        class ShaRecordingFakeClient(FakeGeminiClient):
            """프롬프트 sha16을 호출 기록에 더한다(본문은 남기지 않는다)."""

            def _begin(self, contents: str, mode: str):  # type: ignore[override]
                call = super()._begin(contents, mode)
                call.contents_sha16 = hashlib.sha256(contents.encode("utf-8")).hexdigest()[:16]  # type: ignore[attr-defined]
                return call

            def _end(self, call, outcome: str) -> None:  # type: ignore[override]
                log, self._log_path = self._log_path, None
                try:
                    super()._end(call, outcome)
                finally:
                    self._log_path = log
                if log:
                    import dataclasses as _dc

                    row = _dc.asdict(call)
                    row["candidate_ids"] = "|".join(call.candidate_ids)
                    row["contents_sha16"] = getattr(call, "contents_sha16", "")
                    with open(log, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        _fake_client = ShaRecordingFakeClient(
            mode=os.getenv("V65_FAKE_MODE", "normal"),
            mode_file=os.getenv("V65_FAKE_MODE_FILE") or None,
            log_path=os.getenv("V65_FAKE_LOG") or None,
        )
    return _fake_client


def make_fake_reranker():
    from src.retrieval.gemini_listwise_reranker import GeminiListwiseReranker

    return GeminiListwiseReranker(client=get_fake_client(), configured=True)


# ---------------------------------------------------------------------- 7. 관측

LYRICS_CALLS: List[Dict[str, Any]] = []


def install_observers() -> None:
    from src.retrieval import timing
    from src.vector_db import qdrant_backend as qb
    import src.retrieval.lyrics_exact_search as les

    def _short(collection: str) -> str:
        return collection.split("__")[0].replace("vaguefinder-", "")

    orig_count, orig_search, orig_payloads = qb.QdrantIndex._count, qb.QdrantIndex._search, qb.QdrantIndex._payloads

    def _count(self, collection, query_filter):
        with timing.step("qdrant.count", collection=_short(collection), filtered=query_filter is not None):
            return orig_count(self, collection, query_filter)

    def _search(self, collection, *, query, using, limit, query_filter, with_payload):
        with timing.step("qdrant.search", collection=_short(collection), using=using, limit=int(limit),
                         filtered=query_filter is not None):
            return orig_search(self, collection, query=query, using=using, limit=limit,
                               query_filter=query_filter, with_payload=with_payload)

    def _payloads(self, collection, ids, fields=None):
        with timing.step("qdrant.payloads", collection=_short(collection), n=len(ids)):
            return orig_payloads(self, collection, ids, fields)

    qb.QdrantIndex._count = _count  # type: ignore[assignment]
    qb.QdrantIndex._search = _search  # type: ignore[assignment]
    qb.QdrantIndex._payloads = _payloads  # type: ignore[assignment]

    orig_lyrics = les.LyricsExactSearchService.search

    def lyrics_search(self, clues, top_k=100, snapshot_out=None):
        binding = timing.current()
        request_id = getattr(binding[0], "request_id", "") if binding else ""
        kinds = [getattr(c, "kind", "") for c in clues]
        started = time.perf_counter()
        row: Dict[str, Any] = {"request_id": request_id, "clue_kinds": kinds}
        try:
            result = orig_lyrics(self, clues, top_k=top_k, snapshot_out=snapshot_out)
        except BaseException as exc:
            row.update({"ok": False, "error": type(exc).__name__, "ms": round((time.perf_counter() - started) * 1000, 2)})
            LYRICS_CALLS.append(row)
            raise
        row.update({
            "ok": True, "n_results": len(result), "ms": round((time.perf_counter() - started) * 1000, 2),
            "match_types": sorted({str(getattr(t, "lyric_match_type", "") or "") for t in result}),
        })
        LYRICS_CALLS.append(row)
        return result

    les.LyricsExactSearchService.search = lyrics_search  # type: ignore[assignment]


def describe_config() -> Dict[str, Any]:
    from src.retrieval.gemini_listwise_reranker import GeminiListwiseRerankerConfig

    cfg = GeminiListwiseRerankerConfig.from_env()
    keys = ("ATTR_FILTER_WEIGHT", "ATTR_FILTER_GATE", "ATTR_FILTER_K", "ATTR_FILTER_USE_GENRE", "ATTR_FILTER_GENRE_POLICY",
            "QDRANT_URL", "QDRANT_PATH", "QDRANT_HYBRID_SCAN_LIMIT", "CONTEXT_WEIGHT", "SEARCH_REFERENCE_YEAR",
            "RERANKER_BACKEND", "RERANKER_ENABLED", "FEEDBACK_COLLECTION_ENABLED", "FEEDBACK_INDEX_WORKER_ENABLED",
            "FEEDBACK_RETRIEVAL_ENABLED", "LYRICS_CACHE_TTL_SECONDS", "SEARCH_WARMUP", "GEMINI_RERANK_TIMEOUT_SECONDS",
            "GEMINI_RERANK_BUDGET_SECONDS", "GEMINI_RERANK_MAX_RETRIES", "HF_HUB_OFFLINE", "SEARCH_TIMING_LOG",
            "MONGO_URI", "GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT_ID")
    secretish = {"MONGO_URI", "GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT_ID"}
    env = {}
    for k in keys:
        value = os.getenv(k)
        env[k] = ("set" if value else "unset") if k in secretish else (value if value is not None else "")
    return {
        "env": env,
        "reranker": {
            "model_name": cfg.model_name, "enabled": cfg.enabled, "passes": cfg.passes, "max_retries": cfg.max_retries,
            "max_candidates": cfg.max_candidates, "request_timeout_seconds": cfg.request_timeout_seconds,
            "time_budget_seconds": cfg.time_budget_seconds, "rerank_weight": cfg.rerank_weight,
            "use_search_grounding": cfg.use_search_grounding, "rare_fact_verification": cfg.rare_fact_verification,
        },
    }
