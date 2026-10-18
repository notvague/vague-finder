"""v72 작품명 해석 리랭킹 부분집합 진단 하네스 공용 — 구성(off|work) · main 코드 사본 · 서버 Qdrant · 외부 호출 차단 · 가사 스냅샷 · 기록 · 예산 통제.

v69 `v69_eval_runtime.py`(리뷰된 v68 사본)의 사본이다. 검색·기록·예산 연결 로직은 바꾸지 않았다. 바꾼 것:

1. **구성**: `V72_CONFIG=off`(작품명 해석 꺼짐 = 운영 기본 = A) | `work`(`WORK_NAME_RESOLUTION_ENABLED=1` ·
   `WORK_NAME_RESOLVER=cache:<resolution_cache/v71_rep1.json>` — v71 반복 1의 실제 해석 출력, 해석 호출 0 = B).
   **스위치 하나만 다르다.** 소리 ON(`AUDIO_FUSION_SCALE=1`) · Context 0.5 · D OFF · 피드백 OFF · 보조 가중 기본 0.5.
2. **코드 사본**: `artifacts/src_main_5c2d43e`(`git archive 5c2d43e src` — main = v71 PR #85 병합, src는 v70 `a55e498`과 같다).
   v69 사본(`bb7e6f0`)에는 작품명 해석 코드가 없다. 기록 대상 모듈에 `work_name_resolution`·`work_mentions`를 더했다.
3. **서버**: 전용 컨테이너 `vague-finder-qdrant-v72`(127.0.0.1:6380 — v71 저장소의 파일 복사본, 트리 지문 같음).
4. **해석기 보호**: `work`에서 해석기는 캐시뿐이어야 한다. 실제 해석 어댑터(`VertexWorkNameResolver.resolve`)는 막는다 —
   불리면 실행을 끊는다(승인 범위 밖 호출 방지).
5. 기록 파일 `v72_*.jsonl` · 환경변수 `V72_*`. 모델 `fake`(기본)·`live`·`fault`(자체 검사)는 v69 설명 그대로:
   fake = 결정적 가짜 응답·외부 연결 차단, live = 실제 Vertex(`v72_budget`이 HTTP 시도 단위로 막는다),
   fault = live와 같은 리랭커·예산 통제 경로 + SDK `_request_once` 가짜 전송(countTokens 성공, generate는 504|403).
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
CONFIG = os.environ.get("V72_CONFIG", "")
CODE = CONFIG  # 실행 이름·기록 필드(code)는 구성 이름을 쓴다
AUDIO = "on"  # v72는 소리 ON 고정(run_eval의 소리 실행 검사가 이 값을 본다)
MODE = os.environ.get("V72_MODE", "server")
# 모델: fake(기본, 결정적 가짜 응답·외부 연결 차단) | live(실제 Vertex, 예산 통제)
MODEL = os.environ.get("V72_MODEL", "fake")
if CONFIG not in {"off", "work"} or MODE != "server" or MODEL not in {"fake", "live", "fault"}:
    raise SystemExit("V72_CONFIG=off|work, V72_MODE=server (V72_MODEL=fake|live|fault) 가 필요하다")
COMMIT = "5c2d43ec09982f579db3b14b9767723c6571ae29"
COMMITS = {"off": COMMIT, "work": COMMIT}
SRC_TREE_SHA16 = "eaf159683c73e5d5"  # git ls-tree 5c2d43e src(.py) 지문 — 실행 전에 계산해 고정
CODE_ROOT = ROOT / "artifacts/src_main_5c2d43e"
V61_DIR = ROOT / "experiments/reranking/results_v61_hybrid_scan_audit"
V63_DIR = ROOT / "experiments/reranking/results_v63_hybrid_lyrics_validation"
# 워크트리 루트는 경로에서 뺀다 — 워크트리 `src/`가 사본 대신 잡히는 일을 없앤다(필요한 것은 사본·이 폴더·v63·v61뿐).
for p in list(sys.path):
    if p in ("", ".", str(ROOT)):
        sys.path.remove(p)
for p in reversed([str(CODE_ROOT), str(HERE), str(V63_DIR), str(V61_DIR)]):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
if "src" in sys.modules:
    raise SystemExit("src가 이미 import됐다 — v72_eval_runtime을 가장 먼저 import해야 한다")

SNAPSHOT_DIR = Path("/private/tmp/vague-finder-hybrid-lyrics-validation/artifacts/lyrics_snapshot_v63/snapshot")
QDRANT_SERVER = "http://127.0.0.1:6380"
CONTAINER = "vague-finder-qdrant-v72"
EXPECTED_INDEX_FINGERPRINT = "238299e6c8c17f2a"
EXPECTED_LYRICS = "e0d58f28f1f4a563"
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")
# 작품명 해석: v71 반복 1의 실제 출력(파일 복사본, Git 제외) · 언급 저장소 — 지문이 다르면 시작하지 않는다
WORK_CACHE = HERE / "resolution_cache/v71_rep1.json"
WORK_CACHE_SHA16 = "03ad094834c986e5"  # v71 resolution_cache/rep1.json과 같은 바이트
WORK_MENTIONS = ROOT / "artifacts/work_names/work_mentions_v1.json.gz"
WORK_MENTIONS_SHA16 = "bfd522ebe9da7590"
EXPECTED_WORK_VERSIONS = {"gate": "gate_v2", "validation": "validate_v3", "prompt": "resolver_prompt_v2",
                          "prompt_sha16": "31494c3f69f77481"}

FIXED_ENV: Dict[str, str] = {
    "CONTEXT_WEIGHT": "0.5", "ATTR_FILTER_WEIGHT": "0", "AUDIO_FUSION_SCALE": "1", "SEARCH_REFERENCE_YEAR": "2026",
    "RERANKER_BACKEND": "gemini_listwise", "GEMINI_RERANK_RARE_FACT_VERIFY": "0", "GEMINI_RETRIEVAL_BACKEND": "vertex",
    "GCP_LOCATION": "global", "GEMINI_MODEL_NAME": "gemini-3.1-flash-lite", "GCP_PROJECT_ID": "v72-offline-no-calls",
    "FEEDBACK_COLLECTION_ENABLED": "0", "FEEDBACK_INDEX_WORKER_ENABLED": "0", "FEEDBACK_RETRIEVAL_ENABLED": "0",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "WORK_MENTIONS_PATH": str(WORK_MENTIONS),
    "WORK_NAME_RESOLUTION_ENABLED": "1" if CONFIG == "work" else "0",
    **({"WORK_NAME_RESOLVER": "cache:" + str(WORK_CACHE)} if CONFIG == "work" else {}),
}
REMOVE_ENV = (
    "QDRANT_URL", "QDRANT_PATH", "QDRANT_HYBRID_SCAN_LIMIT", "CLAUDE_EFFORT", "CLAUDE_API", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS", "RERANKER_ENABLED", "GEMINI_RERANK_ENABLED", "GEMINI_RERANK_MODEL_NAME",
    "GEMINI_RERANK_TIMEOUT_SECONDS", "GEMINI_RERANK_BUDGET_SECONDS", "GEMINI_RERANK_MAX_RETRIES", "GEMINI_RERANK_PASSES",
    "GEMINI_RERANK_USE_SEARCH", "LYRICS_CACHE_TTL_SECONDS", "LYRICS_FUZZY_THRESHOLD", "MONGO_URI", "MONGO_DB_NAME",
    "ATTR_FILTER_GATE", "ATTR_FILTER_K", "ATTR_FILTER_USE_GENRE", "ATTR_FILTER_GENRE_POLICY", "VF_CLAUDE_EFFORT",
    "VF_CLAUDE_THINKING", "WORK_NAME_RESOLUTION_ENABLED", "WORK_NAME_RESOLVER", "WORK_NAME_AUX_WEIGHT",
    "WORK_NAME_RESOLVER_TIMEOUT_SECONDS", "WORK_MENTIONS_PATH",
)


LIVE_CREDENTIALS_PRESENT = False


def _file_sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def check_work_inputs() -> Dict[str, Any]:
    """해석 캐시·언급 저장소 바이트 지문(src import 전, 두 구성 모두 같은 저장소를 쓴다 — off는 읽지 않지만 같은 파일을 가리킨다)."""
    out = {"mentions_sha16": _file_sha16(WORK_MENTIONS)}
    if out["mentions_sha16"] != WORK_MENTIONS_SHA16:
        raise SystemExit(f"언급 저장소 지문 {out['mentions_sha16']} ≠ {WORK_MENTIONS_SHA16}")
    if CONFIG == "work":
        out["cache_sha16"] = _file_sha16(WORK_CACHE)
        if out["cache_sha16"] != WORK_CACHE_SHA16:
            raise SystemExit(f"해석 캐시 지문 {out['cache_sha16']} ≠ {WORK_CACHE_SHA16}")
    return out


def fixed_env() -> Dict[str, str]:
    keep = {k: os.environ[k] for k in ("GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS") if MODEL == "live" and os.environ.get(k)}
    for key in REMOVE_ENV:
        os.environ.pop(key, None)
    applied = dict(FIXED_ENV)
    if MODEL == "live":
        # 실제 호출: with_env.py가 운영 .env에서 이 두 키만 넘긴다. 없으면 run_eval이 게이트 확인 뒤 거부한다(값은 기록하지 않는다)
        global LIVE_CREDENTIALS_PRESENT
        LIVE_CREDENTIALS_PRESENT = set(keep) == {"GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"}
        applied.pop("GCP_PROJECT_ID")
        os.environ.update(keep)
    applied["QDRANT_URL"] = QDRANT_SERVER
    os.environ.update(applied)
    return applied


WORK_INPUTS = check_work_inputs()


fixed_env()  # src import 전에 — 모듈 상수가 import 때 환경을 읽는다


class BlockedExternalCall(RuntimeError):
    pass


# gemini_listwise_reranker.rerank_run의 재시도 정정 지시(두 커밋에서 같은 문구)
RETRY_CORRECTION = ("\n\nYour previous output was invalid. Return strict JSON only. "
                    "Use only candidate ids from the supplied list, exactly once each.")


class RunAborted(BaseException):
    """예산 장부 정지 — 리랭커의 `except Exception`이 삼키지 않도록 BaseException으로 실행을 끊는다."""


def _blocked(*_a: Any, **_k: Any) -> Any:
    raise BlockedExternalCall("v72: 외부 클라이언트 생성이 차단됐다")


class _BlockedClient:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise BlockedExternalCall(f"v72: {type(self).__name__} 생성이 차단됐다")


NET_ATTEMPTS: List[str] = []


WORK_RESOLVER_CALLS: List[str] = []


def _install_work_resolver_block() -> None:
    """v72: 실제 작품명 해석 어댑터는 어떤 구성에서도 부르지 않는다(work는 캐시 해석기만). 불리면 실행을 끊는다."""
    import src.retrieval.work_name_resolution as wnr

    def blocked_resolve(self, query, *, timeout_seconds=0.0):  # type: ignore[no-untyped-def]
        WORK_RESOLVER_CALLS.append(type(self).__name__)
        raise RunAborted("v72: 실제 작품명 해석 호출이 시도됐다 — 승인 범위 밖")

    wnr.VertexWorkNameResolver.resolve = blocked_resolve  # type: ignore[assignment]


def install_guards() -> None:
    """Mongo·Claude·가사 Mongo는 항상 막는다. fake면 루프백 밖 네트워크와 Gemini 클라이언트도 막는다(live는 예산 통제로 대신)."""
    _install_always_blocked()
    _install_work_resolver_block()
    if MODEL == "fake":
        _install_fake_only_blocks()
    elif MODEL == "fault":
        _install_network_block()
        import src.common.gemini_client as gc
        from google.auth.credentials import AnonymousCredentials

        gc._load_default_credentials = lambda: AnonymousCredentials()  # type: ignore[assignment]


FAULT_PLANS = ("504", "403")


def install_fault_transport() -> Dict[str, int]:
    """fault 전용: SDK가 HTTP를 한 번 보내는 자리를 가짜 전송으로 바꾼다(예산 통제가 이것을 다시 감싼다). 네트워크 없음."""
    import google.genai._api_client as api
    from google.genai import errors

    plan = os.environ.get("V72_FAULT_PLAN", "")
    if MODEL != "fault" or plan not in FAULT_PLANS:
        raise SystemExit(f"fault 전송은 V72_MODEL=fault · V72_FAULT_PLAN={'|'.join(FAULT_PLANS)}에서만")
    seen = {"count_tokens": 0, "generate": 0}

    def fake_once(client_self, http_request, stream=False):  # type: ignore[no-untyped-def]
        url = str(http_request.url)
        if ":countTokens" in url:
            seen["count_tokens"] += 1
            return api.HttpResponse({}, [json.dumps({"totalTokens": 1})])
        seen["generate"] += 1
        if plan == "504":
            raise errors.ServerError(504, {"error": {"code": 504, "message": "v72 fault: deadline", "status": "DEADLINE_EXCEEDED"}})
        raise errors.ClientError(403, {"error": {"code": 403, "message": "v72 fault: permission denied", "status": "PERMISSION_DENIED"}})

    async def fake_async_once(client_self, http_request, stream=False):  # type: ignore[no-untyped-def]
        return fake_once(client_self, http_request, stream)

    api.BaseApiClient._request_once = fake_once  # type: ignore[assignment]
    api.BaseApiClient._async_request_once = fake_async_once  # type: ignore[assignment]
    return seen


def _install_network_block() -> None:
    import socket

    orig_connect = socket.socket.connect

    def connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0] if isinstance(address, tuple) else address
            if not (str(host).startswith("127.") or str(host) in {"::1", "localhost"}):
                NET_ATTEMPTS.append(f"{host}:{address[1] if isinstance(address, tuple) else ''}")
                raise OSError(f"v72: 외부 연결 차단 {host}")
        return orig_connect(self, address)

    socket.socket.connect = connect  # type: ignore[assignment]
    orig_gai = socket.getaddrinfo

    def gai(host, *a, **k):  # type: ignore[no-untyped-def]
        h = host.decode() if isinstance(host, bytes) else host
        if h is not None and not (str(h).startswith("127.") or str(h) in {"::1", "localhost"}):
            NET_ATTEMPTS.append(f"dns:{h}")
            raise OSError(f"v72: 외부 이름 조회 차단 {h}")
        return orig_gai(host, *a, **k)

    socket.getaddrinfo = gai  # type: ignore[assignment]


def _install_fake_only_blocks() -> None:
    _install_network_block()
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
    if MODEL in ("fake", "fault"):
        before = len(NET_ATTEMPTS)
        try:
            socket.create_connection(("192.0.2.1", 9), timeout=0.5)
            out["network"] = "NOT BLOCKED"
        except OSError as exc:
            out["network"] = "blocked" if "v72" in str(exc) else f"NOT BLOCKED ({type(exc).__name__})"
        del NET_ATTEMPTS[before:]
    out["QDRANT_HYBRID_SCAN_LIMIT"] = "unset" if not os.getenv("QDRANT_HYBRID_SCAN_LIMIT") else "SET"
    bad = {k: v for k, v in out.items() if v not in {"blocked", "unset"}}
    if bad:
        raise RuntimeError(f"v72 차단 확인 실패: {bad}")
    return out


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _commit_blob(commit: str, rel: str) -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", f"{commit}:{rel}"], capture_output=True, text=True, cwd=ROOT,
                          check=True).stdout.strip()


def code_identity() -> Dict[str, Any]:
    """실제로 import된 동점 관련 세 모듈과 평가기·라우터의 파일·blob 해시. 기준 커밋의 blob과 다르거나 사본 밖이면 멈춘다.

    추가로 사본 전체(`src/**/*.py`)의 blob 목록 지문을 남기고, 기준 커밋의 `git ls-tree` 지문과 같은지 본다.
    """
    import subprocess

    import src.vector_db.qdrant_backend as qb
    import src.retrieval.context_qdrant_search as cqs
    import src.feedback.case_index as ci
    import src.retrieval.evaluate_search_accuracy as esa
    import src.retrieval.search_router as sr
    import src.retrieval.work_name_resolution as wnr
    import src.retrieval.work_mentions as wm
    import src.retrieval.gemini_listwise_reranker as glr

    commit = COMMITS[CODE]
    files = {}
    for mod in (qb, cqs, ci, esa, sr, wnr, wm, glr):
        path = Path(mod.__file__).resolve()
        if not str(path).startswith(str(CODE_ROOT.resolve()) + "/"):
            raise RuntimeError(f"{mod.__name__}가 코드 판 사본 밖에서 import됐다: {path}")
        rel = str(path.relative_to(CODE_ROOT.resolve()))
        blob = _git_blob(path.read_bytes())
        expected = _commit_blob(commit, rel)
        if blob != expected:
            raise RuntimeError(f"{rel}: 사본 blob {blob[:12]} ≠ {commit[:7]}의 blob {expected[:12]}")
        files[mod.__name__] = {"path": str(path.relative_to(ROOT)), "git_blob": blob}
    # 사본 전체 = 커밋의 src 트리(파이썬 파일만, __pycache__ 제외)
    tree = subprocess.run(["git", "ls-tree", "-r", commit, "src"], capture_output=True, text=True, cwd=ROOT,
                          check=True).stdout.splitlines()
    want = sorted((line.split("\t")[1], line.split()[2]) for line in tree if line.endswith(".py"))
    have = sorted((str(p.relative_to(CODE_ROOT)), _git_blob(p.read_bytes()))
                  for p in (CODE_ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    if want != have:
        raise RuntimeError(f"코드 판 사본이 {commit[:7]}의 src 트리와 다르다")
    tree_sha = hashlib.sha256(json.dumps(want).encode()).hexdigest()[:16]
    if tree_sha != SRC_TREE_SHA16:
        raise RuntimeError(f"사본 트리 지문 {tree_sha} ≠ 고정값 {SRC_TREE_SHA16}")
    return {"code": CODE, "commit": commit, "code_root": str(CODE_ROOT.relative_to(ROOT)), "files": files,
            "src_py_files": len(want), "src_py_tree_sha16": tree_sha}


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
        self.clap: Dict[str, Any] = {"load_calls": 0, "load_ms": [], "embed_calls": 0}

    def row(self) -> Dict[str, Any]:
        qid = self.current or "_none"
        return self.rows.setdefault(qid, {"query_id": qid, "lyrics": [], "rerank_runs": [], "generates": [],
                                          "qdrant_seqs": [], "protection": None, "final": None,
                                          "clap_embed": [], "audio_queries": 0, "path_failures": []})

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

        # --- CLAP: 로드(첫 임베딩 때 지연 로드)와 텍스트 임베딩 호출. `model`은 load()를 부르는 property라 내부 필드로 본다(v67 재귀 수정)
        from src.embedding.models.audio_clap import CLAPAudioEmbedder

        orig_load, orig_embed = CLAPAudioEmbedder.load, CLAPAudioEmbedder.embed_texts

        def clap_load(emb_self):  # type: ignore[no-untyped-def]
            if emb_self._model is not None and emb_self._processor is not None:
                return orig_load(emb_self)
            started = time.perf_counter()
            out = orig_load(emb_self)
            with rec.lock:
                rec.clap["load_calls"] += 1
                rec.clap["load_ms"].append(round((time.perf_counter() - started) * 1000, 3))
            return out

        def clap_embed(emb_self, texts, *, l2_normalize=True):  # type: ignore[no-untyped-def]
            started = time.perf_counter()
            out = orig_embed(emb_self, texts, l2_normalize=l2_normalize)
            with rec.lock:
                rec.clap["embed_calls"] += 1
            rec.row()["clap_embed"].append({"texts": len(texts), "ms": round((time.perf_counter() - started) * 1000, 3)})
            return out

        CLAPAudioEmbedder.load = clap_load  # type: ignore[assignment]
        CLAPAudioEmbedder.embed_texts = clap_embed  # type: ignore[assignment]

        # --- 경로 실패: 라우터가 예외를 빈 결과로 바꾸기 전에 센다
        orig_path_result = sr._path_result

        def path_result(name, result, recorder):  # type: ignore[no-untyped-def]
            if isinstance(result, BaseException):
                rec.row()["path_failures"].append(f"{name}: {type(result).__name__}: {str(result)[:160]}")
            return orig_path_result(name, result, recorder)

        sr._path_result = path_result  # type: ignore[assignment]

        # 자체 검사 전용(fake만): 소리 경로를 일부러 실패시켜, 실패가 빈 결과로 조용히 넘어가지 않고 실행이 멈추는지 본다
        if MODEL == "fake" and os.environ.get("V72_PATH_FAIL_INJECT") == "audio":
            def broken_audio(router_self, analysis, top_k):  # type: ignore[no-untyped-def]
                raise RuntimeError("v72 자체 검사: 소리 경로 실패 주입")

            sr.SearchRouter._search_audio = broken_audio  # type: ignore[assignment]

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
            if "audio" in collection:
                rec.row()["audio_queries"] += 1
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
            # 이 질의 안 그룹 순번(rerank_runs는 끝난 뒤 붙으므로 지금 길이가 이 그룹의 번호다)과 그룹 안 시도 순번
            rec.local.group_index = len(rec.row()["rerank_runs"])
            rec.local.group_attempt = 0
            started = time.perf_counter()
            run = orig_run(reranker_self, query_text, tracks, top_k, answers, deadline=deadline)
            mixes = {sid: _round(dataclasses.asdict(m)) for sid, m in (run.mixes or {}).items()}
            rec.row()["rerank_runs"].append({
                "input_ids": [str(t.id) for t in tracks], "input_scores": [round(float(t.score or 0.0), 9) for t in tracks],
                "input_lyric": [t.lyric_match_type or "" for t in tracks], "top_k": int(top_k), "status": run.status,
                "model_calls": run.model_calls, "output_ids": [str(t.id) for t in run.tracks],
                "mixes_sha16": sha16(json.dumps(mixes, sort_keys=True)),
                "ms": round((time.perf_counter() - started) * 1000, 3),
                "error": (getattr(run, "error", "") or "")[:200] if hasattr(run, "error") else "",
            })
            return run

        glr.GeminiListwiseReranker.rerank_run = rerank_run  # type: ignore[assignment]

        orig_generate = glr.GeminiListwiseReranker._generate

        def _entry(contents: str, config: Any) -> Dict[str, Any]:
            # 재시도는 리랭커가 정정 지시(RETRY_CORRECTION)를 끝에 붙여 보낸다 — 떼어 낸 본문 sha로 "같은 입력의 재시도"인지 본다
            corrected = contents.endswith(RETRY_CORRECTION)
            base = contents[: -len(RETRY_CORRECTION)] if corrected else contents
            entry = {"group": getattr(rec.local, "group_index", None), "attempt": getattr(rec.local, "group_attempt", 0),
                     "contents_sha16": sha16(contents), "base_sha16": sha16(base), "correction": corrected,
                     "contents_chars": len(contents),
                     "config_sha16": sha16(json.dumps(config, sort_keys=True, default=str))}
            rec.local.group_attempt = entry["attempt"] + 1
            return entry

        def _guard_halted() -> Optional[str]:
            budget = sys.modules.get("v72_budget")
            if budget is not None and budget.GUARD is not None and budget.GUARD.halted:
                return budget.GUARD.halt_reason
            return None

        def generate_live(reranker_self, *, contents, config, deadline, tally=None):
            reason = _guard_halted()
            if reason:
                raise RunAborted(f"예산 통제 정지: {reason}")
            entry = _entry(contents, config)
            started = time.perf_counter()
            try:
                response = orig_generate(reranker_self, contents=contents, config=config, deadline=deadline, tally=tally)
            except Exception as exc:
                entry.update({"error": type(exc).__name__, "error_code": getattr(exc, "code", None),
                              "ms": round((time.perf_counter() - started) * 1000, 3)})
                rec.row()["generates"].append(entry)
                reason = _guard_halted()
                if reason:
                    raise RunAborted(f"예산 통제 정지: {reason}") from exc
                raise
            entry.update({"ms": round((time.perf_counter() - started) * 1000, 3),
                          "response_text_sha16": sha16(getattr(response, "text", "") or ""),
                          "model_version": getattr(response, "model_version", None)})
            rec.row()["generates"].append(entry)
            reason = _guard_halted()
            if reason:
                raise RunAborted(f"예산 통제 정지: {reason}")
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
            rec.row()["generates"].append(_entry(contents, config))
            return SimpleNamespace(text=json.dumps(payload))

        glr.GeminiListwiseReranker._generate = generate if MODEL == "fake" else generate_live  # type: ignore[assignment]  # live·fault

    def save(self, out_dir: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "v72_queries.jsonl").open("w", encoding="utf-8") as fh:
            for row in self.rows.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out_dir / "v72_qdrant_calls.jsonl").open("w", encoding="utf-8") as fh:
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
            "audio_collection_queries": sum(r["audio_queries"] for r in self.rows.values()),
            "queries_with_clap_embed": sum(1 for r in self.rows.values() if r["clap_embed"]),
            "path_failures": [f"{q}:{f}" for q, r in self.rows.items() for f in r["path_failures"]],
            "offline_generate_calls": self.offline_generate_calls,
        }
