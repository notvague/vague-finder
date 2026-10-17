"""v63 공용 — 실행 조건 고정 · 외부 호출 통제 · 가사 스냅샷 자료원 · A/B 하이브리드 구현 · 기록.

서비스 코드는 고치지 않는다. 이 모듈이 프로세스 안에서 바꾸는 자리는 아래뿐이다.

1. **가사 자료원** — `LyricsExactSearchService._source_documents`를 고정 스냅샷(`artifacts/lyrics_snapshot_v63/snapshot`)
   읽기로 바꾼다. 검색 알고리즘·판 수명(TTL)·보호 규칙은 그대로다. 평가기의 가사 지문(`_lyrics_fingerprint`)도 같은
   스냅샷에서 같은 식으로 계산한다(Mongo를 다시 읽지 않는다). Mongo 클라이언트 생성은 항상 막는다.
2. **하이브리드 조회 구현(arm)** — B는 지금 디스크의 `QdrantIndex.query`(v62, 요청별 count 전수 조회) 그대로,
   A는 v61 기준 커밋 `fc20e86`의 `QdrantIndex.query` 함수(모듈 상수 `HYBRID_SCAN_LIMIT`=5000)를 같은 클래스에 꽂는다.
   두 함수의 차이는 하이브리드 분기의 `limit`뿐이다(`git diff fc20e86 f43daf7 -- src/vector_db/qdrant_backend.py`).
   운영 코드에 옛 상한 스위치를 넣지 않는다.
3. **기록** — 하이브리드 호출마다 독립 정확 합산(`exact_hybrid.ExactHybridReference`, v61)과 대조, 가사 검색 결과·예외,
   리랭킹 직전 입력(보호 판정·tier·그룹별 입력 순서·점수)·프롬프트 sha256·모델 응답 상태·점수 합성값.
4. **모델 호출** — offline: Gemini 클라이언트 생성을 막고 `GeminiListwiseReranker._generate`를 **결정적 가짜 응답**
   (입력 순서 그대로의 정상 JSON)으로 바꾼다 — 프롬프트는 실제로 만들어지고 지문만 남긴다. live: 실제 Vertex 호출,
   `v63_budget`이 HTTP 시도 단위로 예약·정산·한도를 건다.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V61_DIR = ROOT / "experiments/reranking/results_v61_hybrid_scan_audit"
for p in (str(ROOT), str(HERE), str(V61_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

LEGACY_COMMIT = "fc20e86"
SNAPSHOT_DIR = ROOT / "artifacts/lyrics_snapshot_v63/snapshot"
QDRANT_COPY = "artifacts/qdrant_v63"
EXPECTED_INDEX_FINGERPRINT = "238299e6c8c17f2a"   # v61·v62 복사본과 같은 내용이어야 한다
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")
EXPECTED_ROWS = {"v06": 57, "v09": 59, "v10": 60}
LYRIC_QUERY_IDS = (
    "q209", "q211", "q213", "q219", "n007", "n009", "n013", "n017", "n039", "n064",
    "t003", "t016", "t018", "t025", "t032", "t060", "t063", "t105", "t110",
)
TRACE_QUERY_IDS = ("n017", "t003", "t016", "t018")

FIXED_ENV: Dict[str, str] = {
    "QDRANT_PATH": QDRANT_COPY,
    "CONTEXT_WEIGHT": "0.5",
    "ATTR_FILTER_WEIGHT": "0",
    "AUDIO_FUSION_SCALE": "1",
    "SEARCH_REFERENCE_YEAR": "2026",
    "RERANKER_BACKEND": "gemini_listwise",
    "GEMINI_RERANK_RARE_FACT_VERIFY": "0",
    "GEMINI_RETRIEVAL_BACKEND": "vertex",
    "GCP_LOCATION": "global",
    "GEMINI_MODEL_NAME": "gemini-3.1-flash-lite",
    "FEEDBACK_COLLECTION_ENABLED": "0",
    "FEEDBACK_INDEX_WORKER_ENABLED": "0",
    "FEEDBACK_RETRIEVAL_ENABLED": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
# 실행에 영향을 줄 수 있는데 이 실험이 정하지 않는 값은 지운다(리랭커·검색 스위치는 코드 기본값)
REMOVE_ENV = (
    "QDRANT_URL", "QDRANT_HYBRID_SCAN_LIMIT", "CLAUDE_EFFORT", "CLAUDE_API", "GEMINI_API_KEY",
    "RERANKER_ENABLED", "GEMINI_RERANK_ENABLED", "GEMINI_RERANK_MODEL_NAME", "GEMINI_RERANK_TIMEOUT_SECONDS",
    "GEMINI_RERANK_BUDGET_SECONDS", "GEMINI_RERANK_MAX_RETRIES", "GEMINI_RERANK_PASSES", "GEMINI_RERANK_USE_SEARCH",
    "LYRICS_CACHE_TTL_SECONDS", "LYRICS_FUZZY_THRESHOLD", "MONGO_URI", "MONGO_DB_NAME",
    "ATTR_FILTER_GATE", "ATTR_FILTER_K", "ATTR_FILTER_USE_GENRE", "ATTR_FILTER_GENRE_POLICY",
)


class BlockedExternalCall(RuntimeError):
    pass


class RunAborted(RuntimeError):
    """예산·호출 한도·인증 오류로 실행을 끊는다 — 완료 표식 없이 끝난다."""


def _blocked(*_a: Any, **_k: Any) -> Any:
    raise BlockedExternalCall("v63: 외부 클라이언트 생성이 차단됐다")


class _BlockedClient:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise BlockedExternalCall(f"v63: {type(self).__name__} 생성이 차단됐다")


def fixed_env(mode: str) -> Dict[str, str]:
    for key in REMOVE_ENV:
        os.environ.pop(key, None)
    applied = dict(FIXED_ENV)
    if mode == "offline":
        # 리랭커 클래스(gemini_listwise)를 live와 같게 만들기 위한 가짜 프로젝트. 클라이언트 생성은 막혀 있다(v62 교훈)
        applied["GCP_PROJECT_ID"] = "v63-offline-no-calls"
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    else:
        for key in ("GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"):
            if not os.getenv(key):
                raise SystemExit(f"live 실행에 {key}가 없다(with_env.py로 넘긴다)")
    for key, value in applied.items():
        os.environ[key] = value
    shown = dict(applied)
    if mode == "live":
        shown["GCP_PROJECT_ID"] = (os.environ["GCP_PROJECT_ID"][:12] + "…")
        shown["GOOGLE_APPLICATION_CREDENTIALS"] = "set"
    return shown


def install_guards(mode: str) -> None:
    """Mongo·Claude는 항상, Gemini는 offline에서 막는다. 서비스 모듈 import 전에 부른다."""
    import pymongo

    pymongo.MongoClient = _BlockedClient  # type: ignore[attr-defined]
    import src.common.mongodb as mongodb

    mongodb.MongoClient = _BlockedClient  # type: ignore[attr-defined]
    mongodb._connect = _blocked  # type: ignore[assignment]
    try:
        import anthropic

        anthropic.Anthropic = _BlockedClient  # type: ignore[attr-defined]
        anthropic.AsyncAnthropic = _BlockedClient  # type: ignore[attr-defined]
    except ImportError:
        pass
    if mode == "offline":
        import src.common.gemini_client as gc
        from google import genai
        import google.genai.client as genai_client_mod

        gc.make_genai_client = _blocked  # type: ignore[assignment]
        genai.Client = _BlockedClient  # type: ignore[attr-defined]
        genai_client_mod.Client = _BlockedClient  # type: ignore[attr-defined]
        # 이미 이 이름을 import한 모듈도 막는다
        import src.retrieval.gemini_listwise_reranker as glr

        glr.make_genai_client = _blocked  # type: ignore[assignment]


def verify_guards(mode: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    import pymongo

    try:
        pymongo.MongoClient("mongodb://127.0.0.1:1")
        out["pymongo.MongoClient"] = "NOT BLOCKED"
    except BlockedExternalCall:
        out["pymongo.MongoClient"] = "blocked"
    import src.common.mongodb as mongodb

    try:
        mongodb.get_collection("songs")
        out["src.common.mongodb"] = "NOT BLOCKED"
    except BlockedExternalCall:
        out["src.common.mongodb"] = "blocked"
    import src.retrieval.lyrics_exact_search as les

    try:
        les._make_lyrics_mongo_client(1.0)
        out["lyrics_mongo_client"] = "NOT BLOCKED"
    except BlockedExternalCall:
        out["lyrics_mongo_client"] = "blocked"
    if mode == "offline":
        import src.common.gemini_client as gc
        import src.retrieval.gemini_listwise_reranker as glr

        for name, fn in (("make_genai_client", gc.make_genai_client), ("reranker.make_genai_client", glr.make_genai_client)):
            try:
                fn(purpose="retrieval")
                out[name] = "NOT BLOCKED"
            except BlockedExternalCall:
                out[name] = "blocked"
        try:
            from google import genai

            genai.Client(vertexai=True, project="x", location="global")
            out["google.genai.Client"] = "NOT BLOCKED"
        except BlockedExternalCall:
            out["google.genai.Client"] = "blocked"
    out["QDRANT_URL"] = "unset" if not os.getenv("QDRANT_URL") else "SET"
    out["QDRANT_HYBRID_SCAN_LIMIT"] = "unset" if not os.getenv("QDRANT_HYBRID_SCAN_LIMIT") else "SET"
    try:
        import aiohttp  # noqa: F401

        out["aiohttp"] = "INSTALLED"  # SDK가 연결 오류에서 내부 재시도 1회를 한다 — 시도 수 집계가 HTTP 계층과 어긋남
    except ImportError:
        out["aiohttp"] = "absent"
    bad = {k: v for k, v in out.items() if v not in {"blocked", "unset", "absent"}}
    if bad:
        raise RuntimeError(f"v63 차단 확인 실패: {bad}")
    return out


# ---------------------------------------------------------------------- 가사 스냅샷

def snapshot_manifest() -> Dict[str, Any]:
    if not (SNAPSHOT_DIR / "COMPLETE").exists():
        raise RuntimeError(f"가사 스냅샷이 게시되지 않았다: {SNAPSHOT_DIR}")
    manifest = json.loads((SNAPSHOT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("consistent_before_full_after"):
        raise RuntimeError("가사 스냅샷 manifest가 완전하지 않다")
    return manifest


def load_snapshot_documents() -> List[Dict[str, Any]]:
    manifest = snapshot_manifest()
    path = SNAPSHOT_DIR / manifest["file"]["name"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest()[:16] != manifest["file"]["sha256_16"]:
        raise RuntimeError("가사 스냅샷 파일 sha가 manifest와 다르다")
    documents = pickle.loads(gzip.decompress(data))
    if len(documents) != manifest["documents"]:
        raise RuntimeError("가사 스냅샷 문서 수가 manifest와 다르다")
    return documents


class LyricsSnapshotSource:
    """`LyricsExactSearchService._source_documents` 자리. 같은 문서 목록을 매번 새 반복자로 돌려준다."""

    def __init__(self, documents: List[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
        self.documents = documents
        self.manifest = manifest
        self.reads = 0
        self.lock = threading.Lock()

    def __call__(self, service_self: Any) -> Any:
        with self.lock:
            self.reads += 1
        return iter(list(self.documents))

    def fingerprint(self) -> Dict[str, Any]:
        """평가기 `_lyrics_fingerprint`와 같은 계산 — 스냅샷에서."""
        from snapshot_lyrics import content_digest

        pairs = [
            (str(d.get("song_id") or ""), str((d.get("lyrics_data") or {}).get("full_lyrics") or ""))
            for d in self.documents
        ]
        info = {"db": self.manifest.get("db_name"), "collection": "songs", **content_digest(pairs)}
        info["source"] = "v63_snapshot"
        info["snapshot_full_document_sha256_16"] = self.manifest.get("full_document_sha256_16")
        info["snapshot_file_sha256_16"] = self.manifest["file"]["sha256_16"]
        info["snapshot_finished_at"] = self.manifest.get("finished_at")
        return info


def install_lyrics_snapshot() -> LyricsSnapshotSource:
    import src.retrieval.lyrics_exact_search as les
    from src.retrieval import evaluate_search_accuracy as esa

    source = LyricsSnapshotSource(load_snapshot_documents(), snapshot_manifest())
    les.LyricsExactSearchService._source_documents = lambda self: source(self)  # type: ignore[assignment]
    fp = source.fingerprint()
    if fp["content_sha256_16"] != source.manifest["content_from_full_read"]["content_sha256_16"]:
        raise RuntimeError("스냅샷에서 다시 계산한 가사 지문이 manifest와 다르다")
    esa._lyrics_fingerprint = lambda: dict(fp)  # type: ignore[assignment]
    return source


# ---------------------------------------------------------------------- A/B 구현

def legacy_backend_module(work_dir: Path):
    src = subprocess.run(
        ["git", "show", f"{LEGACY_COMMIT}:src/vector_db/qdrant_backend.py"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if src.returncode != 0:
        raise RuntimeError(f"git show 실패: {src.stderr.strip()}")
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "qdrant_backend_legacy_fc20e86.py"
    path.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("qdrant_backend_legacy_fc20e86", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, hashlib.sha256(src.stdout.encode("utf-8")).hexdigest()[:16]


def install_arm(arm: str, work_dir: Path) -> Dict[str, Any]:
    from src.vector_db import qdrant_backend as qb

    current_src = (ROOT / "src/vector_db/qdrant_backend.py").read_bytes()
    info: Dict[str, Any] = {
        "arm": arm,
        "current_backend_sha256_16": hashlib.sha256(current_src).hexdigest()[:16],
        "current_hybrid_scan_limit_const": qb.HYBRID_SCAN_LIMIT,
    }
    if arm == "B":
        info["query_impl"] = "src.vector_db.qdrant_backend.QdrantIndex.query (v62, limit=count(exact))"
        return info
    legacy, sha = legacy_backend_module(work_dir)
    if legacy.HYBRID_SCAN_LIMIT != 5000:
        raise RuntimeError(f"옛 구현 상한이 5000이 아니다: {legacy.HYBRID_SCAN_LIMIT}")
    qb.QdrantIndex.query = legacy.QdrantIndex.query  # type: ignore[assignment]
    info.update({
        "query_impl": f"{LEGACY_COMMIT}:src/vector_db/qdrant_backend.py QdrantIndex.query (HYBRID_SCAN_LIMIT=5000)",
        "legacy_backend_sha256_16": sha,
        "legacy_hybrid_scan_limit": legacy.HYBRID_SCAN_LIMIT,
    })
    return info


# ---------------------------------------------------------------------- 기록

def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Recorder:
    """질의 단위 기록. 평가기는 질의를 순차로 돌므로 '현재 질의'를 라우터 search 진입 때 정한다."""

    def __init__(self, mode: str, reference: Any) -> None:
        self.mode = mode
        self.reference = reference
        self.lock = threading.Lock()
        self.query_ids: Dict[str, str] = {}
        self.current: Optional[str] = None
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.hybrid_calls: List[Dict[str, Any]] = []
        self.local = threading.local()
        self.seq = 0
        self.lyrics_loads: List[Dict[str, Any]] = []
        self.offline_generate_calls = 0
        self._filter_cache: Dict[str, Any] = {}

    def row(self) -> Dict[str, Any]:
        qid = self.current or "_none"
        return self.rows.setdefault(qid, {"query_id": qid, "lyrics": [], "rerank_runs": [], "generates": [],
                                          "hybrid_seqs": [], "protection": None, "final": None})

    def install(self) -> None:
        rec = self
        from src.retrieval import search_router as sr
        from src.retrieval import evaluate_search_accuracy as esa
        import src.retrieval.lyrics_exact_search as les
        import src.retrieval.gemini_listwise_reranker as glr
        from src.vector_db import qdrant_backend as qb
        from exact_hybrid import allowed_point_ids, compare_rankings

        # --- 라우터 search: 현재 질의 · 후보 30개
        orig_router_search = sr.SearchRouter.search

        async def router_search(router_self, analysis, *args, **kwargs):
            text = getattr(analysis, "original_query", None)
            rec.current = rec.query_ids.get(text)
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

        # --- 하이브리드 조회: 독립 기준 대조(두 arm 모두)
        orig_query = qb.QdrantIndex.query

        def query(index_self, *, vector, top_k=10, include_metadata=True, namespace=qb.NAMESPACE,
                  filter=None, sparse_vector=None, **kw):  # noqa: A002
            if sparse_vector is None:
                return orig_query(index_self, vector=vector, top_k=top_k, include_metadata=include_metadata,
                                  namespace=namespace, filter=filter, sparse_vector=None, **kw)
            started = time.perf_counter()
            result = orig_query(index_self, vector=vector, top_k=top_k, include_metadata=include_metadata,
                                namespace=namespace, filter=filter, sparse_vector=sparse_vector, **kw)
            elapsed = (time.perf_counter() - started) * 1000
            allowed = None
            if filter:
                key = json.dumps(filter, ensure_ascii=False, sort_keys=True)
                allowed = rec._filter_cache.get(key)
                if allowed is None:
                    collection = qb.collection_name(index_self._index_name, namespace)
                    allowed = allowed_point_ids(index_self._client, collection, qb.translate_filter(filter))
                    rec._filter_cache[key] = allowed
            expected = rec.reference.top(vector, sparse_vector, int(top_k), allowed_point_ids=allowed)
            actual = [{"id": str(m["id"]), "score": float(m["score"])} for m in result["matches"]]
            cmp = compare_rankings(actual, [{"id": e["id"], "score": e["score"]} for e in expected])
            with rec.lock:
                rec.seq += 1
                call = {
                    "seq": rec.seq, "query_id": rec.current, "top_k": int(top_k), "filtered": bool(filter),
                    "filter_allowed": len(allowed) if allowed is not None else None,
                    "elapsed_ms": round(elapsed, 3),
                    "result_ids": [a["id"] for a in actual],
                    "result_scores": [round(a["score"], 9) for a in actual],
                    "ref_identical_ids": cmp["identical_ids"], "ref_same_set": cmp["same_set"],
                    "ref_missing": cmp["missing"], "ref_extra": cmp["extra"],
                    "ref_score_mismatch": _round(cmp["score_mismatch"]),
                    "ref_order_differs_beyond_ties": cmp["order_differs_beyond_ties"],
                    "ref_max_abs_delta": cmp["max_abs_score_delta_common"],
                }
                rec.hybrid_calls.append(call)
                rec.row()["hybrid_seqs"].append(rec.seq)
            return result

        qb.QdrantIndex.query = query  # type: ignore[assignment]

        # --- 가사 검색: 정상 0건과 자료원 실패 구분
        orig_lyrics_search = les.LyricsExactSearchService.search

        def lyrics_search(service_self, clues, top_k=100, snapshot_out=None):
            started = time.perf_counter()
            entry: Dict[str, Any] = {"clues": len(clues)}
            try:
                result = orig_lyrics_search(service_self, clues, top_k=top_k, snapshot_out=snapshot_out)
            except BaseException as exc:
                entry["error"] = type(exc).__name__
                entry["ms"] = round((time.perf_counter() - started) * 1000, 3)
                rec.row()["lyrics"].append(entry)
                raise
            entry.update({
                "ms": round((time.perf_counter() - started) * 1000, 3),
                "results": len(result),
                "top": [{"id": str(t.id), "type": t.lyric_match_type, "score": round(float(t.lyric_match_score or 0), 9),
                         "phrase_sha": sha16((t.lyric_match_detail.matched_phrase if t.lyric_match_detail else "") or ""),
                         "corpus_count": t.lyric_match_detail.corpus_match_count if t.lyric_match_detail else None}
                        for t in result[:30]],
            })
            rec.row()["lyrics"].append(entry)
            return result

        les.LyricsExactSearchService.search = lyrics_search  # type: ignore[assignment]

        orig_build = les.LyricsExactSearchService._build_snapshot

        def build_snapshot(service_self, generation, should_cancel):
            started = time.perf_counter()
            snap = orig_build(service_self, generation, should_cancel)
            with rec.lock:
                rec.lyrics_loads.append({"ts": time.time(), "seconds": round(time.perf_counter() - started, 3),
                                         "documents": len(snap.documents), "generation": generation,
                                         "thread": threading.current_thread().name})
            return snap

        les.LyricsExactSearchService._build_snapshot = build_snapshot  # type: ignore[assignment]

        # --- 보호 판정·tier·정책 생략 · 최종 순서
        orig_rpel = esa.rerank_preserving_exact_lyrics

        def rpel(reranker, analysis, candidates, top_k, answers=None, recorder=esa.NULL_RECORDER, errors=None,
                 statuses=None, fallback_candidates=None, skips=None):
            protected_ids = esa.select_protected_lyric_ids(candidates)
            tiers = {str(t.id): round(float(esa.exact_lyric_constraint_score(analysis, t)), 9)
                     for t in candidates if t.id in protected_ids}
            eligible = bool(getattr(reranker, "enabled", True)) and len(candidates) > 1
            skip = esa.rerank_skip_reason(reranker, analysis, candidates) if eligible else ""
            started = time.perf_counter()
            final = orig_rpel(reranker, analysis, candidates, top_k, answers=answers, recorder=recorder, errors=errors,
                              statuses=statuses, fallback_candidates=fallback_candidates, skips=skips)
            row = rec.row()
            row["protection"] = {"protected_ids": [str(t.id) for t in candidates if t.id in protected_ids],
                                 "tiers": tiers, "policy_skip": skip, "eligible": eligible}
            row["final"] = {"ids": [str(t.id) for t in final],
                            "rerank_scores": [None if t.rerank_score is None else round(float(t.rerank_score), 9) for t in final],
                            "statuses": list(statuses or []), "errors": [e[:200] for e in (errors or [])],
                            "rerank_stage_ms": round((time.perf_counter() - started) * 1000, 3)}
            budget = sys.modules.get("v63_budget")
            if budget is not None and budget.GUARD is not None and budget.GUARD.halted:
                raise RunAborted(f"호출 한도·오류로 중단: {budget.GUARD.halt_reason}")
            return final

        esa.rerank_preserving_exact_lyrics = rpel  # type: ignore[assignment]

        # --- 리랭커 실행 한 번(그룹 하나) · 프롬프트 지문
        orig_run = glr.GeminiListwiseReranker.rerank_run

        def rerank_run(reranker_self, query_text, tracks, top_k, answers=None, *, deadline=None):
            judged = list(tracks[: reranker_self.config.max_candidates])
            rec.local.valid_ids = [str(t.id) for t in judged]
            started = time.perf_counter()
            run = orig_run(reranker_self, query_text, tracks, top_k, answers, deadline=deadline)
            mixes = {sid: _round(dataclasses.asdict(m)) for sid, m in (run.mixes or {}).items()}
            rec.row()["rerank_runs"].append({
                "input_ids": [str(t.id) for t in tracks],
                "input_scores": [round(float(t.score or 0.0), 9) for t in tracks],
                "input_lyric": [t.lyric_match_type or "" for t in tracks],
                "top_k": int(top_k), "status": run.status, "model_calls": run.model_calls,
                "output_ids": [str(t.id) for t in run.tracks],
                "model_order_ids": list(run.model_order_ids or []),
                "order_notes": [list(n) for n in (run.order_notes or [])],
                "mixes_sha16": sha16(json.dumps(mixes, sort_keys=True)),
                "mixes": mixes,
                "unreturned_ids": list(run.unreturned_ids or []),
                "response_stats": dict(run.response_stats or {}),
                "ms": round((time.perf_counter() - started) * 1000, 3),
            })
            return run

        glr.GeminiListwiseReranker.rerank_run = rerank_run  # type: ignore[assignment]

        orig_generate = glr.GeminiListwiseReranker._generate

        def generate(reranker_self, *, contents, config, deadline, tally=None):
            entry = {"contents_sha16": sha16(contents), "contents_chars": len(contents),
                     "config_sha16": sha16(json.dumps(config, sort_keys=True, default=str)),
                     "model": reranker_self.config.model_name}
            started = time.perf_counter()
            if rec.mode == "offline":
                if tally is not None:
                    tally[0] += 1
                ids = list(getattr(rec.local, "valid_ids", []) or [])
                n = max(1, len(ids))
                payload = {"ranking": [{"id": sid, "relevance": round(1.0 - i / n, 6), "reason": "offline"}
                                       for i, sid in enumerate(ids)], "confidence": 0.9}
                with rec.lock:
                    rec.offline_generate_calls += 1
                entry["fake"] = True
                rec.row()["generates"].append(entry)
                return SimpleNamespace(text=json.dumps(payload))
            try:
                response = orig_generate(reranker_self, contents=contents, config=config, deadline=deadline, tally=tally)
            except BaseException as exc:
                entry["error"] = type(exc).__name__
                entry["ms"] = round((time.perf_counter() - started) * 1000, 3)
                rec.row()["generates"].append(entry)
                raise
            entry["ms"] = round((time.perf_counter() - started) * 1000, 3)
            entry["response_text_sha16"] = sha16(getattr(response, "text", "") or "")
            entry["model_version"] = getattr(response, "model_version", None)
            rec.row()["generates"].append(entry)
            return response

        glr.GeminiListwiseReranker._generate = generate  # type: ignore[assignment]

    def save(self, out_dir: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "v63_queries.jsonl").open("w", encoding="utf-8") as fh:
            for qid, row in self.rows.items():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out_dir / "v63_hybrid_calls.jsonl").open("w", encoding="utf-8") as fh:
            for call in self.hybrid_calls:
                fh.write(json.dumps(call, ensure_ascii=False) + "\n")
        calls = self.hybrid_calls
        return {
            "queries_recorded": len([q for q in self.rows if q != "_none"]),
            "unattributed_rows": "_none" in self.rows,
            "hybrid_calls": len(calls),
            "hybrid_calls_matching_reference": sum(1 for c in calls if c["ref_same_set"] and not c["ref_score_mismatch"]
                                                   and not c["ref_order_differs_beyond_ties"]),
            "hybrid_calls_with_missing": sum(1 for c in calls if c["ref_missing"]),
            "hybrid_calls_with_score_mismatch": sum(1 for c in calls if c["ref_score_mismatch"]),
            "lyrics_searches": sum(len(r["lyrics"]) for r in self.rows.values()),
            "lyrics_errors": sum(1 for r in self.rows.values() for e in r["lyrics"] if "error" in e),
            "lyrics_zero_results": sum(1 for r in self.rows.values() for e in r["lyrics"] if e.get("results") == 0),
            "lyrics_snapshot_builds": list(self.lyrics_loads),
            "offline_generate_calls": self.offline_generate_calls,
            "generate_records": sum(len(r["generates"]) for r in self.rows.values()),
        }


def index_fingerprint(client: Any, collection: str):
    from exact_hybrid import ExactHybridReference

    started = time.perf_counter()
    reference = ExactHybridReference.from_client(client, collection)
    fp = reference.fingerprint()
    fp["build_seconds"] = round(time.perf_counter() - started, 3)
    if fp["content_sha256_16"] != EXPECTED_INDEX_FINGERPRINT:
        raise RuntimeError(f"색인 복사본 지문 {fp['content_sha256_16']} ≠ {EXPECTED_INDEX_FINGERPRINT}")
    return reference, fp


def git_state() -> Dict[str, Any]:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short", "src"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {"head": sha, "src_dirty_lines": len(dirty.splitlines()) if dirty else 0}
