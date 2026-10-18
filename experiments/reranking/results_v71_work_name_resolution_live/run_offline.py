"""v71 오프라인 후보 비교 — 실제 해석 출력(반복별 캐시)으로 dev 176 후보 30(리랭킹 전). **외부 모델 호출 0회.**

v70 `run_offline.py` 사본. 바꾼 것: 구성을 `off` · `live_rep1` · `live_rep2`로 바꿨다(정답 주입·오답 모의 제거), 전용 Qdrant를
v71 사본(`vague-finder-qdrant-v71` · 6378)으로, 기록에 가사 보호 곡(`select_protected_lyric_ids`, 서비스와 같은 함수)을 더했다.

| 구성 | 해석기 |
|---|---|
| `off` | 주입 안 함(운영 기본과 같다) — A |
| `live_rep1` · `live_rep2` | 서비스 공장 경로(`WORK_NAME_RESOLUTION_ENABLED=1` · `WORK_NAME_RESOLVER=cache:<resolution_cache/repN.json>`) — B1 · B2 |

조건(소리 ON · Context 0.5 · D OFF · 피드백 OFF · v63 가사 스냅샷 · v49 Vertex 분석 캐시)은 v70과 같다. 정답(relevant_ids)은
채점에만 쓰고 해석기·라우터에 넘기지 않는다. 해석 실패·보류·cache_miss(정지로 못 부른 질의)는 서비스와 같은 상태로 재생된다.

    venv/bin/python experiments/reranking/results_v71_work_name_resolution_live/run_offline.py --config off --out <폴더>
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import os
import pickle
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V63_DIR = ROOT / "experiments/reranking/results_v63_hybrid_lyrics_validation"
for p in list(sys.path):
    if p in ("", ".", str(HERE)):
        sys.path.remove(p)
for p in (str(V63_DIR), str(ROOT)):
    sys.path.insert(0, p)
if "src" in sys.modules:
    raise SystemExit("src가 이미 import됐다")

SNAPSHOT_DIR = Path("/private/tmp/vague-finder-hybrid-lyrics-validation/artifacts/lyrics_snapshot_v63/snapshot")
EXPECTED_LYRICS = "e0d58f28f1f4a563"
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"
SETS = ("v06", "v09", "v10")
CONFIGS = ("off", "live_rep1", "live_rep2")

FIXED_ENV = {
    "CONTEXT_WEIGHT": "0.5", "ATTR_FILTER_WEIGHT": "0", "AUDIO_FUSION_SCALE": "1", "SEARCH_REFERENCE_YEAR": "2026",
    "RERANKER_BACKEND": "gemini_listwise", "GEMINI_RERANK_RARE_FACT_VERIFY": "0", "GEMINI_RETRIEVAL_BACKEND": "vertex",
    "GCP_LOCATION": "global", "GEMINI_MODEL_NAME": "gemini-3.1-flash-lite", "GCP_PROJECT_ID": "v71-offline-no-calls",
    "FEEDBACK_COLLECTION_ENABLED": "0", "FEEDBACK_INDEX_WORKER_ENABLED": "0", "FEEDBACK_RETRIEVAL_ENABLED": "0",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "QDRANT_URL": "http://127.0.0.1:6378",
    # 라우터 공장은 해석기를 만들지 않는다 — 구성별 해석기는 아래에서 직접 주입한다(진단 해석기를 모델로 가장하지 않게)
    "WORK_NAME_RESOLUTION_ENABLED": "0",
}
REMOVE_ENV = (
    "QDRANT_PATH", "QDRANT_HYBRID_SCAN_LIMIT", "CLAUDE_EFFORT", "CLAUDE_API", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS", "RERANKER_ENABLED", "GEMINI_RERANK_ENABLED", "GEMINI_RERANK_MODEL_NAME",
    "GEMINI_RERANK_TIMEOUT_SECONDS", "GEMINI_RERANK_BUDGET_SECONDS", "GEMINI_RERANK_MAX_RETRIES", "GEMINI_RERANK_PASSES",
    "GEMINI_RERANK_USE_SEARCH", "LYRICS_CACHE_TTL_SECONDS", "LYRICS_FUZZY_THRESHOLD", "MONGO_URI", "MONGO_DB_NAME",
    "ATTR_FILTER_GATE", "ATTR_FILTER_K", "ATTR_FILTER_USE_GENRE", "ATTR_FILTER_GENRE_POLICY", "VF_CLAUDE_EFFORT",
    "VF_CLAUDE_THINKING", "WORK_NAME_RESOLVER", "WORK_NAME_AUX_WEIGHT", "WORK_NAME_RESOLVER_TIMEOUT_SECONDS",
    "WORK_MENTIONS_PATH",
)
for key in REMOVE_ENV:
    os.environ.pop(key, None)
os.environ.update(FIXED_ENV)
os.environ["WORK_MENTIONS_PATH"] = str(ROOT / "artifacts/work_names/work_mentions_v1.json.gz")
# live_repN: 서비스와 같은 공장 경로로 켠다(해석기 = 반복 N의 실제 출력 캐시, 호출 없음)
CONFIG_ARG = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else ""
LIVE_CACHE = HERE / f"resolution_cache/rep{CONFIG_ARG[-1]}.json" if CONFIG_ARG.startswith("live_rep") else None
if LIVE_CACHE is not None:
    os.environ["WORK_NAME_RESOLUTION_ENABLED"] = "1"
    os.environ["WORK_NAME_RESOLVER"] = "cache:" + str(LIVE_CACHE)

NET_ATTEMPTS: List[str] = []


class Blocked(RuntimeError):
    pass


def _blocked(*_a: Any, **_k: Any) -> Any:
    raise Blocked("v71: 외부 호출 차단")


class _BlockedClient:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise Blocked("v71: 클라이언트 생성 차단")


def install_guards() -> Dict[str, str]:
    orig_connect = socket.socket.connect

    def connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6):
            host = str(address[0] if isinstance(address, tuple) else address)
            if not (host.startswith("127.") or host in {"::1", "localhost"}):
                NET_ATTEMPTS.append(host)
                raise OSError(f"v71: 외부 연결 차단 {host}")
        return orig_connect(self, address)

    socket.socket.connect = connect  # type: ignore[assignment]
    orig_gai = socket.getaddrinfo

    def gai(host, *a, **k):  # type: ignore[no-untyped-def]
        h = host.decode() if isinstance(host, bytes) else host
        if h is not None and not (str(h).startswith("127.") or str(h) in {"::1", "localhost"}):
            NET_ATTEMPTS.append(f"dns:{h}")
            raise OSError(f"v71: 외부 이름 조회 차단 {h}")
        return orig_gai(host, *a, **k)

    socket.getaddrinfo = gai  # type: ignore[assignment]
    import pymongo
    import src.common.gemini_client as gc
    import src.common.mongodb as mongodb
    import src.retrieval.gemini_listwise_reranker as glr
    import src.retrieval.lyrics_exact_search as les
    from google import genai
    import google.genai.client as genai_client_mod

    pymongo.MongoClient = _BlockedClient  # type: ignore[attr-defined]
    mongodb.MongoClient = _BlockedClient  # type: ignore[attr-defined]
    mongodb._connect = _blocked  # type: ignore[assignment]
    les._make_lyrics_mongo_client = _blocked  # type: ignore[assignment]
    gc.make_genai_client = _blocked  # type: ignore[assignment]
    gc._load_default_credentials = _blocked  # type: ignore[assignment]
    glr.make_genai_client = _blocked  # type: ignore[assignment]
    genai.Client = _BlockedClient  # type: ignore[attr-defined]
    genai_client_mod.Client = _BlockedClient  # type: ignore[attr-defined]
    checks = {}
    for name, fn in {"genai": lambda: genai.Client(vertexai=True, project="x", location="global"),
                     "make_genai_client": lambda: gc.make_genai_client(purpose="retrieval"),
                     "mongo": lambda: pymongo.MongoClient("mongodb://127.0.0.1:1")}.items():
        try:
            fn()
            checks[name] = "NOT BLOCKED"
        except Blocked:
            checks[name] = "blocked"
    try:
        socket.create_connection(("192.0.2.1", 9), timeout=0.5)
        checks["network"] = "NOT BLOCKED"
    except OSError as exc:
        checks["network"] = "blocked" if "v71" in str(exc) else f"NOT BLOCKED ({exc})"
    NET_ATTEMPTS.clear()
    if any(v != "blocked" for v in checks.values()):
        raise SystemExit(f"차단 확인 실패: {checks}")
    return checks


def install_lyrics_snapshot() -> Dict[str, Any]:
    import src.retrieval.lyrics_exact_search as les
    from snapshot_lyrics import content_digest

    manifest = json.loads((SNAPSHOT_DIR / "manifest.json").read_text(encoding="utf-8"))
    data = (SNAPSHOT_DIR / manifest["file"]["name"]).read_bytes()
    if hashlib.sha256(data).hexdigest()[:16] != manifest["file"]["sha256_16"]:
        raise SystemExit("가사 스냅샷 sha 불일치")
    documents = pickle.loads(gzip.decompress(data))
    les.LyricsExactSearchService._source_documents = lambda self: iter(list(documents))  # type: ignore[assignment]
    pairs = [(str(d.get("song_id") or ""), str((d.get("lyrics_data") or {}).get("full_lyrics") or "")) for d in documents]
    digest = content_digest(pairs)
    if digest["content_sha256_16"] != EXPECTED_LYRICS:
        raise SystemExit("가사 지문 불일치")
    return digest


def read_dev() -> List[Dict[str, Any]]:
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.evaluate_search_accuracy import parse_relevant_ids, read_queries

    rows = []
    for s in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{s}_dev_vertex.json")
        for row in read_queries(ROOT / f"experiments/reranking/eval_queries_{s}.csv", "dev"):
            qid = row["query_id"].strip()
            rows.append({"set": s, "qid": qid, "query": row["query"].strip(),
                         "relevant": sorted(parse_relevant_ids(row["relevant_ids"])),
                         "analysis": cache.get(qid, row["query"].strip())})
    if len(rows) != 176 or len({r["qid"] for r in rows}) != 176:
        raise SystemExit(f"dev 질의 {len(rows)}건")
    return rows


async def run(config: str, out_dir: Path) -> None:
    guards = install_guards()
    lyrics = install_lyrics_snapshot()
    import src.backend.api.dependencies as deps
    import src.retrieval.search_router as sr
    from src.retrieval import work_name_resolution as wnr
    from src.retrieval.work_mentions import WorkMentionStore

    rows = read_dev()
    router = deps.get_search_router()
    if LIVE_CACHE is not None:
        if router._work_names is None or not router._work_names.resolver.name.startswith("cache:"):
            raise SystemExit("공장이 캐시 해석기를 만들지 않았다")
    elif router._work_names is not None:
        raise SystemExit("공장이 해석기를 만들었다 — WORK_NAME_RESOLUTION_ENABLED가 새고 있다")
    store = WorkMentionStore.load(Path(os.environ["WORK_MENTIONS_PATH"]))
    real_gate = sr.gate_indirect_work_reference
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for r in rows:
        pool: List[Any] = []
        wres: List[Any] = []
        started = time.perf_counter()
        await router.search(r["analysis"], top_k=30, use_rerank=False, candidate_k=30,
                            candidate_tracks_out=pool, work_resolution_out=wres)
        elapsed = (time.perf_counter() - started) * 1000
        gate_real = real_gate(r["analysis"])
        results.append({
            "set": r["set"], "qid": r["qid"], "relevant": r["relevant"],
            "candidates": [{"id": t.id, "score": round(float(t.score), 9)} for t in pool],
            "search_ms": round(elapsed, 1),
            "gate": {"triggered": gate_real.triggered, "reason": gate_real.reason},
            "work_resolution": wres[0].as_dict() if wres else None,
            "protected_lyric_ids": sorted(sr.select_protected_lyric_ids(pool)),
        })
    router.shutdown()
    if NET_ATTEMPTS:
        raise SystemExit(f"외부 연결 시도 {NET_ATTEMPTS}")
    # 질의 원문·작품명은 원기록(Git 제외)에만 둔다
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    conditions = {
        "config": config, "guards": guards, "lyrics": lyrics, "fixed_env": FIXED_ENV,
        "store": store.fingerprint(), "versions": wnr.versions(store),
        "live_cache": str(LIVE_CACHE.relative_to(HERE)) if LIVE_CACHE else None,
        "live_cache_sha16": hashlib.sha256(LIVE_CACHE.read_bytes()).hexdigest()[:16] if LIVE_CACHE else None,
        "resolver_name": getattr(router._work_names.resolver, "name", "") if router._work_names else "", "net_attempts": len(NET_ATTEMPTS),
        "src_tree": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "src_diff_sha16": hashlib.sha256(os.popen(f"git -C {ROOT} diff HEAD -- src").read().encode()).hexdigest()[:16],
    }
    (out_dir / "run_conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8")
    print(config, "done", len(results))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out}가 이미 있다 — 덮어쓰지 않는다")
    asyncio.run(run(args.config, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
