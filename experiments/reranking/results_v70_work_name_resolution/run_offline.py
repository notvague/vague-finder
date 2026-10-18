"""v70 오프라인 후보 비교 — 작품명 해석 보조 경로(리랭킹 전 후보 30). **외부 모델 호출 0회.**

구성(모두 소리 ON · Context 0.5 · D OFF · 피드백 OFF · 서버 Qdrant `vague-finder-qdrant-v70` · v63 가사 스냅샷 · v49 Vertex 분석 캐시):

| 구성 | 해석기 | 결과 종류(RUN_INFO §1-8) |
|---|---|---|
| `off` | 주입 안 함(운영 기본과 같다) | 기준 |
| `gate_abstain` | 판정만 실제, 해석기는 언제나 보류 | ① 실제 자동 판정(대상 수·지연) — 후보는 off와 같아야 한다 |
| `oracle` | v67 `oracle_work_name.py`의 사람 작성 이름(이미 있는 목록, 새로 만들지 않음), 판정 그대로 | ③ 정답을 알고 넣은 진단 |
| `oracle_nogate` | 같은 이름, 판정 우회(목록 질의만) | ③ 진단 — 판정 누락과 경로 능력을 가른다 |
| `wrong` | 판정 통과 질의마다 그럴듯한 오답 작품명 1개(코퍼스 앨범명 "… OST"에서 고정 시드, 정답 곡을 언급하는 이름은 제외) | ② 모의 — 틀린 이름의 피해 폭 |

해석기 입력은 질의 원문뿐이다. 정답(relevant_ids)은 채점과 `wrong`의 "정답을 언급하는 이름 제외"에만 쓰고 해석기·라우터에 넘기지 않는다.

    venv/bin/python experiments/reranking/results_v70_work_name_resolution/run_offline.py --config off --out <폴더>
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
import random
import re
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V63_DIR = ROOT / "experiments/reranking/results_v63_hybrid_lyrics_validation"
V67_ORACLE = Path("/private/tmp/vague-finder-v67-candidate-recall/experiments/reranking/"
                  "results_v67_candidate_recall_diagnosis/oracle_work_name.py")
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
CONFIGS = ("off", "gate_abstain", "oracle", "oracle_nogate", "wrong", "env_cache")
WRONG_SEED = 70
# 저장소를 만든 자료와 같은 파일(sha는 저장소 manifest에 있다)
SONGS_JSONL = Path("/Users/ozzing727/Desktop/Developer/졸작-RISE/vague-finder/data/all_songs.jsonl")

FIXED_ENV = {
    "CONTEXT_WEIGHT": "0.5", "ATTR_FILTER_WEIGHT": "0", "AUDIO_FUSION_SCALE": "1", "SEARCH_REFERENCE_YEAR": "2026",
    "RERANKER_BACKEND": "gemini_listwise", "GEMINI_RERANK_RARE_FACT_VERIFY": "0", "GEMINI_RETRIEVAL_BACKEND": "vertex",
    "GCP_LOCATION": "global", "GEMINI_MODEL_NAME": "gemini-3.1-flash-lite", "GCP_PROJECT_ID": "v70-offline-no-calls",
    "FEEDBACK_COLLECTION_ENABLED": "0", "FEEDBACK_INDEX_WORKER_ENABLED": "0", "FEEDBACK_RETRIEVAL_ENABLED": "0",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "QDRANT_URL": "http://127.0.0.1:6376",
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
# env_cache: 서비스와 같은 공장 경로(WORK_NAME_RESOLUTION_ENABLED=1 · WORK_NAME_RESOLVER=cache:<경로>)로 켠다 — 배선 확인용
if "--config" in sys.argv and sys.argv[sys.argv.index("--config") + 1] == "env_cache":
    os.environ["WORK_NAME_RESOLUTION_ENABLED"] = "1"
    os.environ["WORK_NAME_RESOLVER"] = "cache:" + os.environ["V70_RESOLUTION_CACHE"]

NET_ATTEMPTS: List[str] = []


class Blocked(RuntimeError):
    pass


def _blocked(*_a: Any, **_k: Any) -> Any:
    raise Blocked("v70: 외부 호출 차단")


class _BlockedClient:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise Blocked("v70: 클라이언트 생성 차단")


def install_guards() -> Dict[str, str]:
    orig_connect = socket.socket.connect

    def connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6):
            host = str(address[0] if isinstance(address, tuple) else address)
            if not (host.startswith("127.") or host in {"::1", "localhost"}):
                NET_ATTEMPTS.append(host)
                raise OSError(f"v70: 외부 연결 차단 {host}")
        return orig_connect(self, address)

    socket.socket.connect = connect  # type: ignore[assignment]
    orig_gai = socket.getaddrinfo

    def gai(host, *a, **k):  # type: ignore[no-untyped-def]
        h = host.decode() if isinstance(host, bytes) else host
        if h is not None and not (str(h).startswith("127.") or str(h) in {"::1", "localhost"}):
            NET_ATTEMPTS.append(f"dns:{h}")
            raise OSError(f"v70: 외부 이름 조회 차단 {h}")
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
        checks["network"] = "blocked" if "v70" in str(exc) else f"NOT BLOCKED ({exc})"
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


def load_oracle_names() -> Dict[str, str]:
    """v67 진단 파일의 WORK_NAMES를 그대로 읽는다(코드 실행 없이 리터럴만). 새 이름을 만들지 않는다."""
    import ast

    tree = ast.parse(V67_ORACLE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "WORK_NAMES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("v67 WORK_NAMES를 찾지 못했다")


def corpus_ost_names(songs_path: Path) -> List[str]:
    """그럴듯한 오답 후보: 코퍼스 앨범명 "<이름> OST …"의 <이름>(실험 전용, 기능에는 쓰지 않는다)."""
    names = set()
    with open(songs_path, encoding="utf-8") as f:
        for line in f:
            album = str((json.loads(line).get("metadata") or {}).get("album") or "")
            m = re.match(r"^(.{2,30}?)\s*(?:\(.*?\)\s*)?OST\b", album, re.I)
            if m:
                name = m.group(1).strip(" -:'\"")
                if len(name) >= 2:
                    names.add(name)
    return sorted(names)


class StaticResolver:
    """진단·모의 해석기 — 질의 원문 → 정해진 이름. 근거 구절은 질의 전체로 둔다(진단 전용)."""

    def __init__(self, by_query: Dict[str, str], label: str) -> None:
        self.by_query = by_query
        self.name = label
        self.calls = 0

    def resolve(self, query: str, *, timeout_seconds: float):
        from src.retrieval.work_name_resolution import ResolverOutput, WorkCandidate

        self.calls += 1
        name = self.by_query.get(query)
        if not name:
            return ResolverOutput("abstain", source=self.name)
        return ResolverOutput("candidates", (WorkCandidate(name=name, media_type="other", evidence=query),),
                              source=self.name)


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
    if config == "env_cache":
        if router._work_names is None or not router._work_names.resolver.name.startswith("cache:"):
            raise SystemExit("공장이 캐시 해석기를 만들지 않았다")
    elif router._work_names is not None:
        raise SystemExit("공장이 해석기를 만들었다 — WORK_NAME_RESOLUTION_ENABLED가 새고 있다")
    store = WorkMentionStore.load(Path(os.environ["WORK_MENTIONS_PATH"]))
    resolver = None
    gate_bypass_qids: set = set()
    if config not in ("off", "env_cache"):
        by_query: Dict[str, str] = {}
        if config in ("oracle", "oracle_nogate"):
            oracle = load_oracle_names()
            by_query = {r["query"]: oracle[r["qid"]] for r in rows if r["qid"] in oracle}
            if config == "oracle_nogate":
                gate_bypass_qids = {r["qid"] for r in rows if r["qid"] in oracle}
        elif config == "wrong":
            names = corpus_ost_names(SONGS_JSONL)
            rng = random.Random(WRONG_SEED)
            for r in rows:
                if not wnr.gate_indirect_work_reference(r["analysis"]).triggered:
                    continue
                rel = set(r["relevant"])
                for _ in range(200):
                    name = rng.choice(names)
                    mentioned = {h.song_id for h in store.songs_mentioning([name])}
                    if mentioned and not mentioned & rel and len(mentioned) <= wnr.MAX_MENTION_SONGS:
                        by_query[r["query"]] = name
                        break
        resolver = StaticResolver(by_query, f"diagnostic:{config}")
        router._work_names = wnr.WorkNameRuntime(resolver, store)
        router._work_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="work_name")
    real_gate = sr.gate_indirect_work_reference
    current: Dict[str, str] = {}
    if gate_bypass_qids:
        def gate(analysis):  # type: ignore[no-untyped-def]
            if current.get("qid") in gate_bypass_qids:
                return wnr.GateDecision(True, "bypassed_for_diagnosis")
            return wnr.GateDecision(False, "bypassed_for_diagnosis")
        sr.gate_indirect_work_reference = gate  # type: ignore[assignment]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for r in rows:
        current["qid"] = r["qid"]
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
        })
    sr.gate_indirect_work_reference = real_gate  # type: ignore[assignment]
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
        "resolver_calls": getattr(resolver, "calls", 0),
        "resolver_name": getattr(router._work_names.resolver, "name", "") if router._work_names else "", "net_attempts": len(NET_ATTEMPTS),
        "src_tree": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "src_diff_sha16": hashlib.sha256(os.popen(f"git -C {ROOT} diff HEAD -- src").read().encode()).hexdigest()[:16],
        "untracked_src_sha16": hashlib.sha256(b"".join(
            (ROOT / p).read_bytes() for p in ("src/retrieval/work_mentions.py", "src/retrieval/work_name_resolution.py")
        )).hexdigest()[:16],
    }
    (out_dir / "run_conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8")
    print(config, "done", len(results), "resolver calls", conditions["resolver_calls"])


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
