"""v65 §4-⑦ 비용 — 수정 전/후 구현을 **같은 프로세스·같은 클라이언트에서 교대로** 바꿔 끼워 지연을 잰다(모델·Mongo 호출 없음).

바뀐 코드는 `src/vector_db/qdrant_backend.py`·`src/retrieval/context_qdrant_search.py` 두 파일뿐이다(`git diff --stat`).
수정 전 판은 `artifacts/src_base_2a6481d`의 같은 두 파일을 별도 모듈 이름으로 읽어 `QdrantIndex.query`·`ContextQdrantSearch._dense`·
`._sparse` 세 메서드만 바꿔 끼운다(나머지 메서드는 두 판에서 같다). 교체가 맞는지는 각 판의 후보 30을 평가기 실행(offline)과
대조해 확인한다.

측정(모드 하나 = 프로세스 하나, `V65_MODE=local|server`, 리랭킹 끔):
  path   경로별 논리 조회를 한 번 기록(캡처)한 뒤 같은 인자로 다시 부른다. 판 순서는 질의마다 ABBA… 로 교대, 각 판 5회.
         Context 사실 조회의 질의 임베딩은 첫 호출 결과를 재사용해 지연에서 뺀다(두 판에 같은 비용이라서).
  full   dev 176 전체의 `router.search(top_k=30, use_rerank=False, candidate_k=30)`, 판 교대 4회(ABBA — 각 판 앞자리 2번).
  준비   프로세스 시작 → 라우터·모델 준비 → 첫 검색(콜드)은 `--cold <arm>`으로 따로 띄운 프로세스에서 잰다.
합격선은 두지 않는다. 출력: bench/<mode>_{path,full}.json · bench/cold_<mode>_<arm>_<rep>.json
"""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("V65_CODE", "after")
import v65_eval_runtime as rt  # noqa: E402

ROOT = rt.ROOT
os.chdir(ROOT)
OUT = HERE / "bench"


def load_before_modules():
    out = {}
    for name, rel in (("qb_before", "src/vector_db/qdrant_backend.py"), ("cqs_before", "src/retrieval/context_qdrant_search.py")):
        spec = importlib.util.spec_from_file_location(name, rt.BASE_SRC / rel)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = mod  # dataclass가 모듈을 찾는다
        spec.loader.exec_module(mod)
        out[name] = mod
    return out


def setup():
    rt.install_guards()
    rt.verify_guards()
    import src.backend.api.dependencies as deps
    from src.backend.schemas.query import QueryAnalysis
    from src.retrieval.analysis_cache import load_cache
    import src.retrieval.context_qdrant_search as cqs
    from src.vector_db import qdrant_backend as qb

    rt.install_lyrics_snapshot()
    before = load_before_modules()
    arms = {
        "before": (before["qb_before"].QdrantIndex.query, before["cqs_before"].ContextQdrantSearch._dense,
                   before["cqs_before"].ContextQdrantSearch._sparse),
        "after": (qb.QdrantIndex.query, cqs.ContextQdrantSearch._dense, cqs.ContextQdrantSearch._sparse),
    }

    def set_arm(arm: str) -> None:
        q, d, s = arms[arm]
        qb.QdrantIndex.query = q  # type: ignore[assignment]
        cqs.ContextQdrantSearch._dense = d  # type: ignore[assignment]
        cqs.ContextQdrantSearch._sparse = s  # type: ignore[assignment]

    analyses = {}
    for s in rt.SETS:
        cache = load_cache(rt.CACHES / f"analysis_cache_{s}_dev_vertex.json")
        for qid, e in cache.entries.items():
            analyses[qid] = QueryAnalysis(**e["analysis"])
    return deps, qb, cqs, set_arm, analyses


def search_ids(router, analysis) -> list:
    tracks = asyncio.run(router.search(analysis, top_k=30, use_rerank=False, candidate_k=30))
    return [t.id for t in tracks]


OFFLINE_REP = int(os.environ.get("V65_BENCH_OFFLINE_REP", "1"))


def offline_candidates(arm: str) -> dict:
    out = {}
    for s in rt.SETS:
        for line in (HERE / "offline" / f"{arm}_{rt.MODE}_rep{OFFLINE_REP}" / f"{s}_dev" / "v65_queries.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["query_id"]] = [c["id"] for c in r["candidates"]]
    return out


def run_full(router, set_arm, analyses) -> dict:
    qids = sorted(analyses)
    # 균형 교대(ABBA): 각 판이 앞자리 2번·뒷자리 2번. 라운드별 값을 남겨 자리 효과를 따로 본다
    orders = [("before", "after"), ("after", "before"), ("after", "before"), ("before", "after")]
    first_pos = {"before": [], "after": []}
    times = {"before": {q: [] for q in qids}, "after": {q: [] for q in qids}}
    ids = {"before": {}, "after": {}}
    for q in qids:  # 예열 1회(두 판 공통, 기록 안 함)
        search_ids(router, analyses[q])
    for order in orders:
        for q in qids:
            for pos, arm in enumerate(order):
                set_arm(arm)
                t0 = time.perf_counter()
                got = search_ids(router, analyses[q])
                ms = (time.perf_counter() - t0) * 1000
                times[arm][q].append(ms)
                if pos == 0:
                    first_pos[arm].append(ms)
                ids[arm].setdefault(q, got)
    set_arm("after")
    match = {arm: sum(1 for q in qids if ids[arm][q] == offline_candidates(arm)[q]) for arm in ("before", "after")}
    med = {arm: {q: statistics.median(v) for q, v in times[arm].items()} for arm in times}
    diffs = [med["after"][q] - med["before"][q] for q in qids]
    second_pos = {arm: [v for q in qids for v in times[arm][q]] for arm in times}
    return {"queries": len(qids), "reps_per_arm": len(orders), "order": "ABBA", "candidates_match_offline": {"rep": OFFLINE_REP, **match},
            "position_effect_ms": {arm: {"first_median": round(statistics.median(first_pos[arm]), 2),
                                         "all_median": round(statistics.median(second_pos[arm]), 2)} for arm in times},
            "median_ms": {arm: round(statistics.median(list(med[arm].values())), 2) for arm in med},
            "p95_ms": {arm: round(sorted(med[arm].values())[int(0.95 * (len(qids) - 1))], 2) for arm in med},
            "per_query_diff_ms": {"median": round(statistics.median(diffs), 2),
                                  "p95": round(sorted(diffs)[int(0.95 * (len(diffs) - 1))], 2),
                                  "max": round(max(diffs), 2), "min": round(min(diffs), 2)},
            "per_query_median_ms": {q: [round(med["before"][q], 2), round(med["after"][q], 2)] for q in qids}}


def run_path(router, qb, cqs, set_arm, analyses) -> dict:
    """경로별 논리 조회를 캡처하고(수정 후 판으로 한 번 검색) 같은 인자로 두 판을 교대로 다시 부른다."""
    captured = []
    orig_query = qb.QdrantIndex.query
    orig_dense, orig_sparse = cqs.ContextQdrantSearch._dense, cqs.ContextQdrantSearch._sparse

    def cap_query(index_self, **kw):
        captured.append(("index", index_self, kw))
        return orig_query(index_self, **kw)

    def cap_dense(s, snapshot, query, limit, *a, **k):
        captured.append(("context_fact", s, (snapshot, query, limit, a, k)))
        return orig_dense(s, snapshot, query, limit, *a, **k)

    def cap_sparse(s, snapshot, query, limit, *a, **k):
        captured.append(("context_profile", s, (snapshot, query, limit, a, k)))
        return orig_sparse(s, snapshot, query, limit, *a, **k)

    qb.QdrantIndex.query, cqs.ContextQdrantSearch._dense, cqs.ContextQdrantSearch._sparse = cap_query, cap_dense, cap_sparse
    for q in sorted(analyses):
        search_ids(router, analyses[q])
    set_arm("after")
    # Context 사실 조회의 질의 임베딩 재사용(두 판 같은 비용이라 지연에서 뺀다)
    memo: dict = {}
    for kind, s, _ in captured:
        if kind == "context_fact" and not getattr(s, "_v65_memo", False):
            real = s._text_embedder

            def cached(snapshot, _real=real):
                emb = _real(snapshot)

                class _Memo:
                    def embed_passages(self, passages, **kw):
                        key = (tuple(passages), tuple(sorted(kw.items())))
                        if key not in memo:
                            memo[key] = emb.embed_passages(passages, **kw)
                        return memo[key]
                return _Memo()

            s._text_embedder = cached
            s._v65_memo = True
    rows = []
    for i, (kind, obj, args) in enumerate(captured):
        times = {"before": [], "after": []}
        for rep in range(5):
            for arm in (("before", "after") if (i + rep) % 2 == 0 else ("after", "before")):
                set_arm(arm)
                t0 = time.perf_counter()
                if kind == "index":
                    qb.QdrantIndex.query(obj, **args)
                elif kind == "context_fact":
                    snapshot, query, limit, a, k = args
                    cqs.ContextQdrantSearch._dense(obj, snapshot, query, limit, *a, **k)
                else:
                    snapshot, query, limit, a, k = args
                    cqs.ContextQdrantSearch._sparse(obj, snapshot, query, limit, *a, **k)
                times[arm].append((time.perf_counter() - t0) * 1000)
        name = obj._index_name.replace("vaguefinder-", "").split("-")[0] if kind == "index" else kind
        if kind == "index" and args.get("sparse_vector") is not None:
            name = "text_hybrid"
        rows.append({"path": name, "before_ms": statistics.median(times["before"][1:]), "after_ms": statistics.median(times["after"][1:])})
    set_arm("after")
    out = {}
    for path in sorted({r["path"] for r in rows}):
        sel = [r for r in rows if r["path"] == path]
        d = [r["after_ms"] - r["before_ms"] for r in sel]
        out[path] = {"calls": len(sel), "before_median_ms": round(statistics.median(r["before_ms"] for r in sel), 3),
                     "after_median_ms": round(statistics.median(r["after_ms"] for r in sel), 3),
                     "diff_median_ms": round(statistics.median(d), 3), "diff_p95_ms": round(sorted(d)[int(0.95 * (len(d) - 1))], 3),
                     "diff_max_ms": round(max(d), 3)}
    return out


def payload_bytes(deps) -> dict:
    """경로별 점 하나의 payload JSON 크기(조회가 늘어난 만큼의 전송·메모리 추정용)."""
    client = deps.get_vector_client().client
    out = {}
    for name in ("vaguefinder-image-768__dev", "vaguefinder-audio-512__dev"):
        pts, _ = client.scroll(name, limit=200, with_payload=True, with_vectors=False)
        sizes = [len(json.dumps(p.payload, ensure_ascii=False).encode()) for p in pts]
        out[name.split("__")[0]] = {"sample": len(sizes), "median_bytes": statistics.median(sizes), "max_bytes": max(sizes)}
    for alias in [a.alias_name for a in client.get_aliases().aliases]:
        pts, _ = client.scroll(alias, limit=200, with_payload=True, with_vectors=False)
        sizes = [len(json.dumps(p.payload, ensure_ascii=False).encode()) for p in pts]
        out[alias.split("__")[0]] = {"sample": len(sizes), "median_bytes": statistics.median(sizes), "max_bytes": max(sizes)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="path,full")
    ap.add_argument("--cold", default="")
    ap.add_argument("--rep", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    deps, qb, cqs, set_arm, analyses = setup()
    if args.cold:
        set_arm(args.cold)
        t0 = time.perf_counter()
        router = deps.get_search_router()
        ready = time.perf_counter() - t0
        first_q = "q205"
        t1 = time.perf_counter()
        search_ids(router, analyses[first_q])
        first = time.perf_counter() - t1
        t2 = time.perf_counter()
        search_ids(router, analyses[first_q])
        second = time.perf_counter() - t2
        res = {"mode": rt.MODE, "arm": args.cold, "rep": args.rep, "process_to_setup_s": round(t0 - PROCESS_STARTED, 2),
               "router_ready_s": round(ready, 2), "first_search_ms": round(first * 1000, 1), "second_search_ms": round(second * 1000, 1),
               "note": "첫 검색에는 가사 판 적재(스냅샷 재생)·모델 첫 추론·Qdrant 첫 조회가 들어 있다"}
        (OUT / f"cold_{rt.MODE}_{args.cold}_{args.rep}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False))
        os._exit(0)
    router = deps.get_search_router()
    search_ids(router, analyses["q205"])
    result = {"mode": rt.MODE, "network_attempts_before": list(rt.NET_ATTEMPTS)}
    if "path" in args.what:
        result["path"] = run_path(router, qb, cqs, set_arm, analyses)
        print("path", json.dumps(result["path"], ensure_ascii=False), flush=True)
    if "full" in args.what:
        result["full"] = run_full(router, set_arm, analyses)
        print("full", json.dumps({k: v for k, v in result["full"].items() if k != "per_query_median_ms"}, ensure_ascii=False), flush=True)
    result["payload_bytes"] = payload_bytes(deps)
    result["network_attempts"] = list(rt.NET_ATTEMPTS)
    (OUT / f"{rt.MODE}_bench.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    os._exit(0)


if __name__ == "__main__":
    main()
