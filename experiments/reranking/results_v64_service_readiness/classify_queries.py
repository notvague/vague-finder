"""지연 표본 선정 — v56 `classify_queries.py`의 규칙을 **현재 코드**(dd5cc3e)로 다시 돌린다. 모델·Mongo 호출 없음(v64_runtime 차단).

범주(D = ATTR_FILTER_WEIGHT 0.5 · era_plus · K 30 · USE_GENRE 1 · conditional, 1턴 · 답 없음):
  closed      `_build_attribute_filter_with_policy`가 None — 보조 경로가 닫힌다
  open_hits   필터가 만들어지고 서비스와 같은 `SearchRouter._search_text(..., metadata_filter)`가 1곡 이상
  open_empty  필터는 만들어지지만 0곡
제외(선정 전 고정, v56과 같다): 분석의 lyric_clues에 verbatim/partial/phonetic이 있는 질의. v56의 제외 이유(Mongo 가사 판 재적재)는
이번 스냅샷 환경에서는 작아졌지만 규칙을 바꾸지 않는다 — v56과 같은 규칙으로 고른 표본이어야 비교 설명이 된다.
선정: 범주별 v06 → v09 → v10 번갈아(세트 안 query_id 순) 10건. 합성 open_empty 10건 = open_hits 표본 분석의 시기 창만 1960~1965로 바꾼 사본.

실행(Qdrant 서버가 떠 있어야 한다, 워크트리 루트):
  QDRANT_URL=http://127.0.0.1:6364 venv/bin/python experiments/reranking/results_v64_service_readiness/classify_queries.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

for key in ("QDRANT_PATH", "QDRANT_HYBRID_SCAN_LIMIT", "MONGO_URI", "GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ.pop(key, None)
D_ENV = {
    "ATTR_FILTER_WEIGHT": "0.5", "ATTR_FILTER_GATE": "era_plus", "ATTR_FILTER_K": "30",
    "ATTR_FILTER_USE_GENRE": "1", "ATTR_FILTER_GENRE_POLICY": "conditional",
    "QDRANT_URL": os.environ.get("QDRANT_URL", "http://127.0.0.1:6364"), "CONTEXT_WEIGHT": "0.5",
    "SEARCH_REFERENCE_YEAR": "2026", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "FEEDBACK_COLLECTION_ENABLED": "0", "FEEDBACK_INDEX_WORKER_ENABLED": "0", "FEEDBACK_RETRIEVAL_ENABLED": "0",
    "RERANKER_BACKEND": "gemini_listwise", "GEMINI_RETRIEVAL_BACKEND": "vertex", "GCP_PROJECT_ID": "v64-blocked-project",
    "GCP_LOCATION": "global",
}
os.environ.update(D_ENV)

import v64_runtime  # noqa: E402

v64_runtime.install_guards()
GUARD = v64_runtime.verify_guards()
v64_runtime.install_lyrics_snapshot()

import src.backend.api.dependencies as deps  # noqa: E402

deps.get_reranker = deps.singleton(v64_runtime.make_fake_reranker)  # type: ignore[assignment]
from src.retrieval import search_router as sr  # noqa: E402
from src.retrieval.analysis_cache import load_cache  # noqa: E402

SETS = {
    "v06": ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v06_dev_vertex.json",
    "v09": ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v09_dev_vertex.json",
    "v10": ROOT / "experiments/reranking/results_v49_vertex_baseline/analysis_cache_v10_dev_vertex.json",
}
PER_CATEGORY = 10
LYRIC_KINDS = {"verbatim", "partial", "phonetic"}


def main() -> int:
    router = deps.get_search_router()
    rows = []
    caches = {s: load_cache(p) for s, p in SETS.items()}
    for set_name, cache in caches.items():
        for qid in sorted(cache.entries):
            entry = cache.entries[qid]
            analysis = cache.get(qid, entry["query"])
            lyric = any(c.kind in LYRIC_KINDS for c in analysis.lyric_clues)
            attr_filter, decision = sr._build_attribute_filter_with_policy(analysis, [])
            category, n_hits, probe_ms = "closed", 0, None
            if attr_filter is not None:
                t0 = time.perf_counter()
                hits = router._search_text(analysis, sr._attr_filter_k(), metadata_filter=attr_filter)
                probe_ms = round((time.perf_counter() - t0) * 1000, 1)
                n_hits = len(hits)
                category = "open_hits" if n_hits else "open_empty"
            era = analysis.release_era
            rows.append({
                "query_id": qid, "set": set_name, "query": entry["query"], "intent_type": analysis.intent_type,
                "lyric_path": int(lyric), "category": category, "attr_hits": n_hits, "probe_ms": probe_ms,
                "genre_decision": getattr(decision, "reason", ""), "use_genre": getattr(decision, "use_genre", ""),
                "era": f"{era.start_year}-{era.end_year}" if analysis.has_release_era else "",
                "vocal_gender": analysis.vocal_gender or "", "genre": analysis.genre or "",
                "filter": json.dumps(attr_filter, ensure_ascii=False) if attr_filter else "",
            })
    with (HERE / "query_categories.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sample: dict = {}
    for category in ("closed", "open_hits", "open_empty"):
        pools = {s: [r for r in rows if r["category"] == category and not r["lyric_path"] and r["set"] == s] for s in SETS}
        picked = []
        while len(picked) < PER_CATEGORY and any(pools.values()):
            for s in SETS:
                if pools[s] and len(picked) < PER_CATEGORY:
                    picked.append(pools[s].pop(0))
        sample[category] = [{"query_id": r["query_id"], "set": r["set"], "query": r["query"], "attr_hits": r["attr_hits"]} for r in picked]
    synthetic = []
    for item in sample.get("open_hits", []):
        cache = caches[item["set"]]
        entry = cache.entries[item["query_id"]]
        analysis = cache.get(item["query_id"], entry["query"])
        era = analysis.release_era.model_copy(update={"start_year": 1960, "end_year": 1965})
        synthetic_analysis = analysis.model_copy(update={"release_era": era})
        attr_filter, _decision = sr._build_attribute_filter_with_policy(synthetic_analysis, [])
        n_hits = len(router._search_text(synthetic_analysis, sr._attr_filter_k(), metadata_filter=attr_filter)) if attr_filter else -1
        synthetic.append({
            "query_id": item["query_id"], "set": item["set"], "query": entry["query"], "attr_hits": n_hits,
            "filter_open": attr_filter is not None, "prior_analysis": synthetic_analysis.model_dump(mode="json"),
        })
    sample["open_empty_synthetic"] = [s for s in synthetic if s["filter_open"] and s["attr_hits"] == 0]
    counts = {c: sum(1 for r in rows if r["category"] == c) for c in ("closed", "open_hits", "open_empty")}
    counts_nolyric = {c: sum(1 for r in rows if r["category"] == c and not r["lyric_path"]) for c in counts}
    summary = {
        "code": "dd5cc3e", "universe": len(rows), "excluded_lyric_path": sum(r["lyric_path"] for r in rows),
        "category_counts_all": counts, "category_counts_after_exclusion": counts_nolyric,
        "selected": {c: len(v) for c, v in sample.items()}, "per_category_target": PER_CATEGORY,
        "d_env": {k: D_ENV[k] for k in list(D_ENV)[:5]}, "qdrant_url": os.environ["QDRANT_URL"],
        "selection_rule": "범주별 v06→v09→v10 번갈아, 세트 안 query_id 순, 가사 경로 질의 제외(v56과 같다)",
        "guard": GUARD, "network_attempts": len(v64_runtime.NET_ATTEMPTS), "blocked_client_calls": list(v64_runtime.BLOCKED_CALLS),
        "sample": sample,
    }
    (HERE / "latency_sample.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, ensure_ascii=False, indent=1))
    for c, items in sample.items():
        print(c, [i["query_id"] for i in items])
    router.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
