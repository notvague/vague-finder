"""지연 표본 선정 — dev 176건을 D 설정의 보조 경로 기준으로 세 범주로 나눈다(모델 호출 없음, 사전 조회만).

범주(§5, D = ATTR_FILTER_WEIGHT 0.5 · era_plus · K 30 · USE_GENRE 1 · conditional, 1턴 · 답 없음):
  closed      `_build_attribute_filter_with_policy` 가 None — 보조 경로가 닫힌다(A와 같은 호출 수)
  open_hits   필터가 만들어지고 서비스와 같은 `SearchRouter._search_text(..., metadata_filter)` 가 1곡 이상 돌려준다
  open_empty  필터는 만들어지지만 0곡 — 경로는 열렸는데 결과가 없는 경우(비용만 드는가?)

제외 규칙(선정 전에 고정): 분석의 lyric_clues 에 verbatim/partial/phonetic 이 있는 질의는 가사 경로(Mongo 가사 판, TTL 300초 ·
재적재 약 350초)를 타므로 지연 표본에서 뺀다 — 보조 경로와 무관한 적재가 A/D 비교를 가린다. 제외 건수를 기록한다.
선정: 범주별로 v06 → v09 → v10 세트를 번갈아(세트 안은 query_id 순) 10건. 부족하면 전부.
사전 조회의 지연은 참고로만 적고 §5 측정값으로 쓰지 않는다.

실행(Qdrant 서버가 떠 있어야 한다):
  QDRANT_URL=http://127.0.0.1:6356 venv/bin/python experiments/reranking/results_v56_attr_runtime_validation/classify_queries.py
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

import v56_runtime  # noqa: E402

v56_runtime.block_real_models()
v56_runtime.verify_guard()

D_ENV = {
    "ATTR_FILTER_WEIGHT": "0.5", "ATTR_FILTER_GATE": "era_plus", "ATTR_FILTER_K": "30",
    "ATTR_FILTER_USE_GENRE": "1", "ATTR_FILTER_GENRE_POLICY": "conditional",
}
os.environ.update(D_ENV)
os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6356")
os.environ.setdefault("SEARCH_REFERENCE_YEAR", "2026")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["FEEDBACK_COLLECTION_ENABLED"] = "0"

import src.backend.api.dependencies as deps  # noqa: E402

deps.get_reranker = deps.singleton(v56_runtime.make_fake_reranker)  # type: ignore[assignment]
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
    for set_name, path in SETS.items():
        cache = load_cache(path)
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
                "genre_decision": decision.reason if decision else "", "use_genre": decision.use_genre if decision else "",
                "era": f"{era.start_year}-{era.end_year}" if analysis.has_release_era else "",
                "vocal_gender": analysis.vocal_gender or "", "genre": analysis.genre or "",
                "filter": json.dumps(attr_filter, ensure_ascii=False) if attr_filter else "",
            })
    out_csv = HERE / "query_categories.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # 선정: 범주별 세트 번갈아
    sample: dict = {}
    for category in ("closed", "open_hits", "open_empty"):
        pools = {s: [r for r in rows if r["category"] == category and not r["lyric_path"] and r["set"] == s] for s in SETS}
        picked = []
        while len(picked) < PER_CATEGORY and any(pools.values()):
            for s in SETS:
                if pools[s] and len(picked) < PER_CATEGORY:
                    picked.append(pools[s].pop(0))
        sample[category] = [{"query_id": r["query_id"], "set": r["set"], "query": r["query"], "attr_hits": r["attr_hits"]} for r in picked]
    # 합성 open_empty — dev 176에 자연 사례가 없을 때를 위한 보조 표본. open_hits 표본의 분석에서 **시기 창만** 곡이 없는
    # 연도(1960~1965, 서버 count 0 확인)로 바꾼 사본을 prior_analysis 로 보낸다(모델 호출 없음, 서비스의 정규 입력 경로).
    # A·D 양쪽에 같은 분석이 가므로 비교는 성립하지만 자연 질의가 아니다 — 결과에 '합성'으로 표시한다.
    synthetic = []
    by_id = {}
    for set_name, path in SETS.items():
        cache = load_cache(path)
        for qid, entry in cache.entries.items():
            by_id[qid] = (cache, entry)
    for item in sample.get("open_hits", []):
        cache, entry = by_id[item["query_id"]]
        analysis = cache.get(item["query_id"], entry["query"])
        era = analysis.release_era.model_copy(update={"start_year": 1960, "end_year": 1965})
        synthetic_analysis = analysis.model_copy(update={"release_era": era})
        attr_filter, decision = sr._build_attribute_filter_with_policy(synthetic_analysis, [])
        n_hits = len(router._search_text(synthetic_analysis, sr._attr_filter_k(), metadata_filter=attr_filter)) if attr_filter else -1
        synthetic.append({
            "query_id": item["query_id"], "set": item["set"], "query": entry["query"], "attr_hits": n_hits,
            "filter_open": attr_filter is not None, "prior_analysis": synthetic_analysis.model_dump(mode="json"),
        })
    sample["open_empty_synthetic"] = [s for s in synthetic if s["filter_open"] and s["attr_hits"] == 0]
    counts = {c: sum(1 for r in rows if r["category"] == c) for c in ("closed", "open_hits", "open_empty")}
    counts_nolyric = {c: sum(1 for r in rows if r["category"] == c and not r["lyric_path"]) for c in counts}
    summary = {
        "universe": len(rows), "excluded_lyric_path": sum(r["lyric_path"] for r in rows),
        "category_counts_all": counts, "category_counts_after_exclusion": counts_nolyric,
        "selected": {c: len(v) for c, v in sample.items()}, "per_category_target": PER_CATEGORY,
        "d_env": D_ENV, "qdrant_url": os.environ["QDRANT_URL"], "selection_rule": "범주별 v06→v09→v10 번갈아, 세트 안 query_id 순, 가사 경로 질의 제외",
        "sample": sample,
    }
    v56_runtime.dump_json(HERE / "latency_sample.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, ensure_ascii=False, indent=1))
    for c, items in sample.items():
        print(c, [i["query_id"] for i in items])
    router.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
