"""v78 무료 검증 §4-2 — 제품 모듈 단독 결과가 v77 실험 검색기 결과와 같은지(모델 호출 0 · 로컬 KoE5만 · 정답 미사용).

dev 176 각 질의에 대해 제품 임베더(`KoE5Embedder()`, 운영과 같은 생성 — revision 미지정)로 `"query: " + search_text`를
임베딩하고, 제품 저장소(`DescriptionSemanticStore`)의 위 10묶음(묶음 · 근거 청크 · 점수 6자리)과 대표곡(곡 · 규칙)을
v77 R1 기록(`rankings.jsonl.gz`의 `S`·`S_reps`)과 비교한다. v77 방식(`SentenceTransformer(revision=237b580a)`)의 질의 벡터와
제품 임베더 벡터의 차도 잰다.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 <venv>/bin/python parity_v77.py --v77-rankings <v77>/runs/R1/rankings.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
SETS = ("v06", "v09", "v10")
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v77-rankings", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "analysis/parity_v77.json")
    args = ap.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        print("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1이 필요하다", file=sys.stderr)
        return 2
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from src.embedding.models.text_koe5 import KoE5Embedder
    from src.retrieval import description_semantic_aux as dsa
    from src.retrieval.analysis_cache import load_cache
    from src.retrieval.search_router import _exact_lyric_terms

    raw = args.v77_rankings.read_bytes()
    v77 = {}
    with gzip.open(args.v77_rankings, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                v77[r["qid"]] = r
    t0 = time.perf_counter()
    store = dsa.DescriptionSemanticStore.load(ROOT / dsa.DEFAULT_RUNTIME_DIR)
    load_s = time.perf_counter() - t0
    product = KoE5Embedder()
    pinned = SentenceTransformer(dsa.MODEL_NAME, revision=dsa.MODEL_REVISION)

    rows = []
    for set_name in SETS:
        cache = load_cache(CACHES / f"analysis_cache_{set_name}_dev_vertex.json")
        for qid in sorted(cache.entries):
            analysis = cache.get(qid, cache.entries[qid]["query"])
            text = dsa.QUERY_PREFIX + analysis.search_text
            qv = product.embed_passages([text], add_e5_prefix=False, normalize=True)[0].astype(np.float32)
            pv = pinned.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            hits = store.search(qv)
            ref = v77[qid]
            mine_groups = [[h.group, round(h.score, 6), h.chunk] for h in hits]
            ref_groups = [list(g) for g in ref["S"][: dsa.GROUPS_K]]
            mine_reps = [[h.group, h.representative or None, h.rule] for h in hits]
            ref_reps = [list(r) for r in ref["S_reps"]]
            rows.append({
                "set": set_name, "qid": qid, "exact_lyric": bool(_exact_lyric_terms(analysis)),
                "input_sha16_matches_v77": hashlib.sha256(text.encode()).hexdigest()[:16] == ref["query_input_sha16"],
                "groups_same": [g[0] for g in mine_groups] == [g[0] for g in ref_groups],
                "chunks_same": [g[2] for g in mine_groups] == [g[2] for g in ref_groups],
                "scores_same_6dp": [g[1] for g in mine_groups] == [g[1] for g in ref_groups],
                "max_score_abs_diff": max(abs(a[1] - b[1]) for a, b in zip(mine_groups, ref_groups)),
                "reps_same": mine_reps == ref_reps,
                "vector_max_abs_diff_vs_pinned": float(np.abs(qv - pv).max()),
                "rules": [h.rule for h in hits],
            })
    applicable = [r for r in rows if not r["exact_lyric"]]

    def count(rs, key):
        return sum(1 for r in rs if r[key])

    out = {
        "v77_rankings_sha256_16": hashlib.sha256(raw).hexdigest()[:16],
        "runtime_fingerprint": store.fingerprint(),
        "store_load_seconds": round(load_s, 3),
        "queries": len(rows), "applicable": len(applicable),
        "all": {k: count(rows, k) for k in ("input_sha16_matches_v77", "groups_same", "chunks_same", "scores_same_6dp",
                                            "reps_same")},
        "applicable_counts": {k: count(applicable, k) for k in ("input_sha16_matches_v77", "groups_same", "chunks_same",
                                                                 "scores_same_6dp", "reps_same")},
        "max_score_abs_diff": max(r["max_score_abs_diff"] for r in rows),
        "max_vector_abs_diff_product_vs_pinned": max(r["vector_max_abs_diff_vs_pinned"] for r in rows),
        "rule_counts_top10": {},
        "mismatches": [r["qid"] for r in rows if not (r["groups_same"] and r["chunks_same"] and r["reps_same"])],
    }
    for r in rows:
        for rule in r["rules"]:
            out["rule_counts_top10"][rule] = out["rule_counts_top10"].get(rule, 0) + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "runtime_fingerprint"}, ensure_ascii=False))
    return 0 if not out["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
