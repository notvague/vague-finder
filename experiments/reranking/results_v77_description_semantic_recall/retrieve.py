"""v77 검색기(실험 전용) — 정답 라벨을 읽지 않는다.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=<worktree> venv/bin/python retrieve.py --run R1

입력: 분석 캐시(v49 Vertex, dev 176) · 코퍼스 · v77 색인 · 기존 텍스트 dense 벡터 · v76 잔여 산출물(비교용).
출력(`runs/<run>/`, 질의별 원기록은 Git 제외): `rankings.jsonl.gz` · `run_meta.json`.

경로(RUN_INFO §2-6):
- S: `query: search_text` KoE5 ↔ 소개 원문 청크, 묶음 최댓값
- L: `korean_tags` BM25 ↔ 같은 청크(같은 자료의 낱말 대조군)
- V: v76 잔여 낱말 1단계(v76 코드 사본 · 산출물 그대로, 비교용)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desc_semantic as ds  # noqa: E402

V76_ROOT = Path(os.environ.get("V76_WORKTREE", "/private/tmp/vague-finder-v76-evidence-recall"))
V76_MODULE = V76_ROOT / "artifacts/src_v76_f3c2e1b737479bdc/src/retrieval/description_aux.py"
V76_STORE = V76_ROOT / "artifacts/description_aux/description_residual_v1.json.gz"


def _rss_now_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 1024


def _maxrss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _exact_lyric(analysis) -> bool:
    """운영 `search_router._exact_lyric_terms`가 비지 않는 질의(가사 전용 텍스트 질의로 바뀌는 질의)와 같은 조건."""
    clues = [c for c in (analysis.lyric_clues or []) if getattr(c, "kind", None) in {"verbatim", "partial"}]
    return bool(clues or list(analysis.lyric_keywords or []))


def _load_v76():
    spec = importlib.util.spec_from_file_location("v76_description_aux", V76_MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v76_description_aux"] = mod
    spec.loader.exec_module(mod)
    store = mod.DescriptionResidualStore.load(V76_STORE)
    song_group = dict(zip(store._ids, store._group))  # noqa: SLF001 — 읽기 전용 비교
    return mod, store, song_group


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        print("HF_HUB_OFFLINE=1 · TRANSFORMERS_OFFLINE=1 이 필요하다", file=sys.stderr)
        return 2
    out = HERE / "runs" / args.run
    if out.exists() and any(out.iterdir()):
        print(f"이미 있다 — 덮어쓰지 않는다: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rss = {"start": _rss_now_mb()}

    manifest = json.loads((ds.INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    songs, groups, _ = ds.load_corpus()
    rows = ds.ordered_chunks(groups)
    chunk_list_sha = ds.sha256_bytes("\n".join(f"{g}\t{c}\t{ds.sha256_bytes(t.encode())}" for g, c, t in rows).encode())
    if chunk_list_sha != manifest["fingerprints"]["chunk_list_sha256"]:
        raise SystemExit("청크 목록 지문이 색인과 다르다 — 규칙·코퍼스가 바뀌었다")
    if ds.sha256_file(ds.CORPUS) != manifest["inputs"]["corpus_sha256"]:
        raise SystemExit("코퍼스 지문이 색인과 다르다")

    rss["before_index"] = _rss_now_mb()
    t0 = time.perf_counter()
    vecs = np.load(ds.INDEX_DIR / "vectors.npy")
    if ds.sha256_file(ds.INDEX_DIR / "vectors.npy") != manifest["fingerprints"]["vectors_sha256"]:
        raise SystemExit("벡터 지문이 다르다")
    chunk_rows = ds.read_jsonl_gz(ds.INDEX_DIR / "chunks.jsonl.gz")
    row_group = [r["group"] for r in chunk_rows]
    row_chunk = [r["chunk"] for r in chunk_rows]
    row_u = np.asarray([r["u"] for r in chunk_rows], dtype=np.int64)
    row_text = [r["text"] for r in chunk_rows]
    group_row_start = {}
    for i, (g, c) in enumerate(zip(row_group, row_chunk)):
        group_row_start.setdefault(g, {})[c] = i
    t_index_load = time.perf_counter() - t0
    rss["after_index"] = _rss_now_mb()

    t0 = time.perf_counter()
    bm25_tok = {r["u"]: r["t"] for r in ds.read_jsonl_gz(ds.INDEX_DIR / "bm25_tokens.jsonl.gz")}
    bm25 = ds.BM25([bm25_tok[int(u)] for u in row_u])
    t_bm25_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    dense_ids, dense_mat, dense_missing = [], [], []
    for sid in sorted(songs, key=ds.song_sort_key):
        p = ds.TEXT_DENSE_DIR / f"{sid}.npy"
        if p.is_file():
            dense_ids.append(sid)
            dense_mat.append(np.load(p).astype(np.float32).reshape(-1))
        else:
            dense_missing.append(sid)
    dense_mat = np.stack(dense_mat)
    dense_row = {sid: i for i, sid in enumerate(dense_ids)}
    t_dense_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    v76mod, v76store, v76_song_group = _load_v76()
    t_v76_load = time.perf_counter() - t0
    rss["after_all_indexes"] = _rss_now_mb()

    from src.embedding.text.korean_bm25_tokenizer import normalize_for_bm25
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(ds.MODEL_NAME, revision=ds.MODEL_REVISION)
    t_model_load = time.perf_counter() - t0
    rss["after_model"] = _rss_now_mb()
    # 예열 1회(기록하지 않음) — 첫 질의가 장치 초기화 비용을 떠안지 않게
    model.encode([ds.QUERY_PREFIX + "예열"], normalize_embeddings=True, convert_to_numpy=True)

    def reps_for(ranked, qv):
        out_reps = []
        for gid, _sc, ci in ranked[: ds.GROUPS_K]:
            group = groups[gid]
            idx = [dense_row[s] for s in group.songs if s in dense_row]
            scores = dense_mat[idx] @ qv if idx else np.zeros(0, dtype=np.float32)
            dscore = {s: float(v) for s, v in zip([s for s in group.songs if s in dense_row], scores)}
            text = row_text[group_row_start[gid][ci]]
            sid, rule = ds.pick_representative(group, text, songs, dscore)
            out_reps.append([gid, sid, rule])
        return out_reps

    records = []
    for set_name, qid, analysis, fallback in ds.iter_queries():
        rec = {"set": set_name, "qid": qid, "fallback": fallback}
        if analysis is None:
            rec["status"] = "analysis_missing"
            records.append(rec)
            continue
        rec["status"] = "ok"
        rec["exact_lyric"] = _exact_lyric(analysis)
        qtext = ds.QUERY_PREFIX + analysis.search_text
        rec["query_input_sha16"] = ds.sha256_bytes(qtext.encode("utf-8"))[:16]

        t0 = time.perf_counter()
        qv = model.encode([qtext], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        uscores = vecs @ qv
        rscores = uscores[row_u]
        s_ranked = ds.rank_groups(row_group, row_chunk, rscores, ds.DIAG_DEPTH)
        s_reps = reps_for(s_ranked, qv)
        t_s = time.perf_counter() - t0

        terms = list(dict.fromkeys(t for t in normalize_for_bm25(" ".join(
            str(x) for x in (analysis.korean_tags or []) if str(x).strip())).split() if t)) if analysis.korean_tags else []
        t0 = time.perf_counter()
        l_ranked = ds.rank_groups(row_group, row_chunk, bm25.scores(terms), ds.DIAG_DEPTH, require_positive=True)
        l_reps = reps_for(l_ranked, qv)
        t_l = time.perf_counter() - t0

        v_terms = v76mod.query_terms(analysis)
        hits = v76store.evidence(v_terms)
        v_groups = v76mod.top_groups(hits, k=ds.DIAG_DEPTH)

        rec.update({
            "S": [[g, round(float(s), 6), c] for g, s, c in s_ranked],
            "S_reps": s_reps,
            "L": [[g, round(float(s), 6), c] for g, s, c in l_ranked],
            "L_reps": l_reps,
            "L_terms": len(terms),
            "V": [[grp.group, round(float(grp.score), 6)] for grp in v_groups],
            "V_evidence_songs": len(hits),
            "ms": {"encode": round(t_enc * 1000, 3), "S_search": round(t_s * 1000, 3), "L_search": round(t_l * 1000, 3)},
        })
        records.append(rec)

    ds.write_jsonl_gz(out / "rankings.jsonl.gz", records)
    # 질의별 근거 청크 원문은 Git 제외 파일에만(사람 검토용)
    ds.write_jsonl_gz(out / "evidence_text.jsonl.gz", (
        {"qid": r["qid"], "S_top10": [[g, row_text[group_row_start[g][c]]] for g, _s, c in r.get("S", [])[:10]]}
        for r in records if r["status"] == "ok"))
    enc = sorted(r["ms"]["encode"] for r in records if r["status"] == "ok")
    srch = sorted(r["ms"]["S_search"] for r in records if r["status"] == "ok")
    med = lambda a: a[len(a) // 2] if a else None  # noqa: E731
    p90 = lambda a: a[min(len(a) - 1, int(len(a) * 0.9))] if a else None  # noqa: E731
    meta = {
        "run": args.run, "started_at": started, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "worktree_head": subprocess.run(["git", "-C", str(ds.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "index_manifest_sha256": ds.sha256_file(ds.INDEX_DIR / "manifest.json"),
        "index_vectors_sha256": manifest["fingerprints"]["vectors_sha256"],
        "code_sha256": {n: ds.sha256_file(HERE / n) for n in ("desc_semantic.py", "retrieve.py")},
        "analysis_caches_sha256": {s: ds.sha256_file(ds.CACHES / f"analysis_cache_{s}_dev_vertex.json") for s in ds.SETS},
        "v76": {"module_sha256": ds.sha256_file(V76_MODULE), "store_sha256": ds.sha256_file(V76_STORE)},
        "model": {"name": ds.MODEL_NAME, "revision": ds.MODEL_REVISION, "device": str(model.device)},
        "queries": len(records), "status_counts": {s: sum(1 for r in records if r["status"] == s) for s in {r["status"] for r in records}},
        "text_dense": {"loaded": len(dense_ids), "missing": dense_missing},
        "timing": {
            "index_load_s": round(t_index_load, 3), "bm25_build_s": round(t_bm25_build, 3),
            "text_dense_load_s": round(t_dense_load, 3), "v76_load_s": round(t_v76_load, 3),
            "model_load_s": round(t_model_load, 3),
            "encode_ms_median": med(enc), "encode_ms_p90": p90(enc), "encode_ms_max": enc[-1] if enc else None,
            "S_search_ms_median": med(srch), "S_search_ms_p90": p90(srch), "S_search_ms_max": srch[-1] if srch else None,
        },
        "rss_mb": {**{k: round(v, 1) for k, v in rss.items()}, "index_increase": round(rss["after_index"] - rss["before_index"], 1),
                   "max": round(_maxrss_mb(), 1)},
        "index_matrix_mb": round(vecs.nbytes / 1e6, 2),
        "external_calls": 0,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("queries", "status_counts", "timing", "rss_mb")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
