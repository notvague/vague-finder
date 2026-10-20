"""v77 앨범 소개 원문 의미 색인 생성기(실험 전용, 로컬 KoE5만).

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=<worktree> venv/bin/python build_index.py

산출물(Git 제외 경로 `artifacts/description_semantic/desc_semantic_v1/`):
- `chunks.jsonl.gz`  청크 행(묶음 id · 청크 번호 · 고유 벡터 행 · 토큰 수 · BM25 토큰 · 원문) — 원문 포함, 커밋하지 않는다
- `groups.json.gz`   묶음 → 곡 목록 · 강제 분할 수
- `vectors.npy`      고유 청크 벡터(float32, L2 정규화)
- `manifest.json`    지문 · 규칙 · 모델 · 시간 · 자원
이미 있으면 덮어쓰지 않는다.
"""
from __future__ import annotations

import json
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import desc_semantic as ds  # noqa: E402


def _rss_mb() -> float:
    # macOS ru_maxrss는 바이트
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main() -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        print("HF_HUB_OFFLINE=1 · TRANSFORMERS_OFFLINE=1 이 필요하다(새 다운로드 금지)", file=sys.stderr)
        return 2
    out = ds.INDEX_DIR
    if out.exists() and any(out.iterdir()):
        print(f"이미 있다 — 덮어쓰지 않는다: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    corpus_sha = ds.sha256_file(ds.CORPUS)
    songs, groups, _texts = ds.load_corpus()
    rows = ds.ordered_chunks(groups)
    no_desc = sorted((s for s, v in songs.items() if v.group is None), key=ds.song_sort_key)
    t_chunk = time.perf_counter() - t_all

    # 고유 청크(글자 같으면 벡터 하나)
    unique_texts: list[str] = []
    unique_index: dict[str, int] = {}
    row_unique: list[int] = []
    for _gid, _ci, chunk in rows:
        k = unique_index.get(chunk)
        if k is None:
            k = len(unique_texts)
            unique_index[chunk] = k
            unique_texts.append(chunk)
        row_unique.append(k)
    chunk_groups_per_text: dict[int, set] = {}
    for (gid, _ci, _c), k in zip(rows, row_unique):
        chunk_groups_per_text.setdefault(k, set()).add(gid)
    shared_texts = sum(1 for v in chunk_groups_per_text.values() if len(v) > 1)

    from src.embedding.text.korean_bm25_tokenizer import normalize_for_bm25

    t0 = time.perf_counter()
    bm25_tokens = [normalize_for_bm25(c).split() for c in unique_texts]
    t_tok = time.perf_counter() - t0

    from sentence_transformers import SentenceTransformer
    import torch

    t0 = time.perf_counter()
    model = SentenceTransformer(ds.MODEL_NAME, revision=ds.MODEL_REVISION)
    t_load = time.perf_counter() - t0
    device = str(model.device)
    max_seq = int(model.max_seq_length)
    tok = model.tokenizer
    passages = [ds.PASSAGE_PREFIX + c for c in unique_texts]
    token_counts = [len(tok(p, add_special_tokens=True)["input_ids"]) for p in passages]
    truncated = sum(1 for n in token_counts if n > max_seq)

    t0 = time.perf_counter()
    vecs = model.encode(passages, batch_size=ds.EMBED_BATCH, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    t_embed = time.perf_counter() - t0

    # 재현 점검: 고정 seed 77로 200개를 다른 배치 구성(무작위 순서)으로 다시 임베딩
    rng = random.Random(77)
    sample = sorted(rng.sample(range(len(passages)), min(200, len(passages))))
    shuffled = sample[:]
    rng.shuffle(shuffled)
    again = model.encode([passages[i] for i in shuffled], batch_size=ds.EMBED_BATCH,
                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    diffs = np.abs(again - vecs[shuffled]).max(axis=1)
    cos = (again * vecs[shuffled]).sum(axis=1)

    np.save(out / "vectors.npy", vecs)
    chunk_rows = [
        {"group": gid, "chunk": ci, "u": k, "tokens": token_counts[k], "chars": len(chunk), "text": chunk}
        for (gid, ci, chunk), k in zip(rows, row_unique)
    ]
    ds.write_jsonl_gz(out / "chunks.jsonl.gz", chunk_rows)
    ds.write_jsonl_gz(out / "bm25_tokens.jsonl.gz", ({"u": k, "t": t} for k, t in enumerate(bm25_tokens)))
    ds.write_jsonl_gz(out / "groups.jsonl.gz", (
        {"group": gid, "songs": groups[gid].songs, "chunks": len(groups[gid].chunks),
         "forced_splits": groups[gid].forced_splits} for gid in sorted(groups)))

    chunk_list_sha = ds.sha256_bytes("\n".join(f"{g}\t{c}\t{ds.sha256_bytes(t.encode())}" for g, c, t in rows).encode())
    tc = sorted(token_counts)
    q = lambda p: tc[min(len(tc) - 1, int(len(tc) * p))]  # noqa: E731
    group_sizes = sorted(len(g.songs) for g in groups.values())
    model_dir = Path.home() / ".cache/huggingface/hub/models--nlpai-lab--KoE5/snapshots" / ds.MODEL_REVISION
    manifest = {
        "rule_version": ds.RULE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {
            "corpus": "<main checkout>/data/all_songs.jsonl", "corpus_sha256": corpus_sha,
            "code_sha256": {name: ds.sha256_file(Path(__file__).resolve().parent / name)
                            for name in ("desc_semantic.py", "build_index.py")},
            "bm25_tokenizer_sha256": ds.sha256_file(ds.ROOT / "src/embedding/text/korean_bm25_tokenizer.py"),
        },
        "model": {
            "name": ds.MODEL_NAME, "revision": ds.MODEL_REVISION,
            "safetensors_blob": os.path.basename(os.path.realpath(model_dir / "model.safetensors")),
            "device": device, "max_seq_length": max_seq, "prefix": ds.PASSAGE_PREFIX,
            "normalize": True, "batch_size": ds.EMBED_BATCH, "dim": int(vecs.shape[1]),
        },
        "rules": {"max_chunk_chars": ds.MAX_CHUNK_CHARS, "sentence_split": r"(?<=[.!?。])\s+",
                  "dedup_within_group": True, "identity_prefix": None},
        "counts": {
            "songs": len(songs), "songs_without_description": len(no_desc),
            "groups": len(groups), "groups_shared": sum(1 for g in groups.values() if len(g.songs) > 1),
            "songs_in_shared_groups": sum(len(g.songs) for g in groups.values() if len(g.songs) > 1),
            "group_size_max": group_sizes[-1],
            "chunk_rows": len(rows), "unique_chunks": len(unique_texts),
            "unique_chunks_in_multiple_groups": shared_texts,
            "chunks_per_group_median": sorted(len(g.chunks) for g in groups.values())[len(groups) // 2],
            "forced_split_sentences": sum(g.forced_splits for g in groups.values()),
            "tokens_median": q(0.5), "tokens_p90": q(0.9), "tokens_p99": q(0.99), "tokens_max": tc[-1],
            "truncated_over_max_seq": truncated,
        },
        "fingerprints": {
            "chunk_list_sha256": chunk_list_sha,
            "vectors_sha256": ds.sha256_file(out / "vectors.npy"),
            "chunks_file_sha256": ds.sha256_file(out / "chunks.jsonl.gz"),
            "groups_file_sha256": ds.sha256_file(out / "groups.jsonl.gz"),
            "bm25_tokens_file_sha256": ds.sha256_file(out / "bm25_tokens.jsonl.gz"),
        },
        "reproducibility_sample": {
            "seed": 77, "n": len(sample), "max_abs_diff": float(diffs.max()),
            "median_abs_diff": float(np.median(diffs)), "min_cosine": float(cos.min()),
        },
        "timing_seconds": {"corpus_and_chunking": round(t_chunk, 3), "bm25_tokenize": round(t_tok, 3),
                           "model_load": round(t_load, 3), "embed": round(t_embed, 3),
                           "total": round(time.perf_counter() - t_all, 3)},
        "resources": {"max_rss_mb": round(_rss_mb(), 1),
                      "vectors_mb": round((out / "vectors.npy").stat().st_size / 1e6, 2),
                      "chunks_gz_mb": round((out / "chunks.jsonl.gz").stat().st_size / 1e6, 2)},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "sentence_transformers": __import__("sentence_transformers").__version__,
                    "transformers": __import__("transformers").__version__, "numpy": np.__version__,
                    "kiwipiepy": __import__("kiwipiepy").__version__, "machine": platform.machine()},
        "external_calls": 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "timing_seconds", "resources", "reproducibility_sample")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
