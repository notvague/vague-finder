"""
pipelines/embed_songs.py
- 임베딩 파이프라인(곡 단위)

목표:
- songs 리스트를 받아 각 song에
  - song["text_dense_values"] (Ko-E5 임베딩)
  - song["text_sparse_values"] (BM25 임베딩)
  - song["image_values"] (SigLIP2 임베딩)
  을 채운 뒤 반환

[변경사항]
    BM25 기반 text_sparse_values 생성 추가
- Dense(Ko-E5)용 passage와 Sparse(BM25)용 passage를 분리
  - build_dense_passage(): Ko-E5 임베딩에만 사용
  - build_sparse_passage(): BM25 임베딩에만 사용
- verbose_passage 출력도 dense/sparse를 각각 확인 가능하도록 개선

[확장]
- 오디오 임베딩(CLAP) 추가:
  - metadata["audio_path"]에 로컬 오디오 경로가 주입되어 있으면 로드/임베딩
  - 실패 시 song["audio_values"] = None

  [스위치 추가]
- CPU 메모리 피크(OOM, Killed) 방지 및 단계별 실행을 위해 모달리티별 스위치 제공
- 텍스트는 dense/sparse를 분리하지 않고 "TEXT" 스위치 하나로 묶음

환경변수 (기본값 "1" = 기존처럼 한번에 전체 실행):
- EMBED_TEXT=1/0   -> Ko-E5(dense) + BM25(sparse) 둘 다 수행
- EMBED_IMAGE=1/0
- EMBED_AUDIO=1/0

실행 예)
- 텍스트만:
  EMBED_IMAGE=0 EMBED_AUDIO=0
- 이미지/오디오만:
  EMBED_TEXT=0
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import shutil
import numpy as np

from src.embedding.embedding_cache import (
    EmbeddingCacheKey,
    save_npz_sparse,
)

from src.embedding.image.image_io import fetch_image
from src.embedding.models.image_siglip2 import SigLIP2Embedder
from src.embedding.models.text_koe5 import KoE5Embedder
from src.embedding.models.text_bm25 import BM25SparseEncoder
from src.embedding.text.passage_builder import build_dense_passage, build_sparse_passage
from src.embedding.audio.audio_io import (
    AUDIO_DECODER,
    AUDIO_TARGET_SAMPLE_RATE,
    load_audio_mono_segments,
)
from src.embedding.image.normalize import l2_normalize_np
from src.embedding.models.audio_clap import CLAP_FEATURE_SEED, CLAPAudioEmbedder
from src.embedding.audio.mean_center import (
    apply_mean_center,
    compute_dataset_signature,
    compute_preprocess_fingerprint,
    get_or_create_mean_for_dataset,
)

TEXT_EMBED_BATCH_SIZE = int(os.getenv("TEXT_EMBED_BATCH_SIZE", "64"))
BM25_PARAMS_PATH = Path(os.getenv("BM25_PARAMS_PATH", "artifacts/bm25_params.json"))
ARTIFACTS_DIR = Path(os.getenv("VAGUEFINDER_ARTIFACTS_DIR", "artifacts"))
TEXT_SPARSE_MODEL_TAG = os.getenv("TEXT_SPARSE_MODEL_TAG", "bm25")
IMAGE_EMBED_BATCH_SIZE = int(os.getenv("IMAGE_EMBED_BATCH_SIZE", "32"))
AUDIO_EMBED_BATCH_SIZE = int(os.getenv("AUDIO_EMBED_BATCH_SIZE", "8"))

def _flag(name: str, default: str = "1") -> bool:
    """환경변수 플래그 파서."""
    v = os.getenv(name, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

EMBED_TEXT = _flag("EMBED_TEXT", "1")
EMBED_IMAGE = _flag("EMBED_IMAGE", "1")
EMBED_AUDIO = _flag("EMBED_AUDIO", "1")

def _chunked(seq, n: int):
    for i in range(0, len(seq), n):
        yield i, seq[i : i + n]


def _debug_print_passages(song: Dict, dense: str, sparse: str) -> None:
    md = song.get("metadata", {})
    artist = md.get("artist", "N/A")
    title = md.get("title", "N/A")

    print("\n" + "=" * 70)
    print(f"[DEBUG] Build *dense* passage for id: {song.get('id', 'N/A')} | artist: {artist} | title: {title}")
    print("\n- passage preview:\n" + dense)

    print("\n" + "=" * 70)
    print(f"[DEBUG] Build *sparse* passage for id: {song.get('id', 'N/A')} | artist: {artist} | title: {title}")
    print("\n- passage preview:\n" + sparse)

    print("\n" + "=" * 70)


def embed_songs(
    songs: List[Dict],
    *,
    text_embedder: Optional[KoE5Embedder] = None,
    bm25_encoder: Optional[BM25SparseEncoder] = None,
    image_embedder: Optional[SigLIP2Embedder] = None,
    audio_embedder: Optional[CLAPAudioEmbedder] = None,
    verbose_passage: bool = False,
) -> List[Dict]:
    """
    songs 리스트를 받아, 스위치가 켜진 모달리티만 임베딩을 채워서 반환.
    - 텍스트: dense + sparse를 같이 수행(EMBED_TEXT)
    - 이미지: EMBED_IMAGE
    - 오디오: EMBED_AUDIO
    """
    # ---------------------------
    # 1) passages 생성 (TEXT가 켜져있거나 verbose_passage가 켜져있으면 필요)
    # ---------------------------
    dense_passages: List[str] = []
    sparse_passages: List[str] = []
    if EMBED_TEXT or verbose_passage:
        for s in songs:
            d = build_dense_passage(s)
            sp = build_sparse_passage(s)
            dense_passages.append(d)
            sparse_passages.append(sp)

            if verbose_passage:
                _debug_print_passages(s, d, sp)

    # ---------------------------
    # 2) TEXT: Ko-E5 dense + BM25 sparse
    # ---------------------------
    if EMBED_TEXT:
        text_embedder = text_embedder or KoE5Embedder()

        # 2-A) dense
        for start, batch_passages in _chunked(dense_passages, TEXT_EMBED_BATCH_SIZE):
            emb = text_embedder.embed_passages(
                batch_passages,
                add_e5_prefix=True,
                normalize=True,
                batch_size=TEXT_EMBED_BATCH_SIZE,
                show_progress_bar=False,
            )
            for i, vec in enumerate(emb):
                songs[start + i]["text_dense_values"] = np.asarray(vec)

        # 2-B) sparse(BM25)
        bm25_encoder = bm25_encoder or BM25SparseEncoder(params_path=BM25_PARAMS_PATH)
        bm25_encoder.fit(sparse_passages)

        try:
            bm25_encoder.dump()
        except Exception as e:
            print(f"[WARN] BM25 params dump failed: {e}")

        sparse_list = bm25_encoder.encode_documents(sparse_passages)  # list[dict]

        cache_dir = ARTIFACTS_DIR / "embeddings" / "text_sparse" / TEXT_SPARSE_MODEL_TAG
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        for i, sv in enumerate(sparse_list):
            tid = str(songs[i]["id"])
            songs[i]["text_sparse_values"] = sv  # {"indices":[...], "values":[...]}

            indices = np.asarray(sv.get("indices", []), dtype=np.int32)
            values = np.asarray(sv.get("values", []), dtype=np.float32)

            save_npz_sparse(
                ARTIFACTS_DIR,
                EmbeddingCacheKey("text_sparse", TEXT_SPARSE_MODEL_TAG, tid),
                indices=indices,
                values=values,
            )

    # ---------------------------
    # 3) IMAGE: SigLIP2 (STREAMING: batch 단위 fetch->embed)
    # ---------------------------
    if EMBED_IMAGE:
        image_embedder = image_embedder or SigLIP2Embedder()

        # 커버가 있는 곡 인덱스만 모으고(없으면 None 처리)
        image_song_indices: List[int] = []
        for idx, song in enumerate(songs):
            src = song.get("metadata", {}).get("album_cover")
            if not src:
                song["image_values"] = None
                continue
            image_song_indices.append(idx)

        # batch 단위로 "fetch -> 임베딩 -> 저장" (전체 이미지를 RAM에 쌓지 않음)
        for start, batch_indices in _chunked(image_song_indices, IMAGE_EMBED_BATCH_SIZE):
            batch_images = []
            batch_keep_indices = []

            # 1) 이 배치에 해당하는 이미지만 fetch
            for song_idx in batch_indices:
                src = songs[song_idx].get("metadata", {}).get("album_cover")
                if not src:
                    songs[song_idx]["image_values"] = None
                    continue

                try:
                    img = fetch_image(src)
                    batch_images.append(img)
                    batch_keep_indices.append(song_idx)
                except Exception:
                    songs[song_idx]["image_values"] = None

            if not batch_images:
                continue

            # 2) 로드된 것만 임베딩
            vecs = image_embedder.embed_images(batch_images, l2_normalize=True)

            # 3) 결과 저장
            for i, vec in enumerate(vecs):
                songs[batch_keep_indices[i]]["image_values"] = np.asarray(vec)

    # ---------------------------
    # 4) AUDIO: CLAP
    # ---------------------------
    if EMBED_AUDIO:
        audio_embedder = audio_embedder or CLAPAudioEmbedder()

        # 세그먼트 샘플링 파라미터 (환경변수로 조절 가능)
        AUDIO_WINDOW_SECONDS = int(os.getenv("AUDIO_WINDOW_SECONDS", "120"))
        AUDIO_SEGMENT_SECONDS = int(os.getenv("AUDIO_SEGMENT_SECONDS", "12"))
        AUDIO_NUM_SEGMENTS = int(os.getenv("AUDIO_NUM_SEGMENTS", "8"))

        # 곡별 누적합/카운트 (세그먼트 임베딩 평균을 만들기 위함)
        # sum_vecs[song_idx] = Σ segment_vec
        # cnts[song_idx] = segment 개수
        sum_vecs: Dict[int, np.ndarray] = {}
        cnts: Dict[int, int] = {}

        # 배치에 넣을 세그먼트들과 “어느 곡의 세그먼트인지” 매핑
        batch_audios: List[np.ndarray] = []
        batch_song_indices: List[int] = []

        def _flush_batch():
            """현재 batch_audios를 임베딩해서 곡별 누적(sum/count)에 반영 후 배치 비움."""
            nonlocal batch_audios, batch_song_indices, sum_vecs, cnts
            if not batch_audios:
                return

            vecs = audio_embedder.embed_audios(
                batch_audios,
                sampling_rate=48_000,
                l2_normalize=False,
            )

            for v, si in zip(vecs, batch_song_indices):
                v = np.asarray(v)
                if si not in sum_vecs:
                    sum_vecs[si] = v.copy()
                    cnts[si] = 1
                else:
                    sum_vecs[si] += v
                    cnts[si] += 1

            # 메모리 피크 방지를 위해 즉시 비움
            batch_audios = []
            batch_song_indices = []

        # 오디오가 있는 곡들만 순회하며 “세그먼트 단위”로 배치 구성
        for song_idx, song in enumerate(songs):
            audio_path = song.get("metadata", {}).get("audio_path")
            if not audio_path:
                song["audio_values"] = None
                continue

            # 기본값 None으로 깔아두고, 성공하면 아래에서 채움
            song["audio_values"] = None

            try:
                segments = load_audio_mono_segments(
                    audio_path,
                    target_sr=48_000,
                    window_seconds=AUDIO_WINDOW_SECONDS,
                    segment_seconds=AUDIO_SEGMENT_SECONDS,
                    num_segments=AUDIO_NUM_SEGMENTS,
                )
            except Exception:
                # 디코딩 실패/파일 문제 등 → None 유지
                continue

            # 세그먼트들을 배치에 넣고, 찼으면 flush
            for seg in segments:
                batch_audios.append(seg)
                batch_song_indices.append(song_idx)
                if len(batch_audios) >= AUDIO_EMBED_BATCH_SIZE:
                    _flush_batch()

        # 마지막 남은 배치 처리
        _flush_batch()

        # 곡별 평균(mean) + 최종 L2 정규화
        for si, svec in sum_vecs.items():
            c = cnts.get(si, 0)
            if c <= 0:
                continue
            avg = svec / float(c)
            songs[si]["audio_values"] = l2_normalize_np(avg)


    # ---------------------------
    # 5) AUDIO mean-centering (Policy A + auto versioning)
    # ---------------------------
    # 기본값:
    # - AUDIO_MEAN_CENTER=1  (센터링 기본 ON)
    # - AUDIO_MEAN_REBUILD=0 (평균 재계산 강제 OFF)
    #
    # 데이터셋 변경 자동 감지:
    # - data/raw 아래 audio.m4a 목록(경로+size)로 dataset_signature 계산
    # - signature가 바뀌면 manifest version 자동 증가 + 새 mean 생성(audio_mean_v{n}.npy)
    if EMBED_AUDIO and _flag("AUDIO_MEAN_CENTER", "1"):
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
        rebuild = _flag("AUDIO_MEAN_REBUILD", "0")

        dataset_sig = compute_dataset_signature(raw_dir, audio_filename="audio.m4a")

        # 디코더·시드·구간이 바뀌면 곡 목록이 같아도 평균을 다시 만들어야 한다.
        preprocess_fp = compute_preprocess_fingerprint(
            model_tag=os.getenv("AUDIO_MODEL_TAG", "laion__clap-htsat-fused"),
            decoder=AUDIO_DECODER,
            clap_seed=CLAP_FEATURE_SEED,
            target_sr=AUDIO_TARGET_SAMPLE_RATE,
            window_seconds=AUDIO_WINDOW_SECONDS,
            segment_seconds=AUDIO_SEGMENT_SECONDS,
            num_segments=AUDIO_NUM_SEGMENTS,
        )

        # songs에 채워진 audio_values(곡 평균 + L2 normalize)로 μ를 만들고/재사용
        vecs = [s.get("audio_values") for s in songs]
        mean, manifest, status = get_or_create_mean_for_dataset(
            vecs,
            artifacts_dir=artifacts_dir,
            dataset_signature=dataset_sig,
            preprocess_fingerprint=preprocess_fp,
            rebuild=rebuild,
        )

        if mean is not None:
            for s in songs:
                v = s.get("audio_values")
                if v is None:
                    continue
                s["audio_values"] = apply_mean_center(v, mean)

        # 로그(디버깅/재현성)
        if status == "reused" and manifest is not None:
            print(f"[AUDIO_MEAN] reused: {manifest.mean_filename} (v{manifest.version})")
        elif status == "rebuilt" and manifest is not None:
            print(f"[AUDIO_MEAN] rebuilt: {manifest.mean_filename} (v{manifest.version})")
        else:
            print("[AUDIO_MEAN] skipped (no audio vectors)")

    return songs