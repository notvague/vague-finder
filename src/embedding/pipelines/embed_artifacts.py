from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import json
import shutil
import numpy as np

from src.embedding.embedding_cache import (
    EmbeddingCacheKey,
    plan_tracks,
    save_npy,
    save_npz_sparse,
)

from src.embedding.text.passage_builder import build_dense_passage, build_sparse_passage
from src.embedding.models.text_koe5 import KoE5Embedder
from src.embedding.models.text_bm25 import BM25SparseEncoder
from src.embedding.models.image_siglip2 import SigLIP2Embedder
from src.embedding.models.audio_clap import CLAPAudioEmbedder

from src.embedding.image.image_io import fetch_image
from src.embedding.image.normalize import l2_normalize_np
from src.embedding.audio.audio_io import (
    AUDIO_DECODER,
    AUDIO_TARGET_SAMPLE_RATE,
    load_audio_mono_segments,
)
from src.embedding.models.audio_clap import CLAP_FEATURE_SEED
from src.embedding.audio.mean_center import (
    apply_mean_center,
    compute_dataset_signature,
    compute_preprocess_fingerprint,
)

# 배치 사이즈
TEXT_DENSE_BATCH = int(os.getenv("TEXT_DENSE_BATCH", "64"))
TEXT_SPARSE_BATCH = int(os.getenv("TEXT_SPARSE_BATCH", "256"))
IMAGE_BATCH = int(os.getenv("IMAGE_BATCH", "32"))
AUDIO_BATCH = int(os.getenv("AUDIO_BATCH", "8"))


def _purge_other_model_tag_dirs(artifacts_dir: Path, *, modality: str, keep_model_tag: str) -> None:
    """
    artifacts/embeddings/<modality>/ 아래에서 keep_model_tag 디렉토리만 남기고 나머지 삭제.
    """
    base = Path(artifacts_dir) / "embeddings" / modality
    if not base.exists():
        return

    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name == keep_model_tag:
            continue
        shutil.rmtree(child, ignore_errors=True)
        print(f"[CLEAN] removed old dir: {child}")
        # 캐시를 지웠으면 그 태그의 manifest도 지운다. 남겨 두면 "기록은 있는데
        # 벡터는 없는" 상태가 되어 읽는 사람을 헷갈리게 한다.
        #
        # audio_cache_manifest__* 는 오디오 전용이다. 이 함수는 텍스트·이미지도 쓰므로
        # modality를 확인하지 않으면, 텍스트와 오디오가 같은 태그를 쓸 때 텍스트 태그를
        # 바꾸는 것만으로 정상인 오디오 캐시가 무효화된다.
        if modality != "audio":
            continue
        orphan = Path(artifacts_dir) / f"audio_cache_manifest__{child.name}.json"
        if orphan.exists():
            orphan.unlink()
            print(f"[CLEAN] removed old manifest: {orphan.name}")


def _purge_cache_dir(artifacts_dir: Path, *, modality: str, model_tag: str) -> None:
    """
    artifacts/embeddings/<modality>/<model_tag>/ 를 통째로 삭제.
    --force 시 '남아있는 찌꺼기 파일'까지 제거하기 위해 필요.
    """
    cache_dir = Path(artifacts_dir) / "embeddings" / modality / model_tag
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[CACHE] purged: {cache_dir}")


def _chunks(n: int, b: int):
    for s in range(0, n, b):
        yield s, min(n, s + b)


# -------- TEXT DENSE (Ko-E5) --------
def embed_text_dense(
        songs: List[Dict], 
        *, 
        artifacts_dir: Path, 
        model_tag: str, 
        force: bool
) -> None:
    modality = "text_dense"
    track_ids = [str(s["id"]) for s in songs]

    if force:
        _purge_cache_dir(artifacts_dir, modality=modality, model_tag=model_tag)

    todo_ids, skipped_ids = plan_tracks(artifacts_dir, modality=modality, model_tag=model_tag, track_ids=track_ids, force=force)
    print(f"[TEXT_DENSE] tag={model_tag} total={len(track_ids)} todo={len(todo_ids)} skipped={len(skipped_ids)}")
    if not todo_ids:
        _purge_other_model_tag_dirs(artifacts_dir, modality=modality, keep_model_tag=model_tag)
        return

    id2song = {str(s["id"]): s for s in songs}
    todo_songs = [id2song[tid] for tid in todo_ids]
    passages = [build_dense_passage(s) for s in todo_songs]

    embedder = KoE5Embedder()

    for s, e in _chunks(len(passages), TEXT_DENSE_BATCH):
        vecs = embedder.embed_passages(
            passages[s:e], 
            add_e5_prefix=True, 
            normalize=True, 
            batch_size=TEXT_DENSE_BATCH, 
            show_progress_bar=True
        )
        for i, v in enumerate(vecs):
            tid = str(todo_songs[s + i]["id"])
            save_npy(artifacts_dir, EmbeddingCacheKey(modality, model_tag, tid), np.asarray(v, dtype=np.float32))

    _purge_other_model_tag_dirs(artifacts_dir, modality=modality, keep_model_tag=model_tag)

# -------- TEXT SPARSE (BM25) --------
def embed_text_sparse_bm25(
    songs: List[Dict],
    *,
    artifacts_dir: Path,
    model_tag: str,
) -> None:
    """
    - --force 여부와 무관하게 "항상 재생성"한다.
      즉, 매 실행마다:
        1) 기존 npz 캐시(artifacts/embeddings/text_sparse/<model_tag>/)를 전부 삭제
        2) 현재 songs 전체 corpus로 BM25를 새로 fit
        3) 전 곡 sparse 벡터를 새로 계산
        4) 각 곡별로 .npz를 저장(축적/관리)
        5) songs[*]["text_sparse_values"]에 주입

    저장 위치:
      - 문서별 sparse 캐시:
          artifacts/embeddings/text_sparse/<model_tag>/<track_id>.npz
      - BM25 파라미터(json):
          artifacts/bm25_params.json (매 실행 덮어쓰기)
    """
    modality = "text_sparse"
    track_ids = [str(s["id"]) for s in songs]

    # 0) 기존 npz 캐시 삭제(강제 재구축)
    _purge_cache_dir(artifacts_dir, modality=modality, model_tag=model_tag)

    # 1) corpus 생성
    corpus = [build_sparse_passage(s) for s in songs]

    # 2) BM25 fit + params 저장(덮어쓰기)
    encoder = BM25SparseEncoder(params_path=artifacts_dir / "bm25_params.json")
    encoder.fit(corpus)
    try:
        encoder.dump()
    except Exception as e:
        print(f"[WARN] BM25 params dump failed: {e}")

    # 3) 전 곡 sparse 생성
    sparse_vecs = encoder.encode_documents(corpus)

    # 4) 전 곡 npz 저장 + songs에 주입
    for s, sp in zip(songs, sparse_vecs):
        tid = str(s["id"])
        indices = np.asarray(sp.get("indices", []), dtype=np.int64)
        values = np.asarray(sp.get("values", []), dtype=np.float32)

        save_npz_sparse(
            artifacts_dir,
            EmbeddingCacheKey(modality, model_tag, tid),
            indices=indices,
            values=values,
        )

        s["text_sparse_values"] = {
            "indices": [int(i) for i in indices.tolist()],
            "values": [float(v) for v in values.tolist()],
        }


# -------- IMAGE (SigLIP2) --------
def embed_image(
    songs: List[Dict], 
    *, 
    artifacts_dir: Path, 
    model_tag: str, 
    force: bool
) -> None:
    modality = "image"
    candidates = [s for s in songs if s.get("metadata", {}).get("album_cover")]
    track_ids = [str(s["id"]) for s in candidates]

    if force:
        _purge_cache_dir(artifacts_dir, modality=modality, model_tag=model_tag)

    todo_ids, skipped_ids = plan_tracks(artifacts_dir, modality=modality, model_tag=model_tag, track_ids=track_ids, force=force)
    print(f"[IMAGE] tag={model_tag} total={len(track_ids)} todo={len(todo_ids)} skipped={len(skipped_ids)}")
    if not todo_ids:
        _purge_other_model_tag_dirs(artifacts_dir, modality=modality, keep_model_tag=model_tag)
        return

    id2song = {str(s["id"]): s for s in candidates}
    todo_songs = [id2song[tid] for tid in todo_ids]

    embedder = SigLIP2Embedder()
    batch_imgs, batch_ids = [], []

    def flush():
        nonlocal batch_imgs, batch_ids
        if not batch_imgs:
            return
        vecs = embedder.embed_images(batch_imgs, l2_normalize=True)
        for tid, v in zip(batch_ids, vecs):
            save_npy(artifacts_dir, EmbeddingCacheKey(modality, model_tag, tid), np.asarray(v, dtype=np.float32))
        batch_imgs, batch_ids = [], []

    for s in todo_songs:
        tid = str(s["id"])
        src = s.get("metadata", {}).get("album_cover")
        try:
            img = fetch_image(src)
        except Exception:
            continue
        batch_imgs.append(img)
        batch_ids.append(tid)
        if len(batch_imgs) >= IMAGE_BATCH:
            flush()
    flush()

    _purge_other_model_tag_dirs(artifacts_dir, modality=modality, keep_model_tag=model_tag)


# -------- AUDIO (CLAP) --------
def embed_audio(
    songs: List[Dict],
    *,
    data_raw_dir: Path,
    artifacts_dir: Path,
    model_tag: str,
    force: bool,
    window_seconds: int,
    segment_seconds: int,
    num_segments: int,
    mean_center: bool,
) -> Optional[str]:
    modality = "audio"
    candidates = [s for s in songs if s.get("metadata", {}).get("audio_path")]
    track_ids = [str(s["id"]) for s in candidates]

    effective_tag = f"{model_tag}__mean" if mean_center else model_tag

    # 두 산출물은 수명이 다르므로 manifest를 나눈다.
    #   - 곡별 벡터 캐시: 전처리 지문에만 좌우된다 (센터링 여부와 무관)
    #   - 평균 벡터: 지문 + 곡 구성(dataset_signature)에 좌우된다
    # 한 파일에 섞어 두면 센터링을 껐을 때 "평균 설정"과 "캐시 설정"이 구별되지 않아,
    # 새 지문을 남길 자리가 없어진다(그러면 매 실행이 전곡 재임베딩으로 떨어진다).
    mean_path = Path(artifacts_dir) / "audio_mean.npy"
    mean_manifest_path = Path(artifacts_dir) / "audio_mean_manifest.json"
    cache_manifest_path = Path(artifacts_dir) / f"audio_cache_manifest__{effective_tag}.json"
    cache_dir = Path(artifacts_dir) / "embeddings" / modality / effective_tag

    # dataset_signature로 "곡 구성 변화" 감지
    dataset_sig = compute_dataset_signature(Path(data_raw_dir), audio_filename="audio.m4a") if mean_center else None

    # 곡 구성 말고도 벡터를 바꾸는 입력들(디코더·시드·구간·모델 태그)을 지문으로 본다.
    # dataset_signature만 보던 때는 시드를 바꿔도 todo=0으로 예전 벡터를 재사용했다.
    preprocess_fp = compute_preprocess_fingerprint(
        model_tag=model_tag,
        decoder=AUDIO_DECODER,
        clap_seed=CLAP_FEATURE_SEED,
        target_sr=AUDIO_TARGET_SAMPLE_RATE,
        window_seconds=window_seconds,
        segment_seconds=segment_seconds,
        num_segments=num_segments,
    )

    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    cache_fp = _read_json(cache_manifest_path).get("preprocess_fingerprint")
    mean_meta = _read_json(mean_manifest_path)
    prev_sig, mean_fp = mean_meta.get("dataset_signature"), mean_meta.get("preprocess_fingerprint")

    # 남아 있는 산출물이 어떤 설정으로 만들어졌는지 모르면 재사용하지 않는다.
    # manifest가 없는 캐시·평균도 여기에 걸린다(출처 불명 = 무효).
    cache_has_files = cache_dir.exists() and any(cache_dir.glob("*.npy"))

    rebuild_all = False
    if cache_has_files and cache_fp != preprocess_fp:
        rebuild_all = True
        reason = "preprocess_changed" if cache_fp else "cache_fingerprint_unknown"
        print(f"[AUDIO] rebuild_all=True ({reason}) cache_fp={cache_fp} new_fp={preprocess_fp}")

    prev_members = mean_meta.get("mean_track_ids")

    if mean_center and not rebuild_all:
        if not mean_path.exists():
            rebuild_all = True
            print(f"[AUDIO] rebuild_all=True (mean_missing) new_sig={dataset_sig}")
        elif mean_fp != preprocess_fp:
            rebuild_all = True
            reason = "mean_preprocess_changed" if mean_fp else "mean_fingerprint_unknown"
            print(f"[AUDIO] rebuild_all=True ({reason}) mean_fp={mean_fp} new_fp={preprocess_fp}")
        elif not isinstance(prev_members, list):
            # 구성원 기록이 없으면 평균이 어느 곡들로 만들어졌는지 알 수 없다. 그러면
            # 실패했던 곡이 복구돼도 평균이 낡았는지 판단할 수 없으므로 지금 다시 만든다.
            rebuild_all = True
            print(f"[AUDIO] rebuild_all=True (mean_members_unknown) new_fp={preprocess_fp}")
        elif prev_sig != dataset_sig:
            rebuild_all = True
            print(f"[AUDIO] rebuild_all=True (signature_changed) prev_sig={prev_sig} new_sig={dataset_sig}")

    if force:
        rebuild_all = True

    if rebuild_all:
        # 캐시를 지금 지우지 않는다. 임베딩이 중간에 실패하면 기존 산출물만 사라지고
        # 새 것은 없는 상태가 된다(예전에 905곡을 날려 백업에서 복구한 그 구조).
        # 새 벡터를 임시 위치에 다 만든 뒤 마지막에 교체한다.
        todo_ids = track_ids[:]
        skipped_ids = []
    else:
        todo_ids, skipped_ids = plan_tracks(
            artifacts_dir,
            modality=modality,
            model_tag=effective_tag,
            track_ids=track_ids,
            force=force,
        )

    # 새로 임베딩할 것이 없어도 평균 구성원이 어긋나 있을 수 있다(예: 이전 실행이
    # 산출물 반영 중간에 죽어 벡터만 새 것으로 남은 경우). 그대로 반환하면 아무도
    # 그 불일치를 고치지 못하므로, 조기 반환 전에 한 번 본다.
    if mean_center and not rebuild_all and not todo_ids and isinstance(prev_members, list):
        if sorted(prev_members) != sorted(skipped_ids):
            print(
                f"[AUDIO] rebuild_all=True (mean_members_changed) "
                f"prev={len(prev_members)} now={len(skipped_ids)}"
            )
            rebuild_all = True
            todo_ids, skipped_ids = track_ids[:], []

    print(f"[AUDIO] tag={effective_tag} total={len(track_ids)} todo={len(todo_ids)} skipped={len(skipped_ids)}")

    if not todo_ids:
        _purge_other_model_tag_dirs(artifacts_dir, modality=modality, keep_model_tag=effective_tag)
        return effective_tag

    id2song = {str(s["id"]): s for s in candidates}

    embedder = CLAPAudioEmbedder()

    def embed_ids(ids: List[str]) -> Dict[str, np.ndarray]:
        """주어진 곡들을 세그먼트 단위로 임베딩해 곡별 평균(L2)까지 낸다.

        디코딩에 실패한 곡은 결과에서 빠진다 — 반환된 키가 "실제로 성공한 곡"이다.
        """
        sum_vecs: Dict[str, np.ndarray] = {}
        cnts: Dict[str, int] = {}
        batch_audios, batch_ids = [], []

        def flush():
            nonlocal batch_audios, batch_ids
            if not batch_audios:
                return
            seg_vecs = embedder.embed_audios(batch_audios, sampling_rate=48_000, l2_normalize=False)
            for v, tid in zip(seg_vecs, batch_ids):
                v = np.asarray(v, dtype=np.float32)
                if tid not in sum_vecs:
                    sum_vecs[tid] = v.copy()
                    cnts[tid] = 1
                else:
                    sum_vecs[tid] += v
                    cnts[tid] += 1
            batch_audios, batch_ids = [], []

        for tid in ids:
            s = id2song[tid]
            ap = s.get("metadata", {}).get("audio_path")

            if not ap:
                print(f"[AUDIO][SKIP] tid={tid} reason=no_audio_path")
                continue

            try:
                segments = load_audio_mono_segments(
                    ap,
                    target_sr=48_000,
                    window_seconds=window_seconds,
                    segment_seconds=segment_seconds,
                    num_segments=num_segments,
                )
            except Exception as e:
                print(f"[AUDIO][SKIP] tid={tid} path={ap} error={type(e).__name__}: {e}")
                continue

            for seg in segments:
                batch_audios.append(seg)
                batch_ids.append(tid)
                if len(batch_audios) >= AUDIO_BATCH:
                    flush()
        flush()

        out: Dict[str, np.ndarray] = {}
        for tid, svec in sum_vecs.items():
            c = cnts.get(tid, 0)
            if c <= 0:
                continue
            out[tid] = l2_normalize_np(svec / float(c))
        return out

    song_vecs = embed_ids(todo_ids)

    # 평균에 실제로 들어간 곡이 달라졌으면 평균과 센터링을 다시 해야 한다.
    #
    # 일부 곡이 디코딩에 실패하면 성공한 곡만으로 평균이 만들어진다. 그 곡이 나중에
    # 복구되면 평균은 여전히 예전 구성원 기준이라, 처음부터 전곡이 성공한 실행과
    # 최종 벡터가 달라진다. 캐시에 남은 벡터는 이미 센터링된 것이고
    # apply_mean_center = l2_normalize(v - mean) 은 되돌릴 수 없으므로,
    # 고치려면 전곡을 다시 임베딩해야 한다. 그래서 이 자리에서 한 번 더 돈다.
    #
    # 영구 실패 곡이 있어도 반복되지 않는다 — 그 곡은 계속 성공 집합에 안 들어오므로
    # 구성원이 기록과 같아지고, 다음 실행부터는 이 분기를 타지 않는다.
    if mean_center and not rebuild_all:
        now_members = sorted(set(song_vecs) | set(skipped_ids))
        if isinstance(prev_members, list) and sorted(prev_members) != now_members:
            print(
                f"[AUDIO] rebuild_all=True (mean_members_changed) "
                f"prev={len(prev_members)} now={len(now_members)}"
            )
            # 여기서도 기존 캐시를 지우지 않는다. 아래 교체 단계까지 그대로 남겨 둔다.
            rebuild_all = True
            todo_ids, skipped_ids = track_ids[:], []
            song_vecs = embed_ids(todo_ids)

    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    new_mean = None
    if mean_center and song_vecs:
        if rebuild_all:
            # 전곡(현재 데이터셋) 기반 mean. 파일로 쓰는 것은 교체 단계에서 한다.
            mat = np.stack(list(song_vecs.values()), axis=0).astype(np.float32)
            new_mean = mat.mean(axis=0).astype(np.float32)
            mean = new_mean
        else:
            # 기존 mean 로드
            mean = np.load(mean_path).astype(np.float32)

        # mean 적용
        for tid in list(song_vecs.keys()):
            song_vecs[tid] = apply_mean_center(song_vecs[tid], mean)

    if not song_vecs:
        # 한 곡도 만들지 못했다. 기존 산출물은 그대로 두고 실패로 돌려준다.
        # effective_tag를 돌려주면 호출부가 "이 태그로 캐시를 읽으면 된다"고 믿는다.
        print("[AUDIO] 성공한 곡이 없다 -> 기존 산출물 보존, 태그 없이 반환")
        return None

    # ── 산출물 반영 ────────────────────────────────────────────────────────────
    # 벡터·평균·manifest 두 개를 **임시 위치에 모두 완성한 뒤** 한꺼번에 제자리로 옮긴다.
    # 하나라도 실패하면 옮긴 것을 되돌려 이전 상태로 복구한다.
    #
    # 순서대로 반영하면 중간 실패에서 섞인 상태가 남는다. 예를 들어 벡터만 새 것으로
    # 바뀌고 평균·manifest가 예전 것이면, 다음 실행은 지문이 같아 todo=0으로 끝나고
    # 아무도 그 불일치를 고치지 못한다.
    staging_root = Path(artifacts_dir) / f".staging_{modality}_{os.getpid()}"
    shutil.rmtree(staging_root, ignore_errors=True)

    def _commit(pairs: List[tuple[Path, Path]]) -> None:
        """(임시, 제자리) 쌍을 전부 반영하거나 전부 되돌린다."""
        moved: List[tuple[Path, Optional[Path]]] = []   # (제자리, 밀어둔 이전 것)
        try:
            for staged, live in pairs:
                retired = live.with_name(f"{live.name}.retired_{os.getpid()}")
                if live.exists():
                    os.rename(live, retired)
                else:
                    retired = None
                # 밀어두는 데 성공한 즉시 복구 목록에 넣는다. 새 것을 제자리에 넣은
                # 뒤에 등록하면, 그 사이에 실패했을 때 이 항목이 목록에 없어서
                # 이전 것이 .retired_* 에 갇히고 제자리는 빈 상태로 남는다.
                moved.append((live, retired))
                live.parent.mkdir(parents=True, exist_ok=True)
                os.rename(staged, live)
        except Exception:
            for live, retired in reversed(moved):
                if live.exists():
                    shutil.rmtree(live, ignore_errors=True) if live.is_dir() else live.unlink(missing_ok=True)
                if retired is not None and retired.exists():
                    os.rename(retired, live)
            raise
        for _, retired in moved:
            if retired is None:
                continue
            shutil.rmtree(retired, ignore_errors=True) if retired.is_dir() else retired.unlink(missing_ok=True)

    try:
        pairs: List[tuple[Path, Path]] = []

        # 벡터. 전곡 재구축이면 디렉터리를 통째로 교체하고, 증분이면 제자리에 더한다
        # (증분은 지우는 것이 없으므로 되돌릴 이전 상태도 없다).
        if rebuild_all:
            for tid, v in song_vecs.items():
                save_npy(staging_root, EmbeddingCacheKey(modality, effective_tag, tid), v)
            pairs.append((staging_root / "embeddings" / modality / effective_tag, cache_dir))
        else:
            for tid, v in song_vecs.items():
                save_npy(artifacts_dir, EmbeddingCacheKey(modality, effective_tag, tid), v)

        # 평균과 그 manifest는 항상 같이 움직인다.
        if new_mean is not None:
            staging_root.mkdir(parents=True, exist_ok=True)
            # 이름이 .npy로 끝나야 한다 — np.save는 아니면 제멋대로 .npy를 덧붙인다.
            staged_mean = staging_root / mean_path.name
            np.save(str(staged_mean), new_mean)
            pairs.append((staged_mean, mean_path))

            # mean_track_ids는 평균에 **실제로 들어간** 곡이다. num_tracks(후보 수)와
            # 다를 수 있다 — 디코딩에 실패한 곡은 평균에 못 들어간다. 이 목록이 있어야
            # 나중에 그 곡이 복구됐을 때 평균이 낡았다는 것을 알아낼 수 있다.
            staged_mean_manifest = staging_root / mean_manifest_path.name
            _write_json(staged_mean_manifest, {
                "dataset_signature": dataset_sig,
                "preprocess_fingerprint": preprocess_fp,
                "model_tag": model_tag,
                "num_tracks": len(track_ids),
                "mean_num_tracks": len(song_vecs),
                "mean_track_ids": sorted(song_vecs),
            })
            pairs.append((staged_mean_manifest, mean_manifest_path))

        # 캐시 manifest. 센터링을 껐을 때도 남겨야 다음 실행이 "같은 설정"임을 알아본다.
        staging_root.mkdir(parents=True, exist_ok=True)
        staged_cache_manifest = staging_root / cache_manifest_path.name
        _write_json(staged_cache_manifest, {
            "preprocess_fingerprint": preprocess_fp,
            "model_tag": model_tag,
            "effective_tag": effective_tag,
            "mean_centered": bool(mean_center),
            "num_tracks": len(track_ids),
        })
        pairs.append((staged_cache_manifest, cache_manifest_path))

        _commit(pairs)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    if rebuild_all:
        print(f"[CACHE] swapped in {len(song_vecs)} vectors -> {cache_dir.name}")
    if new_mean is not None:
        print(f"[AUDIO_MEAN] recompute+overwrite -> {mean_path.name}")

    # 예전 태그 정리는 새 산출물이 제자리에 들어간 뒤에만 한다. 먼저 지우면
    # 새 태그가 전부 실패했을 때 쓸 수 있는 캐시까지 없어진다.
    _purge_other_model_tag_dirs(artifacts_dir, modality=modality, keep_model_tag=effective_tag)

    return effective_tag