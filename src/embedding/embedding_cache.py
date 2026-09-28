"""
src/embedding/artifacts_cache.py

dense(.npy) / sparse(.npz) 임베딩 캐시를 저장/로드하는 공용 유틸.

요구사항:
- 생성된 npy/npz는 Vague-Finder/artifacts 아래에 저장
- 파일이 존재하면: 해당 트랙은 "해당 모달리티 + model_tag" 조합으로 임베딩 완료로 간주
- --force 없으면: 없던 것만 임베딩
- --force 있으면: 전부 재임베딩(덮어쓰기)
- model_tag를 경로에 포함해 모델/버전 변경 대비

캐시 경로 규칙:
- dense: artifacts/embeddings/<modality>/<model_tag>/<track_id>.npy
- sparse(BM25): artifacts/embeddings/text_sparse/<model_tag>/<track_id>.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EmbeddingCacheKey:
    """
    캐시 키

    modality:
      - "text_dense", "text_sparse", "image", "audio"
    model_tag:
      - 모델/버전/실험 설정을 나타내는 문자열 (경로 분리용)
      - 예: "ko-e5-2026_02_22", "siglip2-2026_02_22", "bm25-2026_02_22", "clap-2026_02_22"
    track_id:
      - 곡 고유 ID (meta.json의 id 또는 폴더명)
    """
    modality: str
    model_tag: str
    track_id: str


def _ext_for_modality(modality: str) -> str:
    """
    모달리티별 확장자 결정.
    - text_sparse: npz
    - 그 외: npy
    """
    return "npz" if modality == "text_sparse" else "npy"


def cache_path(artifacts_dir: Path, key: EmbeddingCacheKey) -> Path:
    """
    캐시 파일 경로를 반환.

    dense: artifacts/embeddings/<modality>/<model_tag>/<track_id>.npy
    sparse: artifacts/embeddings/text_sparse/<model_tag>/<track_id>.npz
    """
    ext = _ext_for_modality(key.modality)
    return Path(artifacts_dir) / "embeddings" / key.modality / key.model_tag / f"{key.track_id}.{ext}"


def exists(artifacts_dir: Path, key: EmbeddingCacheKey) -> bool:
    """
    파일이 존재하면 "이미 해당 조합으로 임베딩 완료"로 간주.
    """
    return cache_path(artifacts_dir, key).exists()


def plan_tracks(
    artifacts_dir: Path,
    *,
    modality: str,
    model_tag: str,
    track_ids: list[str],
    force: bool,
) -> Tuple[list[str], list[str]]:
    if force:
        return list(track_ids), []

    todo, skipped = [], []
    for tid in track_ids:
        if exists(artifacts_dir, EmbeddingCacheKey(modality, model_tag, tid)):
            skipped.append(tid)
        else:
            todo.append(tid)
    return todo, skipped


def purge_track_artifacts(
    artifacts_dir: Path,
    *,
    rejected_ids: set[str],
    accepted_ids: set[str] | None = None,
) -> list[Path]:
    """
    failed_raw로 이동된 곡의 과거 임베딩 파일을 모두 삭제한다.

    검사 대상:
    - text dense .npy
    - text sparse .npz
    - image .npy
    - audio .npy
    - 모든 model_tag 폴더
    """
    protected_ids = accepted_ids or set()

    purge_ids = {
        str(track_id)
        for track_id in rejected_ids
        if str(track_id) not in protected_ids
    }

    embeddings_dir = (
        Path(artifacts_dir)
        / "embeddings"
    )

    if (
        not purge_ids
        or not embeddings_dir.exists()
    ):
        return []

    removed: list[Path] = []

    for artifact_path in embeddings_dir.rglob("*"):
        if not artifact_path.is_file():
            continue

        if artifact_path.suffix.lower() not in {
            ".npy",
            ".npz",
        }:
            continue

        if artifact_path.stem not in purge_ids:
            continue

        artifact_path.unlink()
        removed.append(artifact_path)

    return removed


# -----------------------------
# Dense (.npy)
# -----------------------------
def save_npy(artifacts_dir: Path, key: EmbeddingCacheKey, vec: np.ndarray) -> Path:
    """
    dense 벡터를 float32 1D로 저장.
    """
    if key.modality == "text_sparse":
        raise ValueError("save_npy() called for text_sparse; use save_npz_sparse() instead.")

    p = cache_path(artifacts_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    np.save(str(p), vec)
    return p


def load_npy(artifacts_dir: Path, key: EmbeddingCacheKey) -> Optional[np.ndarray]:
    """
    dense 벡터 로드.
    """
    if key.modality == "text_sparse":
        raise ValueError("load_npy() called for text_sparse; use load_npz_sparse() instead.")

    p = cache_path(artifacts_dir, key)
    if not p.exists():
        return None
    try:
        v = np.load(str(p))
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        return v
    except Exception:
        return None


# -----------------------------
# Sparse (.npz) - BM25
# -----------------------------
def save_npz_sparse(
    artifacts_dir: Path,
    key: EmbeddingCacheKey,
    *,
    indices: np.ndarray,
    values: np.ndarray,
) -> Path:
    """
    BM25 sparse를 npz로 저장.

    저장 포맷:
      - indices: int32 1D
      - values : float32 1D
    """
    if key.modality != "text_sparse":
        raise ValueError("save_npz_sparse() is only for modality='text_sparse'.")

    p = cache_path(artifacts_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)

    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    val = np.asarray(values, dtype=np.float32).reshape(-1)

    if idx.shape[0] != val.shape[0]:
        raise ValueError(f"sparse length mismatch: indices={idx.shape[0]} values={val.shape[0]}")

    np.savez_compressed(str(p), indices=idx, values=val)
    return p


def load_npz_sparse(artifacts_dir: Path, key: EmbeddingCacheKey) -> Optional[dict]:
    """
    BM25 sparse npz 로드 후 Pinecone sparse_values 포맷으로 반환:
      {"indices": [...], "values": [...]}
    """
    if key.modality != "text_sparse":
        raise ValueError("load_npz_sparse() is only for modality='text_sparse'.")

    p = cache_path(artifacts_dir, key)
    if not p.exists():
        return None

    try:
        z = np.load(str(p))
        idx = np.asarray(z["indices"], dtype=np.int64).reshape(-1)
        val = np.asarray(z["values"], dtype=np.float32).reshape(-1)
        if idx.shape[0] != val.shape[0]:
            return None
        return {"indices": [int(i) for i in idx.tolist()], "values": [float(v) for v in val.tolist()]}
    except Exception:
        return None