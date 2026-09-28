# src/embedding/audio/mean_center.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List

import numpy as np


# -----------------------------
# Basic vector utils
# -----------------------------
def l2_normalize_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def compute_mean_vector(vectors: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    vecs = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
    if not vecs:
        return None
    mat = np.stack(vecs, axis=0)  # [N, D]
    return mat.mean(axis=0)


def apply_mean_center(v: np.ndarray, mean: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    return l2_normalize_np(v - mean)


def save_mean(mean: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.asarray(mean, dtype=np.float32))


def load_mean(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(str(path))


# -----------------------------
# Dataset signature + versioning
# -----------------------------
# 전처리 규약 버전. 디코더·시드·구간 계산이 바뀌면 손으로 올린다.
#   audio_v2_seeded: ffmpeg 고정 디코딩 + CLAP 특징 추출 시드 고정
PREPROCESS_VERSION = "audio_v2_seeded"


def compute_preprocess_fingerprint(
    *,
    model_tag: str,
    decoder: str,
    clap_seed: int,
    target_sr: int,
    window_seconds: int,
    segment_seconds: int,
    num_segments: int,
) -> str:
    """벡터 값을 좌우하는 설정을 한 문자열로 접는다.

    dataset_signature는 "어떤 곡이 있는지"만 본다. 그래서 시드나 디코더를 바꿔도
    곡 목록이 같으면 캐시가 그대로 재사용됐다. 벡터를 바꾸는 입력은 곡 목록만이
    아니므로 그것들을 따로 지문으로 만들어 manifest에 남긴다.

    model_tag까지 넣는 이유: 태그만 바꾸면 곡 벡터는 새 폴더에 새로 쌓이는데
    audio_mean.npy는 예전 모델로 만든 평균이 재사용될 수 있다.
    """
    payload = {
        "preprocess_version": PREPROCESS_VERSION,
        "model_tag": model_tag,
        "decoder": decoder,
        "clap_seed": int(clap_seed),
        "target_sr": int(target_sr),
        "window_seconds": int(window_seconds),
        "segment_seconds": int(segment_seconds),
        "num_segments": int(num_segments),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class AudioMeanManifest:
    dataset_signature: str
    preprocess_fingerprint: Optional[str] = None


def _manifest_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "audio_mean_manifest.json"


def _mean_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "audio_mean.npy"


def _read_manifest(path: Path) -> Optional[AudioMeanManifest]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fp = data.get("preprocess_fingerprint")
        return AudioMeanManifest(
            dataset_signature=str(data["dataset_signature"]),
            # 지문 이전에 쓰인 manifest에는 이 값이 없다. None은 "모른다"는 뜻이고,
            # 호출부에서 재구축 사유로 다룬다.
            preprocess_fingerprint=str(fp) if fp is not None else None,
        )
    except Exception:
        return None


def _write_manifest(path: Path, manifest: AudioMeanManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset_signature": manifest.dataset_signature}
    if manifest.preprocess_fingerprint is not None:
        payload["preprocess_fingerprint"] = manifest.preprocess_fingerprint
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_dataset_signature(
    raw_data_dir: Path,
    *,
    audio_filename: str = "audio.m4a",
) -> str:
    """
    data/raw 아래 오디오 파일 목록으로 데이터셋 "시그니처"를 만든다.
    - 목적: 데이터셋이 바뀌었는지 자동 감지(버전업)
    - 안정성: 경로+파일크기 기반(내용 해시보다 빠르고 대부분 충분)
    """
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        # data 폴더가 없는 환경(예: demo)에서도 동작하도록 고정값 반환
        return "no-data-dir"

    files = sorted(raw_data_dir.rglob(audio_filename))

    h = hashlib.sha256()
    h.update(str(raw_data_dir.resolve()).encode("utf-8"))
    h.update(b"\n")

    for p in files:
        try:
            rel = p.relative_to(raw_data_dir).as_posix()
        except Exception:
            rel = p.as_posix()

        try:
            size = p.stat().st_size
        except Exception:
            size = -1

        h.update(rel.encode("utf-8"))
        h.update(b"\t")
        h.update(str(size).encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()


def get_or_create_mean_for_dataset(
    vectors: List[Optional[np.ndarray]],
    *,
    artifacts_dir: Path,
    dataset_signature: str,
    preprocess_fingerprint: Optional[str] = None,
    rebuild: bool = False,
) -> tuple[Optional[np.ndarray], Optional[AudioMeanManifest], str]:
    """
    Policy B (single-file, overwrite):
    - manifest가 존재하고 signature·지문이 모두 같으며 mean 파일이 존재하면: 재사용
    - 아니면: mean을 새로 계산해서 audio_mean.npy에 덮어쓰기 + manifest 갱신

    지문(preprocess_fingerprint)이 왜 필요한지는 compute_preprocess_fingerprint 참고.
    manifest에 지문이 없으면(지문 도입 전에 만든 파일) 평균을 다시 만든다 —
    어떤 설정으로 만든 평균인지 알 수 없으므로 재사용하면 안 된다.

    returns:
      (mean_vector or None, manifest or None, status_str)
      status_str ∈ {"reused", "rebuilt", "skipped"}
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mpath = _manifest_path(artifacts_dir)
    manifest = _read_manifest(mpath)
    mean_path = _mean_path(artifacts_dir)

    # 1) 재사용 가능한지 판단
    reusable = (
        (not rebuild)
        and manifest is not None
        and manifest.dataset_signature == dataset_signature
        and mean_path.exists()
    )
    if reusable and preprocess_fingerprint is not None:
        reusable = manifest.preprocess_fingerprint == preprocess_fingerprint

    if reusable:
        mean = load_mean(mean_path)
        if mean is not None:
            return mean, manifest, "reused"

    # 2) 새로 생성(덮어쓰기)
    mean = compute_mean_vector([v for v in vectors if v is not None])
    if mean is None:
        return None, manifest, "skipped"

    save_mean(mean, mean_path)
    new_manifest = AudioMeanManifest(
        dataset_signature=dataset_signature,
        preprocess_fingerprint=preprocess_fingerprint,
    )
    _write_manifest(mpath, new_manifest)

    return mean, new_manifest, "rebuilt"