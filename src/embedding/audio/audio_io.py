"""
audio/audio_io.py

- data/raw/*/audio.m4a 를 로컬에서 읽어 mono waveform으로 반환
- CLAP 입력 권장 스펙(48kHz, mono, float32)에 맞게 다운믹스·리샘플
- "긴 오디오를 그대로 다 넣으면" 메모리/시간/배치패딩 비용이 커지므로
  기본으로 max_seconds (120초)로 잘라서 사용
- 디코딩은 ffmpeg가 한다(AUDIO_DECODER 참고). ffmpeg는 별도 설치가 필요하다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Union

import numpy as np
import torch

import torchaudio


AUDIO_TARGET_SAMPLE_RATE = int(os.getenv("AUDIO_TARGET_SAMPLE_RATE", "48000"))
AUDIO_MAX_SECONDS = int(os.getenv("AUDIO_MAX_SECONDS", "120"))

# 디코더. 기본은 ffmpeg — 기존 코퍼스가 이 경로로 만들어졌고, 바꿔 끼우면 파형이 달라진다.
# 어느 쪽을 고르든 조용히 다른 쪽으로 넘어가지 않는다. 실패는 예외로 드러낸다.
AUDIO_DECODER = os.getenv("AUDIO_DECODER", "ffmpeg").strip().lower()


def _ffmpeg_binary() -> str | None:
    """ffmpeg 실행 파일. 없으면 None."""
    return os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")


def _load_with_ffmpeg(path: Path, *, target_sr: int, max_seconds: int) -> np.ndarray:
    """ffmpeg로 직접 디코딩한다. 이것이 기본 경로다.

    torchaudio 2.10은 디코딩을 torchcodec에 넘기는데, torchcodec은 FFmpeg 4~8용
    바이너리만 들고 온다(`libtorchcodec_core4~8.dylib`). FFmpeg 9가 깔린 환경에서는
    요구하는 `libavutil.56~60`이 하나도 없어 다섯 개 전부 dlopen에 실패하고,
    `torchaudio.load`가 곡마다 예외를 던진다. 곡 수만큼 200줄짜리 트레이스백이 쌓이고
    audio_values=None이 되어 오디오 임베딩이 통째로 비는데 파이프라인은 정상 종료한다.
    캐시를 먼저 비우는 구조라 기존 벡터까지 사라진다.

    그래서 torchaudio를 먼저 시도하지 않고 ffmpeg를 바로 부른다. 디코더를 바꿔 끼우면
    파형이 달라져 기존 코퍼스와 어긋나므로(실측: CLAP 코사인 0.989) 경로를 하나로 고정한다.

    ffmpeg가 다운믹스와 리샘플까지 해 주므로 결과는 이 함수의 계약과 같다 —
    float32 mono, target_sr, 앞부분 max_seconds.
    """
    binary = _ffmpeg_binary()
    if binary is None:
        raise RuntimeError(
            "오디오 디코딩에 ffmpeg가 필요한데 찾을 수 없습니다. "
            "ffmpeg를 설치하거나 FFMPEG_BINARY로 경로를 지정하세요."
        )
    command = [binary, "-v", "error", "-i", str(path)]
    if max_seconds and max_seconds > 0:
        command += ["-t", str(max_seconds)]
    command += ["-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-"]
    done = subprocess.run(command, capture_output=True)
    if done.returncode != 0 or not done.stdout:
        raise RuntimeError(
            f"ffmpeg 디코딩 실패 ({path.name}): {done.stderr.decode('utf-8', 'replace')[:200]}"
        )
    return np.frombuffer(done.stdout, dtype=np.float32).copy()


def _load_with_torchaudio(path: Path, *, target_sr: int, max_seconds: int) -> np.ndarray:
    """torchaudio로 디코딩한다. AUDIO_DECODER=torchaudio 일 때만 쓴다.

    기본 경로가 아니다. 남겨 둔 이유는 두 가지다 —
    torchcodec이 정상인 환경에서 비교 측정을 할 수 있어야 하고,
    다운믹스 방식이 ffmpeg와 다르기 때문이다(여기는 mean = (L+R)/2,
    ffmpeg `-ac 1`은 (L+R)/√2로 3dB 더 크다). 코퍼스를 섞어 만들면 안 된다.

    **실패하면 ffmpeg로 대체하지 않고 예외를 올린다.** 대체해 버리면 비교 실험에서
    양쪽 모두 ffmpeg로 돌아가면서 "차이가 없다"는 결론이 조용히 나온다. 명시적으로
    고른 디코더가 안 되면 그 사실이 드러나야 한다.

    처리 단계:
    1) torchaudio.load -> waveform (C, T), sample_rate
    2) stereo/multi-channel이면 평균내서 mono로 변환
    3) sample_rate != target_sr 이면 torchaudio.functional.resample로 리샘플
    4) max_seconds 적용 (앞부분 잘라내기)
    5) float32 numpy 반환
    """
    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception as exc:  # 디코더가 없거나 FFmpeg 버전이 안 맞는 경우
        raise RuntimeError(
            f"AUDIO_DECODER=torchaudio로 골랐지만 torchaudio가 {path.name}을 디코딩하지 "
            "못했습니다. ffmpeg로 대체하지 않습니다 — 비교 실험이 조용히 무의미해지기 "
            "때문입니다. torchcodec을 설치된 FFmpeg 세대에 맞춰 깔거나, "
            "AUDIO_DECODER를 지우고 기본 ffmpeg 경로로 돌아가세요."
        ) from exc

    # 1) mono 변환: 채널이 여러 개면 평균
    if waveform.dim() != 2:
        raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)} (expected [C, T])")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

    # 2) 리샘플
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    # 3) trim (앞부분 max_seconds 초만)
    if max_seconds and max_seconds > 0:
        max_len = int(sr * max_seconds)
        if waveform.size(1) > max_len:
            waveform = waveform[:, :max_len]

    # 4) numpy 변환 (mono이므로 [1, T] -> [T])
    wav_1d = waveform.squeeze(0)

    # 5) dtype float32로 정규화
    wav_1d = wav_1d.to(torch.float32)

    return wav_1d.detach().cpu().numpy()


def load_audio_mono(
    path: Union[str, Path],
    *,
    target_sr: int = AUDIO_TARGET_SAMPLE_RATE,
    max_seconds: int = AUDIO_MAX_SECONDS,
) -> np.ndarray:
    """
    로컬 오디오 파일을 mono waveform으로 로드한다.

    파라미터:
    - path: audio.m4a (또는 ffmpeg가 읽을 수 있는 포맷)
    - target_sr: 목표 샘플레이트
    - max_seconds: 앞부분 max_seconds 초만 사용

    반환:
    - np.ndarray (shape: [T], dtype float32)

    디코딩은 ffmpeg가 한다. 예전에는 torchaudio.load를 먼저 시도하고 실패하면
    ffmpeg로 넘겼는데, torchcodec이 FFmpeg 9와 맞지 않는 환경에서는 그 시도가
    **곡마다** 실패하면서 로그만 쌓였다(자세한 사정은 _load_with_ffmpeg 주석).
    기존 코퍼스도 ffmpeg 경로로 만들어졌으므로 경로를 하나로 고정한다.

    비교 측정이 필요하면 AUDIO_DECODER=torchaudio로 바꿔 끼운다. 그때는 실패해도
    ffmpeg로 대체하지 않고 예외가 난다 — 그래야 두 경로를 실제로 비교한 것이 된다.
    """
    path = Path(path)
    if AUDIO_DECODER == "torchaudio":
        return _load_with_torchaudio(path, target_sr=target_sr, max_seconds=max_seconds)
    return _load_with_ffmpeg(path, target_sr=target_sr, max_seconds=max_seconds)

def load_audio_mono_segments(
    path: Union[str, Path],
    *,
    target_sr: int = AUDIO_TARGET_SAMPLE_RATE,
    window_seconds: int = 120,
    segment_seconds: int = 12,
    num_segments: int = 8,
) -> list[np.ndarray]:
    """
    오디오 파일에서 여러 개의 짧은 세그먼트를 샘플링해 반환한다.
    - 목적: 한 구간(특히 보컬 들어간 구간)에 임베딩이 과도하게 휘둘리는 것을 줄이고,
            곡 전체의 '사운드/편곡/분위기' 쪽을 더 반영하기 위한 평균 풀링(mean pooling) 입력으로 사용.
    """
    path = Path(path)

    # 1) 앞부분 window_seconds만 로드
    wav = load_audio_mono(path, target_sr=target_sr, max_seconds=window_seconds)  # [T]
    seg_len = int(target_sr * segment_seconds)

    if seg_len <= 0:
        raise ValueError(f"segment_seconds must be > 0 (got {segment_seconds})")

    # 오디오가 너무 짧은 경우: 1개만 패딩해서 반환
    if wav.shape[0] <= seg_len:
        padded = np.zeros((seg_len,), dtype=np.float32)
        padded[: wav.shape[0]] = wav.astype(np.float32, copy=False)
        return [padded]

    max_start = wav.shape[0] - seg_len

    # max_start가 0이면 사실상 한 위치뿐
    if max_start <= 0 or num_segments <= 1:
        starts = [0]
    else:
        # 균등 간격 샘플링
        starts = np.linspace(0, max_start, num=num_segments, dtype=np.int64).tolist()
        # 중복 start 제거 (짧은 wav에서 num_segments가 과하면 중복이 생길 수 있음)
        starts = sorted(set(int(s) for s in starts))

    segments: list[np.ndarray] = []
    for st in starts:
        seg = wav[st : st + seg_len]
        if seg.shape[0] < seg_len:
            padded = np.zeros((seg_len,), dtype=np.float32)
            padded[: seg.shape[0]] = seg.astype(np.float32, copy=False)
            segments.append(padded)
        else:
            segments.append(seg.astype(np.float32, copy=False))

    return segments