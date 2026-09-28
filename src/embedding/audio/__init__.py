"""
- 로컬 오디오 파일 로딩/전처리(리샘플링, mono 변환, trim 등) 유틸 제공
"""

from src.embedding.audio.audio_io import load_audio_mono

__all__ = ["load_audio_mono"]