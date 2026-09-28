"""
image/image_io.py
- 이미지 입력을 다루는 I/O 모듈

[변경사항]
- 기존: URL(requests)만 지원
- 변경: 로컬 파일 경로도 지원
  - metadata["album_cover"]에 "data/raw/.../cover.jpg" 같은 경로가 들어와도 동작
"""
from __future__ import annotations

from pathlib import Path

import requests
from PIL import Image


def fetch_image(src: str, timeout: int = 20) -> Image.Image:
    """
    입력:
    -src: 이미지 URL 또는 로컬 파일 경로
     - URL: https://...
     - local: data/raw/.../cover.jpg
    """
    if src.startswith("http://") or src.startswith("https://"):
        resp = requests.get(src, stream=True, timeout=timeout)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")

    path = Path(src)
    if not path.is_absolute():
        path = Path.cwd() / path

    return Image.open(path).convert("RGB")