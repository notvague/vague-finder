"""
image/normalize.py
- 벡터 유틸(정규화 등)을 모아두는 곳
- 텍스트/이미지 모두 유사도 비교에서 L2 정규화를 반복적으로 사용 가능
- 중복 방지, 정규화 방식 변경 시 한 군데만 수정하면 됨
"""
from __future__ import annotations

import numpy as np


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    # L2 정규화
    return x / (np.linalg.norm(x) + 1e-12)