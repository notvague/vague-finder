"""
runtime.py
- embedding 모듈 전반에서 공통으로 쓴느 실행 환경 관련 유틸
- device 선택 (CPU/GPU), 환경변수 기반 강제 옵션 등을 관리
"""
from __future__ import annotations

import os
from typing import Optional

import torch

def get_device(force_cpu: Optional[bool] = None) -> torch.device:
    """
    디바이스 선택:
    - 기본: CUDA 가능하면 cuda, 아니면 cpu
    - 강제 CPU: force_cpu=True 또는 ENV FORCE_CPU=1

    주의:
    - Docker에서 GPU가 있어도 컨테이너에 GPU 접근 설정이 없으면 is_available()는 False가 될 수 있다.
    - 따라서 이 함수는 "가능하면 GPU" 전략이며, GPU 사용을 보장하지는 않는다.
    """
    if force_cpu is None:
        force_cpu = os.getenv("FORCE_CPU", "0") in {"1", "true", "True", "YES", "yes"}

    if force_cpu:
        return torch.device("cpu")

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")