"""
models/image_siglip2.py
- AutoModel / AutoProcessor 로딩 캡슐화
- device 이동, eval 모드, 이미지/텍스트 배치 임베딩을 통일된 인터페이스로 제공
- 결과 벡터를 L2 정규화하여 코사인 유사도 기반 비교를 쉽게 함

[2026-05-25 수정] cross-modal alignment 버그 fix
- 이전: AutoImageProcessor + AutoTokenizer 따로 로드 + padding=True
  -> SigLIP family는 고정 길이로 학습되어 dynamic padding 사용 시 텍스트 임베딩이
    이미지 공간과 정렬되지 않음. 측정 결과 text<->image cosine ≈ 0.06 평탄
    (정상 매칭이라면 0.1+ 기대) → vague × image R@10 = 0 의 직접 원인
- 수정: AutoProcessor (text+image 통합) + processor.tokenizer + padding="max_length"
  → matched cosine 0.10~0.13, mismatched ~0, +0.20 으로 정상 작동 확인
- 부수 발견: checkpoint config.model_type 이 'siglip'(SigLIP1) — 모델 ID가
  'siglip2-base-patch16-224' 지만 실제 가중치는 SigLIP1. transformers
  4.49+ 에 추가된 Siglip2Model 클래스(NaFlex 패치 임베딩)로는 로드 불가.
  AutoModel 에 위임하면 SiglipModel 로 올바르게 로드됨.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from src.embedding.image.normalize import l2_normalize_np

# base-patch-224: 비교적 가볍고 빠른 버전(224X224 입력 기준)
# 주의: 모델 ID는 'siglip2' 이지만 실제 config 는 SigLIP1 (위 docstring 참조).
DEFAULT_SIGLIP2_CKPT = "google/siglip2-base-patch16-224"

SIGLIP_TEXT_MAX_LENGTH = 64


@dataclass
class SigLIP2Embedder:
    """
    SigLIP2 임베딩 생성 클래스.

    필드:
    - ckpt: HF 체크포인트 ID
    - device: torch.device (None이면 load 시점에 자동 결정)
    - _model, _processor: 내부 캐시 (한 번 로드 후 재사용)
    """
    ckpt: str = DEFAULT_SIGLIP2_CKPT
    device: Optional[torch.device] = None
    _model: Optional[torch.nn.Module] = None
    _processor: Optional[object] = None
    _load_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    def load(self) -> "SigLIP2Embedder":
        """
        모델 및 processor 로드.
        - device 가 None 이면 CUDA → MPS → CPU 순으로 자동 선택
        - model 은 eval() 모드로 전환해 추론 최적화/드롭아웃 비활성화
        - processor 는 AutoProcessor (text+image 통합) — tokenizer 는 .tokenizer 로 접근
        """
        if self._model is not None and self._processor is not None:
            return self

        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return self

            if self.device is not None:
                device = self.device
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            from transformers import AutoModel, AutoProcessor

            # model/processor를 로컬 변수에서 모두 준비한 뒤 한 번에 공개한다.
            # 어느 한쪽 로드가 실패해도 다른 스레드가 부분 초기화 상태를 보지 않는다.
            processor = self._processor
            if processor is None:
                processor = AutoProcessor.from_pretrained(self.ckpt)

            model = self._model
            if model is None:
                model = (
                    AutoModel
                    .from_pretrained(self.ckpt)
                    .to(device)
                    .eval()
                )

            self.device = device
            self._processor = processor
            self._model = model

        return self

    @property
    def model(self) -> torch.nn.Module:
        self.load()
        assert self._model is not None
        return self._model

    @property
    def processor(self):
        self.load()
        assert self._processor is not None
        return self._processor

    def embed_images(
        self,
        images: List[Image.Image],
        *,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        PIL 이미지 리스트를 SigLIP2 이미지 임베딩으로 변환한다.
        """

        self.load()
        assert self.device is not None

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                out = self.model.get_image_features(**inputs)
            else:
                out = self.model(**inputs)

        feats = self._extract_features(out)
        vecs = feats.detach().cpu().numpy()

        if l2_normalize:
            vecs = np.stack(
                [l2_normalize_np(vector) for vector in vecs],
                axis=0,
            )

        return vecs

    def embed_texts(
        self,
        texts: List[str],
        *,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        텍스트 리스트를 이미지 공간에 정렬된 텍스트 임베딩으로 변환

        - 소문자화
        - padding="max_length"
        - max_length=64
        - truncation=True

        따라서 검색 문장의 실제 길이에 상관없이 input_ids의 shape은
        항상 [batch_size, 64]가 됨
        """

        self.load()
        assert self.device is not None

        normalized_texts = [
            str(text).strip().lower()
            for text in texts
        ]

        inputs = self.processor(
            text=normalized_texts,
            padding="max_length",
            max_length=SIGLIP_TEXT_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        actual_length = inputs["input_ids"].shape[1]

        if actual_length != SIGLIP_TEXT_MAX_LENGTH:
            raise RuntimeError(
                "SigLIP2 text input length mismatch: "
                f"expected={SIGLIP_TEXT_MAX_LENGTH}, "
                f"actual={actual_length}"
            )

        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                out = self.model.get_text_features(**inputs)
            else:
                out = self.model(**inputs)

        feats = self._extract_features(out)
        vecs = feats.detach().cpu().numpy()

        if l2_normalize:
            vecs = np.stack(
                [l2_normalize_np(vector) for vector in vecs],
                axis=0,
            )

        return vecs

    @staticmethod
    def _extract_features(out) -> torch.Tensor:
        """
        Transformers 버전에 따라 달라질 수 있는 모델 출력에서
        실제 임베딩 텐서를 추출
        """
        if isinstance(out, torch.Tensor):
            return out
        
        for attribute_name in (
            "pooler_output",
            "text_embeds",
            "image_embeds",
        ):
            value = getattr(out, attribute_name, None)

            if value is not None:
                return value

        raise TypeError(
            f"Unexpected SigLIP model output: {type(out)}"
        )
