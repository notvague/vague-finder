"""
models/text_koe5.py
- Ko-E5 텍스트 임베딩 모델 래퍼
- SentenceTransformer 모델 로딩을 필요할 때만 하도록 분리 / 파이프라인에서 쉽게 재사용

[변경사항]
- embed_passages에 batch_size/show_progress_bar 옵션 추가
  (data 폴더 곡 수가 많아질 때 메모리/시간 관리 용이)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Hugging Face 모델 ID
DEFAULT_KOE5_MODEL = "nlpai-lab/KoE5"


@dataclass
class KoE5Embedder:
    """
    Ko-E5 임베딩을 생성하는 클래스

    필드:
    - model_name: 사용할 HF 모델 ID
    - _model: 내부 캐시(한 번 로드되면 재사용)
    """
    model_name: str = DEFAULT_KOE5_MODEL
    # 지정하면 Hugging Face 모델 저장소의 정확한 revision/commit을 고정한다.
    # 기존 호출부는 None을 사용하므로 동작이 바뀌지 않는다.
    revision: Optional[str] = None
    _model: Optional[object] = None
    _load_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    def load(self) -> "KoE5Embedder":
        """SentenceTransformer를 인스턴스당 한 번만 안전하게 로드한다."""
        if self._model is not None:
            return self

        with self._load_lock:
            # 첫 번째 스레드가 lock을 기다리는 동안 로드를 끝냈을 수 있으므로
            # lock 내부에서 다시 확인한다(double-checked locking).
            if self._model is not None:
                return self

            from sentence_transformers import SentenceTransformer

            # 생성이 실패하면 부분 객체를 캐시에 남기지 않는다. 다음 요청은
            # 깨끗한 상태에서 다시 시도할 수 있다.
            kwargs = {"revision": self.revision} if self.revision else {}
            model = SentenceTransformer(self.model_name, **kwargs)
            self._model = model
        return self

    @property
    def model(self):
        """
        외부에서는 model 속성으로 안전하게 접근
        호출 시 load()가 보장되도록
        """
        self.load()
        assert self._model is not None
        return self._model

    def embed_passages(
        self,
        passages: List[str],
        *,
        add_e5_prefix: bool = True,
        normalize: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        passages를 입력받아 임베딩을 생성
        - E5 계열 권장에 따라 "passage: " prefix 적용
        - normalize_embeddings=True로 코사인 유사도 계산이 쉬워지도록 L2 정규화 진행
        """
        inputs = [f"passage: {p}" for p in passages] if add_e5_prefix else passages
        emb = self.model.encode(
            inputs,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(emb)
