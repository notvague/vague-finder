"""
models/audio_clap.py
- transformers의 CLAPModel / CLAPProcessor 기반 오디오 임베딩 래퍼

설계 목표
1) 모델 로딩/디바이스 이동/추론(eval+no_grad)을 캡슐화
2) 입력: "이미 로딩된 waveform(float32 1D numpy)" 리스트
3) 출력: numpy (batch, dim) 임베딩 + L2 normalize
"""

from __future__ import annotations

import os
import threading
from typing import List

import numpy as np
import torch

from src.embedding.image.normalize import l2_normalize_np

DEFAULT_CLAP_CKPT = "laion/clap-htsat-fused"

# CLAP 특징 추출 시드.
#
# laion/clap-htsat-fused의 특징 추출기는 truncation="fusion"으로 동작한다. 입력이
# max_length_s(10초)보다 길면 여러 구간을 **np.random으로** 골라 합치는데, 시드를
# 잡지 않으면 같은 파일을 다시 임베딩해도 다른 벡터가 나온다(실측 코사인 0.961).
# 그러면 sound 맵이 재생성마다 움직이고 오디오 쪽 측정을 기준선으로 쓸 수 없다.
#
# 배치 단위로 한 번만 시드를 잡는 것으로는 부족하다. 한 배치 안에서 앞 항목이 난수를
# 몇 개 쓰느냐에 따라 뒤 항목의 구간이 달라지므로, 같은 곡이라도 배치 위치가 바뀌면
# 벡터가 바뀐다(실측 0.967~0.989). 곡을 이어서 추가하는 이 프로젝트에서는 치명적이다.
# 그래서 **항목마다** 같은 시드로 되돌린다(실측 1.000000, 배치 구성과 무관).
CLAP_FEATURE_SEED = int(os.getenv("CLAP_SEED", "0"))

# 시드를 심는 대상이 numpy의 **전역** 난수 상태다(특징 추출기가 np.random을 직접 쓰므로
# 다른 방법이 없다). 두 스레드가 동시에 들어오면 서로의 시드를 덮어써 재현성이 깨진다.
# 특징 추출 구간을 직렬화해서 막는다. 추론은 이 구간 밖에서 배치로 돈다.
_FEATURE_RNG_LOCK = threading.Lock()


class CLAPAudioEmbedder:
    """
    CLAP 오디오 임베딩 생성 클래스

    필드:
    - ckpt: HF 체크포인트 ID
    - device: torch.device (None이면 load 시점에 자동 결정)
    - _model, _processor: 내부 캐시(한 번 로드 후 재사용)
    """
    def __init__(self, ckpt: str = DEFAULT_CLAP_CKPT, device: torch.device | None = None):
        self.ckpt = ckpt
        self.device = device
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()

    def load(self) -> "CLAPAudioEmbedder":
        """
        모델/프로세서 로드 및 디바이스 세팅
        - device가 None이면 CUDA 가능 시 cuda, 아니면 cpu
        - model.eval() 로 드롭아웃 비활성화/추론 최적화
        """
        if self._model is not None and self._processor is not None:
            return self

        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return self

            device = self.device or (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            from transformers import ClapModel, ClapProcessor

            # 두 구성요소가 모두 준비되기 전에는 공유 캐시를 갱신하지 않는다.
            # 실패 시 다음 요청이 부분 초기화 객체를 재사용하는 문제도 차단한다.
            processor = self._processor
            if processor is None:
                processor = ClapProcessor.from_pretrained(self.ckpt)

            model = self._model
            if model is None:
                model = (
                    ClapModel
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

    def embed_audios(
        self,
        audios: List[np.ndarray],
        *,
        sampling_rate: int = 48_000,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        오디오 리스트를 받아 임베딩을 생성.

        1) processor(audio=..., sampling_rate=..., return_tensors="pt")
           - CLAP_FEATURE_SEED 참고: 항목마다 시드를 되돌려 특징을 뽑고 배치로 합친다.
             배치로 한 번에 넘기면 같은 곡이라도 배치 위치에 따라 벡터가 달라진다.
        2) model.get_audio_features(**inputs)   ← 추론은 배치로 한 번만
        3) cpu numpy 변환
        4) (옵션) L2 normalize
        """
        self.load()
        assert self.device is not None

        if not audios:
            raise ValueError("embed_audios: 빈 리스트는 임베딩할 수 없습니다.")

        # 전역 numpy 난수 상태를 빌려 쓰고 그대로 돌려놓는다. 호출자(예: 맵의 UMAP)가
        # 자기 난수 흐름을 쓰고 있을 수 있으므로 흔들지 않는다.
        with _FEATURE_RNG_LOCK:
            rng_state = np.random.get_state()
            try:
                per_item = []
                for audio in audios:
                    np.random.seed(CLAP_FEATURE_SEED)
                    per_item.append(
                        self.processor(
                            audios=[audio],
                            sampling_rate=sampling_rate,
                            return_tensors="pt",
                        )
                    )
            finally:
                np.random.set_state(rng_state)

        # 특징 추출기는 input_features와 is_longer만 돌려준다. 둘 다 배치 축이 0번이다.
        inputs = {
            key: torch.cat([item[key] for item in per_item], dim=0)
            for key in per_item[0]
        }

        # processor 결과는 보통 dict-like. to(device) 가능한 텐서들만 이동.
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_audio_features(**inputs)

        if isinstance(outputs, torch.Tensor):
            feats = outputs
        elif hasattr(outputs, "audio_embeds") and outputs.audio_embeds is not None:
            feats = outputs.audio_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            raise RuntimeError(
                f"Unexpected output from get_audio_features(): {type(outputs)}. "
                "No audio_embeds/pooler_output found."
            )

        vecs = feats.detach().cpu().numpy()


        if l2_normalize:
            vecs = np.stack([l2_normalize_np(v) for v in vecs], axis=0)

        return vecs

    def embed_texts(self, texts: List[str], *, l2_normalize: bool = True) -> np.ndarray:
        """
        텍스트 쿼리를 받아 CLAP 오디오 임베딩 공간으로 매핑
        """
        self.load()
        assert self.device is not None

        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        vecs = outputs.detach().cpu().numpy()
        if l2_normalize:
            vecs = np.stack([l2_normalize_np(v) for v in vecs], axis=0)

        return vecs
