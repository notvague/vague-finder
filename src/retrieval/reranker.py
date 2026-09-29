"""
[Retrieval] Music Cross-Encoder Reranker

RRF/규칙 기반 부스팅으로 생성된 소수 후보를
한국어 Cross-Encoder로 다시 점수화하여 최종 순위를 정한다.

기본 모델: dragonkue/bge-reranker-v2-m3-ko
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.backend.schemas.search import MatchingTrack
from src.retrieval.explain import (
    RERANK_APPLIED,
    RERANK_SKIPPED,
    RerankRun,
    ScoreMix,
)

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RerankerConfig:
    """환경변수로 조정 가능한 리랭커 설정."""

    model_name: str = DEFAULT_RERANKER_MODEL
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 512
    # 운영 기본값 (2026-09-23 결정, dev 53 + test 23 측정 근거).
    #
    # **spread_ref와 쌍으로 움직인다.** 작은 spread 구간의 CE 민감도는
    # rerank_weight/spread_ref = 45라는 비율이 정하고, 큰 spread 구간은
    # rerank_weight가 상한이 된다. 한쪽만 바꾸면 아무도 측정하지 않은 조합이 된다.
    rerank_weight: float = 0.45
    # 0 이면 기존 결합식을 유지한다. 값이 있으면 rerank score spread 기반
    # 적응형 가중치를 쓴다. 위 rerank_weight와 비율 45를 이루도록 함께 정한다.
    spread_ref: float = 0.01
    # spread가 기준보다 작을 때는 상위 N개 안에서만 재정렬한다.
    # 낮은 신뢰도에서 11~30위 후보가 top10으로 튀어오르는 것을 막는다.
    low_conf_top_n: int = 5
    # spread가 이 값보다 작으면 **재정렬을 아예 생략**한다. 0이면 끈다(기본).
    #
    # low_conf_top_n과 다르다 — 그것은 상위 N개 안에서 여전히 재정렬한다.
    # 이쪽은 CE가 후보를 거의 구분하지 못하는 구간에서 검색 순서를 그대로 둔다.
    # 실험용 스위치다(RERANKER_MIN_SPREAD).
    min_spread: float = 0.0
    # **넘겨받은 후보 중 앞 N개만 채점한다.** 0이면 전부 채점한다(기본).
    #
    # CE 비용은 `후보 수 × 토큰 수`에 거의 비례한다(2026-09-24 측정: 후보 30개
    # 약 500토큰에 2.7초). 배치 크기로는 줄지 않으므로 줄일 곳은 둘뿐인데,
    # 이쪽이 채점할 후보 수를 줄이는 손잡이다.
    #
    # `low_conf_top_n`과 다르다 — 그것은 **전부 채점한 뒤** 상위 N개만 재정렬하므로
    # 시간이 줄지 않는다. 이쪽은 아예 채점하지 않는다.
    #
    # 채점하지 않은 후보는 **입력 순서 그대로 뒤에 붙는다.** 점수가 없으니 순서를
    # 정할 근거도 없다. 그 곡들은 `judged_ids`에도 들어가지 않는다 — 모델이 보지
    # 않은 곡에 "리랭킹됐다"를 붙이면 설명이 거짓이 된다.
    #
    # 실험용 스위치다(RERANKER_SCORE_TOP_N). **순위를 바꾸므로** 켜려면 재측정해야 한다.
    score_top_n: int = 0
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        # 두 값은 쌍이다. 한쪽만 .env에 남아 있으면 나머지는 새 기본값에서 와서
        # 측정한 적 없는 비율이 된다 — 예전 .env의 SPREAD_REF=0.02만 남으면
        # 0.45/0.02 = 22.5가 되어 dev/test 어느 측정과도 맞지 않는다.
        set_weight = os.getenv("RERANKER_WEIGHT") is not None
        set_spread = os.getenv("RERANKER_SPREAD_REF") is not None
        if set_weight != set_spread:
            logger.warning(
                "[MusicReranker] RERANKER_WEIGHT와 RERANKER_SPREAD_REF 중 하나만 "
                "설정돼 있습니다(%s). 둘은 쌍으로 움직입니다 — 한쪽만 두면 측정한 적 "
                "없는 비율이 됩니다. .env.example의 값을 함께 맞춰 주세요.",
                "WEIGHT만" if set_weight else "SPREAD_REF만",
            )

        weight = float(os.getenv("RERANKER_WEIGHT", "0.45"))
        return cls(
            model_name=os.getenv("RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL),
            device=os.getenv("RERANKER_DEVICE", "auto"),
            batch_size=max(1, int(os.getenv("RERANKER_BATCH_SIZE", "8"))),
            max_length=max(64, int(os.getenv("RERANKER_MAX_LENGTH", "512"))),
            rerank_weight=min(1.0, max(0.0, weight)),
            spread_ref=max(0.0, float(os.getenv("RERANKER_SPREAD_REF", "0.01"))),
            low_conf_top_n=max(1, int(os.getenv("RERANKER_LOW_CONF_TOP_N", "5"))),
            score_top_n=max(0, int(os.getenv("RERANKER_SCORE_TOP_N", "0"))),
            min_spread=max(0.0, float(os.getenv("RERANKER_MIN_SPREAD", "0"))),
            enabled=_env_bool("RERANKER_ENABLED", True),
        )


class MusicReranker:
    """
    query-document 쌍을 함께 입력하는 Cross-Encoder 리랭커.

    모델은 첫 리랭킹 요청 때 지연 로딩한다. 모델 다운로드/로딩 실패 시
    호출부(SearchRouter)가 기존 검색 순서로 안전하게 폴백할 수 있도록
    예외를 그대로 전달한다.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig.from_env()
        self._tokenizer = None
        self._model = None
        self._device: Optional[torch.device] = None
        self._load_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _resolve_device(self) -> torch.device:
        requested = self.config.device.strip().lower()
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self) -> "MusicReranker":
        """토크나이저와 분류 모델을 한 번만 로드한다."""
        if self._model is not None and self._tokenizer is not None:
            return self

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return self

            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = self._resolve_device()
            model_kwargs = {}
            if device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

            logger.info(
                "[MusicReranker] 모델 로딩 시작 model=%s device=%s",
                self.config.model_name,
                device,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                **model_kwargs,
            )
            model.to(device).eval()

            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            logger.info("[MusicReranker] 모델 로딩 완료")

        return self

    @staticmethod
    def build_document(track: MatchingTrack) -> str:
        """
        벡터 DB 메타데이터를 리랭커가 비교하기 좋은 자연어 문서로 변환한다.
        빈 필드는 제외하여 불필요한 토큰 사용을 줄인다.
        """

        def add(parts: List[str], label: str, value) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple)):
                text = ", ".join(str(x).strip() for x in value if str(x).strip())
            else:
                text = str(value).strip()
            if text:
                parts.append(f"{label}: {text}")

        parts: List[str] = []
        add(parts, "곡 제목", track.title)
        add(parts, "아티스트", track.artist)
        add(parts, "발매일", track.release_date)
        add(parts, "장르", track.genre)
        add(parts, "발매일", track.release_date)
        add(parts, "아티스트 구성", track.type)
        add(parts, "보컬 성별", track.vocal_gender)
        add(parts, "가수 구성", track.artist_types)
        add(parts, "검색용 분위기 설명", track.search_style_summary)
        add(parts, "분위기 태그", track.mood_tags)
        add(parts, "시간·날씨 태그", track.time_weather_tags)
        add(parts, "장소·활동 태그", track.place_activity_tags)
        add(parts, "감정 태그", track.emotion_tags)
        add(parts, "바이브 태그", track.vibe_tags)
        add(parts, "관계·상황 태그", track.relation_context_tags)
        add(parts, "색감 태그", track.color_tags)
        add(parts, "사운드 태그", track.sound_tags)
        add(parts, "플레이리스트 태그", track.melon_playlist_tags)
        add(parts, "시각적 이미지", track.visual_imagery)
        add(parts, "가사 핵심 구절", track.lyrics_highlight)
        add(parts, "가사 내용 요약", track.lyrics_summary)
        add(parts, "청취자 반응", track.sentiment_summary)
        add(parts, "팬 태그", track.fans_tags)
        add(parts, "대표 감정", track.major_emotion)
        return "\n".join(parts)

    def score(self, query: str, tracks: Sequence[MatchingTrack]) -> List[float]:
        """각 (query, track document) 쌍의 sigmoid 관련도 점수를 반환한다."""
        if not tracks:
            return []

        self.load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        documents = [self.build_document(track) for track in tracks]
        scores: List[float] = []

        for start in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[start : start + self.config.batch_size]
            batch_queries = [query] * len(batch_docs)
            features = self._tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            features = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in features.items()
            }

            with torch.inference_mode():
                logits = self._model(**features).logits.reshape(-1)
                batch_scores = torch.sigmoid(logits).detach().float().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)

        return scores

    @staticmethod
    def _minmax(values: Sequence[float]) -> List[float]:
        if not values:
            return []
        low = min(values)
        high = max(values)
        if high - low < 1e-12:
            return [1.0] * len(values)
        return [(value - low) / (high - low) for value in values]

    def rerank(
        self,
        query: str,
        tracks: Sequence[MatchingTrack],
        top_k: int,
    ) -> List[MatchingTrack]:
        """
        최종 점수 = rerank_weight * CrossEncoder 점수
                  + (1-rerank_weight) * 정규화된 기존 검색 점수

        기존 RRF/명시 단서 부스팅을 완전히 버리지 않고 함께 사용하므로
        제목·아티스트·가사 키워드 같은 강한 단서도 보존된다.
        """
        return self.rerank_run(query, tracks, top_k).tracks

    def rerank_run(
        self,
        query: str,
        tracks: Sequence[MatchingTrack],
        top_k: int,
    ) -> RerankRun:
        """결과와 함께 **실행 상태**를 돌려준다.

        호출 여부만으로는 리랭킹이 실제로 일어났는지 알 수 없다. 꺼져 있거나
        후보가 없으면 입력을 그대로 돌려주므로, 그것을 '리랭킹했다'고 기록하면
        설명이 거짓이 된다.
        """
        if not tracks:
            return RerankRun([], RERANK_SKIPPED, [])
        if not self.enabled:
            return RerankRun(list(tracks)[:top_k], RERANK_SKIPPED, [])

        # 앞 N개만 채점한다. 나머지는 입력 순서 그대로 뒤에 붙는다.
        cap = self.config.score_top_n
        judged = list(tracks) if cap <= 0 else list(tracks)[:cap]
        unjudged = [] if cap <= 0 else list(tracks)[cap:]

        # **검색 점수는 받은 후보 전체로 정규화한다.** 채점을 앞 N개로 줄였다고
        # 검색 점수의 기준까지 좁힐 이유가 없다 — 그 값은 이미 30개 모두에 대해
        # 손에 있고, CE 호출도 더 들지 않는다.
        #
        # 좁히면 합성의 저울이 통째로 달라진다. q201에서 정답−오답 합성 점수 차이가
        # +0.000849(30곡 채점) → **−0.010278**(20곡 채점)로 뒤집혔는데, 같은 CE 결과에
        # 검색 정규화만 30곡 기준으로 되돌리면 +0.000848로 돌아왔다. 주범은 spread의
        # 미세한 변화가 아니라 **이 범위 축소**였다.
        #
        # CE 쪽(`normalized_rerank`, `spread`)은 채점한 곡으로만 낼 수밖에 없다.
        # 채점하지 않은 곡의 CE 점수는 존재하지 않는다.
        all_retrieval = [float(track.score) for track in tracks]
        all_normalized = self._minmax(all_retrieval)
        retrieval_scores = all_retrieval[: len(judged)]
        normalized_retrieval = all_normalized[: len(judged)]
        rerank_scores = self.score(query, judged)

        # spread_ref가 0이면 기존 결합식을 그대로 쓴다. 값이 있으면:
        #   - score 폭이 충분히 클 때만 reranker 영향력을 키우고
        #   - score 폭이 작아 구분력이 낮을 때는 retrieval 순서를 더 존중한다.
        spread = (max(rerank_scores) - min(rerank_scores)) if rerank_scores else 0.0

        if self.config.min_spread > 0.0 and spread < self.config.min_spread:
            # CE가 후보를 거의 구분하지 못한다. 모델은 돌았지만 **순서를 정하지 않는다.**
            #
            # 점수도 검색 점수를 그대로 둔다 — 합성 점수만 넣고 순서를 유지하면
            # 점수와 순서가 어긋나 "이 곡이 왜 위에 있나"를 설명할 수 없다.
            # rerank_score는 진단용으로 붙인다(평가의 spread 열이 이것을 읽는다).
            #
            # 상태는 skipped다. failed가 아니고(실패가 아니다) applied도 아니다
            # (최종 순서가 모델의 것이 아니다).
            logger.debug(
                "[MusicReranker] spread=%.8g < min_spread=%.8g — 재정렬 생략",
                spread,
                self.config.min_spread,
            )
            kept = [
                track.model_copy(update={"rerank_score": float(score)})
                for track, score in zip(judged, rerank_scores)
            ]
            return RerankRun([*kept, *unjudged][:top_k], RERANK_SKIPPED, [])

        if self.config.spread_ref > 0.0:
            confidence = min(1.0, spread / self.config.spread_ref)
            weight = self.config.rerank_weight * confidence
            normalized_rerank = self._minmax(rerank_scores)
        else:
            confidence = 1.0
            weight = self.config.rerank_weight
            normalized_rerank = list(rerank_scores)

        logger.debug(
            "[MusicReranker] spread=%.8g spread_ref=%.8g confidence=%.3f effective_weight=%.3f",
            spread,
            self.config.spread_ref,
            confidence,
            weight,
        )

        # 합성식을 곡별로 남긴다. rerank_score만 보면 "CE 점수가 가장 높은 곡이
        # 1위"일 것 같지만 실제로는 아니다 — 검색 점수가 함께 섞이기 때문이다.
        # 그 비율을 남기지 않으면 설명이 순서의 원인을 리랭커에 전부 돌리게 된다.
        mixes: Dict[str, ScoreMix] = {}

        reranked: List[MatchingTrack] = []
        for track, retrieval_score, retrieval_norm, rerank_score, rerank_norm in zip(
            judged,
            retrieval_scores,
            normalized_retrieval,
            rerank_scores,
            normalized_rerank,
        ):
            final_score = weight * rerank_norm + (1.0 - weight) * retrieval_norm
            mixes[str(track.id)] = ScoreMix(
                backend="cross_encoder",
                weight=float(weight),
                configured_weight=float(self.config.rerank_weight),
                confidence=float(confidence),
                rerank_component=float(rerank_norm),
                retrieval_component=float(retrieval_norm),
                final=float(final_score),
                # spread_ref가 꺼져 있으면 CE **원점수**를 그대로 섞는다.
                # 그때 "정규화된 리랭커 점수"라고 적으면 사실이 아니다.
                rerank_normalized=self.config.spread_ref > 0.0,
                detail=f"검색 점수는 min-max 정규화, CE spread={spread:.6g}",
            )
            reranked.append(
                track.model_copy(
                    update={
                        "score": float(final_score),
                        "retrieval_score": retrieval_score,
                        "rerank_score": float(rerank_score),
                    }
                )
            )

        # spread가 충분히 크면 candidate 전체를 재정렬한다.
        # spread가 작으면 Cross-Encoder가 후보 전체를 확실히 구분하지 못하는 상태로
        # 보고, 원래 검색 상위 N개 안에서만 재정렬한다. 그 아래 후보는 기존 순서를
        # 그대로 유지해 저신뢰 후보가 top10으로 튀어오르는 것을 막는다.
        if self.config.spread_ref > 0.0 and spread < self.config.spread_ref:
            head_n = min(self.config.low_conf_top_n, len(reranked))
            head = sorted(
                reranked[:head_n],
                key=lambda track: track.score,
                reverse=True,
            )
            reranked = head + reranked[head_n:]
            strategy = f"low-confidence-top{head_n}"
            strategy_label = f"저신뢰 — 상위 {head_n}개만 재정렬"
            # 아래 후보는 합성 점수를 받았지만 **순서에는 쓰이지 않았다.**
            reordered_ids = {str(track.id) for track in head}
        else:
            reranked.sort(key=lambda track: track.score, reverse=True)
            strategy = "full-candidate"
            strategy_label = "후보 전체 재정렬"
            reordered_ids = set(mixes)

        if unjudged:
            # 채점하지 않은 후보는 **맨 뒤**다. 점수가 없으므로 합성 점수도,
            # 순서를 정할 근거도 없다.
            reranked = [*reranked, *unjudged]
            strategy = f"{strategy}-scored{len(judged)}"
            strategy_label = f"{strategy_label} · 앞 {len(judged)}개만 채점"

        for song_id, mix in mixes.items():
            mix.strategy = strategy_label
            mix.reordered = song_id in reordered_ids

        logger.debug(
            "[MusicReranker] strategy=%s low_conf_top_n=%d",
            strategy,
            self.config.low_conf_top_n,
        )
        # 실제로 채점한 곡만 judged_ids에 넣는다. `score_top_n`으로 잘린 뒤쪽
        # 후보는 모델이 본 적이 없으므로 "리랭킹됐다"고 적으면 안 된다.
        return RerankRun(
            reranked[:top_k],
            RERANK_APPLIED,
            [str(track.id) for track in judged],
            mixes=mixes,
        )
