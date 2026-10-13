"""피드백 검토·색인·검색 재사용 설정. 세 스위치는 서로 독립이다.

- `FEEDBACK_COLLECTION_ENABLED` — 수집(검색 기록·제출). `store.collection_enabled`
- `FEEDBACK_INDEX_WORKER_ENABLED` — 승인 사례 색인과 만료·무효 사례 정리
- `FEEDBACK_RETRIEVAL_ENABLED` — 검색 후보 보충

수집을 꺼도 색인 작업은 돌 수 있다 — 이미 쌓인 자료의 정리가 수집 스위치에 묶이면 만료 정리가 멈춘다.
`FEEDBACK_EVIDENCE_ENABLED`(리랭커에 사유 전달)는 후속 작업의 예약 이름이고 **읽는 코드가 없다.**

숫자 설정은 비었거나, 숫자가 아니거나, 유한하지 않거나(nan·inf), 범위를 벗어나면 **기본값**으로 읽는다.
잘못 적은 값이 문턱을 0으로 만들어 넓은 사례까지 들이는 쪽으로 풀리지 않게 하려는 것이다.
"""
from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from src.embedding.models.text_koe5 import DEFAULT_KOE5_MODEL

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes"}
# 잘못된 설정 경고는 값마다 한 번만 낸다. 검색 설정은 요청마다 읽으므로 그대로 두면 로그가 요청 수만큼 쌓인다.
_warned: set = set()


def _warn_once(name: str, raw: str, default: object) -> None:
    if (name, raw) not in _warned:
        _warned.add((name, raw))
        logger.warning("%s 값을 쓸 수 없어 기본값 %s을(를) 쓴다(숫자가 아니거나 범위 밖)", name, default)

# 사례 벡터를 만든 방식. 모델·입력 규약(E5 passage 접두사)·정규화가 바뀌면 이 값을 올린다 —
# 값이 다른 포인트는 검색에서 걸러지고 색인 작업이 다시 만든다.
EMBEDDING_VERSION = f"koe5:{DEFAULT_KOE5_MODEL}:passage:l2:v1"

# 승인 문장과 검토 사유의 길이 상한. 사용자 사유(1,000자)보다 짧게 둔다 — 재사용 단서는 요약이다.
MAX_REUSABLE_TEXT = 500
MAX_REVIEW_NOTE = 1000


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE


def env_float(name: str, default: float, *, low: float, high: float, low_inclusive: bool = True) -> float:
    """범위 `[low, high]`(또는 `(low, high]`) 안의 유한한 수만 받는다. 아니면 기본값."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    below = value < low if low_inclusive else value <= low
    if not math.isfinite(value) or below or value > high:
        _warn_once(name, raw, default)
        return default
    return value


def env_int(name: str, default: int, *, low: int, high: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        _warn_once(name, raw, default)
        return default
    if value < low or value > high:
        _warn_once(name, raw, default)
        return default
    return value


def index_worker_enabled() -> bool:
    return env_flag("FEEDBACK_INDEX_WORKER_ENABLED")


def retrieval_enabled() -> bool:
    return env_flag("FEEDBACK_RETRIEVAL_ENABLED")


@dataclass(frozen=True)
class RetrievalSettings:
    """후보 보충 설정. **실험 전 잠정값이다** — 채택 기준(`docs/feedback.md`)을 넘긴 값으로 바꾼다.

    - top_cases: Qdrant에서 읽을 유사 사례 수
    - max_songs: 보충할 곡 수 상한(같은 곡의 사례는 하나로 접는다)
    - min_similarity: 사례 문장과 질의의 코사인 유사도 문턱. 이보다 낮으면 버린다.
      잠정값 0.65의 근거는 합성 질의 10건 × 사례 14건뿐이다(KoE5, 2026-10-13,
      `experiments/feedback_reuse/results_synthetic_v01`): 같은 곡을 따로 쓴 말로 가리킨 쌍이 0.58~0.75,
      서로 무관한 쌍이 0.63 이하였다. **문턱은 약한 유사성만 거른다** — 넓은 단서("슬픈 발라드")나 잘못
      승인된 사례는 질의와 말이 같으면 0.8을 넘으므로 유사도로는 못 거른다. 그것은 검토에서 거른다.
    - max_songs의 실제 상한은 이 값과 "후보 수 − 표시 수" 가운데 작은 쪽이다(리랭커 없이는 표시되지 않는 자리 수)
    - fusion_weight: **진입 판정**에 쓰는 가중. 후보 밖 곡은 밀려날 꼬리 후보의 보정 전 융합 점수 × 이 값을
      받고(`search_router.fuse_feedback_supplement`), 명시 단서 보정까지 받은 점수가 절단을 넘어야 들어온다.
      1.0이면 꼬리 후보와 보정이 같거나 많을 때, 1보다 작으면 보정이 더 많을 때만 들어온다.
      **들어온 뒤의 자리는 이 값과 무관하게 후보의 끝이다**(`pin_feedback_entrants`)
    - budget_seconds: 기존 경로가 끝난 뒤 보충을 더 기다리는 상한(최신 상태 확인 포함)
    - display_k: **측정 전용.** 호출자가 라우터에 후보 전체를 요청할 때(top_k = 후보 수) 실제 표시 수를 알려 준다.
      기존 측정기(`evaluate_search_accuracy` 등)는 후보 30곡을 통째로 받아 자기 쪽에서 리랭킹하고 상위 10곡만
      쓴다 — 라우터가 보는 top_k는 30이라 "후보 수 − 표시 수"가 0이 되어 보충이 한 건도 일어나지 않는다.
      그때만 이 값을 표시 수로 쓴다. 서비스 요청(top_k < 후보 수)에서는 읽지 않는다.
      **환경 변수로는 정할 수 없다** — 측정 프로세스가 `measurement_display()`로만 정한다.
    """

    top_cases: int = 20
    max_songs: int = 3
    min_similarity: float = 0.65
    fusion_weight: float = 1.0
    budget_seconds: float = 0.6
    display_k: Optional[int] = None

    def display_count(self, top_k: int, candidates: int) -> int:
        """보충 곡이 넘어서면 안 되는 표시 수. 라우터의 top_k가 기본이고, 후보 전체를 요청한 호출에서만 display_k를 쓴다."""
        if top_k >= candidates and self.display_k is not None:
            return min(self.display_k, candidates)
        return top_k

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        base = cls()
        return cls(
            top_cases=env_int("FEEDBACK_TOP_CASES", base.top_cases, low=1, high=100),
            max_songs=env_int("FEEDBACK_SUPPLEMENT_SONGS", base.max_songs, low=1, high=10),
            min_similarity=env_float("FEEDBACK_MIN_SIMILARITY", base.min_similarity, low=0.0, high=1.0),
            fusion_weight=env_float(
                "FEEDBACK_FUSION_WEIGHT", base.fusion_weight, low=0.0, high=2.0, low_inclusive=False,
            ),
            budget_seconds=env_float("FEEDBACK_LOOKUP_BUDGET_SECONDS", base.budget_seconds, low=0.05, high=5.0),
            display_k=_measurement_display_k,
        )


# 측정기가 결과로 쓰는 곡 수. 이 프로세스 안에서만 정해진다(`measurement_display`). 서버는 정하지 않으므로 항상 None이다.
_measurement_display_k: Optional[int] = None
MAX_DISPLAY_K = 100


@contextmanager
def measurement_display(display_k: int) -> Iterator[None]:
    """블록 안에서 읽는 보충 설정(`RetrievalSettings.from_env`)에 측정기의 표시 수를 싣는다. 끝나면 되돌린다.

    환경 변수로 받지 않는 이유: 측정기와 서버가 같은 `.env`를 읽는다. 환경 변수였다면 그 값이 서버 환경에 들어갔을 때
    후보 전체를 요청한 공개 API 호출(top_k = candidate_k)에서 보충 곡이 리랭커 없이도 표시 결과에 실린다.
    """
    global _measurement_display_k
    if isinstance(display_k, bool) or not isinstance(display_k, int) or not 1 <= display_k <= MAX_DISPLAY_K:
        raise ValueError(f"표시 수는 1~{MAX_DISPLAY_K}의 정수다: {display_k!r}")
    previous, _measurement_display_k = _measurement_display_k, display_k
    try:
        yield
    finally:
        _measurement_display_k = previous


@dataclass(frozen=True)
class WorkerSettings:
    """색인·정리 작업 설정. 한 주기에 처리하는 양을 묶어 검색과 모델·DB를 나눠 쓴다."""

    interval_seconds: float = 60.0
    index_batch: int = 16     # 한 주기에 색인할 사례 수
    embed_batch: int = 8      # 한 번에 임베딩할 문장 수
    sweep_batch: int = 200    # 한 주기에 대조할 포인트·문서 수

    @classmethod
    def from_env(cls) -> "WorkerSettings":
        base = cls()
        return cls(
            interval_seconds=env_float(
                "FEEDBACK_WORKER_INTERVAL_SECONDS", base.interval_seconds, low=1.0, high=3600.0,
            ),
            index_batch=env_int("FEEDBACK_INDEX_BATCH", base.index_batch, low=1, high=256),
            embed_batch=env_int("FEEDBACK_EMBED_BATCH", base.embed_batch, low=1, high=64),
            sweep_batch=env_int("FEEDBACK_SWEEP_BATCH", base.sweep_batch, low=1, high=2000),
        )
