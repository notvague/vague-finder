"""
Search evaluation metrics -- 순수 함수.

ranked_ids: 시스템이 반환한 top-K song_id 리스트 (1위가 [0])
positives:  정답 song_id 집합
negatives:  '나오면 안 되는' song_id 집합 (함정)

모든 함수는 의존성 없이 동작하므로 search system 없이 단위 테스트 가능.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Set


def recall_at_k(ranked_ids: List[str], positives: Set[str], k: int) -> float:
    """top-K 안에 들어온 positives 비율 (multi-label fractional recall).

    positives 비어있으면 0.0. low_signal tier 등 정답이 없는 케이스는 호출 측에서
    이 함수 자체를 건너뛰는 것이 의미상 옳다.
    """
    if not positives:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(positives & top_k) / len(positives)


def mrr(ranked_ids: List[str], positives: Set[str]) -> float:
    """첫 번째 positive 의 reciprocal rank. 못 찾으면 0.0."""
    if not positives:
        return 0.0
    for rank, sid in enumerate(ranked_ids, start=1):
        if sid in positives:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: List[str], positives: Set[str], k: int) -> float:
    """Binary relevance NDCG@K.

    DCG = Σ rel_i / log2(i+1)        for i in 1..K
    IDCG = DCG of perfect ranking    = Σ 1/log2(i+1) for i in 1..min(|P|, K)
    """
    if not positives:
        return 0.0
    dcg = 0.0
    for rank, sid in enumerate(ranked_ids[:k], start=1):
        if sid in positives:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_count = min(len(positives), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0 else 0.0


def negative_hit_rate_at_k(ranked_ids: List[str], negatives: Set[str], k: int) -> float:
    """top-K 에 negative 가 등장한 비율. 낮을수록 좋음 (오탐 적음).

    negatives 비어있으면 0.0 (함정 정의 안 된 query 는 평가 제외).
    """
    if not negatives:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(negatives & top_k) / len(negatives)


def mean_negative_rank_when_present(
    ranked_ids: List[str], negatives: Set[str]
) -> Optional[float]:
    """탑K 에 *실제로 등장한* negatives 의 평균 순위.

    아무 negative 도 안 들어왔으면 None 반환 (좋은 신호).
    값이 작을수록 나쁨 (함정이 상위에 노출됨).
    """
    if not negatives:
        return None
    ranks = [
        rank for rank, sid in enumerate(ranked_ids, start=1) if sid in negatives
    ]
    if not ranks:
        return None
    return sum(ranks) / len(ranks)


def diversity_at_k(
    ranked_ids: List[str], get_artist: Callable[[str], str], k: int
) -> float:
    """top-K 의 unique 아티스트 비율. low_signal query 의 graceful handling 평가용.

    1.0 = 모두 다른 아티스트, 0.1 = 한 아티스트가 K곡 독점.
    """
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    artists = {get_artist(sid) for sid in top_k}
    return len(artists) / len(top_k)
