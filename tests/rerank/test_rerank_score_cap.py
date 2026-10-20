"""CE가 **채점할 후보 수**를 줄이는 스위치(`RERANKER_SCORE_TOP_N`).

왜 필요한가. CE 비용은 `후보 수 × 토큰 수`에 거의 비례한다(2026-09-24 측정:
후보 30개 약 500토큰에 2.7초). 배치 크기로는 줄지 않는 것을 확인했으므로
(`results_v20_ce_batch`), 남은 손잡이 중 하나가 채점할 후보 수다.

`low_conf_top_n`과 다르다 — 그것은 **전부 채점한 뒤** 상위 N개만 재정렬하므로
시간이 줄지 않는다. 이쪽은 아예 채점하지 않는다.

여기서 고정하는 것.

1. 꺼져 있으면(기본 0) 동작이 하나도 달라지지 않는다
2. 채점하지 않은 후보는 **입력 순서 그대로 맨 뒤**에 붙는다
3. 그 곡들은 `judged_ids`에 없다 — 모델이 본 적 없는 곡에 "리랭킹됐다"를 붙이면 안 된다
4. 그 곡들은 합성 점수(`mixes`)도 받지 않는다
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence

from src.backend.schemas.search import MatchingTrack
from src.retrieval.explain import RERANK_APPLIED
from src.retrieval.reranker import MusicReranker, RerankerConfig


class _StubCE(MusicReranker):
    """모델을 올리지 않는다. **무엇을 채점했는지** 기록한다.

    점수는 입력 순서를 뒤집는다 — 재정렬이 실제로 일어났는지 눈에 보이게 하려고.
    """

    def __init__(self, **config_kwargs):
        super().__init__(replace(RerankerConfig(), **config_kwargs))
        self.scored_batches: List[List[str]] = []

    def score(self, query: str, tracks: Sequence[MatchingTrack]) -> List[float]:
        self.scored_batches.append([t.id for t in tracks])
        # 뒤에 있을수록 높은 점수 → 채점된 묶음 안에서 순서가 뒤집힌다
        return [0.1 + 0.05 * i for i in range(len(tracks))]


def _tracks(n: int = 30) -> List[MatchingTrack]:
    return [
        MatchingTrack(id=f"s{i:02d}", score=1.0 - i * 0.01, title=f"곡{i}")
        for i in range(n)
    ]


def test_cap_off_scores_everything():
    ce = _StubCE(score_top_n=0)
    run = ce.rerank_run("질의", _tracks(30), top_k=10)

    assert ce.scored_batches == [[t.id for t in _tracks(30)]]
    assert len(run.judged_ids) == 30
    assert run.status == RERANK_APPLIED


def test_cap_scores_only_the_leading_candidates():
    ce = _StubCE(score_top_n=20)
    ce.rerank_run("질의", _tracks(30), top_k=10)

    assert ce.scored_batches == [[f"s{i:02d}" for i in range(20)]], "앞 20개만 채점해야 한다"


def test_unscored_candidates_are_not_called_judged():
    """모델이 본 적 없는 곡에 '리랭킹됐다'를 붙이면 설명이 거짓이 된다."""
    ce = _StubCE(score_top_n=20)
    run = ce.rerank_run("질의", _tracks(30), top_k=30)

    assert set(run.judged_ids) == {f"s{i:02d}" for i in range(20)}
    for song_id in (f"s{i:02d}" for i in range(20, 30)):
        assert song_id not in run.judged_ids


def test_unscored_candidates_keep_input_order_at_the_back():
    ce = _StubCE(score_top_n=20)
    run = ce.rerank_run("질의", _tracks(30), top_k=30)

    ids = [t.id for t in run.tracks]
    tail = ids[-10:]
    assert tail == [f"s{i:02d}" for i in range(20, 30)], "입력 순서가 아니다"
    # 채점된 곡은 전부 꼬리보다 앞이다
    assert set(ids[:20]) == {f"s{i:02d}" for i in range(20)}


def test_unscored_candidates_get_no_mixed_score():
    ce = _StubCE(score_top_n=20)
    run = ce.rerank_run("질의", _tracks(30), top_k=30)

    for song_id in (f"s{i:02d}" for i in range(20, 30)):
        assert song_id not in run.mixes, f"{song_id}: 채점 안 한 곡에 합성 점수가 붙었다"
    tail = [t for t in run.tracks if t.id >= "s20"]
    assert all(t.rerank_score is None for t in tail)


def test_the_strategy_says_the_cap_was_used():
    """화면이 지어내지 않도록 전략 이름에 남는다."""
    ce = _StubCE(score_top_n=20)
    run = ce.rerank_run("질의", _tracks(30), top_k=10)
    labels = {mix.strategy for mix in run.mixes.values()}
    assert labels and all("앞 20개만 채점" in label for label in labels), labels


def test_a_cap_larger_than_the_pool_changes_nothing():
    ce = _StubCE(score_top_n=100)
    run = ce.rerank_run("질의", _tracks(30), top_k=10)
    assert len(run.judged_ids) == 30
    assert all("만 채점" not in mix.strategy for mix in run.mixes.values())


def test_cap_reads_the_environment(monkeypatch):
    monkeypatch.setenv("RERANKER_SCORE_TOP_N", "20")
    assert RerankerConfig.from_env().score_top_n == 20
    monkeypatch.delenv("RERANKER_SCORE_TOP_N")
    assert RerankerConfig.from_env().score_top_n == 0


# ---------------------------------------------------------------------------
# 검색 점수 정규화는 **받은 후보 전체** 기준이어야 한다
#
# 채점을 앞 N개로 줄였다고 검색 점수의 min-max 범위까지 좁히면 합성의 저울이
# 통째로 달라진다. 그 값은 이미 전부 손에 있고 CE 호출도 더 들지 않는다.
# q201이 이것 때문에 1위 → 2위로 내려갔다.
# ---------------------------------------------------------------------------

def test_retrieval_normalisation_uses_every_candidate():
    """앞 20개만 채점해도 검색 점수는 30개 기준으로 정규화한다."""
    tracks = _tracks(30)
    ce = _StubCE(score_top_n=20)
    run = ce.rerank_run("질의", tracks, top_k=30)

    scores = [t.score for t in tracks]
    low, high = min(scores), max(scores)
    for track in tracks[:20]:
        expected = (track.score - low) / (high - low)
        got = run.mixes[track.id].retrieval_component
        assert abs(got - expected) < 1e-9, (
            f"{track.id}: 검색 정규화가 {got}, 30곡 기준이면 {expected}"
        )


def test_capping_does_not_stretch_the_retrieval_scale():
    """20곡 기준으로 정규화하면 20번째 곡이 0이 된다 — 그러면 안 된다."""
    ce = _StubCE(score_top_n=20)
    run = ce.rerank_run("질의", _tracks(30), top_k=30)
    last_judged = run.mixes["s19"].retrieval_component
    assert last_judged > 0.0, "채점 범위 끝 곡이 0으로 눌렸다(20곡 기준 정규화)"


def test_uncapped_normalisation_is_unchanged():
    """캡이 꺼져 있으면 예전과 완전히 같아야 한다."""
    tracks = _tracks(30)
    full = _StubCE(score_top_n=0).rerank_run("질의", tracks, top_k=30)
    scores = [t.score for t in tracks]
    low, high = min(scores), max(scores)
    for track in tracks:
        expected = (track.score - low) / (high - low)
        assert abs(full.mixes[track.id].retrieval_component - expected) < 1e-9
