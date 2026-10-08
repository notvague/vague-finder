"""단계 3 — 재질문 정책 측정 하네스.

벡터 DB·Gemini 없이 하네스의 판단 로직만 고정한다. 실제 검색 품질이 아니라
"정책을 올바르게 시뮬레이션하는가"를 본다.

여기서 고정하는 것
1. 정답이 이미 Top-10에 있으면 개입하지 않는다 — 검색을 더 돌리지 않는다.
2. 거절한 곡은 다음 턴에 다시 나오지 않는다.
3. "잘 모르겠어요"는 Reject-only와 같은 결과여야 한다. 다르면 상태가 새는 것이다.
4. oracle:best는 천장이다 — 어떤 슬롯 답변보다 나쁠 수 없다.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval.evaluate_clarification import (
    POLICY_ORDER,
    add_oracle_best,
    evaluate_query,
    fill_missing_policies,
    noisy_value,
    oracle_value,
    summarize,
    as_answer,
)

TARGET = "s030"          # 정답. 기본 풀에서 31번째라 Top-10 밖이다.
POOL = [f"s{i:03d}" for i in range(50)]


def _analysis(query: str = "비 오는 날 발라드") -> QueryAnalysis:
    return QueryAnalysis(
        original_query=query,
        intent_type="mood",
        image_english_query="",
        audio_english_query="",
    )


def _song(gender: str = "여성", types=("듀오",), genre="발라드", date="2013.03.04") -> dict:
    return {
        "song_id": TARGET,
        "metadata": {
            "title": "정답곡", "artist": ["가수"], "album": "앨범",
            "release_date": date, "genre": [genre], "type": list(types),
            "vocal_gender": gender,
        },
    }


class _FakeRouter:
    """분석 내용에 따라 정답의 위치를 바꾸는 가짜 검색기.

    · vocal_gender=여성  → 정답을 맨 앞으로 (정확한 답변이 도움이 되는 상황)
    · vocal_gender=남성  → 정답을 풀에서 제거 (틀린 답변이 밀어내는 상황)
    그 외 슬롯은 순위에 영향을 주지 않는다.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def search(self, analysis, **kwargs) -> List[MatchingTrack]:
        self.calls.append({"analysis": analysis, **kwargs})
        excluded = set(kwargs.get("exclude_ids") or [])
        pool = [sid for sid in POOL if sid not in excluded]

        # 실제 라우터와 같은 자리에서 반응한다 — 답변은 분석이 아니라 answers로
        # 오고, 후보 풀을 재검색하지 않고 순위만 바꾼다.
        gender = analysis.vocal_gender
        for answer in (kwargs.get("answers") or []):
            if answer.slot == "vocal_gender" and not answer.skipped and answer.value:
                gender = answer.value

        if gender == "여성":
            pool = [TARGET] + [sid for sid in pool if sid != TARGET]
        elif gender == "남성":
            pool = [sid for sid in pool if sid != TARGET]

        candidate_k = kwargs.get("candidate_k") or 30
        pool = pool[:candidate_k]

        out = kwargs.get("candidate_ids_out")
        if out is not None:
            out.extend(pool)
        return [
            MatchingTrack(id=sid, score=1.0 - i * 0.001, title=f"곡{sid}")
            for i, sid in enumerate(pool)
        ]


class _FakeReranker:
    """순서를 바꾸지 않는 리랭커. 하네스 로직만 보기 위한 것."""

    def rerank(self, query, candidates, top_k):
        return list(candidates)[:top_k]


def _turn(rank):
    from src.retrieval.evaluate_clarification import TurnResult
    return TurnResult(rank=rank, candidate_rank=rank, candidate_recall=1.0)


def _run(router, analysis, song, top_k=10, candidate_k=30):
    results, _slot = asyncio.run(
        evaluate_query(
            router, _FakeReranker(), analysis, song, {TARGET},
            top_k=top_k, candidate_k=candidate_k,
        )
    )
    return results


def _run_with_slot(router, analysis, song, top_k=10, candidate_k=30):
    return asyncio.run(
        evaluate_query(
            router, _FakeReranker(), analysis, song, {TARGET},
            top_k=top_k, candidate_k=candidate_k,
        )
    )


# ---------------------------------------------------------------------------
# 슬롯 값 추출
# ---------------------------------------------------------------------------

def test_oracle_values_come_from_the_target_song() -> None:
    song = _song(gender="여성", types=("듀오", "여성"), genre="발라드", date="2013.03.04")
    assert oracle_value(song, "vocal_gender") == "여성"
    assert oracle_value(song, "type") == "듀오"
    assert oracle_value(song, "genre") == "발라드"
    assert oracle_value(song, "release_era") == "2010년대"


def test_free_form_types_use_the_matchers_normaliser() -> None:
    """평가기는 판정기(clarify.canonical_artist_types)와 같은 기준으로 type을 읽는다.

    회귀 이력: 평가기가 자체 표를 들고 있어 ['혼성', '듀오']를 '그룹'으로,
    판정기는 '듀오'로 읽었다. oracle 답변이 정답 곡에 보너스를 주지 못했다.
    """
    assert oracle_value(_song(types=("혼성", "듀오")), "type") == "듀오"
    assert oracle_value(_song(types=("혼성그룹",)), "type") == "그룹"
    assert oracle_value(_song(types=("듀엣",)), "type") == "듀오"
    # 판정기가 유형으로 읽지 못하는 값은 답변할 수 없다고 본다.
    assert oracle_value(_song(types=("혼성",)), "type") is None


def test_oracle_answer_matches_its_own_target_and_noisy_does_not() -> None:
    """oracle 답변은 정답 곡에 반드시 일치하고, noisy 답변은 절대 일치하지 않아야 한다.

    이게 깨지면 oracle 정책이 '정답을 말했는데 보너스가 없는' 상태가 되어
    슬롯별 효과와 oracle:best가 전부 과소평가된다. 실제로 c608·c707(둘 다
    ['혼성', '듀오'])에서 그랬다.
    """
    from src.backend.schemas.search import ClarifyAnswer
    from src.retrieval.clarify import answer_matches

    def track_from(song):
        m = song["metadata"]
        return MatchingTrack(
            id=song["song_id"], score=1.0, title=m["title"],
            vocal_gender=m.get("vocal_gender"),
            genre=", ".join(m.get("genre") or []),
            artist_types=list(m.get("type") or []),
            release_date=m.get("release_date"),
        )

    cases = [
        _song(gender="혼성", types=("혼성", "듀오"), genre="발라드", date="2016.02.24"),
        _song(gender="여성", types=("솔로",), genre="댄스", date="1999.10.01"),
        _song(gender="남성", types=("밴드",), genre="록/메탈", date="2004.06.01"),
        _song(gender="혼성", types=("혼성그룹", "듀엣"), genre="랩/힙합", date="2013.01.01"),
    ]
    for song in cases:
        track = track_from(song)
        for slot in ("vocal_gender", "genre", "type", "release_era"):
            truth, wrong = oracle_value(song, slot), noisy_value(song, slot)
            if truth is not None:
                assert answer_matches(track, ClarifyAnswer(slot=slot, value=truth)), (slot, truth)
            if wrong is not None:
                assert not answer_matches(track, ClarifyAnswer(slot=slot, value=wrong)), (slot, wrong)


def test_oracle_answers_match_every_real_eval_target() -> None:
    """평가 세트의 실제 정답 곡 전부에 대해 같은 성질을 확인한다 (코퍼스가 있을 때)."""
    import json
    from pathlib import Path
    from src.backend.schemas.search import ClarifyAnswer
    from src.retrieval.clarify import answer_matches

    corpus = Path("data/all_songs.jsonl")
    if not corpus.exists():
        pytest.skip("data/는 git 추적 대상이 아니다")
    songs = {}
    for line in corpus.open(encoding="utf-8"):
        if line.strip():
            song = json.loads(line)
            songs[song["song_id"]] = song
    queries = json.loads(Path("docs/eval/queries.json").read_text(encoding="utf-8"))["queries"]
    targets = {q["positives"][0] for q in queries if q.get("positives")}

    failures = []
    for sid in sorted(targets):
        song = songs[sid]; m = song["metadata"]
        track = MatchingTrack(id=sid, score=1.0, title=m["title"],
                              vocal_gender=m.get("vocal_gender"),
                              genre=", ".join(m.get("genre") or []),
                              artist_types=list(m.get("type") or []),
                              release_date=m.get("release_date"))
        for slot in ("vocal_gender", "genre", "type", "release_era"):
            truth, wrong = oracle_value(song, slot), noisy_value(song, slot)
            if truth and not answer_matches(track, ClarifyAnswer(slot=slot, value=truth)):
                failures.append((sid, slot, "oracle 불일치", truth))
            if wrong and answer_matches(track, ClarifyAnswer(slot=slot, value=wrong)):
                failures.append((sid, slot, "noisy 일치", wrong))
    assert not failures, failures[:5]


def test_noisy_values_differ_from_the_truth() -> None:
    song = _song(gender="여성", types=("듀오",), genre="발라드", date="2013.03.04")
    for slot in ("vocal_gender", "type", "genre", "release_era"):
        assert noisy_value(song, slot) != oracle_value(song, slot)


def test_noisy_values_are_deterministic() -> None:
    """무작위면 재측정 때 값이 달라져 정책 비교가 깨진다."""
    song = _song()
    assert noisy_value(song, "vocal_gender") == noisy_value(song, "vocal_gender")
    assert noisy_value(song, "release_era") == "2000년대"


def test_oldest_era_shifts_forward() -> None:
    """1990년대보다 앞은 코퍼스에 없다. 뒤로 밀어야 답변이 의미를 갖는다."""
    assert noisy_value(_song(date="1999.10.01"), "release_era") == "2000년대"


# ---------------------------------------------------------------------------
# 개입 판정
# ---------------------------------------------------------------------------

def test_no_intervention_when_answer_is_already_shown() -> None:
    """정답이 Top-10에 있으면 사용자는 '이 중에는 없어요'를 누르지 않는다."""
    router = _FakeRouter()
    analysis = _analysis()
    analysis.vocal_gender = "여성"          # 정답이 1위로 올라온다
    results = _run(router, analysis, _song())

    assert set(results) == {"initial"}
    assert results["initial"].rank == 1
    assert len(router.calls) == 1, "개입이 없는데 검색을 더 돌렸다"


def test_intervention_runs_all_policies() -> None:
    router = _FakeRouter()
    results = _run(router, _analysis(), _song())

    assert results["initial"].rank is None
    assert "reject_only" in results and "reject_twice" in results and "skip" in results
    assert any(k.startswith("oracle:") for k in results)
    assert any(k.startswith("noisy:") for k in results)


def test_rejected_songs_do_not_come_back() -> None:
    router = _FakeRouter()
    results = _run(router, _analysis(), _song())

    rejected = set(results["initial"].shown_ids)
    assert rejected, "1턴에서 보여준 곡이 없다"
    assert not rejected & set(results["reject_only"].shown_ids)
    assert not rejected & set(results["reject_twice"].shown_ids)


def test_two_turns_accumulate_rejections() -> None:
    router = _FakeRouter()
    results = _run(router, _analysis(), _song())

    shown_1 = set(results["initial"].shown_ids)
    shown_2 = set(results["reject_only"].shown_ids)
    assert not (shown_1 | shown_2) & set(results["reject_twice"].shown_ids)


def test_reject_only_promotes_the_next_ranks() -> None:
    """Reject-only의 전부 — 거절한 만큼 뒤에 있던 곡이 올라온다."""
    router = _FakeRouter()
    results = _run(router, _analysis(), _song())
    assert results["reject_only"].shown_ids[0] == "s010"
    assert results["reject_twice"].shown_ids[0] == "s020"


# ---------------------------------------------------------------------------
# 답변 정책
# ---------------------------------------------------------------------------

def test_skip_equals_reject_only() -> None:
    """'잘 모르겠어요'는 슬롯을 채우지 않으므로 Reject-only와 같아야 한다.

    다르면 왕복 어딘가에서 분석 상태가 새고 있다는 뜻이다.
    """
    router = _FakeRouter()
    results = _run(router, _analysis(), _song())
    assert results["skip"].shown_ids == results["reject_only"].shown_ids
    assert results["skip"].rank == results["reject_only"].rank


def test_correct_answer_can_recover_the_target() -> None:
    router = _FakeRouter()
    results = _run(router, _analysis(), _song(gender="여성"))
    assert results["oracle:vocal_gender"].rank == 1


def test_wrong_answer_can_push_the_target_out_of_the_pool() -> None:
    """틀린 답변이 정답을 후보 풀에서 밀어내는 상황을 하네스가 포착해야 한다.

    이 수치("오답 시 후보@30 유지율")가 재질문 기능의 안전성 근거다.
    """
    router = _FakeRouter()
    results = _run(router, _analysis(), _song(gender="여성"))
    noisy = results["noisy:vocal_gender"]
    assert noisy.candidate_recall == 0.0
    assert noisy.rank is None


def test_answers_never_touch_the_analysis() -> None:
    """답변은 분석을 바꾸지 않는다 — 재검색이 아니라 후보 재정렬로만 반영된다.

    분석에 병합하면 analysis.genre가 보조 검색 경로의 sparse 질의로 흘러가
    맞는 답변도 순위를 해친다(q200: 후보 4위 → 15위).
    """
    router = _MetaRouter()
    analysis = _analysis()
    before = analysis.model_dump()
    _run(router, analysis, _song(gender="여성"))

    assert analysis.model_dump() == before
    # 라우터에 넘어간 분석도 원본 그대로여야 한다.
    for call in router.calls:
        assert call["analysis"].vocal_gender == before["vocal_gender"]
    # 답변은 별도 인자로 전달된다.
    assert any(call.get("answers") for call in router.calls)


def test_as_answer_builds_a_plain_answer() -> None:
    answer = as_answer("vocal_gender", "여성")
    assert answer.slot == "vocal_gender" and answer.value == "여성"
    assert not answer.skipped


def test_oracle_best_is_the_ceiling() -> None:
    """어떤 슬롯 답변도 oracle:best보다 나을 수 없다 — 질문 선택의 천장이다."""
    router = _FakeRouter()
    results = _run(router, _analysis(), _song(gender="여성"))
    add_oracle_best(results)

    best = results["oracle:best"].rank or 10**6
    for key, turn in results.items():
        if key.startswith("oracle:") and key != "oracle:best":
            assert best <= (turn.rank or 10**6)


# ---------------------------------------------------------------------------
# 규칙 기반 질문 선택 정책
# ---------------------------------------------------------------------------

class _MetaRouter(_FakeRouter):
    """후보에 성별·장르를 실어 주는 라우터. pick_question이 고를 수 있게 한다."""

    async def search(self, analysis, **kwargs):
        tracks = await super().search(analysis, **kwargs)
        enriched = [
            t.model_copy(update={
                "vocal_gender": "남성" if i % 2 else "여성",
                "genre": "발라드",
            })
            for i, t in enumerate(tracks)
        ]
        out = kwargs.get("candidate_tracks_out")
        if out is not None:
            out.clear()
            out.extend(enriched)
        return enriched


def test_rule_falls_back_to_reject_only_when_nothing_is_worth_asking() -> None:
    """후보에 메타데이터가 없으면 물을 게 없다 — 설계대로 Reject-only가 된다."""
    router = _FakeRouter()
    results, picked = _run_with_slot(router, _analysis(), _song())

    assert picked is None
    for mode in ("oracle", "noisy", "skip"):
        assert results[f"rule:{mode}"].rank == results["reject_only"].rank


def test_rule_asks_the_slot_the_selector_picked() -> None:
    """성별이 고르게 갈리면 성별을 묻고, 그 답변으로 재검색한다."""
    router = _MetaRouter()
    results, picked = _run_with_slot(router, _analysis(), _song(gender="여성"))

    assert picked == "vocal_gender"
    # 오라클 답변("여성")이 반영되면 가짜 라우터가 정답을 1위로 올린다.
    assert results["rule:oracle"].rank == 1


def test_rule_skip_matches_plain_skip() -> None:
    """질문을 던져도 '잘 모르겠어요'면 슬롯이 안 채워져 Reject-only와 같다."""
    router = _MetaRouter()
    results, _ = _run_with_slot(router, _analysis(), _song())
    assert results["rule:skip"].shown_ids == results["reject_only"].shown_ids


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

def test_summary_separates_full_and_intervention_sets() -> None:
    rows = [
        {"query_id": "a", "policy": "reject_only", "ran": "0", "hit1": 1.0, "hit5": 1.0,
         "hit10": 1.0, "mrr10": 1.0, "candidate_recall": 1.0},
        {"query_id": "b", "policy": "reject_only", "ran": "1", "hit1": 0.0, "hit5": 0.0,
         "hit10": 0.0, "mrr10": 0.0, "candidate_recall": 0.0},
    ]
    summary = summarize(rows, ["reject_only"], intervention_ids={"b"})[0]

    assert summary["n_full"] == 2
    assert summary["n_intervention"] == 1
    assert summary["full_hit@10"] == 0.5
    # 개입 집합만 보면 b 하나뿐이라 0이다 — 전체 지표에 희석되면 안 된다.
    assert summary["int_hit@10"] == 0.0
    assert summary["int_candidate_recall@30"] == 0.0
    # 실제로 개입한 것은 b 하나뿐 — a는 채워 넣은 값이다.
    assert summary["n_executed"] == 1


# ---------------------------------------------------------------------------
# 분모 — 정책마다 다르면 비교가 성립하지 않는다
# ---------------------------------------------------------------------------

def test_unrun_policies_fall_back_to_what_the_user_would_have_seen() -> None:
    """실행되지 않은 정책도 모든 질의에 결과가 있어야 한다.

    회귀 이력: reject_twice는 1턴에서 정답을 찾으면 실행되지 않는다. 그 행을
    비워 두면 그 정책만 분모가 작아져 평균이 부풀려진다(스모크에서 reject_twice의
    분모가 혼자 1이 됐다). 2턴이 일어나지 않았다면 사용자가 본 것은 1턴 결과다.
    """
    router = _FakeRouter()
    analysis = _analysis()
    analysis.vocal_gender = "여성"          # 정답이 1위 — 개입 없음
    results = _run(router, analysis, _song())
    executed = fill_missing_policies(results, POLICY_ORDER)

    assert set(results) >= set(POLICY_ORDER)
    assert executed == {"initial"}
    # 개입이 없었으므로 모든 정책의 결과가 최초 검색과 같다.
    for policy in POLICY_ORDER:
        assert results[policy].rank == results["initial"].rank


def test_fallback_after_first_turn_success_is_the_first_turn() -> None:
    """1턴에서 찾으면 2턴은 일어나지 않는다 — reject_only 결과가 곧 그 정책의 결과다."""
    results = {
        "initial": _turn(None),
        "reject_only": _turn(3),
    }
    fill_missing_policies(results, ["initial", "reject_only", "reject_twice"])
    assert results["reject_twice"].rank == 3


# ---------------------------------------------------------------------------
# 서비스 2턴 흐름 (flow:*) — routes/search.py · map.js와 같은 규칙인지
# ---------------------------------------------------------------------------

from src.backend.schemas.search import ClarifyOption, ClarifyQuestion  # noqa: E402
from src.retrieval.evaluate_clarification import (  # noqa: E402
    FLOW_POLICIES,
    TurnResult,
    check_against_labels,
    choose_answer,
    next_question,
    run_flow,
)


def _target(gender: str = "여성") -> MatchingTrack:
    return MatchingTrack(id=TARGET, score=0.0, title="정답곡", vocal_gender=gender, genre="발라드")


def _question(*values: str) -> ClarifyQuestion:
    return ClarifyQuestion(
        slot="vocal_gender", question="?",
        options=[ClarifyOption(value=v, count=5) for v in values],
    )


def test_oracle_picks_the_option_that_matches_the_target() -> None:
    answer, kind = choose_answer(_question("남성", "여성"), "oracle", _target("여성"), _song("여성"))
    assert (answer.value, kind) == ("여성", "oracle")


def test_oracle_cannot_pick_a_value_the_screen_does_not_offer() -> None:
    """화면에 없는 값은 고를 수 없다 — 정확히 기억해도 '잘 모르겠어요'가 된다."""
    answer, kind = choose_answer(_question("남성", "여성"), "oracle", _target("혼성"), _song("혼성"))
    assert answer.skipped and kind == "skip"


def test_noisy_picks_an_offered_wrong_option() -> None:
    answer, kind = choose_answer(_question("여성", "남성", "혼성"), "noisy", _target("여성"), _song("여성"))
    assert (answer.value, kind) == ("남성", "noisy")   # 하네스의 인접 오답이 선택지에 있으면 그것


def test_no_question_after_the_last_turn_or_full_rejections() -> None:
    result = TurnResult(rank=None, candidate_rank=None, candidate_recall=0.0)
    assert next_question(_analysis(), result, 3, [], [], 10) is None          # 3턴 응답에는 질문 없음
    assert next_question(_analysis(), result, 2, ["vocal_gender", "genre"], [], 10) is None
    assert next_question(_analysis(), result, 2, [], [f"x{i}" for i in range(11)], 10) is None


def _flow(router, mode, gender="여성"):
    results, _slot = _run_with_slot(router, _analysis(), _song(gender))
    initial = results["initial"]
    return asyncio.run(run_flow(
        router, _FakeReranker(), _analysis(), {TARGET}, initial, mode,
        _target(gender), _song(gender), top_k=10, candidate_k=30,
    ))


def test_flow_stops_when_the_target_is_shown() -> None:
    steps = _flow(_MetaRouter(), "oracle")
    assert [s.turn for s in steps] == [1, 2]
    assert steps[1].answer_kind == "oracle" and steps[1].answer_value == "여성"
    assert steps[-1].result.found


def test_flow_accumulates_answers_and_rejections_up_to_the_limit() -> None:
    router = _MetaRouter()
    steps = _flow(router, "noisy")
    assert [s.turn for s in steps] == [1, 2, 3]          # 최대 두 번 다시 찾는다
    assert [s.rejected for s in steps] == [0, 10, 20]    # 보여 준 곡이 누적된다
    assert steps[1].answer_kind == "noisy"
    # 성별은 이미 물었고 장르는 후보를 가르지 못한다 — 2턴에는 질문 없이 거절만 한다
    assert steps[2].answer_kind == "none"
    last = router.calls[-1]
    assert len(last["exclude_ids"]) == 20
    assert [a.value for a in last["answers"]] == ["남성"]   # 앞 턴의 답변을 계속 싣는다


def test_reject_flow_never_answers() -> None:
    steps = _flow(_MetaRouter(), "reject")
    assert [s.answer_kind for s in steps[1:]] == ["none", "none"]


def test_flow_policies_are_reported() -> None:
    assert set(FLOW_POLICIES) <= set(POLICY_ORDER)


def test_label_check_stops_on_any_mismatch(tmp_path) -> None:
    from src.eval.schema import EvalQuery

    q = EvalQuery(query_id="x1", query="질의", category="mixed", split="dev",
                  query_set="v04", positives=["a"])
    path = tmp_path / "labels.csv"
    header = "query_id,split,query_type,query,relevant_ids\n"
    path.write_text(header + "x1,dev,search,질의,a\n", encoding="utf-8")
    check_against_labels([q], path, whole_split="dev")

    path.write_text(header + "x1,dev,search,질의,b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        check_against_labels([q], path)

    path.write_text(header + "x1,dev,search,질의,a\nx2,dev,search,다른 질의,c\n", encoding="utf-8")
    with pytest.raises(ValueError):
        check_against_labels([q], path, whole_split="dev")


# ---------------------------------------------------------------------------
# 목표 곡 — 답변·종료·성공을 같은 곡으로 판정한다 (복수 정답 질의)
# ---------------------------------------------------------------------------

OTHER = "s045"   # 같은 질의의 다른 정답. 기본 풀에서 46번째라 처음엔 후보 밖이다


class _OtherAnswerRouter(_MetaRouter):
    """'여성'이라고 답하면 다른 정답(OTHER)이 1위로 온다. 목표 곡(TARGET)은 그대로 둔다."""

    async def search(self, analysis, **kwargs):
        answers = kwargs.get("answers") or []
        female = any(a.slot == "vocal_gender" and a.value == "여성" for a in answers)
        tracks = await super().search(analysis, **{**kwargs, "answers": []})
        if female:
            other = MatchingTrack(id=OTHER, score=2.0, title="다른 정답", vocal_gender="여성", genre="발라드")
            tracks = [other] + [t for t in tracks if t.id not in (OTHER, TARGET)]
            out = kwargs.get("candidate_tracks_out")
            if out is not None:
                out.clear()
                out.extend(tracks)
        return tracks


def test_another_answer_song_is_not_the_conversation_target() -> None:
    """m303 회귀: 목표 곡은 남성인데 '여성'(틀린 답)을 고르자 다른 정답(여성)이 1위에 떴다.
    목표 곡으로 판정하면 실패, 정답 아무 곡으로 판정하면 성공 — 예전 평가기는 성공으로 셌다."""
    def run(goal):
        router = _OtherAnswerRouter()
        results, _ = _run_with_slot(router, _analysis(), _song("남성"))
        return asyncio.run(run_flow(
            router, _FakeReranker(), _analysis(), goal, results["initial"], "noisy",
            _target("남성"), _song("남성"), top_k=10, candidate_k=30,
        ))

    assert run({TARGET})[1].answer_value == "여성"
    assert not any(s.result.found for s in run({TARGET}))
    assert any(s.result.found for s in run({TARGET, OTHER}))


def test_flow_summary_counts_queries_not_conversations() -> None:
    def row(qid, target, hit, seen):
        return {"query_id": qid, "policy": "flow:oracle", "target_id": target, "ran": "1",
                "hit1": hit, "hit5": hit, "hit10": hit, "mrr10": hit, "candidate_recall": 1.0,
                "found_turn": 2 if hit else "", "final_relaxed_hit10": hit,
                "relaxed_seen_turn": 2 if seen else ""}
    rows = [row("a", "a1", 1.0, True), row("a", "a2", 0.0, True), row("b", "b1", 1.0, True)]
    summary = summarize(rows, ["flow:oracle"], intervention_ids={"a", "b"})[0]
    assert summary["n_intervention"] == 2 and summary["n_conversations"] == 3
    assert summary["int_hit@10"] == 0.75          # a는 두 대화의 평균 0.5, b는 1
    assert summary["int_final_relaxed"] == 0.75   # 마지막 화면 기준
    assert summary["int_relaxed_seen"] == 1.0     # 대화 중 한 번이라도 보였다


from src.retrieval.evaluate_clarification import (  # noqa: E402
    intervention_ids_from_detail,
    summarize_by_scope,
)


def _flow_row(qid, target, found_turn, cand="", policy="flow:oracle"):
    hit = 1.0 if found_turn else 0.0
    return {"query_id": qid, "policy": policy, "target_id": target, "ran": "1",
            "hit1": hit, "hit5": hit, "hit10": hit, "mrr10": hit, "candidate_recall": hit,
            "candidate_rank@30": cand, "found_turn": found_turn, "final_relaxed_hit10": hit,
            "relaxed_seen_turn": found_turn}


def test_flow_summary_counts_conversations_left_outside_the_candidates() -> None:
    """찾지 못했고 마지막 검색에서도 후보 30 밖인 대화 — 답변이 닿지 않는 대화 수다."""
    rows = [_flow_row("a", "a1", 2, cand=3), _flow_row("a", "a2", "", cand=""),
            _flow_row("b", "b1", "", cand=12),   # 후보 안까지만 (c603 같은 경우)
            _flow_row("c", "c1", "", cand="")]
    summary = summarize(rows, ["flow:oracle"], intervention_ids={"a", "b", "c"})[0]
    assert summary["n_conv_out_of_candidates"] == 2


def test_summary_reads_saved_detail_the_same_way() -> None:
    """--resummarize는 CSV 문자열을 읽는다 — found_turn이 "2"여도 2턴 안으로 센다."""
    rows = [_flow_row("a", "a1", 2, cand=3), _flow_row("b", "b1", "", cand="")]
    saved = [{k: str(v) for k, v in r.items()} for r in rows]
    assert summarize(saved, ["flow:oracle"], {"a", "b"}) == summarize(rows, ["flow:oracle"], {"a", "b"})


def test_scope_summary_splits_categorical_queries_and_keeps_the_whole() -> None:
    rows = [_flow_row("s1", "s1a", 1, cand=1), _flow_row("s2", "s2a", "", cand=""),
            _flow_row("k1", "k1a", "", cand=""), _flow_row("k1", "k1b", 3, cand=2)]
    scopes = {"s1": "specific", "s2": "specific", "k1": "categorical"}
    out = summarize_by_scope(rows, ["flow:oracle"], {"s1", "s2", "k1"}, scopes)
    by = {r["scope"]: r for r in out}
    assert {k: v for k, v in by["all"].items() if k != "scope"} == \
        summarize(rows, ["flow:oracle"], {"s1", "s2", "k1"})[0]
    assert (by["specific"]["n_intervention"], by["specific"]["int_hit@10"]) == (2, 0.5)
    assert (by["categorical"]["n_conversations"], by["categorical"]["int_hit@10"]) == (2, 0.5)
    assert by["categorical"]["n_conv_out_of_candidates"] == 1


def test_scope_summary_stops_on_a_query_missing_from_the_label_file() -> None:
    with pytest.raises(KeyError):
        summarize_by_scope([_flow_row("zz", "z1", 1)], ["flow:oracle"], {"zz"}, {"a": "specific"})


def test_intervention_set_is_recovered_from_saved_detail() -> None:
    """개입 대상 = reject_only를 실제로 실행한 질의. 실행 안 한 행은 최초 검색을 채운 것이다."""
    rows = [{"query_id": "a", "policy": "reject_only", "ran": "1"},
            {"query_id": "b", "policy": "reject_only", "ran": "0"},
            {"query_id": "c", "policy": "initial", "ran": "1"}]
    assert intervention_ids_from_detail(rows) == {"a"}
