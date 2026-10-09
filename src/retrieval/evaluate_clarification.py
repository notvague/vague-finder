#!/usr/bin/env python
"""
재질문 정책 측정 하네스 (단계 3).

"거절하고 다시 물으면 정말 나아지는가"를 정책별로 재는 도구다.
비교 기준은 언제나 **Reject-only**다 — 질문을 던지는 정책은 이것보다 나아야
의미가 있다. 질문 없이 거절만으로 얻는 이득을 질문의 공으로 돌리지 않기 위해서다.

측정하는 정책

    initial            최초 검색. 개입 없음.
    reject_only        보여준 Top-10을 거절하고 재검색. **필수 비교 대상**.
    reject_twice       2턴 누적 거절 (최대 20곡 제외).
    skip               거절 + "잘 모르겠어요". 정의상 reject_only와 같아야 한다.
    oracle:<slot>      거절 + 해당 슬롯에 정답 곡의 실제 값으로 답변.
    noisy:<slot>       거절 + 틀린 값으로 답변.

Oracle과 Noisy를 반드시 나눠 본다. Oracle만 제시하면 과대평가다 — 사람은
자기 기억을 틀리게 답한다.

위 정책은 **거절 한 번 뒤 답변 한 번**의 효과만 잰다(reject_twice는 답변 없이 거절만 두 번).
서비스의 2턴 흐름은 아래 flow:*로 따로 잰다.

    flow:<mode>        서비스의 재질문을 그대로 밟는다 — 거절 → 질문 → 답변을 최대 두 번.
                       질문 선택·누적 답변·거절 목록·물은 슬롯·턴 제한을 서비스와 같은 규칙으로
                       쌓는다(`routes/search.py`의 `_can_ask_another`, 화면 `map.js`의 submitTurn).
                       답은 **화면이 내준 선택지 안에서만** 고른다.
                         oracle  정답 곡과 맞는 선택지 (없으면 "잘 모르겠어요")
                         noisy   정답 곡과 맞지 않는 선택지 (없으면 "잘 모르겠어요")
                         skip    매번 "잘 모르겠어요"
                         reject  질문을 쓰지 않고 거절만 (= reject_only → reject_twice)
                       턴마다 정답 순위와 후보 ID를 clarify_turns.csv에 남긴다.

정답 기준
    주 지표는 원래 타깃(positives)이다. 개입 여부도 원래 타깃이 보였는지로 정한다.
    flow:*의 사용자는 곡 하나를 찾는다 — 정답이 여럿이면 **목표 곡마다 따로 대화**하고 답변·종료·
    성공을 모두 그 곡으로 판정한다(다른 정답이 떠도 목표 곡이 아니면 계속 거절한다). 집계는
    질의 안에서 먼저 평균한다. 한 번 답변 정책(rule:*·oracle:*·noisy:*)은 v07과 같은 정의로 남겼다 —
    답변은 첫 정답 기준인데 성공은 정답 아무 곡이나 보이면 인정하므로 복수 정답 질의에서는 어긋난다.
    허용 정답(allowed)을 넣은 열은 참고로 따로 낸다. final_relaxed_*는 마지막 화면 기준, relaxed_seen_turn은
    대화 중 한 번이라도 보인 턴이다. 허용 정답이 이미 보였다면 사용자는 거절하지 않았을 수 있고, 질문에 대한
    맞는 답도 곡마다 다르다.

분석 캐시
    --analysis-cache를 주면 질의 분석을 새로 하지 않는다(evaluate_search_accuracy와 같은 캐시).
    캐시 누락·원문 불일치·폴백 분석은 **모델과 DB를 올리기 전에** 전부 확인하고 멈춘다.
    --labels(기본 eval_queries_v06.csv)와 질의 ID·원문·정답·split이 하나라도 다르면 시작하지 않는다
    — 이 평가기는 docs/eval/queries.json을 읽으므로 측정 기준 세트와 어긋나지 않는지 본다.

가장 중요한 수치는 **오답 응답 시 정답의 후보@30 유지율**이다.
"답변은 필터가 아니라 부스팅이라 안전하다"는 주장은 이 수치로만 증명된다.
유지율이 낮으면 틀린 답변 한 번에 정답이 후보 풀에서 사라진다는 뜻이고,
그러면 재질문은 사용자를 돕는 게 아니라 해치는 기능이 된다.

개입 대상만 잰다
    정답이 이미 Top-10에 있으면 사용자는 "이 중에는 없어요"를 누르지 않는다.
    그런 질의는 모든 정책이 동일하므로 최초 검색 결과를 그대로 쓴다.
    전체 집합 지표와 개입 집합 지표를 함께 보고한다 — 전자는 서비스 전체
    효과, 후자는 정책 간 우열을 본다.

  python -m src.retrieval.evaluate_clarification --split dev \\
      --analysis-cache experiments/reranking/analysis_cache_v06_dev.json \\
      --output-dir experiments/reranking/results_clarify_vNN
  python -m src.retrieval.evaluate_clarification --split dev --limit 3   # 스모크
  python -m src.retrieval.evaluate_clarification --resummarize \\
      --output-dir experiments/reranking/results_clarify_v08          # 요약만 다시 (모델·DB 없음)

범주형 집계
    docs/eval/queries.json에서 target_scope=categorical인 질의(질의가 일반 속성만 말해 원래 타깃을
    특정할 수 없는 질의)를 나눠 본 요약을 clarify_summary_by_scope.csv로 함께 낸다. 전체 지표는
    바꾸지 않는다 — 범주형 질의도 전체 집계(clarify_summary.csv)에 그대로 들어간다.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, get_args

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.api.dependencies import (
    get_query_analyzer,
    get_reranker,
    get_search_router,
    get_vector_client,
)
from src.backend.api.routes.search import _can_ask_another
from src.retrieval.clarify import (
    analysis_with_answers,
    answer_matches,
    canonical_artist_types,
    pick_question,
)
from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import (
    MAX_REJECTED_IDS,
    ClarifyAnswer,
    ClarifyQuestion,
    MatchingTrack,
)
from src.eval.loader import DEFAULT_EVAL_PATH, load_eval_set, load_target_scopes
from src.eval.schema import TARGET_SCOPES, EvalQuery, V05_QUERY_SETS, QuerySet
from src.retrieval.analysis_cache import (
    AnalysisCacheError,
    analyzer_fingerprint,
    load_cache,
)
from src.retrieval.evaluate_search_accuracy import (
    collect_run_info,
    first_relevant_rank,
    mean,
    metric_bundle,
    recall_at_k,
    rerank_preserving_exact_lyrics,
)

DEFAULT_CORPUS = Path("data/all_songs.jsonl")
DEFAULT_LABELS = Path("experiments/reranking/eval_queries_v06.csv")
DEFAULT_OUTPUT_DIR = Path("experiments/reranking/results_clarify_v01")

# 재질문에 쓸 슬롯. 계획서의 4슬롯과 같다. 실제로 묻는 것은 clarify.ALLOWED_SLOTS의
# 둘뿐이지만, 여기서는 슬롯별 효과를 비교해야 하므로 넷을 모두 잰다.
SLOTS: Sequence[str] = ("vocal_gender", "type", "genre", "release_era")

# 틀린 답변을 만들 때 쓰는 대체값. 사용자가 헷갈릴 법한 인접 값으로 고른다.
# 유형 답변을 고를 때의 결정적 순서. 무작위면 재측정 간 비교가 깨진다.
_TYPE_ORDER = ("솔로", "그룹", "듀오", "밴드")

_NOISY_GENDER = {"남성": "여성", "여성": "남성", "혼성": "남성"}
_NOISY_GENRE = {"발라드": "댄스", "댄스": "발라드", "랩/힙합": "R&B/Soul",
                "R&B/Soul": "랩/힙합", "록/메탈": "발라드", "국내드라마": "발라드"}


# ---------------------------------------------------------------------------
# 정답 곡에서 슬롯 값 뽑기
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> Dict[str, dict]:
    songs: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                song = json.loads(line)
                songs[song["song_id"]] = song
    return songs


def _first(value) -> Optional[str]:
    if isinstance(value, list):
        return next((str(v) for v in value if str(v).strip()), None)
    text = str(value or "").strip()
    return text or None


def oracle_value(song: dict, slot: str) -> Optional[str]:
    """정답 곡의 실제 값. 사용자가 정확히 기억한 경우의 답변."""
    md = song.get("metadata", {})
    if slot == "vocal_gender":
        value = _first(md.get("vocal_gender"))
        return value if value in ("남성", "여성", "혼성") else None
    if slot == "type":
        # 판정기(clarify.answer_matches)와 **같은 함수**로 정규화한다. 평가기가 따로
        # 표를 들고 있었을 때 ['혼성', '듀오']를 평가기는 '그룹', 판정기는 '듀오'로
        # 읽어서, oracle 답변이 정작 정답 곡에 보너스를 주지 못했다(정답 72곡 중 5곡).
        types = canonical_artist_types(md.get("type") or [])
        return next((t for t in _TYPE_ORDER if t in types), None)
    if slot == "genre":
        return _first(md.get("genre"))
    if slot == "release_era":
        year = str(md.get("release_date") or "")[:4]
        return f"{int(year) // 10 * 10}년대" if year.isdigit() else None
    raise ValueError(f"알 수 없는 슬롯: {slot}")


def noisy_value(song: dict, slot: str) -> Optional[str]:
    """틀린 답변. 사용자가 잘못 기억한 경우.

    무작위가 아니라 결정적으로 고른다 — 재측정 때 같은 값이 나와야 비교가 된다.
    """
    truth = oracle_value(song, slot)
    if truth is None:
        return None
    if slot == "vocal_gender":
        return _NOISY_GENDER.get(truth)
    if slot == "type":
        # 정답 곡의 유형 중 어느 것과도 겹치지 않는 값이어야 진짜 오답이다.
        actual = canonical_artist_types(song.get("metadata", {}).get("type") or [])
        return next((t for t in _TYPE_ORDER if t not in actual), None)
    if slot == "genre":
        return _NOISY_GENRE.get(truth, "발라드" if truth != "발라드" else "댄스")
    if slot == "release_era":
        decade = int(truth[:4])
        # 한 시대 앞으로. 1990년대보다 앞은 코퍼스에 없으므로 뒤로 민다.
        return f"{decade - 10}년대" if decade > 1990 else f"{decade + 10}년대"
    return None


# ---------------------------------------------------------------------------
# 한 번의 검색
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """검색 한 번의 결과. 정책 비교에 필요한 것만 담는다."""
    rank: Optional[int]                     # 사용자에게 보여준 top_k 안에서의 순위
    candidate_rank: Optional[int]           # 후보 풀 안에서의 순위
    candidate_recall: float                 # 후보 풀에 정답이 있는가 (0/1)
    shown_ids: List[str] = field(default_factory=list)
    candidate_ids: List[str] = field(default_factory=list)
    candidate_tracks: List = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def remaining(self) -> List:
        """보여준 곡을 뺀 나머지 후보. 질문은 이걸 기준으로 만든다."""
        shown = set(self.shown_ids)
        return [t for t in self.candidate_tracks if str(t.id) not in shown]

    @property
    def found(self) -> bool:
        return self.rank is not None


async def run_search(
    router,
    reranker,
    analysis: QueryAnalysis,
    relevant_ids: set,
    *,
    top_k: int,
    candidate_k: int,
    exclude_ids: Optional[Sequence[str]] = None,
    answers: Optional[Sequence[ClarifyAnswer]] = None,
    answer_multiplier: Optional[float] = None,
) -> TurnResult:
    """실서비스와 같은 순서로 검색한다 — 후보 검색 → 리랭킹 → 상위 top_k.

    evaluate_search_accuracy.py와 동일한 파이프라인을 쓴다. 다르게 만들면
    여기서 나온 수치를 기준선과 비교할 수 없다.
    """
    pool: List[str] = []
    tracks: List = []
    candidates = await router.search(
        analysis,
        top_k=candidate_k,
        use_rerank=False,
        candidate_k=candidate_k,
        exclude_ids=list(exclude_ids or []),
        candidate_ids_out=pool,
        candidate_tracks_out=tracks,
        answers=list(answers or []),
        answer_multiplier=answer_multiplier,
    )
    if not candidates:
        return TurnResult(rank=None, candidate_rank=None, candidate_recall=0.0,
                          metrics=metric_bundle([], relevant_ids, top_k))

    # 리랭크의 가사 exact 그룹 정렬은 질의의 성별·장르를 다시 본다. 원래 분석을
    # 넘기면 사용자가 정정한 답이 그 자리에서 무시된다(라우터와 같은 이유).
    reranked_all = rerank_preserving_exact_lyrics(
        reranker, analysis_with_answers(analysis, answers or []), candidates, candidate_k,
        answers=list(answers or []),
    )
    shown = [str(t.id) for t in reranked_all[:top_k]]
    candidate_ids = [str(t.id) for t in candidates]

    return TurnResult(
        rank=first_relevant_rank(shown, relevant_ids),
        candidate_rank=first_relevant_rank(candidate_ids, relevant_ids),
        candidate_recall=recall_at_k(candidate_ids, relevant_ids, candidate_k),
        shown_ids=shown,
        candidate_ids=candidate_ids,
        candidate_tracks=tracks or list(candidates),
        metrics=metric_bundle(shown, relevant_ids, top_k),
    )


def as_answer(slot: str, value: str) -> ClarifyAnswer:
    """실서비스와 같은 형태의 답변. 분석에 병합하지 않고 그대로 넘긴다."""
    return ClarifyAnswer(slot=slot, value=value)


# ---------------------------------------------------------------------------
# 질의 하나에 대해 모든 정책 실행
# ---------------------------------------------------------------------------

async def evaluate_query(
    router,
    reranker,
    analysis: QueryAnalysis,
    song: Optional[dict],
    relevant_ids: set,
    *,
    top_k: int,
    candidate_k: int,
    answer_multiplier: Optional[float] = None,
) -> Tuple[Dict[str, TurnResult], Optional[str]]:
    """(정책별 결과, 규칙이 고른 슬롯)을 돌려준다.

    개입이 필요 없는 질의는 initial만 담고 슬롯은 None이다.
    """
    results: Dict[str, TurnResult] = {}
    initial = await run_search(
        router, reranker, analysis, relevant_ids,
        top_k=top_k, candidate_k=candidate_k,
    )
    results["initial"] = initial

    if initial.found:
        # 정답이 이미 보였으므로 사용자는 "이 중에는 없어요"를 누르지 않는다.
        return results, None

    rejected = list(initial.shown_ids)

    reject_only = await run_search(
        router, reranker, analysis, relevant_ids,
        top_k=top_k, candidate_k=candidate_k, exclude_ids=rejected,
    )
    results["reject_only"] = reject_only

    # 2턴 누적 — 1턴에서도 못 찾았을 때만 의미가 있다.
    if not reject_only.found:
        twice = await run_search(
            router, reranker, analysis, relevant_ids,
            top_k=top_k, candidate_k=candidate_k,
            exclude_ids=rejected + list(reject_only.shown_ids),
        )
        results["reject_twice"] = twice

    # "잘 모르겠어요" — 스킵 답변은 아무 후보에도 보너스를 주지 않으므로 reject_only와
    # 같아야 한다. 같지 않으면 왕복 어딘가에서 상태가 새고 있다는 뜻이다.
    results["skip"] = await run_search(
        router, reranker, analysis, relevant_ids,
        top_k=top_k, candidate_k=candidate_k, exclude_ids=rejected,
        answers=[ClarifyAnswer(slot="vocal_gender", skipped=True)],
    )

    # --- 규칙 기반 질문 선택 (단계 4) -------------------------------------
    # oracle:<slot>이 "그 슬롯을 물었다면"이라면, rule:*은 "우리 규칙이 고른
    # 슬롯을 물었다면"이다. reject_only를 넘고 oracle:best에 얼마나 닿는지가
    # 질문 선택기를 만든 이유 그 자체다.
    question = pick_question(analysis, initial.remaining)
    picked = question.slot if question else None

    if question is None:
        # 물을 게 없다 — 설계대로 Reject-only로 진행한다.
        for mode in ("oracle", "noisy", "skip"):
            results[f"rule:{mode}"] = reject_only
    else:
        results["rule:skip"] = results["skip"]
        if song is not None:
            for mode, getter in (("oracle", oracle_value), ("noisy", noisy_value)):
                value = getter(song, picked)
                if not value:
                    results[f"rule:{mode}"] = reject_only
                    continue
                results[f"rule:{mode}"] = await run_search(
                    router, reranker, analysis, relevant_ids,
                    top_k=top_k, candidate_k=candidate_k, exclude_ids=rejected,
                    answers=[as_answer(picked, value)],
                    answer_multiplier=answer_multiplier,
                )

    if song is None:
        return results, picked

    for slot in SLOTS:
        truth, wrong = oracle_value(song, slot), noisy_value(song, slot)
        if truth:
            results[f"oracle:{slot}"] = await run_search(
                router, reranker, analysis, relevant_ids,
                top_k=top_k, candidate_k=candidate_k, exclude_ids=rejected,
                answers=[as_answer(slot, truth)],
                answer_multiplier=answer_multiplier,
            )
        if wrong:
            results[f"noisy:{slot}"] = await run_search(
                router, reranker, analysis, relevant_ids,
                top_k=top_k, candidate_k=candidate_k, exclude_ids=rejected,
                answers=[as_answer(slot, wrong)],
                answer_multiplier=answer_multiplier,
            )
    return results, picked


# ---------------------------------------------------------------------------
# 서비스 2턴 흐름 (flow:*)
# ---------------------------------------------------------------------------

FLOW_MODES: Sequence[str] = ("reject", "skip", "oracle", "noisy")
FLOW_POLICIES: List[str] = [f"flow:{mode}" for mode in FLOW_MODES]


@dataclass
class FlowStep:
    """흐름의 검색 한 번. turn 1은 최초 검색이다."""
    turn: int
    result: TurnResult
    slot: str = ""          # 이 검색 직전에 답한 질문의 슬롯
    answer_kind: str = ""   # oracle · noisy · skip · none(질문 없이 거절) — turn 1은 빈칸
    answer_value: str = ""
    options: str = ""       # 화면이 내준 선택지 "값:곡수|…"
    rejected: int = 0       # 이 검색에서 뺀 곡 수


def choose_answer(
    question: ClarifyQuestion,
    mode: str,
    target: Optional[MatchingTrack],
    song: Optional[dict],
) -> Tuple[ClarifyAnswer, str]:
    """화면에서 고를 수 있는 답만 고른다 — 선택지 하나 또는 "잘 모르겠어요".

    맞는지는 서비스의 판정기(`answer_matches`)로 본다. 정답 곡은 후보와 같은
    출처(벡터 DB 페이로드)에서 읽은 MatchingTrack이어야 한다.
    """
    skip = ClarifyAnswer(slot=question.slot, skipped=True)
    if mode == "skip" or target is None:
        return skip, "skip"
    values = [option.value for option in question.options]
    matching = [v for v in values if answer_matches(target, as_answer(question.slot, v))]
    if mode == "oracle":
        return (as_answer(question.slot, matching[0]), "oracle") if matching else (skip, "skip")
    wrong = [v for v in values if v not in matching]
    if not wrong:
        return skip, "skip"
    # 결정적으로 고른다. 하네스의 인접 오답이 선택지에 있으면 그것, 아니면 가장 많은 쪽.
    preferred = noisy_value(song, question.slot) if song is not None else None
    return as_answer(question.slot, preferred if preferred in wrong else wrong[0]), "noisy"


def next_question(
    analysis: QueryAnalysis,
    result: TurnResult,
    turn: int,
    asked: Sequence[str],
    rejected: Sequence[str],
    top_k: int,
) -> Optional[ClarifyQuestion]:
    """이번 턴 응답에 실릴 질문. 라우트와 같은 조건(`_can_ask_another`)으로 정한다."""
    if not _can_ask_another(turn, list(asked), list(rejected), top_k):
        return None
    return pick_question(analysis, result.remaining, list(asked))


async def run_flow(
    router,
    reranker,
    analysis: QueryAnalysis,
    relevant_ids: set,
    initial: TurnResult,
    mode: str,
    target: Optional[MatchingTrack],
    song: Optional[dict],
    *,
    top_k: int,
    candidate_k: int,
    answer_multiplier: Optional[float] = None,
) -> List[FlowStep]:
    """서비스의 재질문 대화를 한 번 밟는다. 정답이 보이면 사용자는 멈춘다.

    매 턴 화면이 보내는 것 — 이전 분석(다시 분석하지 않는다), 지금까지의 답변 전부,
    지금까지 보여 준 곡 전부(거절 목록), 물은 슬롯. 거절 목록이 한도(20)를 넘으면
    더 거절할 수 없다(`map.js`의 canRejectMore).
    """
    steps = [FlowStep(turn=1, result=initial)]
    current, turn = initial, 1
    rejected: List[str] = []
    answers: List[ClarifyAnswer] = []
    asked: List[str] = []
    question = None if mode == "reject" else next_question(
        analysis, current, turn, asked, rejected, top_k
    )
    while not current.found:
        nxt = list(dict.fromkeys([*rejected, *current.shown_ids]))
        if not current.shown_ids or len(nxt) > MAX_REJECTED_IDS:
            break
        step = FlowStep(turn=turn + 1, result=current, answer_kind="none")
        if question is not None:
            answer, kind = choose_answer(question, mode, target, song)
            answers = [*answers, answer]
            if answer.slot not in asked:
                asked.append(answer.slot)
            step.slot, step.answer_kind, step.answer_value = question.slot, kind, answer.value
            step.options = "|".join(f"{o.value}:{o.count}" for o in question.options)
        rejected, turn = nxt, turn + 1
        current = await run_search(
            router, reranker, analysis, relevant_ids,
            top_k=top_k, candidate_k=candidate_k, exclude_ids=rejected,
            answers=answers, answer_multiplier=answer_multiplier,
        )
        step.result, step.rejected = current, len(rejected)
        steps.append(step)
        question = None if mode == "reject" else next_question(
            analysis, current, turn, asked, rejected, top_k
        )
    return steps


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

# 한 번 답변 정책 — v07과 같은 정의다. 답변은 첫 정답(positives[0]) 기준인데 성공은 정답 중
# 아무 곡이나 보이면 인정하므로, 복수 정답 질의에서는 두 기준이 어긋난다(아래 flow:*는 맞췄다).
LEGACY_ORDER = ["initial", "reject_only", "reject_twice", "skip"] + \
               [f"oracle:{s}" for s in SLOTS] + [f"noisy:{s}" for s in SLOTS] + \
               ["rule:oracle", "rule:noisy", "rule:skip", "oracle:best"]
POLICY_ORDER = LEGACY_ORDER + FLOW_POLICIES


def add_oracle_best(results: Dict[str, TurnResult]) -> None:
    """슬롯을 완벽하게 고르는 질문 선택기의 천장.

    단계 4의 pick_question이 아무리 잘해도 이보다 나을 수 없다. 이 값이
    reject_only와 차이가 없으면 질문 선택 로직을 만들 이유 자체가 없다.
    """
    oracles = [(k, v) for k, v in results.items() if k.startswith("oracle:")]
    if not oracles:
        return
    best = min(oracles, key=lambda kv: (kv[1].rank or 10**6))
    results["oracle:best"] = best[1]


def fill_missing_policies(results: Dict[str, TurnResult], policies: Sequence[str]) -> set:
    """실행되지 않은 정책을 그 상황에서 사용자가 실제로 겪었을 결과로 채운다.

    · 개입 없음        → 최초 검색 결과. 사용자는 거절 버튼을 누르지 않는다.
    · 1턴에서 찾음      → 2턴은 일어나지 않는다. reject_only 결과가 곧 그 정책의 결과다.
    · 슬롯에 답할 값 없음 → 그 질문은 던질 수 없다. 질문 없이 진행한 것과 같다.

    채우지 않으면 정책마다 분모가 달라져 평균을 비교할 수 없다. 실제로
    스모크에서 reject_twice의 분모가 혼자 1이 되는 문제가 있었다.

    실제로 실행된 정책의 키 집합을 돌려준다 — 채운 값과 구분해야 나중에
    "이 정책이 몇 번이나 실제로 개입했는가"를 셀 수 있다.
    """
    executed = set(results)
    fallback = results.get("reject_only") or results["initial"]
    for policy in policies:
        results.setdefault(policy, fallback)
    return executed


def summarize(
    rows: List[dict],
    policies: Sequence[str],
    intervention_ids: set,
) -> List[dict]:
    """정책별 집계. 전체 집합과 개입 집합을 함께 낸다.

    **질의 단위로 센다.** flow:*는 복수 정답 질의에서 목표 곡마다 한 행(대화 하나)이 있으므로
    질의 안에서 먼저 평균한 뒤 질의끼리 평균한다. 한 행뿐인 정책은 그대로다.

    int_found_by_turn2  flow:*에서 첫 재질문(두 번째 검색) 안에 목표 곡이 보인 비율
    int_final_relaxed   마지막 화면에 원래 타깃 또는 허용 정답이 있는 비율
    int_relaxed_seen    flow:* 대화 중 한 번이라도 원래 타깃 또는 허용 정답이 보인 비율. 대화는 목표 곡을
                        찾을 때까지 진행하므로, 앞에서 본 허용 정답을 거절하고 지나갔을 수 있다 — 참고값이다
    n_conv_out_of_candidates
                        flow:*에서 목표 곡을 찾지 못했고 마지막 검색에서도 후보 30 밖이었던 대화 수(질의 평균이
                        아니라 대화 수). 답변은 후보 안에서만 순위를 바꾸므로 재질문이 닿지 않는 대화다
    """

    def by_query(group: List[dict], value) -> List[float]:
        per: Dict[str, List[float]] = {}
        for r in group:
            per.setdefault(r["query_id"], []).append(float(value(r)))
        return [sum(v) / len(v) for v in per.values()]

    def avg(group: List[dict], value) -> float:
        vals = by_query(group, value)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    out: List[dict] = []
    for policy in policies:
        full = [r for r in rows if r["policy"] == policy]
        if not full:
            continue
        inter = [r for r in full if r["query_id"] in intervention_ids]
        is_flow = policy in FLOW_POLICIES
        has_relaxed = "final_relaxed_hit10" in full[0]
        out.append({
            "policy": policy,
            "n_full": len({r["query_id"] for r in full}),
            "n_intervention": len({r["query_id"] for r in inter}),
            "n_conversations": len(inter),
            "n_executed": len({r["query_id"] for r in full if r["ran"] == "1"}),
            "full_hit@1": avg(full, lambda r: r["hit1"]),
            "full_hit@5": avg(full, lambda r: r["hit5"]),
            "full_hit@10": avg(full, lambda r: r["hit10"]),
            "full_mrr@10": avg(full, lambda r: r["mrr10"]),
            "int_hit@1": avg(inter, lambda r: r["hit1"]) if inter else "",
            "int_hit@5": avg(inter, lambda r: r["hit5"]) if inter else "",
            "int_hit@10": avg(inter, lambda r: r["hit10"]) if inter else "",
            "int_candidate_recall@30": avg(inter, lambda r: r["candidate_recall"]) if inter else "",
            "int_found_by_turn2": (
                avg(inter, lambda r: str(r.get("found_turn", "")) == "2") if inter and is_flow else ""
            ),
            "int_final_relaxed": (
                avg(inter, lambda r: r["final_relaxed_hit10"]) if inter and has_relaxed else ""
            ),
            "int_relaxed_seen": (
                avg(inter, lambda r: r.get("relaxed_seen_turn") not in ("", None))
                if inter and is_flow and has_relaxed else ""
            ),
            "n_conv_out_of_candidates": (
                sum(1 for r in inter
                    if r.get("found_turn") in ("", None) and r.get("candidate_rank@30") in ("", None))
                if inter and is_flow else ""
            ),
        })
    return out


def summarize_by_scope(
    rows: List[dict],
    policies: Sequence[str],
    intervention_ids: set,
    scopes: Dict[str, str],
) -> List[dict]:
    """범주형 집계 — 전체 · 범주형 제외(specific) · 범주형으로 나눠 summarize를 다시 돈다.

    all은 clarify_summary.csv와 같다. 질의가 없는 묶음은 행이 없다.
    """
    unknown = sorted({r["query_id"] for r in rows} - set(scopes))
    if unknown:
        raise KeyError(f"queries.json에 없는 질의 — 범주형 여부를 알 수 없다: {', '.join(unknown)}")
    out: List[dict] = []
    for scope in ("all",) + TARGET_SCOPES:
        group = [r for r in rows if scope == "all" or scopes[r["query_id"]] == scope]
        ids = {r["query_id"] for r in group}
        for row in summarize(group, policies, intervention_ids & ids):
            out.append({"scope": scope, **row})
    return out


def intervention_ids_from_detail(rows: List[dict]) -> set:
    """저장된 detail에서 개입 대상을 되찾는다 — reject_only를 실제로 실행한 질의다."""
    return {r["query_id"] for r in rows if r["policy"] == "reject_only" and str(r["ran"]) == "1"}


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        # LF로 쓴다. csv 모듈 기본값(\r\n)은 저장소에 CRLF를 남긴다.
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 실행 전 확인 — 모델·DB를 올리기 전에 끝낸다
# ---------------------------------------------------------------------------

def select_queries(
    path: Path,
    split: Optional[str],
    limit: Optional[int] = None,
    query_ids: Optional[Sequence[str]] = None,
    query_sets: Optional[Sequence[str]] = None,
) -> List[EvalQuery]:
    """측정 대상 질의. query_sets를 주지 않으면 v0.5 세트(v06 기준선)만 고른다.

    queries.json에는 v09(2차 세트)가 같은 split으로 들어 있다. split만으로 고르면 v09 dev가
    섞여 기본 --labels(v06 CSV)와 맞지 않아 멈추고, --split test는 봉인 test 38건을 연다 (PR #20 리뷰).
    """
    chosen = tuple(query_sets) if query_sets else V05_QUERY_SETS
    queries = [
        q for q in load_eval_set(path).queries
        if q.is_scorable and q.query_set in chosen
    ]
    if split:
        queries = [q for q in queries if q.split == split]
    if query_ids:
        wanted = {qid.strip() for qid in query_ids if qid.strip()}
        queries = [q for q in queries if q.query_id in wanted]
    queries.sort(key=lambda q: q.query_id)
    return queries[:limit] if limit else queries


def _ids(raw: str) -> set:
    return {part.strip() for part in (raw or "").split("|") if part.strip()}


def check_against_labels(
    queries: Sequence[EvalQuery],
    path: Path,
    whole_split: Optional[str] = None,
) -> None:
    """평가기가 읽는 queries.json이 측정 기준 CSV와 같은 질의·정답인지.

    whole_split을 주면 그 split의 CSV 질의가 빠짐없이 들어왔는지도 본다.
    """
    with open(path, encoding="utf-8-sig") as f:
        rows = {r["query_id"].strip(): r for r in csv.DictReader(f)}
    problems: List[str] = []
    for q in queries:
        row = rows.get(q.query_id)
        if row is None:
            problems.append(f"{q.query_id}: 라벨 CSV에 없다")
            continue
        if row["query"].strip() != q.query.strip():
            problems.append(f"{q.query_id}: 질의 원문이 다르다")
        if _ids(row["relevant_ids"]) != set(q.positives):
            problems.append(f"{q.query_id}: 정답이 다르다")
        if row["split"].strip() != q.split:
            problems.append(f"{q.query_id}: split이 다르다")
    if whole_split:
        selected = {q.query_id for q in queries}
        extra = sorted(
            qid for qid, row in rows.items()
            if row["split"].strip() == whole_split
            and row.get("query_type", "search").strip().lower() == "search"
            and _ids(row["relevant_ids"]) and qid not in selected
        )
        if extra:
            problems.append(f"라벨 CSV에만 있는 질의: {', '.join(extra)}")
    if problems:
        raise ValueError(
            f"질의 세트가 {path}와 다르다 ({len(problems)}건)\n  " + "\n  ".join(problems[:10])
        )


def prepare_analyses(
    cache_path: Optional[Path],
    queries: Sequence[EvalQuery],
    allow_fallback: bool = False,
) -> Tuple[Optional[Dict[str, QueryAnalysis]], dict]:
    """분석 캐시를 읽어 **선택한 질의 전부**를 꺼내 본다. 문제가 있으면 여기서 멈춘다.

    캐시가 없으면 (None, 출처)를 돌려준다 — 호출부가 질의마다 새로 분석한다.
    """
    if not cache_path:
        print("경고: 분석 캐시 없이 실행합니다. 같은 측정을 두 번 해도 질의 분석이 달라져 "
              "숫자가 갈릴 수 있습니다(--analysis-cache 권장).", flush=True)
        return None, {"mode": "live", "analyzer": analyzer_fingerprint()}

    cache = load_cache(Path(cache_path))
    ids = [q.query_id for q in queries]
    missing = cache.missing(ids)
    if missing:
        raise AnalysisCacheError(f"캐시에 없는 질의 {len(missing)}건: {', '.join(missing[:10])}")
    fallbacks = cache.fallback_ids(ids)
    if fallbacks and not allow_fallback:
        raise AnalysisCacheError(
            f"규칙 폴백으로 저장된 질의 {len(fallbacks)}건: {', '.join(fallbacks[:10])} "
            "— 캐시를 다시 채우거나 --allow-fallback-analysis를 주세요"
        )
    analyses: Dict[str, QueryAnalysis] = {}
    problems: List[str] = []
    for q in queries:
        try:
            analyses[q.query_id] = cache.get(q.query_id, q.query)
        except AnalysisCacheError as exc:
            problems.append(str(exc))
        except Exception as exc:  # noqa: BLE001 - 스키마가 깨진 항목
            problems.append(f"{q.query_id}: {type(exc).__name__}: {exc}")
    if problems:
        raise AnalysisCacheError(
            f"캐시 항목 {len(problems)}건을 쓸 수 없습니다.\n" + "\n".join(problems[:10])
        )
    drift = cache.fingerprint_drift()
    if drift:
        print(f"경고: 캐시를 만든 분석 조건이 지금과 다릅니다({', '.join(drift)}).", flush=True)
    print(f"분석 캐시 사용: {cache_path} (질의 {len(ids)}건, 폴백 {len(fallbacks)}건)", flush=True)
    return analyses, {
        "mode": "cache",
        "path": str(cache_path),
        "meta": cache.meta,
        "fallback_query_ids": fallbacks,
        "fingerprint_drift": drift,
    }


def load_target_tracks(song_ids: Sequence[str]) -> Dict[str, MatchingTrack]:
    """정답 곡을 후보와 **같은 출처**(텍스트 컬렉션 페이로드)에서 읽는다.

    flow:oracle·noisy가 선택지와 맞는지 판정할 때 쓴다. 코퍼스 JSONL로 만들면 장르 표기
    등이 후보와 달라 같은 곡인데도 선택지와 안 맞을 수 있다.
    """
    from src.retrieval.search_service import SearchService
    from src.vector_db.qdrant_backend import collection_name, point_id
    from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME

    ids = list(dict.fromkeys(str(s) for s in song_ids))
    if not ids:
        return {}
    records = get_vector_client().client.retrieve(
        collection_name=collection_name(TEXT_HYBRID_INDEX_NAME, NAMESPACE),
        ids=[point_id(s) for s in ids],
        with_payload=True,
        with_vectors=False,
    )
    out: Dict[str, MatchingTrack] = {}
    for record in records:
        payload = dict(record.payload or {})
        song_id = str(payload.pop("song_id", record.id))
        out[song_id] = SearchService.track_from_match(
            {"id": song_id, "score": 0.0, "metadata": payload}
        )
    return out


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def retarget(result: TurnResult, target_ids: set, top_k: int) -> TurnResult:
    """같은 검색 결과를 다른 정답 기준으로 다시 잰 사본. 검색은 다시 하지 않는다."""
    return TurnResult(
        rank=first_relevant_rank(result.shown_ids, target_ids),
        candidate_rank=first_relevant_rank(result.candidate_ids, target_ids),
        candidate_recall=recall_at_k(result.candidate_ids, target_ids, len(result.candidate_ids) or 1),
        shown_ids=result.shown_ids,
        candidate_ids=result.candidate_ids,
        candidate_tracks=result.candidate_tracks,
        metrics=metric_bundle(result.shown_ids, target_ids, top_k),
    )


def _row(q: EvalQuery, policy: str, ran: bool, picked_slot: Optional[str], turn: TurnResult,
         relaxed_ids: set, target_id: str = "", steps: Optional[List[FlowStep]] = None) -> dict:
    final_relaxed = first_relevant_rank(turn.shown_ids, relaxed_ids)
    flow = steps is not None
    return {
        "query_id": q.query_id,
        "split": q.split,
        "query_set": q.query_set,
        "tier": q.tier,
        "policy": policy,
        "target_id": target_id,
        "ran": "1" if ran else "0",
        "picked_slot": picked_slot or "",
        "rank": turn.rank if turn.rank is not None else "",
        "candidate_rank@30": turn.candidate_rank if turn.candidate_rank is not None else "",
        "candidate_recall": round(turn.candidate_recall, 6),
        "hit1": turn.metrics["hit1"],
        "hit5": turn.metrics["hit5"],
        "hit10": turn.metrics["hit10"],
        "mrr10": round(float(turn.metrics["mrr10"]), 6),
        "ndcg10": round(float(turn.metrics["ndcg10"]), 6),
        "final_relaxed_rank": final_relaxed or "",
        "final_relaxed_hit10": int(final_relaxed is not None),
        "found_turn": next((s.turn for s in steps if s.result.found), "") if flow else "",
        "relaxed_seen_turn": next(
            (s.turn for s in steps if first_relevant_rank(s.result.shown_ids, relaxed_ids)), ""
        ) if flow else "",
        "searches": len(steps) if flow else "",
        "shown_ids": "|".join(turn.shown_ids),
    }


async def evaluate(args: argparse.Namespace) -> None:
    queries = select_queries(
        args.queries, args.split, args.limit,
        args.query_ids.split(",") if args.query_ids else None,
        query_sets=args.query_set,
    )
    if not queries:
        raise ValueError("평가할 질의가 없습니다. --split과 label_status를 확인하세요.")

    # 모델·DB를 올리기 전에 끝낸다 — 중간에 멈추면 앞 질의 결과만 남은 폴더가 생긴다.
    if args.labels:
        check_against_labels(
            queries, Path(args.labels),
            whole_split=args.split if not (args.limit or args.query_ids) else None,
        )
    analyses, analysis_source = prepare_analyses(
        Path(args.analysis_cache) if args.analysis_cache else None,
        queries, allow_fallback=args.allow_fallback_analysis,
    )

    corpus = load_corpus(args.corpus) if args.corpus.exists() else {}
    if not corpus:
        print(f"[경고] {args.corpus} 없음 — 답변 정책(oracle/noisy)을 건너뜁니다.", flush=True)

    analyzer = get_query_analyzer() if analyses is None else None
    router, reranker = get_search_router(), get_reranker()
    reranker.load()
    targets = load_target_tracks([sid for q in queries for sid in q.positives])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plain_args = argparse.Namespace(
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    )
    run_info = collect_run_info(plain_args, reranker=reranker, analysis_source=analysis_source)
    run_info["clarify"] = {
        "flow_modes": list(FLOW_MODES),
        "flow_target": "정답마다 따로 대화한다 — 답변·종료·성공 판정이 같은 목표 곡",
        "max_rejected_ids": MAX_REJECTED_IDS,
        "targets_from_vector_db": len(targets),
    }
    (args.output_dir / "clarify_runinfo.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )

    detail: List[dict] = []
    turn_rows: List[dict] = []
    intervention_ids: set = set()
    started = time.perf_counter()

    for index, q in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] {q.query_id}: {q.query[:58]}", flush=True)
        relevant = set(q.positives)
        relaxed_ids = relevant | set(q.allowed)
        analysis = analyses[q.query_id] if analyses is not None else analyzer.analyze(q.query)
        song = corpus.get(q.positives[0]) if corpus else None

        results, picked_slot = await evaluate_query(
            router, reranker, analysis, song, relevant,
            top_k=args.top_k, candidate_k=args.candidate_k,
            answer_multiplier=args.answer_multiplier,
        )
        add_oracle_best(results)
        intervened = "reject_only" in results
        if intervened:
            intervention_ids.add(q.query_id)
        executed = fill_missing_policies(results, LEGACY_ORDER)

        for policy in LEGACY_ORDER:
            detail.append(_row(q, policy, policy in executed, picked_slot, results[policy], relaxed_ids))

        # flow:* — 사용자는 곡 하나를 찾는다. 정답이 여럿이면 목표 곡마다 따로 대화하고,
        # 답변·종료·성공을 모두 그 곡으로 판정한다.
        for mode in FLOW_MODES:
            policy = f"flow:{mode}"
            if not intervened:
                # 정답이 이미 보였다 — 대화는 최초 검색 한 번으로 끝난다.
                steps = [FlowStep(turn=1, result=results["initial"])]
                detail.append(_row(q, policy, False, picked_slot, results["initial"], relaxed_ids,
                                   steps=steps))
                continue
            for target_id in q.positives:
                goal = {target_id}
                steps = await run_flow(
                    router, reranker, analysis, goal,
                    retarget(results["initial"], goal, args.top_k), mode,
                    targets.get(target_id), corpus.get(target_id) if corpus else None,
                    top_k=args.top_k, candidate_k=args.candidate_k,
                    answer_multiplier=args.answer_multiplier,
                )
                detail.append(_row(q, policy, True, picked_slot, steps[-1].result, relaxed_ids,
                                   target_id=target_id, steps=steps))
                for step in steps:
                    result = step.result
                    turn_rows.append({
                        "query_id": q.query_id,
                        "split": q.split,
                        "tier": q.tier,
                        "policy": policy,
                        "target_id": target_id,
                        "turn": step.turn,
                        "slot": step.slot,
                        "answer_kind": step.answer_kind,
                        "answer_value": step.answer_value,
                        "options": step.options,
                        "rejected": step.rejected,
                        "rank": result.rank if result.rank is not None else "",
                        "candidate_rank@30": result.candidate_rank if result.candidate_rank is not None else "",
                        "relaxed_rank": first_relevant_rank(result.shown_ids, relaxed_ids) or "",
                        "shown_ids": "|".join(result.shown_ids),
                        "candidate_ids": "|".join(result.candidate_ids),
                    })

        write_csv(args.output_dir / "clarify_detail.csv", detail)
        write_csv(args.output_dir / "clarify_turns.csv", turn_rows)

    summary, by_scope = write_summaries(
        args.output_dir, detail, intervention_ids, {q.query_id: q.target_scope for q in queries},
    )

    elapsed = time.perf_counter() - started
    from src.retrieval.clarify import ANSWER_MATCH_MULTIPLIER
    weight = args.answer_multiplier if args.answer_multiplier is not None else ANSWER_MATCH_MULTIPLIER
    print(f"\n질의 {len(queries)}개 · 개입 대상 {len(intervention_ids)}개 · "
          f"답변 보너스 ×{weight} · {elapsed:.0f}초")
    print_summaries(args.output_dir, summary, by_scope, intervention_ids)
    print(f"상세: {args.output_dir / 'clarify_detail.csv'}")
    print(f"턴별: {args.output_dir / 'clarify_turns.csv'}")


def write_summaries(
    output_dir: Path, detail: List[dict], intervention_ids: set, scopes: Dict[str, str],
) -> Tuple[List[dict], List[dict]]:
    summary = summarize(detail, POLICY_ORDER, intervention_ids)
    write_csv(output_dir / "clarify_summary.csv", summary)
    by_scope = summarize_by_scope(detail, POLICY_ORDER, intervention_ids, scopes)
    write_csv(output_dir / "clarify_summary_by_scope.csv", by_scope)
    return summary, by_scope


def print_summaries(
    output_dir: Path, summary: List[dict], by_scope: List[dict], intervention_ids: set,
) -> None:
    print(f"개입 대상: {', '.join(sorted(intervention_ids)) or '없음'}\n")

    header = (f"{'정책':<18}{'실행':>5}{'전체 Hit@10':>12}{'개입 Hit@10':>12}"
              f"{'개입 후보유지':>13}{'2턴 안':>8}{'확장(마지막)':>12}{'확장(대화중)':>12}")
    print(header)
    print("-" * len(header))
    for row in summary:
        print(f"{row['policy']:<18}{row['n_executed']:>5}{row['full_hit@10']:>12}"
              f"{str(row['int_hit@10']):>12}{str(row['int_candidate_recall@30']):>13}"
              f"{str(row['int_found_by_turn2']):>8}{str(row['int_final_relaxed']):>12}"
              f"{str(row['int_relaxed_seen']):>12}")

    print("\n범주형 집계 — 개입 대상에서 3턴 안에 목표 곡을 본 비율 · 끝까지 후보 밖인 대화 수")
    for row in by_scope:
        if row["policy"] in FLOW_POLICIES and row["n_intervention"]:
            print(f"  {row['scope']:<12}{row['policy']:<14}개입 {row['n_intervention']:>2} · "
                  f"3턴 안 {row['int_hit@10']:<6} · 대화 {row['n_conversations']:>2} 중 후보 밖 "
                  f"{row['n_conv_out_of_candidates']}")
    print(f"\n요약: {output_dir / 'clarify_summary.csv'}")
    print(f"범주형 집계: {output_dir / 'clarify_summary_by_scope.csv'}")


def resummarize(args: argparse.Namespace) -> None:
    """이미 잰 clarify_detail.csv로 요약 두 개를 다시 쓴다. 모델·DB를 올리지 않는다."""
    detail_path = args.output_dir / "clarify_detail.csv"
    with detail_path.open(encoding="utf-8") as f:
        detail = list(csv.DictReader(f))
    if not detail:
        raise ValueError(f"detail이 비어 있다: {detail_path}")
    intervention_ids = intervention_ids_from_detail(detail)
    summary, by_scope = write_summaries(
        args.output_dir, detail, intervention_ids, load_target_scopes(args.queries),
    )
    print(f"{detail_path} — 질의 {len({r['query_id'] for r in detail})}개 · "
          f"개입 대상 {len(intervention_ids)}개 (재집계만)")
    print_summaries(args.output_dir, summary, by_scope, intervention_ids)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="재질문 정책별 검색 성능 비교")
    p.add_argument("--queries", type=Path, default=DEFAULT_EVAL_PATH)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument(
        "--query-set", action="append", choices=list(get_args(QuerySet)), default=None,
        help="질의 출처 세트. 반복 지정 가능. 생략하면 v0.5 세트(v04·modality_v1·clarify_v1 = v06 기준선). "
             "v09를 재려면 명시하고 --labels도 eval_queries_v09.csv로 바꾼다",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-k", type=int, default=30)
    p.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (스모크용)")
    p.add_argument("--query-ids", default="", help="쉼표로 구분한 query_id만 (예: q115,c601)")
    p.add_argument(
        "--answer-multiplier", type=float, default=None,
        help="답변 일치 보너스 배수 (생략하면 clarify.ANSWER_MATCH_MULTIPLIER). 보정용",
    )
    p.add_argument(
        "--analysis-cache", default="",
        help="build_analysis_cache로 만든 QueryAnalysis 캐시. 주면 질의 분석을 새로 하지 않는다",
    )
    p.add_argument(
        "--allow-fallback-analysis", action="store_true",
        help="캐시에 규칙 폴백(Gemini 실패)으로 저장된 질의가 있어도 진행한다",
    )
    p.add_argument(
        "--labels", default=str(DEFAULT_LABELS),
        help="질의·정답이 같은지 대조할 기준 CSV. 빈 문자열이면 대조하지 않는다",
    )
    p.add_argument(
        "--resummarize", action="store_true",
        help="측정하지 않고 --output-dir의 clarify_detail.csv로 요약(전체·범주형)만 다시 쓴다",
    )
    return p


if __name__ == "__main__":
    _args = build_parser().parse_args()
    if _args.resummarize:
        resummarize(_args)
    else:
        asyncio.run(evaluate(_args))
