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

가장 중요한 수치는 **오답 응답 시 정답의 후보@30 유지율**이다.
"답변은 필터가 아니라 부스팅이라 안전하다"는 주장은 이 수치로만 증명된다.
유지율이 낮으면 틀린 답변 한 번에 정답이 후보 풀에서 사라진다는 뜻이고,
그러면 재질문은 사용자를 돕는 게 아니라 해치는 기능이 된다.

개입 대상만 잰다
    정답이 이미 Top-10에 있으면 사용자는 "이 중에는 없어요"를 누르지 않는다.
    그런 질의는 모든 정책이 동일하므로 최초 검색 결과를 그대로 쓴다.
    전체 집합 지표와 개입 집합 지표를 함께 보고한다 — 전자는 서비스 전체
    효과, 후자는 정책 간 우열을 본다.

  python -m src.retrieval.evaluate_clarification --split dev
  python -m src.retrieval.evaluate_clarification --split dev --limit 3   # 스모크
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
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.api.dependencies import (
    get_query_analyzer,
    get_reranker,
    get_search_router,
)
from src.retrieval.clarify import (
    analysis_with_answers,
    canonical_artist_types,
    pick_question,
)
from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import ClarifyAnswer
from src.eval.loader import DEFAULT_EVAL_PATH, load_eval_set
from src.eval.schema import EvalQuery
from src.retrieval.evaluate_search_accuracy import (
    first_relevant_rank,
    mean,
    metric_bundle,
    recall_at_k,
    rerank_preserving_exact_lyrics,
)

DEFAULT_CORPUS = Path("data/all_songs.jsonl")
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
# 집계
# ---------------------------------------------------------------------------

POLICY_ORDER = ["initial", "reject_only", "reject_twice", "skip"] + \
               [f"oracle:{s}" for s in SLOTS] + [f"noisy:{s}" for s in SLOTS] + \
               ["rule:oracle", "rule:noisy", "rule:skip", "oracle:best"]


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
    """정책별 집계. 전체 집합과 개입 집합을 함께 낸다."""
    out: List[dict] = []
    for policy in policies:
        full = [r for r in rows if r["policy"] == policy]
        if not full:
            continue
        inter = [r for r in full if r["query_id"] in intervention_ids]
        out.append({
            "policy": policy,
            "n_full": len(full),
            "n_intervention": len(inter),
            "n_executed": sum(1 for r in full if r["ran"] == "1"),
            "full_hit@1": round(mean(full, "hit1"), 4),
            "full_hit@5": round(mean(full, "hit5"), 4),
            "full_hit@10": round(mean(full, "hit10"), 4),
            "full_mrr@10": round(mean(full, "mrr10"), 4),
            "int_hit@1": round(mean(inter, "hit1"), 4) if inter else "",
            "int_hit@5": round(mean(inter, "hit5"), 4) if inter else "",
            "int_hit@10": round(mean(inter, "hit10"), 4) if inter else "",
            "int_candidate_recall@30": round(mean(inter, "candidate_recall"), 4) if inter else "",
        })
    return out


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
# 실행
# ---------------------------------------------------------------------------

def select_queries(
    path: Path,
    split: Optional[str],
    limit: Optional[int] = None,
    query_ids: Optional[Sequence[str]] = None,
) -> List[EvalQuery]:
    queries = [q for q in load_eval_set(path).queries if q.is_scorable]
    if split:
        queries = [q for q in queries if q.split == split]
    if query_ids:
        wanted = {qid.strip() for qid in query_ids if qid.strip()}
        queries = [q for q in queries if q.query_id in wanted]
    queries.sort(key=lambda q: q.query_id)
    return queries[:limit] if limit else queries


async def evaluate(args: argparse.Namespace) -> None:
    queries = select_queries(
        args.queries, args.split, args.limit,
        args.query_ids.split(",") if args.query_ids else None,
    )
    if not queries:
        raise ValueError("평가할 질의가 없습니다. --split과 label_status를 확인하세요.")

    corpus = load_corpus(args.corpus) if args.corpus.exists() else {}
    if not corpus:
        print(f"[경고] {args.corpus} 없음 — 답변 정책(oracle/noisy)을 건너뜁니다.", flush=True)

    analyzer, router, reranker = get_query_analyzer(), get_search_router(), get_reranker()
    reranker.load()

    detail: List[dict] = []
    intervention_ids: set = set()
    started = time.perf_counter()

    for index, q in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] {q.query_id}: {q.query[:58]}", flush=True)
        relevant = set(q.positives)
        analysis = analyzer.analyze(q.query)   # Gemini는 질의당 한 번만
        song = corpus.get(q.positives[0]) if corpus else None

        results, picked_slot = await evaluate_query(
            router, reranker, analysis, song, relevant,
            top_k=args.top_k, candidate_k=args.candidate_k,
            answer_multiplier=args.answer_multiplier,
        )
        add_oracle_best(results)

        if "reject_only" in results:
            intervention_ids.add(q.query_id)
        executed = fill_missing_policies(results, POLICY_ORDER)

        for policy in POLICY_ORDER:
            turn = results[policy]
            detail.append({
                "query_id": q.query_id,
                "split": q.split,
                "query_set": q.query_set,
                "tier": q.tier,
                "policy": policy,
                "ran": "1" if policy in executed else "0",
                "picked_slot": picked_slot or "",
                "rank": turn.rank if turn.rank is not None else "",
                "candidate_rank@30": turn.candidate_rank if turn.candidate_rank is not None else "",
                "candidate_recall": round(turn.candidate_recall, 6),
                "hit1": turn.metrics["hit1"],
                "hit5": turn.metrics["hit5"],
                "hit10": turn.metrics["hit10"],
                "mrr10": round(float(turn.metrics["mrr10"]), 6),
                "ndcg10": round(float(turn.metrics["ndcg10"]), 6),
                "shown_ids": "|".join(turn.shown_ids),
            })

        write_csv(args.output_dir / "clarify_detail.csv", detail)

    summary = summarize(detail, POLICY_ORDER, intervention_ids)
    write_csv(args.output_dir / "clarify_summary.csv", summary)

    elapsed = time.perf_counter() - started
    from src.retrieval.clarify import ANSWER_MATCH_MULTIPLIER
    weight = args.answer_multiplier if args.answer_multiplier is not None else ANSWER_MATCH_MULTIPLIER
    print(f"\n질의 {len(queries)}개 · 개입 대상 {len(intervention_ids)}개 · "
          f"답변 보너스 ×{weight} · {elapsed:.0f}초")
    print(f"개입 대상: {', '.join(sorted(intervention_ids)) or '없음'}\n")

    header = (f"{'정책':<18}{'실행':>5}{'전체 Hit@10':>12}"
              f"{'개입 Hit@10':>12}{'개입 후보유지':>13}")
    print(header)
    print("-" * len(header))
    for row in summary:
        print(f"{row['policy']:<18}{row['n_executed']:>5}{row['full_hit@10']:>12}"
              f"{str(row['int_hit@10']):>12}{str(row['int_candidate_recall@30']):>13}")
    print(f"\n상세: {args.output_dir / 'clarify_detail.csv'}")
    print(f"요약: {args.output_dir / 'clarify_summary.csv'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="재질문 정책별 검색 성능 비교")
    p.add_argument("--queries", type=Path, default=DEFAULT_EVAL_PATH)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-k", type=int, default=30)
    p.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (스모크용)")
    p.add_argument("--query-ids", default="", help="쉼표로 구분한 query_id만 (예: q115,c601)")
    p.add_argument(
        "--answer-multiplier", type=float, default=None,
        help="답변 일치 보너스 배수 (생략하면 clarify.ANSWER_MATCH_MULTIPLIER). 보정용",
    )
    return p


if __name__ == "__main__":
    asyncio.run(evaluate(build_parser().parse_args()))
