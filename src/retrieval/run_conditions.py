"""측정 실행(runinfo)의 조건을 서로 견준다 — 집계 도구 공용 (모델·DB 없음).

여러 실행을 한 표에 나란히 놓는 도구(`diagnose_genre_gap report` · `experiments/genre_memory_gap/summarize_clarify.py`)가
재는 조건이 다른 실행을 표시 없이 섞지 않게 한다. 조건을 하나씩 골라 보지 않고 runinfo의 조건 블록을 통째로 견준다 —
스위치가 새로 생겨도 빠지지 않는다.

리랭킹 상태도 여기서 읽는다. **모르는 것을 정상으로 치지 않는다**: 정상으로 아는 상태가 아니면 실패로 세고, 상태 집계가
아예 없으면 "실패 0건"이 아니라 "모름"이다.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# runinfo에서 "어떻게 쟀는가"를 담는 블록(`evaluate_search_accuracy.collect_run_info` + 재질문 하네스의 `clarify`).
CONDITION_BLOCKS: Tuple[str, ...] = (
    "args", "analysis", "corpus", "bm25", "lyrics_source", "reranker", "gemini", "ranking_switches",
    "search_reference_year_env", "clarify",
)
# 조건이 아닌 칸 — 결과를 쓴 곳 · 걸린 시간 · 결과 집계.
# 색인 폴더(`corpus.qdrant_path`)는 조건이다. runinfo에는 색인 내용의 지문이 없어(색인 이름 · 곡 수뿐) 폴더가 다르면 같은
# 내용인지 runinfo로는 알 수 없다 — 곡 수와 BM25 지문은 메타 수정이나 Context 색인 유무를 가르지 못한다. 같은 색인의
# 복사본을 나눠 쓴 실행도 그 칸이 다르다고 나온다. 그것이 사실이고, 내용이 같다는 근거는 따로 남긴다.
NOT_CONDITIONS = frozenset({
    "args.output_dir", "clarify.elapsed_seconds", "clarify.rerank_status_counts",
})
# 다르다고만 적으면 오해하기 쉬운 칸에 붙이는 말.
DIFFERENCE_NOTES = {
    "corpus.qdrant_path": "색인 폴더가 다르다 — 같은 내용인지는 runinfo로 알 수 없다",
}
MISSING = "(없음)"

Conditions = Dict[str, Tuple[str, ...]]


def flatten(info: Mapping[str, Any]) -> Dict[str, str]:
    """조건 블록을 `블록.칸` → 값(JSON 문자열)으로 편다. 중첩된 dict는 끝까지 편다."""
    out: Dict[str, str] = {}

    def walk(prefix: str, value: Any) -> None:
        if prefix in NOT_CONDITIONS:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(f"{prefix}.{key}", item)
        else:
            out[prefix] = json.dumps(value, ensure_ascii=False, sort_keys=True)

    for block in CONDITION_BLOCKS:
        if block in info:
            walk(block, info[block])
    return out


def run_conditions(infos: Sequence[Mapping[str, Any]]) -> Conditions:
    """한 실행의 조건. 실행이 세트별 폴더 여럿으로 나뉘어 있으면 칸마다 폴더 순서대로 값을 늘어놓는다."""
    flats = [flatten(info) for info in infos]
    return {key: tuple(flat.get(key, MISSING) for flat in flats)
            for key in dict.fromkeys(key for flat in flats for key in flat)}


def condition_differences(base: Conditions, other: Conditions) -> List[str]:
    """기준 실행과 값이 다른 칸 — `칸: 기준 → 이 실행`. 한쪽에만 있는 칸은 `(없음)`으로 적는다.

    폴더가 여럿인 실행은 같은 값을 한 번만 적는다. 값의 종류는 같고 폴더 자리만 다르면 자리대로 전부 적는다.
    """
    lines: List[str] = []
    for key in dict.fromkeys([*base, *other]):
        mine, theirs = base.get(key, (MISSING,)), other.get(key, (MISSING,))
        if mine == theirs:
            continue
        before, after = " | ".join(dict.fromkeys(mine)), " | ".join(dict.fromkeys(theirs))
        if before == after:
            before, after = " | ".join(mine), " | ".join(theirs)
        note = f"({DIFFERENCE_NOTES[key]})" if key in DIFFERENCE_NOTES else ""
        lines.append(f"{key}: {before} → {after}{note}")
    return lines


# 리랭킹이 실패가 아닌 결말: 적용 · 부를 조건이 아님 · 정책 생략. 평가기의 `rerank_outcome`과 재질문 하네스의 상태 집계가
# 같은 이름을 쓴다(하네스는 정책 생략에 `policy_skipped:사유`를 붙인다).
RERANK_OK_STATES = frozenset({"applied", "skipped", "policy_skipped"})


def rerank_state_ok(name: str) -> bool:
    """정상으로 **아는** 상태인가. 모르는 이름(빈칸 · fallback · unknown …)은 정상으로 치지 않는다."""
    return name in RERANK_OK_STATES or name.startswith("policy_skipped:")


def rerank_failures(counts: Optional[Mapping[str, Any]]) -> Optional[int]:
    """상태별 건수에서 정상이 아닌 것의 수. 집계가 없거나 비어 있으면 None — "실패 없음"이 아니라 "모름"이다."""
    if not counts:
        return None
    return sum(int(count) for name, count in counts.items() if not rerank_state_ok(str(name)))
