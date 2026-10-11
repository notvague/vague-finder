"""발표 순서 그대로 한 번 돌려 본다 — **실제 서버에 실제 질의로.**

왜 따로 두는가. 질의를 하나씩 손으로 넣어 보면 "텍스트는 되더라"까지만 알 수 있다.
발표는 텍스트 → 이미지 → 가사 → 재질문을 **이어서** 하고, 그때 처음 쓰는 경로가
그 자리에서 모델을 올린다. 이어서 돌려야 그 비용이 어디서 튀는지 보인다.

`measure_search_latency`와 다른 일을 한다 — 그쪽은 dev 53건으로 분포를 재고,
여기서는 **발표에서 실제로 누를 순서** 몇 개를 한 번씩 밟는다.

    venv/bin/uvicorn src.backend.main:app        # 먼저 띄우고
    venv/bin/python -m src.backend.demo_rehearsal

확인하는 것: 단계마다 몇 초 걸리는가 · 결과가 나오는가 · 실행 기록이 성한가
(질의 분석 폴백·죽은 경로·리랭킹 실패가 없는가).

요청은 **화면(map.js)이 보내는 것과 같은 모양**이다 — 검색은 `defer_clarify`로 보내고, 재질문은
"이 중에는 없어요"를 누를 때처럼 `/search/clarify`로 질문을 조회한 뒤 답을 실어 다시 검색한다.
화면의 흐름이 바뀌면 여기도 같이 바꾼다. 다른 모양으로 재면 발표에서 누르지 않을 경로의 시간이 나온다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.backend.schemas.search import MAX_PREVIOUS_CANDIDATES

DEFAULT_URL = "http://127.0.0.1:8000"
SEARCH_PATH = "/api/v1/search"
CLARIFY_PATH = "/api/v1/search/clarify"
TOP_K = 10

# 발표에서 누를 순서. 경로가 겹치지 않게 한 단계에 하나씩 맡긴다.
SEQUENCE: List[Dict[str, str]] = [
    {
        "name": "1. 텍스트 — 분위기만 말한다",
        "query": "비 오는 날 혼자 듣기 좋은 잔잔한 발라드",
        "expect": "text_hybrid",
    },
    {
        "name": "2. 이미지 — 앨범 표지를 말한다",
        "query": "앨범 커버가 온통 빨간색이었던 노래를 찾고 있어",
        "expect": "image",
    },
    {
        "name": "3. 가사 — 기억나는 구절을 말한다",
        # 실제 가사 한 줄이어야 한다. 제목을 넣으면 가사 경로가 잡을 것이 없다
        # — 처음에 "사랑은 가슴이 시킨다"(버즈)로 넣었다가 0건이었다.
        "query": "가사에 '가시처럼 깊게 박힌 기억은'이 나오는 노래",
        "expect": "lyrics_surface",
    },
]


@dataclass
class Step:
    name: str
    seconds: float = 0.0
    status: int = 0
    results: int = 0
    top: str = ""
    note: str = ""
    problems: List[str] = field(default_factory=list)


def _health(url: str) -> Dict[str, Any]:
    return json.loads(urllib.request.urlopen(f"{url}/health", timeout=10).read())


def _post(url: str, body: Dict[str, Any], timeout: float, path: str = SEARCH_PATH) -> tuple:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            payload = json.loads(res.read())
            return time.perf_counter() - started, res.status, payload
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - started, exc.code, json.loads(exc.read() or b"{}")


def _read_run(payload: Dict[str, Any]) -> tuple:
    """실행 기록 한 줄과, 성하지 않은 점들."""
    explain = payload.get("explain") or {}
    parts = [explain.get("reorder_stage_label", "")]
    if explain.get("reranker", "none") != "none":
        parts.append(explain["reranker"])
    problems: List[str] = []
    if explain.get("analysis_fallback"):
        problems.append("질의 분석이 규칙 폴백이다(쿼터·타임아웃 확인)")
    for path in explain.get("failed_paths", []):
        problems.append(path["label"])
    if explain.get("rerank_error"):
        problems.append(f"리랭킹 실패: {explain['rerank_error']}")
    return " · ".join(p for p in parts if p), problems


def _paths_of_top(payload: Dict[str, Any]) -> List[str]:
    results = payload.get("results") or []
    if not results:
        return []
    return [p["path"] for p in ((results[0].get("explain") or {}).get("paths") or [])]


def _clarify_body(convo: Dict[str, Any]) -> Dict[str, Any]:
    """화면이 "이 중에는 없어요"를 누를 때 보내는 질문 조회. 직전 검색 응답의 상태를 그대로 돌려준다."""
    return {
        "asked_slots": convo["asked_slots"],
        "previous_candidate_ids": convo["candidate_ids"][:MAX_PREVIOUS_CANDIDATES],
        "shown_ids": [r["id"] for r in convo["results"]],
        "rejected_ids": convo["rejected_ids"],
        "turn": convo["turn"],
    }


def rehearse(url: str, timeout: float) -> List[Step]:
    steps: List[Step] = []

    # 0. 예열 상태 — ready가 아니면 첫 검색이 혼자 11초를 낸다
    warm = _health(url)["warmup"]
    failed = [name for name, stage in warm.get("stages", {}).items()
              if str(stage.get("outcome", "")).startswith("failed")]
    step = Step(name="0. 예열", note=f"{warm['state']} · {warm['elapsed_ms']:.0f}ms")
    if warm["state"] != "ready":
        step.problems.append(f"예열이 끝나지 않았다({warm['state']}) — 첫 검색이 느리다")
    if failed:
        step.problems.append(f"예열 실패 단계: {', '.join(failed)}")
    steps.append(step)

    # 서버가 "거절하면 질문을 조회하라"고 한 응답들. 화면은 이런 결과에서만 질문을 조회한다.
    deferred: List[Dict[str, Any]] = []
    for spec in SEQUENCE:
        seconds, status, payload = _post(
            url,
            {"query": spec["query"], "top_k": TOP_K, "explain": True, "defer_clarify": True},
            timeout,
        )
        note, problems = _read_run(payload)
        results = payload.get("results") or []
        step = Step(
            name=spec["name"], seconds=seconds, status=status,
            results=len(results), note=note, problems=problems,
        )
        if status != 200:
            step.problems.append(f"HTTP {status}: {payload.get('detail', '')}")
        elif not results:
            step.problems.append("결과가 0건이다")
        else:
            step.top = f"{results[0]['title']} — {results[0].get('artist', '')}"
            paths = _paths_of_top(payload)
            if spec["expect"] not in paths:
                step.problems.append(
                    f"1위 곡에 {spec['expect']} 경로 기여가 없다(실제: {paths or '없음'})"
                )
        steps.append(step)
        if status == 200 and payload.get("clarify_deferred"):
            deferred.append(payload)

    # 4. 재질문 — 화면이 하는 것과 같은 왕복: 거절 버튼 → 질문 조회 → 답 → 다시 검색
    step = Step(name="4. 재질문 — '이 중에는 없어요' 뒤 답하기")
    convo: Optional[Dict[str, Any]] = None
    question: Optional[Dict[str, Any]] = None
    lookup_seconds = 0.0
    # 질문은 후보가 갈릴 때만 나온다. 앞 단계의 결과를 차례로 눌러 보고 처음 질문이 뜨는 것으로 답한다.
    for candidate in deferred:
        seconds, status, payload = _post(url, _clarify_body(candidate), timeout, path=CLARIFY_PATH)
        lookup_seconds += seconds
        if status != 200:
            step.problems.append(f"질문 조회 HTTP {status}: {payload.get('detail', '')}")
            continue
        if payload.get("clarify"):
            convo, question = candidate, payload["clarify"]
            break
    if not deferred:
        step.problems.append("앞 단계의 어느 응답도 질문을 조회하라고 하지 않았다(clarify_deferred)")
    elif question is None:
        step.problems.append("앞 단계의 어느 결과에서도 물을 질문이 나오지 않았다")
    if convo is not None and question is not None:
        shown = [r["id"] for r in convo["results"]]
        answer = {
            "slot": question["slot"],
            "value": question["options"][0]["value"] if question["options"] else "",
            "skipped": not question["options"],
        }
        body = {
            "query": convo["analysis"]["original_query"],
            "top_k": TOP_K,
            "explain": True,
            "defer_clarify": True,
            "prior_analysis": convo["analysis"],
            "answers": [answer],
            "asked_slots": convo["asked_slots"],
            "previous_candidate_ids": convo["candidate_ids"][:MAX_PREVIOUS_CANDIDATES],
            "rejected_ids": list(dict.fromkeys([*convo["rejected_ids"], *shown])),
            "turn": convo["turn"],
        }
        seconds, status, payload = _post(url, body, timeout)
        note, problems = _read_run(payload)
        results = payload.get("results") or []
        step.seconds, step.status, step.results = seconds, status, len(results)
        step.note = (
            f"{question['question']} → {answer['value'] or '건너뜀'}"
            f" (질문 조회 {lookup_seconds:.2f}초)"
        )
        step.problems.extend(problems)
        if status != 200:
            step.problems.append(f"HTTP {status}: {payload.get('detail', '')}")
        elif not results:
            step.problems.append("재질문 뒤 결과가 0건이다")
        else:
            step.top = f"{results[0]['title']} — {results[0].get('artist', '')}"
            if set(r["id"] for r in results) & set(shown):
                step.problems.append("거절한 곡이 다시 나왔다")
            if payload.get("turn") != convo["turn"] + 1:
                step.problems.append(f"턴이 하나 가지 않았다({convo['turn']} → {payload.get('turn')})")
    steps.append(step)
    return steps


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    try:
        steps = rehearse(args.url, args.timeout)
    except urllib.error.URLError as exc:
        print(f"서버에 닿지 못했다({args.url}): {exc}", file=sys.stderr)
        return 2

    print(f"{'단계':<40} {'초':>7}  결과  1위 / 기록")
    print("-" * 100)
    for step in steps:
        seconds = f"{step.seconds:.2f}" if step.seconds else "-"
        tail = step.top or step.note
        print(f"{step.name:<40} {seconds:>7}  {step.results or '-':>3}   {tail}")
        if step.top and step.note:
            print(f"{'':<40} {'':>7}        {step.note}")
        for problem in step.problems:
            print(f"{'':<40} {'':>7}   ⚠ {problem}")

    slow = max((s for s in steps if s.seconds), key=lambda s: s.seconds, default=None)
    if slow:
        print(f"\n가장 느린 단계: {slow.name} — {slow.seconds:.2f}초")
    trouble = [s for s in steps if s.problems]
    print(f"단계 {len(steps)} · 짚을 것 {sum(len(s.problems) for s in trouble)}건")
    return 1 if trouble else 0


if __name__ == "__main__":
    raise SystemExit(main())
