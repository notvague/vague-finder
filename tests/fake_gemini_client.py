"""`GeminiListwiseReranker(client=...)`에 주입하는 **가짜 google-genai 클라이언트**.

실제 리랭커의 재시도·JSON 파서·시간 예산(`asyncio.wait_for` 취소)·폴백 상태 보고를 그대로 지나게 하려고
리랭커 자체가 아니라 그 아래의 SDK 클라이언트만 바꾼다. 비동기 면(`aio.models.generate_content`)을 제공하므로
리랭커는 운영과 같은 경로(`_generate` → `asyncio.wait_for`)로 호출하고 느린 응답은 실제로 취소된다.

모드(생성자 인자 또는 `mode_file`의 JSON `{"mode": ..., "sleep": ...}` — 서버 프로세스 밖에서 요청마다 바꿀 수 있다):
  normal         프롬프트의 후보 id를 **역순**으로 돌려준다(리랭킹이 실제로 반영됐는지 눈에 보이게)
  identity       입력 순서 그대로(정상 응답이지만 순위 변화 없음)
  exception      호출마다 예외(`FakeAPIError("499 CANCELLED")`) — 리랭커가 3회 재시도 뒤 failed 로 폴백
  slow           응답하지 않고 잔다 → 리랭커의 벽시계 제한이 호출을 취소한다(요청 제한 20초 · 전체 예산 30초)
  slow_within    `sleep`초 뒤 정상 응답(예산 안의 느린 응답)
  invalid_json   JSON이 아닌 본문 → 파서 실패 → 재시도 3회 → failed
  foreign_ids    후보에 없는 id만 돌려준다 → v59부터 유효 후보 0개는 무효 응답(재시도 뒤 failed). v56까지는 applied·순위 그대로였다
  empty_ranking  `{"ranking": []}` → 같은 무효 응답
  wrong_types    정상 JSON이지만 자료형이 어긋남(최상위 배열) → 무효 응답
  partial_ids    후보의 앞 절반만 역순으로, 첫 id는 두 번, 후보 밖 id 하나 섞어 돌려준다 → 일부 누락·중복·무효 혼합(정상, 보충 유지)
  invalid_then_normal  모드를 켠 뒤 첫 호출만 foreign_ids, 다음부터 normal → 무효 응답 뒤 재시도 성공
  fail_group_first  모드를 켠 뒤 **처음 본 후보 집합**(가사 보호 그룹 등)만 계속 실패시키고 다른 집합은 normal
                    → 그룹 하나만 실패하는 부분 실패를 만든다

모델 호출·네트워크는 없다. 호출 기록은 `calls`(메모리)와 `log_path`(JSONL, 본문 없음)에 남는다.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_ID_RE = re.compile(r'"id":"([^"]+)"')
_CANDIDATES_MARKER = "Candidates (JSON):\n"
FOREIGN_ID_PREFIX = "999000"


def candidate_ids_from_prompt(contents: str) -> List[str]:
    """실제 프롬프트(`_build_prompt`)의 후보 JSON 블록에서 id를 프롬프트 순서대로 꺼낸다.

    스키마 예시(`"id":"candidate id"`)는 후보가 아니므로 후보 블록만 읽는다. 블록을 못 찾으면 정규식으로
    대신하되 예시 문자열은 뺀다.
    """
    text = contents or ""
    start = text.find(_CANDIDATES_MARKER)
    if start >= 0:
        body = text[start + len(_CANDIDATES_MARKER):]
        end = body.find("\n\nReturn one JSON object only")
        block = body[:end] if end >= 0 else body
        try:
            return [str(item.get("id")) for item in json.loads(block.strip()) if isinstance(item, dict)]
        except ValueError:
            pass
    return [song_id for song_id in _ID_RE.findall(text) if song_id != "candidate id"]


class FakeAPIError(Exception):
    """원격 API 오류를 흉내 낸다(메시지만 다르고 처리 경로는 일반 예외와 같다)."""


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text


@dataclass
class FakeCall:
    index: int
    mode: str
    n_candidates: int
    candidate_ids: List[str]
    started_at: float
    elapsed_s: float = 0.0
    outcome: str = ""  # returned | raised | cancelled


def normal_ranking(candidate_ids: List[str], *, reverse: bool = True) -> str:
    ordered = list(reversed(candidate_ids)) if reverse else list(candidate_ids)
    n = max(1, len(ordered))
    return json.dumps({
        "query_confidence": 0.9,
        "ranking": [
            {"id": song_id, "relevance": round(1.0 - i / n, 3), "reason": "v56 가짜 응답"}
            for i, song_id in enumerate(ordered)
        ],
    }, ensure_ascii=False)


class FakeGeminiClient:
    def __init__(
        self,
        mode: str = "normal",
        *,
        mode_file: Optional[str] = None,
        log_path: Optional[str] = None,
        sleep_seconds: float = 8.0,
    ) -> None:
        self._mode = mode
        self._sleep = sleep_seconds
        self._mode_file = mode_file
        self._log_path = log_path
        self.calls: List[FakeCall] = []
        self._fail_set: Optional[frozenset] = None
        self._fail_set_mode_stamp: Optional[float] = None
        self.aio = _Aio(self)
        self.models = _SyncModels(self)

    # --- 모드 ---------------------------------------------------------------
    def set_mode(self, mode: str, *, sleep_seconds: Optional[float] = None) -> None:
        self._mode = mode
        if sleep_seconds is not None:
            self._sleep = sleep_seconds
        self._fail_set = None

    def _current(self) -> Dict[str, Any]:
        if self._mode_file and os.path.exists(self._mode_file):
            try:
                with open(self._mode_file, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                stamp = os.path.getmtime(self._mode_file)
                if stamp != self._fail_set_mode_stamp:
                    # 모드 파일이 바뀌면 fail_group_first 의 "처음 본 집합"도 다시 잡는다
                    self._fail_set = None
                    self._fail_set_mode_stamp = stamp
                return {"mode": str(payload.get("mode", "normal")), "sleep": float(payload.get("sleep", self._sleep))}
            except (OSError, ValueError):
                pass
        return {"mode": self._mode, "sleep": self._sleep}

    # --- 호출 ---------------------------------------------------------------
    def _begin(self, contents: str, mode: str) -> FakeCall:
        ids = candidate_ids_from_prompt(contents)
        call = FakeCall(index=len(self.calls) + 1, mode=mode, n_candidates=len(ids), candidate_ids=ids, started_at=time.time())
        self.calls.append(call)
        return call

    def _end(self, call: FakeCall, outcome: str) -> None:
        call.elapsed_s = round(time.time() - call.started_at, 3)
        call.outcome = outcome
        if self._log_path:
            try:
                with open(self._log_path, "a", encoding="utf-8") as handle:
                    row = asdict(call)
                    row["candidate_ids"] = "|".join(call.candidate_ids)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def _respond(self, call: FakeCall, mode: str) -> _Response:
        ids = call.candidate_ids
        if mode == "exception":
            raise FakeAPIError("499 CANCELLED (v56 simulated API failure)")
        if mode == "invalid_json":
            return _Response("리랭킹 결과입니다: 1위는 첫 번째 곡 (JSON 아님 — v56)")
        if mode == "foreign_ids":
            fake_ids = [f"{FOREIGN_ID_PREFIX}{i:04d}" for i in range(len(ids))]
            return _Response(normal_ranking(fake_ids, reverse=False))
        if mode == "empty_ranking":
            return _Response(json.dumps({"query_confidence": 0.9, "ranking": []}))
        if mode == "wrong_types":
            # 정상 JSON이지만 최상위가 배열 — `payload.get`이 없는 형태
            return _Response(json.dumps([{"id": song_id} for song_id in ids]))
        if mode == "partial_ids":
            half = ids[: max(1, len(ids) // 2)]
            items = [*reversed(half), half[0], f"{FOREIGN_ID_PREFIX}9999"]
            return _Response(json.dumps({
                "query_confidence": 0.9,
                "ranking": [{"id": song_id, "relevance": 0.5, "reason": "v59 가짜 응답"} for song_id in items],
            }))
        if mode == "invalid_then_normal":
            if self._fail_set is None:
                self._fail_set = frozenset({"__first_call_done__"})
                fake_ids = [f"{FOREIGN_ID_PREFIX}{i:04d}" for i in range(len(ids))]
                return _Response(normal_ranking(fake_ids, reverse=False))
            return _Response(normal_ranking(ids))
        if mode == "identity":
            return _Response(normal_ranking(ids, reverse=False))
        if mode == "fail_group_first":
            key = frozenset(ids)
            if self._fail_set is None:
                self._fail_set = key
            if key == self._fail_set:
                raise FakeAPIError("v56 simulated failure for the first candidate group")
            return _Response(normal_ranking(ids))
        return _Response(normal_ranking(ids))

    async def _generate_async(self, contents: str) -> _Response:
        current = self._current()
        mode = current["mode"]
        call = self._begin(contents, mode)
        try:
            if mode == "slow":
                await asyncio.sleep(3600)
            elif mode == "slow_within":
                await asyncio.sleep(current["sleep"])
            response = self._respond(call, mode)
        except asyncio.CancelledError:
            self._end(call, "cancelled")
            raise
        except Exception:
            self._end(call, "raised")
            raise
        self._end(call, "returned")
        return response

    def _generate_sync(self, contents: str) -> _Response:
        current = self._current()
        mode = current["mode"]
        call = self._begin(contents, mode)
        try:
            if mode == "slow":
                time.sleep(current["sleep"])
            elif mode == "slow_within":
                time.sleep(current["sleep"])
            response = self._respond(call, mode)
        except Exception:
            self._end(call, "raised")
            raise
        self._end(call, "returned")
        return response


class _AioModels:
    def __init__(self, owner: FakeGeminiClient) -> None:
        self._owner = owner

    async def generate_content(self, *, model: str, contents: str, config: Any = None) -> _Response:
        return await self._owner._generate_async(contents)


class _Aio:
    def __init__(self, owner: FakeGeminiClient) -> None:
        self.models = _AioModels(owner)

    async def aclose(self) -> None:
        return None


class _SyncModels:
    def __init__(self, owner: FakeGeminiClient) -> None:
        self._owner = owner

    def generate_content(self, *, model: str, contents: str, config: Any = None) -> _Response:
        return self._owner._generate_sync(contents)
