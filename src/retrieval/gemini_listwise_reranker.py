"""
[Retrieval] Gemini listwise reranker 

기존 검색기가 만든 candidate_k 후보만 대상으로 Gemini가 한 번에 순위를 다시 매긴다.
- 벡터 DB·임베딩 모델을 새로 만들지 않는다.
- 후보 밖의 곡을 생성하지 않는다.
- 사용자의 기억 오류를 허용하고, 희소하고 검증 가능한 단서를 우선한다.
- 제목 구조처럼 결정론적으로 확인 가능한 단서는 별도 rescue 규칙으로 보호한다.
- 정확/음성 가사 일치가 retrieval 1위에 있는 경우 LLM이 과도하게 뒤집지 못하게 보호한다.
- 선택적으로 Gemini Google Search grounding을 이용해 OST/방송 삽입/유명 맥락 같은 외부 사실을 검증한다.
- API 실패 시 원래 retrieval 순서를 그대로 반환한다.
"""
from __future__ import annotations

import json
import logging
import asyncio
import contextlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.backend.schemas.search import ClarifyAnswer, MatchingTrack
from src.retrieval.explain import (
    RERANK_APPLIED,
    RERANK_FAILED,
    RERANK_SKIPPED,
    RerankRun,
    ScoreMix,
)
from src.common.gemini_client import gemini_configured, make_genai_client
from src.common.title_features import analyze_title_structure, base_title

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _clip_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _compact_list(values: Iterable[Any], limit: int = 8) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = _clip_text(value, 48)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
        if len(result) >= limit:
            break
    return result


_KOREAN_NUMBER = {
    "한": 1,
    "두": 2,
    "세": 3,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}


def _extract_title_shape_clue(query: str) -> dict[str, Any]:
    """원질의에서 매우 명시적인 제목 구조 단서만 보수적으로 추출한다."""
    q = query or ""
    script: Optional[str] = None
    if re.search(r"(?:알파벳|영어|영문).{0,8}(?:제목|글자|이름)", q) or re.search(
        r"(?:제목|이름).{0,8}(?:알파벳|영어|영문)", q
    ):
        script = "latin"
    elif re.search(r"(?:한자).{0,8}(?:제목|글자|이름)", q) or re.search(
        r"(?:제목|이름).{0,8}(?:한자)", q
    ):
        script = "hanja"
    elif re.search(r"(?:한글).{0,8}(?:제목|글자|이름)", q) or re.search(
        r"(?:제목|이름).{0,8}(?:한글)", q
    ):
        script = "hangul"

    char_count: Optional[int] = None
    m = re.search(r"(\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*글자", q)
    if m:
        token = m.group(1)
        try:
            char_count = int(token)
        except ValueError:
            char_count = _KOREAN_NUMBER.get(token)

    return {
        "script": script,
        "char_count": char_count,
        "strong": bool(script is not None and char_count is not None),
    }


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Search grounding 응답의 코드펜스/설명 문구를 허용해 JSON object만 추출한다."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _rare_fact_reason(query: str) -> str:
    """웹 사실검증이 유효한 희소 단서 질의만 보수적으로 감지한다."""
    if _extract_title_shape_clue(query).get("strong"):
        # 제목 구조는 결정론적 rescue가 더 안전하다.
        return ""

    q = query or ""
    if re.search(r"(?:애니|애니메이션|영화|드라마|예능|방송|광고|게임|웹툰|OST|삽입곡|삽입되어|배경음악|BGM)", q, re.IGNORECASE):
        return "media_context"

    sound_hits = len(re.findall(
        r"(?:휘파람|나레이션|내레이션|사이렌|박수|허밍|하모니카|아코디언|도입부|인트로|아웃트로|엔딩)",
        q,
        flags=re.IGNORECASE,
    ))
    role_hits = len(re.findall(
        r"(?:듀엣|랩|래퍼|피처링|남자.{0,8}랩|여자.{0,8}노래|남성.{0,8}랩|여성.{0,8}보컬)",
        q,
        flags=re.IGNORECASE,
    ))
    if sound_hits >= 1 and role_hits >= 1:
        return "performance_fact"
    return ""


_ANSWER_SLOT_LABELS = {
    "vocal_gender": "vocal gender",
    "genre": "genre",
    "type": "artist type (solo/group/duo/band)",
    "release_era": "release era",
}


def _corrections_block(answers: Optional[Sequence[ClarifyAnswer]]) -> str:
    """재질문 답변을 프롬프트 블록으로 만든다. 반영할 답이 없으면 빈 문자열.

    답변은 사용자가 결과를 보고 거절한 **뒤에** 직접 고른 값이라 최초 질의보다
    믿을 만하다. 이게 없으면 리랭커는 질의의 틀린 기억을 그대로 따른다 — dev c705는
    "남자가 부르는"을 "여성"으로 정정해 후보 2위까지 올라왔지만 Top-10 밖으로 밀렸다.
    '잘 모르겠어요'(skipped)는 정보가 없으므로 넣지 않는다.
    """
    lines = [
        f"- {_ANSWER_SLOT_LABELS.get(a.slot, a.slot)}: {a.value}"
        for a in answers or []
        if not a.skipped and a.value
    ]
    if not lines:
        return ""
    return (
        "\nUser corrections (chosen by the user AFTER rejecting earlier results; "
        "more reliable than the original query):\n"
        + "\n".join(lines)
        + "\nWhen a correction conflicts with the original query, trust the correction. "
        "Rank candidates that match the corrections above otherwise similar candidates "
        "that contradict them.\n"
    )


def _run_coroutine_blocking(coro):
    """동기 문맥에서 코루틴을 끝까지 돌린다. 이 스레드에 루프가 돌고 있으면 별도 스레드의 루프에서 돌린다.

    rerank_run은 라우터가 실행기 스레드에서 부르므로 보통 루프가 없다. 혹시 루프 안에서 직접 불리면
    asyncio.run이 거부하므로 그때만 스레드 하나를 띄운다 — 그 루프 안에서도 wait_for 취소는 그대로 동작한다.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


@dataclass(frozen=True)
class GeminiListwiseRerankerConfig:
    """기본값은 **1패스 · Google Search 끔**이다 (2026-10-09, results_v32~v34).

    2패스는 Hit@10 +1에 리랭킹 10초, Search grounding은 Hit@1·MRR을 더 올리지만 13~15초가 든다.
    1패스·끔은 리랭킹 중앙값 4.5~5.4초에 이득의 대부분(세 세트 합산 Hit@10 +17, 손실 0)을 남긴다.
    희소 사실 교차검증(rare_fact_verification)도 **끈다** (results_v35). 3,010곡 listwise 기록 전부에서 구조 규칙이
    3번 발동했고 셋 다 오답을 9위로 올렸다. 끄면 dev 116건 Hit@10 84 → 84, 외부 맥락 질의 리랭킹 15초 → 4~5초.
    GEMINI_RERANK_PASSES=2 · GEMINI_RERANK_USE_SEARCH=1 · GEMINI_RERANK_RARE_FACT_VERIFY=1로 되돌릴 수 있다.
    """
    model_name: str = "gemini-3.1-flash-lite"
    enabled: bool = True
    rerank_weight: float = 0.85
    spread_ref: float = 0.0
    max_candidates: int = 30
    max_retries: int = 3
    passes: int = 1
    low_confidence_threshold: float = 0.55
    use_search_grounding: bool = False
    search_evidence_chars: int = 7000
    rare_fact_verification: bool = False
    rare_fact_batch_size: int = 15
    rare_fact_min_support: float = 0.86
    rare_fact_min_confidence: float = 0.78
    rare_fact_min_margin: float = 0.06
    rare_fact_insert_rank: int = 9
    # 원격 호출 제한. 기본 경로가 된 뒤로는 멈춘 응답이 서버 워커를 붙잡지 않아야 한다 (PR 리뷰 P1).
    #   request_timeout_seconds: HTTP 연결·읽기 제한. SDK HttpOptions.timeout(ms)으로 넘긴다
    #   time_budget_seconds: 리랭킹 한 번의 전체 예산. 넘으면 남은 호출(검증 배치·패스)을 하지 않고
    #                        검색 순서를 돌려준다. 호출 하나의 벽시계 제한 = min(요청 제한, 남은 예산)이고
    #                        넘기면 진행 중인 HTTP 연결까지 취소하므로 최악 대기 ≈ 예산
    #   pass_reserve_seconds: 예산 중 본 패스 몫. 앞 단계(Search evidence·희소 사실 검증)는
    #                        `deadline - 몫`을 마감으로 받아, 검증 배치가 예산을 다 써서 본 패스가
    #                        0회 도는 일(PR 리뷰 3차)을 막는다. 실제 몫 = min(이 값, 요청 제한, 예산/2)
    request_timeout_seconds: float = 20.0
    time_budget_seconds: float = 30.0
    pass_reserve_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "GeminiListwiseRerankerConfig":
        weight = float(os.getenv("GEMINI_RERANK_WEIGHT", "0.85"))
        return cls(
            model_name=os.getenv(
                "GEMINI_RERANK_MODEL_NAME",
                os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite"),
            ),
            # 기존 공통 스위치(RERANKER_ENABLED)로 리랭킹을 꺼 둔 환경이 백엔드만 바꿨다고
            # 다시 켜지면 안 된다. 둘 중 하나라도 꺼져 있으면 끈다.
            enabled=_env_bool("RERANKER_ENABLED", True)
            and _env_bool("GEMINI_RERANK_ENABLED", True),
            rerank_weight=min(1.0, max(0.0, weight)),
            spread_ref=0.0,
            max_candidates=max(10, min(50, int(os.getenv("GEMINI_RERANK_MAX_CANDIDATES", "30")))),
            max_retries=max(1, min(5, int(os.getenv("GEMINI_RERANK_MAX_RETRIES", "3")))),
            passes=max(1, min(3, int(os.getenv("GEMINI_RERANK_PASSES", "1")))),
            low_confidence_threshold=min(
                0.95,
                max(0.0, float(os.getenv("GEMINI_RERANK_LOW_CONFIDENCE", "0.55"))),
            ),
            use_search_grounding=_env_bool("GEMINI_RERANK_USE_SEARCH", False),
            search_evidence_chars=max(
                2000,
                min(12000, int(os.getenv("GEMINI_RERANK_SEARCH_EVIDENCE_CHARS", "7000"))),
            ),
            rare_fact_verification=_env_bool("GEMINI_RERANK_RARE_FACT_VERIFY", False),
            rare_fact_batch_size=max(5, min(20, int(os.getenv("GEMINI_RERANK_RARE_FACT_BATCH", "15")))),
            rare_fact_min_support=min(0.99, max(0.50, float(os.getenv("GEMINI_RERANK_RARE_FACT_MIN_SUPPORT", "0.86")))),
            rare_fact_min_confidence=min(0.99, max(0.50, float(os.getenv("GEMINI_RERANK_RARE_FACT_MIN_CONFIDENCE", "0.78")))),
            rare_fact_min_margin=min(0.50, max(0.0, float(os.getenv("GEMINI_RERANK_RARE_FACT_MIN_MARGIN", "0.06")))),
            rare_fact_insert_rank=max(5, min(10, int(os.getenv("GEMINI_RERANK_RARE_FACT_INSERT_RANK", "9")))),
            request_timeout_seconds=max(1.0, float(os.getenv("GEMINI_RERANK_TIMEOUT_SECONDS", "20"))),
            time_budget_seconds=max(1.0, float(os.getenv("GEMINI_RERANK_BUDGET_SECONDS", "30"))),
            pass_reserve_seconds=max(0.0, float(os.getenv("GEMINI_RERANK_PASS_RESERVE_SECONDS", "10"))),
        )


class GeminiListwiseReranker:
    """후보 목록 전체를 비교하는 LLM 기반 listwise reranker."""

    # 앨범 커버를 보지 못하는 텍스트 리랭커라 이미지 지배 질의는 건드리지 않는다.
    # search_router.should_skip_rerank_for_image가 이 값을 본다.
    skips_image_dominant = True
    # 재질문 답변을 프롬프트에 반영한다. search_router.call_reranker가 이 값을 보고
    # answers를 넘긴다 (Cross-Encoder는 질의-곡 쌍만 보므로 받지 않는다).
    uses_clarify_answers = True

    def __init__(
        self,
        config: Optional[GeminiListwiseRerankerConfig] = None,
        api_key: Optional[str] = None,
        client: Any = None,
    ):
        self.config = config or GeminiListwiseRerankerConfig.from_env()
        # 직접 넘긴 키만 담는다. 없으면 환경(GCP_PROJECT_ID → Vertex, GEMINI_API_KEY → AI Studio)을 따른다.
        self._api_key = api_key or ""
        self._configured = gemini_configured(self._api_key)
        self._client = client
        if not self._configured and client is None:
            logger.warning(
                "[GeminiListwiseReranker] Gemini 설정 없음(GCP_PROJECT_ID·GEMINI_API_KEY) — retrieval 순서를 유지합니다."
            )

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and (self._configured or self._client is not None))

    def load(self) -> "GeminiListwiseReranker":
        return self

    @property
    def _gemini(self):
        if self._client is None:
            from google.genai import types

            # 연결·읽기 제한. 없으면 멈춘 응답을 무기한 기다리고, 스레드에 wait_for를 씌워도
            # 진행 중인 HTTP 요청은 끝나지 않는다.
            self._client = make_genai_client(
                api_key=self._api_key or None,
                http_options=types.HttpOptions(
                    timeout=int(self.config.request_timeout_seconds * 1000)
                ),
            )
        return self._client

    @staticmethod
    def _over_budget(deadline: Optional[float]) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _pass_reserve_seconds(self) -> float:
        """본 패스에 남겨 둘 예산. 앞 단계는 이만큼 이른 마감을 받는다.

        검증 배치 하나의 제한이 min(요청 제한 20초, 남은 예산)이라, 30곡(배치 2회)이면 두 배치가
        30초 예산을 다 쓰고 본 패스가 한 번도 못 돌 수 있다(PR 리뷰 3차). 기본 10초를 남기고,
        예산이 작으면 절반까지만 남긴다 — 앞 단계가 0초가 되지는 않는다.
        """
        return max(
            0.0,
            min(
                self.config.pass_reserve_seconds,
                self.config.request_timeout_seconds,
                self.config.time_budget_seconds / 2.0,
            ),
        )

    def _new_client(self):
        """호출마다 새 비동기 클라이언트. 주입된 클라이언트(테스트)가 있으면 그것을 쓴다.

        rerank_run은 동기 함수라 호출마다 이벤트 루프를 새로 연다. SDK의 비동기 클라이언트는
        연결 풀이 루프에 묶이므로 루프마다 새로 만들고 끝나면 닫는다 — 취소된 호출의 연결도 함께 닫힌다.
        """
        if self._client is not None:
            return self._client
        from google.genai import types

        return make_genai_client(
            api_key=self._api_key or None,
            http_options=types.HttpOptions(timeout=int(self.config.request_timeout_seconds * 1000)),
        )

    def _generate(self, *, contents: str, config: Any, deadline: Optional[float]):
        """Gemini 호출 한 번. **벽시계 제한**을 건다 — min(요청 제한, 남은 예산).

        `HttpOptions.timeout`은 연결·읽기 단계마다 따로 세고 읽기는 *조각 하나*를 기다리는 값이라,
        조금씩 오는 응답은 그 제한을 넘겨 계속된다(0.2초 제한 + 0.3초 예산에 0.89초 확인, PR 리뷰).
        그래서 비동기 호출을 `asyncio.wait_for`로 감싼다 — 시간이 다 되면 진행 중인 HTTP 요청이
        취소되고 연결이 닫힌다. 스레드 대기만 끊는 방식은 버려진 요청이 계속 돌아 부족하다.
        """
        remaining = (deadline - time.monotonic()) if deadline is not None else self.config.request_timeout_seconds
        if remaining <= 0:
            raise TimeoutError(f"리랭킹 시간 예산 {self.config.time_budget_seconds:.0f}초 초과")
        timeout = min(self.config.request_timeout_seconds, remaining)
        client = self._new_client()
        aio = getattr(client, "aio", None)
        if aio is None:
            # 비동기 면이 없는 클라이언트(테스트용 가짜). 동기 호출 — 취소는 못 하지만 예산 검사는 밖에서 한다.
            return client.models.generate_content(
                model=self.config.model_name, contents=contents, config=config
            )

        async def _call():
            try:
                return await asyncio.wait_for(
                    aio.models.generate_content(
                        model=self.config.model_name, contents=contents, config=config
                    ),
                    timeout=timeout,
                )
            finally:
                if client is not self._client:
                    with contextlib.suppress(Exception):
                        await aio.aclose()

        return _run_coroutine_blocking(_call())

    @staticmethod
    def _candidate_payload(track: MatchingTrack, retrieval_rank: int) -> dict[str, Any]:
        title_info = analyze_title_structure(track.title or "")
        return {
            "id": str(track.id),
            "retrieval_rank": retrieval_rank,
            "title": _clip_text(track.title, 90),
            "base_title": _clip_text(base_title(track.title or ""), 60),
            "title_script": title_info.get("script"),
            "title_char_count": title_info.get("char_count"),
            "title_word_count": title_info.get("word_count"),
            "title_contains_number": title_info.get("contains_number"),
            "artist": _clip_text(track.artist, 70),
            "album": _clip_text(track.album, 70),
            "release_date": _clip_text(track.release_date, 16),
            "genre": _clip_text(track.genre, 40),
            "vocal_gender": _clip_text(track.vocal_gender, 16),
            "artist_types": _compact_list(track.artist_types, 4),
            "search_style_summary": _clip_text(track.search_style_summary, 180),
            "lyrics_highlight": _clip_text(track.lyrics_highlight, 120),
            "lyrics_summary": _clip_text(track.lyrics_summary, 180),
            "lyric_match_type": track.lyric_match_type,
            "lyric_match_score": track.lyric_match_score,
            "sound_tags": _compact_list(track.sound_tags, 8),
            "mood_tags": _compact_list(track.mood_tags, 6),
            "vibe_tags": _compact_list(track.vibe_tags, 6),
            "relation_context_tags": _compact_list(track.relation_context_tags, 6),
            "fame": _clip_text(track.fame, 80),
        }

    @classmethod
    def _build_prompt(
        cls,
        query: str,
        tracks: Sequence[MatchingTrack],
        retrieval_rank: Optional[dict[str, int]] = None,
        external_evidence: str = "",
        answers: Optional[Sequence[ClarifyAnswer]] = None,
    ) -> str:
        rank_map = retrieval_rank or {
            str(track.id): rank for rank, track in enumerate(tracks, start=1)
        }
        candidates = [
            cls._candidate_payload(track, rank_map[str(track.id)])
            for track in tracks
        ]
        candidate_json = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
        evidence_block = ""
        if external_evidence:
            evidence_block = (
                "\nExternal factual evidence gathered with Google Search (may contain noise; "
                "use it only when it directly supports a distinctive clue):\n"
                + external_evidence
                + "\n"
            )
        return f"""\
You are the final reranker for a Korean vague-song finder.

The retrieval system has already narrowed the catalog to the candidate songs below.
Your job is ONLY to rank these given candidates for the user's memory-based query.

Important reasoning rules:
1. The user may remember one or more details incorrectly. Do NOT reject a candidate only because one generic detail (gender, era, solo/group, genre) conflicts.
2. Prioritize rare, diagnostic combinations over generic mood words. Examples: exact/phonetic lyric fragment, title shape (one letter / three Latin letters / Hanja), distinctive instrument or sound effect, featured rapper/singer roles, famous media-use context, unusual story detail.
3. title_script/title_char_count/base_title are deterministic features. When the user explicitly remembers a title shape, candidates that exactly satisfy the whole shape deserve very strong weight.
4. exact/phonetic lyric_match_type is strong direct evidence and normally outweighs generic semantic similarities.
5. Use the supplied metadata AND your general knowledge of Korean songs when useful. Media appearances or well-known song facts may not exist in metadata.
6. Treat retrieval_rank as a useful prior, not ground truth. A candidate at rank 20-30 may be the correct answer when a rare clue strongly identifies it.
7. Never invent a song and never output an id that is not in the candidate list.
8. Rank by likelihood that this is the song the user is trying to remember, not merely semantic similarity of descriptions.
9. If the query contains uncertainty markers such as '같아', '기억', '?', '아마', discount the uncertain detail.
10. Return ALL candidate ids exactly once.
{evidence_block}
User query:
{query}
{_corrections_block(answers)}
Candidates (JSON):
{candidate_json}

Return one JSON object only, with this schema:
{{
  "query_confidence": 0.0-1.0,
  "ranking": [
    {{"id":"candidate id","relevance":0.0-1.0,"reason":"short Korean reason"}}
  ]
}}

The ranking array must contain every candidate exactly once in best-to-worst order.
"""

    @staticmethod
    def _build_search_evidence_prompt(query: str, tracks: Sequence[MatchingTrack]) -> str:
        lines = []
        for rank, track in enumerate(tracks, start=1):
            lines.append(f"{track.id} | {track.artist or ''} - {track.title} | retrieval_rank={rank}")
        candidate_text = "\n".join(lines)
        return f"""\
You are a factual investigator helping a Korean vague-song finder.
Use Google Search to verify ONLY distinctive factual clues that can separate the fixed candidate songs below.
Focus especially on: title-shape facts, intro sounds/instruments, featured rapper/singer roles, famous OST/TV/animation/movie insertion, Cyworld/BGM-era associations, memorable story/context, or other unusual clues.
Do not spend time verifying generic moods such as 'sad' or 'emotional'.
The user's memory may contain one wrong detail, so prefer combinations of several rare clues.
Do not suggest songs outside the candidate list.

User query:
{query}

Fixed candidates:
{candidate_text}

Return concise Korean evidence notes. Whenever you find support for a candidate, include its candidate id and artist-title explicitly. Also mention contradictions when useful.
"""

    def _collect_search_evidence(
        self, query: str, tracks: Sequence[MatchingTrack], deadline: Optional[float] = None,
    ) -> str:
        if not self.config.use_search_grounding or not tracks:
            return ""
        if self._over_budget(deadline):
            logger.warning("[GeminiListwiseReranker] 시간 예산 초과 — Search evidence 생략")
            return ""
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
            )
            response = self._generate(
                contents=self._build_search_evidence_prompt(query, tracks),
                config=config,
                deadline=deadline,
            )
            if self._over_budget(deadline):
                logger.warning("[GeminiListwiseReranker] Search evidence가 예산 뒤에 도착 — 버린다")
                return ""
            return _clip_text(getattr(response, "text", "") or "", self.config.search_evidence_chars)
        except Exception as exc:
            # SDK/모델이 Google Search grounding을 지원하지 않아도 일반 listwise는 계속한다.
            logger.warning("[GeminiListwiseReranker] Google Search evidence 수집 실패 — 일반 listwise로 계속: %s", exc)
            return ""

    @staticmethod
    def _has_strong_surface_lyric_anchor(tracks: Sequence[MatchingTrack]) -> bool:
        if not tracks:
            return False
        top = tracks[0]
        if top.lyric_match_type == "exact":
            return True
        if top.lyric_match_type != "phonetic":
            return False
        competitors = [
            t for t in tracks[1:]
            if t.lyric_match_type in {"exact", "phonetic"}
        ]
        return bool((top.lyric_match_score or 0.0) >= 0.70 or not competitors)

    @staticmethod
    def _build_rare_fact_verification_prompt(
        query: str,
        tracks: Sequence[MatchingTrack],
        global_rank: dict[str, int],
        reason: str,
        answers: Optional[Sequence[ClarifyAnswer]] = None,
    ) -> str:
        candidates = []
        for track in tracks:
            payload = GeminiListwiseReranker._candidate_payload(
                track, global_rank[str(track.id)]
            )
            # 웹 검색 judge가 검색어를 만들기 쉽도록 이름 중심으로 간결하게 유지한다.
            candidates.append(payload)
        candidate_json = json.dumps(
            candidates, ensure_ascii=False, separators=(",", ":")
        )
        return f"""\
You are a high-precision fact verifier for a Korean vague-song finder.
Use Google Search aggressively, but ONLY to check rare, externally verifiable clues in the user's memory.

Verification mode: {reason}

Rules:
1. The candidate list is fixed. Never suggest a song outside it.
2. Ignore generic mood similarity. Verify distinctive facts such as:
   - exact media/animation/movie/TV insertion or OST usage, named scene/character/context
   - intro/outro sound effects or unusual instruments (whistling, narration, siren, etc.)
   - featured artist roles (male rap + female chorus, duet role split)
   - other unusual factual combinations explicitly stated by the user
3. Search candidate artist/title together with the rare clue. Do not infer support merely because metadata sounds similar.
4. User memory may contain one wrong generic detail. A strong verified rare fact can outweigh gender/era/group uncertainty.
5. support=0.90+ only when the rare clue is directly supported by search evidence or a clearly documented source.
6. matched_clues must contain only facts you actually verified.
7. Return every candidate in this batch exactly once.
8. Output strict JSON only.

User query:
{query}
{_corrections_block(answers)}
Candidates (JSON):
{candidate_json}

Return:
{{
  "verification_confidence": 0.0-1.0,
  "verdicts": [
    {{
      "id": "candidate id",
      "support": 0.0-1.0,
      "matched_clues": ["verified rare clue"],
      "contradictions": ["verified contradiction"],
      "evidence": "very short Korean evidence summary"
    }}
  ]
}}
"""

    @staticmethod
    def _parse_rare_fact_response(
        raw_text: str,
        valid_ids: Sequence[str],
    ) -> tuple[dict[str, dict[str, Any]], float]:
        payload = _extract_json_object(raw_text)
        valid = set(valid_ids)
        result: dict[str, dict[str, Any]] = {}
        for item in payload.get("verdicts") or []:
            if not isinstance(item, dict):
                continue
            song_id = str(item.get("id") or "").strip()
            if song_id not in valid:
                continue
            try:
                support = min(1.0, max(0.0, float(item.get("support", 0.0))))
            except (TypeError, ValueError):
                support = 0.0
            matched = _compact_list(item.get("matched_clues") or [], 6)
            contradictions = _compact_list(item.get("contradictions") or [], 4)
            result[song_id] = {
                "support": support,
                "matched_clues": matched,
                "contradictions": contradictions,
                "evidence": _clip_text(item.get("evidence"), 240),
            }
        try:
            confidence = min(
                1.0,
                max(0.0, float(payload.get("verification_confidence", 0.0))),
            )
        except (TypeError, ValueError):
            confidence = 0.0
        return result, confidence

    def _verify_rare_facts(
        self,
        query: str,
        tracks: Sequence[MatchingTrack],
        answers: Optional[Sequence[ClarifyAnswer]] = None,
        deadline: Optional[float] = None,
    ) -> tuple[dict[str, dict[str, Any]], float, str]:
        if not self.config.rare_fact_verification or not tracks:
            return {}, 0.0, ""
        reason = _rare_fact_reason(query)
        if not reason:
            return {}, 0.0, ""
        # q219 유형처럼 강한 표면 가사 단서가 이미 1위인 경우 웹 judge가 건드리지 않는다.
        if self._has_strong_surface_lyric_anchor(tracks):
            return {}, 0.0, ""

        try:
            from google.genai import types
        except Exception as exc:
            logger.warning(
                "[GeminiListwiseReranker] rare-fact verifier SDK 로딩 실패: %s", exc
            )
            return {}, 0.0, reason

        global_rank = {
            str(track.id): rank for rank, track in enumerate(tracks, start=1)
        }
        merged: dict[str, dict[str, Any]] = {}
        confidences: list[float] = []
        batch_size = self.config.rare_fact_batch_size
        for start in range(0, len(tracks), batch_size):
            if self._over_budget(deadline):
                logger.warning(
                    "[GeminiListwiseReranker] 시간 예산 초과 — rare-fact 검증 %d곡부터 생략", start + 1
                )
                break
            batch = list(tracks[start : start + batch_size])
            valid_ids = [str(t.id) for t in batch]
            prompt = self._build_rare_fact_verification_prompt(
                query, batch, global_rank, reason, answers
            )
            try:
                response = self._generate(
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        temperature=0.0,
                    ),
                    deadline=deadline,
                )
                if self._over_budget(deadline):
                    logger.warning("[GeminiListwiseReranker] rare-fact 검증 응답이 예산 뒤에 도착 — 버린다")
                    break
                verdicts, confidence = self._parse_rare_fact_response(
                    getattr(response, "text", "") or "", valid_ids
                )
                confidences.append(confidence)
                merged.update(verdicts)
            except Exception as exc:
                logger.warning(
                    "[GeminiListwiseReranker] rare-fact batch 검증 실패 (%d-%d): %s",
                    start + 1,
                    min(start + len(batch), len(tracks)),
                    exc,
                )

        confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        return merged, confidence, reason

    def _apply_rare_fact_rescue(
        self,
        original: Sequence[MatchingTrack],
        reranked: Sequence[MatchingTrack],
        verdicts: dict[str, dict[str, Any]],
        verification_confidence: float,
        reason: str,
        order_notes: Optional[List[Tuple[str, str, str]]] = None,
    ) -> list[MatchingTrack]:
        if not verdicts or verification_confidence < self.config.rare_fact_min_confidence:
            return list(reranked)

        required_clues = 1 if reason == "media_context" else 2
        eligible: list[tuple[float, str]] = []
        for track in original:
            song_id = str(track.id)
            verdict = verdicts.get(song_id) or {}
            support = float(verdict.get("support") or 0.0)
            matched = verdict.get("matched_clues") or []
            contradictions = verdict.get("contradictions") or []
            if support < self.config.rare_fact_min_support:
                continue
            if len(matched) < required_clues:
                continue
            # 강한 모순이 여러 개 확인된 후보는 rescue하지 않는다.
            if len(contradictions) >= 2 and support < 0.94:
                continue
            eligible.append((support, song_id))

        if not eligible:
            return list(reranked)
        eligible.sort(reverse=True)
        best_support, best_id = eligible[0]
        # 경쟁 상대는 승격 자격과 무관하게 **모든** 후보 중에서 고른다. 자격 통과자끼리만
        # 비교하면 0.87 vs 0.85처럼 사실상 동률인데도, 0.85가 최소 support(0.86)에서
        # 먼저 빠져 차이가 0.87 − 0으로 계산되고 승격된다.
        second_support = max(
            (
                float((verdicts.get(str(track.id)) or {}).get("support") or 0.0)
                for track in original
                if str(track.id) != best_id
            ),
            default=0.0,
        )
        if best_support - second_support < self.config.rare_fact_min_margin:
            return list(reranked)

        current = list(reranked)
        current_index = next(
            (i for i, track in enumerate(current) if str(track.id) == best_id),
            None,
        )
        if current_index is None:
            return current

        target_index = self.config.rare_fact_insert_rank - 1
        if current_index <= target_index:
            return current
        rescued = current.pop(current_index)
        current.insert(min(target_index, len(current)), rescued)
        if order_notes is not None:
            order_notes.append(
                (
                    best_id,
                    "gemini_rare_fact_rescue",
                    f"{reason} 단서 외부 검증 support={best_support:.2f} "
                    f"(확신도 {verification_confidence:.2f}) — "
                    f"모델 순서 {current_index + 1}위 → {target_index + 1}위",
                )
            )
        logger.info(
            "[GeminiListwiseReranker] rare-fact rescue: id=%s support=%.3f confidence=%.3f reason=%s -> rank %d",
            best_id,
            best_support,
            verification_confidence,
            reason,
            target_index + 1,
        )
        return current

    @staticmethod
    def _parse_response(
        raw_text: str, valid_ids: Sequence[str]
    ) -> tuple[list[str], dict[str, float], float, dict[str, str]]:
        """응답에서 순서·관련도·신뢰도와 함께 **모델이 쓴 이유 문장**을 꺼낸다.

        이유 문장은 프롬프트가 이미 요구하고 있는데(`"reason":"short Korean reason"`)
        여기서 버려 왔다. 검증된 기록은 아니지만 "모델이 무엇을 보고 그렇게 말했나"를
        확인할 때 유일한 단서라 보존한다. 확인된 근거와는 끝까지 분리해서 다룬다.
        """
        payload = json.loads(raw_text)
        ranking = payload.get("ranking") or []
        valid = set(valid_ids)
        ordered: list[str] = []
        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        seen: set[str] = set()

        for item in ranking:
            if isinstance(item, str):
                song_id = item.strip()
                relevance = None
                reason = ""
            elif isinstance(item, dict):
                song_id = str(item.get("id") or "").strip()
                relevance = item.get("relevance")
                reason = str(item.get("reason") or "").strip()
            else:
                continue

            if song_id not in valid or song_id in seen:
                continue
            seen.add(song_id)
            ordered.append(song_id)
            if reason:
                reasons[song_id] = reason
            try:
                if relevance is not None:
                    scores[song_id] = min(1.0, max(0.0, float(relevance)))
            except (TypeError, ValueError):
                pass

        ordered.extend(song_id for song_id in valid_ids if song_id not in seen)
        try:
            confidence = min(1.0, max(0.0, float(payload.get("query_confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        return ordered, scores, confidence, reasons

    @staticmethod
    def _rank_score(rank: int, n: int) -> float:
        if n <= 1:
            return 1.0
        return 1.0 - (rank - 1) / (n - 1)

    @staticmethod
    def _aggregate_passes(
        valid_ids: Sequence[str],
        pass_results: Sequence[
            tuple[Sequence[str], dict[str, float], float, dict[str, str]]
        ],
    ) -> tuple[list[str], dict[str, float], float, dict[str, list[str]]]:
        n = len(valid_ids)
        if not pass_results:
            return list(valid_ids), {}, 0.0, {}

        rank_totals = {song_id: 0.0 for song_id in valid_ids}
        relevance_totals = {song_id: 0.0 for song_id in valid_ids}
        relevance_counts = {song_id: 0 for song_id in valid_ids}
        confidences: list[float] = []
        # pass마다 다른 문장이 올 수 있다. 하나를 골라 대표로 쓰면 나머지를 숨기는
        # 셈이라 전부 남긴다 — 최종 순서는 pass 평균에서 나오기 때문이다.
        reasons: dict[str, list[str]] = {}

        for order, relevance, confidence, pass_reasons in pass_results:
            rank_map = {song_id: rank for rank, song_id in enumerate(order, start=1)}
            for song_id in valid_ids:
                rank = rank_map.get(song_id, n)
                rank_totals[song_id] += 1.0 if n <= 1 else 1.0 - (rank - 1) / (n - 1)
                if song_id in relevance:
                    relevance_totals[song_id] += relevance[song_id]
                    relevance_counts[song_id] += 1
            for song_id, text in pass_reasons.items():
                cleaned = " ".join(str(text).split())
                if cleaned and cleaned not in reasons.setdefault(song_id, []):
                    reasons[song_id].append(cleaned)
            confidences.append(confidence)

        pass_count = float(len(pass_results))
        aggregate_score: dict[str, float] = {}
        aggregate_relevance: dict[str, float] = {}
        for song_id in valid_ids:
            rank_score = rank_totals[song_id] / pass_count
            if relevance_counts[song_id]:
                rel = relevance_totals[song_id] / relevance_counts[song_id]
            else:
                rel = rank_score
            aggregate_relevance[song_id] = rel
            aggregate_score[song_id] = 0.75 * rank_score + 0.25 * rel

        original_rank = {song_id: rank for rank, song_id in enumerate(valid_ids, start=1)}
        order = sorted(
            valid_ids,
            key=lambda song_id: (-aggregate_score[song_id], original_rank[song_id]),
        )
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return order, aggregate_relevance, confidence, reasons

    def _fuse(
        self,
        tracks: Sequence[MatchingTrack],
        llm_order: Sequence[str],
        llm_relevance: dict[str, float],
        query_confidence: float,
        mixes: Optional[Dict[str, ScoreMix]] = None,
    ) -> List[MatchingTrack]:
        """모델 순서와 검색 순서를 섞어 최종 점수를 만든다.

        mixes를 넘기면 곡별 합성식을 채운다. 이 백엔드가 섞는 것은 **점수가 아니라
        순위 점수**다 — 검색 점수 원값이 아니라 검색 순위를 0~1로 환산한 값을 쓴다.
        그 차이를 남기지 않으면 Cross-Encoder와 같은 식인 것처럼 보인다.
        """
        n = len(tracks)
        retrieval_rank = {str(track.id): rank for rank, track in enumerate(tracks, start=1)}
        llm_rank = {song_id: rank for rank, song_id in enumerate(llm_order, start=1)}

        llm_weight = self.config.rerank_weight
        confidence_factor = 1.0
        if query_confidence < self.config.low_confidence_threshold:
            confidence_factor = max(
                0.35, query_confidence / max(self.config.low_confidence_threshold, 1e-6)
            )
            llm_weight *= confidence_factor

        rescored: list[MatchingTrack] = []
        for track in tracks:
            song_id = str(track.id)
            r_rank = retrieval_rank[song_id]
            l_rank = llm_rank.get(song_id, n)
            retrieval_norm = self._rank_score(r_rank, n)
            llm_rank_norm = self._rank_score(l_rank, n)
            llm_score = llm_relevance.get(song_id, llm_rank_norm)
            llm_norm = 0.70 * llm_rank_norm + 0.30 * llm_score
            final_score = (1.0 - llm_weight) * retrieval_norm + llm_weight * llm_norm
            if mixes is not None:
                mixes[song_id] = ScoreMix(
                    backend="gemini_listwise",
                    weight=float(llm_weight),
                    configured_weight=float(self.config.rerank_weight),
                    confidence=float(confidence_factor),
                    rerank_component=float(llm_norm),
                    retrieval_component=float(retrieval_norm),
                    final=float(final_score),
                    rerank_normalized=True,
                    strategy="후보 전체 재정렬",
                    reordered=True,
                    detail=(
                        f"모델 쪽 = 모델 순위 {l_rank}위 점수 70% + 관련도 30%, "
                        f"검색 쪽 = 검색 순위 {r_rank}위 점수, "
                        f"모델이 말한 질의 확신도 {query_confidence:.2f}"
                    ),
                )
            rescored.append(
                track.model_copy(
                    update={
                        "score": float(final_score),
                        "retrieval_score": float(track.retrieval_score if track.retrieval_score is not None else track.score),
                        "rerank_score": float(llm_norm),
                    }
                )
            )

        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored

    @staticmethod
    def _protect_strong_surface_lyric_anchor(
        original: Sequence[MatchingTrack],
        reranked: Sequence[MatchingTrack],
        order_notes: Optional[List[Tuple[str, str, str]]] = None,
    ) -> list[MatchingTrack]:
        """retrieval 1위의 강한 가사 표면일치를 LLM이 일반 의미 유사도로 뒤집지 못하게 보호."""
        if not original or not reranked:
            return list(reranked)
        top = original[0]
        match_type = top.lyric_match_type
        if match_type not in {"exact", "phonetic"}:
            return list(reranked)

        if match_type == "exact":
            strong = True
        else:
            score = top.lyric_match_score
            competing = [
                t for t in original[1:]
                if t.lyric_match_type in {"exact", "phonetic"}
            ]
            strong = (score is not None and score >= 0.70) or not competing

        if not strong:
            return list(reranked)

        top_id = str(top.id)
        ordered = list(reranked)
        for index, track in enumerate(ordered):
            if str(track.id) == top_id:
                # 자리를 실제로 옮겼을 때만 남긴다. 이미 1위였으면 규칙이 한 일이 없다.
                if index > 0 and order_notes is not None:
                    order_notes.append(
                        (
                            top_id,
                            "gemini_lyric_anchor_protect",
                            f"모델 순서 {index + 1}위 → 1위 (표면일치 {match_type})",
                        )
                    )
                return [track, *ordered[:index], *ordered[index + 1 :]]
        return ordered

    @staticmethod
    def _apply_title_shape_rescue(
        query: str,
        reranked: Sequence[MatchingTrack],
        top_k_boundary: int = 10,
        order_notes: Optional[List[Tuple[str, str, str]]] = None,
    ) -> list[MatchingTrack]:
        """명시적인 '영문+N글자' 같은 강한 제목 구조 완전일치 후보를 top10에 보존."""
        clue = _extract_title_shape_clue(query)
        if not clue.get("strong"):
            return list(reranked)

        def matches(track: MatchingTrack) -> bool:
            info = analyze_title_structure(track.title or "")
            return (
                info.get("script") == clue["script"]
                and info.get("char_count") == clue["char_count"]
            )

        current = list(reranked)
        top, rest = current[:top_k_boundary], current[top_k_boundary:]
        promote = [track for track in rest if matches(track)]
        if not promote:
            return current

        # 이미 경계 안에 있는 일치 곡은 제자리에 둔다. 전부 빼서 경계에 다시 넣으면
        # 1위였던 정답이 9위로 내려간다("알파벳 세 글자": TTL 1위 → 9위).
        # 경계 안의 비일치 곡을 아래에서부터 밀어내고 그 자리에 밖의 일치 곡을 올린다.
        demotable = [i for i, track in enumerate(top) if not matches(track)]
        count = min(len(promote), len(demotable))
        if count == 0:
            return current
        dropped_index = set(demotable[-count:])
        kept = [track for i, track in enumerate(top) if i not in dropped_index]
        dropped = [track for i, track in enumerate(top) if i in dropped_index]
        promoted = promote[:count]
        promoted_ids = {str(track.id) for track in promoted}
        remainder = [track for track in rest if str(track.id) not in promoted_ids]
        if order_notes is not None:
            shape = f"{clue['script']} {clue['char_count']}글자"
            for track in promoted:
                order_notes.append(
                    (
                        str(track.id),
                        "gemini_title_shape_rescue",
                        f"제목이 {shape} 구조와 완전일치 — {top_k_boundary}위 경계 안으로 이동",
                    )
                )
            # 밀려난 곡도 기록한다. 모델이 내린 것이 아니라 규칙이 자리를 뺀 것이다.
            for track in dropped:
                order_notes.append(
                    (
                        str(track.id),
                        "gemini_title_shape_demoted",
                        f"{shape} 제목 일치 곡 {len(promoted)}곡에 경계 안 자리를 내줌",
                    )
                )
        return [*kept, *promoted, *dropped, *remainder]

    def rerank(
        self,
        query: str,
        tracks: Sequence[MatchingTrack],
        top_k: int,
        answers: Optional[Sequence[ClarifyAnswer]] = None,
    ) -> List[MatchingTrack]:
        return self.rerank_run(query, tracks, top_k, answers=answers).tracks

    def rerank_run(
        self,
        query: str,
        tracks: Sequence[MatchingTrack],
        top_k: int,
        answers: Optional[Sequence[ClarifyAnswer]] = None,
    ) -> RerankRun:
        """결과와 **실행 상태**를 함께 돌려준다.

        이 백엔드는 재시도가 전부 실패해도 예외를 던지지 않고 입력 순서를 그대로
        돌려준다(아래 '최종 실패' 경로). 호출 횟수만 보면 성공과 구분되지 않으므로,
        상태를 명시적으로 알려야 설명이 거짓말을 하지 않는다.
        """
        if not tracks:
            return RerankRun([], RERANK_SKIPPED, [])
        if len(tracks) == 1 or not self.enabled:
            return RerankRun(list(tracks)[:top_k], RERANK_SKIPPED, [])

        judged = list(tracks[: self.config.max_candidates])
        tail = list(tracks[self.config.max_candidates :])
        valid_ids = [str(track.id) for track in judged]
        retrieval_rank = {song_id: rank for rank, song_id in enumerate(valid_ids, start=1)}
        pass_results: list[
            tuple[list[str], dict[str, float], float, dict[str, str]]
        ] = []
        last_error: Optional[Exception] = None
        # 리랭킹 한 번의 전체 예산. 검증 배치·패스 사이마다 확인하고, 넘으면 더 부르지 않는다.
        deadline = time.monotonic() + self.config.time_budget_seconds
        # 앞 단계(Search evidence·희소 사실 검증)는 본 패스 몫을 남긴 이른 마감을 받는다.
        # 같은 마감을 주면 검증 배치 2회가 예산을 다 써 본 패스가 0회 돌고 결과가 failed로 끝난다.
        pre_pass_deadline = deadline - self._pass_reserve_seconds()

        # 일반 Search evidence는 한 번만 수행하고 두 listwise pass가 공유한다.
        external_evidence = self._collect_search_evidence(query, judged, deadline=pre_pass_deadline)
        # 희소 사실이 있는 질의만 15곡 단위로 더 엄격하게 교차검증한다.
        rare_verdicts, rare_confidence, rare_reason = self._verify_rare_facts(
            query, judged, answers, deadline=pre_pass_deadline
        )

        for pass_index in range(self.config.passes):
            if self._over_budget(deadline):
                last_error = last_error or TimeoutError(
                    f"리랭킹 시간 예산 {self.config.time_budget_seconds:.0f}초 초과 (pass {pass_index + 1} 전)"
                )
                break
            prompt_tracks = judged if pass_index % 2 == 0 else list(reversed(judged))
            prompt = self._build_prompt(
                query,
                prompt_tracks,
                retrieval_rank,
                external_evidence=external_evidence,
                answers=answers,
            )
            correction = ""

            for attempt in range(self.config.max_retries):
                if self._over_budget(deadline):
                    last_error = last_error or TimeoutError(
                        f"리랭킹 시간 예산 {self.config.time_budget_seconds:.0f}초 초과 (pass {pass_index + 1}, 시도 {attempt + 1} 전)"
                    )
                    break
                try:
                    response = self._generate(
                        contents=prompt + correction,
                        config={
                            "response_mime_type": "application/json",
                            "temperature": 0.0,
                            "top_p": 1.0,
                        },
                        deadline=deadline,
                    )
                    if self._over_budget(deadline):
                        # 응답이 예산 뒤에 왔다. 반영하면 "30초 안에 끝낸다"는 약속이 거짓이 된다.
                        last_error = TimeoutError(
                            f"리랭킹 시간 예산 {self.config.time_budget_seconds:.0f}초 초과 (pass {pass_index + 1} 응답이 늦게 도착)"
                        )
                        break
                    pass_results.append(self._parse_response(response.text, valid_ids))
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "[GeminiListwiseReranker] pass=%d 호출/파싱 실패 (%d/%d): %s",
                        pass_index + 1,
                        attempt + 1,
                        self.config.max_retries,
                        exc,
                    )
                    correction = (
                        "\n\nYour previous output was invalid. Return strict JSON only. "
                        "Use only candidate ids from the supplied list, exactly once each."
                    )
                    if attempt + 1 < self.config.max_retries:
                        time.sleep(1)

        if pass_results and self._over_budget(deadline):
            # 최종 반영 직전 검사. 받은 결과가 있어도 예산을 넘겼으면 쓰지 않는다 — 검색 순서 + failed.
            last_error = last_error or TimeoutError(
                f"리랭킹 시간 예산 {self.config.time_budget_seconds:.0f}초 초과 (반영 전)"
            )
            pass_results = []

        if pass_results:
            order, relevance, confidence, model_reasons = self._aggregate_passes(
                valid_ids, pass_results
            )
            mixes: Dict[str, ScoreMix] = {}
            reranked = self._fuse(judged, order, relevance, confidence, mixes)

            # 여기까지가 **리랭킹 단계가 만든 순서**다. 아래 rescue 규칙들이 이 뒤에
            # 순위를 또 바꾸므로, 그 전 순서를 따로 남겨야 모델의 정렬과 규칙의
            # 이동을 구분할 수 있다. 섞어 두면 규칙이 3위로 올린 곡을 "모델이
            # 12위에서 3위로 올렸다"고 설명하게 된다.
            model_order_ids = [str(track.id) for track in reranked]
            order_notes: List[Tuple[str, str, str]] = []

            reranked = self._apply_title_shape_rescue(
                query, reranked, top_k_boundary=10, order_notes=order_notes
            )
            reranked = self._apply_rare_fact_rescue(
                judged,
                reranked,
                rare_verdicts,
                rare_confidence,
                rare_reason,
                order_notes=order_notes,
            )
            # 최종 보호막: exact/phonetic 가사 1위는 어떤 rescue보다 우선한다.
            reranked = self._protect_strong_surface_lyric_anchor(
                judged, reranked, order_notes=order_notes
            )
            # 모델이 본 것은 judged 뿐이다. tail은 평가 없이 뒤에 붙는다.
            return RerankRun(
                [*reranked, *tail][:top_k],
                RERANK_APPLIED,
                list(valid_ids),
                mixes=mixes,
                model_notes=model_reasons,
                order_notes=order_notes,
                model_order_ids=model_order_ids,
            )

        logger.warning(
            "[GeminiListwiseReranker] 최종 실패 — retrieval 순서 유지: %s",
            last_error,
        )
        return RerankRun(list(tracks)[:top_k], RERANK_FAILED, [])
