"""
[Shared] LLM Utility Module

역할:
- Gemini 기반 댓글 감성 선별 로직(`build_comment_selection_prompt`,
  `select_emotional_comments_with_llm`)을 collector(`collect_melon_data`,
  `collect_reaction`)와 정제 모듈(`refine_data`) 양쪽에서 공유하기 위해
  중립 모듈로 분리한다.

배경:
- 이전에는 collector가 `refine_data`를 import하면서 모듈 그래프가
  Step1(Melon) ↔ Step2(YouTube) ↔ Step3(LLM 정제) 한 줄로 엮였다.
- `refine_data`가 깨지면 수집 단계까지 동반 마비되는 구조였기 때문에,
  공통으로 쓰이는 댓글 선별 함수를 본 모듈로 추출하여 의존 방향을
    llm_utils ← collect_melon_data / collect_reaction / refine_data
  의 단방향으로 정리한다 (BUG-002 해소).
"""

import os
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google import genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY가 설정되지 않았습니다.")


def build_comment_selection_prompt(
        batch_comments: List[Dict[str, Any]],
        source_name: str = "YouTube"
) -> str:
    """
    댓글 배치 판별용 프롬프트
    - 음악 감성 검색용 데이터셋 구축 목적
    """

    comment_lines = []
    for idx, item in enumerate(batch_comments):
        text = item.get("text", "")
        likes = item.get("like_count", 0)
        comment_lines.append(f'[{idx}] likes={likes} | text="{text}"')

    joined = "\n".join(comment_lines)

    return f"""
You are building a dataset for a **music search engine based on emotional memory**.

The source of comments is: {source_name}

Users often remember songs like this:

- "a sad indie band song with a blue album cover"
- "a melancholic song that feels like a rainy night"
- "a song that feels like driving alone at night"
- "a quiet song that feels like winter air"

Therefore we want comments that describe **the feeling, scene, or atmosphere of the song**.

GOOD COMMENTS usually contain:

• emotions
  (슬프다, 먹먹하다, 위로된다, 설렌다)

• atmosphere / vibe
  (몽환적, 잔잔한, 따뜻한, 차분한)

• time
  (새벽, 밤, 겨울밤, 해질녘)

• weather
  (비 오는 날, 흐린 날, 눈 오는 날)

• place
  (창가, 버스, 지하철, 카페, 드라이브)

• situations
  (이별 후, 혼자 있을 때, 공부할 때, 산책할 때)

• visual imagery
  (회색 하늘, 파란 느낌, 별빛 같은 노래)

GOOD EXAMPLES:

- "비 오는 새벽에 창가에서 듣기 좋은 노래"
- "이별 후 밤에 혼자 걸으면서 듣기 좋은 곡"
- "회색 하늘 같은 분위기의 슬픈 발라드"
- "겨울밤 공기 같은 몽환적인 노래"

BAD COMMENTS:

- fan praise ("레전드", "최고", "미쳤다")
- appearance praise
- "2024년에 듣는 사람?"
- "보러 옴"
- "파트 확인"
- lyrics copy
- random jokes
- arguments

IMPORTANT:
Prefer comments that describe:
- scenes
- atmosphere
- emotions
- imagery
- weather / time / place / situations
- color-like impressions if present

If a comment only contains fan praise, jokes, viewer check-ins, meme reactions,
or meta discussion, do NOT keep it unless it still clearly describes the song's emotional scene.

TASK:
For each comment decide:

keep:
True only if the comment helps describe
the **emotion, scene, or atmosphere of the song**.

score:
5 = perfect emotional scene description
4 = strong emotional context
3 = somewhat useful
2 = weak but related
1 = barely useful
0 = useless

Return JSON ONLY.

Format:

[
  {{
    "index": 0,
    "keep": true,
    "score": 5,
    "reason": "비 오는 밤 장면과 감정이 함께 묘사됨",
    "aspects": ["emotion","weather","time","mood"]
  }}
]

Comments:
{joined}
"""


class CommentSelectionUnavailable(Exception):
    """후보 댓글이 있는데 LLM 판정을 한 건도 못 했다(쿼터 초과·장애).

    '판정 못 함'과 '쓸 댓글이 없음'은 다르다. 이 예외를 삼키고 빈 목록으로 저장하면, 원본
    댓글이 수십 개인 곡이 몇 분짜리 장애 때문에 '댓글 없는 곡'으로 영구히 굳는다. 수집 단계
    실패로 올려 다음 실행에서 다시 시도하게 한다.
    """

    def __init__(self, source_name: str, candidates: int, failed_batches: int):
        super().__init__(
            f"{source_name} 댓글 {candidates}개를 판정하지 못했다(실패 배치 {failed_batches}개)"
        )
        self.source_name = source_name
        self.candidates = candidates
        self.failed_batches = failed_batches


@dataclass
class CommentSelection:
    """선별 결과와 그 근거. texts만 쓰는 호출부는 select_emotional_comments_with_llm을 쓴다."""

    texts: List[str]
    evaluated: int = 0      # LLM이 판정한 댓글 수
    unevaluated: int = 0    # API 실패로 판정하지 못한 댓글 수
    failed_batches: int = 0

    @property
    def api_failed_entirely(self) -> bool:
        return self.evaluated == 0 and self.unevaluated > 0


# 배치 하나가 실패하면 이만큼 다시 시도한다.
BATCH_RETRIES = 2
BATCH_RETRY_DELAY_SEC = 2.0


def _evaluate_batch(client, batch: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    """배치 하나를 LLM에 보내 판정 레코드로 바꾼다. 실패는 예외로 올린다."""
    prompt = build_comment_selection_prompt(batch, source_name=source_name)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    parsed = json.loads(response.text)
    if not isinstance(parsed, list):
        raise ValueError("댓글 판정 응답이 JSON 배열이 아님")

    records: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if idx is None or isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(batch):
            continue

        original = batch[idx]
        keep = bool(item.get("keep", False))
        # score는 안전하게 0~5로 정규화
        try:
            score = int(item.get("score", 0) or 0)
        except Exception:
            score = 0
        score = max(0, min(5, score))

        records.append({
            "text": original.get("text", ""),
            "like_count": int(original.get("like_count", 0) or 0),
            "keep": keep,
            "score": score,
            "reason": str(item.get("reason", "")),
            "aspects": item.get("aspects", []),
        })
    return records


def select_emotional_comments_detailed(
    comments: List[Dict[str, Any]],
    target_count: int = 20,
    batch_size: int = 15,
    source_name: str = "YouTube"
) -> CommentSelection:
    """
    좋아요 순 후보 댓글을 배치 단위로 LLM이 평가해서
    감성 서술형 댓글을 선별한다.

    fallback 정책:
    1) keep=true 댓글을 우선 사용
    2) 부족하면 keep=false 중 score >= 3 만 추가
    3) 그래도 부족하면 score >= 2 만 추가
    4) score 0~1 은 절대 추가하지 않음
    5) **품질 기준에 맞는 댓글이 없으면 빈 목록이다.**

    예전에는 선별 결과가 비면 '안전 fallback'이라며 판정 목록에서 좋아요순으로 채웠다.
    그래서 LLM이 keep=false, score=0으로 버린 댓글이 최종 결과에 들어갔다. 평가 후
    부적합 판정을 받은 댓글은 어떤 경우에도 돌아오지 않는다.

    API 실패로 판정하지 못한 댓글은 별도로 센다(unevaluated). 평가된 댓글처럼
    돌려주지 않는다 — 판정을 못 했다는 것과 통과했다는 것은 다르다.
    target_count는 "최대" 목표라 품질이 부족하면 그보다 적게 끝난다.
    """
    if not comments:
        return CommentSelection(texts=[])

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY가 없어 댓글 LLM 선별을 수행할 수 없습니다.")
        return CommentSelection(
            texts=[c.get("text", "") for c in comments[:target_count]],
            unevaluated=len(comments),
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    kept_items: List[Dict[str, Any]] = []
    reviewed_items: List[Dict[str, Any]] = []
    unevaluated = 0
    failed_batches = 0

    # 좋아요 순으로 정렬
    comments = sorted(comments, key=lambda x: x.get("like_count", 0), reverse=True)

    total_candidates = len(comments)
    logger.info(f"[LLM COMMENT SELECT:{source_name}] total_candidates={total_candidates}")

    for start in range(0, total_candidates, batch_size):
        batch = comments[start:start + batch_size]
        if not batch:
            break

        records: Optional[List[Dict[str, Any]]] = None
        for attempt in range(BATCH_RETRIES + 1):
            try:
                records = _evaluate_batch(client, batch, source_name)
                break
            except Exception as e:
                logger.warning(
                    f"[LLM COMMENT SELECT] 배치 평가 실패 (시도 {attempt + 1}/{BATCH_RETRIES + 1}): {e}"
                )
                if attempt < BATCH_RETRIES:
                    time.sleep(BATCH_RETRY_DELAY_SEC)

        if records is None:
            # 판정하지 못한 배치. 후보 목록에 넣지 않는다 — 넣으면 판정 없이 통과한다.
            failed_batches += 1
            unevaluated += len(batch)
            continue

        reviewed_items.extend(records)
        kept_items.extend(r for r in records if r["keep"])

        logger.info(
            f"[LLM COMMENT SELECT] batch=({start}-{start + len(batch) - 1}) "
            f"kept_so_far={len(kept_items)} reviewed_so_far={len(reviewed_items)}"
        )

        # keep=true가 충분히 모였으면 중단
        if len(kept_items) >= target_count:
            break

    def _take(pool: List[Dict[str, Any]], selected: List[Dict[str, Any]], seen: set) -> None:
        for item in sorted(pool, key=lambda x: (x.get("score", 0), x.get("like_count", 0)), reverse=True):
            if len(selected) >= target_count:
                return
            if item["text"] in seen:
                continue
            selected.append(item)
            seen.add(item["text"])

    selected: List[Dict[str, Any]] = []
    seen: set = set()
    _take(kept_items, selected, seen)
    if len(selected) < target_count:
        _take([x for x in reviewed_items if x.get("score", 0) >= 3], selected, seen)
    if len(selected) < target_count:
        _take([x for x in reviewed_items if x.get("score", 0) >= 2], selected, seen)

    if unevaluated:
        logger.warning(
            f"[LLM COMMENT SELECT:{source_name}] API 실패로 판정하지 못한 댓글 {unevaluated}개 "
            f"(배치 {failed_batches}개). 판정 없이 채택하지 않는다."
        )
    if not selected:
        logger.warning(
            f"[LLM COMMENT SELECT:{source_name}] 품질 기준을 넘는 댓글이 없어 빈 목록 "
            f"(평가 {len(reviewed_items)}개, 미평가 {unevaluated}개)"
        )

    logger.info(
        f"[LLM COMMENT SELECT:{source_name}] final_selected={len(selected)} / requested={target_count}"
    )
    return CommentSelection(
        texts=[item["text"] for item in selected[:target_count]],
        evaluated=len(reviewed_items),
        unevaluated=unevaluated,
        failed_batches=failed_batches,
    )


def select_emotional_comments_with_llm(
    comments: List[Dict[str, Any]],
    target_count: int = 20,
    batch_size: int = 15,
    source_name: str = "YouTube"
) -> List[str]:
    """select_emotional_comments_detailed의 텍스트만 돌려주는 래퍼.

    후보가 있는데 판정을 한 건도 못 했으면 CommentSelectionUnavailable을 올린다. 예전에는
    이 경우도 빈 목록이라 호출부가 '쓸 댓글이 없는 곡'으로 처리했다.
    API 키가 없는 환경은 예외 없이 후보를 그대로 돌려준다(detailed가 그렇게 동작한다).
    """
    selection = select_emotional_comments_detailed(
        comments, target_count=target_count, batch_size=batch_size, source_name=source_name
    )
    if selection.api_failed_entirely and not selection.texts:
        raise CommentSelectionUnavailable(source_name, len(comments), selection.failed_batches)
    return selection.texts
