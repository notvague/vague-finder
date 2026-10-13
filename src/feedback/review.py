"""검토 상태와 "재사용할 수 있는 사례"의 판정. DB·모델을 부르지 않는 순수 함수만 둔다.

검토 화면·색인 작업·검색이 **같은 함수**로 판정한다. 사본이 갈리면 색인은 넣었는데 검색은 버리거나,
검색이 철회된 사례를 쓰는 일이 생긴다.

두 가지 번호를 구분한다.

- `revision` — 사용자 제출 판. 선택·사유를 바꾸거나 철회할 때 오른다(`store.submit`).
- `review_version` — 검토 판. 운영자가 검토 결과나 재사용 문장을 저장할 때마다 오른다.
  사용자 수정은 `review`를 비우지만 이 번호는 **되돌리지 않는다** — 문서 수명 동안 단조 증가라
  같은 번호가 다른 문장을 가리키는 일이 없다.

Qdrant 포인트는 두 번호를 함께 싣는다. 검색은 Mongo의 최신 문서와 두 번호가 모두 같을 때만 쓴다.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

REVIEW_PENDING = "pending"
REVIEW_APPROVED = "approved"
REVIEW_ON_HOLD = "on_hold"
REVIEW_REJECTED = "rejected"
REVIEW_STATES = (REVIEW_PENDING, REVIEW_APPROVED, REVIEW_ON_HOLD, REVIEW_REJECTED)
# 운영자가 내린 결정. pending은 "아직 안 봤다"이고 결정이 아니다.
REVIEW_DECISIONS = (REVIEW_APPROVED, REVIEW_ON_HOLD, REVIEW_REJECTED)

REVIEW_STATE_LABELS = {
    REVIEW_PENDING: "대기",
    REVIEW_APPROVED: "승인",
    REVIEW_ON_HOLD: "보류",
    REVIEW_REJECTED: "제외",
}

# 사례별 Qdrant 포인트 ID를 만드는 이름공간. 바꾸면 기존 포인트를 찾지 못한다.
_POINT_NAMESPACE = uuid.UUID("6a1f6f0e-5b1c-4a55-9d0c-0c3a5f1f7b21")

_WHITESPACE = re.compile(r"\s+")
# 제어 문자(개행·탭 제외)는 화면과 임베딩 입력에 넣지 않는다.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(value: Any) -> str:
    """앞뒤 공백을 지우고 연속 공백·개행을 한 칸으로 줄인다. 문자열이 아니면 빈 문자열."""
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", _CONTROL.sub(" ", value)).strip()


def as_utc(value: Any) -> Optional[datetime]:
    """Mongo가 시간대 없이 돌려준 시각은 UTC로 읽는다. 시각이 아니면 None."""
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def is_expired(document: Mapping[str, Any], now: datetime) -> bool:
    """만료 시각이 없거나 읽을 수 없으면 **만료로 본다** — 보관 기한을 모르는 자료는 쓰지 않는다."""
    expires = as_utc(document.get("expires_at"))
    return expires is None or expires <= now


def review_version(document: Mapping[str, Any]) -> int:
    value = document.get("review_version")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def effective_review_state(document: Mapping[str, Any]) -> str:
    """지금 유효한 검토 상태. `review`가 없거나, 모양이 다르거나, **다른 제출 판을 본 검토**면 대기다."""
    review = document.get("review")
    if not isinstance(review, Mapping):
        return REVIEW_PENDING
    state = review.get("state")
    if state not in REVIEW_DECISIONS:
        return REVIEW_PENDING
    if review.get("source_revision") != document.get("revision"):
        return REVIEW_PENDING
    if review.get("version") != review_version(document):
        return REVIEW_PENDING
    return state


@dataclass(frozen=True)
class CaseVersion:
    """재사용할 수 있는 사례의 한 판. 색인·검색이 이 네 값으로 같은 판인지 가린다."""

    feedback_id: str
    song_id: str
    source_revision: int
    review_version: int

    def matches_payload(self, payload: Mapping[str, Any]) -> bool:
        return (
            payload.get("feedback_id") == self.feedback_id
            and str(payload.get("song_id")) == self.song_id
            and payload.get("source_revision") == self.source_revision
            and payload.get("review_version") == self.review_version
        )


def reusable_case(document: Optional[Mapping[str, Any]], now: datetime) -> Optional[CaseVersion]:
    """이 문서가 **지금** 검색에 재사용할 수 있는 사례면 그 판을, 아니면 None을 돌려준다.

    조건: 문서가 있음 · 활성 제출(철회 아님) · 미만료 · 승인 · 검토가 최신 제출 판을 봤음 ·
    검토 판 일치 · 재사용 문장과 선택 곡이 있음. 하나라도 어긋나면 쓰지 않는다.
    """
    if not document:
        return None
    if document.get("status") != "pending" or is_expired(document, now):
        return None
    if effective_review_state(document) != REVIEW_APPROVED:
        return None
    if not clean_text(document.get("reusable_text")):
        return None
    song_id = document.get("selected_song_id")
    revision = document.get("revision")
    if not song_id or not isinstance(revision, int):
        return None
    return CaseVersion(
        feedback_id=str(document["_id"]),
        song_id=str(song_id),
        source_revision=revision,
        review_version=review_version(document),
    )


def point_uuid(feedback_id: str) -> str:
    """사례의 Qdrant 포인트 ID. 사례당 하나 — 새 판을 올리면 같은 자리를 덮어쓴다.

    Qdrant는 부호 없는 정수나 UUID만 받는다. 피드백 ID(32자리 16진수)를 그대로 쓰지 않고 이름공간
    UUID로 접어, ID 형식이 바뀌어도 포인트 ID 규칙은 그대로 둔다.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, str(feedback_id)))
