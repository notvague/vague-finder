"""평가용 사례 파일을 **격리된** 사례 저장소로 올린다. 운영 Mongo·운영 Qdrant에 쓰지 않는다.

사례 파일(JSONL)의 한 줄:

    {"case_id": "t01", "song_id": "syn017", "reusable_text": "도입부에 휘파람이 나오는 노래", "group": "target"}

`group`은 측정에서 사례를 나눠 보는 이름이다.

- `target` — 평가 질의의 정답 곡을 가리키는 사례(질의와 **따로 쓴** 단서여야 한다)
- `similar_clue_other_song` — 단서는 비슷한데 다른 곡을 가리키는 사례
- `other_clue_same_song` — 정답 곡을 가리키지만 질의와 다른 단서
- `wrong_reason` — 잘못된 사유로 승인된 사례(곡과 맞지 않는 단서)
- `broad` — "슬픈 발라드"처럼 넓은 단서

사례 벡터는 메모리 Qdrant에만 올리고, Mongo 대신 이 파일 내용으로 "승인된 최신 문서"를 돌려준다.
검색이 쓰는 조회 서비스(`FeedbackRetriever`)와 판정 함수는 운영 경로와 같은 것이다.

실제 피드백을 내보낸 파일로 측정할 때는 그 파일도 **원래 만료 시각까지만** 둔다(`expires_at` 필드). 만료가
지난 줄은 읽지 않는다. **줄에 `expires_at`이 없으면 읽기를 거부한다** — 빠뜨린 채로 받으면 원래 보관 기간을
넘겨 쓰게 된다. 합성 사례 파일만 `require_expiry=False`로 그 검사를 뺀다(실행하는 동안만 유효한 사례로 읽는다).
오래 남길 회귀 자료는 합성 사례로 만든다(`docs/feedback.md`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.feedback.case_index import FeedbackCaseIndex
from src.feedback.retrieval import FeedbackRetriever
from src.feedback.review import CaseVersion, clean_text, utcnow
from src.feedback.settings import EMBEDDING_VERSION, MAX_REUSABLE_TEXT, RetrievalSettings

CASE_GROUPS = ("target", "similar_clue_other_song", "other_clue_same_song", "wrong_reason", "broad")
# 합성 사례의 만료. 파일에 만료가 없으면 실행하는 동안만 유효하게 둔다(메모리에만 있다).
SYNTHETIC_TTL = timedelta(days=1)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    song_id: str
    reusable_text: str
    group: str
    expires_at: Optional[datetime] = None

    @property
    def feedback_id(self) -> str:
        """운영 사례 ID와 같은 모양(32자리 16진수). 파일의 case_id에서 정해진다."""
        return hashlib.sha256(f"eval-case:{self.case_id}".encode()).hexdigest()[:32]


def load_cases(path: Path, now: Optional[datetime] = None, *, require_expiry: bool = True) -> List[EvalCase]:
    """사례 파일을 읽는다. `require_expiry`는 합성 사례 파일을 읽을 때만 끈다."""
    now = now or utcnow()
    cases: List[EvalCase] = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        case_id, song_id = str(row.get("case_id", "")).strip(), str(row.get("song_id", "")).strip()
        text, group = clean_text(row.get("reusable_text")), str(row.get("group", "target")).strip()
        if not case_id or not song_id or not text:
            raise ValueError(f"{path}:{number} case_id·song_id·reusable_text가 필요하다")
        if len(text) > MAX_REUSABLE_TEXT:
            raise ValueError(f"{path}:{number} reusable_text가 {MAX_REUSABLE_TEXT}자를 넘는다")
        if group not in CASE_GROUPS:
            raise ValueError(f"{path}:{number} 알 수 없는 group: {group}")
        if case_id in seen:
            raise ValueError(f"{path}:{number} case_id가 겹친다: {case_id}")
        seen.add(case_id)
        expires = row.get("expires_at")
        if not expires and require_expiry:
            raise ValueError(
                f"{path}:{number} expires_at이 없다. 실사용 사례는 원래 만료 시각을 함께 내보낸다"
                "(합성 사례 파일이면 --synthetic-cases)"
            )
        expires_at = datetime.fromisoformat(expires) if expires else None
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at is not None and expires_at <= now:
            continue  # 보관 기간이 지난 사례는 평가에도 쓰지 않는다
        cases.append(EvalCase(case_id, song_id, text, group, expires_at))
    return cases


def cases_digest(cases: Iterable[EvalCase]) -> str:
    """사례 묶음의 지문. 두 실행이 같은 사례 판으로 돌았는지 runinfo에서 맞춰 본다."""
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda case: case.case_id):
        digest.update(f"{case.case_id}\x00{case.song_id}\x00{case.reusable_text}\x00{case.group}\x01".encode())
    return digest.hexdigest()[:16]


class CaseFileStore:
    """`FeedbackStore.fetch_cases`의 자리. 파일의 사례를 '승인된 최신 문서'로 돌려준다."""

    def __init__(self, cases: Iterable[EvalCase], now: Optional[datetime] = None):
        now = now or utcnow()
        self.documents: Dict[str, dict] = {}
        for case in cases:
            self.documents[case.feedback_id] = {
                "_id": case.feedback_id, "status": "pending", "revision": 1, "review_version": 1,
                "selected_song_id": case.song_id, "reusable_text": case.reusable_text,
                "review": {"state": "approved", "source_revision": 1, "version": 1},
                "expires_at": case.expires_at or now + SYNTHETIC_TTL, "source": "eval_file",
            }

    def fetch_cases(self, ids: Iterable[str], *, budget_seconds: Optional[float] = None) -> Dict[str, dict]:
        return {key: dict(self.documents[key]) for key in ids if key in self.documents}


@dataclass
class FileRetriever:
    retriever: FeedbackRetriever
    case_index: FeedbackCaseIndex
    cases: List[EvalCase]
    close: Callable[[], None]

    def group_of(self, feedback_id: str) -> str:
        return self._groups.get(feedback_id, "")

    def __post_init__(self) -> None:
        self._groups = {case.feedback_id: case.group for case in self.cases}
        self._case_ids = {case.feedback_id: case.case_id for case in self.cases}

    def case_id_of(self, feedback_id: str) -> str:
        return self._case_ids.get(feedback_id, "")


def build_file_retriever(
    cases: List[EvalCase],
    *,
    embedder: Any,
    song_client: Any,
    settings: Optional[Callable[[], RetrievalSettings]] = None,
    embed_batch: int = 8,
    dim: Optional[int] = None,
) -> FileRetriever:
    """사례를 KoE5로 임베딩해 메모리 Qdrant에 올리고, 곡 정보는 `song_client`(곡 색인)에서 읽는 조회 서비스.

    `dim`은 시험용이다(작은 가짜 임베딩). 주지 않으면 KoE5 차원을 쓴다.
    """
    from src.vector_db.qdrant_backend import QdrantVectorClient
    from src.vector_db.settings import TEXT_DENSE_DIM

    case_client = QdrantVectorClient(path=":memory:")
    store = CaseFileStore(cases)
    case_index = FeedbackCaseIndex(case_client, dim=dim or TEXT_DENSE_DIM)
    case_index.ensure()
    for start in range(0, len(cases), embed_batch):
        batch = cases[start:start + embed_batch]
        vectors = embedder.embed_passages([case.reusable_text for case in batch], batch_size=embed_batch, normalize=True)
        case_index.upsert(
            [(CaseVersion(case.feedback_id, case.song_id, 1, 1), vector, store.documents[case.feedback_id]["expires_at"])
             for case, vector in zip(batch, vectors)],
            embedding_version=EMBEDDING_VERSION, source="eval_file",
        )
    retriever = FeedbackRetriever(case_index, lambda: store, lambda: embedder, song_client, settings=settings)
    return FileRetriever(retriever, case_index, list(cases), case_client.close)
