"""v58 HTTP 검증 공용 — 실제 FastAPI 앱에 **가짜 가사 자료원**(느리게 막을 수 있음)을 꽂고, 모델·Mongo 호출을 차단한다.

v56의 `v56_runtime.block_real_models`(Gemini/Claude 클라이언트 생성 차단)를 그대로 쓰고 여기서 더하는 것:
  - `block_mongo()` — `src.common.mongodb._connect`를 예외로 바꿔 실제 Mongo에 닿지 않는다(가사·지문·어디서든)
  - `FakeLyricsSource` — 합성 가사 문서(실제 곡 아님). 제어 파일(`V58_LYRICS_CTL`)의 mode에 따라
      normal: 바로 전부 낸다 · gate: 첫 문서 뒤 gate 파일이 생길 때까지 **읽기 안에서** 막힌다(Mongo의 긴 읽기 흉내) ·
      fail: 첫 문서 전에 ConnectionError. 호출·종료·오류를 `V58_SOURCE_LOG`에 JSONL로 남긴다(문서 내용 없음)
  - `make_lyrics_service()` — 운영과 같은 `LyricsExactSearchService`(TTL·재시도 간격은 환경변수) + 가짜 자료원.
      `V58_OLD_SERVICE=1`이면 수정 **전** 코드(`git show <main>:src/retrieval/lyrics_exact_search.py`)로 같은 서비스를 만들어
      전후를 같은 HTTP 경로에서 비교한다
  - `FixedLyricsAnalyzer` — 검증 질의 하나에 고정 분석(가사 intent + verbatim 단서)을 돌려준다. 다른 질의는 예외(조용한 폴백 금지)
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v56_runtime  # noqa: E402  (모델 차단 · 가짜 리랭커 · fake_verify_vertex_setup)

logger = logging.getLogger("v58")

# 합성 단서·가사(실제 곡 아님). 기존 테스트(test_lyric_evidence)와 같은 가상 문장.
QUERY = "가사에 창문을 여는 척 웃었고가 나오는 노래"
CLUE_TEXT = "창문을 여는 척 웃었고"
TARGET_ID = "v58-target"
TARGET_LYRICS = "모래처럼 흩어진 오후의 약속은\n창문을 여는 척 웃었고!\nI FOUND  the WAY, oh\n"
SYLLABLES = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호"


class BlockedMongoCall(RuntimeError):
    """실제 Mongo 연결이 시도됐다 — 검증 조건 위반."""


def block_mongo() -> None:
    import src.common.mongodb as mongodb

    def _blocked(*_a: Any, **_k: Any) -> Any:
        raise BlockedMongoCall("v58: 실제 MongoDB 연결이 차단됐다(가짜 가사 자료원만 허용)")

    mongodb._connect = _blocked  # type: ignore[assignment]
    mongodb.get_db = _blocked  # type: ignore[assignment]
    mongodb.get_collection = _blocked  # type: ignore[assignment]
    # 가사 서비스는 공용 클라이언트가 아니라 전용 클라이언트를 만든다 — 그 자리도 막는다.
    import src.retrieval.lyrics_exact_search as lyrics_mod

    lyrics_mod._make_lyrics_mongo_client = _blocked  # type: ignore[assignment]


def verify_mongo_guard() -> str:
    import src.common.mongodb as mongodb
    import src.retrieval.lyrics_exact_search as lyrics_mod

    try:
        mongodb.get_collection("songs")
        return "NOT BLOCKED"
    except BlockedMongoCall:
        pass
    try:
        lyrics_mod._make_lyrics_mongo_client(60.0)
        return "NOT BLOCKED (lyrics client)"
    except BlockedMongoCall:
        return "blocked"


def synthetic_documents(count: int = 300, seed: int = 58, version: str = "v1") -> List[Dict[str, Any]]:
    """합성 곡 문서. `version`이 바뀌면 모든 가사 끝에 표식이 붙어 판 해시가 달라진다(교체 확인용)."""
    rng = random.Random(seed)
    docs: List[Dict[str, Any]] = []
    for index in range(count):
        lines = []
        for _ in range(8):
            lines.append("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(6, 14))))
        text = "\n".join(lines) + f"\n{version}\n"
        docs.append({
            "song_id": f"v58-{index:04d}",
            "metadata": {"title": f"합성곡 {index}", "artist": "합성 가수"},
            "lyrics_data": {"full_lyrics": text},
        })
    docs.append({
        "song_id": TARGET_ID,
        "metadata": {"title": "가상 곡", "artist": "가수A"},
        "lyrics_data": {"full_lyrics": TARGET_LYRICS + f"{version}\n"},
    })
    return docs


class FakeLyricsSource:
    """`LyricsExactSearchService(documents_source=...)` 자리에 꽂는 자료원. 제어 파일을 호출마다 읽는다."""

    def __init__(self, ctl_path: Path, log_path: Optional[Path]) -> None:
        self.ctl_path = ctl_path
        self.log_path = log_path
        self.calls = 0
        self._lock = threading.Lock()

    def _ctl(self) -> Dict[str, Any]:
        try:
            return json.loads(self.ctl_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"mode": "normal"}

    def _log(self, **payload: Any) -> None:
        if self.log_path is None:
            return
        payload.setdefault("ts", time.time())
        with self._lock, self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def __call__(self) -> Iterable[Mapping[str, Any]]:
        ctl = self._ctl()
        with self._lock:
            self.calls += 1
            call = self.calls
        mode = str(ctl.get("mode", "normal"))
        version = str(ctl.get("version", "v1"))
        docs = synthetic_documents(int(ctl.get("count", 300)), version=version)
        self._log(call=call, event="start", mode=mode, version=version, docs=len(docs), thread=threading.current_thread().name)
        return _Cursor(self, call, docs, ctl)


class _Cursor:
    def __init__(self, source: FakeLyricsSource, call: int, docs: List[Dict[str, Any]], ctl: Dict[str, Any]) -> None:
        self._source = source
        self._call = call
        self._docs = docs
        self._ctl = ctl
        self._index = 0
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        mode = str(self._ctl.get("mode", "normal"))
        if mode == "fail" and self._index == 0:
            self._source._log(call=self._call, event="error", mode=mode)
            raise ConnectionError("v58: 가짜 자료원 오류(fail 모드)")
        if mode == "gate" and self._index == int(self._ctl.get("gate_after", 1)):
            gate = Path(str(self._ctl.get("gate_file", "")))
            self._source._log(call=self._call, event="gate_wait", gate=str(gate))
            deadline = time.monotonic() + float(self._ctl.get("gate_max_seconds", 900))
            while not gate.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("v58: gate 파일이 열리지 않았다")
                time.sleep(0.1)
            self._source._log(call=self._call, event="gate_open")
        if self._index >= len(self._docs):
            self._source._log(call=self._call, event="end", yielded=self._index)
            raise StopIteration
        sleep = float(self._ctl.get("per_doc_sleep", 0.0))
        if sleep > 0:
            time.sleep(sleep)
        item = self._docs[self._index]
        self._index += 1
        return item

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._source._log(call=self._call, event="close", yielded=self._index, exhausted=self._index >= len(self._docs))


_source: Optional[FakeLyricsSource] = None


def get_source() -> FakeLyricsSource:
    global _source
    if _source is None:
        _source = FakeLyricsSource(
            Path(os.environ["V58_LYRICS_CTL"]),
            Path(os.environ["V58_SOURCE_LOG"]) if os.getenv("V58_SOURCE_LOG") else None,
        )
    return _source


def _load_old_service_class(ref: str):
    """수정 전 코드의 `LyricsExactSearchService`. 비교용 — 운영 코드가 아니다."""
    text = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{ref}:src/retrieval/lyrics_exact_search.py"], text=True)
    spec = importlib.util.spec_from_loader("lyrics_exact_search_old", loader=None)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    module.__dict__["__file__"] = f"<git:{ref}:src/retrieval/lyrics_exact_search.py>"
    sys.modules["lyrics_exact_search_old"] = module  # dataclass가 cls.__module__을 sys.modules에서 찾는다
    exec(compile(text, module.__dict__["__file__"], "exec"), module.__dict__)
    return module.LyricsExactSearchService


def make_lyrics_service():
    ttl = float(os.getenv("LYRICS_CACHE_TTL_SECONDS", "300"))
    source = get_source()
    if os.getenv("V58_OLD_SERVICE", "0") == "1":
        cls = _load_old_service_class(os.getenv("V58_OLD_REF", "origin/main"))
        service = cls(fuzzy_threshold=0.78, cache_ttl_seconds=ttl)
        service._source_documents = source  # 수정 전 코드에는 주입 자리가 없다 — 같은 방식으로 자료원만 바꾼다
        return service
    from src.retrieval.lyrics_exact_search import LyricsExactSearchService

    return LyricsExactSearchService(
        fuzzy_threshold=0.78,
        cache_ttl_seconds=ttl,
        documents_source=source,
        refresh_retry_seconds=float(os.getenv("LYRICS_REFRESH_RETRY_SECONDS", "60")),
    )


class FixedLyricsAnalyzer:
    """검증 질의 하나만 안다. `analyze_async`는 두지 않는다(라우트가 동기 분석기를 스레드로 옮기는 분기)."""

    analysis_budget_seconds = 20.0

    def analyze(self, query: str):
        from src.backend.schemas.query import LyricClue, QueryAnalysis

        if query != QUERY:
            raise RuntimeError(f"v58: 검증 질의가 아니다 — 모델 호출 없이 중단: {query!r}")
        return QueryAnalysis(
            original_query=query, intent_type="lyrics", korean_tags=["가사"],
            lyric_clues=[LyricClue(text=CLUE_TEXT, kind="verbatim", confidence=0.95)],
            image_english_query="", audio_english_query="",
            modality_weights={"text": 1.0, "image": 0.0, "audio": 0.0}, confidence=0.9,
        )


def describe_config() -> Dict[str, Any]:
    keys = ("QDRANT_URL", "QDRANT_PATH", "LYRICS_CACHE_TTL_SECONDS", "LYRICS_REFRESH_RETRY_SECONDS", "SEARCH_WARMUP",
            "FEEDBACK_COLLECTION_ENABLED", "RERANKER_BACKEND", "V58_OLD_SERVICE", "V58_OLD_REF", "V58_LYRICS_CTL",
            "V58_SOURCE_LOG", "SEARCH_TIMING_LOG")
    return {k: os.getenv(k, "") for k in keys}
