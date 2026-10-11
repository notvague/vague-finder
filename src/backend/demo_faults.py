"""데모 실패 경로 확인용 모의 서버 — **화면이 장애를 어떻게 말하는가.**

발표 중에 Gemini 쿼터가 터지거나 리랭커가 죽으면 화면에 무엇이 뜨는지 아직 본 적이
없다. 실제 서버로 재현하려면 쿼터를 진짜로 태우거나 모델을 망가뜨려야 하므로,
**장애만 모의하고 나머지는 전부 실물**을 쓴다.

실물인 것 — 검색·질문 조회 라우트(`api/routes/search.py`), 응답 스키마, 실행 기록
직렬화(`schemas/explain.py`), 그리고 화면(`frontend/static/*`). 모의인 것은 질의 분석,
검색 라우터, 후보 조회(벡터 DB) **세 개의 의존성뿐**이다. 그래서 여기서 본 화면은
실제 서버가 같은 장애를 만났을 때의 화면과 같다.

모델을 올리지 않으므로 기동이 즉시 끝나고 로컬 Qdrant 폴더도 열지 않는다 —
개발 서버가 떠 있는 채로 함께 돌려도 된다. 라우트가 벡터 DB에 기대는 의존성을 새로
쓰면 여기서도 대체해야 한다(`tests/test_demo_faults.py`가 실물이 열리지 않는지 본다).

    venv/bin/python -m src.backend.demo_faults        # 8010 포트
    curl -X POST localhost:8010/__fault -d '{"mode":"rerank_fail"}'
    curl localhost:8010/__fault                       # 현재 상태와 모드 목록

모드는 요청 사이에 언제든 바꿀 수 있다. 화면을 새로 고칠 필요가 없다.
"""

from __future__ import annotations

import json
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.backend.api.dependencies import (
    get_query_analyzer,
    get_search_router,
    get_vector_client,
)
from src.backend.api.routes.search import router as search_route
from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import MatchingTrack
from src.retrieval import query_analyzer as analyzer_module
from src.retrieval.explain import (
    RERANK_APPLIED,
    RERANK_FAILED,
    ExplainRecorder,
    ScoreMix,
)

STATIC_DIR = Path(__file__).resolve().parents[1] / "frontend" / "static"

# 모드 이름 → 무엇을 재현하는가. `GET /__fault`가 그대로 돌려준다.
MODES: Dict[str, str] = {
    "ok": "정상 — 비교 기준",
    "analysis_quota": "Gemini 쿼터 초과: 3회 실패 후 규칙 폴백(confidence 0.0)으로 검색은 계속된다",
    "analysis_hang": "Gemini가 응답하지 않음: 분석이 seconds만큼 멈춘다(release로 풀 수 있다)",
    "path_image_fail": "이미지 경로가 예외로 죽음: 가중치는 남고 기여만 사라진다",
    "rerank_fail": "Cross-Encoder 리랭킹 실패: 검색 순서로 폴백",
    "empty": "결과 0건",
    "server_error": "벡터 DB에 닿지 못함: 검색도 질문 조회도 예외 → 500",
    "clarify_fail": "질문 조회만 실패(500) — 검색은 정상이라 거절만으로 다음 결과로 간다",
    "clarify_hang": "질문 조회가 seconds만큼 응답하지 않음 — 화면의 질문 조회 제한 시간 확인(release로 풀 수 있다)",
    "slow_first": "첫 요청만 seconds 지연, 이후는 즉시 — 응답 순서 뒤바뀜 재현",
}


class Fault:
    """지금 무엇을 고장 낼 것인가. 요청 사이에 바뀔 수 있어 잠금을 쓴다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.mode = "ok"
        self.seconds = 8.0
        # 실제 warmup.status() 대신 내보낼 상태. 빈 문자열이면 실물을 쓴다.
        self.warmup = ""
        self.calls = 0
        # analysis_hang·clarify_hang을 밖에서 풀어 주는 손잡이. 기다림을 끊고 다음으로 넘어간다.
        self.release = threading.Event()

    def set(self, mode: str, seconds: Optional[float], warmup: Optional[str]) -> None:
        with self._lock:
            self.mode = mode
            if seconds is not None:
                self.seconds = seconds
            if warmup is not None:
                self.warmup = warmup
            self.calls = 0
            self.release.clear()

    def peek(self) -> tuple:
        """지금 모드와 초. **세지 않는다** — 한 요청이 분석과 검색을 모두 지나므로,
        양쪽에서 세면 `slow_first`의 '첫 요청'이 분석 쪽으로 가 버린다."""
        with self._lock:
            return self.mode, self.seconds

    def take(self) -> tuple:
        """검색 한 번. (모드, 초, 몇 번째 검색인가)."""
        with self._lock:
            self.calls += 1
            return self.mode, self.seconds, self.calls


fault = Fault()


# ---------------------------------------------------------------------------
# 모의 질의 분석
# ---------------------------------------------------------------------------

def _normal_analysis(query: str) -> QueryAnalysis:
    """Gemini가 제대로 답했을 때의 모양. 세 모달리티를 모두 쓴다."""
    return QueryAnalysis(
        original_query=query,
        intent_type="mood",
        korean_tags=["잔잔한", "새벽"],
        # 이미지 경로를 실제로 켜 둔다. has_visual_clue가 False면 스키마가
        # image_english_query를 지우고 가중치를 0으로 재정규화한다 —
        # 그러면 path_image_fail로 끌 이미지 경로가 애초에 없다.
        has_visual_clue=True,
        image_english_query="a moody album cover with rain on a window",
        audio_english_query="a slow piano ballad with soft female vocals",
        confidence=0.9,
    )


class _FaultAnalyzer:
    """`QueryAnalyzer.analyze`와 같은 자리. 동기 함수인 것까지 같다.

    라우트가 `asyncio.to_thread`로 옮겨 부르므로, 여기서 자도 이벤트 루프는 막히지
    않는다 — 그 구조가 실제와 같아야 "분석이 멈춰도 다른 요청은 받는다"를
    화면에서 확인할 수 있다.
    """

    def analyze(self, query: str) -> QueryAnalysis:
        mode, seconds = fault.peek()
        if mode == "analysis_hang":
            # release가 올 때까지, 늦어도 seconds까지 기다린다.
            fault.release.wait(timeout=seconds)
            return analyzer_module._fallback(query)
        if mode == "analysis_quota":
            # 실제 analyzer는 3회 시도 사이에 1초씩 잔다(재시도 2회 = 2초).
            time.sleep(2.0)
            return analyzer_module._fallback(query)
        return _normal_analysis(query)


# ---------------------------------------------------------------------------
# 모의 검색 라우터
# ---------------------------------------------------------------------------

_songs_cache: Optional[List[Dict[str, Any]]] = None


def _songs() -> List[Dict[str, Any]]:
    """지도에 실제로 있는 곡을 쓴다 — 결과를 누르면 지도에서 찾아가져야 한다."""
    global _songs_cache
    if _songs_cache is None:
        data = json.loads((STATIC_DIR / "map_data.json").read_text(encoding="utf-8"))
        _songs_cache = data["songs"]
    return _songs_cache


def _metadata(song: Dict[str, Any]) -> Dict[str, Any]:
    """지도 데이터 한 곡 → 벡터 DB 페이로드 모양. **검색 결과와 질문 조회가 함께 쓴다.**

    둘이 다른 값을 내면 화면의 질문("남성 12 / 여성 8")이 목록에 보이는 곡들과 맞지 않는다.
    """
    return {
        "title": song.get("ft") or song.get("t", ""),
        "artist": song.get("a"),
        "genre": song.get("g"),
        "release_date": song.get("d"),
        "cover_url": song.get("c"),
        "vocal_gender": "여성" if int(str(song["id"])[-1]) % 2 else "남성",
    }


def _track(song: Dict[str, Any], score: float) -> MatchingTrack:
    return MatchingTrack(id=str(song["id"]), score=score, retrieval_score=score, **_metadata(song))


class _FaultRouter:
    """`SearchRouter.search`와 같은 호출 모양. 기록도 실제와 같은 순서로 남긴다."""

    async def search(
        self,
        analysis: QueryAnalysis,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
        candidate_k: Optional[int] = None,
        exclude_ids: Optional[List[str]] = None,
        candidate_ids_out: Optional[List[str]] = None,
        candidate_tracks_out: Optional[List[MatchingTrack]] = None,
        answers: Any = None,
        recorder: ExplainRecorder = None,  # type: ignore[assignment]
        timer: Any = None,
        **_: Any,
    ) -> List[MatchingTrack]:
        import asyncio

        mode, seconds, calls = fault.take()

        if mode == "slow_first" and calls == 1:
            await asyncio.sleep(seconds)
        if mode == "server_error":
            raise RuntimeError("벡터 DB에 연결하지 못했습니다(모의)")

        excluded = set(exclude_ids or [])
        # 질의마다 다른 곡이 나와야 한다. 같은 결과를 돌려주면 "늦게 온 응답이
        # 새 결과를 덮는가"를 아예 확인할 수 없다 — 덮어도 화면이 같다.
        offset = zlib.crc32(analysis.original_query.encode()) % 200
        songs = _songs()
        rotated = songs[offset:] + songs[:offset]
        pool = [s for s in rotated if str(s["id"]) not in excluded]
        width = candidate_k or max(top_k * 3, 30)
        # 거절이 쌓여도 후보 풀이 얇아지지 않게 — 실제 라우터와 같은 성질.
        candidates = [_track(s, 0.9 - i * 0.01) for i, s in enumerate(pool[:width])]

        if mode == "empty":
            candidates = []

        if candidate_ids_out is not None:
            candidate_ids_out.extend(t.id for t in candidates)
        if candidate_tracks_out is not None:
            candidate_tracks_out.extend(candidates)

        results = candidates[:top_k]
        self._record(mode, analysis, candidates, results, recorder, use_rerank)
        return results

    # -- 실행 기록 --------------------------------------------------------
    def _record(
        self,
        mode: str,
        analysis: QueryAnalysis,
        candidates: List[MatchingTrack],
        results: List[MatchingTrack],
        recorder: Optional[ExplainRecorder],
        use_rerank: bool,
    ) -> None:
        if recorder is None:
            return
        weights = analysis.modality_weights
        recorder.set_weights(weights.text, weights.image, weights.audio)
        if mode == "path_image_fail":
            # 실제 라우터의 `_path_result`와 같은 모양으로 남긴다.
            recorder.note_path_failed(
                "image", "RuntimeError: SigLIP2 추론 실패(모의)"
            )
        if not candidates:
            return
        recorder.set_reranker("MusicReranker")

        for rank, track in enumerate(candidates, start=1):
            recorder.path(track.id, "text_hybrid", rank, 0.0163 - rank * 0.0002,
                          f"가중치 {weights.text:.2f}")
            # path_image_fail: 이미지 경로가 예외로 죽으면 **기여가 통째로 없다.**
            # 가중치는 그대로 남는다 — 실제 라우터가 그렇게 동작한다.
            if mode != "path_image_fail" and weights.image > 0 and rank <= 8:
                recorder.path(track.id, "image", rank, 0.0091 - rank * 0.0003,
                              f"가중치 {weights.image:.2f}")
            if weights.audio > 0 and rank <= 6:
                recorder.path(track.id, "audio", rank, 0.0072 - rank * 0.0004,
                              f"가중치 {weights.audio:.2f}")
            recorder.set_fused(track.id, track.score)
            recorder.set_retrieval(track.id, track.score)

        recorder.set_rank_before([t.id for t in candidates])

        if not use_rerank:
            recorder.set_rank_after([t.id for t in results])
            return

        recorder.set_reorder_attempted(True)
        if mode == "rerank_fail":
            # 실제 라우터의 except 절과 같은 순서 — 실패를 남기고 반영을 취소한다.
            recorder.note_rerank_run(RERANK_FAILED, [])
            recorder.revoke_reorder()
            recorder.set_rerank_error("RuntimeError: CUDA out of memory(모의)")
            recorder.set_rank_after([t.id for t in results])
            return

        judged = [t.id for t in candidates]
        recorder.note_rerank_run(RERANK_APPLIED, judged)
        # 모델에 귀속되는 변화는 묶음 안에서의 이동뿐이다 — 실제 라우터와 같게
        # (후보 전체 순서, 모델이 낸 순서, 평가한 곡)으로 남긴다.
        recorder.set_group_ranks(judged, [t.id for t in results], judged)
        recorder.commit_reorder()
        recorder.set_rank_after([t.id for t in results])
        for rank, track in enumerate(results, start=1):
            ce = max(0.05, 0.99 - rank * 0.07)
            recorder.set_rerank_score(track.id, ce)
            recorder.set_score_mix(track.id, ScoreMix(
                backend="cross_encoder",
                weight=0.45,
                configured_weight=0.45,
                confidence=1.0,
                rerank_component=ce,
                retrieval_component=track.score,
                final=0.45 * ce + 0.55 * track.score,
                rerank_normalized=False,
                strategy="전체 재정렬",
                reordered=True,
            ))

    def shutdown(self) -> None:
        """실제 라우터와 같은 자리. 여기서는 할 일이 없다."""


# ---------------------------------------------------------------------------
# 모의 후보 조회 — 질문 조회(/search/clarify)가 읽는 벡터 DB 자리
# ---------------------------------------------------------------------------

_songs_by_id_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _songs_by_id() -> Dict[str, Dict[str, Any]]:
    global _songs_by_id_cache
    if _songs_by_id_cache is None:
        _songs_by_id_cache = {str(song["id"]): song for song in _songs()}
    return _songs_by_id_cache


class _FaultIndex:
    """`QdrantIndex.fetch`와 같은 호출 모양. 모의 검색이 낸 후보와 **같은 메타데이터**를 돌려준다."""

    def fetch(
        self,
        ids: Any,
        namespace: Optional[str] = None,
        fields: Optional[Any] = None,
    ) -> Dict[str, Any]:
        mode, seconds = fault.peek()  # 세지 않는다 — slow_first의 '첫 요청'은 검색 몫이다
        if mode == "clarify_hang":
            # 라우트가 스레드로 옮겨 부르므로 여기서 기다려도 다른 요청은 받는다.
            fault.release.wait(timeout=seconds)
        if mode == "server_error":
            raise RuntimeError("벡터 DB에 연결하지 못했습니다(모의)")
        if mode == "clarify_fail":
            raise RuntimeError("후보 메타데이터를 읽지 못했습니다(모의)")
        matches = []
        for song_id in dict.fromkeys(str(i) for i in ids):
            song = _songs_by_id().get(song_id)
            if song is None:
                continue
            metadata = _metadata(song)
            if fields:
                metadata = {key: metadata.get(key) for key in fields}
            matches.append({"id": song_id, "score": 0.0, "metadata": metadata})
        return {"matches": matches}


class _FaultVectorClient:
    """`get_vector_client()`가 돌려주는 것의 자리. 라우트는 `Index(name).fetch(...)`만 부른다."""

    def Index(self, name: str) -> _FaultIndex:  # noqa: N802 - 실물의 메서드 이름을 따른다
        return _FaultIndex()


# ---------------------------------------------------------------------------
# 앱
# ---------------------------------------------------------------------------

class FaultIn(BaseModel):
    mode: str = Field(..., description=f"{', '.join(MODES)} 중 하나")
    seconds: Optional[float] = Field(default=None, ge=0, le=600)
    warmup: Optional[str] = Field(
        default=None,
        description="/health가 내보낼 예열 상태. 빈 문자열이면 실물",
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Vague-Finder 실패 경로 모의 서버")
    app.include_router(search_route, prefix="/api/v1")
    app.dependency_overrides[get_query_analyzer] = _FaultAnalyzer
    app.dependency_overrides[get_search_router] = _FaultRouter
    # 질문 조회가 후보 메타데이터를 읽는 자리. 대체하지 않으면 실물이 로컬 Qdrant 폴더를 연다 —
    # 개발 서버가 떠 있으면 500이고, 없으면 이 서버가 폴더를 잡아 개발 서버·적재를 막는다.
    app.dependency_overrides[get_vector_client] = _FaultVectorClient

    @app.get("/__fault")
    async def show() -> Dict[str, Any]:
        return {"mode": fault.mode, "seconds": fault.seconds,
                "warmup": fault.warmup or "(실물)", "modes": MODES}

    @app.post("/__fault")
    async def switch(body: FaultIn) -> Dict[str, Any]:
        if body.mode not in MODES:
            return {"error": f"모르는 모드: {body.mode}", "modes": list(MODES)}
        fault.set(body.mode, body.seconds, body.warmup)
        return {"mode": fault.mode, "seconds": fault.seconds,
                "warmup": fault.warmup or "(실물)"}

    @app.post("/__fault/release")
    async def release() -> Dict[str, Any]:
        """멈춰 있는 analysis_hang·clarify_hang 요청을 지금 풀어 준다."""
        fault.release.set()
        return {"released": True}

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        from src.backend import warmup

        state = warmup.status()
        if fault.warmup:
            state = {"state": fault.warmup, "elapsed_ms": 0.0, "stages": {
                "koe5": {"ms": 6800.0, "outcome": "ok"},
                "clap": {"ms": 4100.0, "outcome": "failed: OSError: 모의"},
            }}
        return {"status": "ok", "service": "vague-finder-demo-faults", "warmup": state}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/map")
    async def map_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html",
                            headers={"Cache-Control": "no-cache"})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010, log_level="warning")
