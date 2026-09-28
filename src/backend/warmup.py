"""기동 직후 한 번, 모델을 올리고 **대표 입력으로 첫 추론까지** 끝낸다.

**왜 필요한가.** 2026-09-23 측정에서 서버 기동 후 첫 검색이 17.2초, 같은 질의의
두 번째가 6.0초였다. 차이 11초는 전부 지연 로딩이다 — KoE5 6.8초, BM25 2.4초,
CE 5.0초(로딩 포함). 게다가 모달리티마다 첫 요청이 따로 있어서, 그 세션에서 처음
이미지 경로를 쓴 질의가 `image.embed`에만 5.0초를 냈다. 발표에서 이 비용은 아무
때나 한 번 튀어나온다.

**불러오기만으로는 모자라다.** `load()`는 가중치를 메모리에 올릴 뿐이고, 첫 추론에는
커널 컴파일·지연 초기화가 또 붙는다. 그래서 각 모델을 대표 입력으로 한 번 실제로
돌린다.

**Gemini는 예열하지 않는다.** 네트워크 호출이라 미리 부른다고 다음이 빨라지지
않는다. 쿼터만 쓴다.

**막지 않는다.** 백그라운드 스레드에서 돌리고 `/health`가 상태를 알려준다. 예열이
끝나기 전에 들어온 검색은 지금과 같은 값을 낼 뿐, 더 나빠지지 않는다. 기동을
막으면 `--reload` 개발이 매 저장마다 11초씩 느려진다.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# 예열용 대표 입력. 실제 질의와 같은 모양이면 된다 — 결과는 버린다.
_TEXT_PROBE = "비 오는 날 듣기 좋은 잔잔한 발라드"
_IMAGE_PROBE = "a moody album cover with rain on a window"
_AUDIO_PROBE = "a slow piano ballad with soft female vocals"
_LYRIC_PROBE = "예열용으로만 쓰는 구절"

# 상태: not_started → running → ready | partial_failure | failed | stopped | skipped
#
# **ready는 "다 올라왔다"는 뜻이어야 한다.** 한 단계가 실패했는데 ready로 끝나면,
# 발표 전에 이 값만 보고 준비됐다고 오인한다. 실패한 단계가 있으면 partial_failure다.
# 의도적으로 안 하는 것(리랭커가 꺼져 있음, 원격 리랭커)은 실패가 아니라 skipped다.
READY = "ready"
PARTIAL_FAILURE = "partial_failure"
STOPPED = "stopped"

_state: Dict[str, Any] = {"state": "not_started", "stages": {}, "elapsed_ms": 0.0}
_lock = threading.Lock()
# 종료가 걸리면 다음 단계로 넘어가지 않는다. 특히 마지막 vector_db 단계가
# 클라이언트를 닫은 뒤에 실행되면 닫힌 저장소를 다시 연다.
_stop = threading.Event()
_thread: "threading.Thread | None" = None


class SkipStage(Exception):
    """이 단계는 **할 일이 없다**. 실패가 아니다.

    리랭커가 꺼져 있거나 원격 백엔드라 예열할 로컬 모델이 없는 경우가 이것이다.
    실패로 세면 partial_failure가 되어 진짜 실패와 구분이 안 된다.
    """


def enabled() -> bool:
    """`SEARCH_WARMUP=0`이면 건너뛴다. 모델 없이 화면만 손볼 때가 있다."""
    return os.getenv("SEARCH_WARMUP", "1").strip().lower() not in {"0", "false", "no"}


def status() -> Dict[str, Any]:
    with _lock:
        return {
            "state": _state["state"],
            "elapsed_ms": round(_state["elapsed_ms"], 1),
            "stages": dict(_state["stages"]),
        }


def _run_stage(name: str, work: Callable[[], Any]) -> str:
    """한 단계. **실패해도 다음 단계로 간다.** 결과 문자열을 돌려준다.

    예열은 편의이지 전제가 아니다. Mongo가 없어서 가사 스냅샷을 못 읽는 자리에서도
    나머지 모델은 올려 두는 편이 낫고, 서버는 어차피 떠 있어야 한다. 다만 **삼키지는
    않는다** — 무엇이 실패했는지 상태에 남고 전체 결과도 ready가 아니게 된다.
    """
    started = time.perf_counter()
    try:
        work()
        outcome = "ok"
    except SkipStage as exc:
        outcome = f"skipped: {exc}"
        logger.info("[warmup] %s 건너뜀: %s", name, exc)
    except Exception as exc:
        outcome = f"failed: {type(exc).__name__}: {exc}"
        logger.warning("[warmup] %s 예열 실패(계속 진행): %s", name, exc)
    elapsed = (time.perf_counter() - started) * 1000
    with _lock:
        _state["stages"][name] = {"ms": round(elapsed, 1), "outcome": outcome}
    logger.info("[warmup] %s %.0fms %s", name, elapsed, outcome)
    return outcome


def _stages() -> List[tuple]:
    """(이름, 할 일). 여기서 임포트해야 예열을 끌 때 모델 모듈도 안 건드린다."""
    from src.backend.api.dependencies import (
        get_audio_embedder,
        get_bm25_encoder,
        get_image_embedder,
        get_lyrics_exact_search_service,
        get_reranker,
        get_search_service,
        get_text_embedder,
    )
    from src.backend.schemas.query import LyricClue
    from src.backend.schemas.search import MatchingTrack
    from src.retrieval.reranker import MusicReranker

    def koe5() -> None:
        # SearchService가 쓰는 것과 같은 호출 모양. prefix 규칙까지 같게 둔다.
        get_text_embedder().embed_passages(
            [f"query: {_TEXT_PROBE}"], add_e5_prefix=False, normalize=True
        )

    def bm25() -> None:
        encoder = get_bm25_encoder()
        if not getattr(encoder, "_is_fitted", False):
            raise RuntimeError("BM25 파라미터가 없다(artifacts/bm25_params.json)")
        encoder.encode_queries(_TEXT_PROBE)

    def siglip2() -> None:
        get_image_embedder().embed_texts([_IMAGE_PROBE], l2_normalize=True)

    def clap() -> None:
        get_audio_embedder().embed_texts([_AUDIO_PROBE], l2_normalize=True)

    def reranker() -> None:
        model = get_reranker()
        # **로컬 CE만 예열한다.** RERANKER_BACKEND=gemini_listwise면 여기서
        # rerank()를 부르는 순간 서버가 뜰 때마다 Gemini를 호출한다 — 예열로
        # 빨라지지도 않는 원격 호출에 쿼터만 쓴다. 종류를 먼저 본다.
        if not isinstance(model, MusicReranker):
            raise SkipStage(
                f"로컬 CE가 아니다({type(model).__name__}) — 원격 호출은 예열하지 않는다"
            )
        if not model.enabled:
            raise SkipStage("리랭커가 꺼져 있다(설정)")
        # load()만으로는 첫 추론 비용이 남는다. 두 곡짜리 실제 추론까지 돌린다.
        model.load()
        model.rerank(
            _TEXT_PROBE,
            [
                MatchingTrack(id="_warmup_a", score=1.0, title="예열", artist="예열"),
                MatchingTrack(id="_warmup_b", score=0.9, title="예열2", artist="예열"),
            ],
            top_k=2,
        )

    def lyrics() -> None:
        # Mongo에서 905곡 스냅샷을 읽어 정규화·3-gram까지 만들어 둔다(측정 2.5~3.1초).
        # 공개 경로를 그대로 쓴다 — 어차피 검색이 부르는 길이 이것뿐이다.
        get_lyrics_exact_search_service().search(
            [LyricClue(text=_LYRIC_PROBE, kind="verbatim", confidence=0.1)], top_k=1
        )

    def vector_db() -> None:
        # 인덱스 핸들과 첫 질의까지. 벡터는 KoE5 예열에서 만든 것과 같은 모양이면 된다.
        service = get_search_service()
        service.search_text(_TEXT_PROBE, top_k=1, alpha=0.5)

    return [
        ("koe5", koe5),
        ("bm25", bm25),
        ("siglip2", siglip2),
        ("clap", clap),
        ("reranker", reranker),
        ("lyrics", lyrics),
        ("vector_db", vector_db),
    ]


def run(reset: bool = True) -> Dict[str, Any]:
    """순서대로 예열한다. 이 함수는 블로킹이다 — 스레드에서 부른다.

    reset: 지난 종료 신호를 지우고 시작한다. **백그라운드로 띄울 때는 False다** —
        시작하는 쪽이 스레드를 만들기 전에 이미 지웠고, 여기서 또 지우면 그 사이에
        들어온 종료 신호를 스레드가 덮어써 버린다(기동 직후에 내리면 그 창이 열린다).
    """
    if not enabled():
        with _lock:
            _state["state"] = "skipped"
        logger.info("[warmup] SEARCH_WARMUP=0 — 예열을 건너뛴다")
        return status()

    # 지난 번 종료 신호가 남아 있으면 새 예열이 첫 단계에서 바로 멈춘다.
    if reset:
        _stop.clear()
    with _lock:
        _state["state"] = "running"
        _state["stages"] = {}
    started = time.perf_counter()
    logger.info("[warmup] 시작 — 모델을 올리고 첫 추론까지 돌린다")

    # 한 벌씩 차례로 올린다. 동시에 올리면 CPU를 나눠 쓰느라 전체가 더 걸리고,
    # 어느 단계가 얼마나 걸렸는지도 섞인다.
    failures: List[str] = []
    stopped = False
    try:
        for name, work in _stages():
            # 종료가 걸렸으면 여기서 멈춘다. 특히 마지막 vector_db 단계는 클라이언트가
            # 닫힌 뒤에 돌면 닫힌 저장소를 다시 연다.
            if _stop.is_set():
                logger.info("[warmup] 종료 신호 — %s부터 중단한다", name)
                stopped = True
                break
            if _run_stage(name, work).startswith("failed"):
                failures.append(name)
        state = STOPPED if stopped else (PARTIAL_FAILURE if failures else READY)
    except Exception as exc:  # 단계 목록을 만들다 실패한 경우(임포트 등)
        logger.exception("[warmup] 예열을 준비하지 못했다: %s", exc)
        state = "failed"

    if failures:
        # ready로 끝내면 발표 전에 이 값만 보고 준비됐다고 오인한다.
        logger.warning("[warmup] 실패한 단계: %s", ", ".join(failures))

    elapsed = (time.perf_counter() - started) * 1000
    with _lock:
        _state["state"] = state
        _state["elapsed_ms"] = elapsed
    logger.info("[warmup] %s — %.0fms", state, elapsed)
    return status()


def start_background() -> None:
    """기동을 막지 않고 예열을 시작한다.

    앞 예열이 아직 돌고 있으면 새로 시작하지 않는다. 시작하면서 `_stop`을 지우는데,
    그 신호는 살아 있는 앞 스레드도 보고 있다 — 지워 버리면 멈추라고 해 둔 스레드가
    다시 다음 단계로 넘어간다(lifespan을 반복하는 자리에서 드러난다).
    """
    global _thread
    if not enabled():
        with _lock:
            _state["state"] = "skipped"
        return
    if _thread is not None and _thread.is_alive():
        logger.warning("[warmup] 앞 예열이 아직 돌고 있다 — 새로 시작하지 않는다")
        return
    # 스레드를 만들기 **전에** 지운다. 안에서 지우면 그 사이에 걸린 종료 신호를
    # 놓친다 — 기동하자마자 서버를 내리는 경우가 그렇다.
    _stop.clear()
    _thread = threading.Thread(
        target=run, kwargs={"reset": False}, name="search_warmup", daemon=True
    )
    _thread.start()


def stop_and_wait(timeout: Optional[float] = None) -> bool:
    """남은 단계를 취소하고, 돌고 있는 단계가 **끝날 때까지** 기다린다.

    **벡터 클라이언트를 닫기 전에 불러야 한다.** 예열 중에 서버를 내리면 뒤쪽
    `vector_db` 단계가 종료 처리 뒤에 실행되어 닫힌 저장소를 다시 연다. 로컬
    Qdrant는 폴더를 한 번에 하나만 열 수 있으므로, 남은 핸들이 다음 기동을 막는다.

    이미 시작한 단계는 중간에 끊을 수 없다 — 모델 로딩에는 취소 지점이 없다.
    그래서 "다음 단계로 안 넘어간다 + 지금 것은 끝나기를 기다린다"가 할 수 있는
    전부이고, **기다리는 쪽에 제한을 두지 않는 것이 기본**이다. 제한을 두면 시간이
    지난 뒤에도 그 단계는 계속 돌고, 호출부는 닫아도 된다고 오해한다 — 고치려던
    문제가 그대로 돌아온다.

    timeout: 굳이 제한을 두는 호출부를 위해 남긴다. 시간 안에 못 끝내면 **False**를
        돌려주고 스레드 핸들도 **그대로 둔다**. False를 받은 쪽은 클라이언트를 닫으면
        안 된다. 핸들을 버리면 다음 기동이 살아 있는 스레드를 모르는 채로 시작한다.
    """
    global _thread
    _stop.set()
    thread = _thread
    if thread is None or not thread.is_alive():
        _thread = None
        return True
    thread.join(timeout)
    if thread.is_alive():
        logger.warning(
            "[warmup] %s초 안에 끝나지 않았다 — 아직 도는 중이다(핸들 유지)", timeout
        )
        return False
    _thread = None
    return True
