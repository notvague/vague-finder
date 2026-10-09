#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Iterable, Optional, Sequence, Any

# 이 파일을 retrieval/ 아래에 두고 실행하는 것을 기준으로
# 프로젝트 루트를 Python import 경로에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.api.dependencies import (
    get_query_analyzer,
    DEFAULT_RERANKER_BACKEND,
    get_reranker,
    get_search_router,
)
from src.backend.schemas.explain import to_track_explain
from src.eval.relaxed_metrics import top10_path, write_top10
from src.retrieval.analysis_cache import (
    AnalysisCacheError,
    analyzer_fingerprint,
    load_cache,
)
from src.retrieval.explain import (
    NULL_RECORDER,
    RERANK_APPLIED,
    RERANK_FAILED,
    RERANK_UNKNOWN,
    ExplainRecorder,
    render_ko,
)
from src.common.gemini_client import gemini_route_info
from src.retrieval import search_router as _search_router
from src.retrieval.clarify import reranker_corrections_mode, reranker_input_order_mode
from src.retrieval.gemini_listwise_reranker import type_slot_label
from src.retrieval.search_router import (
    call_reranker,
    exact_lyric_constraint_score,
    lyric_priority_rule,
    select_protected_lyric_ids,
    should_skip_rerank_for_image,
)



def _actual_backend(reranker: Any) -> str:
    """실제로 만들어진 리랭커의 백엔드 이름. 환경변수가 아니라 객체에서 읽는다."""
    name = type(reranker).__name__
    return {"GeminiListwiseReranker": "gemini_listwise", "MusicReranker": "cross_encoder"}.get(name, name)

def parse_relevant_ids(raw: str) -> set[str]:
    """여러 정답 ID는 | 로 구분한다."""
    return {part.strip() for part in (raw or "").split("|") if part.strip()}


def first_relevant_rank(result_ids: Sequence[str], relevant_ids: set[str]) -> int | None:
    for rank, song_id in enumerate(result_ids, start=1):
        if song_id in relevant_ids:
            return rank
    return None


def hit_at_k(result_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    return float(any(song_id in relevant_ids for song_id in result_ids[:k]))


def recall_at_k(result_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    found = len(set(result_ids[:k]) & relevant_ids)
    return found / len(relevant_ids)


def mrr_at_k(result_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    rank = first_relevant_rank(result_ids[:k], relevant_ids)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(result_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """정답/오답의 이진 관련도를 이용한 nDCG."""
    if not relevant_ids:
        return 0.0

    dcg = 0.0
    for index, song_id in enumerate(result_ids[:k]):
        if song_id in relevant_ids:
            dcg += 1.0 / math.log2(index + 2)

    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return 0.0 if idcg == 0 else dcg / idcg


def metric_bundle(result_ids: Sequence[str], relevant_ids: set[str], top_k: int) -> dict[str, float | int | None]:
    return {
        "rank": first_relevant_rank(result_ids[:top_k], relevant_ids),
        "hit1": hit_at_k(result_ids, relevant_ids, 1),
        "hit5": hit_at_k(result_ids, relevant_ids, min(5, top_k)),
        "hit10": hit_at_k(result_ids, relevant_ids, min(10, top_k)),
        "recall10": recall_at_k(result_ids, relevant_ids, min(10, top_k)),
        "mrr10": mrr_at_k(result_ids, relevant_ids, min(10, top_k)),
        "ndcg10": ndcg_at_k(result_ids, relevant_ids, min(10, top_k)),
    }


def mean(rows: Iterable[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def rerank_preserving_exact_lyrics(
    reranker,
    analysis,
    candidates,
    top_k: int,
    answers=None,
    recorder=NULL_RECORDER,
    errors=None,
    statuses=None,
):
    """실서비스와 동일하게 이미지 지배/가사 exact 보호 규칙을 적용한다.

    answers는 재질문 답변이다. 답변을 쓰는 리랭커에만 전달된다(call_reranker).

    recorder는 SearchRouter.search()와 **같은 것을 같은 순서로** 기록한다. 이 함수는
    라우터의 보호 로직을 평가용으로 옮겨 놓은 사본이라, 기록도 함께 옮겨야 평가
    결과의 설명이 실서비스와 같은 의미를 갖는다. 라우터 쪽을 고치면 여기도 고쳐야 한다.

    statuses에 리스트를 넘기면 백엔드가 보고한 실행 상태가 담긴다. **기록기와는
    별개의 통로다** — Gemini는 예외 없이 실패 상태를 돌려주므로, 설명 기록을 끄면
    그 실패가 아무 데도 남지 않는다. 그러면 측정에서 실패가 "리랭킹 효과 없음"으로
    조용히 섞인다.

    errors에 리스트를 넘기면 폴백 사유가 담긴다. 예외를 그대로 올리지 않는 이유는
    실서비스가 그렇게 하지 않기 때문이다 — 라우터는 리랭킹 실패를 잡아 기존 검색
    순서로 돌아간다. 평가만 예외로 멈추면 질의 하나의 추론 실패로 **측정 전체가
    중단되고 요약 파일도 생성되지 않는다.**
    """
    if should_skip_rerank_for_image(reranker, analysis):
        # 시도 자체가 없었다. attempted를 올리면 "실패"로 읽힌다.
        return list(candidates)[:top_k]

    recorder.set_reorder_attempted(True)
    try:
        return _rerank_with_lyric_protection(
            reranker, analysis, candidates, top_k, answers, recorder, statuses
        )
    except Exception as exc:
        # 라우터와 같은 순서로 취소한다 — 잠정 기록을 남기면 반영되지 않은 처리가
        # 설명에 남는다.
        recorder.note_rerank_run(RERANK_FAILED, [])
        recorder.revoke_reorder()
        message = f"{type(exc).__name__}: {exc}"
        recorder.set_rerank_error(message)
        if errors is not None:
            errors.append(message)
        print(f"  경고: 리랭킹 실패 — 기존 검색 순서로 폴백 ({message})", flush=True)
        final = list(candidates)[:top_k]
        recorder.set_rank_after([track.id for track in final])
        return final


def _rerank_with_lyric_protection(
    reranker, analysis, candidates, top_k: int, answers, recorder, statuses=None
):
    """보호 규칙 본체. 예외는 호출부가 폴백으로 처리한다."""
    # 보호 판정은 라우터와 **같은 함수**로 한다. 규칙을 바꿀 때 한쪽만 고치면
    # 평가와 서비스가 다른 규칙으로 동작한다.
    protected_ids = select_protected_lyric_ids(candidates)
    protected = [track for track in candidates if track.id in protected_ids]
    if not protected:
        reranked = call_reranker(
            reranker, analysis.original_query, candidates, top_k, answers, recorder,
            runs_out=statuses,
        )
        recorder.commit_reorder()
        recorder.set_rank_after([track.id for track in reranked])
        for track in reranked:
            recorder.set_rerank_score(track.id, track.rerank_score)
        return reranked

    others = [track for track in candidates if track not in protected]
    tiered = {}
    for track in protected:
        tiered.setdefault(exact_lyric_constraint_score(analysis, track), []).append(track)
    ordered_protected = []
    for tier in sorted(tiered, reverse=True):
        group = tiered[tier]
        slots = top_k - len(ordered_protected)
        if slots <= 0:
            break
        if len(group) > 1:
            group = call_reranker(
                reranker,
                analysis.original_query,
                group,
                min(slots, len(group)),
                answers,
                recorder,
                runs_out=statuses,
            )
        ordered_protected.extend(group[:slots])
    protected = ordered_protected
    remaining = max(0, top_k - len(protected))
    reranked_others = (
        call_reranker(
            reranker,
            analysis.original_query,
            others,
            remaining,
            answers,
            recorder,
            runs_out=statuses,
        )
        if remaining and others
        else []
    )
    final = [*protected, *reranked_others][:top_k]
    # 보호 배치가 실제로 성사된 뒤에만 기록한다(라우터와 동일).
    final_ids = {track.id for track in final}
    for track in protected:
        if track.id in final_ids:
            recorder.order(
                track.id,
                lyric_priority_rule(track),
                "일반 후보 아래로 내려가지 않음",
            )
    recorder.note_behind_protected(
        [track.id for track in reranked_others if track.id in final_ids],
        len(protected),
    )
    recorder.note_order_rules_applied()
    recorder.commit_reorder()
    recorder.set_rank_after([track.id for track in final])
    for track in final:
        recorder.set_rerank_score(track.id, track.rerank_score)
    return final


def read_queries(path: Path, split: str | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"query_id", "split", "query_type", "query", "relevant_ids"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"평가 CSV에 필요한 열이 없습니다: {sorted(missing)}")

    selected = []
    for row in rows:
        if row["query_type"].strip().lower() != "search":
            continue
        if split and row["split"].strip().lower() != split.lower():
            continue
        if not parse_relevant_ids(row["relevant_ids"]):
            continue
        selected.append(row)

    if not selected:
        raise ValueError("평가할 search 질의가 없습니다. split과 relevant_ids를 확인하세요.")
    return selected


def write_detail_checkpoint(path: Path, rows: list[dict]) -> None:
    """질의 1건이 끝날 때마다 detail CSV를 원자적으로 갱신한다."""
    if not rows:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


# 질의 한 건의 리랭킹이 어떻게 끝났는가.
OUTCOME_EXCEPTION = "exception"            # 예외 → 검색 순서로 폴백
OUTCOME_FALLBACK = "fallback"              # 예외 없이 실패, 반영 0건
OUTCOME_PARTIAL_FAILURE = "partial_failure"  # 일부 그룹만 실패
OUTCOME_APPLIED = "applied"                # 모델이 순서를 정했다
OUTCOME_SKIPPED = "skipped"                # 부를 조건이 아니었다(정상)
OUTCOME_UNKNOWN = "unknown"                # 상태를 보고하지 않는 백엔드


def rerank_outcome(statuses, rerank_error: str) -> str:
    """리랭킹 결말 한 단어. **실패와 정상 건너뜀을 섞지 않는다.**

    `skipped`를 실패로 세면 안 된다 — 후보가 1곡뿐이거나 리랭킹이 꺼져 있으면
    백엔드는 정상적으로 호출을 생략하고 `skipped`를 돌려준다. 그것을 폴백으로
    집계하면 아무 문제 없는 질의가 실패 목록에 올라간다.

    `unknown`도 실패가 아니다. 상태를 보고하지 않는 백엔드라 **모른다**는 뜻이고,
    모르는 것을 실패로 단정하면 이 기록의 원칙이 무너진다.
    """
    if rerank_error:
        return OUTCOME_EXCEPTION
    failed = any(status == RERANK_FAILED for status in statuses)
    applied = any(status == RERANK_APPLIED for status in statuses)
    if failed and applied:
        return OUTCOME_PARTIAL_FAILURE
    if failed:
        return OUTCOME_FALLBACK
    if applied:
        return OUTCOME_APPLIED
    if any(status == RERANK_UNKNOWN for status in statuses):
        return OUTCOME_UNKNOWN
    return OUTCOME_SKIPPED


def rerank_fell_back(statuses, rerank_error: str) -> bool:
    """최종 순서에 리랭킹 결과가 **하나도** 반영되지 않았는가.

    예외만 보면 안 된다. Gemini는 재시도가 전부 실패해도 예외를 던지지 않고 입력
    순서를 그대로 돌려준다 — 호출은 있었고 예외는 없었지만 리랭킹은 없었다.
    그것을 집계에서 빠뜨리면 실패가 "리랭킹 효과 없음"으로 조용히 섞인다.
    """
    return rerank_outcome(statuses, rerank_error) in {
        OUTCOME_EXCEPTION,
        OUTCOME_FALLBACK,
    }


def rerank_partially_failed(statuses, rerank_error: str) -> bool:
    """일부 호출만 실패했는가. 보호 그룹이 여럿일 때 생긴다."""
    return rerank_outcome(statuses, rerank_error) == OUTCOME_PARTIAL_FAILURE


def collect_run_info(args, *, reranker, analysis_source: dict) -> dict:
    """이 측정이 **무엇을 대상으로** 돌았는지 남긴다.

    숫자만 남기면 나중에 비교할 수 없다. 905곡 측정과 952곡 측정을 구분할 수 없어서
    이미 한 번 혼선이 있었다. 코퍼스·인덱스·BM25·리랭커 설정을 같이 적어 둔다.
    """
    from src.vector_db import settings as vsettings

    bm25_path = Path(os.getenv("BM25_PARAMS_PATH", "artifacts/bm25_params.json"))
    bm25 = {"path": str(bm25_path), "exists": bm25_path.exists()}
    if bm25_path.exists():
        bm25["sha256_12"] = hashlib.sha256(bm25_path.read_bytes()).hexdigest()[:12]
        bm25["bytes"] = bm25_path.stat().st_size

    reranker_config = {}
    config = getattr(reranker, "config", None)
    if config is not None:
        reranker_config = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        } or dataclasses.asdict(config)

    info = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": {k: v for k, v in vars(args).items()},
        "analysis": analysis_source,
        "corpus": {
            "vector_backend": "qdrant",
            "namespace": vsettings.NAMESPACE,
            "text_index": vsettings.TEXT_HYBRID_INDEX_NAME,
            "image_index": vsettings.IMAGE_INDEX_NAME,
            "audio_index": vsettings.AUDIO_INDEX_NAME,
            "qdrant_path": os.getenv("QDRANT_PATH", ""),
            "point_count": None,
        },
        "bm25": bm25,
        # 가사 표면 검색은 Qdrant가 아니라 **MongoDB의 full_lyrics를 직접 읽는다**
        # (`lyrics_exact_search.py`). 같은 분석 캐시·같은 인덱스로 돌려도 중간에
        # 가사 데이터가 바뀌면 phonetic·exact 후보가 달라진다. 후속 실험에서
        # 조건 간 비교가 성립하려면 이것도 고정돼 있어야 하므로 지문을 남긴다.
        "lyrics_source": _lyrics_fingerprint(),
        "reranker": {
            # 요청(환경변수)과 실제 선택을 구분해 적는다 — 키가 없어 CE로 내려가거나
            # 기본값이 바뀌면 환경변수만으로는 무엇이 돌았는지 알 수 없다 (PR 리뷰 P2).
            "backend": _actual_backend(reranker),
            "requested_backend": os.getenv("RERANKER_BACKEND", DEFAULT_RERANKER_BACKEND),
            "class": type(reranker).__name__,
            "enabled": bool(getattr(reranker, "enabled", False)),
            "config": reranker_config,
        },
        # Gemini를 어느 경로로 불렀는지(vertex·api_key). 같은 모델이어도 경로가 바뀌면 출력이 달라질 수 있다.
        "gemini": gemini_route_info(),
        # 실험 스위치. **실제로 적용된 값을 코드에서 읽는다** — 환경변수를 다시
        # 읽으면 기록과 적용이 갈릴 수 있고, 폴더명과 수기 문서에만 의존하면
        # 나중에 어느 조건의 결과인지 알 수 없다.
        "ranking_switches": _ranking_switches(config),
        "search_reference_year_env": os.getenv("SEARCH_REFERENCE_YEAR", "").strip(),
    }

    # 코퍼스 규모는 최선 노력으로 센다. 실패해도 측정을 막을 이유는 없지만,
    # 왜 못 셌는지는 남겨야 나중에 "몇 곡 기준이었나"를 추측하지 않는다.
    try:
        info["corpus"]["point_count"] = _count_points(vsettings)
    except Exception as exc:  # noqa: BLE001 - 기록용이므로 사유만 남긴다
        info["corpus"]["point_count_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _ranking_switches(config: Any) -> dict:
    """순위를 바꾸는 스위치의 **실제 적용값**. 환경 변수 문자열이 아니라 search_router가 읽는 함수를 부른다.

    가산 배율 세 개(`BOOST_SCALE`·`ATTR_BOOST_SCALE`·`GENDER_MISMATCH_SCALE`, PR #16~#18)가 빠져 있어
    v27~v31의 runinfo로는 조건을 구분할 수 없었다(PR #16~#22 리뷰). 기본값이어도 적는다.
    """
    return {
        "boost_scale": _search_router._explicit_boost_scale(),
        "attr_boost_scale": _search_router._attr_boost_scale(),
        "gender_mismatch_scale": _search_router._gender_mismatch_scale(),
        "lyric_protect_phonetic_top1": _search_router._protect_phonetic_top1(),
        "lyric_protect_min_confidence": _search_router._lyric_protect_min_confidence(),
        "lyric_boost_scale": _search_router._lyric_boost_scale(),
        "reranker_min_spread": float(getattr(config, "min_spread", 0.0) or 0.0)
        if config is not None
        else None,
        # 재질문 답변이 LLM 리랭커 프롬프트에 들어가는 방식 (results_clarify_v10_corrections 리뷰)
        "gemini_rerank_corrections": reranker_corrections_mode(),
        "gemini_rerank_type_slot_label": type_slot_label(),
        "clarify_rerank_input_order": reranker_input_order_mode(),
    }


def _lyrics_fingerprint() -> dict:
    """가사 표면 검색이 읽는 MongoDB 데이터의 지문.

    **내용 해시를 쓴다.** 문서 수와 길이 합만으로는 부족하다 — "너를 사랑해"와
    "너를 미워해"는 길이가 같다. 곡 ID와 가사를 묶어 해시하므로 한 글자만 바뀌어도
    값이 달라진다. 곡 순서에 흔들리지 않게 ID로 정렬한 뒤 계산한다.

    길이 합도 함께 남긴다 — 해시가 달라졌을 때 얼마나 달라졌는지 가늠할 수 있다.
    """
    info = {
        "db": os.getenv("MONGO_DB_NAME", "vaguefinder"),
        "collection": "songs",
        # URI는 자격증명이 들어 있을 수 있으므로 호스트도 남기지 않는다.
        "document_count": None,
        "full_lyrics_songs": None,
        "full_lyrics_total_chars": None,
        "content_sha256_16": None,
    }
    try:
        from src.common.mongodb import get_collection

        rows = []
        total = 0
        with_lyrics = 0
        for document in get_collection("songs").find(
            {}, {"song_id": 1, "lyrics_data.full_lyrics": 1}
        ):
            text = (document.get("lyrics_data") or {}).get("full_lyrics") or ""
            rows.append((str(document.get("song_id") or ""), text))
            total += len(text)
            with_lyrics += 1 if text else 0

        digest = hashlib.sha256()
        for song_id, text in sorted(rows):
            digest.update(song_id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(text.encode("utf-8"))
            digest.update(b"\x01")

        info["document_count"] = len(rows)
        info["full_lyrics_songs"] = with_lyrics
        info["full_lyrics_total_chars"] = total
        info["content_sha256_16"] = digest.hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001 - 기록용이므로 사유만 남긴다
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def candidate_digest(candidates, reranker=None) -> str:
    """리랭커에 **실제로 들어간 입력**의 지문.

    id·순서·검색 점수뿐 아니라 **Cross-Encoder가 읽는 곡 설명 문서**까지 해시한다.
    id와 점수가 같아도 메타데이터가 바뀌면 CE는 다른 문서를 보고 다른 점수를 낸다 —
    그러면 조건 간 비교가 성립하지 않는데 지문은 같아 보인다.

    detail CSV의 `baseline_top_ids`는 상위 10개뿐이고 `candidate_rank@30`은 정답
    위치만 나타내므로 둘 다 이 용도로는 부족하다.

    문서를 만들 수 없는 백엔드(Gemini는 프롬프트를 따로 만든다)면 id·점수까지만
    해시하고 접두사로 그 사실을 남긴다 — 범위를 숨기지 않기 위해서다.
    """
    build = getattr(reranker, "build_document", None)
    digest = hashlib.sha256()
    for track in candidates:
        digest.update(f"{track.id}:{float(track.score):.17g}".encode("utf-8"))
        digest.update(b"\x00")
        if build is not None:
            digest.update(build(track).encode("utf-8"))
        digest.update(b"\x01")
    prefix = "d" if build is not None else "s"   # document / score-only
    return f"{prefix}{digest.hexdigest()[:15]}"


def _count_points(vsettings) -> Optional[int]:
    """텍스트 인덱스의 벡터 수. 952곡인지 905곡인지를 사후에 확인하기 위한 값.

    **검색이 쓰는 클라이언트를 그대로 재사용한다.** 로컬 Qdrant는 저장 폴더를 한
    프로세스에서 하나만 열 수 있어서, 세는 용도로 새 클라이언트를 만들면
    "already accessed by another instance"로 실패한다.
    """
    from src.backend.api.dependencies import get_vector_client

    from src.vector_db.qdrant_backend import collection_name

    client = get_vector_client()
    name = collection_name(vsettings.TEXT_HYBRID_INDEX_NAME, vsettings.NAMESPACE)
    return int(client.client.count(collection_name=name, exact=True).count)


def explain_columns(
    record, *, enabled: bool, rerank_error: str = "", statuses=None
) -> dict:
    """detail CSV의 실행 상태 열.

    **백엔드가 보고한 상태(`rerank_run_statuses`)와 폴백 사유는 기록 여부와 무관하게
    남긴다.** 리랭킹이 실제로 일어났는지는 설명 기능의 부산물이 아니라 측정 결과를
    읽는 데 필요한 사실이다.

    나머지 열은 기록을 끄면 빈 값으로 남긴다. NULL_RECORDER의 빈 기록은 "리랭킹이
    없었다"가 아니라 **"기록하지 않았다"**인데, 그대로 저장하면 실제로 리랭킹이
    돌아간 실행에 not_attempted / 호출 0회가 남는다.
    """
    statuses = list(statuses or [])
    outcome = rerank_outcome(statuses, rerank_error)
    always = {
        "rerank_run_statuses": "|".join(statuses),
        "rerank_outcome": outcome,
        "rerank_error": rerank_error,
        "rerank_fell_back": int(outcome in {OUTCOME_EXCEPTION, OUTCOME_FALLBACK}),
        "rerank_partial_failure": int(outcome == OUTCOME_PARTIAL_FAILURE),
    }
    if not enabled:
        return {
            "reorder_stage": "",
            "rerank_calls": "",
            "rerank_calls_applied": "",
            **always,
        }
    return {
        "reorder_stage": record.reorder_stage,
        "rerank_calls": len(record.rerank_runs),
        "rerank_calls_applied": record.rerank_runs_applied,
        **always,
        "rerank_error": rerank_error or record.rerank_error,
    }


def build_explain_payload(
    *,
    query_id: str,
    split: str,
    query: str,
    relevant_ids: set[str],
    record,
    ordered,
    candidates,
    top_k: int,
) -> dict:
    """질의 한 건의 실행 기록을 저장 형태로 만든다.

    결과 곡만 담지 않고 **정답 곡도 함께** 담는다. 재측정에서 알아야 하는 것은
    "1위가 왜 1위인가"보다 "정답이 왜 그 자리인가"이고, 그 둘은 다른 곡이다.

    정답이 후보 풀에 아예 없으면 기록이 없다 — 그것 자체가 가장 중요한 구분이다.
    검색이 못 찾은 것과 순위가 밀린 것은 고치는 방법이 다르다.
    """
    stage = record.reorder_stage
    meta = {
        str(track.id): {"title": track.title, "artist": track.artist}
        for track in candidates
    }
    rank_by_id = {str(track.id): rank for rank, track in enumerate(ordered, start=1)}

    def entry(song_id: str, role: str) -> dict:
        song = record.songs.get(song_id)
        row = {
            "role": role,
            "song_id": song_id,
            "rank": rank_by_id.get(song_id),
            "relevant": song_id in relevant_ids,
            "in_candidates": song_id in meta,
            **meta.get(song_id, {"title": None, "artist": None}),
        }
        if song is None:
            # 후보 풀에 없었다. 근거가 없는 것이 아니라 검색이 닿지 않은 것이다.
            row["explain"] = None
            return row
        row["explain"] = to_track_explain(song, reorder_stage=stage).model_dump()
        return row

    songs = [entry(str(track.id), "result") for track in ordered[:top_k]]
    shown = {row["song_id"] for row in songs}
    songs.extend(
        entry(song_id, "relevant")
        for song_id in sorted(relevant_ids)
        if song_id not in shown
    )

    return {
        "query_id": query_id,
        "split": split,
        "query": query,
        "relevant_ids": sorted(relevant_ids),
        "modality_weights": dict(record.modality_weights),
        "reranker": record.reranker,
        "reorder_stage": stage,
        "rerank_calls": len(record.rerank_runs),
        "rerank_calls_applied": record.rerank_runs_applied,
        "rerank_error": record.rerank_error,
        "songs": songs,
    }


async def evaluate(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # "all"은 분할 필터를 걸지 않는다는 뜻이다. read_queries는 None을 그렇게 읽는다.
    split = None if args.split == "all" else args.split
    rows = read_queries(input_path, split)

    if args.query_ids:
        requested_ids = [part.strip() for part in args.query_ids.split(",") if part.strip()]
        row_by_id = {row["query_id"].strip(): row for row in rows}
        rows = [row_by_id[qid] for qid in requested_ids if qid in row_by_id]
        if not rows:
            raise ValueError(f"--query-ids에 해당하는 질의가 없습니다: {requested_ids}")

    if args.exclude_query_ids:
        excluded_ids = {
            part.strip()
            for part in args.exclude_query_ids.split(",")
            if part.strip()
        }
        rows = [row for row in rows if row["query_id"].strip() not in excluded_ids]
        if not rows:
            raise ValueError("--exclude-query-ids 적용 후 평가할 질의가 없습니다.")

    suffix = args.split
    detail_path = output_dir / f"search_eval_{suffix}_detail.csv"
    summary_path = output_dir / f"search_eval_{suffix}_summary.csv"
    explain_path = output_dir / f"search_eval_{suffix}_explain.jsonl"

    # 분석 캐시를 쓰면 분석기를 만들지 않는다 — 만들어 두면 누락 질의를 조용히
    # 새로 분석하는 경로가 생긴다.
    cache = None
    analyzer = None
    cached: dict = {}
    if args.analysis_cache:
        cache = load_cache(Path(args.analysis_cache))
        needed = [row["query_id"].strip() for row in rows]

        missing = cache.missing(needed)
        if missing:
            # 측정 중에 새로 분석하면 그 질의만 다른 조건으로 측정된다.
            raise AnalysisCacheError(
                f"캐시에 없는 질의 {len(missing)}건: {', '.join(missing[:10])}"
                + (" ..." if len(missing) > 10 else "")
                + "\n  build_analysis_cache로 먼저 채우세요."
            )

        fallbacks = cache.fallback_ids(needed)
        if fallbacks and not args.allow_fallback_analysis:
            raise AnalysisCacheError(
                f"규칙 폴백으로 저장된 질의 {len(fallbacks)}건: "
                f"{', '.join(fallbacks[:10])}"
                + (" ..." if len(fallbacks) > 10 else "")
                + "\n  이 질의는 제목·가사 단서가 비어 있어 검색과 무관한 이유로"
                "\n  점수가 떨어집니다. 캐시를 다시 채우거나"
                " --allow-fallback-analysis를 주세요."
            )

        # **선택된 모든 질의를 여기서 전부 꺼내 본다.** 원문 불일치와 스키마 오류는
        # get()에서만 드러나므로, 루프 안에서 처음 만나면 모델·DB를 올리고 파일을 쓴
        # 뒤에 중단된다 — 앞 질의 결과만 남은 결과 폴더가 생긴다.
        problems = []
        for row in rows:
            try:
                cached[row["query_id"].strip()] = cache.get(
                    row["query_id"].strip(), row["query"].strip()
                )
            except AnalysisCacheError as exc:
                problems.append(str(exc))
            except Exception as exc:  # noqa: BLE001 - 스키마가 깨진 항목
                problems.append(f"{row['query_id'].strip()}: {type(exc).__name__}: {exc}")
        if problems:
            raise AnalysisCacheError(
                f"캐시 항목 {len(problems)}건을 쓸 수 없습니다.\n"
                + "\n".join(problems[:10])
                + ("\n  ..." if len(problems) > 10 else "")
            )

        drift = cache.fingerprint_drift()
        if drift:
            print(
                f"경고: 캐시를 만든 분석 조건이 지금과 다릅니다({', '.join(drift)}). "
                "검색·랭킹 비교에는 문제 없지만 기록해 둡니다.",
                flush=True,
            )
        print(
            f"분석 캐시 사용: {args.analysis_cache} "
            f"(질의 {len(needed)}건, 폴백 {len(fallbacks)}건)",
            flush=True,
        )
    else:
        analyzer = get_query_analyzer()
        print(
            "경고: 분석 캐시 없이 실행합니다. 같은 측정을 두 번 해도 질의 분석이 "
            "달라져 숫자가 갈릴 수 있습니다(--analysis-cache 권장).",
            flush=True,
        )

    router = get_search_router()
    reranker = get_reranker()

    if cache is not None:
        analysis_source = {
            "mode": "cache",
            "path": str(args.analysis_cache),
            "meta": cache.meta,
            "fallback_query_ids": cache.fallback_ids(
                [row["query_id"].strip() for row in rows]
            ),
            "fingerprint_drift": cache.fingerprint_drift(),
        }
    else:
        analysis_source = {"mode": "live", "analyzer": analyzer_fingerprint()}

    run_info_path = output_dir / f"search_eval_{suffix}_runinfo.json"
    run_info_path.write_text(
        json.dumps(
            collect_run_info(args, reranker=reranker, analysis_source=analysis_source),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    detail_rows: list[dict] = []
    lyric_rows: list[dict] = []
    lyric_path = output_dir / f"search_eval_{suffix}_lyric_boosts.csv"
    warmed_up = False

    # 질의마다 한 줄씩 즉시 flush한다. detail CSV와 같은 이유 — 중간에 죽어도
    # 끝난 질의의 기록은 남아야 한다.
    explain_file = explain_path.open("w", encoding="utf-8") if args.explain else None

    for index, row in enumerate(rows, start=1):
        query_id = row["query_id"].strip()
        query = row["query"].strip()
        relevant_ids = parse_relevant_ids(row["relevant_ids"])

        print(f"[{index}/{len(rows)}] {query_id}: {query}", flush=True)

        # 캐시가 있으면 시작 전에 검증해 둔 분석을 쓴다. 없으면 질의당 한 번만 분석한다.
        analysis = cached[query_id] if cache is not None else analyzer.analyze(query)

        # 끄면 아무것도 기록하지 않는 null object가 들어간다.
        recorder = ExplainRecorder(query) if args.explain else NULL_RECORDER

        # baseline과 reranking이 완전히 같은 후보 목록을 사용하도록
        # candidate_k개를 한 번만 검색한다.
        retrieval_started = time.perf_counter()
        candidates = await router.search(
            analysis,
            top_k=args.candidate_k,
            use_rerank=False,
            candidate_k=args.candidate_k,
            recorder=recorder,
            path_k=args.path_k,
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        rerank_errors: list[str] = []
        rerank_statuses: list[str] = []
        if not candidates:
            print("  경고: 검색 후보가 없습니다.")
            baseline = []
            reranked_all = []
            reranked = []
            rerank_ms = 0.0
        else:
            baseline = candidates[: args.top_k]

            # 모델 로딩 시간은 정확도와 무관하므로 타이머 전에 모델만 로드한다.
            # 기존 코드는 첫 질의를 한 번 실제 rerank한 뒤 곧바로 다시 rerank하여
            # 첫 질의 추론을 2번 수행했다. 정확도 평가는 load()만으로 충분하다.
            if not warmed_up and len(candidates) > 1:
                reranker.load()
                warmed_up = True

            rerank_started = time.perf_counter()
            # 진단용 score spread를 candidate_k 전체에서 계산하기 위해
            # 전체 후보를 한 번 리랭킹한 뒤 평가 지표는 top_k만 사용한다.
            # 실서비스(SearchRouter)와 동일하게 full_lyrics exact 후보를
            # 일반 후보보다 앞에 두는 보호 로직을 거친다.
            reranked_all = rerank_preserving_exact_lyrics(
                reranker,
                analysis,
                candidates,
                args.candidate_k,
                recorder=recorder,
                errors=rerank_errors,
                statuses=rerank_statuses,
            )
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            reranked = reranked_all[: args.top_k]


        rerank_score_values = [
            float(track.rerank_score)
            for track in reranked_all
            if track.rerank_score is not None
        ]
        rerank_score_min = min(rerank_score_values) if rerank_score_values else 0.0
        rerank_score_max = max(rerank_score_values) if rerank_score_values else 0.0
        rerank_spread = rerank_score_max - rerank_score_min

        mw = analysis.modality_weights
        effective_modality_weights = {
            "text": mw.text,
            "image": mw.image if analysis.has_visual_clue else 0.0,
            "audio": mw.audio if analysis.intent_type != "lyrics" else 0.0,
        }
        dominant_modality = max(
            effective_modality_weights,
            key=effective_modality_weights.get,
        )

        spread_ref = float(getattr(reranker.config, "spread_ref", 0.0) or 0.0)
        rerank_confidence = (
            min(1.0, rerank_spread / spread_ref) if spread_ref > 0.0 else 1.0
        )
        effective_rerank_weight = float(reranker.config.rerank_weight) * rerank_confidence

        candidate_ids = [str(track.id) for track in candidates]
        baseline_ids = [str(track.id) for track in baseline]
        rerank_ids = [str(track.id) for track in reranked]

        candidate_rank = first_relevant_rank(candidate_ids, relevant_ids)
        candidate_recall = recall_at_k(candidate_ids, relevant_ids, args.candidate_k)
        baseline_metrics = metric_bundle(baseline_ids, relevant_ids, args.top_k)
        rerank_metrics = metric_bundle(rerank_ids, relevant_ids, args.top_k)

        baseline_rank_for_compare = baseline_metrics["rank"] or (args.top_k + 1)
        rerank_rank_for_compare = rerank_metrics["rank"] or (args.top_k + 1)

        # 가사 부스트가 **어떤 단서에** 걸렸나. 짧거나 흔한 구절을 완화할지
        # 판단하려면 점수가 아니라 구절 자체를 봐야 한다.
        #
        # **부스트를 받은 곡마다 한 행씩** 남긴다. 질의당 대표 1곡만 남기면
        # 한 질의에서 정답과 오답이 같은 크기의 부스트를 받은 경우가 사라진다
        # (q212·q301이 실제로 그랬다). 그러면 "오답에 걸린 부스트"라는 통계가
        # 정답에도 걸렸다는 사실을 숨긴다.
        final_rank = {str(track.id): rank for rank, track in enumerate(reranked_all, start=1)}
        for song_id, song in recorder.record.songs.items():
            if not song.lyric_match:
                continue
            match = song.lyric_match
            delta = max(
                (p.delta for p in song.paths if p.path == "lyrics_surface"), default=0.0
            )
            lyric_rows.append({
                "query_id": query_id,
                "split": row["split"],
                "song_id": song_id,
                "is_answer": int(song_id in relevant_ids),
                "clue_text": match.get("clue_text", ""),
                "clue_kind": match.get("clue_kind", ""),
                "match_type": match.get("match_type", ""),
                "matched_phrase": match.get("matched_phrase", ""),
                "is_variant": int(bool(match.get("is_variant"))),
                "normalized_length": match.get("normalized_length", ""),
                "corpus_match_count": match.get("corpus_match_count", ""),
                "confidence": match.get("confidence", ""),
                "boost_delta": round(delta, 8),
                "rank_before_boost": song.lyric_boost_rank_before or "",
                "rank_after_boost": song.lyric_boost_rank_after or "",
                "final_rank": final_rank.get(song_id, ""),
                "negative_rules": "|".join(a.rule for a in song.adjustments if a.delta < 0),
            })

        # CSV에서 바로 읽을 수 있는 문장 두 개. 1위와 **정답**을 나란히 둔다 —
        # 진단에 필요한 것은 "정답이 왜 그 자리인가"이고 그건 1위와 다른 곡이다.
        stage = recorder.record.reorder_stage
        explain_top1 = ""
        explain_relevant = ""
        if args.explain:
            if reranked:
                top_song = recorder.record.songs.get(str(reranked[0].id))
                if top_song is not None:
                    explain_top1 = render_ko(top_song, reorder_stage=stage)
            best_relevant = next(
                (song_id for song_id in rerank_ids if song_id in relevant_ids),
                next(
                    (song_id for song_id in candidate_ids if song_id in relevant_ids),
                    None,
                ),
            )
            if best_relevant is None:
                # 순위가 밀린 것이 아니라 검색이 닿지 않았다. 둘은 고치는 방법이 다르다.
                explain_relevant = "정답이 후보 풀에 없음"
            else:
                song = recorder.record.songs.get(best_relevant)
                explain_relevant = (
                    render_ko(song, reorder_stage=stage)
                    if song is not None
                    else "기록 없음"
                )

        detail_rows.append(
            {
                "query_id": query_id,
                "split": row["split"],
                "query": query,
                "relevant_ids": "|".join(sorted(relevant_ids)),
                "intent_type": analysis.intent_type,
                "korean_tags": "|".join(analysis.korean_tags),
                "lyric_keywords": "|".join(analysis.lyric_keywords),
                "lyric_clues": json.dumps(
                    [clue.model_dump() for clue in analysis.lyric_clues],
                    ensure_ascii=False,
                ),
                "lyric_semantic_query": analysis.lyric_semantic_query,
                "title_constraints": json.dumps(
                    analysis.title_constraints.model_dump(),
                    ensure_ascii=False,
                ),
                "title_meaning_clue": json.dumps(
                    analysis.title_meaning_clue.model_dump(),
                    ensure_ascii=False,
                ),
                "vocal_gender": analysis.vocal_gender or "",
                "genre": analysis.genre,
                "image_english_query": analysis.image_english_query,
                "audio_english_query": analysis.audio_english_query,
                "modality_weights": json.dumps(
                    analysis.modality_weights.model_dump(),
                    ensure_ascii=False,
                ),
                "text_alpha": analysis.text_alpha,
                "release_era": json.dumps(
                    analysis.release_era.model_dump(),
                    ensure_ascii=False,
                ),
                "artist_type": json.dumps(
                    analysis.artist_type.model_dump(),
                    ensure_ascii=False,
                ),
                "performance_clues": json.dumps(
                    analysis.performance_clues.model_dump(),
                    ensure_ascii=False,
                ),
                "analysis_json": analysis.model_dump_json(),
                "dominant_modality": dominant_modality,
                "modality_text_weight": round(float(effective_modality_weights["text"]), 6),
                "modality_image_weight": round(float(effective_modality_weights["image"]), 6),
                "modality_audio_weight": round(float(effective_modality_weights["audio"]), 6),
                "rerank_score_min": round(rerank_score_min, 10),
                "rerank_score_max": round(rerank_score_max, 10),
                "rerank_spread": round(rerank_spread, 10),
                "rerank_confidence": round(rerank_confidence, 6),
                "effective_rerank_weight": round(effective_rerank_weight, 6),
                f"candidate_rank@{args.candidate_k}": candidate_rank or "",
                f"candidate_recall@{args.candidate_k}": round(candidate_recall, 6),
                "baseline_rank": baseline_metrics["rank"] or "",
                "rerank_rank": rerank_metrics["rank"] or "",
                "rank_change_positive_is_better": baseline_rank_for_compare - rerank_rank_for_compare,
                "baseline_hit@1": baseline_metrics["hit1"],
                "rerank_hit@1": rerank_metrics["hit1"],
                "baseline_hit@5": baseline_metrics["hit5"],
                "rerank_hit@5": rerank_metrics["hit5"],
                "baseline_hit@10": baseline_metrics["hit10"],
                "rerank_hit@10": rerank_metrics["hit10"],
                "baseline_recall@10": round(float(baseline_metrics["recall10"]), 6),
                "rerank_recall@10": round(float(rerank_metrics["recall10"]), 6),
                "baseline_mrr@10": round(float(baseline_metrics["mrr10"]), 6),
                "rerank_mrr@10": round(float(rerank_metrics["mrr10"]), 6),
                "baseline_ndcg@10": round(float(baseline_metrics["ndcg10"]), 6),
                "rerank_ndcg@10": round(float(rerank_metrics["ndcg10"]), 6),
                "candidate_digest": candidate_digest(candidates, reranker),
                **explain_columns(
                    recorder.record,
                    enabled=args.explain,
                    rerank_error=rerank_errors[0] if rerank_errors else "",
                    statuses=rerank_statuses,
                ),
                "explain_top1": explain_top1,
                "explain_relevant": explain_relevant,
                "retrieval_ms": round(retrieval_ms, 2),
                "rerank_only_ms": round(rerank_ms, 2),
                "baseline_top_ids": "|".join(baseline_ids),
                "rerank_top_ids": "|".join(rerank_ids),
                "baseline_lyric_match_types": "|".join(
                    track.lyric_match_type or "" for track in baseline
                ),
                "rerank_lyric_match_types": "|".join(
                    track.lyric_match_type or "" for track in reranked
                ),
                "baseline_top_titles": " | ".join(
                    f"{track.artist or ''} - {track.title}" for track in baseline
                ),
                "rerank_top_titles": " | ".join(
                    f"{track.artist or ''} - {track.title}" for track in reranked
                ),
            }
        )

        # 외부 SIGTERM/OOM 등으로 중간 종료되더라도 완료된 질의 결과를 잃지 않도록
        # 매 질의마다 detail CSV를 즉시 저장한다.
        write_detail_checkpoint(detail_path, detail_rows)
        write_detail_checkpoint(lyric_path, lyric_rows)
        if explain_file is not None:
            explain_file.write(
                json.dumps(
                    build_explain_payload(
                        query_id=query_id,
                        split=row["split"],
                        query=query,
                        relevant_ids=relevant_ids,
                        record=recorder.record,
                        ordered=reranked_all,
                        candidates=candidates,
                        top_k=args.top_k,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            explain_file.flush()
        improved_now = sum(int(r["rank_change_positive_is_better"] > 0) for r in detail_rows)
        unchanged_now = sum(int(r["rank_change_positive_is_better"] == 0) for r in detail_rows)
        worsened_now = sum(int(r["rank_change_positive_is_better"] < 0) for r in detail_rows)
        print(
            f"  checkpoint 저장 ({len(detail_rows)}건) | "
            f"누적 개선/동일/악화: {improved_now}/{unchanged_now}/{worsened_now}",
            flush=True,
        )

    # 정상 완료 시에도 마지막 상태를 한 번 더 확정 저장한다.
    write_detail_checkpoint(detail_path, detail_rows)
    write_detail_checkpoint(lyric_path, lyric_rows)
    # detail은 제목·설명 문장이 있어 커밋하지 않는다. 곡 ID만 담은 경량본을 따로 남긴다.
    write_top10(detail_rows, top10_path(output_dir, suffix))
    if explain_file is not None:
        explain_file.close()

    summary_rows = []
    metrics = [
        ("Hit@1", "baseline_hit@1", "rerank_hit@1"),
        ("Hit@5", "baseline_hit@5", "rerank_hit@5"),
        ("Hit@10", "baseline_hit@10", "rerank_hit@10"),
        ("Recall@10", "baseline_recall@10", "rerank_recall@10"),
        ("MRR@10", "baseline_mrr@10", "rerank_mrr@10"),
        ("nDCG@10", "baseline_ndcg@10", "rerank_ndcg@10"),
    ]

    for label, baseline_key, rerank_key in metrics:
        baseline_value = mean(detail_rows, baseline_key)
        rerank_value = mean(detail_rows, rerank_key)
        summary_rows.append(
            {
                "metric": label,
                "baseline": round(baseline_value, 6),
                "rerank": round(rerank_value, 6),
                "delta": round(rerank_value - baseline_value, 6),
            }
        )

    improved = sum(int(row["rank_change_positive_is_better"] > 0) for row in detail_rows)
    unchanged = sum(int(row["rank_change_positive_is_better"] == 0) for row in detail_rows)
    worsened = sum(int(row["rank_change_positive_is_better"] < 0) for row in detail_rows)
    # 폴백한 질의는 리랭킹이 없었던 것이므로 "동일"에 섞인다. 건수를 따로 적지
    # 않으면 추론 실패를 "리랭킹 효과 없음"으로 읽는다.
    #
    # 예외만 세면 안 된다 — Gemini는 예외 없이 실패 상태를 돌려준다.
    # rerank_fell_back이 그 두 경우를 모두 담는다.
    fell_back = [
        str(row["query_id"]) for row in detail_rows if row.get("rerank_fell_back")
    ]
    partially_failed = [
        str(row["query_id"]) for row in detail_rows if row.get("rerank_partial_failure")
    ]
    # 실패가 아니다. 다만 "리랭킹이 실제로 몇 건에 적용됐나"를 알려면 분포가 필요하다.
    outcome_counts = Counter(row.get("rerank_outcome", "") for row in detail_rows)

    summary_rows.extend(
        [
            {
                "metric": f"Candidate Recall@{args.candidate_k}",
                "baseline": round(
                    mean(detail_rows, f"candidate_recall@{args.candidate_k}"), 6
                ),
                "rerank": "",
                "delta": "",
            },
            {
                "metric": "Improved query count",
                "baseline": "",
                "rerank": improved,
                "delta": "",
            },
            {
                "metric": "Unchanged query count",
                "baseline": "",
                "rerank": unchanged,
                "delta": "",
            },
            {
                "metric": "Worsened query count",
                "baseline": "",
                "rerank": worsened,
                "delta": "",
            },
            {
                "metric": "Rerank fallback query count",
                "baseline": "",
                "rerank": len(fell_back),
                "delta": "",
            },
            {
                "metric": "Rerank partial failure query count",
                "baseline": "",
                "rerank": len(partially_failed),
                "delta": "",
            },
            {
                "metric": "Rerank outcome distribution",
                "baseline": "",
                "rerank": " ".join(
                    f"{name}={count}" for name, count in sorted(outcome_counts.items())
                ),
                "delta": "",
            },
        ]
    )

    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "baseline", "rerank", "delta"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"상세 결과: {detail_path}")
    print(f"top-10 경량본: {top10_path(output_dir, suffix)}")
    print(f"요약 결과: {summary_path}")
    if args.explain:
        print(f"실행 기록: {explain_path}")
    print(f"실행 조건: {run_info_path}")
    if lyric_rows:
        answers = sum(int(r["is_answer"]) for r in lyric_rows)
        print(
            f"가사 부스트: {lyric_path} "
            f"({len(lyric_rows)}곡, 정답 {answers}곡 / 오답 {len(lyric_rows)-answers}곡)"
        )
    worsened_ids = [
        str(row["query_id"])
        for row in detail_rows
        if row["rank_change_positive_is_better"] < 0
    ]

    print(f"개선/동일/하락 질의 수: {improved}/{unchanged}/{worsened}")
    if fell_back:
        print(
            f"리랭킹 폴백 질의 {len(fell_back)}건 — 이 질의는 리랭킹이 없었으므로 "
            f"'동일'에 섞여 있다: {', '.join(fell_back)}"
        )
    if partially_failed:
        print(
            f"리랭킹 부분 실패 질의 {len(partially_failed)}건 — 일부 호출만 실패했다: "
            f"{', '.join(partially_failed)}"
        )
    print(
        "악화 질의 ID: "
        + (", ".join(worsened_ids) if worsened_ids else "없음")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="동일 후보 목록을 사용해 baseline과 reranking 검색 정확도를 비교합니다."
    )
    parser.add_argument(
        "--input",
        default="experiments/reranking/eval_queries_v06.csv",
        help=(
            "평가 질의 CSV 경로. 기본값이 기준 세트다(dev 57 + test 25, "
            "docs/eval/queries.json에서 내보낸 것). "
            "eval_queries_smoke3.csv는 3행짜리 스모크용이며 기준선이 아니다"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/reranking/results",
        help="결과 CSV 저장 폴더",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        required=True,
        help=(
            "평가할 데이터 분할. **생략할 수 없다** — 생략 시 전체를 돌리던 예전 "
            "동작에서는 dev만 재려던 실행이 test 23건까지 함께 측정했다. "
            "전체를 의도하면 all을 명시한다"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument(
        "--path-k",
        type=int,
        default=None,
        help=(
            "세 기본 경로(text·image·audio)의 깊이와 RRF 절단 수. 생략하면 "
            "--candidate-k와 같다. 최종 후보 수는 --candidate-k 그대로 두고 "
            "경로 깊이만 넓힐 때 쓴다"
        ),
    )
    parser.add_argument(
        "--query-ids",
        default="",
        help="쉼표로 구분한 query_id만 평가 (예: q201,q206,q208)",
    )
    parser.add_argument(
        "--exclude-query-ids",
        default="",
        help="평가에서 제외할 query_id 목록 (예: q118)",
    )
    parser.add_argument(
        "--analysis-cache",
        default="",
        help=(
            "build_analysis_cache로 만든 QueryAnalysis 캐시 경로. 주면 질의 분석을 "
            "새로 하지 않고 저장된 것만 읽는다 — 같은 측정을 두 번 해도 숫자가 "
            "같아진다. 재측정에는 사용을 권한다"
        ),
    )
    parser.add_argument(
        "--allow-fallback-analysis",
        action="store_true",
        help=(
            "캐시에 규칙 폴백(Gemini 실패)으로 저장된 질의가 있어도 측정을 진행한다. "
            "그 질의는 검색과 무관한 이유로 점수가 떨어진다"
        ),
    )
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "선정 근거를 함께 기록한다(기본 켜짐). 결과 1위와 정답 곡의 설명을 "
            "detail CSV에 넣고, 전체 기록을 *_explain.jsonl 로 저장한다. "
            "순위와 점수는 달라지지 않는다"
        ),
    )
    return parser


if __name__ == "__main__":
    asyncio.run(evaluate(build_parser().parse_args()))
