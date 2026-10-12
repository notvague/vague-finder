"""모달리티 경계만 어긴 응답은 그 모달리티만 내려놓는다.

예전에는 규칙과 모델이 한 모달리티에서 어긋나면 3회 재시도 뒤 **분석 전체**가 규칙
폴백이 됐다. 가사·제목·가수·태그까지 잃었고(results_v38: t009·t071·t086·t099, v09 n002),
규칙이 틀린 경우에도 그랬다("초록"의 록). 이제 재시도는 그대로 하되, 끝내 어긋나면 마지막
응답에서 어긴 모달리티만 끄거나 규칙 프롬프트로 바꾼다.

모델 호출 없음 — Gemini 응답은 전부 가짜다.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

import pytest

from src.backend.schemas.query import QueryAnalysis
from src.retrieval import query_analyzer as qa
from src.retrieval.analysis_cache import looks_like_fallback
from src.retrieval.modality_queries import (
    ModalityQueryValidationError,
    apply_modality_query_safeguards,
)

COVER_QUERY = "빨간 앨범 커버 노래"                       # 규칙: 표지 단서, 규칙 프롬프트 있음
COVER_QUERY_NO_RULE_PROMPT = "앨범 커버에 고양이가 있는 노래"  # 규칙: 표지 단서, 규칙 프롬프트 없음
SOUND_QUERY = "허스키한 여자 보컬 발라드"                  # 규칙: 소리 단서, 규칙 프롬프트 있음
SOUND_QUERY_NO_RULE_PROMPT = "휘파람 소리로 시작하는 노래"   # 규칙: 소리 단서, 규칙 프롬프트 없음


def model_json(**over: Any) -> Dict[str, Any]:
    """모델이 낸 것으로 칠 응답. 모달리티 프롬프트만 바꿔 가며 쓴다."""
    raw: Dict[str, Any] = {
        "intent_type": "mixed", "korean_tags": ["이별", "그리움"],
        "lyric_keywords": [], "lyric_clues": [], "lyric_semantic_query": "",
        "song_title": "", "artist_name": "아이유", "artist_name_alt": ["IU"],
        "vocal_gender": "여성", "genre": "발라드",
        "image_english_query": "", "audio_english_query": "",
        "modality_weights": {"text": 0.4, "image": 0.3, "audio": 0.3},
        "text_alpha": 0.5, "confidence": 0.9,
    }
    raw.update(over)
    return raw


# --- 1. 검증 함수 ------------------------------------------------------------------------

@pytest.mark.parametrize("query, modality", [(COVER_QUERY, "image"), (SOUND_QUERY, "audio")])
def test_strict_mode_still_raises_and_names_the_modality(query: str, modality: str) -> None:
    """기본 동작은 그대로다 — 분석기가 정정을 붙여 다시 물을 수 있어야 한다."""
    with pytest.raises(ModalityQueryValidationError) as caught:
        apply_modality_query_safeguards(query, model_json())
    assert (caught.value.modality, caught.value.kind) == (modality, "missing")


@pytest.mark.parametrize("query, modality", [(COVER_QUERY, "image"), (SOUND_QUERY, "audio")])
def test_a_missing_prompt_turns_only_that_route_off(query: str, modality: str) -> None:
    """모델이 정정을 받고도 프롬프트를 안 냈다 — 규칙 프롬프트로 채우지 않고 끈다.

    규칙 프롬프트가 있는 질의로 본다. 채우면 규칙이 틀렸을 때 그 오염이 검색에 들어간다.
    """
    out = apply_modality_query_safeguards(query, model_json(), degrade=True)
    assert out["degraded_modalities"] == [modality]
    assert (out["image_english_query"], out["audio_english_query"]) == ("", "")
    assert out["has_visual_clue"] is False
    assert out["modality_weights"] == {"text": 1.0, "image": 0.0, "audio": 0.0}
    # 나머지 분석은 손대지 않는다.
    assert (out["artist_name"], out["vocal_gender"], out["korean_tags"]) == ("아이유", "여성", ["이별", "그리움"])


def test_the_other_modality_survives() -> None:
    """표지 프롬프트가 빠졌다고 소리 경로까지 끄지 않는다."""
    query = "빨간 앨범 커버였고 허스키한 여자 보컬 발라드"
    out = apply_modality_query_safeguards(
        query,
        model_json(audio_english_query="A ballad with husky female vocals."),
        degrade=True,
    )
    assert out["degraded_modalities"] == ["image"]
    assert out["image_english_query"] == ""
    assert out["audio_english_query"].startswith("A ballad with husky female vocals")
    assert out["modality_weights"]["image"] == 0.0 and out["modality_weights"]["audio"] > 0.0


def test_a_leaking_audio_prompt_is_replaced_by_the_rule_prompt() -> None:
    """단서가 있다는 데는 규칙과 모델이 같다. 못 믿는 것은 문장이라 규칙 문장으로 바꾼다."""
    leaking = "A husky female pop star singing a ballad."     # 'star'가 시각 낱말로 읽힌다(v09 n025)
    with pytest.raises(ModalityQueryValidationError) as caught:
        apply_modality_query_safeguards(SOUND_QUERY, model_json(audio_english_query=leaking))
    assert (caught.value.modality, caught.value.kind) == ("audio", "leak")

    out = apply_modality_query_safeguards(
        SOUND_QUERY, model_json(audio_english_query=leaking), degrade=True)
    assert out["degraded_modalities"] == ["audio"]
    assert out["audio_english_query"] == (
        "An audio track with female vocals, a ballad style, a husky vocal timbre.")
    assert "star" not in out["audio_english_query"]
    assert out["modality_weights"]["audio"] > 0.0


def test_a_leaking_audio_prompt_without_a_rule_prompt_is_turned_off() -> None:
    out = apply_modality_query_safeguards(
        SOUND_QUERY_NO_RULE_PROMPT,
        model_json(audio_english_query="Whistling over a blue night-sky background."),
        degrade=True,
    )
    assert out["degraded_modalities"] == ["audio"]
    assert out["audio_english_query"] == ""
    assert out["modality_weights"] == {"text": 1.0, "image": 0.0, "audio": 0.0}


def test_a_leaking_image_prompt_is_replaced_or_turned_off() -> None:
    leaking = "A red album cover for a slow ballad with soft vocals."
    replaced = apply_modality_query_safeguards(
        COVER_QUERY, model_json(image_english_query=leaking), degrade=True)
    assert replaced["degraded_modalities"] == ["image"]
    assert replaced["image_english_query"] == "Album cover with red tones."
    assert replaced["has_visual_clue"] is True

    off = apply_modality_query_safeguards(
        COVER_QUERY_NO_RULE_PROMPT, model_json(image_english_query=leaking), degrade=True)
    assert off["degraded_modalities"] == ["image"]
    assert (off["image_english_query"], off["has_visual_clue"]) == ("", False)


def test_a_valid_response_is_the_same_in_both_modes() -> None:
    """내려놓는 길은 어긴 응답에서만 다르다. 멀쩡한 응답의 분석은 한 글자도 바뀌지 않는다."""
    raw = model_json(image_english_query="A red album cover.")
    strict = apply_modality_query_safeguards(COVER_QUERY, dict(raw))
    lenient = apply_modality_query_safeguards(COVER_QUERY, dict(raw), degrade=True)
    assert strict == lenient
    assert strict["degraded_modalities"] == []


def test_the_model_cannot_set_the_marker_itself() -> None:
    raw = model_json(image_english_query="A red album cover.", degraded_modalities=["audio"])
    assert apply_modality_query_safeguards(COVER_QUERY, raw)["degraded_modalities"] == []


def test_stored_analyses_without_the_field_still_load() -> None:
    """이 필드가 생기기 전에 만든 분석 캐시·prior_analysis를 그대로 읽는다."""
    stored = qa.rule_fallback(COVER_QUERY).model_dump(mode="json")
    stored.pop("degraded_modalities")
    assert QueryAnalysis(**stored).degraded_modalities == []


# --- 2. 분석기 ---------------------------------------------------------------------------

class _Scripted:
    """시도마다 정해진 응답을 낸다. dict는 모델 JSON, 예외는 그대로 던진다."""

    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)
        self.prompts: List[str] = []
        self.models = self

    def generate_content(self, *, model: str, contents: str, config: Dict[str, Any]):
        self.prompts.append(contents)
        step = self.steps[min(len(self.prompts), len(self.steps)) - 1]
        if isinstance(step, Exception):
            raise step
        return type("_Response", (), {"text": json.dumps(step, ensure_ascii=False)})()

    @property
    def aio(self):
        outer = self

        class _AsyncModels:
            async def generate_content(self, **kw):
                return outer.generate_content(**kw)

        return type("_Aio", (), {"models": _AsyncModels()})()


def _analyzer(fake: _Scripted, monkeypatch) -> qa.QueryAnalyzer:
    real_sleep = time.sleep
    monkeypatch.setattr(qa.time, "sleep", lambda s: real_sleep(0))

    async def no_wait(seconds: float) -> None:
        return None

    monkeypatch.setattr(qa.asyncio, "sleep", no_wait)
    analyzer = qa.QueryAnalyzer(api_key="시험용")
    analyzer._client = fake
    return analyzer


def _run(analyzer: qa.QueryAnalyzer, query: str, use_async: bool) -> QueryAnalysis:
    if use_async:
        return asyncio.run(analyzer.analyze_async(query))
    return analyzer.analyze(query)


@pytest.mark.parametrize("use_async", [False, True])
def test_three_boundary_failures_keep_the_rest_of_the_analysis(monkeypatch, use_async: bool) -> None:
    fake = _Scripted(model_json())           # 표지 프롬프트를 끝내 안 낸다
    analysis = _run(_analyzer(fake, monkeypatch), COVER_QUERY, use_async)

    assert len(fake.prompts) == 3, "재시도는 그대로 한다 — 정정을 받고 고치는 응답을 잃지 않는다"
    assert ["REQUIRED CORRECTION" in p for p in fake.prompts] == [False, True, True]
    assert analysis.degraded_modalities == ["image"]
    assert analysis.image_english_query == "" and analysis.has_visual_clue is False
    assert analysis.modality_weights.image == 0.0
    # 예전에는 여기서 전부 잃었다.
    assert (analysis.artist_name, analysis.vocal_gender, analysis.confidence) == ("아이유", "여성", 0.9)
    assert analysis.korean_tags == ["이별", "그리움"]
    assert not looks_like_fallback(analysis)


@pytest.mark.parametrize("use_async", [False, True])
def test_a_later_valid_response_wins(monkeypatch, use_async: bool) -> None:
    """정정을 받고 고친 응답이 오면 그것을 쓴다. 내려놓은 것이 없다."""
    fake = _Scripted(model_json(), model_json(image_english_query="A red album cover."))
    analysis = _run(_analyzer(fake, monkeypatch), COVER_QUERY, use_async)
    assert len(fake.prompts) == 2
    assert analysis.degraded_modalities == []
    assert analysis.image_english_query == "A red album cover."


@pytest.mark.parametrize("use_async", [False, True])
def test_a_boundary_failure_is_kept_through_later_transport_errors(monkeypatch, use_async: bool) -> None:
    """1차는 경계만 어겼고 2·3차는 통신 오류다. 1차 응답을 버릴 이유가 없다."""
    fake = _Scripted(model_json(), RuntimeError("일시적 통신 오류"), RuntimeError("일시적 통신 오류"))
    analysis = _run(_analyzer(fake, monkeypatch), COVER_QUERY, use_async)
    assert len(fake.prompts) == 3
    assert analysis.degraded_modalities == ["image"]
    assert analysis.artist_name == "아이유"


def test_the_last_boundary_failure_is_the_one_used(monkeypatch) -> None:
    """정정을 받은 뒤의 응답이 모델의 최종 판단에 가깝다."""
    fake = _Scripted(
        model_json(artist_name="처음"),
        model_json(artist_name="나중"),
        model_json(artist_name="마지막"),
    )
    analysis = _analyzer(fake, monkeypatch).analyze(COVER_QUERY)
    assert analysis.artist_name == "마지막"


@pytest.mark.parametrize("use_async", [False, True])
def test_other_failures_still_fall_back_to_rules(monkeypatch, use_async: bool) -> None:
    """살릴 응답이 없으면 예전과 같다."""
    fake = _Scripted(RuntimeError("일시적 통신 오류"))
    analysis = _run(_analyzer(fake, monkeypatch), COVER_QUERY, use_async)
    assert len(fake.prompts) == 3
    assert looks_like_fallback(analysis)
    assert analysis.degraded_modalities == []


def test_an_unusable_response_falls_back_to_rules(monkeypatch) -> None:
    """모달리티 말고 다른 필드까지 못 쓰는 응답이면 살리지 않는다."""
    fake = _Scripted(model_json(confidence=7.0))     # 스키마 범위 밖
    analysis = _analyzer(fake, monkeypatch).analyze(COVER_QUERY)
    assert looks_like_fallback(analysis)


def test_the_budget_running_out_keeps_the_boundary_failure(monkeypatch) -> None:
    """예산을 다 쓴 경우도 같다 — 이미 받은 응답이 있으면 규칙 폴백보다 낫다."""
    fake = _Scripted(model_json())
    analyzer = _analyzer(fake, monkeypatch)
    clock = iter([0.0, 0.0, 0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(qa.time, "monotonic", lambda: next(clock, 100.0))
    analysis = analyzer.analyze(COVER_QUERY)
    assert len(fake.prompts) == 1
    assert analysis.degraded_modalities == ["image"]
    assert analysis.artist_name == "아이유"
