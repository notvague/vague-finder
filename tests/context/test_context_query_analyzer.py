"""Context 단서 분석의 경계와 로컬 평가 질의의 재현성 검증."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.backend.schemas.query import QueryAnalysis
from src.backend.schemas.search import SearchRequest
from src.retrieval.context_query import apply_context_query_safeguards
from src.retrieval.query_analyzer import QueryAnalyzer, _PROMPT_TEMPLATE, rule_fallback


def _fallback(query: str) -> QueryAnalysis:
    analyzer = QueryAnalyzer(api_key="")
    analyzer._api_key = ""
    return analyzer.analyze(query)


def test_existing_analysis_is_compatible_and_context_is_independent():
    old = QueryAnalysis(
        original_query="옛날에 들었던 노래",
        intent_type="mixed",
        image_english_query="",
        audio_english_query="",
    )
    assert old.context_clues == []
    with_context = {**old.model_dump(), "context_clues": [{
        "target": "짱구는 못말려",
        "relation": "삽입곡",
        "search_query": "짱구는 못말려 삽입곡",
        "confidence": 0.8,
    }]}
    restored = QueryAnalysis.model_validate(with_context)
    assert restored.has_context_clue
    assert old.modality_weights == restored.modality_weights
    assert restored.intent_type == "mixed"
    request = SearchRequest.model_validate({
        "query": old.original_query,
        "prior_analysis": restored.model_dump(),
    })
    assert request.prior_analysis.context_clues[0].target == "짱구는 못말려"


def test_all_pilot_queries_are_separated_from_controls():
    path = Path(__file__).parents[2] / "experiments/namuwiki/queries_context_v01.json"
    if not path.is_file():
        pytest.skip("private Context evaluation queries are not installed")
    cases = json.loads(
        path.read_text(encoding="utf-8")
    )["queries"]
    assert len(cases) == 30
    assert sum(case["segment"] != "control" for case in cases) == 26
    for case in cases:
        analysis = _fallback(case["query"])
        assert bool(analysis.context_clues) == (case["segment"] != "control"), case["query_id"]
        for clue in analysis.context_clues:
            assert clue.relation and clue.search_query and 0 < clue.confidence <= 1
            assert not analysis.image_english_query or analysis.has_visual_clue


def test_multiple_facts_and_named_work_remain_distinct():
    query = (
        "드라마 도깨비 OST 중에 여자 가수가 부른 발라드야. "
        "제작진에게 열두 버전을 보냈는데 전부 퇴짜를 맞았어."
    )
    analysis = _fallback(query)
    assert len(analysis.context_clues) == 2
    assert analysis.context_clues[0].target == "도깨비"
    assert "OST" in analysis.context_clues[0].search_query
    assert "열두 버전" in analysis.context_clues[1].search_query
    assert not analysis.has_visual_clue


def test_narrative_and_acronym_survive_sentence_boundaries():
    narrative = _fallback(
        "남자가 여자 대신 죄를 뒤집어쓰고 감옥에 가는 이야기였어. "
        "빅뱅 옛날 뮤직비디오였던 것 같은데 제목이 생각 안 나."
    )
    assert "감옥" in narrative.context_clues[0].search_query
    assert "뮤직비디오" in narrative.context_clues[0].search_query
    acronym = _fallback(
        "MSG워너비 M.O.M이 부른 곡인데 초기 기획 때 김종국과 KCM이 부를 뻔했대."
    )
    assert "M.O.M" in acronym.context_clues[0].search_query


def test_distinct_events_and_followup_question_do_not_duplicate_clues():
    separate = _fallback(
        "김윤아가 어떤 청춘 드라마의 모티브가 됐다고 말했고, "
        "나중에 윤하가 불후의 명곡에서 커버해 우승한 자우림 곡이 뭐였더라?"
    )
    assert len(separate.context_clues) == 2
    assert "드라마" in separate.context_clues[0].search_query
    assert "불후의 명곡" in separate.context_clues[1].search_query
    followup = _fallback(
        "윤종신 노래였는데 민서가 답가도 냈잖아. 그 원곡이 뭐였는지 찾아줘."
    )
    assert len(followup.context_clues) == 1


def test_music_video_details_are_context_not_album_art():
    analysis = _fallback("뉴진스 뮤직비디오에서 민지가 안무를 틀린 실제 NG 장면이 나온 곡")
    assert analysis.context_clues[0].relation == "뮤직비디오 속 사건"
    assert not analysis.has_visual_clue
    assert not analysis.image_english_query


@pytest.mark.parametrize("query", [
    "가사에 '짱구 OST'라는 말이 나오는 노래",
    "가사 내용이 드라마에서 이별하는 이야기인 노래",
    "앨범 표지에 도깨비 OST라는 글자가 그려진 노래",
    "앨범 표지에 드라마 도깨비 OST라고 적혀 있는 노래",
    "앨범 커버에 뮤직비디오 같은 분위기의 남자가 있는 노래",
    "뮤직비디오처럼 드라마틱하게 들리는 피아노 소리",
    "OST 같은 분위기로 들리는 노래",
    "비 오는 날 애니메이션을 보면서 듣기 좋은 노래",
    "제목에 영화라는 단어가 들어가는 노래",
    "제목이 가요제라는 노래",
    "가사에 가요제라는 말이 나오는 노래",
    "앨범 커버에 가요제라는 글자가 적힌 노래",
    "그 노래 가사에 광고 영상에 쓰였다는 말이 있었어",
    "그 노래 제목이 가요제라는 곡",
])
def test_lyrics_artwork_mood_and_title_do_not_make_context(query):
    assert _fallback(query).context_clues == []


def test_independent_lyrics_artwork_and_external_usage_are_kept():
    analysis = _fallback(
        "가사에 '봄이 오면'이라는 구절이 있고, 앨범 표지에는 파란 달이 있어. "
        "드라마 도깨비 OST로 사용됐던 곡이야."
    )
    assert len(analysis.context_clues) == 1
    assert analysis.context_clues[0].target == "도깨비"
    assert "OST" in analysis.context_clues[0].search_query
    assert "봄이 오면" in analysis.lyric_keywords
    assert analysis.image_english_query


@pytest.mark.parametrize("query", [
    "가사에 봄이 온다는 말이 나오고 드라마 도깨비 OST였어",
    "앨범 표지에 파란 달이 있고 드라마 도깨비 OST였어",
])
def test_context_in_same_sentence_does_not_include_lyrics_or_artwork(query):
    clues = _fallback(query).context_clues
    assert len(clues) == 1
    assert clues[0].target == "도깨비"
    assert clues[0].search_query == "드라마 도깨비 OST였어"


def test_uncertain_external_memory_is_lower_confidence():
    uncertain = _fallback("2024년쯤 한양대 축제에서 앙코르를 했던 곡").context_clues[0]
    assert uncertain.confidence <= 0.6
    assert "2024년" in uncertain.search_query
    assert "2025년" not in uncertain.search_query
    assert _fallback("뉴진스 노래가 밈으로 퍼졌대").context_clues[0].confidence > 0.6


def test_model_cannot_add_unsupported_answer_or_create_context():
    query = "짱구 애니메이션에서 나미리 선생님이 울 때 나왔던 노래"
    unsafe = [{
        "target": "도깨비",
        "relation": "드라마 OST 삽입곡",
        "search_query": "도깨비 OST 가상곡제목 윤도현 짱구 애니메이션",
        "confidence": 0.95,
    }]
    sanitized = apply_context_query_safeguards(query, {"context_clues": unsafe})["context_clues"]
    assert len(sanitized) == 1
    assert "도깨비" not in str(sanitized)
    assert "가상곡제목" not in str(sanitized)
    assert "윤도현" not in str(sanitized)
    two_syllables = apply_context_query_safeguards(
        query,
        {"context_clues": [{**unsafe[0], "target": "토이", "search_query": "토이 OST"}]},
    )["context_clues"]
    assert two_syllables[0]["target"] != "토이"
    assert "토이" not in two_syllables[0]["search_query"]
    invented_relation = apply_context_query_safeguards(
        query,
        {"context_clues": [{
            "target": "나미리",
            "relation": "OST 삽입곡",
            "search_query": "짱구 애니메이션 나미리 OST 삽입곡",
            "confidence": 0.95,
        }]},
    )["context_clues"]
    assert "OST" not in invented_relation[0]["relation"]
    assert "OST" not in invented_relation[0]["search_query"]
    assert apply_context_query_safeguards(
        "가사에 짱구라는 말이 나오는 노래", {"context_clues": unsafe},
    )["context_clues"] == []


def test_gemini_response_passes_through_context_safeguards():
    analyzer = QueryAnalyzer(api_key="fake-key")
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({
            "intent_type": "mixed",
            "image_english_query": "",
            "audio_english_query": "",
            "modality_weights": {"text": 1, "image": 0, "audio": 0},
            "context_clues": [{
                "target": "존재하지않는영화",
                "relation": "삽입곡",
                "search_query": "존재하지않는영화 노래",
                "confidence": 0.99,
            }],
        }, ensure_ascii=False))

    analyzer._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    analysis = analyzer.analyze("짱구 애니메이션에서 나미리 선생님이 울 때 나오던 노래")
    assert len(calls) == 1
    assert len(analysis.context_clues) == 1
    assert "존재하지않는영화" not in analysis.context_clues[0].search_query
    assert "존재하지않는영화" not in analysis.context_clues[0].target


def test_prompt_rendering_keeps_context_rules_and_existing_modalities():
    prompt = _PROMPT_TEMPLATE.format(
        query="드라마 도깨비 OST",
        reference_year=2026,
        reference_year_minus_one=2025,
        recent_start_year=2018,
    )
    assert "context_clues" in prompt
    assert "image_english_query" in prompt
    assert "audio_english_query" in prompt


@pytest.mark.parametrize("query,expected_target", [
    ("예능 프로그램에서 가요제 한다고 만든 곡", ""),
    ("무한도전 가요제 곡", "무한도전"),
    ("도깨비 드라마에 나온 노래인데 제목이 기억 안 나", "도깨비"),
])
def test_reviewed_background_phrasings_reach_fallback(query, expected_target):
    clues = rule_fallback(query).context_clues
    assert len(clues) == 1
    assert clues[0].target == expected_target
    assert clues[0].search_query


@pytest.mark.parametrize("query", [
    "드라마 가진 주인공이 나오는 곡",
    "드라마 별 없이 부른 곡",
    "드라마 배경으로 한 추억이 나오는 곡",
    "2000년대 초반 겨울 드라마 OST였어",
    "사극 드라마 OST였어",
    "다른 가수가 부른 노래인데 배우가 출연한 드라마 OST였어",
    "오래된 동네를 배경으로 한 추억 드라마에 삽입된 곡",
    "랩 없이 부른 드라마 삽입곡",
])
def test_unreliable_after_media_phrases_do_not_become_titles(query):
    clues = apply_context_query_safeguards(query, {})["context_clues"]
    assert not clues or clues[0]["target"] == ""


def test_before_media_discards_time_adverb_and_grounded_model_can_supply_target():
    query = "옛날에 짱구 애니메이션에서 나미리 선생님이 우는 장면에 나온 노래"
    assert rule_fallback(query).context_clues[0].target == "짱구"
    assert rule_fallback(
        "애니메이션 짱구에서 나미리 선생님이 우는 장면에 나온 노래"
    ).context_clues[0].target == "짱구"

    query = "드라마 도깨비에 나온 곡"
    model = {"context_clues": [{
        "target": "도깨비", "relation": "나온", "search_query": query,
        "confidence": 0.7,
    }]}
    assert apply_context_query_safeguards(query, model)["context_clues"][0]["target"] == "도깨비"


def test_unmatched_grounded_model_event_supplements_rules_without_double_counting():
    query = "드라마 도깨비 OST였어, 광고 영상에 쓰였던 곡"
    extra = {"target": "", "relation": "쓰였던", "search_query": "광고 영상에 쓰였던 곡", "confidence": 0.9}
    clues = apply_context_query_safeguards(query, {"context_clues": [extra]})["context_clues"]
    assert len(clues) == 2
    assert clues[0]["target"] == "도깨비"
    assert clues[1] == {**extra, "confidence": 0.5}

    duplicate = {"target": "도깨비", "relation": "OST", "search_query": "드라마 도깨비 OST였어", "confidence": 0.9}
    clues = apply_context_query_safeguards(query, {"context_clues": [duplicate, extra, extra]})["context_clues"]
    assert len(clues) == 2
    assert sum("광고" in clue["search_query"] for clue in clues) == 1

    joined = "드라마 도깨비 OST였고 광고 영상에 쓰였던 곡"
    clues = apply_context_query_safeguards(joined, {"context_clues": [extra]})["context_clues"]
    assert len(clues) == 2
    assert clues[0]["relation"] == "삽입곡·배경음악"
    assert clues[1]["relation"] == "쓰였던"


@pytest.mark.parametrize("query,search", [
    ("가사에 광고 영상에 쓰였던 곡이라는 문장이 있어", "광고 영상에 쓰였던 곡"),
    ("앨범 표지에 광고 영상에 쓰였던 곡이라는 글자가 있어", "광고 영상에 쓰였던 곡"),
    ("비 오는 날 광고 보면서 들을 노래", "광고 보면서 들을 노래"),
])
def test_model_only_context_requires_real_external_relation(query, search):
    unsafe = {"context_clues": [{
        "target": "", "relation": "쓰였던", "search_query": search, "confidence": 0.9,
    }]}
    assert apply_context_query_safeguards(query, unsafe)["context_clues"] == []


def test_model_only_grounding_rejects_invented_target_relation_and_nonfinite_confidence():
    query = "광고 영상에 쓰였던 곡"
    safe = {"target": "", "relation": "쓰였던", "search_query": query, "confidence": 0.95}
    assert apply_context_query_safeguards(query, {"context_clues": [safe]})["context_clues"] == [
        {**safe, "confidence": 0.5}
    ]
    unsafe = [
        {**safe, "target": "없는드라마"},
        {**safe, "search_query": query + " 없는가수"},
        {**safe, "confidence": float("nan")},
        {**safe, "confidence": 2},
    ]
    assert apply_context_query_safeguards(query, {"context_clues": unsafe})["context_clues"] == []
    corrected = apply_context_query_safeguards(
        query, {"context_clues": [{**safe, "relation": "광고 삽입곡"}]},
    )["context_clues"]
    assert len(corrected) == 1
    assert corrected[0]["relation"] == "쓰였"
    assert "삽입곡" not in str(corrected)


def test_model_cannot_promote_temporal_adjective_to_work_title():
    query = "2000년대 초반 겨울 드라마 OST였어"
    for target in ("2000년대 초반 겨울", "겨울"):
        model = {"context_clues": [{
            "target": target, "relation": "OST", "search_query": query, "confidence": 0.9,
        }]}
        clue = apply_context_query_safeguards(query, model)["context_clues"][0]
        assert clue["target"] == ""


def test_context_prompt_shrunk_and_analysis_deadline_remains_bounded(monkeypatch):
    monkeypatch.delenv("QUERY_ANALYSIS_TIMEOUT_SECONDS", raising=False)
    analyzer = QueryAnalyzer(api_key="")
    prompt = analyzer._prompt("드라마 도깨비 OST")
    assert len(prompt) < 24000
    assert analyzer.analysis_budget_seconds == 20


def test_org_public_fallback_and_async_api_keep_context_clues():
    query = "드라마 도깨비 OST로 사용됐던 곡"
    expected = rule_fallback(query)
    assert expected.has_context_clue
    analyzer = QueryAnalyzer(api_key="")
    analyzer._api_key = ""
    assert analyzer.analysis_budget_seconds > 0
    async_result = asyncio.run(analyzer.analyze_async(query))
    assert async_result.context_clues == expected.context_clues
