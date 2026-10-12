"""
tests/test_emotion_vocab.py

대표 감정(major_emotion)을 크롤러가 한국어 통제 어휘로 받는지 검증.

배경: 수집분 961곡의 major_emotion이 **전부 영어**였고, 자유 생성이라 45종으로 흩어져 있었다.

          Joy(125) / Joyful(3) / Happiness(1)
          Sensuality(3) / Sensual(1)
          Longing(14) / Yearning(3)

      새로 수집하는 곡부터 한국어 고정 어휘 중 하나를 고르게 하고, LLM이 목록 밖 값을
      돌려주면 같은 응답의 태그를 근거로 정규화한다.

색인: 기존 영어 값을 이 어휘로 옮기거나 형용사형을 덧붙여 색인하는 변경은 A/B에서 현재 76개
      질의로 개선을 확인하지 못해 제외했다. 감정 표현이 든 질의는 8개뿐이라 일반적으로 효과가 없다는
      입증은 아니다(experiments/reranking/results_emotion_index_ab/RUN_INFO.md).
      passage_builder가 저장된 감정값을 번역·확장하지 않고 그대로 넣는지도 여기서 확인한다.
      (앞으로 한국어 값으로 수집한 곡은 영어 값이던 곡과 토큰이 달라진다.)

실행:
    venv/bin/python -m pytest tests/test_emotion_vocab.py -v
"""
from __future__ import annotations

import copy
import json
import types
from collections import Counter
from pathlib import Path

import pytest

from src.common.emotion_vocab import (
    EMOTION_VOCAB,
    KNOWN_ENGLISH,
    canonical_emotion,
    vocab_prompt_line,
)
import src.crawler.scripts_py.refine_data as refine_module
from src.crawler.scripts_py.refine_data import _normalize_major_emotion, generate_prompt
from src.embedding.fixtures.meta_validation import validate_meta_document

CORPUS = Path("data/all_songs.jsonl")

# 실제 수집분에 나온 45종.
CORPUS_VALUES = [
    "Sadness", "Joy", "Melancholy", "Excitement", "Nostalgia", "Healing",
    "Empowerment", "Longing", "Love", "Confidence", "Passion", "Bittersweet",
    "Defiance", "Sensuality", "Loneliness", "Yearning", "Liberation", "Joyful",
    "Cynicism", "Catharsis", "Determination", "Conflict", "Intensity", "Sensual",
    "Gratitude", "Desire", "Sarcasm", "Mood", "Resentment", "Affection",
    "Mysterious", "Despair", "Frustration", "Cool", "Euphoria", "Relief",
    "Suspense", "Angst", "Captivation", "Obsession", "Ambition", "Hopeful",
    "Romance", "Happiness", "Resilience",
]


# --- 어휘와 분류 --------------------------------------------------------------

@pytest.mark.parametrize("value", CORPUS_VALUES)
def test_every_collected_english_value_is_classified(value) -> None:
    """수집분의 영어 값은 모두 셋 중 하나로 분류돼 있어야 한다.

    옮김 / 근거로 가림 / 대응 없음. 분류가 빠진 값이 조용히 비워지는 일이 없도록.
    """
    assert value.lower() in KNOWN_ENGLISH


@pytest.mark.skipif(not CORPUS.exists(), reason="곡 데이터 없음")
def test_no_corpus_value_is_unclassified() -> None:
    """실제 코퍼스에 분류표에 없는 값이 없어야 한다."""
    unknown = set()
    for line in CORPUS.open(encoding="utf-8"):
        value = (json.loads(line).get("community_feedback") or {}).get("major_emotion")
        if value and value not in EMOTION_VOCAB and value.lower() not in KNOWN_ENGLISH:
            unknown.add(value)
    assert unknown == set()


def test_true_synonyms_are_merged() -> None:
    """뜻이 같은 값은 한 값으로 합친다."""
    assert canonical_emotion("Joy") == canonical_emotion("Happiness") == "기쁨"
    assert canonical_emotion("Longing") == canonical_emotion("Yearning") == "그리움"


def test_korean_values_pass_through() -> None:
    for value in EMOTION_VOCAB:
        assert canonical_emotion(value) == value


def test_unknown_value_is_dropped() -> None:
    """옮길 근거가 없는 값은 비운다."""
    assert canonical_emotion("Whatever") == ""
    assert canonical_emotion("분석실패") == ""
    assert canonical_emotion(None) == ""
    assert canonical_emotion("") == ""


def test_sadness_and_melancholy_stay_apart() -> None:
    """통곡과 가라앉은 정서는 다른 감정이다."""
    assert canonical_emotion("Sadness") != canonical_emotion("Melancholy")


def test_prompt_line_lists_the_whole_vocabulary() -> None:
    line = vocab_prompt_line()
    for value in EMOTION_VOCAB:
        assert value in line


# --- 의미 검증: 빠짐없이 옮기는 것과 맞게 옮기는 것은 별개다 --------------------
# 곡마다 자체 emotion_tags·mood_tags와 대조해서 나온 사례들이다.

@pytest.mark.parametrize(
    "value, tags",
    [
        ("Obsession", ["고독", "절박함", "허무", "광기", "서늘함", "집착"]),    # 에픽하이 'Fan'
        ("Suspense", ["공포", "불안", "집착", "긴장감", "섬뜩함"]),             # '미친거니'
        ("Sarcasm", ["시기심", "유쾌함", "위트", "유머"]),                      # '봄이 좋냐??'
        ("Cynicism", ["씁쓸함", "통쾌함", "풍자", "비판"]),                     # 'Dirty Cash'
        ("Conflict", ["혼란", "갈등", "애틋함"]),                               # '두사랑'
        ("Mood", ["묘함", "설렘", "레트로", "몽환"]),                            # 감정이 아님
    ],
)
def test_no_equivalent_values_are_dropped_even_with_tags(value, tags) -> None:
    """대응하는 어휘가 없는 값은 태그가 뭐든 비운다."""
    assert canonical_emotion(value, tags) == ""


def test_sensuality_is_not_flutter_without_evidence() -> None:
    """'Love Shot' — 섹시·치명적. 근거 없이 '설렘'으로 옮기면 뜻이 바뀐다."""
    assert canonical_emotion("Sensuality", ["갈망", "카타르시스", "매혹", "치명적", "섹시함"]) == ""


def test_excitement_splits_by_the_songs_own_tags() -> None:
    """LLM이 Excitement를 두 뜻으로 썼다. 곡 태그로 가린다."""
    dance = ["흥분", "쾌감", "신나는", "강렬한"]           # '19금 유레카'
    flutter = ["두근거림", "기대감", "설렘", "달콤함"]      # '틈'
    assert canonical_emotion("Excitement", dance) == "신남"
    assert canonical_emotion("Excitement", flutter) == "설렘"


def test_more_evidence_wins_and_ties_keep_the_literal_meaning() -> None:
    both_more_flutter = ["설렘", "두근거림", "경쾌함"]      # 설렘 2 : 신남 1
    tie = ["설렘", "신나는"]                               # 1 : 1 -> 직역(신남)
    assert canonical_emotion("Excitement", both_more_flutter) == "설렘"
    assert canonical_emotion("Excitement", tie) == "신남"


def test_ambiguous_value_without_evidence_is_dropped() -> None:
    """'피 땀 눈물' — Desire인데 태그에 그리움 근거가 없다."""
    assert canonical_emotion("Desire", ["중독", "고뇌", "갈등", "몽환", "퇴폐"]) == ""
    assert canonical_emotion("Excitement") == ""


# --- 목록을 벗어난 한국어 답 ----------------------------------------------------
# 프롬프트가 한국어를 요구하므로 LLM이 목록을 벗어나면 한국어 유사어로 벗어날 가능성이 크다.
# 'Happiness'는 기쁨으로 옮기면서 같은 뜻의 '행복'을 비우면 앞뒤가 맞지 않는다.

@pytest.mark.parametrize(
    "value, expected",
    [
        ("행복", "기쁨"),
        ("외로움", "쓸쓸함"),
        ("설레임", "설렘"),
        ("신나는", "신남"),
    ],
)
def test_korean_near_synonym_maps_by_its_own_stem(value, expected) -> None:
    assert canonical_emotion(value) == expected


def test_korean_answer_matching_several_entries_uses_tags() -> None:
    """'향수'는 그리움·추억 어간에 모두 걸린다. 태그로 가리고, 근거가 없으면 비운다."""
    assert canonical_emotion("향수") == ""
    assert canonical_emotion("향수", ["추억", "회상"]) == "추억"


@pytest.mark.parametrize("value, tags", [("평온", ["평온"]), ("서글픔", []), ("Sad", ["슬픔"])])
def test_off_list_values_without_a_stem_are_still_dropped(value, tags) -> None:
    """답 자체에 어휘 어간이 없으면 태그만 보고 옮기지 않는다."""
    assert canonical_emotion(value, tags) == ""


def test_tags_alone_never_rescue_a_no_equivalent_label() -> None:
    """대응 어휘가 없는 라벨이 태그에 걸린 다른 감정으로 바뀌면 안 된다."""
    assert canonical_emotion("Obsession", ["고독", "쓸쓸함"]) == ""
    assert canonical_emotion("Mood", ["설렘", "몽환"]) == ""


def test_refine_normalizes_a_korean_near_synonym() -> None:
    result = _normalize_major_emotion({"major_emotion": "행복", "emotion_tags": ["행복", "즐거움"]})
    assert result["major_emotion"] == "기쁨"


# --- 크롤러: LLM 응답 정규화 -----------------------------------------------------

def test_prompt_asks_for_korean_from_the_fixed_list() -> None:
    prompt = generate_prompt({"title": "밤편지", "artist": ["아이유"]}, "가사", [], [])
    assert "except major_emotion in English" not in prompt
    assert vocab_prompt_line() in prompt
    assert "__EMOTION_VOCAB__" not in prompt


def test_in_vocab_answer_is_kept() -> None:
    result = _normalize_major_emotion({"major_emotion": "위로", "emotion_tags": ["평온"]})
    assert result["major_emotion"] == "위로"


def test_off_list_answer_is_resolved_with_the_same_responses_tags() -> None:
    """LLM이 목록을 무시하고 영어를 돌려줘도 같은 응답의 태그로 가린다."""
    result = _normalize_major_emotion({
        "major_emotion": "Excitement",
        "emotion_tags": ["두근거림", "기대감"],
        "mood_tags": ["설렘", "달콤함"],
    })
    assert result["major_emotion"] == "설렘"


def test_off_list_answer_without_evidence_is_emptied() -> None:
    result = _normalize_major_emotion({"major_emotion": "Obsession", "emotion_tags": ["집착"]})
    assert result["major_emotion"] == ""


def test_evidence_uses_tags_not_the_sentiment_summary() -> None:
    """반응 요약의 '향수를 느낀다' 같은 상투구가 거짓 근거가 되면 안 된다."""
    result = _normalize_major_emotion({
        "major_emotion": "Desire",
        "emotion_tags": ["중독", "고뇌"],
        "mood_tags": ["몽환", "퇴폐"],
        "sentiment_summary": "발매 당시를 회상하며 각별한 향수와 애틋함을 느낀다.",
    })
    assert result["major_emotion"] == ""


def test_malformed_tags_do_not_break_normalization() -> None:
    result = _normalize_major_emotion({
        "major_emotion": "Excitement",
        "emotion_tags": None,
        "mood_tags": ["신나는", 3, None],
    })
    assert result["major_emotion"] == "신남"


# --- 크롤러 성공 경로에 정규화가 실제로 연결돼 있는지 ------------------------------
# 위 테스트는 헬퍼만 직접 부른다. refine_data가 LLM 응답을 받을 때 헬퍼를 부르지 않게 바뀌어도
# 통과하므로, Gemini 클라이언트를 가짜로 바꿔 전체 경로를 오프라인으로 돈다.

# refine_data는 응답의 타입·필수 필드를 검사한 뒤에야 정규화한다(validate_llm_schema).
# 감정 필드만 든 페이로드는 필드 누락으로 재요청되므로, 나머지는 정상 값으로 채운다.
FULL_LLM_PAYLOAD = {
    "album_summary": "앨범 소개 요약", "artist_type": ["솔로"], "vocal_gender": "여성",
    "lyrics_highlight": "가장 기억에 남는 한 줄", "lyrics_summary": "가사 요약",
    "search_style_summary": "상황 묘사", "mood_tags": ["잔잔함"], "time_weather_tags": ["새벽"],
    "place_activity_tags": ["창가"], "emotion_tags": ["그리움"], "vibe_tags": ["몽환적"],
    "relation_context_tags": ["이별"], "color_tags": ["파랑"], "sound_tags": ["피아노"],
    "visual_imagery": ["비 오는 창가"], "sentiment_summary": "댓글 요약", "fans_tags": ["인생곡"],
    "major_emotion": "슬픔", "context_tags": [], "fact_summary": "",
}


def _fake_gemini(monkeypatch, payload: dict) -> None:
    payload = {**FULL_LLM_PAYLOAD, **payload}

    class _Models:
        def generate_content(self, **kwargs):
            return types.SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))

    class _Client:
        def __init__(self):
            self.models = _Models()

    monkeypatch.setattr(refine_module, "GEMINI_CONFIGURED", True)
    monkeypatch.setattr(refine_module, "make_genai_client", lambda **kwargs: _Client())
    monkeypatch.setattr(refine_module.time, "sleep", lambda seconds: None)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"major_emotion": "Excitement", "emotion_tags": ["두근거림"], "mood_tags": ["설렘"]}, "설렘"),
        ({"major_emotion": "위로", "emotion_tags": ["평온"]}, "위로"),
        ({"major_emotion": "Obsession", "emotion_tags": ["집착"]}, ""),
    ],
)
def test_refine_data_normalizes_the_llm_response(monkeypatch, payload, expected) -> None:
    _fake_gemini(monkeypatch, payload)
    result = refine_module.refine_data({"title": "곡", "artist": ["가수"]}, "가사", None)
    assert result["major_emotion"] == expected


# --- 비운 감정값이 곡을 떨어뜨리지 않는지 ------------------------------------------
# 임베딩 전 메타 검증은 빈 문자열을 문제로 보고, run_demo 기본값에서는 곡 폴더를 옮겨
# 색인에서 뺀다. 틀린 감정을 저장하느니 비우는 정책이 곡 전체를 잃는 결과가 되면 안 된다.

def _valid_record() -> dict:
    return {
        "song_id": "1",
        "metadata": {"title": "곡", "artist": ["가수"]},
        "community_feedback": {"major_emotion": "위로"},
    }


def _emotion_issues(record: dict) -> list:
    return [
        (issue.path, issue.reason)
        for issue in validate_meta_document(record, require_media_files=False)
        if issue.path == "community_feedback.major_emotion"
    ]


def test_blank_major_emotion_passes_meta_validation() -> None:
    record = _valid_record()
    record["community_feedback"]["major_emotion"] = ""
    assert _emotion_issues(record) == []


@pytest.mark.parametrize(
    "value, reason",
    [
        ("분석실패", "누락/실패 대체 문구"),   # LLM 호출이 전부 실패한 레코드는 계속 거른다
        (None, "null 값"),
    ],
)
def test_failure_markers_and_null_are_still_rejected(value, reason) -> None:
    record = _valid_record()
    record["community_feedback"]["major_emotion"] = value
    assert (("community_feedback.major_emotion", reason) in _emotion_issues(record))


def test_missing_major_emotion_key_is_still_rejected() -> None:
    record = _valid_record()
    del record["community_feedback"]["major_emotion"]
    assert _emotion_issues(record) == [("community_feedback.major_emotion", "필수 필드 누락")]


@pytest.mark.skipif(not CORPUS.exists(), reason="곡 데이터 없음")
def test_real_record_with_blank_emotion_has_no_new_issue() -> None:
    """실제 레코드에서 감정값만 비웠을 때 새로 생기는 문제가 없어야 한다."""
    record = next(
        (rec for rec in map(json.loads, CORPUS.open(encoding="utf-8"))
         if rec.get("song_id") == "31324607"),
        None,
    )
    if record is None:
        pytest.skip("로컬 곡 데이터에 31324607이 없음 (data/는 개인별로 다르다)")
    before = validate_meta_document(copy.deepcopy(record), require_media_files=False)
    record["community_feedback"]["major_emotion"] = ""
    after = validate_meta_document(record, require_media_files=False)
    assert after == before


# --- 색인 변경(한국어 변환·형용사형 확장)은 이번 PR에서 제외했다 -----------------------

def _sparse_passage(value, tags=()) -> str:
    from src.embedding.text.passage_builder import build_sparse_passage

    return build_sparse_passage({
        "metadata": {"title": "밤편지", "artist": ["아이유"]},
        "semantic_analysis": {"emotion_tags": list(tags)},
        "community_feedback": {"major_emotion": value},
    })


@pytest.mark.parametrize(
    "value, tags",
    [
        ("Sadness", ()),                       # 기존 영어 값: 번역·덧붙임 없이 그대로
        ("Excitement", ("설렘", "두근거림")),   # 태그 근거가 있어도 색인에서는 옮기지 않는다
        ("슬픔", ()),                           # 새 한국어 값: 형용사형('슬픈') 확장 없음
        ("그리움", ("그립다",)),
    ],
)
def test_passage_builder_adds_major_emotion_verbatim(value, tags) -> None:
    """감정값이 passage에 더하는 것은 저장값 그 한 줄뿐이어야 한다.

    같은 곡을 감정값 없이 만든 passage와 비교해 늘어난 줄만 본다. 번역해 바꾸든, 영어 뒤에
    한국어를 덧붙이든, 한국어 값에만 형용사형을 붙이든 모두 여기서 걸린다. 이 테스트가 깨지면
    제외했던 색인 변경이 다시 들어온 것이므로 변경 전후 측정을 함께 갖춰야 한다.
    """
    def content_lines(passage: str) -> Counter:
        # 블록 구분용 빈 줄은 내용이 아니므로 뺀다
        return Counter(line for line in passage.split("\n") if line.strip())

    added = content_lines(_sparse_passage(value, tags)) - content_lines(_sparse_passage("", tags))
    assert added == Counter({value: 1})
