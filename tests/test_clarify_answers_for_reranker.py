"""LLM 리랭커에 넘길 답변 선별 — 확인용 답 빼기(실험 스위치)와 type 슬롯 문구 스위치."""
from src.backend.schemas.query import ArtistTypeClue, ModalityWeights, QueryAnalysis, ReleaseEra
from src.backend.schemas.search import ClarifyAnswer
from src.retrieval.clarify import answer_confirms_analysis, answers_for_reranker


def _analysis(**kw) -> QueryAnalysis:
    base = dict(
        original_query="커버가 어둡고 차가운 느낌이었는데, 노래는 현악기 깔린 남자 발라드였어",
        intent_type="mood", vocal_gender="남성", genre="발라드",
        image_english_query="dark cold album cover", audio_english_query="",
        modality_weights=ModalityWeights(text=0.6, image=0.4, audio=0.0),
    )
    base.update(kw)
    return QueryAnalysis(**base)


def test_confirming_answer_is_the_same_value_as_the_analysis():
    a = _analysis()
    assert answer_confirms_analysis(a, ClarifyAnswer(slot="vocal_gender", value="남성"))
    assert answer_confirms_analysis(a, ClarifyAnswer(slot="genre", value="발라드"))
    # 분석에 없던 값은 새 정보다 — m402의 '솔로'·'2000년대'
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="type", value="솔로"))
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="release_era", value="2000년대"))
    # 다른 값(정정)도 새 정보다
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="vocal_gender", value="여성"))
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="vocal_gender", skipped=True))


def test_confirming_structured_slots():
    a = _analysis(artist_type=ArtistTypeClue(values=["솔로"], confidence=0.8),
                  release_era=ReleaseEra(start_year=2000, end_year=2009, confidence=0.7))
    assert answer_confirms_analysis(a, ClarifyAnswer(slot="type", value="솔로"))
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="type", value="그룹"))
    assert answer_confirms_analysis(a, ClarifyAnswer(slot="release_era", value="2000년대"))
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="release_era", value="1990년대"))


def test_answers_for_reranker_default_passes_everything(monkeypatch):
    monkeypatch.delenv("GEMINI_RERANK_CORRECTIONS", raising=False)
    a = _analysis()
    answers = [ClarifyAnswer(slot="vocal_gender", value="남성"), ClarifyAnswer(slot="type", value="솔로")]
    assert answers_for_reranker(a, answers) == answers


def test_answers_for_reranker_new_only_drops_confirming_answers(monkeypatch):
    monkeypatch.setenv("GEMINI_RERANK_CORRECTIONS", "new_only")
    a = _analysis()
    answers = [ClarifyAnswer(slot="vocal_gender", value="남성"), ClarifyAnswer(slot="type", value="솔로"),
               ClarifyAnswer(slot="genre", value="댄스")]
    kept = answers_for_reranker(a, answers)
    assert [(x.slot, x.value) for x in kept] == [("type", "솔로"), ("genre", "댄스")]


def test_type_slot_label_switch(monkeypatch):
    from src.retrieval.gemini_listwise_reranker import _corrections_block
    monkeypatch.delenv("GEMINI_RERANK_TYPE_SLOT_LABEL", raising=False)
    block = _corrections_block([ClarifyAnswer(slot="type", value="솔로"), ClarifyAnswer(slot="artist_type", value="그룹")])
    assert "- artist type (solo/group/duo/band): 솔로" in block
    assert "- artist_type: 그룹" in block, "스위치를 끄면 artist_type 슬롯 문구도 전과 같아야 한다 (PR #27 리뷰)"
    monkeypatch.setenv("GEMINI_RERANK_TYPE_SLOT_LABEL", "artist type")
    block = _corrections_block([ClarifyAnswer(slot="type", value="솔로"), ClarifyAnswer(slot="artist_type", value="그룹")])
    assert "- artist type: 솔로" in block and "- artist type: 그룹" in block and "solo/group" not in block


def test_values_merge_answer_ignores_are_not_confirming():
    """목록 밖 성별·유형, 연대로 못 읽는 값은 분석을 안 바꾸지만 '같은 값'이 아니라 정보 없음이다 (리뷰)."""
    a = _analysis(artist_type=ArtistTypeClue(values=["솔로"], confidence=0.8),
                  release_era=ReleaseEra(start_year=2000, end_year=2009, confidence=0.7))
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="vocal_gender", value="남자"))
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="type", value="혼성그룹"))
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="release_era", value="옛날"))
    assert not answer_confirms_analysis(a, ClarifyAnswer(slot="mood", value="슬픔"))


def test_runinfo_records_the_two_clarify_prompt_switches(monkeypatch):
    from src.retrieval.evaluate_search_accuracy import _ranking_switches
    monkeypatch.delenv("GEMINI_RERANK_CORRECTIONS", raising=False)
    monkeypatch.delenv("GEMINI_RERANK_TYPE_SLOT_LABEL", raising=False)
    d = _ranking_switches(None)
    assert d["gemini_rerank_corrections"] == "all"
    assert d["gemini_rerank_type_slot_label"] == "artist type (solo/group/duo/band)"
    monkeypatch.setenv("GEMINI_RERANK_CORRECTIONS", "new_only")
    monkeypatch.setenv("GEMINI_RERANK_TYPE_SLOT_LABEL", "artist type")
    d = _ranking_switches(None)
    assert (d["gemini_rerank_corrections"], d["gemini_rerank_type_slot_label"]) == ("new_only", "artist type")
