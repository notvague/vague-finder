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


def test_candidates_for_reranker_pre_bonus_switch(monkeypatch):
    """(a) 스위치: 답변이 있을 때만 보너스 전 순서를 넘기고, 기본은 보너스 순서다."""
    from src.retrieval.clarify import candidates_for_reranker, reranker_input_order_mode
    pre = [("a", 0.3), ("b", 0.2), ("c", 0.1)]
    post = [("b", 0.5), ("a", 0.3), ("c", 0.1)]
    answers = [ClarifyAnswer(slot="genre", value="발라드")]
    monkeypatch.delenv("CLARIFY_RERANK_INPUT_ORDER", raising=False)
    assert reranker_input_order_mode() == "bonus"
    assert candidates_for_reranker(pre, post, answers) == post
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "pre_bonus")
    assert reranker_input_order_mode() == "pre_bonus"
    assert candidates_for_reranker(pre, post, answers) == pre
    assert candidates_for_reranker(pre, post, []) == post, "답변이 없으면 바꿀 것이 없다"


def test_runinfo_records_the_input_order_switch(monkeypatch):
    from src.retrieval.evaluate_search_accuracy import _ranking_switches
    monkeypatch.delenv("CLARIFY_RERANK_INPUT_ORDER", raising=False)
    assert _ranking_switches(None)["clarify_rerank_input_order"] == "bonus"
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "pre_bonus")
    assert _ranking_switches(None)["clarify_rerank_input_order"] == "pre_bonus"


def test_placebo_mode_shuffles_as_many_positions_as_the_bonus_changed_and_drops_answers(monkeypatch):
    """위약 대조: 보너스가 바꾼 자리 수만큼 고정 시드로 섞고, 답은 프롬프트에 넣지 않는다."""
    from src.retrieval.clarify import candidates_for_reranker, reranker_input_order_mode
    pre = [(str(i), 1.0 - i / 10) for i in range(10)]
    post = list(pre); post[2], post[5], post[7] = pre[5], pre[7], pre[2]  # 보너스가 3자리를 바꿨다
    answers = [ClarifyAnswer(slot="genre", value="발라드")]
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "placebo:3")
    assert reranker_input_order_mode() == "placebo:3"
    out = candidates_for_reranker(pre, post, answers)
    assert set(out) == set(pre) and out != pre, "같은 집합에서 순서만 바뀐다"
    assert sum(1 for a, b in zip(out, pre) if a != b) <= 3
    assert out == candidates_for_reranker(pre, post, answers), "같은 시드면 같은 순서"
    monkeypatch.setenv("CLARIFY_RERANK_INPUT_ORDER", "placebo:4")
    assert candidates_for_reranker(pre, post, answers) != out, "시드가 다르면 다른 순서"
    assert answers_for_reranker(_analysis(), answers) == [], "위약에서는 답을 프롬프트에 넣지 않는다"
    assert candidates_for_reranker(pre, post, []) == post, "답이 없으면 바꿀 것이 없다"
