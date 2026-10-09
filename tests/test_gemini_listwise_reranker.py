import json
from pathlib import Path

import pytest

from src.backend.schemas.search import MatchingTrack
from src.retrieval import gemini_listwise_reranker as gemini_module
from src.retrieval.explain import RULE_LABELS
from src.retrieval.gemini_listwise_reranker import (
    GeminiListwiseReranker,
    GeminiListwiseRerankerConfig,
    _rare_fact_reason,
)


def _track(song_id, title, score, *, lyric_match_type=None, lyric_match_score=None):
    return MatchingTrack(
        id=song_id,
        title=title,
        artist="artist",
        score=score,
        retrieval_score=score,
        lyric_match_type=lyric_match_type,
        lyric_match_score=lyric_match_score,
    )


def test_rare_fact_detector_targets_media_and_distinctive_performance_only():
    assert _rare_fact_reason(
        "옛날 짱구 애니메이션 이별 장면에 OST로 삽입된 남자 노래"
    ) == "media_context"
    assert _rare_fact_reason(
        "도입부 휘파람이 나고 남자는 랩, 여자는 노래하는 듀엣"
    ) == "performance_fact"
    # 제목 구조는 deterministic rescue가 담당하므로 rare web judge는 끈다.
    assert _rare_fact_reason("제목이 알파벳 세 글자인 여자 그룹 댄스곡") == ""
    assert _rare_fact_reason("잔잔한 남자 발라드") == ""


def test_parse_rare_fact_response_accepts_fenced_json():
    raw = '''```json
    {
      "verification_confidence": 0.94,
      "verdicts": [
        {
          "id": "14",
          "support": 0.97,
          "matched_clues": ["도입부 휘파람", "남성 랩과 여성 보컬"],
          "contradictions": [],
          "evidence": "검색 근거 확인"
        }
      ]
    }
    ```'''
    verdicts, confidence = GeminiListwiseReranker._parse_rare_fact_response(
        raw, ["14", "15"]
    )
    assert confidence == 0.94
    assert verdicts["14"]["support"] == 0.97
    assert len(verdicts["14"]["matched_clues"]) == 2


def test_performance_rare_fact_rescue_moves_rank14_to_rank9():
    original = [_track(str(i), f"song-{i}", 1.0 / i) for i in range(1, 31)]
    reranked = list(original)
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            rare_fact_insert_rank=9,
            rare_fact_min_support=0.86,
            rare_fact_min_confidence=0.78,
            rare_fact_min_margin=0.06,
        ),
        client=object(),
    )
    verdicts = {
        "14": {
            "support": 0.97,
            "matched_clues": ["도입부 휘파람", "남성 랩 + 여성 보컬"],
            "contradictions": [],
        },
        "2": {
            "support": 0.60,
            "matched_clues": ["남녀 보컬"],
            "contradictions": [],
        },
    }
    result = reranker._apply_rare_fact_rescue(
        original, reranked, verdicts, 0.95, "performance_fact"
    )
    assert [t.id for t in result].index("14") == 8


def test_media_context_one_direct_fact_is_enough_to_rescue_rank23():
    original = [_track(str(i), f"song-{i}", 1.0 / i) for i in range(1, 31)]
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(rare_fact_insert_rank=9),
        client=object(),
    )
    verdicts = {
        "23": {
            "support": 0.96,
            "matched_clues": ["특정 애니메이션 에피소드 삽입곡으로 확인"],
            "contradictions": [],
        }
    }
    result = reranker._apply_rare_fact_rescue(
        original, original, verdicts, 0.92, "media_context"
    )
    assert [t.id for t in result].index("23") == 8


def test_ambiguous_web_verdict_does_not_rescue():
    original = [_track(str(i), f"song-{i}", 1.0 / i) for i in range(1, 20)]
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            rare_fact_insert_rank=9,
            rare_fact_min_margin=0.06,
        ),
        client=object(),
    )
    verdicts = {
        "14": {
            "support": 0.92,
            "matched_clues": ["휘파람", "듀엣"],
            "contradictions": [],
        },
        "15": {
            "support": 0.89,
            "matched_clues": ["휘파람", "듀엣"],
            "contradictions": [],
        },
    }
    result = reranker._apply_rare_fact_rescue(
        original, original, verdicts, 0.95, "performance_fact"
    )
    assert [t.id for t in result].index("14") == 13


def test_strong_phonetic_anchor_skips_rare_fact_web_verification():
    tracks = [
        _track("correct", "correct", 1.0, lyric_match_type="phonetic", lyric_match_score=0.85),
        _track("other", "other", 0.9),
    ]
    assert GeminiListwiseReranker._has_strong_surface_lyric_anchor(tracks) is True


class _FlowFakeModels:
    def generate_content(self, **kwargs):
        contents = kwargs.get("contents", "")
        if "high-precision fact verifier" in contents:
            # q200형: 14번만 두 희소 사실이 모두 검색으로 확인됐다고 가정.
            ids = []
            import json as _json
            marker = "Candidates (JSON):\n"
            if marker in contents:
                tail = contents.split(marker, 1)[1].split("\n\nReturn:", 1)[0]
                try:
                    ids = [str(x["id"]) for x in _json.loads(tail)]
                except Exception:
                    ids = []
            verdicts = []
            for song_id in ids:
                if song_id == "14":
                    verdicts.append({
                        "id": song_id,
                        "support": 0.98,
                        "matched_clues": ["도입부 휘파람", "남성 랩과 여성 보컬"],
                        "contradictions": [],
                        "evidence": "희소 사실 직접 확인",
                    })
                else:
                    verdicts.append({
                        "id": song_id,
                        "support": 0.20,
                        "matched_clues": [],
                        "contradictions": [],
                        "evidence": "",
                    })
            return type("Response", (), {"text": _json.dumps({
                "verification_confidence": 0.96,
                "verdicts": verdicts,
            }, ensure_ascii=False)})()

        # listwise 자체는 원 retrieval 순서를 그대로 반환하도록 한다.
        import json as _json
        marker = "Candidates (JSON):\n"
        tail = contents.split(marker, 1)[1].split("\n\nReturn one JSON", 1)[0]
        candidates = _json.loads(tail)
        payload = {
            "query_confidence": 0.9,
            "ranking": [
                {"id": str(c["id"]), "relevance": 0.5}
                for c in candidates
            ],
        }
        return type("Response", (), {"text": _json.dumps(payload, ensure_ascii=False)})()


class _FlowFakeClient:
    def __init__(self):
        self.models = _FlowFakeModels()


def test_full_rerank_flow_rare_fact_verifier_rescues_q200_shape(monkeypatch):
    _install_fake_genai_types(monkeypatch)

    tracks = [_track(str(i), f"song-{i}", 1.0 / i) for i in range(1, 31)]
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            rerank_weight=0.85,
            passes=1,
            use_search_grounding=False,
            rare_fact_verification=True,
            rare_fact_batch_size=15,
            rare_fact_insert_rank=9,
        ),
        client=_FlowFakeClient(),
    )
    query = "도입부에 휘파람이 나오고 남자는 랩, 여자는 노래하는 남녀 듀엣이었어"
    result = reranker.rerank(query, tracks, 30)
    assert [t.id for t in result].index("14") == 8


# ---------------------------------------------------------------------------
# 백엔드 선택 · 이미지 지배 질의 생략 범위
# ---------------------------------------------------------------------------

def _image_dominant_analysis():
    from src.backend.schemas.query import ModalityWeights, QueryAnalysis

    return QueryAnalysis(
        original_query="초록 바탕에 흰 도형이 있는 앨범 커버",
        intent_type="mood",
        has_visual_clue=True,
        image_english_query="green cover with white geometric shapes",
        audio_english_query="",
        modality_weights=ModalityWeights(text=0.1, image=0.9, audio=0.0),
    )


def test_image_dominant_skip_applies_only_to_gemini_listwise():
    # 기존 Cross-Encoder는 기준선부터 이미지 질의도 리랭킹해 왔다. 함께 끄면
    # 백엔드를 바꾸지 않은 사람의 결과까지 달라진다.
    from src.retrieval.reranker import MusicReranker
    from src.retrieval.search_router import should_skip_rerank_for_image

    analysis = _image_dominant_analysis()
    assert should_skip_rerank_for_image(
        GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), api_key="x"), analysis
    )
    assert not should_skip_rerank_for_image(MusicReranker(), analysis)


# ---------------------------------------------------------------------------
# 리뷰 2차 — 공통 비활성화 · 제목 구조 보호 · 희소 사실 점수 차이 · 재질문 답변
# ---------------------------------------------------------------------------

def test_common_reranker_switch_also_disables_gemini(monkeypatch):
    # 기존 설정으로 리랭킹을 꺼 둔 환경이 백엔드만 바꿨다고 다시 켜지면 안 된다.
    monkeypatch.delenv("GEMINI_RERANK_ENABLED", raising=False)
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    assert not GeminiListwiseReranker(api_key="x").enabled

    monkeypatch.setenv("RERANKER_ENABLED", "true")
    assert GeminiListwiseReranker(api_key="x").enabled
    monkeypatch.setenv("GEMINI_RERANK_ENABLED", "false")
    assert not GeminiListwiseReranker(api_key="x").enabled


def _titled(song_id, title):
    return _track(song_id, title, 1.0)


def test_title_shape_rescue_keeps_matches_already_inside_boundary():
    # 회귀: 경계 밖에 일치 곡이 하나라도 있으면 일치 곡을 전부 빼서 경계에 다시
    # 넣어, 1위였던 TTL이 9위로 내려갔다.
    tracks = (
        [_titled("ttl", "TTL")]
        + [_titled(str(i), f"노래{i}") for i in range(2, 20)]
        + [_titled("lie", "Lie")]
        + [_titled(str(i), f"곡{i}") for i in range(21, 31)]
    )
    out = [t.id for t in GeminiListwiseReranker._apply_title_shape_rescue(
        "제목이 알파벳 세 글자였어", tracks, top_k_boundary=10)]

    assert out[0] == "ttl"
    assert out.index("lie") < 10
    assert len(out) == len(tracks) and set(out) == {t.id for t in tracks}
    # 밀려난 것은 경계 안의 가장 아래 비일치 곡 하나뿐이다.
    assert out[:9] == [t.id for t in tracks[:9]]
    assert out[10] == tracks[9].id


def test_title_shape_rescue_is_noop_when_all_matches_are_inside():
    tracks = [_titled("a", "노래"), _titled("ttl", "TTL")] + [
        _titled(str(i), f"곡{i}") for i in range(3, 31)
    ]
    out = GeminiListwiseReranker._apply_title_shape_rescue(
        "제목이 알파벳 세 글자였어", tracks, top_k_boundary=10)
    assert [t.id for t in out] == [t.id for t in tracks]


def _rare_verdict(support, clues=1):
    return {"support": support, "matched_clues": ["단서"] * clues, "contradictions": []}


def test_rare_fact_margin_counts_competitors_that_failed_eligibility():
    # 회귀: 0.87 vs 0.85인데 0.85가 최소 support(0.86)에서 먼저 빠져
    # 차이가 0.87 − 0으로 계산되고 9위로 승격됐다.
    reranker = GeminiListwiseReranker(config=GeminiListwiseRerankerConfig(), api_key="x")
    tracks = [_track(str(i), f"곡{i}", 1.0) for i in range(1, 31)]

    def rank_of_25(verdicts):
        out = reranker._apply_rare_fact_rescue(tracks, tracks, verdicts, 0.9, "media_context")
        return [t.id for t in out].index("25") + 1

    assert rank_of_25({"25": _rare_verdict(0.87), "3": _rare_verdict(0.85)}) == 25
    # 경쟁 후보가 단서 수로 탈락해도 support는 경쟁 점수로 센다.
    assert rank_of_25({"25": _rare_verdict(0.87), "3": _rare_verdict(0.85, clues=0)}) == 25
    # 차이가 충분하면 승격은 그대로 된다.
    assert rank_of_25({"25": _rare_verdict(0.95), "3": _rare_verdict(0.80)}) == 9


class _PromptRecordingModels:
    def __init__(self):
        self.prompts = []

    def generate_content(self, model, contents, config=None):
        import json as _json

        self.prompts.append(contents)
        tail = contents.split("Candidates (JSON):\n", 1)[1].split("\n\nReturn one JSON", 1)[0]
        ids = [str(c["id"]) for c in _json.loads(tail)]
        payload = {"query_confidence": 0.9,
                   "ranking": [{"id": i, "relevance": 0.5} for i in ids]}
        return type("Response", (), {"text": _json.dumps(payload)})()


def test_clarify_answers_reach_the_listwise_prompt():
    # 회귀: 프롬프트에 최초 질의만 들어가 "여성"으로 정정해도 질의의 "남자가 부르는"을
    # 따랐다 (dev c705: 후보 2위 → Top-10 밖).
    from src.backend.schemas.search import ClarifyAnswer

    models = _PromptRecordingModels()
    client = type("Client", (), {"models": models})()
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(passes=1, use_search_grounding=False,
                                            rare_fact_verification=False),
        client=client,
    )
    tracks = [_track(str(i), f"곡{i}", 1.0 / i) for i in range(1, 6)]
    query = "남자가 부르는 잔잔한 발라드"

    reranker.rerank(query, tracks, 5, answers=[
        ClarifyAnswer(slot="vocal_gender", value="여성"),
        ClarifyAnswer(slot="genre", skipped=True),
    ])
    prompt = models.prompts[-1]
    assert "User corrections" in prompt
    assert "- vocal gender: 여성" in prompt
    assert "genre:" not in prompt.split("User corrections", 1)[1].split("Candidates", 1)[0]

    reranker.rerank(query, tracks, 5)
    assert "User corrections" not in models.prompts[-1]


def test_call_reranker_passes_answers_only_to_rerankers_that_use_them():
    from src.backend.schemas.search import ClarifyAnswer
    from src.retrieval.search_router import call_reranker

    class _CrossEncoderLike:          # answers 인자가 없는 기존 시그니처
        def rerank(self, query, tracks, top_k):
            return ["ce"]

    class _AnswerAware:
        uses_clarify_answers = True

        def rerank(self, query, tracks, top_k, answers=None):
            return answers

    answers = [ClarifyAnswer(slot="vocal_gender", value="여성")]
    assert call_reranker(_CrossEncoderLike(), "q", [], 10, answers) == ["ce"]
    assert call_reranker(_AnswerAware(), "q", [], 10, answers) == answers


# ---------------------------------------------------------------------------
# 실행 기록 — 모델이 한 일과 규칙이 한 일을 섞지 않는다
# ---------------------------------------------------------------------------

def test_parse_response_keeps_the_model_reason():
    """프롬프트가 요구하는 reason을 버리지 않는다.

    검증된 근거는 아니지만 "모델이 무엇을 보고 그렇게 말했나"를 확인할 때 유일한
    단서다. 파서에서 버리면 아예 되살릴 방법이 없다.
    """
    raw = json.dumps(
        {
            "query_confidence": 0.8,
            "ranking": [
                {"id": "2", "relevance": 0.9, "reason": "비 오는 새벽 정서가 일치"},
                {"id": "1", "relevance": 0.4},
            ],
        }
    )
    order, scores, confidence, reasons = GeminiListwiseReranker._parse_response(
        raw, ["1", "2"]
    )

    assert order == ["2", "1"]
    assert confidence == 0.8
    assert reasons == {"2": "비 오는 새벽 정서가 일치"}
    assert "1" not in reasons, "말하지 않은 이유를 만들어 내면 안 된다"


def test_aggregate_passes_keeps_reasons_from_every_pass():
    """pass마다 다른 문장이 오면 전부 남긴다.

    최종 순서는 pass 평균에서 나온다. 한 문장만 골라 대표로 쓰면 그 순서를 만든
    나머지 근거를 숨기는 셈이다.
    """
    pass_results = [
        (["1", "2"], {"1": 0.9}, 0.8, {"1": "가사 구절 일치"}),
        (["2", "1"], {"2": 0.8}, 0.6, {"1": "제목 구조 일치", "2": "분위기 유사"}),
    ]
    _, _, _, reasons = GeminiListwiseReranker._aggregate_passes(["1", "2"], pass_results)

    assert reasons["1"] == ["가사 구절 일치", "제목 구조 일치"]
    assert reasons["2"] == ["분위기 유사"]


def test_fuse_records_the_mix_that_made_the_score():
    """기록된 합성식이 실제 점수를 만든 그 식이어야 한다.

    이 백엔드는 모델 순서와 **검색 순위 점수**를 섞는다. 그 비율을 남기지 않으면
    "모델이 1위로 뽑았으니 1위"라는 잘못된 설명이 나온다.
    """
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            rerank_weight=0.80,
            low_confidence_threshold=0.55,
        ),
        client=object(),
    )
    tracks = [_track("1", "a", 0.9), _track("2", "b", 0.8), _track("3", "c", 0.7)]
    mixes: dict = {}
    out = reranker._fuse(tracks, ["3", "1", "2"], {"3": 0.9, "1": 0.5, "2": 0.4}, 0.9, mixes)

    by_id = {str(track.id): track for track in out}
    assert set(mixes) == {"1", "2", "3"}
    for song_id, mix in mixes.items():
        assert mix.backend == "gemini_listwise"
        assert mix.final == pytest.approx(by_id[song_id].score), "기록 = 실제 점수"
        recomputed = (
            mix.weight * mix.rerank_component
            + (1.0 - mix.weight) * mix.retrieval_component
        )
        assert recomputed == pytest.approx(mix.final)
        assert mix.weight == pytest.approx(0.80), "확신도가 높으면 깎이지 않는다"
        assert mix.confidence == pytest.approx(1.0)


def test_fuse_records_the_reduced_weight_when_confidence_is_low():
    """확신도가 낮아 가중치를 깎았으면 그 사실이 기록에 남아야 한다."""
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            rerank_weight=0.80,
            low_confidence_threshold=0.55,
        ),
        client=object(),
    )
    tracks = [_track("1", "a", 0.9), _track("2", "b", 0.8)]
    mixes: dict = {}
    reranker._fuse(tracks, ["2", "1"], {}, 0.11, mixes)

    mix = mixes["1"]
    assert mix.configured_weight == pytest.approx(0.80)
    assert mix.confidence < 1.0
    assert mix.weight < mix.configured_weight
    assert mix.weight == pytest.approx(mix.configured_weight * mix.confidence)


def test_title_shape_rescue_reports_promoted_and_demoted():
    """규칙이 올린 곡과 **밀려난 곡**을 모두 남긴다.

    밀려난 곡을 적지 않으면 그 하락이 모델의 판단으로 읽힌다. 실제로는 규칙이
    경계 안의 자리를 뺀 것이다.
    """
    inside = [_titled(str(i), f"곡{i}") for i in range(1, 11)]
    outside = [_titled("99", "TTL")]
    notes: list = []

    out = GeminiListwiseReranker._apply_title_shape_rescue(
        "제목이 알파벳 세 글자인 노래",
        [*inside, *outside],
        top_k_boundary=10,
        order_notes=notes,
    )

    assert [str(track.id) for track in out].index("99") < 10, "규칙이 경계 안으로 올렸다"
    by_rule = {rule: (song_id, detail) for song_id, rule, detail in notes}
    assert "gemini_title_shape_rescue" in by_rule
    assert by_rule["gemini_title_shape_rescue"][0] == "99"
    assert "gemini_title_shape_demoted" in by_rule, "밀려난 곡도 기록해야 한다"
    assert by_rule["gemini_title_shape_demoted"][0] == "10", "경계 맨 아래가 밀린다"


def test_rare_fact_rescue_reports_the_move_it_made():
    original = [_track(str(i), f"song-{i}", 1.0 / i) for i in range(1, 31)]
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            rare_fact_insert_rank=9,
            rare_fact_min_support=0.86,
            rare_fact_min_confidence=0.78,
            rare_fact_min_margin=0.06,
        ),
        client=object(),
    )
    verdicts = {
        "14": {"support": 0.97, "matched_clues": ["도입부 휘파람", "남성 랩"], "contradictions": []}
    }
    notes: list = []
    out = reranker._apply_rare_fact_rescue(
        original, list(original), verdicts, 0.94, "performance_fact", order_notes=notes
    )

    assert str(out[8].id) == "14"
    assert len(notes) == 1
    song_id, rule, detail = notes[0]
    assert (song_id, rule) == ("14", "gemini_rare_fact_rescue")
    assert "14위 → 9위" in detail, "모델 순서에서 어디로 옮겼는지가 설명이다"


def test_lyric_anchor_protect_reports_only_a_real_move():
    """이미 1위였으면 규칙이 한 일이 없다 — 기록도 없어야 한다."""
    anchor = _track("1", "정답", 0.9, lyric_match_type="exact", lyric_match_score=0.95)
    other = _track("2", "다른 곡", 0.8)

    moved: list = []
    GeminiListwiseReranker._protect_strong_surface_lyric_anchor(
        [anchor, other], [other, anchor], order_notes=moved
    )
    assert [note[1] for note in moved] == ["gemini_lyric_anchor_protect"]
    assert "2위 → 1위" in moved[0][2]

    noop: list = []
    GeminiListwiseReranker._protect_strong_surface_lyric_anchor(
        [anchor, other], [anchor, other], order_notes=noop
    )
    assert noop == [], "옮기지 않았는데 규칙이 배치했다고 하면 안 된다"


def test_every_rescue_rule_this_backend_emits_has_a_label():
    """기록에 남는 규칙명은 전부 사람이 읽는 말이 있어야 한다."""
    emitted = {
        "gemini_lyric_anchor_protect",
        "gemini_title_shape_rescue",
        "gemini_title_shape_demoted",
        "gemini_rare_fact_rescue",
    }
    source = Path(gemini_module.__file__).read_text(encoding="utf-8")
    for rule in emitted:
        assert f'"{rule}"' in source, f"{rule}을 더 이상 쓰지 않으면 테스트도 지워야 한다"
        assert rule in RULE_LABELS, f"{rule}에 라벨이 없다"


def test_default_config_is_single_pass_without_search(monkeypatch):
    """2026-10-09 전환: 1패스·Search 끔(results_v32~v34)에 희소 사실 검증 끔(results_v35)이 기본값이어야 한다."""
    for k in ("GEMINI_RERANK_PASSES", "GEMINI_RERANK_USE_SEARCH", "GEMINI_RERANK_RARE_FACT_VERIFY"):
        monkeypatch.delenv(k, raising=False)
    cfg = GeminiListwiseRerankerConfig.from_env()
    assert cfg.passes == 1
    assert cfg.use_search_grounding is False
    assert cfg.rare_fact_verification is False  # results_v35: 구조 규칙이 오답만 올렸고 외부 맥락 질의 +11초
    assert cfg.rerank_weight == 0.85 and cfg.max_candidates == 30


def test_get_reranker_defaults_to_listwise_and_falls_back_without_key(monkeypatch):
    from src.backend.api import dependencies as deps
    from src.retrieval.reranker import MusicReranker

    monkeypatch.delenv("RERANKER_BACKEND", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GEMINI_RETRIEVAL_BACKEND", raising=False)  # 개발자 .env의 검색 경로 고정과 무관하게
    monkeypatch.setenv("GEMINI_API_KEY", "시험용")
    deps.get_reranker.cache_clear()
    assert isinstance(deps.get_reranker(), GeminiListwiseReranker)

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    deps.get_reranker.cache_clear()
    assert isinstance(deps.get_reranker(), MusicReranker), "키가 없으면 CE로 내려가야 한다"

    # Vertex 설정만 있어도 listwise다 — 키를 주석 처리하고 Vertex로 옮긴 뒤 CE로 조용히 내려가면 안 된다
    monkeypatch.setenv("GCP_PROJECT_ID", "시험-프로젝트")
    deps.get_reranker.cache_clear()
    assert isinstance(deps.get_reranker(), GeminiListwiseReranker)
    # 검색을 AI Studio로 고정했는데 키가 없으면 Vertex로 바꿔 타지 않고 CE로 내려간다(로그에 남는다)
    monkeypatch.setenv("GEMINI_RETRIEVAL_BACKEND", "api_key")
    deps.get_reranker.cache_clear()
    assert isinstance(deps.get_reranker(), MusicReranker)
    monkeypatch.delenv("GEMINI_RETRIEVAL_BACKEND", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    monkeypatch.setenv("GEMINI_API_KEY", "시험용")
    monkeypatch.setenv("RERANKER_BACKEND", "cross_encoder")
    deps.get_reranker.cache_clear()
    assert isinstance(deps.get_reranker(), MusicReranker)
    deps.get_reranker.cache_clear()


def test_default_config_has_request_timeout_and_budget(monkeypatch):
    for k in ("GEMINI_RERANK_TIMEOUT_SECONDS", "GEMINI_RERANK_BUDGET_SECONDS"):
        monkeypatch.delenv(k, raising=False)
    cfg = GeminiListwiseRerankerConfig.from_env()
    assert cfg.request_timeout_seconds == 20.0 and cfg.time_budget_seconds == 30.0
    assert cfg.pass_reserve_seconds == 10.0
    assert GeminiListwiseReranker(config=cfg, client=object())._pass_reserve_seconds() == 10.0
    # 예산이 작으면 절반까지만 남긴다 — 앞 단계가 0초가 되지 않는다
    small = GeminiListwiseRerankerConfig(time_budget_seconds=4.0, pass_reserve_seconds=10.0)
    assert GeminiListwiseReranker(config=small, client=object())._pass_reserve_seconds() == 2.0


def test_client_is_built_with_http_timeout(monkeypatch):
    """HTTP 제한이 없으면 멈춘 응답을 무기한 기다린다 (PR 리뷰 P1)."""
    import google.genai as genai_mod
    captured = {}

    def fake_client(api_key, http_options=None):
        captured["timeout_ms"] = getattr(http_options, "timeout", None)
        return object()

    monkeypatch.setattr(genai_mod, "Client", fake_client)
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(request_timeout_seconds=7.5), api_key="시험용",
    )
    reranker._gemini  # noqa: B018 — 클라이언트 생성
    assert captured["timeout_ms"] == 7500


class _CountingModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        raise AssertionError("예산을 넘긴 뒤에는 Gemini를 부르면 안 된다")


def test_exhausted_budget_returns_retrieval_order_without_calling_gemini():
    """전체 예산을 넘기면 남은 호출 없이 검색 순서를 돌려주고 상태는 failed다."""
    from src.retrieval.gemini_listwise_reranker import RERANK_FAILED
    models = _CountingModels()
    client = type("Client", (), {"models": models})()
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            passes=2, use_search_grounding=True, rare_fact_verification=True,
            time_budget_seconds=1.0,
        ),
        client=client,
    )
    # 예산이 이미 지난 것처럼 만든다.
    reranker._over_budget = staticmethod(lambda deadline: True)
    tracks = [_track(str(i), f"곡{i}", 1.0 / i) for i in range(1, 6)]
    run = reranker.rerank_run("드라마에 나온 발라드", tracks, 5)
    assert run.status == RERANK_FAILED
    assert [t.id for t in run.tracks] == [t.id for t in tracks]
    assert models.calls == 0


def test_late_response_after_budget_is_discarded(monkeypatch):
    """예산 뒤에 도착한 응답은 반영하지 않는다 — 검색 순서 + failed (PR 리뷰 P2)."""
    import src.retrieval.gemini_listwise_reranker as mod
    from src.retrieval.gemini_listwise_reranker import RERANK_FAILED
    import json as _json

    clock = {"now": 0.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["now"])

    class _LateModels:
        def generate_content(self, model, contents, config=None):
            clock["now"] += 31.0  # 응답을 받는 데 31초 — 30초 예산을 넘겼다
            tail = contents.split("Candidates (JSON):\n", 1)[1].split("\n\nReturn one JSON", 1)[0]
            ids = [str(c["id"]) for c in _json.loads(tail)]
            payload = {"query_confidence": 0.9,
                       "ranking": [{"id": i, "relevance": 1.0 - 0.1 * k} for k, i in enumerate(reversed(ids))]}
            return type("Response", (), {"text": _json.dumps(payload)})()

    client = type("Client", (), {"models": _LateModels()})()
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(passes=2, use_search_grounding=False,
                                            rare_fact_verification=False, time_budget_seconds=30.0),
        client=client,
    )
    tracks = [_track("a", "곡a", 1.0), _track("b", "곡b", 0.5)]
    run = reranker.rerank_run("잔잔한 발라드", tracks, 2)
    assert run.status == RERANK_FAILED
    assert [t.id for t in run.tracks] == ["a", "b"], "뒤집힌 모델 순서가 반영되면 안 된다"


def test_hung_async_call_is_cancelled_by_wall_clock():
    """읽기 제한은 조각마다 초기화된다 — 벽시계 제한이 진행 중인 호출을 끊어야 한다 (PR 리뷰 P1)."""
    import asyncio as _asyncio
    import time as _time
    from src.retrieval.gemini_listwise_reranker import RERANK_FAILED

    state = {"cancelled": False, "closed": False}

    class _HungAsyncModels:
        async def generate_content(self, model, contents, config=None):
            try:
                await _asyncio.sleep(30)
            except _asyncio.CancelledError:
                state["cancelled"] = True
                raise

    class _Aio:
        models = _HungAsyncModels()

        async def aclose(self):
            state["closed"] = True

    client = type("Client", (), {"models": object(), "aio": _Aio()})()
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            passes=1, use_search_grounding=False, rare_fact_verification=False,
            request_timeout_seconds=0.3, time_budget_seconds=0.5, max_retries=1,
        ),
        client=client,
    )
    tracks = [_track("a", "곡a", 1.0), _track("b", "곡b", 0.5)]
    started = _time.monotonic()
    run = reranker.rerank_run("잔잔한 발라드", tracks, 2)
    elapsed = _time.monotonic() - started
    assert run.status == RERANK_FAILED and [t.id for t in run.tracks] == ["a", "b"]
    assert state["cancelled"], "wait_for가 진행 중인 호출을 취소해야 한다"
    assert elapsed < 2.0, f"벽시계 제한을 넘겨 {elapsed:.2f}초 기다렸다"



def _install_fake_genai_types(monkeypatch):
    """테스트 컨테이너에 google-genai가 없어도 Search tool 타입만 가짜로 주입한다."""
    import sys
    import types as pytypes

    class _GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _GoogleSearch:
        pass

    class _Tool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.types = pytypes.SimpleNamespace(
        GenerateContentConfig=_GenerateContentConfig, GoogleSearch=_GoogleSearch, Tool=_Tool,
    )
    fake_google = pytypes.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)


def test_rare_fact_verification_leaves_budget_for_the_main_pass(monkeypatch):
    """검증 배치가 예산을 다 써도 본 패스는 한 번 돈다 (PR 리뷰 3차).

    30곡 → 검증 배치 2회. 배치 하나가 12초씩이면 같은 마감으로는 24초가 지나 본 패스가
    6초 안에 끝나야 하고, 더 느리면 0회가 된다. 앞 단계는 본 패스 몫 10초를 뺀 20초 마감을
    받으므로 배치 2(24초 도착)는 버려지고, 배치 1의 판정은 남아 본 패스(3초)와 함께 반영된다.
    """
    import src.retrieval.gemini_listwise_reranker as mod
    from src.retrieval.gemini_listwise_reranker import RERANK_APPLIED

    _install_fake_genai_types(monkeypatch)
    clock = {"now": 0.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["now"])
    calls = {"verify": [], "listwise": []}
    inner = _FlowFakeModels()

    class _SlowVerifierModels:
        def generate_content(self, **kwargs):
            contents = kwargs.get("contents", "")
            if "high-precision fact verifier" in contents:
                calls["verify"].append(clock["now"])
                clock["now"] += 12.0
            else:
                calls["listwise"].append(clock["now"])
                clock["now"] += 3.0
            return inner.generate_content(**kwargs)

    client = type("Client", (), {"models": _SlowVerifierModels()})()
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            passes=1, use_search_grounding=False, rare_fact_verification=True,
            rare_fact_batch_size=15, rare_fact_insert_rank=9,
            request_timeout_seconds=20.0, time_budget_seconds=30.0, pass_reserve_seconds=10.0,
        ),
        client=client,
    )
    tracks = [_track(str(i), f"song-{i}", 1.0 / i) for i in range(1, 31)]
    run = reranker.rerank_run("도입부에 휘파람이 나오고 남자는 랩, 여자는 노래하는 남녀 듀엣이었어", tracks, 30)

    assert calls["verify"] == [0.0, 12.0], "배치 2는 앞 단계 마감(20초) 안에 시작한다"
    assert calls["listwise"] == [24.0], "본 패스가 한 번 돌아야 한다"
    assert run.status == RERANK_APPLIED
    assert [t.id for t in run.tracks].index("14") == 8, "배치 1의 판정은 버려지지 않는다"


def test_same_deadline_for_verification_would_starve_the_main_pass(monkeypatch):
    """대조: 본 패스 몫이 0이면 검증 배치 2회가 예산을 다 써 결과가 failed로 끝난다."""
    import src.retrieval.gemini_listwise_reranker as mod
    from src.retrieval.gemini_listwise_reranker import RERANK_FAILED

    _install_fake_genai_types(monkeypatch)
    clock = {"now": 0.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["now"])
    inner = _FlowFakeModels()
    listwise_calls = []

    class _SlowVerifierModels:
        def generate_content(self, **kwargs):
            if "high-precision fact verifier" in kwargs.get("contents", ""):
                clock["now"] += 15.0
            else:
                listwise_calls.append(clock["now"])
            return inner.generate_content(**kwargs)

    client = type("Client", (), {"models": _SlowVerifierModels()})()
    reranker = GeminiListwiseReranker(
        config=GeminiListwiseRerankerConfig(
            passes=1, use_search_grounding=False, rare_fact_verification=True,
            rare_fact_batch_size=15, time_budget_seconds=30.0, pass_reserve_seconds=0.0,
        ),
        client=client,
    )
    tracks = [_track(str(i), f"song-{i}", 1.0 / i) for i in range(1, 31)]
    run = reranker.rerank_run("드라마 OST였는데 남자는 랩, 여자는 노래하는 듀엣", tracks, 30)
    assert listwise_calls == [] and run.status == RERANK_FAILED
