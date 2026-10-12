"""A grounded rewrite must not erase an unnamed media memory's constraints.

All public test queries are synthetic; private evaluation IDs/answers are not
used to choose behavior. Live corpus OFF/ON results are a separate run.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.namuwiki.install_context_media_specificity import (
    edit_context_media_safeguards, install,
)
from src.backend.schemas.query import ContextClue
from src.retrieval.context_clue_specificity import preserve_context_media_search
from src.retrieval.context_evidence import (
    context_candidate_matches_media_description, context_evidence_for_result,
    select_context_fact_for_result,
)
from src.retrieval.context_query import (
    apply_context_query_safeguards, context_media_description_requires_support,
    context_media_target_terms, context_search_queries,
)
from src.retrieval.query_analyzer import QueryAnalyzer, rule_fallback


DETAILS = (
    "랩으로 유명한 남자 가수가 랩 없이 부른 영화 삽입곡인데 1990년대 후반 곡이었어",
    "여자 가수가 독창으로 부른 드라마 삽입곡",
    "2000년대 중반에 방영한 드라마 삽입곡",
    "옛날 겨울 드라마 삽입곡",
    "특별한 후각을 쓰는 탐정이 등장하는 드라마 삽입곡",
    "군인이 파병 나가는 드라마 삽입곡",
    "드라마 마지막 회의 이별 장면에 나온 삽입곡",
    "비 오는 골목에서 두 주인공이 헤어지는 영화 삽입곡",
    "게임에서 보스가 등장할 때 나온 삽입곡",
    "애니메이션에서 친구가 떠날 때 나온 삽입곡",
)
BROAD = ("OST", "삽입곡", "OST 삽입곡", "배경음악", "BGM", "주제가")


def cue(query, *, target="", relation="삽입곡·배경음악"):
    return ContextClue(target=target, relation=relation, search_query=query, confidence=0.8)


def payload(search, *, relation="삽입곡·배경음악", target=""):
    return dict(context_clues=[dict(target=target, relation=relation,
                                   search_query=search, confidence=0.8)],
                audio_english_query="unchanged audible description",
                image_english_query="unchanged artwork description",
                korean_tags=["보컬"], lyric_keywords=["가상문구"])


def hit_for(clue, text=None, *, sid="fictional_song", alternative=None):
    fact = None if text is None else SimpleNamespace(
        song_id=sid, record_id=f"nw:{sid}:fictional_fact", fact_text=text,
        source_url="https://namu.wiki/w/fictional", title="가상 곡",
        artists=("가상 가수",), category="media_usage", section="여담", score=0.7,
    )
    fused = SimpleNamespace(song_id=sid, dense_song=None if fact is None else
                            SimpleNamespace(best_fact=fact), dense_facts=() if fact is None else (fact,))
    return SimpleNamespace(song_id=sid, clue=clue, fused_hit=fused,
                           alternatives=() if alternative is None else ((clue, alternative.fused_hit),))


@pytest.mark.parametrize("query", DETAILS)
@pytest.mark.parametrize("search", BROAD)
def test_qualified_source_cannot_become_generic_enumeration(query, search):
    assert preserve_context_media_search(search, source_query=query,
                                        relation="삽입곡·배경음악") == query


@pytest.mark.parametrize("query", DETAILS)
@pytest.mark.parametrize("search", ("OST", "삽입곡", "OST 삽입곡"))
def test_actual_context_parser_retains_source_and_keeps_other_fields(query, search):
    raw = payload(search)
    frozen = deepcopy(raw)
    result = apply_context_query_safeguards(query, raw)
    assert raw == frozen
    assert {k: v for k, v in result.items() if k != "context_clues"} == {
        k: v for k, v in raw.items() if k != "context_clues"
    }
    assert len(result["context_clues"]) == 1
    clue = ContextClue(**result["context_clues"][0])
    assert clue.confidence == 0.8
    assert context_media_description_requires_support(clue)
    assert clue.search_query == query
    assert context_media_target_terms(clue, original_query=query) == ()
    assert context_search_queries(clue, original_query=query) == (query,)


@pytest.mark.parametrize("source", ("드라마 OST", "영화 삽입곡", "드라마 OST 추천", "게임 배경음악"))
def test_broad_user_requests_still_allow_profile_only_retrieval(source):
    search = preserve_context_media_search("OST", source_query=source,
                                           relation="삽입곡·배경음악")
    # Keeping the original media category (e.g. games) is also a valid broad
    # lookup. The contract here is Sparse recall, not a particular rewrite.
    assert search in {"OST", source}
    clue = cue(search)
    hit = hit_for(clue)
    assert context_candidate_matches_media_description(hit, query_clues=[clue])
    assert context_evidence_for_result(hit, query_clues=[clue]) is None


@pytest.mark.parametrize("target", ("가상 작품", "가상극"))
def test_literal_named_work_keeps_a_focused_query(target):
    query = f"{target}에서 두 사람이 이별할 때 흘러나온 삽입곡"
    search = f"{target} 삽입곡"
    assert preserve_context_media_search(search, source_query=query,
                                        target=target, relation="삽입곡·배경음악") == search
    assert context_media_target_terms(cue(search, target=target), original_query=query) == (target,)


@pytest.mark.parametrize("target", ("가상 작품", "가상극"))
@pytest.mark.parametrize("search", BROAD)
def test_literal_work_name_cannot_be_removed_from_the_primary_lookup(target, search):
    query = f"{target}에서 두 사람이 이별할 때 흘러나온 삽입곡"
    assert preserve_context_media_search(search, source_query=query,
                                        target=target, relation="삽입곡·배경음악") == query


@pytest.mark.parametrize("query", DETAILS)
def test_informative_grounded_rewrite_is_not_forced_to_expand(query):
    assert preserve_context_media_search(query, source_query=query + " 기억이 안 나",
                                        relation="삽입곡·배경음악") == query


@pytest.mark.parametrize("relation", ("제작·발매 비화", "커버·리메이크·답가", "뮤직비디오 속 사건", "차트·기록"))
def test_non_media_relations_are_unchanged(relation):
    assert preserve_context_media_search("짧은 사건", source_query="길게 기억하는 외부 사건",
                                        relation=relation) == "짧은 사건"


@pytest.mark.parametrize("query", DETAILS)
@pytest.mark.parametrize("text", (None, "전혀 다른 드라마의 OST로 이 곡이 삽입되었다."))
def test_restored_clue_blocks_generic_profile_and_unrelated_use_fact(query, text):
    result = apply_context_query_safeguards(query, payload("삽입곡"))
    clue = ContextClue(**result["context_clues"][0])
    hit = hit_for(clue, text)
    assert not context_candidate_matches_media_description(hit, query_clues=[clue])
    assert context_evidence_for_result(hit, query_clues=[clue]) is None


def test_relevant_use_fact_can_still_promote_and_supply_the_same_evidence_fact():
    query = "후각 탐정 드라마에 삽입된 노래"
    result = apply_context_query_safeguards(query, payload("드라마 삽입곡"))
    clue = ContextClue(**result["context_clues"][0])
    hit = hit_for(clue, "후각 탐정 드라마의 배경음악으로 이 곡이 삽입되었다.")
    assert context_candidate_matches_media_description(hit, query_clues=[clue])
    evidence = select_context_fact_for_result(hit, query_clues=[clue])
    assert evidence.record_id == hit.fused_hit.dense_song.best_fact.record_id


def test_another_song_snapshot_cannot_supply_missing_support():
    clue = cue("후각 탐정 드라마에 삽입된 노래")
    unrelated = hit_for(clue, "후각 탐정 드라마에 이 곡이 삽입되었다.", sid="different_song")
    hit = hit_for(clue, alternative=unrelated)
    assert not context_candidate_matches_media_description(hit, query_clues=[clue])


def test_hallucinated_target_and_query_are_rejected_before_specificity_repair():
    query = DETAILS[0]
    raw = payload("다른 작품 삽입곡", target="다른 작품")
    result = apply_context_query_safeguards(query, raw)
    assert result["context_clues"][0]["target"] == ""
    assert "다른 작품" not in result["context_clues"][0]["search_query"]


@pytest.mark.parametrize("query", (
    "가사에 드라마 삽입곡이라는 말이 들어가는 노래",
    "앨범 표지에 OST라는 글씨와 우는 사람이 그려져 있었어",
    "드라마 OST 같은 감성적인 피아노 소리였어",
))
def test_lyric_artwork_and_audio_analogy_do_not_open_context(query):
    assert apply_context_query_safeguards(query, payload("OST"))["context_clues"] == []


def test_real_analyzer_parse_keeps_the_boundary_between_event_audio_and_artwork():
    query = ("특별한 후각을 가진 탐정 드라마에 나온 삽입곡이었어. "
             "반주에는 바이올린 소리가 들렸고 앨범 표지는 파란 바탕에 흰 꽃이 그려져 있었어.")
    raw = dict(intent_type="mixed", confidence=0.91, korean_tags=["드라마OST"],
               context_clues=payload("드라마 삽입곡")["context_clues"],
               audio_english_query="A melodic song accompanied by a violin.",
               image_english_query="White flowers on a blue album cover.",
               has_visual_clue=True, modality_weights=dict(text=0.6, audio=0.2, image=0.2))
    result = QueryAnalyzer(api_key="synthetic-test-key")._parse(query, SimpleNamespace(text=json.dumps(raw)))
    assert result.audio_english_query == raw["audio_english_query"]
    assert result.image_english_query == raw["image_english_query"]
    assert result.confidence == 0.91
    assert len(result.context_clues) == 1
    assert "후각" in result.context_clues[0].search_query
    assert "앨범 표지" not in result.context_clues[0].search_query
    assert context_media_description_requires_support(result.context_clues[0])


def test_rule_fallback_does_not_lose_the_same_supplied_media_memory():
    result = rule_fallback(DETAILS[0])
    assert result.context_clues and "1990년대" in result.context_clues[0].search_query


PARSER = '''# keep this merged module/comment\nfrom __future__ import annotations
def apply_context_query_safeguards(query, raw):
    """Existing rule and grounding logic is preserved."""
    rule = dict(search_query=query, relation="삽입곡·배경음악", target="")
    search = raw["search"]
    if search and len(search) <= 400 and _grounded(search, query, allow_search_words=True):
        rule["search_query"] = search
    return rule

def unrelated():
    return "keep merged behavior 그대로"
'''


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("bom", (b"", b"\xef\xbb\xbf"))
def test_targeted_install_is_idempotent_and_preserves_encoding_and_unrelated_body(newline, bom):
    before = bom + PARSER.replace("\n", newline).encode()
    after = edit_context_media_safeguards(before)
    assert after.startswith(bom)
    ast.parse(after.decode("utf-8-sig"))
    assert edit_context_media_safeguards(after) == after
    assert after.split(b"def unrelated():")[1] == before.split(b"def unrelated():")[1]
    if newline == "\r\n":
        assert b"\n" not in after.replace(b"\r\n", b"")
    assert b"_grounded(search, query, allow_search_words=True)" in after


@pytest.mark.parametrize("text", (
    "def different_function():\n    return 1\n",
    PARSER.replace('rule["search_query"] = search', 'rule["search_query"] = search\n        rule["search_query"] = search'),
    PARSER.replace('and _grounded(search, query, allow_search_words=True)', ''),
    PARSER.replace('rule["search_query"] = search', 'rule["search_query"] = raw["different"]'),
))
def test_unknown_parser_body_is_not_overwritten(text):
    with pytest.raises(ValueError):
        edit_context_media_safeguards(text.encode())


def test_installer_preserves_router_backend_settings_labels_and_existing_reports(tmp_path):
    project = tmp_path / "project"
    root = project / "artifacts/new_run"
    files = {
        "src/retrieval/context_query.py": PARSER.encode(),
        "src/retrieval/context_clue_specificity.py": b"# installed helper\n",
        "src/retrieval/search_router.py": b"# merged router intact\n",
        "src/vector_db/qdrant_backend.py": b"# merged backend intact\n",
        ".env": b"CONTEXT_WEIGHT=0.5\nMONGO_URI=synthetic-private\n",
        "data/context/eval.csv": b"fixed labels intact\n",
        "artifacts/prior/report.json": b'{"status":"prior"}\n',
    }
    for name, value in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    report = install(project, root)
    assert report["changed_files"] == ["src/retrieval/context_query.py"]
    assert (root / "specificity/backups/src/retrieval/context_query.py").read_bytes() == files["src/retrieval/context_query.py"]
    for name, value in files.items():
        if name != "src/retrieval/context_query.py":
            assert (project / name).read_bytes() == value
    repeat = install(project, root)
    assert repeat["changed_files"] == []
    assert repeat["first_install_source_sha256"] == report["source_sha256"]


def test_install_missing_helper_fails_without_source_mutation(tmp_path):
    path = tmp_path / "src/retrieval/context_query.py"
    path.parent.mkdir(parents=True)
    path.write_text(PARSER)
    with pytest.raises(FileNotFoundError):
        install(tmp_path, tmp_path / "artifacts/new_run")
    assert path.read_text() == PARSER


def test_report_write_failure_rolls_back_the_parser_change(tmp_path, monkeypatch):
    from experiments.namuwiki import install_context_media_specificity as installer
    path = tmp_path / "src/retrieval/context_query.py"
    path.parent.mkdir(parents=True)
    path.write_text(PARSER)
    (path.parent / "context_clue_specificity.py").write_text("# installed helper\n")
    original_write = installer.atomic_write
    def fail_report(destination, data):
        if destination.name == "install_report.json":
            raise PermissionError("simulated report lock")
        return original_write(destination, data)
    monkeypatch.setattr(installer, "atomic_write", fail_report)
    with pytest.raises(PermissionError):
        install(tmp_path, tmp_path / "artifacts/new_run")
    assert path.read_text() == PARSER
