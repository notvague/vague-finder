"""Evaluation guards must detect failures without changing retrieval settings."""

from copy import deepcopy

import pytest

from experiments.namuwiki.context_fixed_core import (
    EXPECTED_NEW, SETTING, fixed_report, inspect_path, paired_exact_p,
)
from experiments.namuwiki.evaluate_context_step8 import Query, evaluate_pair, metrics
from src.backend.schemas.query import ContextClue, QueryAnalysis


def row(qid="case", group="context_context", *, off=None, on=1, required=""):
    positive = group in {"context_context", "context_mixed"}
    query = Query(qid, "dev", "context" if group.startswith("context_") else "regression",
                  group, "", "synthetic query", frozenset({"answer"}))
    analysis = QueryAnalysis(original_query=query.query, intent_type="mixed", image_english_query="", audio_english_query="",
                             context_clues=[ContextClue(target="work", relation="used in",
                                                        search_query=query.query, confidence=0.8)] if positive else [])

    def arm(rank):
        candidates = [f"other{i}" for i in range(30)]
        if rank is not None:
            candidates[rank - 1] = "answer"
        return dict(candidates=candidates, top=candidates[:10], tracks=[], context_hits=[])

    before, after = arm(off), arm(on)
    value = evaluate_pair(query, before, after, frozenset({"answer"}), analysis)
    value.update(category="synthetic", required_path=required, target_evidence_record_id="",
                 displayed_evidence=[], off_elapsed_ms=10, on_elapsed_ms=11, rerank_executed=False)
    for name, result in metrics(before["candidates"], before["top"], frozenset({"answer"})).items():
        value[f"ref_{name}"] = result
    for name in ("off", "on"):
        value[f"{name}_required_path_verified"] = bool(required)
        value[f"{name}_path_errors"] = []
        value[f"{name}_image_executed"] = required == "image"
        value[f"{name}_audio_executed"] = required == "audio"
        value[f"{name}_context_executed"] = name == "on" and positive
    return value


def independent_rows():
    values = []
    for group, count in EXPECTED_NEW.items():
        for n in range(count):
            positive = group in {"context_context", "context_mixed"}
            required = {"regression_image": "image", "regression_mood_sound": "audio"}.get(group, "")
            values.append(row(f"{group}{n}", group, off=None if positive else 1, on=1, required=required))
    return values


def test_fixed_settings_equal_the_previous_diagnostic_without_a_grid():
    assert SETTING.as_dict() == dict(weight=0.5, fact_k=100, sparse_k=100,
                                     candidate_k=30, top_k=10, named_media_multiplier=2.0)


@pytest.mark.parametrize("names,executed,queried", [
    ([], False, False), (["path.image"], True, False),
    (["image.embed", "image.query"], False, True),
    (["path.image", "image.embed"], True, False),
    (["path.image", "image.embed", "image.query"], True, True),
])
def test_weights_or_branch_entry_do_not_prove_an_image_query(names, executed, queried):
    check = inspect_path([{"name": n} for n in names], "image")
    assert check.executed == executed and check.queried == queried
    assert check.verified == (executed and queried)


@pytest.mark.parametrize("failed", ["path.audio", "audio.embed", "audio.query"])
def test_embed_or_db_errors_are_not_successful_path_coverage(failed):
    spans = [dict(name=n, error="TimeoutError" if n == failed else "")
             for n in ("path.audio", "audio.embed", "audio.query")]
    check = inspect_path(spans, "audio")
    assert check.executed and check.queried and not check.verified
    assert check.error == "TimeoutError"


@pytest.mark.parametrize("gain,loss,expected", [(0, 0, 1.0), (1, 0, 1.0), (6, 0, 0.03125), (3, 3, 1.0)])
def test_exact_test_uses_discordant_question_pairs(gain, loss, expected):
    assert paired_exact_p(gain, loss) == expected
    assert paired_exact_p(loss, gain) == expected


def test_effect_denominator_excludes_easy_no_fact_controls():
    data = [row("positive"), row("control", "context_control", off=1, on=1)]
    report = fixed_report(data, phase="dev", expected=2)
    assert report["status"] == "passed"
    assert report["groups"]["context_positive"]["queries"] == 1
    assert report["groups"]["context_positive"]["metrics"]["hit10"]["delta"] == 1
    assert report["groups"]["context_all"]["metrics"]["hit10"]["delta"] == 0.5
    assert report["paired_context_tests"]["hit10"]["gained"] == 1
    assert report["paired_context_tests"]["hit10"]["bootstrap_95_delta"] == [1.0, 1.0]


@pytest.mark.parametrize("kind", ["empty", "wrong_count", "duplicate", "unindexed", "partial_label"])
def test_denominator_and_unique_query_inventory_cannot_silently_shrink(kind):
    data = [row("a"), row("b")]
    expected = 2
    if kind == "empty":
        data = []
    elif kind == "wrong_count":
        expected = 3
    elif kind == "duplicate":
        data[1]["query_id"] = "a"
    elif kind == "unindexed":
        data[0]["scoreable"] = 0
    else:
        data[0]["unavailable_ids"] = "another_answer"
    with pytest.raises(ValueError):
        fixed_report(data, phase="dev", expected=expected)


@pytest.mark.parametrize("field,value,issue", [
    ("context_clue_count", 0, "missing_context_clue"),
    ("on_context_executed", False, "context_path_not_executed"),
    ("off_context_executed", True, "context_executed_in_off_arm"),
    ("rerank_executed", True, "rerank_executed"),
    ("on_path_errors", [{"error": "TimeoutError"}], "on_retrieval_error"),
])
def test_positive_case_failures_block_a_claim(field, value, issue):
    value_row = row()
    value_row[field] = value
    report = fixed_report([value_row], phase="dev", expected=1)
    assert report["status"] == "blocked"
    assert f"case: {issue}" in report["violations"]


@pytest.mark.parametrize("group", ["context_control", "regression_lyrics", "regression_mood_sound", "regression_image"])
def test_context_must_not_run_for_unrelated_controls_and_regressions(group):
    value = row(group=group, off=1, on=1)
    value["context_clue_count"] = 1
    assert "case: unexpected_context_clue" in fixed_report([value], phase="dev", expected=1)["violations"]


def test_no_clue_order_change_is_reported_even_when_the_label_stays_first():
    value = row(group="regression_image", off=1, on=1, required="image")
    value["on_candidate_ids"] += "|bad_extra"
    assert "case: no_clue_candidate_order_changed" in fixed_report([value], phase="test", expected=1)["violations"]


@pytest.mark.parametrize("qid", ["nw001", "q203"])
def test_previously_missing_key_candidates_are_still_required(qid):
    assert f"{qid}: required_candidate30_missing" in fixed_report([row(qid, on=None)], phase="dev", expected=1)["violations"]


def test_protected_rank_deterioration_is_visible_without_a_top10_loss():
    value = row(group="regression_other", off=6, on=9)
    report = fixed_report([value], phase="test", expected=1)
    assert "case: protected_mrr10_loss" in report["violations"]
    assert "case: hit10_loss" not in report["violations"]


def test_positive_top10_loss_blocks_even_when_still_a_candidate():
    value = row(off=8, on=13)
    assert "case: hit10_loss" in fixed_report([value], phase="test", expected=1)["violations"]


def test_path_coverage_requires_both_arms_with_errors_checked():
    value = row(group="regression_image", off=1, on=1, required="image")
    value["on_required_path_verified"] = False
    report = fixed_report([value], phase="test", expected=1)
    assert report["path_coverage"]["image"]["off_verified"] == 1
    assert report["path_coverage"]["image"]["on_verified"] == 0
    assert "case: on_image_path_not_verified" in report["violations"]


def test_unmatched_citation_is_review_required_instead_of_a_precision_claim():
    value = row()
    value["displayed_evidence"] = [dict(song_id="answer", record_id="another_fact", label_match=False)]
    report = fixed_report([value], phase="dev", expected=1)
    assert report["status"] == "evidence_review_required"
    assert report["evidence"]["label_matched_count"] == 0
    assert report["evidence"]["unreviewed_count"] == 1


def test_no_fact_control_must_not_display_a_context_citation():
    value = row(group="context_control", off=1, on=1)
    value["displayed_evidence"] = [dict(song_id="answer", record_id="fact", label_match=False)]
    assert "case: control_has_displayed_evidence" in fixed_report([value], phase="test", expected=1)["violations"]


def test_independent_inventory_and_positive_hit10_improvement_are_both_required():
    data = independent_rows()
    report = fixed_report(data, phase="independent", expected=60)
    assert report["status"] == "passed" and report["groups"]["context_positive"]["queries"] == 36
    assert report["path_coverage"]["image"]["on_verified"] == 6
    assert report["path_coverage"]["audio"]["on_verified"] == 6
    for value in data:
        if value["group"] in {"context_context", "context_mixed"}:
            value["on_hit10"] = value["off_hit10"]
    assert "no_positive_context_hit10_improvement" in fixed_report(data, phase="independent", expected=60)["violations"]
    data[0]["group"] = "regression_other"
    with pytest.raises(ValueError, match="inventory"):
        fixed_report(data, phase="independent", expected=60)


def test_synthetic_ci_and_categories_are_deterministic():
    data = [row("one", off=None, on=1), row("two", off=2, on=1)]
    first = fixed_report(data, phase="dev", expected=2)
    second = fixed_report(deepcopy(data), phase="dev", expected=2)
    assert first == second
    assert first["groups"]["category_synthetic"]["queries"] == 2
