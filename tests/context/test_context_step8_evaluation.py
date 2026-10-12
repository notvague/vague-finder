"""Step 8 guards paired search and denominators, not the ranking implementation."""

import asyncio
import csv
import json
from types import SimpleNamespace

import pytest

from experiments.namuwiki.evaluate_context_step8 import (
    REGRESSION_GROUP_IDS,
    Query,
    evaluate_pair,
    load_queries,
    metrics,
    paired_search,
    summarize,
)


def _row(song_id="target", *, source="context", group="context_context", segment="context"):
    return Query("nw001", "dev", source, group, segment,
                 "외부 작품의 삽입곡", frozenset([song_id]))


def _arm(ids, *, context_ids=()):
    return {"candidates": ids, "top": ids[:10],
            "context_hits": [SimpleNamespace(song_id=song_id) for song_id in context_ids],
            "tracks": [SimpleNamespace(id=song_id) for song_id in ids[:10]]}


class Router:
    def __init__(self):
        self.calls = []

    async def search(self, analysis, **kwargs):
        self.calls.append((analysis, dict(kwargs)))
        assert kwargs["candidate_k"] == 30
        assert kwargs["top_k"] == 10
        assert kwargs["use_rerank"] is False
        pool = [f"other{index}" for index in range(30)]
        if kwargs["use_context"] and analysis.has_context_clue:
            pool[4] = "target"
            kwargs["context_hits_out"].append(SimpleNamespace(song_id="target"))
        kwargs["candidate_ids_out"].extend(pool)
        return [SimpleNamespace(id=song_id, title="fixture", artist="fixture")
                for song_id in pool[:10]]


def test_same_analysis_same_candidate_width_rerank_is_always_disabled():
    router = Router()
    analysis = SimpleNamespace(has_context_clue=True, context_clues=[SimpleNamespace(confidence=1)])
    off, on = asyncio.run(paired_search(router, analysis))
    assert [call[1]["use_context"] for call in router.calls] == [False, True]
    assert router.calls[0][0] is router.calls[1][0] is analysis
    assert "target" not in off["candidates"] and on["candidates"][4] == "target"
    assert len(on["context_hits"]) == 1
    result = evaluate_pair(_row(), off, on, frozenset({"target"}), analysis)
    assert result["off_candidate_rank"] is None and result["on_candidate_rank"] == 5
    assert result["off_top10_rank"] is None and result["on_top10_rank"] == 5
    assert result["on_hit10"] == 1.0 and result["on_mrr10"] == 0.2


def test_no_clue_query_must_be_identical_in_both_arms():
    analysis = SimpleNamespace(has_context_clue=False, context_clues=[])
    off, on = asyncio.run(paired_search(Router(), analysis))
    assert off["candidates"] == on["candidates"]

    class BrokenRouter(Router):
        async def search(self, analysis, **kwargs):
            result = await super().search(analysis, **kwargs)
            if kwargs["use_context"]:
                kwargs["candidate_ids_out"][0] = "changed"
                result[0].id = "changed"
            return result

    with pytest.raises(RuntimeError, match="no-clue query changed"):
        asyncio.run(paired_search(BrokenRouter(), analysis))


def test_missing_text_index_labels_are_reported_and_not_scored():
    analysis = SimpleNamespace(context_clues=[SimpleNamespace(confidence=0.8)])
    unavailable = evaluate_pair(_row("absent"), _arm(["other"]),
                                _arm(["other"]), frozenset(), analysis)
    present = evaluate_pair(_row(), _arm(["other"]),
                            _arm(["target"], context_ids=["target"]),
                            frozenset({"target"}), analysis)
    report = summarize([unavailable, present])
    section = report["groups"]["context_all"]
    assert section["queries"] == 2 and section["scored"] == 1
    assert section["unavailable_queries"] == ["nw001"]
    assert section["metrics"]["hit10"]["off"] == 0.0
    assert section["metrics"]["hit10"]["on"] == 1.0
    assert section["metrics"]["hit10"]["gained"] == ["nw001"]
    assert unavailable["on_hit10"] is None


def test_gains_cannot_hide_regression_losses_in_group_mean():
    analysis = SimpleNamespace(context_clues=[])
    winner = evaluate_pair(_row(source="regression", group="regression_image", segment=""),
                           _arm(["other"]), _arm(["target"]),
                           frozenset({"target"}), analysis)
    loser = evaluate_pair(_row(source="regression", group="regression_image", segment=""),
                          _arm(["target"]), _arm(["other"]),
                          frozenset({"target"}), analysis)
    loser["query_id"] = "q309"
    report = summarize([winner, loser])
    metric = report["groups"]["regression_image"]["metrics"]["hit10"]
    assert metric["delta"] == 0.0
    assert metric["gained"] == ["nw001"] and metric["lost"] == ["q309"]


def _write_csv(path, entries):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "split", "query_type", "query", "relevant_ids"])
        writer.writeheader()
        writer.writerows(entries)


def test_csv_split_groups_and_source_consistency(tmp_path):
    context = tmp_path / "context.csv"
    regression = tmp_path / "v05.csv"
    source = tmp_path / "source.json"
    _write_csv(context, [
        dict(query_id="nw001", split="dev", query_type="search", query="애니 삽입곡", relevant_ids="song1"),
        dict(query_id="nw027", split="test", query_type="search", query="곡 제목", relevant_ids="song2"),
    ])
    _write_csv(regression, [
        dict(query_id=query_id, split="test" if query_id == "q309" else "dev",
             query_type="search", query="기존 단서", relevant_ids="song3")
        for query_id in sorted(set().union(*REGRESSION_GROUP_IDS.values()) | {"m104"})
    ])
    source.write_text(json.dumps({"queries": [
        {"query_id": "nw001", "split": "dev", "song_id": "song1",
         "query": "애니 삽입곡", "segment": "context"},
        {"query_id": "nw027", "split": "test", "song_id": "song2",
         "query": "곡 제목", "segment": "control"},
    ]}), encoding="utf-8")
    rows, sources = load_queries(context, regression, "dev", (), source)
    assert {r.group for r in rows} >= {"context_context", "regression_lyrics", "regression_mood_sound"}
    assert {r.query_id for r in rows}.isdisjoint({"q309", "nw027"})
    assert sources["context_source_sha256_16"] != "unavailable"
    with pytest.raises(ValueError, match="not in split"):
        load_queries(context, regression, "dev", ["nw027"], source)
    with pytest.raises(ValueError, match="unknown"):
        load_queries(context, regression, "all", ["missing"], source)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["queries"][0]["segment"] = "control"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        load_queries(context, regression, "dev", (), source)


def test_multi_label_hit_and_recall_do_not_inflate():
    answer = frozenset({"a", "b"})
    found = metrics(["x", "b", "y"], ["x", "b", "y"], answer)
    assert found["candidate_hit30"] == 1.0
    assert found["candidate_recall30"] == 0.5
    assert found["mrr10"] == 0.5
    with pytest.raises(ValueError, match="denominator"):
        metrics(["a"], ["a"], frozenset())


def test_analyzer_misroutes_are_reported_separately_from_retrieval_scores():
    good = SimpleNamespace(context_clues=[SimpleNamespace(confidence=0.8)])
    absent = SimpleNamespace(context_clues=[])
    missing = evaluate_pair(_row(), _arm(["other"]), _arm(["other"]),
                            frozenset({"target"}), absent)
    control = evaluate_pair(_row(source="context", group="context_control", segment="control"),
                            _arm(["target"]), _arm(["target"]),
                            frozenset({"target"}), good)
    control["query_id"] = "nw027"
    image = evaluate_pair(_row(source="regression", group="regression_image", segment=""),
                          _arm(["target"]), _arm(["target"]),
                          frozenset({"target"}), good)
    image["query_id"] = "q204"
    issues = summarize([missing, control, image])["issues"]
    assert issues == {
        "missing_context_clue": ["nw001"],
        "unexpected_control_clue": ["nw027"],
        "unexpected_regression_clue": ["q204"],
    }
