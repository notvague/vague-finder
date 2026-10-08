"""Selection must protect existing searches and never tune with test labels."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from experiments.namuwiki.context_step10_core import (
    Setting, blind_verdict, compare, display_depth_tradeoffs, heldout_verdict, select_dev,
)
from experiments.namuwiki.evaluate_context_step10 import (
    GRID, _blind_queries, _item, _stamp, _state, _write_blind_report, _write_report,
)
from experiments.namuwiki.inspect_context_step10_v2 import inspect
from experiments.namuwiki.evaluate_context_step8 import Query, evaluate_pair, metrics


def _arm(song_id: str, *, rank: int | None) -> dict:
    ids = [f"wrong{n}" for n in range(30)]
    if rank is not None:
        ids[rank - 1] = song_id
    return dict(candidates=ids, top=ids[:10], tracks=[], context_hits=[])


def _row(n: int, split: str, *, context: bool, off: int | None,
         on: int | None, group: str | None = None) -> dict:
    song_id = f"answer{n}"
    query = Query(f"{'nw' if context else 'q'}{n:03d}", split,
                  "context" if context else "regression",
                  group or ("context_context" if context else "regression_lyrics"),
                  "context" if context else "", f"질의 {n}", frozenset({song_id}))
    clue = SimpleNamespace(confidence=0.8) if context else None
    analysis = SimpleNamespace(context_clues=[clue] if clue else [],
                               has_context_clue=bool(clue))
    before, after = _arm(song_id, rank=off), _arm(song_id, rank=on)
    value = evaluate_pair(query, before, after, frozenset({song_id}), analysis)
    value.update(category="media_usage" if context else "unlabeled",
                 target_evidence_record_id="", target_evidence_labeled=None,
                 displayed_evidence=[], off_elapsed_ms=10.0, on_elapsed_ms=12.0,
                 ref_candidate_ids="|".join(before["candidates"]),
                 off_candidate_hit_width=float(off is not None),
                 on_candidate_hit_width=float(on is not None))
    for key, score in metrics(before["candidates"], before["top"], frozenset({song_id})).items():
        value[f"ref_{key}"] = score
    return value


def _split(split="dev", *, weight=0.25):
    context, regression = ((21, 53) if split == "dev" else (9, 23))
    rows = [_row(n, split, context=True, off=None, on=1)
            for n in range(context)]
    rows += [_row(1000 + n, split, context=False, off=1, on=1)
             for n in range(regression)]
    return Setting(weight, 100, 100, 30), rows


def test_grid_is_frozen_and_includes_fact_sparse_and_pool_variants():
    assert len(GRID) == 24 and len({item.key for item in GRID}) == 24
    assert {(item.fact_k, item.sparse_k) for item in GRID} == {
        (100, 100), (160, 100), (100, 160),
    }
    assert {item.candidate_k for item in GRID} == {30, 50}
    assert {item.weight for item in GRID if item.candidate_k == 30
            and item.fact_k == item.sparse_k == 100} == {
        0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    }
    assert {item.named_media_multiplier for item in GRID} == {1.0, 1.5, 2.0}
    assert Setting(0.5, 100, 100, 30, named_media_multiplier=1.5).key == "w0p5_d100_s100_c30_m1p5"
    with pytest.raises(ValueError, match="weight"):
        Setting(float("nan"), 100, 100, 30)
    with pytest.raises(ValueError, match="top_k"):
        Setting(0.5, 100, 100, 30, top_k=5)
    with pytest.raises(ValueError, match="multiplier"):
        Setting(0.5, 100, 100, 30, named_media_multiplier=float("inf"))


def test_dev_selection_is_label_free_on_test_and_prefers_cheaper_ties():
    lower, rows = _split()
    higher = Setting(1.0, 160, 100, 50)
    decision = select_dev({higher.key: (higher, rows), lower.key: (lower, rows)})
    assert decision["status"] == "locked_for_test"
    assert decision["selected"] == lower.key
    assert decision["candidates"][lower.key]["reference"]["context_all"]["metrics"]["hit10"]["delta"] == 1
    assert len(decision["candidates"][lower.key]["reference"]["context_all"]["metrics"]["hit10"]["bootstrap_95_delta"]) == 2

    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="test split"):
        select_dev({lower.key: (lower, rows)})


def test_required_shinchan_queries_must_enter_first_thirty_candidates():
    setting, rows = _split()
    rows[1] = _row(1, "dev", context=True, off=None, on=None)
    missing = compare(rows, setting)
    assert "nw001: required Context candidate@30 missing" in missing["violations"]
    assert select_dev({setting.key: (setting, rows)})["status"] == "blocked"

    rows[1] = _row(1, "dev", context=True, off=None, on=12)
    prior = _row(203, "dev", context=False, group="regression_other", off=None, on=None)
    prior["context_clue_count"] = 1
    rows[-1] = prior
    assert "q203: required Context candidate@30 missing" in compare(rows, setting)["violations"]
    rows[-1] = _row(203, "dev", context=False, group="regression_other", off=None, on=20)
    rows[-1]["context_clue_count"] = 1
    assert select_dev({setting.key: (setting, rows)})["status"] == "locked_for_test"


def test_previously_wrong_fact_cannot_pass_the_selection_or_validation_gate():
    setting, rows = _split()
    for query_id in ("nw005", "nw021", "nw024"):
        row = rows[5]
        row["query_id"] = query_id
        row["target_evidence_labeled"] = False
        assert f"{query_id}: displayed fact differs from reviewed event" in compare(rows, setting)["violations"]


def test_one_regression_loss_blocks_even_when_average_gains_cancel():
    setting, rows = _split()
    rows[-1] = _row(2000, "dev", context=False, off=1, on=None)
    result = select_dev({setting.key: (setting, rows)})
    assert result["status"] == "blocked"
    assert any("protected hit10 loss" in item for item in result["candidates"][setting.key]["violations"])


def test_no_clue_paths_identical_and_control_has_no_evidence():
    setting, rows = _split()
    candidates = rows[-1]["on_candidate_ids"].split("|")
    candidates[-1] = "unexpected"
    rows[-1]["on_candidate_ids"] = "|".join(candidates)
    assert any("no-clue" in value for value in compare(rows, setting)["violations"])
    rows[-1] = _row(2000, "dev", context=True, group="context_control", off=1, on=1)
    rows[-1]["context_clue_count"] = 0
    rows[-1]["target_evidence_record_id"] = "nw:2000:fake"
    assert any("control target has evidence" in value
               for value in compare(rows, setting)["violations"])


def test_heldout_failure_is_not_a_new_tuning_selection():
    setting, rows = _split("test")
    rows[-1] = _row(3000, "test", context=False, off=8, on=None)
    verdict = heldout_verdict(rows, setting)
    assert verdict["status"] == "not_recommended"
    assert "q3000: protected hit10 loss" in verdict["violations"]
    assert "must not tune this setting" in verdict["policy"]
    with pytest.raises(ValueError, match="duplicate query"):
        compare(rows + [rows[0]], setting)


def test_candidate_width_is_not_mistaken_for_candidate_at_30():
    target = "target"
    row = Query("nw001", "dev", "context", "context_context", "context",
                "질의", frozenset({target}))
    analysis = SimpleNamespace(context_clues=[SimpleNamespace(confidence=0.8)],
                               has_context_clue=True)
    setting = Setting(0.5, 100, 100, 50)
    off = _arm(target, rank=None)
    off["candidates"] += [f"extra{n}" for n in range(20)]
    off["elapsed_ms"] = 12.0
    on = dict(off, candidates=list(off["candidates"]), evidence=[], elapsed_ms=20.0)
    on["candidates"][39] = target
    on["top"] = on["candidates"][:10]
    on["context_hits"] = [SimpleNamespace(song_id=target)]
    ref = _arm(target, rank=None)
    ref["elapsed_ms"] = 11.0
    value = _item(row, setting, analysis, off, on, ref,
                  frozenset({target}), {"nw001": {"clue_category": "media_usage"}})
    assert value["on_candidate_hit30"] == 0
    assert value["on_candidate_hit_width"] == 1
    assert value["on_candidate_rank"] == 40


def test_display_depth_is_measured_without_searching_or_changing_the_top10_contract():
    row = _row(8, "dev", context=True, off=None, on=15)
    values = display_depth_tradeoffs([row])["context_all"]
    assert values["hit5"]["on"] == values["hit10"]["on"] == 0
    assert values["hit20"]["ref"] == 0
    assert values["hit20"]["on"] == 1


def test_reused_test_is_provisional_and_new_blind_needs_coverage():
    setting, rows = _split("test")
    assert heldout_verdict(rows, setting)["status"] == "provisional"
    with pytest.raises(ValueError, match="blind set needs"):
        blind_verdict(rows, setting)


def test_checkpoint_rejects_old_corpus_or_code_stamp(tmp_path):
    path = tmp_path / "dev_checkpoint.json"
    state = _state(path, {"build": "new"}, "dev")
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="another corpus"):
        _state(path, {"build": "old"}, "dev")
    with pytest.raises(ValueError, match="another corpus"):
        _state(path, {"build": "new"}, "test")


def test_blind_registration_is_part_of_frozen_dev_stamp(tmp_path):
    before = _stamp({"labels": "fixed"}, "cached", {"build": "active"}, tmp_path)
    assert before["blind_registration_sha256"] is None
    (tmp_path / "blind_registration.json").write_text('{"registered":true}', encoding="utf-8")
    after = _stamp({"labels": "fixed"}, "cached", {"build": "active"}, tmp_path)
    assert after["blind_registration_sha256"] != before["blind_registration_sha256"]


def test_new_blind_labels_are_disjoint_and_fact_bound(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact_dir = tmp_path / "artifacts/context/songs"
    artifact_dir.mkdir(parents=True)
    path = tmp_path / "blind.csv"
    groups = ["context_context", "context_mixed", "context_control",
              "regression_lyrics", "regression_mood_sound", "regression_image"]
    categories = ["media_usage", "production", "music_video",
                  "performance", "meme", "version"]
    labels = []
    for n in range(30):
        song_id, group = str(9000 + n), groups[n % 6]
        category = (categories[((n // 6) * 2 + n % 6) % 6]
                    if group.startswith("context_") and group != "context_control"
                    else "no_context")
        record_id = f"nw:{song_id}:fact"
        fact = {"record_id": record_id, "category": category,
                "evidence_text": "검토된 문장"} if category != "no_context" else None
        artifact = {"song": {"song_id": song_id},
                    "status": "ok" if fact else "not_found",
                    "source": {"url": "https://namu.wiki/w/test"},
                    "retrieval": {"records": [fact] if fact else []}}
        (artifact_dir / f"{song_id}.json").write_text(json.dumps(artifact), encoding="utf-8")
        labels.append({"query_id": f"blind{n:03d}", "query": f"새 질의 {n}",
                       "relevant_ids": song_id, "group": group,
                       "clue_category": category,
                       "evidence_record_ids": record_id if fact else "",
                       "review_note": "사실 관계를 확인함"})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    original = {"q203": Query("q203", "dev", "context", "context_context",
                              "context", "원래 질의", frozenset({"837567"}))}
    parsed, cases = _blind_queries(path, original)
    assert len(parsed) == 30 and len(cases) == 30
    assert parsed["blind000"].relevant_ids == frozenset({"9000"})
    labels[0]["relevant_ids"] = "837567"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    with pytest.raises(ValueError, match="overlaps an old label"):
        _blind_queries(path, original)


def test_reports_show_scope_losses_and_only_blind_can_recommend(tmp_path):
    setting, dev = _split()
    decision = select_dev({setting.key: (setting, dev)})
    _, test = _split("test")
    provisional = heldout_verdict(test, setting)
    corpus = {"context_build_id": "full-build", "source_song_count": 3016,
              "dense_record_count": 3995, "sparse_profile_count": 478}
    (tmp_path / "dev_selection.json").write_text(
        json.dumps({"decision": decision}), encoding="utf-8")
    report = tmp_path / "final_report.md"
    _write_report(report, {"stamp": {"corpus": corpus},
                           "locked_setting": setting.key, "verdict": provisional},
                  tmp_path / "dev_selection.json")
    content = report.read_text(encoding="utf-8")
    assert "provisional" in content and "3016곡" in content
    assert "새 블라인드 데이터가 아니다" in content

    blind = []
    groups = ["context_context", "context_mixed", "context_control",
              "regression_lyrics", "regression_mood_sound", "regression_image"]
    for n in range(30):
        group = groups[n % 6]
        row = _row(n, "blind", context=group.startswith("context_"),
                   group=group, off=1 if group == "context_control" or group.startswith("regression") else None,
                   on=1)
        if group == "context_control":
            row["context_clue_count"] = 0
            row["category"] = "no_context"
        elif group.startswith("context_"):
            row["category"] = f"fact_type_{n % 6 + n // 6}"
        blind.append(row)
    verdict = blind_verdict(blind, setting)
    assert verdict["status"] == "recommended"
    blind_report = tmp_path / "blind_report.md"
    _write_blind_report(blind_report, {"stamp": {"corpus": corpus},
                                       "locked_setting": setting.key, "verdict": verdict})
    content = blind_report.read_text(encoding="utf-8")
    assert "recommended" in content and "CONTEXT_WEIGHT=0.25" in content


def test_visible_review_refuses_missing_recall_or_wrong_fact(tmp_path, capsys):
    key = "w0p5_d100_s100_c30"
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"queries": [
        {"query_id": qid, "evidence_record_id": f"correct:{qid}"}
        for qid in ("nw001", "nw005", "nw021", "nw024")
    ]}), encoding="utf-8")
    (tmp_path / "final_report.json").write_text(json.dumps({
        "locked_setting": key, "verdict": {"status": "provisional"},
    }), encoding="utf-8")
    rows = {qid: dict(on_candidate_rank=4, ref_candidate_rank=None,
                      on_top10_rank=4, ref_top10_rank=None,
                      target_evidence_record_id="", eligible_ids="123",
                      displayed_evidence=[]) for qid in ("nw001", "q203", "nw005", "nw021", "nw024")}

    def save():
        for split, query_ids in (("dev", ("nw001", "q203", "nw005")),
                                 ("test", ("nw021", "nw024"))):
            (tmp_path / f"{split}_checkpoint.json").write_text(json.dumps({
                "runs": {key: {qid: rows[qid] for qid in query_ids}},
            }), encoding="utf-8")

    save()
    inspect(tmp_path, labels)
    assert "[PASS]" in capsys.readouterr().out
    rows["nw005"]["target_evidence_record_id"] = "wrong:episode"
    save()
    with pytest.raises(AssertionError, match="unrelated fact"):
        inspect(tmp_path, labels)
    rows["nw005"]["target_evidence_record_id"] = ""
    rows["q203"]["on_candidate_rank"] = None
    save()
    with pytest.raises(AssertionError, match="Candidate@30"):
        inspect(tmp_path, labels)
