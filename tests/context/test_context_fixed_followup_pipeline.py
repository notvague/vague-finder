"""Frozen provenance, drift detection, label-free observation and safe export."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from experiments.namuwiki.context_fixed_core import SETTING
from experiments.namuwiki.diagnose_context_fixed_followup import (
    ObservedContext, compare_row, frozen_analyses, read_prior,
)
from experiments.namuwiki.evaluate_context_step8 import Query
from experiments.namuwiki.pack_context_fixed_followup import pack
from src.backend.schemas.query import QueryAnalysis


def prior():
    query = Query("diagnostic", "dev", "regression", "regression_lyrics", "",
                  "synthetic question", frozenset({"answer"}))
    value = QueryAnalysis(original_query=query.query, intent_type="mixed",
                          image_english_query="", audio_english_query="")
    raw = value.model_dump(mode="json")
    registration = dict(setting=SETTING.as_dict(), rerank_enabled=False)
    binding = hashlib.sha256(json.dumps(registration, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache = dict(meta=dict(fixed_registration_sha256=binding, fixed_analysis_batch="dev"),
                 entries={query.query_id: dict(query=query.query, fallback=False, analysis=raw)})
    report = dict(registration=registration, phase="dev", evaluated=1,
                  analysis_cache_sha256="original_cache_sha",
                  question_results=[dict(query_id=query.query_id, query=query.query, analysis=raw)])
    return dict(
        **{"registration.json": registration, "dev_report.json": report,
           "dev_analysis_cache.json": cache,
           "dev_analysis_cache.json.sha256": "original_cache_sha"},
    ), {query.query_id: query}


def test_frozen_analysis_ignores_current_analyzer_fingerprint_but_preserves_every_field():
    data, selected = prior()
    frozen = deepcopy(data)
    analyses, _ = frozen_analyses(data, selected, "dev")
    assert analyses["diagnostic"].model_dump(mode="json") == data["dev_analysis_cache.json"]["entries"]["diagnostic"]["analysis"]
    assert data == frozen  # Diagnostic never edits the original cache.


@pytest.mark.parametrize("mutation", [
    "fallback", "query", "analysis", "registration", "missing", "duplicate",
    "cache_hash", "binding", "split", "schema_default",
])
def test_frozen_analysis_rejects_mixed_or_edited_provenance(mutation):
    data, selected = prior()
    cache, report = data["dev_analysis_cache.json"], data["dev_report.json"]
    if mutation == "fallback":
        cache["entries"]["diagnostic"]["fallback"] = True
    elif mutation == "query":
        cache["entries"]["diagnostic"]["query"] = "another question"
    elif mutation == "analysis":
        report["question_results"][0]["analysis"] = {"original_query": "edited"}
    elif mutation == "registration":
        report["registration"] = {"edited": True}
    elif mutation == "missing":
        cache["entries"] = {}
    elif mutation == "duplicate":
        report["question_results"].append(deepcopy(report["question_results"][0]))
    elif mutation == "cache_hash":
        report["analysis_cache_sha256"] = "different"
    elif mutation == "binding":
        cache["meta"]["fixed_registration_sha256"] = "another registration"
    elif mutation == "split":
        cache["meta"]["fixed_analysis_batch"] = "test"
    else:
        # Removing a default is still an edited schema representation. The
        # diagnostic cannot silently repair it and claim exact same analysis.
        raw = deepcopy(cache["entries"]["diagnostic"]["analysis"])
        raw.pop("artist_name")
        cache["entries"]["diagnostic"]["analysis"] = raw
        report["question_results"][0]["analysis"] = raw
    with pytest.raises(ValueError):
        frozen_analyses(data, selected, "dev")


def test_baseline_drift_is_separate_from_the_on_fix_and_citation_removal():
    before = dict(query_id="synthetic", displayed_evidence=[dict(song_id="wrong", record_id="bad")])
    for arm in ("off", "on"):
        before.update({arm + "_" + key: value for key, value in dict(
            candidate_rank=8, top10_rank=8, candidate_hit30=1, hit1=0,
            hit5=0, hit10=1, mrr10=1 / 8, candidate_ids=["answer"],
            top10_ids=["answer"],
        ).items()})
    after = deepcopy(before)
    after["displayed_evidence"] = []
    after["on_candidate_rank"] = 2
    comparison = compare_row(before, after)
    assert comparison["off_baseline_drift"] == []
    assert comparison["previous_on_candidate_rank"] == 8
    assert comparison["fixed_on_candidate_rank"] == 2
    assert comparison["removed_displayed_evidence"] == [dict(song_id="wrong", record_id="bad")]
    after["off_top10_rank"] = 7
    assert compare_row(before, after)["off_baseline_drift"] == ["top10_rank"]


def test_rank_observer_never_passes_answer_ids_into_the_search():
    calls = []
    class Search:
        generation = "unchanged"
        def search_fused_songs(self, query, **kwargs):
            calls.append((query, kwargs))
            return (SimpleNamespace(song_id="answer", dense_rank=1, sparse_rank=3, dense_song=None),)
    observer = ObservedContext(Search())
    observer.targets = frozenset({"answer"})
    kwargs = dict(fact_k=100, sparse_k=100, media_targets=("supplied work",))
    results = observer.search_fused_songs("literal user clue", **kwargs)
    assert calls == [("literal user clue", kwargs)]
    assert observer.generation == "unchanged" and results[0].song_id == "answer"
    assert observer.calls[0]["target_raw_ranks"]["answer"] == dict(
        fused_rank=1, dense_rank=1, sparse_rank=3, candidate_fact_id=None,
    )


def make_archive(path, *, tamper=False, duplicate=False):
    files = {"registration.json": dict(setting=SETTING.as_dict(), rerank_enabled=False)}
    for phase in ("dev", "test"):
        files[phase + "_report.json"] = {}
        files[phase + "_analysis_cache.json"] = {}
    manifest = dict(schema="context_fixed_feedback_v1", files={})
    with ZipFile(path, "w") as z:
        for name, value in files.items():
            content = json.dumps(value).encode()
            manifest["files"][name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            z.writestr(name, content)
        if tamper:
            manifest["files"]["dev_report.json"]["sha256"] = "incorrect"
        if duplicate:
            with pytest.warns(UserWarning):
                z.writestr("dev_report.json", b"{}")
        z.writestr("export_manifest.json", json.dumps(manifest))


def test_feedback_bytes_are_verified_before_parsing_analyses(tmp_path):
    archive = tmp_path / "feedback.zip"
    make_archive(archive)
    assert read_prior(archive)["registration.json"]["setting"] == SETTING.as_dict()
    make_archive(archive, tamper=True)
    with pytest.raises(ValueError, match="checksum"):
        read_prior(archive)
    make_archive(archive, duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        read_prior(archive)


def test_followup_export_whitelists_reports_and_keeps_stage_names(tmp_path):
    root = tmp_path / "output"
    for phase, name in (("diagnostic", "diagnostic_report.json"), ("fresh", "validation_report.json")):
        path = root / phase / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status": "partial"}', encoding="utf-8")
    (root / ".env").write_text("private", encoding="utf-8")
    (root / "fresh" / "unknown_audio.mp4").write_bytes(b"private")
    archive = tmp_path / "feedback.zip"
    manifest = pack(root, archive)
    with ZipFile(archive) as z:
        assert set(z.namelist()) == {"diagnostic/diagnostic_report.json", "fresh/validation_report.json", "export_manifest.json"}
        assert manifest["schema"] == "context_fixed_followup_feedback_v1"
        for name, info in manifest["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == info["sha256"]
    assert root.joinpath(".env").read_text() == "private"


def test_followup_export_rejects_symbolic_report_links(tmp_path):
    root = tmp_path / "output"
    (root / "fresh").mkdir(parents=True)
    original = tmp_path / "private.json"
    original.write_text("{}", encoding="utf-8")
    (root / "fresh" / "failure_report.json").symlink_to(original)
    with pytest.raises(ValueError, match="links"):
        pack(root, tmp_path / "feedback.zip")
