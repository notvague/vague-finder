"""The repair diagnostic refuses changed baselines, settings and inputs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.namuwiki import diagnose_context_step10_candidate_fix as diagnostic
from experiments.namuwiki import evaluate_context_step10 as step10
from experiments.namuwiki.context_step10_core import Setting
from experiments.namuwiki.evaluate_context_step8 import Query
from src.backend.schemas.query import QueryAnalysis
from src.retrieval.explain import ExplainRecorder


@pytest.fixture
def trial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    query = "팬 요청으로 더블 타이틀로 바꾼 가상 가수의 곡"
    row = Query("synthetic001", "blind", "context", "context_context", "", query,
                frozenset({"target"}))
    original = QueryAnalysis(original_query=query, intent_type="mixed", confidence=0.8,
                             image_english_query="", audio_english_query="")
    files = tuple(Path("src/retrieval") / name for name in (
        "search_router.py", "context_query.py", "context_evidence.py", "context_qdrant_search.py",
    ))
    for path in (*files, Path("src/retrieval/analysis_cache.py")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"controlled current source: {path}", encoding="utf-8")
    code = {str(path): diagnostic._hash(path) for path in files}
    old_code = dict(code)
    old_code["src/retrieval/search_router.py"] = "old-router"
    old_code["src/retrieval/context_evidence.py"] = "old-evidence"
    old_code["src/retrieval/context_query.py"] = "old-query"
    previous_code = dict(code)
    previous_code["src/retrieval/search_router.py"] = "old-router"
    previous_code["src/retrieval/context_evidence.py"] = "previous-evidence"
    corpus = dict(context_build_id="controlled-build", source_song_count=3016,
                  dense_record_count=20, sparse_profile_count=10, text_points=3016,
                  context_manifest_sha256="manifest", state_manifest_sha256="state")
    setting = Setting(0.5, 100, 100, 30, named_media_multiplier=2.0)
    blind_csv = Path("data/context/eval_blind_step10.csv")
    blind_csv.parent.mkdir(parents=True)
    blind_csv.write_text("controlled labels", encoding="utf-8")
    cache_path = diagnostic.OLD / "blind_analysis_cache.json"
    diagnostic._write(cache_path, dict(format=1, meta={}, entries={
        row.query_id: dict(query=query, fallback=False, analysis=original.model_dump(mode="json")),
    }))
    stamp = dict(blind_csv_sha256=diagnostic._hash(blind_csv),
                 blind_analysis_cache_sha256=diagnostic._hash(cache_path),
                 sources={"source": "frozen"}, code_sha256=old_code, corpus=corpus)
    old_results = {row.query_id: dict(ref_candidate_ids="old", off_candidate_rank=None,
                                    on_candidate_rank=None, off_top10_rank=None, on_top10_rank=None)}
    diagnostic._write(diagnostic.OLD / "blind_report.json", dict(
        stamp=stamp, locked_setting=setting.key,
        verdict=dict(status="not_recommended", setting=setting.as_dict()),
    ))
    diagnostic._write(diagnostic.OLD / "blind_checkpoint.json", dict(
        stamp=stamp, runs={setting.key: old_results},
    ))
    previous_stamp = dict(
        schema="disclosed_step10_context_fix_diagnostic_v1",
        blind_csv_sha256=diagnostic._hash(blind_csv),
        old_report_sha256=diagnostic._hash(diagnostic.OLD / "blind_report.json"),
        frozen_analysis_sha256=diagnostic._hash(cache_path),
        analysis_cache_code_sha256=diagnostic._hash(Path("src/retrieval/analysis_cache.py")),
        code_sha256=previous_code, corpus=corpus, setting=setting.as_dict(),
    )
    diagnostic._write(diagnostic.PREVIOUS / "diagnostic_report.json", dict(
        stamp=previous_stamp, rows=[dict(query_id=row.query_id, context_clue_count=1,
                                       after={key: None for key in (
                                           "off_candidate_rank", "on_candidate_rank",
                                           "off_top10_rank", "on_top10_rank")})],
        evidence_needs_review=[],
    ))
    monkeypatch.setattr(diagnostic, "CODE_FILES", files)
    monkeypatch.setattr(diagnostic, "_preflight_labels", lambda: ({}, stamp["sources"], {}))
    monkeypatch.setattr(diagnostic, "_blind_queries", lambda *_: ({row.query_id: row}, {}))
    monkeypatch.setattr(diagnostic, "_corpus", lambda _: (corpus, SimpleNamespace(), row.relevant_ids))
    monkeypatch.setattr(step10, "_assert_unchanged", lambda *_: None)
    closed = []
    monkeypatch.setattr(diagnostic, "close_vector_client", lambda: closed.append(True))
    calls = []
    async def search(_router, _analysis, *, setting, candidate_k):
        calls.append(setting is not None)
        ids = ["target", "old"] if setting else ["old"]
        return dict(candidates=ids, top=ids, tracks=[], context_hits=[], evidence=[], elapsed_ms=1)
    monkeypatch.setattr(diagnostic, "_search", search)
    args = SimpleNamespace(old_output=diagnostic.OLD, previous_output=diagnostic.PREVIOUS,
                           output_dir=diagnostic.OUT, blind_csv=blind_csv,
                           all=True, query_ids=[])
    return SimpleNamespace(args=args, closed=closed, calls=calls, corpus=corpus)


def test_disclosed_diagnosis_saves_gains_and_reuses_matching_checkpoint(trial):
    assert asyncio.run(diagnostic.run(trial.args)) == 0
    report = diagnostic._read(diagnostic.OUT / "diagnostic_report.json")
    group = report["groups"]["context_positive"]
    assert report["evaluated"] == 1
    assert group["after_off_top10"] == 0 and group["after_on_top10"] == 1
    assert group["after_off_mrr10"] == 0 and group["after_on_mrr10"] == 1
    assert group["after_off_to_on_top10_gained"] == ["synthetic001"]
    assert "NOT a fresh blind pass" in report["label"]
    assert trial.calls == [False, True] and trial.closed
    assert asyncio.run(diagnostic.run(trial.args)) == 0
    assert trial.calls == [False, True]


@pytest.mark.parametrize("name", ["context_build_id", "text_points", "source_song_count"])
def test_changed_corpus_is_rejected_before_search(trial, name):
    trial.corpus[name] = "changed"
    with pytest.raises(ValueError, match="corpus changed"):
        asyncio.run(diagnostic.run(trial.args))
    assert not trial.calls


def test_unexpected_code_change_is_rejected(trial):
    Path("src/retrieval/context_qdrant_search.py").write_text("unexpected change")
    with pytest.raises(ValueError, match="other retrieval/evaluation code changed"):
        asyncio.run(diagnostic.run(trial.args))


def test_changed_frozen_cache_is_rejected(trial):
    path = diagnostic.OLD / "blind_analysis_cache.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="old analysis cache has changed"):
        asyncio.run(diagnostic.run(trial.args))


def test_changed_off_baseline_fails_and_closes_vector_client(trial, monkeypatch):
    async def different(*_args, **_kwargs):
        return dict(candidates=["changed"], top=["changed"])
    monkeypatch.setattr(diagnostic, "_search", different)
    with pytest.raises(RuntimeError, match="OFF@30 baseline changed"):
        asyncio.run(diagnostic.run(trial.args))
    assert trial.closed


def test_old_report_and_previous_report_cannot_be_overwritten(trial):
    for path in (diagnostic.OLD, diagnostic.PREVIOUS):
        trial.args.output_dir = path
        with pytest.raises(ValueError, match="separate output directory"):
            asyncio.run(diagnostic.run(trial.args))


@pytest.mark.parametrize("path", [Path("data/output"), Path("artifacts/../data"), Path("/tmp/output")])
def test_private_diagnostic_paths_cannot_escape_artifacts(path):
    with pytest.raises(ValueError, match="ignored artifacts"):
        diagnostic._private(path)


def test_precut_scores_remain_inspectable_for_a_dropped_target():
    recorder = ExplainRecorder("synthetic query")
    recorder.path("target", "context", 1, 0.4 / 61)
    recorder.set_fused("target", 0.4 / 61)
    recorder.adjust("target", "artist_match", 1 / 61)
    row = diagnostic._target_scores(recorder, {"target"}, ["other"])[0]
    assert row["in_final_pool"] is False
    assert row["context_paths"] == [dict(rank=1, contribution=0.4 / 61)]
    assert row["score_before_cut"] == pytest.approx(1.4 / 61)


def test_recording_proxy_passes_the_same_search_arguments():
    recorder = ExplainRecorder("synthetic query")
    seen = []
    async def search(analysis, **kwargs):
        seen.append((analysis, kwargs))
        return ["unchanged result"]
    proxy = diagnostic._RecordingRouter(SimpleNamespace(search=search), recorder)
    assert asyncio.run(proxy.search("analysis", top_k=10, use_rerank=False)) == ["unchanged result"]
    assert seen == [("analysis", dict(top_k=10, use_rerank=False, recorder=recorder))]
