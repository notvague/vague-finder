"""Paired execution, label binding, immutable runs, resumption, and exports."""

import asyncio
import csv
import hashlib
import json
import sys
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from experiments.namuwiki import evaluate_context_fixed as fixed
from experiments.namuwiki.context_fixed_core import EXPECTED_NEW, SETTING, fixed_report
from experiments.namuwiki.evaluate_context_step8 import Query
from experiments.namuwiki.pack_context_fixed_results import pack
from src.backend.schemas.query import ContextClue, ModalityWeights, QueryAnalysis
from tests.context.test_context_fixed_core import independent_rows, row


def analysis(query="synthetic query", *, context=False):
    return QueryAnalysis(original_query=query, intent_type="mixed", confidence=0.9,
                         has_visual_clue=True,
                         image_english_query="a pale square cover", audio_english_query="piano music",
                         modality_weights=ModalityWeights(text=0.5, image=0.5, audio=0.0),
                         context_clues=[ContextClue(target="work", relation="used in",
                                                    search_query=query, confidence=0.8)] if context else [])


class ObservedRouter:
    def __init__(self, *, skip_query=False, query_error=False, bad_order=False, rerank=False):
        self.calls = []
        self.skip_query, self.query_error, self.bad_order, self.rerank = skip_query, query_error, bad_order, rerank

    async def search(self, value, **kwargs):
        self.calls.append((value, kwargs))
        assert kwargs["top_k"] == 10 and kwargs["candidate_k"] == 30
        assert kwargs["use_rerank"] is False
        timer = kwargs["timer"]
        with timer.step("path.image"):
            with timer.step("image.embed"):
                pass
            if not self.skip_query:
                try:
                    with timer.step("image.query"):
                        if self.query_error:
                            raise TimeoutError("synthetic DB error")
                except TimeoutError:
                    pass  # Real router also records a failing path and refills.
        if value.context_clues and kwargs["use_context"]:
            with timer.step("path.context"):
                pass
        if self.rerank:
            with timer.step("search.rerank"):
                pass
        candidates = ["answer", *(f"other{i}" for i in range(29))]
        kwargs["candidate_ids_out"].extend(candidates)
        top = candidates[1:11] if self.bad_order else candidates[:10]
        return [SimpleNamespace(id=sid) for sid in top]


def test_actual_spans_and_identical_analysis_are_used_in_both_arms():
    query = Query("fresh_test", "independent", "regression", "regression_image", "",
                  "synthetic query", frozenset({"answer"}))
    router, value = ObservedRouter(), analysis()
    result = asyncio.run(fixed.evaluate_question(query, value, router, frozenset({"answer"}), {}, required_path="image"))
    assert len(router.calls) == 2 and all(call[0] is value for call in router.calls)
    assert {call[1]["use_context"] for call in router.calls} == {False, True}
    assert value.modality_weights.image == 0.5  # Evaluator never fixes coverage by forcing weights.
    assert result["off_required_path_verified"] and result["on_required_path_verified"]
    assert result["off_top10_rank"] == result["on_top10_rank"] == 1
    assert result["off_target_paths"] == {"answer": []}


@pytest.mark.parametrize("kwargs,field", [
    ({"skip_query": True}, "on_required_path_verified"),
    ({"query_error": True}, "on_required_path_verified"),
    ({"rerank": True}, "rerank_executed"),
])
def test_real_observation_exposes_skips_errors_and_unexpected_reranking(kwargs, field):
    query = Query("image", "test", "regression", "regression_image", "",
                  "synthetic query", frozenset({"answer"}))
    result = asyncio.run(fixed.evaluate_question(query, analysis(), ObservedRouter(**kwargs),
                                               frozenset({"answer"}), {}, required_path="image"))
    assert result[field] is (field == "rerank_executed")
    report = fixed_report([result], phase="test", expected=1)
    assert report["status"] == "blocked"


def test_candidate_top10_contract_is_not_silently_repaired_by_evaluation():
    with pytest.raises(RuntimeError, match="contract"):
        asyncio.run(fixed._arm(ObservedRouter(bad_order=True), analysis(), True))


def test_search_cannot_mutate_the_analysis_shared_with_the_other_arm():
    class MutatingRouter(ObservedRouter):
        async def search(self, value, **kwargs):
            result = await super().search(value, **kwargs)
            value.korean_tags.append("mutated")
            return result
    with pytest.raises(RuntimeError, match="mutated the shared"):
        asyncio.run(fixed._arm(MutatingRouter(), analysis(), False))


@pytest.fixture
def inputs(tmp_path):
    existing = {"prior": Query("prior", "dev", "context", "context_context", "context",
                               "previous question", frozenset({"old_target"}))}
    disclosed = tmp_path / "disclosed.csv"
    disclosed.write_text("query_id,relevant_ids\ndisclosed,another_old_target\n", encoding="utf-8")
    artifacts = tmp_path / "songs"
    artifacts.mkdir()
    raw, reviews = [], []
    for group, count in EXPECTED_NEW.items():
        for _ in range(count):
            qid = f"fresh{len(raw)+1:03d}"
            sid = str(10000 + len(raw))
            positive = group in {"context_context", "context_mixed"}
            record_id = f"nw:{sid}:synthetic"
            query = f"new question {qid}"
            raw.append(dict(query_id=qid, query=query, relevant_ids=sid, group=group,
                            clue_category="production" if positive else "no_context",
                            evidence_record_ids=record_id if positive else "", review_note="reviewed",
                            required_path={"regression_image": "image", "regression_mood_sound": "audio"}.get(group, "")))
            artifact = dict(song={"song_id": sid}, status="ok" if positive else "not_found",
                            source={"url": "https://example.test/fact"},
                            retrieval={"records": [dict(record_id=record_id, category="production",
                                                        evidence_text="A synthetic event about the current song.")]
                                       if positive else []})
            (artifacts / (sid + ".json")).write_text(json.dumps(artifact), encoding="utf-8")
            reviews.append(dict(query_id=qid, song_id=sid, source_url="https://example.test/fact",
                                reviewed_facts=[dict(record_id=record_id, fact_text="A synthetic event about the current song.")]
                                if positive else [], cover_url="https://example.test/cover.jpg", cover_sha256="a" * 64,
                                cover_visual_review=query, indexed_songs_with_same_cover=1))
    csv_path = tmp_path / "new.csv"
    audit_path = tmp_path / "review.json"

    def save():
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
            writer.writeheader()
            writer.writerows(raw)
        fixed.old._write(audit_path, dict(schema="authored_independent_context_review_v1", case_count=60,
                                         group_counts=EXPECTED_NEW, fact_category_counts={"production": 36},
                                         csv_sha256=fixed.old._hash(csv_path),
                                         excluded_disclosed_csv_sha256=fixed.old._hash(disclosed),
                                         excluded_target_ids=["old_target", "another_old_target"], reviews=reviews))

    save()
    return SimpleNamespace(existing=existing, disclosed=disclosed, artifacts=artifacts,
                           raw=raw, reviews=reviews, csv_path=csv_path, audit_path=audit_path, save=save)


def load(data):
    return fixed.load_independent(data.csv_path, data.audit_path, data.existing, data.disclosed, artifact_dir=data.artifacts)


def test_all_new_cases_require_reviewed_song_bound_labels_and_fixed_counts(inputs):
    queries, cases = load(inputs)
    assert len(queries) == 60
    assert len({sid for q in queries.values() for sid in q.relevant_ids}) == 60
    assert all(q.split == "independent" for q in queries.values())
    assert sum(c["required_path"] == "image" for c in cases.values()) == 6


@pytest.mark.parametrize("kind", ["prior_song", "disclosed_song", "prior_qid", "duplicate_song", "duplicate_query", "empty_review", "missing_path"])
def test_overlap_and_incomplete_reviews_cannot_enter_new_evaluation(inputs, kind):
    if kind == "prior_song":
        inputs.raw[0]["relevant_ids"] = "old_target"
    elif kind == "disclosed_song":
        inputs.raw[0]["relevant_ids"] = "another_old_target"
    elif kind == "prior_qid":
        inputs.raw[0]["query_id"] = "prior"
    elif kind == "duplicate_song":
        inputs.raw[1]["relevant_ids"] = inputs.raw[0]["relevant_ids"]
    elif kind == "duplicate_query":
        inputs.raw[1]["query"] = inputs.raw[0]["query"]
    elif kind == "empty_review":
        inputs.raw[0]["review_note"] = ""
    else:
        inputs.raw[-1]["required_path"] = ""
    inputs.save()
    with pytest.raises(ValueError):
        load(inputs)


@pytest.mark.parametrize("kind", ["review_song", "source_url", "fact_text", "category", "missing_fact", "cover_hash", "cover_text", "cover_ambiguity", "duplicate_review"])
def test_song_event_and_actual_cover_review_binding(inputs, kind):
    if kind == "review_song":
        inputs.reviews[0]["song_id"] = "another"
    elif kind == "source_url":
        inputs.reviews[0]["source_url"] = "https://example.test/another"
    elif kind == "fact_text":
        inputs.reviews[0]["reviewed_facts"][0]["fact_text"] = "different fact"
    elif kind == "category":
        inputs.raw[0]["clue_category"] = "another_category"
    elif kind == "missing_fact":
        inputs.raw[0]["evidence_record_ids"] = "not_stored"
    elif kind == "cover_hash":
        inputs.reviews[-1]["cover_sha256"] = "unverified"
    elif kind == "cover_text":
        inputs.reviews[-1]["cover_visual_review"] = "changed picture"
    elif kind == "cover_ambiguity":
        inputs.reviews[-1]["indexed_songs_with_same_cover"] = 2
    else:
        inputs.reviews[-1]["query_id"] = inputs.reviews[-2]["query_id"]
    inputs.save()
    with pytest.raises(ValueError):
        load(inputs)


def test_dataset_bytes_are_checked_before_any_model_or_db_use(inputs):
    with inputs.csv_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        load(inputs)


def test_no_fact_control_cannot_be_an_empty_label_for_a_song_with_facts(inputs):
    control = next(r for r in inputs.raw if r["group"] == "context_control")
    p = inputs.artifacts / (control["relevant_ids"] + ".json")
    data = fixed.old._read(p)
    data["retrieval"]["records"] = [dict(record_id="unexpected", evidence_text="has a fact")]
    fixed.old._write(p, data)
    with pytest.raises(ValueError, match="control has facts"):
        load(inputs)


def mock_registration(monkeypatch, inputs, root):
    monkeypatch.setattr(fixed.old, "_preflight_labels", lambda: (inputs.existing, {"old": "sha"}, {}))
    original = fixed.load_independent
    monkeypatch.setattr(fixed, "load_independent", lambda a, b, c, d: original(a, b, c, d, artifact_dir=inputs.artifacts))
    monkeypatch.setattr(fixed, "_code_paths", lambda: [inputs.audit_path])
    return lambda: fixed._registration(root, inputs.csv_path, inputs.audit_path, inputs.disclosed)


def test_registration_is_immutable_and_does_not_overwrite_old_analysis(monkeypatch, inputs, tmp_path):
    old_cache = tmp_path / "old_analysis_cache.json"
    old_cache.write_text("keep", encoding="utf-8")
    root = tmp_path / "experiment"
    register = mock_registration(monkeypatch, inputs, root)
    first = register()[0]
    assert register()[0] == first
    assert old_cache.read_text() == "keep"
    assert (root / "registered_new_labels.csv").read_bytes() == inputs.csv_path.read_bytes()
    monkeypatch.setenv("CONTEXT_WEIGHT", "0.9")
    with pytest.raises(ValueError, match="registered code/input/settings changed"):
        register()


@pytest.mark.parametrize("filename", ["dev_analysis_cache.json", "dev_checkpoint.json", "dev_report.json"])
def test_new_run_refuses_unregistered_cached_or_evaluated_results(monkeypatch, inputs, tmp_path, filename):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / filename).write_text("{}", encoding="utf-8")
    register = mock_registration(monkeypatch, inputs, root)
    with pytest.raises(ValueError, match="unregistered"):
        register()


def test_public_report_environment_excludes_secrets(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://user:private@example.test")
    monkeypatch.setenv("CLAP_API_KEY", "private")
    monkeypatch.setenv("SEARCH_TOKEN", "private")
    monkeypatch.setenv("QDRANT_IMAGE_COLLECTION", "image_index")
    result = fixed._environment()
    assert "QDRANT_URL" not in result and "CLAP_API_KEY" not in result and "SEARCH_TOKEN" not in result
    assert result["QDRANT_IMAGE_COLLECTION"] == "image_index"


@pytest.fixture
def backend(monkeypatch):
    module = types.ModuleType("src.backend.api.dependencies")
    module.closed = []
    module.close_vector_client = lambda: module.closed.append(True)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fixed, "text_bm25_identity", lambda _: {})
    return module


def test_new_analysis_calls_analyzer_and_resume_only_reuses_this_runs_cache(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_REFERENCE_YEAR", "2026")
    calls = []
    backend.get_query_analyzer = lambda: SimpleNamespace(analyze=lambda q: (calls.append(q), analysis(q, context=True))[1])
    query = Query("new", "dev", "context", "context_context", "context", "new query", frozenset({"answer"}))
    old_cache = tmp_path / "old_cache.json"
    old_cache.write_text("old cache is not consulted", encoding="utf-8")
    path = tmp_path / "new_cache.json"
    first, sha = fixed.old._analyses({"new": query}, path, populate=True)
    second, again = fixed.old._analyses({"new": query}, path, populate=True)
    assert calls == ["new query"] and sha == again
    assert first["new"].model_dump() == second["new"].model_dump()
    assert old_cache.read_text() == "old cache is not consulted"


def test_fallback_is_not_saved_as_a_valid_fresh_analysis(backend, monkeypatch, tmp_path):
    from src.retrieval.query_analyzer import rule_fallback

    monkeypatch.setenv("SEARCH_REFERENCE_YEAR", "2026")
    backend.get_query_analyzer = lambda: SimpleNamespace(analyze=rule_fallback)
    query = Query("bad", "dev", "context", "context_context", "context", "short query", frozenset({"answer"}))
    path = tmp_path / "new_cache.json"
    with pytest.raises(ValueError, match="fallback"):
        fixed.old._analyses({"bad": query}, path, populate=True)
    assert not path.exists()


def test_fresh_cache_has_experiment_binding_and_rejects_a_renamed_old_cache(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_REFERENCE_YEAR", "2026")
    calls = []
    backend.get_query_analyzer = lambda: SimpleNamespace(analyze=lambda q: (calls.append(q), analysis(q, context=True))[1])
    query = Query("fresh", "dev", "context", "context_context", "context", "new query", frozenset({"answer"}))
    path = tmp_path / "fresh_cache.json"
    registered = {"code": "frozen"}
    fixed.fresh_analyses({"fresh": query}, path, registered, "dev")
    fixed.fresh_analyses({"fresh": query}, path, registered, "dev")
    assert calls == ["new query"]
    with pytest.raises(ValueError, match="registered experiment"):
        fixed.fresh_analyses({"fresh": query}, path, registered, "test")
    prior = tmp_path / "renamed_old_cache.json"
    fixed.old._analyses({"fresh": query}, prior, populate=True)
    with pytest.raises(ValueError, match="registered experiment"):
        fixed.fresh_analyses({"fresh": query}, prior, registered, "dev")


def test_phase_preflight_failure_still_closes_an_opened_client(backend, monkeypatch, tmp_path):
    query = Query("dev1", "dev", "context", "context_context", "context", "query", frozenset({"answer"}))
    monkeypatch.setattr(fixed.old, "_analyses", lambda *a, **k: ({}, "cache"))
    monkeypatch.setattr(fixed.old, "_corpus", lambda *a: ({"build": "fixed"}, object(), frozenset({"answer"})))
    def fail(*_):
        raise ValueError("image target missing")
    monkeypatch.setattr(fixed, "_modality_identity", fail)
    with pytest.raises(ValueError, match="image target missing"):
        asyncio.run(fixed.measure_phase("dev", tmp_path, {}, {"dev1": query}, {}))
    assert backend.closed == [True]


def test_resume_keeps_completed_pairs_and_rejects_a_changed_corpus(backend, monkeypatch, tmp_path):
    queries = {key: Query(key, "dev", "context", "context_context", "context", key, frozenset({"answer"}))
               for key in ("first", "second")}
    registration = {"phase_counts": {"dev": 2}}
    monkeypatch.setattr(fixed, "fresh_analyses", lambda *a: ({key: object() for key in queries}, "same_cache"))
    identity = {"build": "same"}
    monkeypatch.setattr(fixed.old, "_corpus", lambda *_: (deepcopy(identity), SimpleNamespace(), frozenset({"answer"})))
    monkeypatch.setattr(fixed.old, "_assert_unchanged", lambda *_: None)
    monkeypatch.setattr(fixed, "_assert_registered_files", lambda *_: None)
    monkeypatch.setattr(fixed, "_modality_identity", lambda *_: {})
    calls = []
    async def interrupted(q, *_args, **_kwargs):
        calls.append(q.query_id)
        if q.query_id == "second":
            raise RuntimeError("temporary failure")
        return row(q.query_id)
    monkeypatch.setattr(fixed, "evaluate_question", interrupted)
    with pytest.raises(RuntimeError, match="temporary failure"):
        asyncio.run(fixed.measure_phase("dev", tmp_path, registration, queries, {}))
    assert list(fixed.old._read(tmp_path / "dev_checkpoint.json")["rows"]) == ["first"]
    async def completed(q, *_args, **_kwargs):
        calls.append(q.query_id)
        return row(q.query_id)
    monkeypatch.setattr(fixed, "evaluate_question", completed)
    report = asyncio.run(fixed.measure_phase("dev", tmp_path, registration, queries, {}))
    assert calls == ["first", "second", "second"]
    assert report["evaluated"] == 2
    identity["build"] = "changed"
    with pytest.raises(ValueError, match="corpus"):
        asyncio.run(fixed.measure_phase("dev", tmp_path, registration, queries, {}))
    assert backend.closed == [True, True, True]


def test_reanalysis_uses_predeclared_groups_and_is_not_added_to_the_independent_denominator(backend, monkeypatch, tmp_path):
    data = independent_rows()
    queries = {r["query_id"]: Query(r["query_id"], "independent", r["source"], r["group"], "",
                                   r["query_id"], frozenset({"answer"})) for r in data}
    cases = {key: {"required_path": {"regression_image": "image", "regression_mood_sound": "audio"}.get(q.group, "")}
             for key, q in queries.items()}
    main = fixed_report(data, phase="independent", expected=60)
    main.update(corpus={"same": True}, question_results=[dict(r, analysis={}) for r in data])
    selected_keys = []
    def analyses(selected, *_):
        selected_keys.extend(selected)
        return {key: object() for key in selected}, "cache"
    monkeypatch.setattr(fixed, "fresh_analyses", analyses)
    monkeypatch.setattr(fixed.old, "_corpus", lambda *_: ({"same": True}, SimpleNamespace(), frozenset({"answer"})))
    monkeypatch.setattr(fixed.old, "_assert_unchanged", lambda *_: None)
    monkeypatch.setattr(fixed, "_assert_registered_files", lambda *_: None)
    monkeypatch.setattr(fixed, "_modality_identity", lambda *_: {})
    async def measured(q, *_args, required_path, **_kwargs):
        original = next(r for r in data if r["query_id"] == q.query)
        return dict(original, query_id=q.query_id, required_path=required_path, analysis={}, off_path_errors=[], on_path_errors=[])
    monkeypatch.setattr(fixed, "evaluate_question", measured)
    result = asyncio.run(fixed.measure_variation(tmp_path, {}, queries, cases, main))
    assert result["question_count"] == result["extra_fresh_analyses"] == 10
    assert len(selected_keys) == len(set(selected_keys)) == 10
    assert all(key.startswith("repeat_") for key in selected_keys)
    assert not any("control" in key for key in selected_keys)
    assert main["evaluated"] == 60 and not result["violations"]


def test_existing_regression_failure_prevents_consuming_new_independent_cases(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_REFERENCE_YEAR", "2026")
    registration = {"fixed": True}
    monkeypatch.setattr(fixed, "_registration", lambda *a: (registration, {}, {}, {}))
    phases = []
    async def measure(phase, *_):
        phases.append(phase)
        report = fixed_report([row(phase)], phase=phase, expected=1)
        report.update(corpus={"same": True}, registration=registration)
        if phase == "test":
            report["violations"] = ["synthetic regression loss"]
        return report
    monkeypatch.setattr(fixed, "measure_phase", measure)
    args = SimpleNamespace(output_dir=tmp_path, new_csv=None, new_audit=None, disclosed_csv=None, phase="all")
    assert asyncio.run(fixed.run(args)) == 4
    assert phases == ["dev", "test"]
    assert fixed.old._read(tmp_path / "validation_report.json")["status"] == "blocked_before_independent"


def test_corpus_lock_prevents_new_queries_after_same_count_index_content_changes(tmp_path):
    before = dict(text_points=3016, local_collection_sha256={"image": "before"})
    fixed.lock_corpus(tmp_path, before)
    fixed.lock_corpus(tmp_path, deepcopy(before))
    with pytest.raises(ValueError, match="before new analysis/search"):
        fixed.lock_corpus(tmp_path, dict(before, local_collection_sha256={"image": "after"}))


def test_code_input_and_environment_changes_are_checked_again_at_phase_end(monkeypatch, tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"frozen")
    registration = dict(source_hashes={str(path): fixed.old._hash(path)}, code_sha256={},
                        environment=fixed._environment(), runtime=fixed._runtime())
    fixed._assert_registered_files(registration)
    path.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="registered input/code changed"):
        fixed._assert_registered_files(registration)


def test_missing_text_sparse_baseline_is_not_silently_compared_as_context_gain(tmp_path):
    path = tmp_path / "bm25.json"
    fixed.old._write(path, {"n_docs": 12})
    encoder = SimpleNamespace(params_path=path, _is_fitted=True)
    router = SimpleNamespace(_text_svc=SimpleNamespace(bm25_encoder=encoder))
    result = fixed.text_bm25_identity(router)
    assert result["text_bm25_fit_documents"] == 12
    assert result["text_bm25_sha256"] == fixed.old._hash(path)
    encoder._is_fitted = False
    with pytest.raises(ValueError, match="working baseline"):
        fixed.text_bm25_identity(router)


def test_phase_reports_from_another_registered_run_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_REFERENCE_YEAR", "2026")
    monkeypatch.setattr(fixed, "_registration", lambda *a: ({"current": True}, {}, {}, {}))
    fixed.old._write(tmp_path / "dev_report.json", {"registration": {"old": True}})
    args = SimpleNamespace(output_dir=tmp_path, new_csv=None, new_audit=None, disclosed_csv=None, phase="test")
    with pytest.raises(ValueError, match="another registered run"):
        asyncio.run(fixed.run(args))


def test_export_is_a_whitelist_with_bytes_hashes_and_no_credentials_or_models(tmp_path):
    root = tmp_path / "report"
    root.mkdir()
    for name in ("registration.json", "validation_report.json", "failure_report.json"):
        (root / name).write_text(json.dumps({"status": "synthetic"}), encoding="utf-8")
    for name in (".env", "model.bin", "analysis_cache.json", "raw_song.mp4"):
        (root / name).write_bytes(b"not exported")
    archive = tmp_path / "feedback.zip"
    result = pack(root, archive)
    with ZipFile(archive) as zip_file:
        assert set(zip_file.namelist()) == {"registration.json", "validation_report.json", "failure_report.json", "export_manifest.json"}
        for name, expected in result["files"].items():
            assert hashlib.sha256(zip_file.read(name)).hexdigest() == expected["sha256"]


def test_empty_export_does_not_create_a_misleading_results_zip(tmp_path):
    with pytest.raises(ValueError, match="No evaluation reports"):
        pack(tmp_path / "missing", tmp_path / "empty.zip")


def test_cli_interruption_is_saved_without_deleting_completed_rows(monkeypatch, tmp_path):
    checkpoint = tmp_path / "dev_checkpoint.json"
    checkpoint.write_text('{"completed": 3}', encoding="utf-8")
    async def interrupted(_):
        raise ValueError("query42: analyzer fallback")
    monkeypatch.setattr(fixed, "run", interrupted)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--output-dir", str(tmp_path)])
    assert fixed.main() == 2
    assert fixed.old._read(tmp_path / "failure_report.json")["error_type"] == "ValueError"
    assert checkpoint.read_text() == '{"completed": 3}'


def test_report_table_is_contiguous_and_never_asserts_universal_or_ui_optimality(tmp_path):
    report = fixed_report([row()], phase="dev", expected=1)
    fixed._write_report(tmp_path, {"dev": report}, {}, status="partial")
    markdown = (tmp_path / "validation_report.md").read_text(encoding="utf-8")
    assert "partial" in markdown and "최적값을 입증한 것은 아니다" in markdown
    saved = fixed.old._read(tmp_path / "validation_report.json")
    assert saved["completed_phases"] == ["dev"]
