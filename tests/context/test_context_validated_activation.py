"""Activation requires real completed evidence and verifies the HTTP contract."""

import json

import pytest

from experiments.namuwiki.apply_validated_context_settings import (
    DEPENDENCIES, ROUTER_VALUES, SETTING, apply, rewrite_factory, sha, validate_pass,
)
from experiments.namuwiki.verify_context_api_ready import check_openapi, check_response, select_cases, verify, wait_ready
from experiments.namuwiki import verify_context_api_ready as api
from experiments.namuwiki.capture_context_cover_dance_fix import attach_console, main as capture_main


def passed_report():
    registration = dict(schema="context_fixed_registered_v1", setting=SETTING, rerank_enabled=False,
                        reference_year=2026, source_hashes={}, code_sha256={})
    reports = {phase: dict(status="passed", evaluated=count, setting=SETTING, rerank_enabled=False,
                          violations=[], evidence=dict(unreviewed_count=0),
                          corpus=dict(build="synthetic-build", source_scope_complete=True))
               for phase, count in dict(dev=74, test=32, independent=60).items()}
    report = dict(schema="context_fixed_validation_v1", status="passed_observed_tests", registration=registration,
                  completed_phases=list(reports), reports=reports,
                  analyzer_repeat_check=dict(question_count=10, violations=[], evidence=dict(unreviewed_count=0)))
    return report, registration


@pytest.mark.parametrize("failure", [
    "in_progress", "blocked", "partial", "evidence_review_required", "missing_independent",
    "wrong_count", "phase_error", "citation_review", "rerank_on", "different_setting",
    "different_corpus", "no_repeat", "repeat_loss", "repeat_citation", "wrong_repeat_count",
    "different_registration", "wrong_reference_year", "partial_corpus",
])
def test_partial_failed_unreviewed_or_inconsistent_measurements_cannot_apply(failure):
    report, registration = passed_report()
    if failure in {"in_progress", "blocked", "partial", "evidence_review_required"}:
        report["status"] = failure
    elif failure == "missing_independent":
        report["reports"].pop("independent")
    elif failure == "wrong_count":
        report["reports"]["independent"]["evaluated"] = 59
    elif failure == "phase_error":
        report["reports"]["test"]["violations"] = ["loss"]
    elif failure == "citation_review":
        report["reports"]["independent"]["evidence"]["unreviewed_count"] = 1
    elif failure == "rerank_on":
        report["reports"]["dev"]["rerank_enabled"] = True
    elif failure == "different_setting":
        registration["setting"] = {**SETTING, "weight": 1.0}
    elif failure == "different_corpus":
        report["reports"]["test"]["corpus"] = dict(build="changed", source_scope_complete=True)
    elif failure == "partial_corpus":
        report["reports"]["test"]["corpus"]["source_scope_complete"] = False
    elif failure == "no_repeat":
        report["analyzer_repeat_check"] = None
    elif failure == "repeat_loss":
        report["analyzer_repeat_check"]["violations"] = ["regression"]
    elif failure == "repeat_citation":
        report["analyzer_repeat_check"]["evidence"]["unreviewed_count"] = 1
    elif failure == "wrong_repeat_count":
        report["analyzer_repeat_check"]["question_count"] = 9
    elif failure == "different_registration":
        report["registration"] = {**registration, "extra": "different"}
    else:
        registration["reference_year"] = 2027
    with pytest.raises(ValueError):
        validate_pass(report, registration)


@pytest.mark.parametrize("data", [
    b"def get_search_router():\n    return SearchRouter(context_search=object())\n",
    b"def get_search_router():\n    return SearchRouter(context_search=object(),)\n",
    b"def get_search_router():\n    return SearchRouter(\n        context_search=object()\n    )\n",
    b"def get_search_router():\n    return SearchRouter(\n        context_search=object(), # keep comment\n    )\n",
    b"def get_search_router():\n    return SearchRouter(context_search=object(), context_weight=float('1.0'), context_sparse_k=80)\n",
    b"def get_search_router():\n    return SearchRouter(\n        context_search=object(),\n        context_weight=1.0,\n        context_fact_k=160,\n        context_sparse_k=160,\n        context_named_media_multiplier=1.0,\n        default_candidate_k=40,\n    )\n",
    "# 한글을 포함한 UTF-8 소스\ndef get_search_router():\n    return SearchRouter(context_search='한글 값', context_weight=1.0)\n".encode(),
    b"\xef\xbb\xbfdef get_search_router():\r\n    return SearchRouter(\r\n        context_search=object(), # keep comment\r\n    )\r\n",
])
def test_precise_factory_edit_preserves_source_and_applies_exact_values(data):
    prefix = b"# untouched preceding code\n"
    suffix = b"\ndef unrelated():\n    return SearchRouter(context_weight=4.0)\n"
    bom = b"\xef\xbb\xbf" if data.startswith(b"\xef\xbb\xbf") else b""
    source = bom + prefix + data[len(bom):] + suffix
    changed = rewrite_factory(source)
    assert changed.startswith(bom + prefix) and changed.endswith(suffix)
    if b"keep comment" in source:
        assert b"# keep comment" in changed
    if b"\r\n" in source:
        assert b"default_candidate_k=30,\r\n" in changed
    namespace = dict(SearchRouter=lambda **kwargs: kwargs)
    exec(compile(changed.decode("utf-8-sig"), "synthetic.py", "exec"), namespace)
    actual = namespace["get_search_router"]()
    assert {key: actual[key] for key in ROUTER_VALUES} == ROUTER_VALUES
    assert namespace["unrelated"]()["context_weight"] == 4.0
    assert rewrite_factory(changed) == changed


@pytest.mark.parametrize("source", [
    "def get_search_router():\n    return SearchRouter()\n",
    "def other():\n    return SearchRouter(context_search=object())\n",
    "def get_search_router():\n    return SearchRouter(context_search=None)\n",
    "def get_search_router():\n    return SearchRouter(context_search=object(), **settings)\n",
    "def get_search_router():\n    a = SearchRouter(context_search=object())\n    return SearchRouter(context_search=object())\n",
])
def test_unknown_factory_shapes_are_refused_without_writing_source(source):
    with pytest.raises(ValueError):
        rewrite_factory(source.encode())


def installation(tmp_path):
    project = tmp_path / "project"
    output = project / "artifacts/current"
    source = project / DEPENDENCIES
    source.parent.mkdir(parents=True)
    source.write_text("# remain unchanged\ndef get_search_router():\n    return SearchRouter(context_search=object())\n", encoding="utf-8")
    fixture = project / "data/context/input.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("synthetic reviewed input")
    report, registration = passed_report()
    registration["source_hashes"] = {"data/context/input.txt": sha(fixture.read_bytes())}
    registration["code_sha256"] = {DEPENDENCIES: sha(source.read_bytes())}
    (output / "fresh").mkdir(parents=True)
    (output / "fresh/validation_report.json").write_text(json.dumps(report))
    (output / "fresh/registration.json").write_text(json.dumps(registration))
    return project, output, source


def test_host_application_backs_up_source_preserves_reports_and_can_resume_api_stage(tmp_path):
    project, output, source = installation(tmp_path)
    before = source.read_bytes()
    frozen_report = (output / "fresh/validation_report.json").read_bytes()
    result = apply(project, output)
    after = source.read_bytes()
    assert before != after and (project / result["backup"]).read_bytes() == before
    assert result["dependencies_after_sha256"] == sha(after) and result["api_verified"] is False
    assert (output / "fresh/validation_report.json").read_bytes() == frozen_report
    result = apply(project, output)
    assert not result["source_changed"] and source.read_bytes() == after
    assert (output / "fresh/validation_report.json").read_bytes() == frozen_report


@pytest.mark.parametrize("change", ["code", "input", "unresolved_failure", "incomplete"])
def test_application_stops_before_writing_when_registration_or_gate_is_wrong(tmp_path, change):
    project, output, source = installation(tmp_path)
    if change == "code":
        source.write_text(source.read_text() + "# another edit\n")
    elif change == "input":
        (project / "data/context/input.txt").write_text("changed input")
    elif change == "unresolved_failure":
        (output / "fresh/failure_report.json").write_text("{}")
    else:
        value = json.loads((output / "fresh/validation_report.json").read_text())
        value["status"] = "in_progress"
        (output / "fresh/validation_report.json").write_text(json.dumps(value))
    before = source.read_bytes()
    with pytest.raises(ValueError):
        apply(project, output)
    assert source.read_bytes() == before and not (output / "activation/activation_report.json").exists()


def response(path="context"):
    ids = [str(index) for index in range(1, 31)]
    return dict(analysis=dict(original_query="합성 질문", context_clues=[{}] if path == "context" else [],
                              modality_weights={path: 0.4}, **{path + "_english_query": "Synthetic dedicated prompt"}),
                explain=dict(analysis_fallback=False, failed_paths=[], rerank_calls=0, rerank_calls_applied=0),
                candidate_ids=ids,
                results=[dict(id=sid, title="Synthetic track", context_evidence=None,
                              explain=dict(paths=[dict(path=path, delta=0.01, rank=1, dropped=False)]))
                         for sid in ids[:10]])


def case(path="context"):
    return dict(query_id="synthetic", query="합성 질문", required_path=path,
                expected_targets=["20"] if path == "context" else [], no_context=path != "context")


def request_timing(path="context", query="합성 질문", request_id="synthetic-request"):
    return dict(request_id=request_id, query=query, attrs=dict(status=200),
                spans=[dict(id=1, name="path." + path, parent=0, run_ms=10.0, wait_ms=0.0),
                       dict(id=2, name=path + ".embed", parent=1, run_ms=3.0, wait_ms=0.0),
                       dict(id=3, name=path + ".query", parent=1, run_ms=4.0, wait_ms=0.0)])


@pytest.mark.parametrize("path", ["context", "image", "audio", "text_hybrid"])
def test_real_api_contract_checks_observed_positive_contributions(path):
    result = check_response(case(path), response(path), timing_record=request_timing(path))
    assert path in result["positive_paths"] and result["candidate_count"] == 30


@pytest.mark.parametrize("failure", [
    "fallback", "path_error", "rerank", "duplicate_candidate", "too_few_candidates",
    "out_of_pool", "rejected_returned", "missing_target", "unknown_title", "wrong_query",
    "no_positive_path", "dropped_path", "nonfinite", "unexpected_context", "evidence_without_clue",
    "empty_fact", "null_fact", "bad_url", "url_credentials", "fact_invalid_port", "null_title",
])
def test_api_smoke_cannot_turn_bad_runtime_behavior_into_a_pass(failure):
    c, body, rejected = case(), response(), ()
    if failure == "fallback":
        body["explain"]["analysis_fallback"] = True
    elif failure == "path_error":
        body["explain"]["failed_paths"] = [dict(path="context", reason="error")]
    elif failure == "rerank":
        body["explain"]["rerank_calls"] = 1
    elif failure == "duplicate_candidate":
        body["candidate_ids"][-1] = "1"
    elif failure == "too_few_candidates":
        body["candidate_ids"].pop()
    elif failure == "out_of_pool":
        body["results"][0]["id"] = "outside"
    elif failure == "rejected_returned":
        rejected = ("1",)
    elif failure == "missing_target":
        c["expected_targets"] = ["unknown"]
    elif failure in {"unknown_title", "null_title"}:
        body["results"][0]["title"] = "Unknown" if failure == "unknown_title" else None
    elif failure == "wrong_query":
        body["analysis"]["original_query"] = "changed"
    elif failure in {"no_positive_path", "dropped_path", "nonfinite"}:
        for track in body["results"]:
            track["explain"]["paths"][0].update(delta=0 if failure == "no_positive_path" else
                                               float("nan") if failure == "nonfinite" else 0.01,
                                               dropped=failure == "dropped_path")
    elif failure == "unexpected_context":
        c["no_context"] = True
    else:
        fact = dict(record_id="synthetic-fact", fact_text="A synthetic reported event.",
                    source_url="https://namu.wiki/w/Synthetic")
        if failure == "evidence_without_clue":
            body["analysis"]["context_clues"] = []
        elif failure == "empty_fact":
            fact["fact_text"] = " "
        elif failure == "null_fact":
            fact["record_id"] = None
        elif failure == "bad_url":
            fact["source_url"] = "http://example.invalid/source"
        elif failure == "url_credentials":
            fact["source_url"] = "https://user:password@namu.wiki/w/Synthetic"
        else:
            fact["source_url"] = "https://namu.wiki:8443/w/Synthetic"
        body["results"][0]["context_evidence"] = fact
    with pytest.raises(ValueError):
        check_response(c, body, rejected=rejected)


def test_optional_none_fact_can_be_omitted_but_step7_schema_is_required():
    body = response()
    for track in body["results"]:
        track.pop("context_evidence")
    check_response(case(), body)
    document = {"paths": {"/api/v1/search": {"post": {"responses": {"200": {"content": {
        "application/json": {"schema": {"$ref": "#/components/schemas/Response"}}
    }}}}}}, "components": {"schemas": {
        "Response": {"properties": {"results": {"items": {"$ref": "#/components/schemas/Track"}}}},
        "Track": {"properties": {"context_evidence": {}}},
    }}}
    check_openapi(document)
    document["components"]["schemas"]["Track"]["properties"].clear()
    with pytest.raises(ValueError, match="context_evidence"):
        check_openapi(document)


def test_api_failure_report_is_saved_without_an_accuracy_claim(tmp_path, monkeypatch):
    project, output, _ = installation(tmp_path)
    apply(project, output)
    monkeypatch.setattr(api, "wait_ready", lambda *args: (_ for _ in ()).throw(TimeoutError("warmup incomplete")))
    result = verify(project, output, "http://127.0.0.1:8000", 0)
    assert result["status"] == "failed_api_smoke" and result["rows"] == []
    assert json.loads((output / "activation/api_smoke_report.json").read_text())["error_type"] == "TimeoutError"


def test_readiness_requires_warmup_ready_not_just_http_200(monkeypatch):
    states = iter([dict(status="ok", warmup=dict(state="loading")),
                   dict(status="ok", warmup=dict(state="ready"))])
    calls = []
    monkeypatch.setattr(api, "request_json", lambda *args, **kwargs: calls.append(args) or next(states))
    monkeypatch.setattr(api.time, "sleep", lambda *_: None)
    wait_ready("http://127.0.0.1:8000", 1)
    assert len(calls) == 2


def test_runtime_reports_are_attached_but_backups_are_not_exported(tmp_path):
    archive, log = tmp_path / "feedback.zip", tmp_path / "run/console_run.log"
    log.parent.mkdir()
    log.write_text("captured API failure\n")
    from zipfile import ZipFile
    payload = b'{"status":"passed_observed_tests"}'
    with ZipFile(archive, "w") as z:
        z.writestr("fresh/validation_report.json", payload)
        z.writestr("export_manifest.json", json.dumps(dict(files={"fresh/validation_report.json":
                     dict(bytes=len(payload), sha256=sha(payload))})))
    root = log.parent / "activation"
    (root / "backups").mkdir(parents=True)
    (root / "activation_report.json").write_text('{"status":"settings_applied"}')
    (root / "api_smoke_report.json").write_text('{"status":"failed_api_smoke"}')
    (root / "backups/private.py").write_text("do not export")
    combined = attach_console(archive, log, child_exit=1)
    with ZipFile(combined) as z:
        assert "activation/api_smoke_report.json" in z.namelist()
        assert "activation/activation_report.json" in z.namelist()
        assert not any("backups" in name for name in z.namelist())
        assert z.read("fresh/validation_report.json") == payload


@pytest.mark.parametrize("option", ["--check-only", "--package-only", "--inspect", "--activate-only"])
def test_application_cannot_be_accidentally_requested_in_check_or_package_mode(option):
    with pytest.raises(SystemExit) as exc:
        capture_main(["--activate-after-pass", option])
    assert exc.value.code == 2


@pytest.mark.parametrize("observation", ["positive_all", "audio_not_in_top10", "audio_query_error",
                                        "all_audio_not_in_top10", "mixed_audio_only_survives"])
def test_full_activation_api_sequence_uses_private_observed_cases_and_refills(tmp_path, monkeypatch, observation):
    project, output, _ = installation(tmp_path)
    apply(project, output)
    entries, rows = {}, []
    for qid, query, group, path, target in (
        ("nw001", "context one", "context_context", "context", "20"),
        ("q203", "context two", "context_context", "context", "20"),
        ("synthetic-image", "image only", "regression_image", "image", "1"),
        ("synthetic-audio", "audio only", "regression_mood_sound", "audio", "1"),
        ("synthetic-control", "control only", "context_control", "text_hybrid", "1"),
    ):
        entries[qid] = dict(query=query, analysis={path + "_english_query": "synthetic prompt"})
        rows.append(dict(query_id=qid, group=group, eligible_ids=target))
    import csv
    (output / "fresh/dev_analysis_cache.json").write_text(json.dumps(dict(entries=entries)))
    with (output / "fresh/dev_detail.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "group", "eligible_ids"])
        writer.writeheader()
        writer.writerows(rows)
    extra_entries, extra_rows = {}, []
    for path, group in (("image", "regression_image"), ("audio", "regression_mood_sound")):
        for index in range(6):
            qid = f"synthetic-{path}-required-{index}"
            extra_entries[qid] = dict(query=f"{path} required {index}", analysis={path + "_english_query": "Prompt"})
            extra_rows.append(dict(query_id=qid, group=group, eligible_ids="1", required_path=path))
    (output / "fresh/independent_analysis_cache.json").write_text(json.dumps(dict(entries=extra_entries)))
    with (output / "fresh/independent_detail.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "group", "eligible_ids", "required_path"])
        writer.writeheader()
        writer.writerows(extra_rows)
    selected = select_cases(output)
    assert len(selected) == 17
    assert all(case["query"] in {entry["query"] for entry in {**entries, **extra_entries}.values()} for case in selected)
    requests = []
    log = project / "artifacts/timing/current.jsonl"
    log.parent.mkdir(parents=True)

    def fake_request(url, body=None, **kwargs):
        if url.endswith("/health"):
            return dict(status="ok", warmup=dict(state="ready"))
        if url.endswith("/openapi.json"):
            return {"paths": {"/api/v1/search": {"post": {"responses": {"200": {"content": {
                "application/json": {"schema": {"properties": {"results": {"items": {
                    "properties": {"context_evidence": {}}
                }}}}}
            }}}}}}}
        raise AssertionError("Search requests must retain their actual HTTP response metadata")

    def fake_search(url, body, **kwargs):
        requests.append(body)
        assert body["use_rerank"] is False and body["candidate_k"] == 30 and body["top_k"] == 10
        path = body["query"].split()[0]
        if path == "control":
            path = "text_hybrid"
        value = response(path)
        value["analysis"]["original_query"] = body["query"]
        if ((body["query"] == "audio only" and observation == "audio_not_in_top10")
                or path == "audio" and observation in {"all_audio_not_in_top10", "mixed_audio_only_survives"}):
            for track in value["results"]:
                track["explain"]["paths"] = [dict(path="text_hybrid", delta=0.01, dropped=False)]
        if body.get("rejected_ids"):
            ids = [str(index) for index in range(1, 100) if str(index) not in body["rejected_ids"]][:30]
            value["candidate_ids"] = ids
            for track, sid in zip(value["results"], ids):
                track["id"] = sid
        rid = f"synthetic-request-{len(requests)}"
        record = request_timing(path, body["query"], rid)
        if body["query"] == "context two" and observation == "mixed_audio_only_survives":
            value["analysis"].update(audio_english_query="Synthetic sound", modality_weights=dict(context=.4, audio=.3))
            value["results"][0]["explain"]["paths"].append(dict(path="audio", delta=.003, dropped=False))
            record["spans"].extend([
                dict(id=4, parent=0, name="path.audio", run_ms=3., wait_ms=0.),
                dict(id=5, parent=4, name="audio.embed", run_ms=1., wait_ms=0.),
                dict(id=6, parent=4, name="audio.query", run_ms=1., wait_ms=0.),
            ])
        if body["query"] == "audio only" and observation == "audio_query_error":
            record["spans"][-1]["error"] = "SyntheticIndexError"
        with log.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return dict(status=200, body=value, request_id=rid)

    monkeypatch.setattr(api, "request_json", fake_request)
    monkeypatch.setattr(api, "request_search", fake_search)
    monkeypatch.setattr(api, "timing_path", lambda *args: log)
    result = verify(project, output, "http://127.0.0.1:8000", 1)
    failed = observation in {"audio_query_error", "all_audio_not_in_top10"}
    assert result["status"] == ("failed_api_smoke" if failed else "passed_api_smoke")
    assert len(result["rows"]) == 18
    assert len(requests) == 18 and "prior_analysis" not in requests[0] and "prior_analysis" in requests[-1]
    assert result["rows"][-1]["rejected_ids"] == ["1", "2", "3", "4", "5"]
    assert result["setting"] == SETTING
    assert all(row["media_execution"]["verified"] for row in result["rows"]
               if row["status"] == "passed" and row["required_path"] in {"image", "audio"})
    activation = json.loads((output / "activation/activation_report.json").read_text())
    assert activation["api_verified"] is (not failed)
    audio_row = next(row for row in result["rows"] if row["query_id"] == "synthetic-audio")
    if observation == "audio_not_in_top10":
        assert audio_row["required_path_in_top10"] is False and audio_row["media_execution"]["verified"] is True
    if observation == "audio_query_error":
        assert len(result["failures"]) == 1 and audio_row["stage"] == "response_contract"
        diagnostic = next(row for row in result["requests"] if row["query_id"] == "synthetic-audio")
        assert diagnostic["response"]["analysis"]["original_query"] == "audio only"
        assert diagnostic["timing"]["spans"][-1]["error"] == "SyntheticIndexError"
    if observation == "all_audio_not_in_top10":
        assert result["media_coverage"]["audio"]["execution_verified_queries"]
        assert result["media_coverage"]["audio"]["top10_positive_verified_queries"] == []
        assert len(result["failures"]) == 1
        assert result["failures"][0]["query_id"] == "audio:aggregate_hit_confirmation"
    if observation == "mixed_audio_only_survives":
        assert result["media_coverage"]["audio"]["top10_positive_verified_queries"] == ["q203"]
