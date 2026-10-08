"""HTTP media execution needs same-request evidence, not a Top-10 assumption."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from experiments.namuwiki.context_api_timing_checks import (
    read_request_timing, timing_path, verify_media_execution,
)
from experiments.namuwiki.verify_context_api_ready import check_response, request_search


def record(name="audio", query="합성 질의", rid="new-request"):
    return dict(request_id=rid, query=query, attrs=dict(status=200),
                spans=[dict(id=1, parent=0, name="path." + name, run_ms=8.0, wait_ms=1.0),
                       dict(id=2, parent=1, name=name + ".embed", run_ms=2.0, wait_ms=0.0),
                       dict(id=3, parent=1, name=name + ".query", run_ms=3.0, wait_ms=0.0)])


def analyzed(name="audio"):
    return dict(original_query="합성 질의", context_clues=[], modality_weights={name: .4},
                **{name + "_english_query": "Synthetic dedicated description"})


def media_body(name="audio", *, in_top10=False):
    ids = [str(index) for index in range(30)]
    return dict(analysis=analyzed(name), candidate_ids=ids,
                explain=dict(analysis_fallback=False, failed_paths=[], rerank_calls=0, rerank_calls_applied=0),
                results=[dict(id=sid, title="Synthetic track", explain=dict(paths=[
                    dict(path=name if in_top10 else "text_hybrid", delta=.01, dropped=False)
                ])) for sid in ids[:10]])


def media_case(name="audio"):
    return dict(query_id="synthetic", query="합성 질의", required_path=name, no_context=True, expected_targets=[])


@pytest.mark.parametrize("name", ["image", "audio"])
@pytest.mark.parametrize("in_top10", [True, False])
def test_media_may_execute_successfully_without_surviving_top10(name, in_top10):
    result = check_response(media_case(name), media_body(name, in_top10=in_top10), timing_record=record(name))
    assert result["media_execution"]["verified"] is True
    assert result["required_path_in_top10"] is in_top10


@pytest.mark.parametrize("name", ["image", "audio"])
def test_even_positive_top10_contribution_does_not_replace_execution_proof(name):
    with pytest.raises(ValueError, match="Same-request timing"):
        check_response(media_case(name), media_body(name, in_top10=True))


@pytest.mark.parametrize("name", ["image", "audio"])
@pytest.mark.parametrize("fault", [
    "no_job", "no_embed", "no_query", "job_error", "embed_error", "query_error",
    "unrelated_embed", "unrelated_query", "split_jobs", "zero_weight", "negative_weight",
    "nan_weight", "bool_weight", "string_weight", "empty_prompt", "null_prompt", "duplicate_id",
    "bool_id", "negative_duration", "nan_duration", "missing_duration", "different_query",
    "http_failure", "reranking", "cyclic_children", "cyclic_job", "orphan_parent", "bool_parent",
])
def test_missing_failed_or_unrelated_execution_cannot_pass(name, fault):
    data, proof = media_body(name, in_top10=True), record(name)
    if fault in {"no_job", "no_embed", "no_query"}:
        proof["spans"].pop({"no_job": 0, "no_embed": 1, "no_query": 2}[fault])
    elif fault in {"job_error", "embed_error", "query_error"}:
        proof["spans"][{"job_error": 0, "embed_error": 1, "query_error": 2}[fault]]["error"] = "SyntheticError"
    elif fault in {"unrelated_embed", "unrelated_query"}:
        proof["spans"][1 if fault == "unrelated_embed" else 2]["parent"] = 0
    elif fault == "split_jobs":
        proof["spans"].append(dict(id=4, parent=0, name="path." + name, run_ms=3., wait_ms=0.))
        proof["spans"][2]["parent"] = 4
    elif fault in {"zero_weight", "negative_weight", "nan_weight", "bool_weight", "string_weight"}:
        data["analysis"]["modality_weights"][name] = {
            "zero_weight": 0, "negative_weight": -.2, "nan_weight": float("nan"),
            "bool_weight": True, "string_weight": ".4",
        }[fault]
    elif fault in {"empty_prompt", "null_prompt"}:
        data["analysis"][name + "_english_query"] = " " if fault == "empty_prompt" else None
    elif fault in {"duplicate_id", "bool_id"}:
        proof["spans"][1]["id"] = 1 if fault == "duplicate_id" else True
    elif fault in {"negative_duration", "nan_duration", "missing_duration"}:
        if fault == "missing_duration":
            proof["spans"][2].pop("run_ms")
        else:
            proof["spans"][2]["run_ms"] = -1 if fault == "negative_duration" else float("nan")
    elif fault == "different_query":
        proof["query"] = "different source query"
    elif fault == "http_failure":
        proof["attrs"]["status"] = 500
    elif fault == "reranking":
        proof["spans"].append(dict(id=4, parent=0, name="search.rerank", run_ms=1., wait_ms=0.))
    elif fault in {"cyclic_job", "orphan_parent", "bool_parent"}:
        proof["spans"][0]["parent"] = {"cyclic_job": 1, "orphan_parent": 999, "bool_parent": True}[fault]
    else:
        proof["spans"][1]["parent"] = 3
        proof["spans"][2]["parent"] = 2
    with pytest.raises(ValueError):
        check_response(media_case(name), data, timing_record=proof)


@pytest.mark.parametrize("name", ["image", "audio"])
def test_nested_successful_children_still_prove_the_same_job(name):
    proof = record(name)
    proof["spans"].append(dict(id=4, parent=1, name="nested-stage", run_ms=5., wait_ms=0.))
    proof["spans"][1]["parent"] = 4
    proof["spans"][2]["parent"] = 4
    assert verify_media_execution(analyzed(name), proof, name)["verified"] is True


def write_record(path, value):
    with path.open("ab") as stream:
        stream.write(json.dumps(value, ensure_ascii=False).encode("utf-8") + b"\n")


def test_log_reader_uses_appended_offset_and_exact_response_id(tmp_path):
    path = tmp_path / "timing.jsonl"
    write_record(path, record(rid="old-request"))
    offset = path.stat().st_size
    write_record(path, record(rid="concurrent-request"))
    write_record(path, record())
    assert read_request_timing(path, offset, "new-request", "합성 질의", wait_seconds=0) == record()


@pytest.mark.parametrize("fault", ["old_only", "different_id", "wrong_query", "failed_http", "duplicate", "no_header", "partial"])
def test_old_other_incomplete_or_failed_requests_are_not_execution_proof(tmp_path, fault):
    path = tmp_path / "timing.jsonl"
    write_record(path, record())
    offset = path.stat().st_size
    value = record()
    if fault == "different_id":
        value["request_id"] = "other"
    elif fault == "wrong_query":
        value["query"] = "other"
    elif fault == "failed_http":
        value["attrs"]["status"] = 500
    if fault == "partial":
        with path.open("ab") as stream:
            stream.write(json.dumps(value).encode())
    elif fault != "old_only":
        write_record(path, value)
        if fault == "duplicate":
            write_record(path, value)
    with pytest.raises(ValueError):
        read_request_timing(path, offset, "" if fault == "no_header" else "new-request", "합성 질의", wait_seconds=0)


def test_log_rotation_is_not_silently_replaced_with_old_history(tmp_path):
    path = tmp_path / "timing.jsonl"
    write_record(path, record())
    offset = path.stat().st_size
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="rotated"):
        read_request_timing(path, offset, "new-request", "합성 질의", wait_seconds=0)


@pytest.mark.parametrize("value", ["", "../outside.jsonl", "artifacts/log.txt", "artifacts/../../outside.jsonl"])
def test_timing_config_must_stay_in_private_project_artifacts(tmp_path, monkeypatch, value):
    monkeypatch.setenv("SEARCH_TIMING_LOG", value)
    with pytest.raises(ValueError):
        timing_path(tmp_path, {})


def test_registered_timing_location_does_not_substitute_for_current_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("SEARCH_TIMING_LOG", raising=False)
    path = timing_path(tmp_path, {"environment": {"SEARCH_TIMING_LOG": "artifacts/timing/run.jsonl"}})
    assert path == tmp_path / "artifacts/timing/run.jsonl"
    with pytest.raises(ValueError, match="No new timing record"):
        read_request_timing(path, 0, "new-request", "합성 질의", wait_seconds=0)


@pytest.mark.parametrize("header_present", [True, False])
def test_real_urllib_request_and_utf8_response_need_server_request_id(tmp_path, header_present):
    """Use a real loopback HTTP exchange; retrieval stages are synthetic fixtures."""
    path = tmp_path / "timing.jsonl"
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            write_record(path, record(query=body["query"]))
            encoded = json.dumps(media_body(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            if header_present:
                self.send_header("x-request-id", "new-request")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    payload = dict(query="합성 질의", top_k=10, candidate_k=30, use_rerank=False, explain=True)
    try:
        received = request_search(f"http://127.0.0.1:{server.server_port}/api/v1/search", payload, timeout=2)
        assert received["body"]["analysis"]["original_query"] == payload["query"]
        if header_present:
            proof = read_request_timing(path, 0, received["request_id"], payload["query"], wait_seconds=0)
            assert check_response(media_case(), received["body"], timing_record=proof)["media_execution"]["verified"]
        else:
            with pytest.raises(ValueError, match="X-Request-Id"):
                read_request_timing(path, 0, received["request_id"], payload["query"], wait_seconds=0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert seen == [payload]
