"""Candidate recall, positive Context votes and Top-10 are different claims."""

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from experiments.namuwiki.context_api_candidate_proof import candidate_probe_payload
from experiments.namuwiki.context_api_timing_checks import read_request_timing
from experiments.namuwiki.verify_context_api_ready import check_response, request_search


def fixture(rank=22, *, rejected=()):
    query = "가상 작품의 이별 장면에서 밴드가 부른 OST"
    ids = [str(i) for i in range(1, 80) if str(i) not in rejected][:30]
    target = ids[rank - 1]
    analysis = dict(original_query=query, context_clues=[dict(target="가상 작품", relation="OST",
                    search_query="가상 작품 이별 장면 OST", confidence=.8)],
                    modality_weights=dict(text=.8, image=0, audio=0))
    tracks = [dict(id=sid, title="Synthetic track " + sid, artist="Synthetic artist", score=.03 - i / 10000,
                   context_evidence=None, explain=dict(paths=[dict(
                       path="context" if sid == target else "text_hybrid", rank=i + 1, delta=.01, dropped=False)]))
              for i, sid in enumerate(ids)]
    primary = dict(analysis=analysis, candidate_ids=ids, results=tracks[:10],
                   explain=dict(analysis_fallback=False, failed_paths=[], rerank_calls=0, rerank_calls_applied=0))
    request = dict(query=query, top_k=10, candidate_k=30, explain=True, use_rerank=False)
    if rejected:
        request.update(rejected_ids=list(rejected), turn=1, previous_candidate_ids=[str(i) for i in range(1, 31)])
    case = dict(query_id="synthetic-context", query=query, required_path="context", no_context=False, expected_targets=[target])
    timing = dict(request_id="primary-request", query=query, attrs=dict(status=200, top_k=10),
                  spans=[dict(id=1, parent=0, name="search", run_ms=15., wait_ms=0.),
                         dict(id=2, parent=1, name="path.context", run_ms=10., wait_ms=1.)])
    full = {**deepcopy(primary), "results": deepcopy(tracks)}
    probe_timing = deepcopy(timing)
    probe_timing.update(request_id="probe-request", attrs=dict(status=200, top_k=30, analysis_mode="prior"))
    probe = dict(request=candidate_probe_payload(request, primary), response=full,
                 http_status=200, request_id="probe-request", timing=probe_timing)
    return case, primary, timing, probe, request


@pytest.mark.parametrize("rank", [1, 10, 11, 16, 22, 24, 30])
def test_positive_target_in_full_candidates_does_not_need_top10_promotion(rank):
    case, primary, timing, probe, request = fixture(rank)
    before = deepcopy((primary, probe, request))
    row = check_response(case, primary, timing_record=timing, candidate_probe=probe, primary_request=request)
    proof = row["context_candidate_proof"]
    assert proof["verified"] and proof["same_analysis"] and proof["same_top10"]
    assert proof["positive_target_candidate_ranks"] == {case["expected_targets"][0]: rank}
    assert row["required_path_in_top10"] is (rank <= 10)
    assert proof["target_in_top10"] == {case["expected_targets"][0]: rank <= 10}
    assert (primary, probe, request) == before


def test_a_context_job_and_a_candidate_id_without_its_positive_vote_cannot_pass():
    case, primary, timing, _, _ = fixture()
    with pytest.raises(ValueError, match="Candidate@30 HTTP proof"):
        check_response(case, primary, timing_record=timing)


@pytest.mark.parametrize("fault", [
    "missing_clue", "empty_search_query", "invalid_confidence", "no_context_job", "job_failed",
    "duplicate_span", "cycle", "orphan", "invalid_duration", "primary_rerank", "wrong_primary_query",
    "wrong_probe_query", "wrong_probe_analysis", "new_gemini_analysis", "no_prior_mode", "same_request_id",
    "wrong_timing_id", "http_failure", "wrong_candidate_limit", "rerank_on", "wrong_state", "no_prior_analysis",
    "candidate_reorder", "candidate_replacement", "partial_probe", "probe_track_reorder", "changed_top10",
    "changed_top10_score", "changed_top10_explain", "unknown_probe_title", "duplicate_probe_track",
    "fallback_probe", "path_error_probe", "probe_rerank", "zero_target_vote", "negative_target_vote",
    "dropped_target_vote", "nan_target_vote", "bool_target_vote", "string_target_vote", "no_target_vote",
    "invalid_context_rank", "duplicate_context_vote", "no_probe_context_job", "failed_probe_context_job",
    "other_positive_song_only", "top10_positive_but_target_has_no_vote",
])
def test_inconsistent_failed_or_noncausal_candidate_proof_is_rejected(fault):
    case, primary, timing, probe, request = fixture()
    full, record = probe["response"], probe["timing"]
    vote = full["results"][21]["explain"]["paths"][0]
    if fault == "missing_clue":
        primary["analysis"]["context_clues"] = []
    elif fault == "empty_search_query":
        primary["analysis"]["context_clues"][0]["search_query"] = " "
    elif fault == "invalid_confidence":
        primary["analysis"]["context_clues"][0]["confidence"] = float("nan")
    elif fault == "no_context_job":
        timing["spans"].pop()
    elif fault == "job_failed":
        timing["spans"][-1]["error"] = "SyntheticIndexError"
    elif fault == "duplicate_span":
        timing["spans"][-1]["id"] = 1
    elif fault == "cycle":
        timing["spans"][-1]["parent"] = 2
    elif fault == "orphan":
        timing["spans"][-1]["parent"] = 999
    elif fault == "invalid_duration":
        timing["spans"][-1]["run_ms"] = -1
    elif fault in {"primary_rerank", "probe_rerank"}:
        (timing if fault == "primary_rerank" else record)["spans"].append(
            dict(id=3, parent=1, name="search.rerank", run_ms=1., wait_ms=0.))
    elif fault == "wrong_primary_query":
        timing["query"] = "wrong query"
    elif fault == "wrong_probe_query":
        record["query"] = "wrong query"
    elif fault == "wrong_probe_analysis":
        full["analysis"]["context_clues"][0]["target"] = "different work"
    elif fault == "new_gemini_analysis":
        record["spans"].append(dict(id=3, parent=0, name="analysis.gemini", run_ms=1., wait_ms=0.))
    elif fault == "no_prior_mode":
        record["attrs"].pop("analysis_mode")
    elif fault == "same_request_id":
        probe["request_id"] = record["request_id"] = timing["request_id"]
    elif fault == "wrong_timing_id":
        record["request_id"] = "different"
    elif fault == "http_failure":
        probe["http_status"] = 500
    elif fault == "wrong_candidate_limit":
        probe["request"]["candidate_k"] = 31
    elif fault == "rerank_on":
        probe["request"]["use_rerank"] = True
    elif fault == "wrong_state":
        probe["request"]["rejected_ids"] = ["1"]
    elif fault == "no_prior_analysis":
        probe["request"].pop("prior_analysis")
    elif fault == "candidate_reorder":
        full["candidate_ids"].reverse()
    elif fault == "candidate_replacement":
        full["candidate_ids"][-1] = "outside"
    elif fault == "partial_probe":
        full["results"].pop()
    elif fault == "probe_track_reorder":
        full["results"][20], full["results"][21] = full["results"][21], full["results"][20]
    elif fault == "changed_top10":
        full["results"][0]["title"] = "Another canonical title"
    elif fault == "changed_top10_score":
        full["results"][0]["score"] += .001
    elif fault == "changed_top10_explain":
        full["results"][0]["explain"]["paths"][0]["delta"] += .001
    elif fault == "unknown_probe_title":
        full["results"][-1]["title"] = "Unknown"
    elif fault == "duplicate_probe_track":
        full["results"][-1]["id"] = "1"
    elif fault == "fallback_probe":
        full["explain"]["analysis_fallback"] = True
    elif fault == "path_error_probe":
        full["explain"]["failed_paths"] = ["context"]
    elif fault in {"zero_target_vote", "negative_target_vote", "nan_target_vote", "bool_target_vote", "string_target_vote"}:
        vote["delta"] = {"zero_target_vote": 0, "negative_target_vote": -1, "nan_target_vote": float("nan"),
                         "bool_target_vote": True, "string_target_vote": ".01"}[fault]
    elif fault == "dropped_target_vote":
        vote["dropped"] = True
    elif fault == "no_target_vote":
        vote["path"] = "text_hybrid"
    elif fault == "invalid_context_rank":
        vote["rank"] = 0
    elif fault == "duplicate_context_vote":
        full["results"][21]["explain"]["paths"].append(deepcopy(vote))
    elif fault == "no_probe_context_job":
        record["spans"].pop()
    elif fault == "failed_probe_context_job":
        record["spans"][-1]["error"] = "SyntheticError"
    elif fault == "other_positive_song_only":
        vote["path"] = "text_hybrid"
        full["results"][25]["explain"]["paths"][0]["path"] = "context"
    else:
        vote["path"] = "text_hybrid"
        for body in (primary, full):
            body["results"][0]["explain"]["paths"][0]["path"] = "context"
    with pytest.raises(ValueError):
        check_response(case, primary, timing_record=timing, candidate_probe=probe, primary_request=request)


def test_rejected_ids_are_kept_identical_in_the_probe_and_never_reintroduced():
    case, primary, timing, probe, request = fixture(rejected=("1", "2", "3", "4", "5"))
    assert probe["request"]["rejected_ids"] == request["rejected_ids"]
    assert probe["request"]["previous_candidate_ids"] == request["previous_candidate_ids"]
    row = check_response(case, primary, rejected=tuple(request["rejected_ids"]), timing_record=timing,
                         candidate_probe=probe, primary_request=request)
    assert row["context_candidate_proof"]["verified"]
    probe["response"]["candidate_ids"][-1] = "1"
    with pytest.raises(ValueError):
        check_response(case, primary, rejected=tuple(request["rejected_ids"]), timing_record=timing,
                       candidate_probe=probe, primary_request=request)


def test_real_http_replay_reuses_analysis_and_retains_all_candidate_contributions(tmp_path):
    """Actual UTF-8 HTTP exchange; synthetic retrieval output, not real-model accuracy."""
    case, primary, timing, probe, request = fixture()
    log, seen = tmp_path / "timing.jsonl", []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            value, record = (primary, timing) if payload["top_k"] == 10 else (probe["response"], probe["timing"])
            with log.open("ab") as stream:
                stream.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
            encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("X-Request-Id", record["request_id"])
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/v1/search"
    try:
        first = request_search(url, request, timeout=2)
        record = read_request_timing(log, 0, first["request_id"], case["query"], wait_seconds=0)
        offset = log.stat().st_size
        payload = candidate_probe_payload(request, first["body"])
        full = request_search(url, payload, timeout=2)
        replay = dict(request=payload, response=full["body"], http_status=full["status"], request_id=full["request_id"],
                      timing=read_request_timing(log, offset, full["request_id"], case["query"], wait_seconds=0))
        row = check_response(case, first["body"], timing_record=record, candidate_probe=replay, primary_request=request)
        assert row["context_candidate_proof"]["positive_target_candidate_ranks"] == {"22": 22}
        assert row["required_path_in_top10"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(seen) == 2 and "prior_analysis" not in seen[0] and seen[1]["prior_analysis"] == primary["analysis"]


def test_existing_fastapi_route_supports_full_candidate_replay_without_analyzer_reentry():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.backend.api.dependencies import get_query_analyzer, get_search_router, get_lyrics_exact_search_service
    from src.backend.api.routes.search import router
    from src.backend.schemas.query import QueryAnalysis, ContextClue
    from src.backend.schemas.search import MatchingTrack
    calls = []

    class Analyzer:
        def analyze(self, query):
            calls.append(query)
            return QueryAnalysis(original_query=query, intent_type="mixed", confidence=.9,
                                 image_english_query="", audio_english_query="", context_clues=[
                ContextClue(target="가상 작품", relation="OST", search_query="가상 작품 OST", confidence=.8)])

    class Searcher:
        async def search(self, analysis, **kwargs):
            ids = [str(i) for i in range(1, 31)]
            kwargs["candidate_ids_out"].extend(ids)
            tracks = [MatchingTrack(id=sid, title="Synthetic track " + sid, artist="Synthetic artist", score=.03)
                      for sid in ids]
            kwargs["candidate_tracks_out"].extend(tracks)
            recorder = kwargs["recorder"]
            for sid in ids:
                recorder.path(sid, "context" if sid == "22" else "text_hybrid", int(sid), .01)
            return tracks[:kwargs["top_k"]]

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_query_analyzer] = Analyzer
    app.dependency_overrides[get_search_router] = Searcher
    app.dependency_overrides[get_lyrics_exact_search_service] = lambda: None
    payload = dict(query="가상 작품 OST", top_k=10, candidate_k=30, explain=True, use_rerank=False)
    with TestClient(app) as client:
        first = client.post("/search", json=payload)
        assert first.status_code == 200, first.text
        full = client.post("/search", json=candidate_probe_payload(payload, first.json()))
        assert full.status_code == 200, full.text
        assert full.json()["analysis"] == first.json()["analysis"]
        assert full.json()["candidate_ids"] == first.json()["candidate_ids"]
        assert full.json()["results"][:10] == first.json()["results"]
        assert len(full.json()["results"]) == 30
        assert full.json()["results"][21]["explain"]["paths"][0]["path"] == "context"
    assert calls == [payload["query"]]
