"""피드백 후보 보충 측정 도구(⑤). KoE5·운영 저장소 없이 — 사례 파일·집계·채택 기준 점검·봉인 보호만 본다."""
import argparse
import asyncio
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src.feedback.eval_cases import CaseFileStore, build_file_retriever, cases_digest, load_cases
from src.feedback.review import reusable_case
from src.feedback.settings import RetrievalSettings, measurement_display
from src.retrieval import evaluate_feedback_reuse as tool
from src.retrieval import search_router as sr
from tests.feedback_fakes import TEST_DIM, KeywordEmbedder, memory_qdrant, seed_songs
from tests.search.test_search_explain import _analysis, _FakeClient, _FakeEmbedder, _FakeTextService, _track


def _write_cases(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def test_case_file_is_validated_and_expired_rows_are_skipped(tmp_path):
    now = datetime.now(timezone.utc)
    path = _write_cases(tmp_path / "cases.jsonl", [
        {"case_id": "t1", "song_id": "f1", "reusable_text": "  휘파람   노래 ", "group": "target"},
        {"case_id": "old", "song_id": "f2", "reusable_text": "만료된 사례", "group": "target",
         "expires_at": (now - timedelta(days=1)).isoformat()},
        {"case_id": "b1", "song_id": "f3", "reusable_text": "슬픈 발라드", "group": "broad",
         "expires_at": (now + timedelta(days=3)).isoformat()},
    ])
    with pytest.raises(ValueError, match="expires_at"):
        load_cases(path, now)                 # 첫 줄에 만료가 없다 — 실사용 사례 파일이면 거부한다
    cases = load_cases(path, now, require_expiry=False)
    assert [(c.case_id, c.reusable_text, c.group) for c in cases] == [("t1", "휘파람 노래", "target"), ("b1", "슬픈 발라드", "broad")], \
        "내보낸 사례도 원래 만료를 넘겨 쓰지 않는다"
    assert len(cases[0].feedback_id) == 32 and cases_digest(cases) == cases_digest(list(reversed(cases)))
    store = CaseFileStore(cases, now)
    document = store.fetch_cases([cases[1].feedback_id])[cases[1].feedback_id]
    assert reusable_case(document, now).song_id == "f3" and reusable_case(document, now + timedelta(days=4)) is None
    for bad in ({"case_id": "", "song_id": "x", "reusable_text": "a"}, {"case_id": "x", "song_id": "x", "reusable_text": " "},
                {"case_id": "x", "song_id": "x", "reusable_text": "a", "group": "unknown"},
                {"case_id": "x", "song_id": "x", "reusable_text": "가" * 501}):
        with pytest.raises(ValueError):
            load_cases(_write_cases(tmp_path / "bad.jsonl", [bad]), require_expiry=False)
    with pytest.raises(ValueError, match="겹친다"):
        load_cases(_write_cases(tmp_path / "dup.jsonl", [{"case_id": "a", "song_id": "x", "reusable_text": "a"}] * 2),
                   require_expiry=False)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """대역 텍스트 경로 + 사례 파일로 만든 조회 서비스. 실제 라우터를 그대로 쓴다."""
    cases = load_cases(_write_cases(tmp_path / "cases.jsonl", [
        {"case_id": "t1", "song_id": "f1", "reusable_text": "휘파람 노래", "group": "target"},
        {"case_id": "w1", "song_id": "f2", "reusable_text": "피아노 노래", "group": "wrong_reason"},
    ]), require_expiry=False)
    songs = memory_qdrant()
    seed_songs(songs, {"f1": {"title": "정답 곡", "artist": "가수"}, "f2": {"title": "오답 곡", "artist": "가수"}})
    file_cases = build_file_retriever(cases, embedder=KeywordEmbedder(), song_client=songs, dim=TEST_DIM,
                                      settings=lambda: RetrievalSettings(min_similarity=0.86, budget_seconds=2.0))
    hits = [_track(f"s{i}", title=f"곡{i}", score=1.0 - i * 0.01) for i in range(1, 41)]
    router = sr.SearchRouter(search_service=_FakeTextService(hits), image_embedder=_FakeEmbedder(), audio_embedder=_FakeEmbedder(),
                             vector_client=_FakeClient(), max_workers=1, feedback_retriever=file_cases.retriever)
    made = tool.Bench(router, lambda row: _analysis(original_query=row["query"]), file_cases, {"corpus": "test"})
    yield made
    router.shutdown()
    file_cases.close()
    songs.close()


def _args(**kw):
    return argparse.Namespace(**{"runs": 2, "top_k": 15, "candidate_k": 30, "rerank": False, "reject_check": True, **kw})


def test_compare_alternates_off_and_on_and_reports_entries_and_gates(bench, monkeypatch):
    monkeypatch.delenv(tool.SWITCH, raising=False)
    rows = [
        {"query_id": "q1", "query": "휘파람 불던 노래", "relevant_ids": "f1"},      # 후보 밖 정답 — 사례가 들인다
        {"query_id": "q2", "query": "피아노 치던 노래", "relevant_ids": "s3"},      # 잘못된 사례가 오답을 들인다
        {"query_id": "q3", "query": "여름 노래", "relevant_ids": "s1"},            # 걸리는 사례 없음
    ]
    detail = asyncio.run(tool.run_compare(bench, rows, _args()))
    assert len(detail) == 3 * 2 * 2 and __import__("os").environ[tool.SWITCH] == "0", "끝나면 스위치를 꺼 둔다"
    by = {(row["run"], row["mode"], row["query_id"]): row for row in detail}
    assert by[(1, "off", "q1")]["candidate_hit"] == 0 and by[(1, "off", "q1")]["feedback_status"] == ""
    on = by[(1, "on", "q1")]
    assert on["candidate_hit"] == 1 and on["candidate_rank"] == 30 and on["entered_ids"] == "f1" and on["displaced_ids"] == "s30"
    assert on["supplemented_cases"] == "t1:target" and on["rejected_reappeared"] == 0
    assert by[(1, "on", "q2")]["supplemented_cases"] == "w1:wrong_reason" and by[(1, "on", "q3")]["feedback_status"] == "no_case"
    assert all(row["hit10"] == by[(1, "off", row["query_id"])]["hit10"] for row in detail), "리랭커가 없으면 순위 지표는 그대로다"

    summary = tool.summarize(detail, 2)
    means = {row["mode"]: row for row in summary if row["run"] == "mean"}
    assert means["off"]["candidate_recall"] == pytest.approx(2 / 3, abs=1e-4) and means["on"]["candidate_recall"] == 1.0
    assert means["on"]["supplement_applied"] == 2 and means["on"]["entered"] == 2 and means["off"]["entered"] == 0
    change_rows = tool.changes(detail, 2)
    first = {row["query_id"]: row for row in change_rows if row["run"] == 1}
    assert first["q1"]["candidate"] == "recovered" and first["q2"]["candidate"] == "" and first["q3"]["candidate"] == ""
    report = tool.gates(detail, summary, change_rows, 2, reranked=False)
    assert report["hit10_no_loss"] is True and report["rejected_reappear_zero"] is True and report["off_unchanged"] is True
    assert report["gain_exceeds_run_variation"] is None, "리랭킹 없이 잰 값으로는 순위 개선을 판정하지 않는다"
    assert report["comparable"] is True and report["candidate_lost_queries"] == 0


def test_compare_reports_whether_entrants_were_inside_the_rerankers_judged_range(bench, monkeypatch):
    """리뷰 FR-08. 보충 곡은 후보 끝에 선다 — 리랭커가 후보 앞쪽만 평가하면 들어와도 표시될 수 없다. 측정에서 그 경우를 가린다."""
    from tests.search.test_search_explain import _StubReranker

    monkeypatch.delenv(tool.SWITCH, raising=False)
    rows = [{"query_id": "q1", "query": "휘파람 불던 노래", "relevant_ids": "f1"}]
    outcomes = {}
    for name, reranker in (("all", _StubReranker(identity=True)), ("front_20", _StubReranker(identity=True, max_judged=20))):
        bench.router._reranker = reranker
        detail = asyncio.run(tool.run_compare(bench, rows, _args(rerank=True, reject_check=False)))
        summary = tool.summarize(detail, 2)
        on = next(row for row in detail if row["mode"] == "on")
        outcomes[name] = (on["entered_ids"], on["entered_judged_ids"],
                          tool.gates(detail, summary, tool.changes(detail, 2), 2, reranked=True))
    assert outcomes["all"][:2] == ("f1", "f1") and outcomes["all"][2]["entrants_outside_judged_range"] is False
    assert outcomes["front_20"][:2] == ("f1", ""), "들어왔지만 리랭커가 보지 않았다"
    report = outcomes["front_20"][2]
    assert report["entered_total"] == 2 and report["entered_judged_by_reranker"] == 0
    assert report["entrants_outside_judged_range"] is True


def test_the_split_guard_itself_passes_only_dev():
    """인자 파서(choices)에 기대지 않고 보호 함수만으로도 dev만 통과한다."""
    for split in ("dev", " DEV "):
        tool.require_open_split(split, False)
    for split in ("test", "all", "", None, "dev,test", "train"):
        with pytest.raises(SystemExit, match="dev 분할만"):
            tool.require_open_split(split, False)
        tool.require_open_split(split, True)       # 최종 1회 측정을 명시했을 때만


def test_gates_flag_losses_reappearing_rejections_and_unstable_baselines():
    def row(run, mode, qid, **kw):
        base = {"run": run, "mode": mode, "query_id": qid, "relevant_ids": "a", "candidate_hit": 1, "hit1": 0, "hit10": 1,
                "hit15": 1, "mrr10": 0.5, "latency_ms": 100.0, "rerank_runs": "applied", "feedback_status": "",
                "supplemented_ids": "", "supplemented_cases": "", "entered_ids": "", "displaced_ids": "", "rank": 2,
                "candidate_rank": 2, "rejected_reappeared": "", "candidate_ids": "a|b", "result_ids": "b|a"}
        return {**base, **kw}

    detail = [
        row(1, "off", "q1"), row(1, "on", "q1", hit10=0, rank="", latency_ms=1500.0, supplemented_ids="z", result_ids="z|b",
                                 feedback_status="applied", rejected_reappeared=1),
        row(2, "off", "q1", candidate_ids="a|c"), row(2, "on", "q1", rerank_runs="failed"),
    ]
    summary = tool.summarize(detail, 2)
    change_rows = tool.changes(detail, 2)
    assert change_rows[0]["hit10"] == "lost" and change_rows[0]["wrong_supplement_in_top10"] == "z"
    report = tool.gates(detail, summary, change_rows, 2, reranked=True)
    assert report["hit10_no_loss"] is False and report["hit10_lost_queries"] == 1
    assert report["rejected_reappear_zero"] is False and report["off_unchanged"] is False
    assert report["latency_within_budget"] is True and report["latency_median_delta_ms"] == 700.0
    assert report["rerank_failed_rows"] == 1 and report["comparable"] is False, "리랭킹 실패가 섞인 측정은 비교하지 않는다"


def test_live_outputs_carry_no_case_ids(bench, monkeypatch):
    """운영 사례(--cases live)로 잰 산출물에는 사례 ID·묶음 이름을 쓰지 않는다 — 곡 ID와 건수만 남는다."""
    monkeypatch.delenv(tool.SWITCH, raising=False)
    feedback_ids = {case.feedback_id for case in bench.file_cases.cases}
    live = tool.Bench(bench.router, bench.analyze, None, {"corpus": "test"})      # file_cases가 없으면 live로 본다
    detail = asyncio.run(tool.run_compare(live, [{"query_id": "q1", "query": "휘파람 불던 노래", "relevant_ids": "f1"}],
                                          _args(runs=1)))
    on = next(row for row in detail if row["mode"] == "on")
    assert on["entered_ids"] == "f1" and on["supplemented_cases"] == ""
    blob = json.dumps(detail, ensure_ascii=False) + json.dumps(tool.changes(detail, 1), ensure_ascii=False)
    assert not any(feedback_id in blob for feedback_id in feedback_ids)
    assert "t1:" not in blob and "target" not in blob and "wrong_reason" not in blob, "사례 이름·묶음도 남기지 않는다"


@pytest.mark.parametrize("argv", [
    ["compare", "--split", "test", "--synthetic"],
    ["calibrate", "--split", "test"],
])
def test_compare_and_calibrate_open_only_the_dev_split(tmp_path, argv):
    with pytest.raises(SystemExit, match="dev 분할만"):
        tool.main([*argv, "--output-dir", str(tmp_path)])


@pytest.mark.parametrize("split", ["", "all", "TEST", "dev,test"])
def test_split_values_other_than_dev_or_test_are_not_accepted(tmp_path, split, capsys):
    """빈 값·all은 질의 읽기가 dev+test를 모두 돌려준다. 인자 단계에서 받지 않는다(리뷰 FR-03)."""
    for command in (["compare", "--synthetic"], ["calibrate"]):
        with pytest.raises(SystemExit):
            tool.main([*command, "--split", split, "--output-dir", str(tmp_path)])
    capsys.readouterr()


@pytest.mark.parametrize("rest", [
    ["--split", "test"], ["--split=test"], ["--spl", "test"], ["--split", "all"], ["--split", ""],
    ["--split", "dev", "--split", "test"], ["--input", "x.csv"],
], ids=["test", "equals", "abbrev", "all", "empty", "repeated", "missing"])
def test_harness_passes_only_an_explicit_dev_split(monkeypatch, rest):
    """측정기 인자의 --split이 어떤 형식이든 dev가 아니면 측정기를 실행하지 않는다(리뷰 FR-03)."""
    ran = []

    @contextmanager
    def installed(*args, **kwargs):
        ran.append("installed")
        yield None

    monkeypatch.setattr(tool.runpy, "run_module", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(tool, "installed_cases", installed)
    with pytest.raises(SystemExit, match="dev"):
        tool.main(["harness", "--cases", "live", "--module", "src.retrieval.evaluate_search_accuracy", "--", *rest])
    assert ran == [], "거부했으면 사례를 끼우지도, 측정기를 돌리지도 않는다"


def test_harness_runs_the_module_for_dev_and_for_an_explicit_sealed_final(monkeypatch):
    ran = []
    monkeypatch.setattr(tool.runpy, "run_module", lambda name, **k: ran.append(
        (name, list(tool.sys.argv[1:]), tool.os.environ.get(tool.SWITCH))))
    monkeypatch.setattr(tool.sys, "argv", ["x", "원래 인자"])
    monkeypatch.delenv(tool.SWITCH, raising=False)
    module = "src.retrieval.evaluate_search_accuracy"
    tool.main(["harness", "--cases", "live", "--module", module, "--", "--split=dev", "--input", "a.csv"])
    tool.main(["harness", "--cases", "live", "--module", module, "--sealed-final", "--", "--split", "test"])
    assert ran == [(module, ["--split=dev", "--input", "a.csv"], "1"), (module, ["--split", "test"], "1")], \
        "측정기가 도는 동안에는 스위치가 켜져 있다"
    assert tool.sys.argv == ["x", "원래 인자"] and tool.SWITCH not in tool.os.environ, "끝나면 argv·스위치를 되돌린다"
    assert RetrievalSettings.from_env().display_k is None


@pytest.mark.parametrize("extra, rest, expected", [
    ([], ["--split", "dev"], "10"),                               # 측정기 기본 표시 수
    ([], ["--split", "dev", "--top-k", "15"], "15"),              # 측정기 인자의 --top-k
    ([], ["--split", "dev", "--top-k=5"], "5"),
    (["--display-k", "7"], ["--split", "dev", "--top-k", "15"], "7"),   # 명시값이 우선
])
def test_harness_tells_the_router_how_many_results_the_measurement_displays(monkeypatch, extra, rest, expected):
    """측정기는 라우터에 후보 전체를 요청한다(top_k = candidate_k). 표시 수를 알려 주지 않으면 보충이 0건이 된다."""
    seen = []
    monkeypatch.setattr(tool.runpy, "run_module", lambda name, **k: seen.append(
        (RetrievalSettings.from_env().display_k, tool.os.environ.get("FEEDBACK_DISPLAY_K"))))
    monkeypatch.setattr(tool.sys, "argv", ["x"])
    monkeypatch.delenv(tool.SWITCH, raising=False)
    monkeypatch.delenv("FEEDBACK_DISPLAY_K", raising=False)
    with measurement_display(99):                                 # 실행 전에 있던 값은 끝나면 돌아온다
        tool.main(["harness", "--cases", "live", "--module", "src.retrieval.evaluate_search_accuracy", *extra, "--", *rest])
        assert RetrievalSettings.from_env().display_k == 99
    assert seen == [(int(expected), None)], "표시 수는 이 프로세스 안의 값으로만 넘긴다 — 환경 변수에 쓰지 않는다"
    assert RetrievalSettings.from_env().display_k is None


@pytest.mark.parametrize("extra, rest", [
    (["--display-k", "-4"], ["--split", "dev"]),
    (["--display-k", "0"], ["--split", "dev"]),
    (["--display-k", "101"], ["--split", "dev"]),
    ([], ["--split", "dev", "--top-k", "0"]),
    ([], ["--split", "dev", "--top-k=-1"]),
])
def test_harness_refuses_an_unusable_display_count_instead_of_measuring_nothing(monkeypatch, extra, rest):
    """쓸 수 없는 표시 수를 그대로 넘기면 보충이 0건인 채 조용히 돌아 on이 off와 같아진다. 측정기를 돌리기 전에 멈춘다."""
    ran = []
    monkeypatch.setattr(tool.runpy, "run_module", lambda name, **k: ran.append(name))
    monkeypatch.setattr(tool.sys, "argv", ["x"])
    monkeypatch.delenv(tool.SWITCH, raising=False)
    with pytest.raises(SystemExit) as stopped:
        tool.main(["harness", "--cases", "live", "--module", "src.retrieval.evaluate_search_accuracy", *extra, "--", *rest])
    assert "1~100" in str(stopped.value) and ran == []
    assert tool.SWITCH not in tool.os.environ and RetrievalSettings.from_env().display_k is None
    with pytest.raises(SystemExit, match="측정기"):
        tool.main(["harness", "--cases", "live", "--module", "os", "--", "--split", "dev"])


# ---------------------------------------------------------------------------
# 실제 코퍼스 경로(real_bench) — 팩토리 교체·복원과 정리는 실물, 코퍼스·모델·분석 캐시만 대역 (리뷰 FR-09)
# ---------------------------------------------------------------------------

class _WideEmbedder(KeywordEmbedder):
    """KoE5 차원(1024)으로 넓힌 가짜 임베딩. 사례 색인이 실제 차원으로 만들어지므로 맞춰야 한다."""

    def embed_passages(self, passages, **kwargs):
        from src.vector_db.settings import TEXT_DENSE_DIM

        small = super().embed_passages(passages, **kwargs)
        wide = np.zeros((len(passages), TEXT_DENSE_DIM), dtype=np.float32)
        wide[:, : small.shape[1]] = small
        return wide


@pytest.fixture
def real_stack(monkeypatch, tmp_path):
    """`dependencies`의 무거운 팩토리만 가벼운 것으로 바꾼다. `get_feedback_retriever`와 `close_vector_client`는 실물이다."""
    from src.backend.api import dependencies
    from src.retrieval import analysis_cache

    client = memory_qdrant()
    seed_songs(client, {"f1": {"title": "정답 곡", "artist": "가수"}})
    hits = [_track(f"s{i}", title=f"곡{i}", score=1.0 - i * 0.01) for i in range(1, 41)]
    monkeypatch.setattr(dependencies, "get_text_embedder", dependencies.singleton(_WideEmbedder))
    monkeypatch.setattr(dependencies, "get_vector_client", dependencies.singleton(lambda: client))
    monkeypatch.setattr(dependencies, "get_search_router", dependencies.singleton(lambda: sr.SearchRouter(
        search_service=_FakeTextService(hits), image_embedder=_FakeEmbedder(), audio_embedder=_FakeEmbedder(),
        vector_client=_FakeClient(), max_workers=1, feedback_retriever=dependencies.get_feedback_retriever())))
    monkeypatch.setattr(analysis_cache, "load_cache", lambda path: SimpleNamespace(
        meta={"stub": True}, get=lambda query_id, query: _analysis(original_query=query)))
    monkeypatch.delenv(tool.SWITCH, raising=False)
    cases = _write_cases(tmp_path / "cases.jsonl", [
        {"case_id": "t1", "song_id": "f1", "reusable_text": "휘파람 노래", "group": "target"}])
    queries = tmp_path / "queries.csv"
    queries.write_text("query_id,split,query_type,query,relevant_ids\nd1,dev,search,휘파람 불던 노래,f1\n", encoding="utf-8")
    argv = ["compare", "--cases", str(cases), "--synthetic-cases", "--input", str(queries), "--split", "dev",
            "--analysis-cache", "stub.json", "--runs", "1", "--output-dir", str(tmp_path / "out")]
    yield SimpleNamespace(dependencies=dependencies, original=dependencies.get_feedback_retriever, argv=argv,
                          out=tmp_path / "out", cases=cases)
    dependencies.get_search_router.cache_clear()


OUTPUTS = ["feedback_reuse_changes.csv", "feedback_reuse_detail.csv", "feedback_reuse_runinfo.json",
           "feedback_reuse_summary.csv"]


def test_harness_runs_the_real_evaluator_and_the_supplement_reaches_its_reranker(monkeypatch, tmp_path):
    """harness → 기존 측정기(`evaluate_search_accuracy`)를 실제로 끝까지 돌린다.

    측정기 모듈·`get_search_router` 팩토리·`SearchRouter`·`SearchService`·사례 끼우기/되돌리기는 실물이다. 무거운 것만
    대역이다(메모리 Qdrant의 합성 곡 60개 · 주제어 임베딩 · 규칙 분석 · 보충 곡을 올리는 리랭커). 측정기는 라우터에
    후보 30곡을 통째로 요청하므로(top_k = candidate_k), 표시 수를 알려 주지 않으면 보충이 0건이라 on이 off와 같아진다.
    """
    import csv

    from qdrant_client import models

    from src.backend.api import dependencies
    from src.common import mongodb
    from src.retrieval.explain import RERANK_APPLIED, RerankRun
    from src.retrieval.query_analyzer import rule_fallback
    from src.vector_db.qdrant_backend import DENSE_VECTOR, collection_name, point_id
    from src.vector_db.settings import TEXT_HYBRID_INDEX_NAME

    class Promote:
        """리랭커 대역 — 정답 곡이 후보에 있으면 1위로 올린다(모델이 그렇게 판단했다고 가정)."""
        enabled = True
        uses_clarify_answers = False
        config = SimpleNamespace(rerank_weight=1.0, model_name="stub")

        def load(self):
            return self

        def rerank_run(self, query, tracks, top_k):
            ordered = sorted(tracks, key=lambda t: t.id != "target")
            return RerankRun([t.model_copy(update={"rerank_score": 0.9}) for t in ordered][:top_k], RERANK_APPLIED,
                             [t.id for t in tracks])

        def rerank(self, query, tracks, top_k):
            return self.rerank_run(query, tracks, top_k).tracks

    embedder = _WideEmbedder()
    client = memory_qdrant()
    from src.vector_db.settings import TEXT_DENSE_DIM
    client.ensure_collection(TEXT_HYBRID_INDEX_NAME, dim=TEXT_DENSE_DIM)
    songs = [(f"p{i}", f"새벽 노래 {i}") for i in range(1, 60)] + [("target", "여름 노래")]     # 정답 곡 요약에는 단서가 없다
    client.client.upsert(collection_name(TEXT_HYBRID_INDEX_NAME), points=[
        models.PointStruct(id=point_id(sid), vector={DENSE_VECTOR: vector.tolist()},
                           payload={"song_id": sid, "title": text, "artist": "합성가수"})
        for (sid, text), vector in zip(songs, embedder.embed_passages([text for _, text in songs]))])
    for name, factory in {
        "get_text_embedder": lambda: embedder, "get_vector_client": lambda: client,
        "get_bm25_encoder": lambda: SimpleNamespace(_is_fitted=False), "get_image_embedder": _FakeEmbedder,
        "get_audio_embedder": _FakeEmbedder, "get_lyrics_exact_search_service": lambda: None,
        "get_query_analyzer": lambda: SimpleNamespace(analyze=rule_fallback), "get_reranker": Promote,
    }.items():
        monkeypatch.setattr(dependencies, name, dependencies.singleton(factory))
    dependencies.get_search_router.cache_clear()
    dependencies.get_search_service.cache_clear()
    monkeypatch.setattr(mongodb, "get_collection", lambda name: (_ for _ in ()).throw(RuntimeError("시험에서는 Mongo를 열지 않는다")))
    monkeypatch.setattr(tool.sys, "argv", ["x"])
    monkeypatch.delenv(tool.SWITCH, raising=False)
    original = dependencies.get_feedback_retriever

    queries = tmp_path / "queries.csv"
    queries.write_text("query_id,split,query_type,query,relevant_ids\nd1,dev,search,새벽에 휘파람 불던 노래,target\n", encoding="utf-8")
    cases = _write_cases(tmp_path / "cases.jsonl", [
        {"case_id": "c1", "song_id": "target", "reusable_text": "새벽에 휘파람 부는 노래", "group": "target"}])

    def measure(out, *harness_args):
        rest = ["--input", str(queries), "--split", "dev", "--output-dir", str(out), "--explain"]
        tool.main(["harness", "--cases", str(cases), "--synthetic-cases", "--module", "src.retrieval.evaluate_search_accuracy",
                   *harness_args, "--", *rest])
        with (out / "search_eval_dev_detail.csv").open(encoding="utf-8-sig") as file:
            return next(csv.DictReader(file))

    try:
        on = measure(tmp_path / "on")
        assert (on["candidate_rank@30"], on["rerank_rank"], on["rerank_hit@1"]) == ("30", "1", "1.0"), \
            "보충 곡이 후보 끝(30위)으로 들어오고 측정기의 리랭커가 그것을 평가한다"
        # 표시 수를 후보 수와 같게 알려 주면 숨은 자리가 없어 보충하지 않는다 — 꺼진 것과 같은 결과다.
        none = measure(tmp_path / "no_room", "--display-k", "30")
        assert (none["candidate_rank@30"], none["candidate_recall@30"], none["rerank_rank"]) == ("", "0.0", "")
    finally:
        dependencies.get_search_router.cache_clear()
        dependencies.get_search_service.cache_clear()
        client.close()
    assert dependencies.get_feedback_retriever is original
    assert tool.SWITCH not in tool.os.environ and RetrievalSettings.from_env().display_k is None


def test_compare_with_a_case_file_on_the_real_stack_finishes_and_restores_the_factory(real_stack, capsys):
    """`compare --cases <파일>`: 사례를 끼우고 → 재고 → 닫고 → 산출물이 남는다. 끝나면 조회 서비스 팩토리는 원래 것이다."""
    assert tool.main(real_stack.argv) == 0
    assert "닫는 중 오류" not in capsys.readouterr().err, "정리도 오류 없이 끝나야 한다(경고로 삼켜진 실패가 아니다)"
    assert sorted(path.name for path in real_stack.out.iterdir()) == OUTPUTS
    detail = (real_stack.out / "feedback_reuse_detail.csv").read_text(encoding="utf-8")
    assert ",on,d1,f1," in detail and "t1:target" in detail, "끼운 사례가 실제로 후보를 보충했다"
    runinfo = json.loads((real_stack.out / "feedback_reuse_runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["cases"]["count"] == 1 and runinfo["bench"]["corpus"] == "live"
    assert real_stack.dependencies.get_feedback_retriever is real_stack.original, "바꿔 끼운 팩토리를 되돌린다"
    assert real_stack.dependencies.get_search_router.peek() is None, "파일 사례를 쥔 라우터가 캐시에 남지 않는다"


def test_the_swapped_factory_keeps_the_singleton_shape_while_installed(real_stack):
    """`close_vector_client()`는 벡터 클라이언트를 쥔 팩토리마다 cache_clear를 부른다. 바꿔 끼운 것도 그 모양이어야 한다."""
    dependencies = real_stack.dependencies
    with tool.installed_cases(str(real_stack.cases), True) as file_cases:
        assert dependencies.get_feedback_retriever is not real_stack.original
        assert dependencies.get_feedback_retriever() is file_cases.retriever
        assert dependencies.get_search_router()._feedback is file_cases.retriever
        for factory in dependencies._vector_client_dependents():
            assert callable(getattr(factory, "cache_clear", None)) and callable(getattr(factory, "peek", None))
        dependencies.close_vector_client()            # 정리 코드 실물 — 여기서 AttributeError가 났다
    assert dependencies.get_feedback_retriever is real_stack.original
    with tool.installed_cases("live", False) as nothing:
        assert nothing is None and dependencies.get_feedback_retriever is real_stack.original


def test_a_router_built_before_the_swap_is_not_reused_while_installed_or_afterwards(real_stack):
    """같은 프로세스에 이미 만들어진 라우터가 있어도, 끼운 동안에는 파일 사례를 쥔 새 라우터가 나가고 끝나면 그것도 버린다."""
    dependencies = real_stack.dependencies
    try:
        before = dependencies.get_search_router()         # 끼우기 전의 라우터 — 운영 조회 서비스를 쥐고 있다
        assert before._feedback is real_stack.original()
        with tool.installed_cases(str(real_stack.cases), True) as file_cases:
            inside = dependencies.get_search_router()
            assert inside is not before and inside._feedback is file_cases.retriever
        after = dependencies.get_search_router()
        assert after is not inside and after._feedback is real_stack.original()
    finally:
        dependencies.get_feedback_retriever.cache_clear()
        dependencies.get_feedback_case_index.cache_clear()


@pytest.mark.parametrize("failing", ["close", "measure"])
def test_results_survive_a_cleanup_failure_and_the_factory_is_restored_on_any_exit(real_stack, monkeypatch, capsys, failing):
    dependencies = real_stack.dependencies
    if failing == "close":
        # 정리 중 오류는 경고로만 남는다. 다 잰 결과를 버리지 않는다.
        monkeypatch.setattr(dependencies, "close_vector_client", lambda: (_ for _ in ()).throw(RuntimeError("정리 실패")))
        assert tool.main(real_stack.argv) == 0
        assert sorted(path.name for path in real_stack.out.iterdir()) == OUTPUTS
        assert "닫는 중 오류" in capsys.readouterr().err
    else:
        # 측정이 중간에 죽어도 끝난 질의의 detail은 남고, 팩토리는 되돌아온다.
        calls = []
        original = tool._search_once

        async def flaky(*args, **kwargs):
            calls.append(True)
            if len(calls) > 2:                # 첫 질의(off·on)는 끝났다
                raise RuntimeError("측정 중 오류")
            return await original(*args, **kwargs)

        monkeypatch.setattr(tool, "_search_once", flaky)
        queries = real_stack.out.parent / "queries.csv"
        queries.write_text(queries.read_text(encoding="utf-8") + "d2,dev,search,휘파람 부는 노래,f1\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="측정 중 오류"):
            tool.main(real_stack.argv)
        assert ",on,d1,f1," in (real_stack.out / "feedback_reuse_detail.csv").read_text(encoding="utf-8")
    assert dependencies.get_feedback_retriever is real_stack.original
    assert tool.os.environ.get(tool.SWITCH) in (None, "0")


def test_harness_with_a_case_file_installs_for_the_run_and_restores_afterwards(real_stack, monkeypatch):
    dependencies = real_stack.dependencies
    seen = []

    def measure(name, **kwargs):
        retriever = dependencies.get_search_router()._feedback        # 측정기가 하는 일: 라우터를 받아 쓴다
        seen.append((retriever is not None and retriever.enabled, dependencies.get_feedback_retriever is not real_stack.original))

    monkeypatch.setattr(tool.runpy, "run_module", measure)
    monkeypatch.setattr(tool.sys, "argv", ["x"])
    tool.main(["harness", "--cases", str(real_stack.cases), "--synthetic-cases",
               "--module", "src.retrieval.evaluate_search_accuracy", "--", "--split", "dev"])
    assert seen == [(True, True)], "측정기가 도는 동안에는 파일 사례가 끼워져 있고 스위치가 켜져 있다"
    assert dependencies.get_feedback_retriever is real_stack.original and dependencies.get_search_router.peek() is None
    assert tool.SWITCH not in tool.os.environ


def test_uncached_real_runs_and_expiryless_case_files_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="analysis-cache"):
        tool.main(["compare", "--cases", "live", "--output-dir", str(tmp_path)])
    exported = _write_cases(tmp_path / "exported.jsonl", [{"case_id": "r1", "song_id": "x", "reusable_text": "만료를 빠뜨린 사례"}])
    args = argparse.Namespace(cases=str(exported), synthetic=False, synthetic_cases=False)
    assert tool.cases_are_synthetic(args) is False
    assert tool.cases_are_synthetic(argparse.Namespace(cases=str(exported), synthetic_cases=True)) is True
    assert tool.cases_are_synthetic(argparse.Namespace(cases=str(tool.SYNTHETIC_DIR / "cases.jsonl"))) is True
    with pytest.raises(ValueError, match="expires_at"):
        tool.main(["calibrate", "--cases", str(exported), "--output-dir", str(tmp_path)])


def test_the_committed_synthetic_set_is_consistent():
    """합성 자료의 사례·질의가 가리키는 곡이 합성 곡 목록에 있다(생성기를 고치고 파일을 안 고친 경우를 잡는다)."""
    from src.retrieval.evaluate_search_accuracy import parse_relevant_ids, read_queries

    songs = {json.loads(line)["song_id"] for line in (tool.SYNTHETIC_DIR / "songs.jsonl").read_text(encoding="utf-8").splitlines()}
    cases = load_cases(tool.SYNTHETIC_DIR / "cases.jsonl", require_expiry=False)
    queries = read_queries(tool.SYNTHETIC_DIR / "queries.csv", "dev")
    assert len(songs) == 402 and len(cases) == 14 and len(queries) == 10
    assert {case.song_id for case in cases} <= songs
    assert all(parse_relevant_ids(row["relevant_ids"]) <= songs for row in queries)
    assert {case.group for case in cases} == set(tool.CASE_GROUPS), "대조군이 모두 들어 있다"
