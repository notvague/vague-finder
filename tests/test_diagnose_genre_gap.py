"""등록 장르·체감 장르 간격 진단 도구와 비교 질의 세트의 계약. 모델·DB 없음.

질의 세트는 dev 전용 실험 자료다. 여기서 지키는 것은 자료의 모양(쌍이 장르 낱말 하나만 다르다 · 제목·가수
이름이 없다 · 넓은 질의는 단일 정답 평가에 들어가지 않는다)과 도구의 집계 규칙이다 — 검색 품질이 아니다.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from src.retrieval import diagnose_genre_gap as tool
from src.retrieval.evaluate_search_accuracy import read_queries
from src.retrieval.explain import ExplainRecorder

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / tool.DEFAULT_SOURCES
QUERIES = ROOT / tool.DEFAULT_QUERIES
EVAL_SETS = [ROOT / f"experiments/reranking/eval_queries_{name}.csv" for name in ("v06", "v09", "v10")]


@pytest.fixture(scope="module")
def sources() -> dict:
    return json.loads(SOURCES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(sources) -> list:
    return tool.build_query_rows(sources)


# ---------------------------------------------------------------------------
# 질의 세트
# ---------------------------------------------------------------------------

def test_committed_csv_is_the_export_of_the_sources(rows) -> None:
    with QUERIES.open(encoding="utf-8", newline="") as f:
        committed = list(csv.DictReader(f))
    assert committed == rows, "query_sources.json을 고쳤으면 `diagnose_genre_gap export`로 CSV를 다시 만든다"


def test_set_shape(sources, rows) -> None:
    songs = sources["songs"]
    assert 15 <= len(songs) <= 20
    variants = {v: sum(1 for r in rows if r["variant"] == v) for v in {r["variant"] for r in rows}}
    assert variants[tool.VARIANT_REGISTERED] == variants[tool.VARIANT_OMITTED] == variants[tool.VARIANT_IDENTITY] == len(songs)
    assert variants[tool.VARIANT_PERCEIVED] == sum(1 for s in songs if s["group"] == "gap")
    assert len({s["song_id"] for s in songs}) == len(songs), "곡이 겹치지 않는다"
    assert {s["group"] for s in songs} == {"gap", "control"}
    for song in songs:
        # 간격 사례는 체감 장르가 등록 표현과 달라야 하고, 대조군은 체감 장르 변형이 없다.
        assert bool(song["perceived_term"]) == (song["group"] == "gap"), song["pair_id"]
        assert song["perceived_term"] != song["registered_term"]
    assert all(r["split"] == "dev" for r in rows), "dev 전용이다"
    assert sources["review_status"] == "author_draft" and all(r["review_status"] == "author_draft" for r in rows)


def test_registered_perceived_and_omitted_differ_only_in_the_genre_word(sources, rows) -> None:
    by_id = {r["query_id"]: r for r in rows}
    for song in sources["songs"]:
        left, right = song["template"].split("{genre} ")
        registered, omitted = by_id[f"{song['pair_id']}r"], by_id[f"{song['pair_id']}o"]
        assert registered["query"] == f"{left}{song['registered_term']} {right}"
        assert omitted["query"] == left + right and omitted["variant"] == tool.VARIANT_OMITTED
        assert omitted["stated_genre"] == "" and omitted["relevant_ids"] == song["song_id"]
        for word in (song["registered_term"], song["perceived_term"]):
            assert not word or word not in omitted["query"], (song["pair_id"], word)
        if song["perceived_term"]:
            perceived = by_id[f"{song['pair_id']}p"]
            assert perceived["query"] == f"{left}{song['perceived_term']} {right}"
            assert registered["relevant_ids"] == perceived["relevant_ids"] == song["song_id"]


def _name_tokens(text: str) -> list:
    """제목·가수 이름에서 질의에 나오면 안 되는 덩어리(괄호 안팎, 2자 이상)."""
    parts = re.split(r"[()\[\],]|feat\.?|Feat\.?", text)
    return [p.strip() for p in parts if len(p.strip()) >= 2]


def test_queries_do_not_name_the_title_or_artist(sources, rows) -> None:
    for song in sources["songs"]:
        tokens = _name_tokens(song["title"]) + _name_tokens(song["artist"])
        for row in (r for r in rows if r["pair_id"] == song["pair_id"]):
            compact = "".join(row["query"].split()).lower()
            for token in tokens:
                assert "".join(token.split()).lower() not in compact, (row["query_id"], token)


def test_broad_queries_stay_out_of_single_answer_evaluation(rows) -> None:
    broad = [r for r in rows if r["target_scope"] == tool.SCOPE_BROAD]
    assert len(broad) >= 5 and all(r["query_type"] == "broad" and not r["relevant_ids"] for r in broad)
    evaluated = read_queries(QUERIES, "dev")          # 평가기가 실제로 읽는 행
    assert {r["query_id"] for r in evaluated} == {r["query_id"] for r in rows if r["target_scope"] == tool.SCOPE_SPECIFIC}
    assert all(json.loads(r["conditions"]) is not None for r in broad)


def test_ids_do_not_collide_with_existing_eval_sets(rows) -> None:
    existing = set()
    for path in EVAL_SETS:
        with path.open(encoding="utf-8-sig", newline="") as f:
            existing.update(r["query_id"].strip() for r in csv.DictReader(f))
    assert not existing & {r["query_id"] for r in rows}


@pytest.mark.parametrize("template", ["장르 자리가 없는 문장", "{genre} 노래와 {genre} 노래", "붙여 쓴 {genre}노래"])
def test_templates_must_hold_exactly_one_genre_slot(sources, template: str) -> None:
    broken = json.loads(json.dumps(sources))
    broken["songs"][0]["template"] = template
    with pytest.raises(ValueError):
        tool.build_query_rows(broken)


# ---------------------------------------------------------------------------
# analyze — 있는 분석은 그대로 두고 없는 것만 더한다
# ---------------------------------------------------------------------------

def _query_file(path: Path, items: dict) -> Path:
    tool.write_csv(path, [
        {**{c: "" for c in tool.QUERY_COLUMNS}, "split": "dev", "query_type": "search", "query_id": qid,
         "query": query, "relevant_ids": "1"} for qid, query in items.items()
    ], tool.QUERY_COLUMNS)
    return path


def test_analyze_adds_missing_queries_and_never_rewrites_existing_ones(tmp_path, monkeypatch) -> None:
    from tests.test_search_explain import _analysis

    calls: list = []

    class Analyzer:
        attempts: list = []

        def analyze(self, query: str):
            calls.append(query)
            self.attempts.append({"genre": f"모델-{len(calls)}"})
            return _analysis(original_query=query, genre=f"모델-{len(calls)}", confidence=0.9)

    monkeypatch.setattr(tool, "_capturing_analyzer", lambda: Analyzer())
    monkeypatch.setenv("SEARCH_REFERENCE_YEAR", "2026")
    cache, raw = tmp_path / "cache.json", tmp_path / "raw.jsonl"
    args = ["--output", str(cache), "--model-raw", str(raw)]

    queries = _query_file(tmp_path / "q.csv", {"a": "첫 질의", "b": "둘째 질의"})
    assert tool.main(["analyze", "--input", str(queries), *args]) == 0
    first = json.loads(cache.read_text(encoding="utf-8"))["entries"]
    assert calls == ["첫 질의", "둘째 질의"]

    _query_file(queries, {"a": "첫 질의", "b": "둘째 질의", "c": "나중에 더한 질의"})
    assert tool.main(["analyze", "--input", str(queries), *args]) == 0
    second = json.loads(cache.read_text(encoding="utf-8"))["entries"]
    assert calls == ["첫 질의", "둘째 질의", "나중에 더한 질의"], "있는 질의는 다시 분석하지 않는다"
    assert {k: second[k] for k in first} == first and set(second) == {"a", "b", "c"}
    assert [json.loads(line)["query_id"] for line in raw.read_text(encoding="utf-8").splitlines()] == ["a", "b", "c"]
    assert set(tool.load_model_raw(raw)) == {"a", "b", "c"}

    assert tool.main(["analyze", "--input", str(queries), *args]) == 0 and len(calls) == 3, "더할 것이 없으면 부르지 않는다"

    _query_file(queries, {"a": "원문을 고친 질의", "b": "둘째 질의", "c": "나중에 더한 질의"})
    assert tool.main(["analyze", "--input", str(queries), *args]) == 2, "원문이 바뀐 질의를 옛 분석과 섞지 않는다"
    assert json.loads(cache.read_text(encoding="utf-8"))["entries"] == second

    monkeypatch.setattr("src.retrieval.analysis_cache.analyzer_fingerprint", lambda provider=None: {"prompt_sha": "다른 프롬프트"})
    _query_file(queries, {"a": "첫 질의", "b": "둘째 질의", "c": "나중에 더한 질의", "d": "조건이 바뀐 뒤의 질의"})
    assert tool.main(["analyze", "--input", str(queries), *args]) == 2, "분석 조건이 달라졌으면 섞지 않는다"
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# 읽기 — dev만, 기존 평가 CSV도
# ---------------------------------------------------------------------------

def test_reader_is_dev_only_and_accepts_plain_eval_csvs(tmp_path) -> None:
    import inspect

    assert list(inspect.signature(tool.read_rows).parameters) == ["path"], "분할을 고르는 인자가 없다"
    plain = tmp_path / "plain.csv"
    plain.write_text(
        "query_id,split,query_type,query,relevant_ids\n"
        "a1,dev,search,질의 하나,111|222\n"
        "a2,test,search,봉인 질의,333\n"
        "a3,dev,clarify,재질문용,444\n", encoding="utf-8")
    rows = tool.read_rows(plain)
    assert [r["query_id"] for r in rows] == ["a1"], "test 분할과 search가 아닌 행은 읽지 않는다"
    sealed = tmp_path / "sealed.csv"
    sealed.write_text("query_id,split,query_type,query,relevant_ids\na2,test,search,봉인 질의,333\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tool.read_rows(sealed)          # test 행만 있으면 읽을 것이 없다
    assert (rows[0]["variant"], rows[0]["group"], rows[0]["pair_id"]) == (tool.VARIANT_ALL, "plain", "a1")
    renamed = tmp_path / "eval_queries_v06.csv"
    renamed.write_text(plain.read_text(encoding="utf-8"), encoding="utf-8")
    assert tool.read_rows(renamed)[0]["group"] == "v06", "기존 평가 CSV는 세트 이름으로 묶는다"
    assert tool.relevant_ids(rows[0]) == ["111", "222"]


def test_scales_must_start_at_the_current_value_and_stay_in_range() -> None:
    assert tool.parse_scales("1,0.5,0.25,0") == [1.0, 0.5, 0.25, 0.0]
    for bad in ("0.5,1", "1,2", "1,-0.5", ""):
        with pytest.raises(ValueError):
            tool.parse_scales(bad)


# ---------------------------------------------------------------------------
# 모델 원본 — 없으면 없다고 적는다
# ---------------------------------------------------------------------------

def test_model_raw_is_reported_only_when_it_backs_the_final_analysis(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in [
        {"query_id": "ok", "fallback": False, "model_attempts": [
            {"genre": "댄스"}, {"genre": "발라드", "artist_type": {"values": ["그룹"]}, "vocal_gender": "여성"}]},
        {"query_id": "fallback", "fallback": True, "model_attempts": [{"genre": "발라드"}]},
        {"query_id": "unparsed", "fallback": False, "model_attempts": [{"_unparsed": "JSONDecodeError"}]},
    ]) + "\n", encoding="utf-8")
    raw = tool.load_model_raw(path)
    assert set(raw) == {"ok"}, "규칙 폴백과 못 읽은 응답은 최종 분석의 바탕이 아니다"
    assert tool._raw_fields(raw["ok"]) == {"raw_genre": "발라드", "raw_artist_type": "그룹", "raw_vocal_gender": "여성"}
    assert set(tool._raw_fields(None).values()) == {tool.RAW_UNAVAILABLE}
    assert tool.load_model_raw(tmp_path / "없는파일.jsonl") == {}


# ---------------------------------------------------------------------------
# 단계 판정 · 집계
# ---------------------------------------------------------------------------

def test_stage_and_genre_match_follow_the_router_rules() -> None:
    rec = ExplainRecorder("q")
    for song_id, rank in (("in", 1), ("cut", 40), ("late", 2)):
        rec.path(song_id, "text_hybrid", rank, 1.0 / (60 + rank))
    rec.mark_paths_dropped(["cut"])
    rec.set_fused("in", 0.03)
    rec.set_fused("late", 0.02)
    record = rec.record
    assert tool.stage_of("in", record, ["in"]) == tool.IN_POOL
    assert tool.stage_of("none", record, ["in"]) == tool.NO_PATH
    assert tool.stage_of("cut", record, ["in"]) == tool.FUSION_CUT
    assert tool.stage_of("late", record, ["in"]) == tool.FINAL_CUT

    assert tool.genre_matches("록", "인디음악, 록/메탈") and tool.genre_matches("R&B/Soul", "R&B/Soul")
    assert not tool.genre_matches("발라드", "R&B/Soul") and not tool.genre_matches("", "발라드")
    assert not tool.genre_matches("알앤비", "R&B/Soul"), "서비스의 부분 일치 그대로다 — 동의어를 더하지 않는다"


def _detail(qid, variant, scale, rank, *, group="gap", pair=None, stage=None, raw="발라드", genre="발라드", stated="발라드"):
    return {
        "query_id": qid, "pair_id": pair or qid[:3], "group": group, "variant": variant, "song_id": "1",
        "genre_match_scale": scale, "pool_rank": "" if rank is None else rank,
        "stage": stage or (tool.IN_POOL if rank else tool.FINAL_CUT),
        "stated_genre": stated, "analysis_genre": genre, "raw_genre": raw,
        "raw_artist_type": "", "analysis_artist_type": "", "target_registered_genre": "R&B/Soul",
        "target_genre_match": 0, "target_type_match": 1, "pool_genre_matched": 20,
        "text_hybrid_rank": 12, "audio_rank": "", "fused_rank_deep": 9, "boosted_rank_deep": 35,
        "target_adjustments": "", "target_paths": "metadata:3" if rank else "", "blanked_genre": "",
        "performance_search_rank": "-",
    }


def test_sweep_counts_queries_once_and_reports_candidate_stage_metrics() -> None:
    detail = [
        _detail("g01p", "perceived", 1.0, None), _detail("g01p", "perceived", 0.0, 8),
        _detail("g02p", "perceived", 1.0, 3), _detail("g02p", "perceived", 0.0, 1),
        _detail("g01r", "registered", 1.0, 2), _detail("g01r", "registered", 0.0, 14),
    ]
    summary = {(r["variant"], r["genre_match_scale"]): r for r in tool.summarize_sweep(detail, top_k=10)}
    assert (summary[("perceived", 1.0)]["in_pool"], summary[("perceived", 1.0)]["pool_top10"]) == (1, 1)
    assert (summary[("perceived", 0.0)]["in_pool"], summary[("perceived", 0.0)]["pool_top1"]) == (2, 1)
    assert summary[("registered", 0.0)]["pool_top10"] == 0, "등록 장르 표현의 손실도 같은 표에 나온다"
    assert summary[("(전체)", 1.0)]["n"] == 3
    assert summary[("perceived", 0.0)]["pool_mrr@10"] == pytest.approx((1 / 8 + 1) / 2, abs=1e-4)


def test_stage_summary_separates_genre_boost_effects_from_retrieval_misses() -> None:
    detail = [
        _detail("g01p", "perceived", 1.0, None), _detail("g01p", "perceived", 0.0, 8),
        _detail("g02p", "perceived", 1.0, None, stage=tool.NO_PATH),
        _detail("g02p", "perceived", 0.0, None, stage=tool.NO_PATH),
        _detail("g03r", "registered", 1.0, 5), _detail("g03r", "registered", 0.0, None),
        _detail("g04i", "identity_sound", 1.0, 4, raw="", genre="", stated=""),
        _detail("g04i", "identity_sound", 0.0, 4, raw="", genre="", stated=""),
        _detail("g05p", "perceived", 1.0, 9, raw=tool.RAW_UNAVAILABLE), _detail("g05p", "perceived", 0.0, 3, raw=tool.RAW_UNAVAILABLE),
        _detail("g06p", "perceived", 1.0, 9, raw="", genre="", stated="인디"), _detail("g06p", "perceived", 0.0, 9, raw="", genre="", stated="인디"),
    ]
    by_id = {r["query_id"]: r for r in tool.summarize_stages(detail, [1.0, 0.0])}
    assert by_id["g01p"]["genre_scale_effect"] == "가산을 낮추면 후보에 듦"
    # 가산을 낮춰도 검색 경로에 없으면 장르 가산의 문제가 아니다.
    assert by_id["g02p"]["genre_scale_effect"] == "어느 배율에서도 후보 밖" and by_id["g02p"]["stage@baseline"] == tool.NO_PATH
    assert by_id["g03r"]["genre_scale_effect"] == "가산을 낮추면 후보에서 빠짐"
    assert by_id["g04i"]["genre_scale_effect"] == "변화 없음"
    assert by_id["g05p"]["genre_scale_effect"] == "후보 안에서 오름(9→3)"
    assert by_id["g05p"]["analysis_note"] == tool.RAW_UNAVAILABLE, "원본이 없으면 역추정하지 않는다"
    assert by_id["g06p"]["analysis_note"] == "모델이 장르 낱말을 genre에 담지 않음"
    assert by_id["g01p"]["analysis_note"] == "모델 genre = 최종 genre"
    # 후보는 보조 신호로도 든다 — 어느 목록으로 들어왔는지가 함께 남는다(후보 밖이면 빈칸).
    assert (by_id["g01p"]["target_paths@baseline"], by_id["g01p"]["target_paths@0"]) == ("", "metadata:3")


def test_paths_text_marks_contributions_dropped_at_the_fusion_cut() -> None:
    """융합 절단에서 빠졌다가 다른 길로 다시 든 곡은 빠진 기여에 `*`가 붙는다 — 어느 길로 들어왔는지 헷갈리지 않게."""
    recorder = ExplainRecorder("q")
    recorder.path("T", "text_hybrid", 31, 0.01)
    recorder.mark_paths_dropped(["T"])
    recorder.path("T", "metadata", 3, 0.02)
    recorder.path("U", "text_hybrid", 1, 0.03)
    assert tool.paths_text(recorder.record, "T") == "text_hybrid:31*|metadata:3"
    assert tool.paths_text(recorder.record, "U") == "text_hybrid:1"
    assert tool.paths_text(recorder.record, "어느 목록에도 없던 곡") == ""


def test_queries_with_several_answers_are_read_by_their_best_answer() -> None:
    """정답이 여럿이면 후보 순위가 가장 좋은 정답으로 본다(sweep · stage · pair 표가 같은 기준)."""
    first = {**_detail("m1", "all", 1.0, None, pair="m1"), "song_id": "A"}
    second = {**_detail("m1", "all", 1.0, 7, pair="m1"), "song_id": "B"}
    low_first = {**_detail("m1", "all", 0.0, 12, pair="m1"), "song_id": "A"}
    low_second = {**_detail("m1", "all", 0.0, None, pair="m1"), "song_id": "B"}
    detail = [first, second, low_first, low_second]
    stage = tool.summarize_stages(detail, [1.0, 0.0])[0]
    assert (stage["pool_rank@baseline"], stage["pool_rank@0"]) == (7, 12)
    assert stage["genre_scale_effect"] == "후보 안에서 내림(7→12)"
    pair = tool.summarize_pairs(detail, [1.0, 0.0])[0]
    assert (pair["all:m1@1"], pair["all:m1@0"], pair["all_variants_in_pool@baseline"]) == (7, 12, 1)
    sweep = {r["genre_match_scale"]: r for r in tool.summarize_sweep(detail, top_k=10) if r["variant"] == "all"}
    assert (sweep[1.0]["pool_top10"], sweep[0.0]["pool_top10"]) == (1, 0)


def test_pair_summary_flags_pairs_where_some_variant_misses_the_pool() -> None:
    detail = [
        _detail("g01r", "registered", 1.0, 2, pair="g01"), _detail("g01r", "registered", 0.0, 6, pair="g01"),
        _detail("g01p", "perceived", 1.0, None, pair="g01"), _detail("g01p", "perceived", 0.0, 8, pair="g01"),
    ]
    line = tool.summarize_pairs(detail, [1.0, 0.0])[0]
    assert line["all_variants_in_pool@baseline"] == 0
    assert (line["registered:g01r@1"], line["perceived:g01p@1"], line["perceived:g01p@0"]) == (2, "", 8)


def test_structural_fit_checks_only_verifiable_fields() -> None:
    from src.backend.schemas.search import MatchingTrack

    group = MatchingTrack(id="a", score=0.0, title="곡", vocal_gender="여성", artist_types=["그룹"])
    solo = MatchingTrack(id="b", score=0.0, title="곡", vocal_gender="여성", artist_types=["솔로"])
    assert tool.structural_fit(group, {"vocal_gender": "여성", "type_any": ["그룹"]})
    assert not tool.structural_fit(solo, {"vocal_gender": "여성", "type_any": ["그룹"]})
    assert tool.structural_fit(solo, {}), "확인할 조건이 없으면 참이다 — '아이돌'·'잔잔한'은 여기서 판정하지 않는다"


def test_answer_summary_counts_only_intervention_rows() -> None:
    rows = [
        {"variant": "identity_sound", "intervention": 0, "answer_kind": "(개입 없음)"},
        {"variant": "identity_sound", "intervention": 1, "answer_kind": "genre_perceived", "genre_answer_scale": 1.0,
         "pool_rank_after": "", "answer_offered": 1},
        {"variant": "identity_sound", "intervention": 1, "answer_kind": "genre_perceived", "genre_answer_scale": 0.0,
         "pool_rank_after": 4, "answer_offered": 1},
    ]
    summary = {(r["variant"], r["answer_kind"], r["genre_answer_scale"]): r for r in tool.summarize_answers(rows, 10)}
    assert summary[("identity_sound", "genre_perceived", 1.0)]["in_pool"] == 0
    assert summary[("identity_sound", "genre_perceived", 0.0)]["in_top_k"] == 1
    assert ("identity_sound", "(개입 없음)", "") not in summary


# ---------------------------------------------------------------------------
# report — 리랭커를 켠 결과를 변형별로
# ---------------------------------------------------------------------------

def _eval_dir(path: Path, ranks: dict, scale: float, top_k: int = 10, candidate_k: int = 30, path_k=None,
              with_args: bool = True, index: str = "artifacts/index") -> Path:
    path.mkdir(parents=True)
    pool_column = f"candidate_rank@{candidate_k}"
    with (path / "search_eval_dev_detail.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", pool_column, "rerank_rank", "rerank_outcome"])
        writer.writeheader()
        for qid, (pool, rank, outcome) in ranks.items():
            writer.writerow({"query_id": qid, pool_column: pool, "rerank_rank": rank, "rerank_outcome": outcome})
    info = {
        # output_dir은 폴더마다 다르다 — 재는 조건이 아니다. 색인 폴더(qdrant_path)는 조건이다.
        "args": {"top_k": top_k, "candidate_k": candidate_k, "path_k": path_k, "output_dir": str(path)},
        "analysis": {"path": "cache.json"}, "corpus": {"qdrant_path": index, "point_count": 60},
        "ranking_switches": {"genre_match_scale": scale, "genre_answer_scale": 1.0},
        "reranker": {"backend": "gemini_listwise"}, "gemini": {"backend": "api_key"},
    }
    if not with_args:
        del info["args"]
    (path / "search_eval_dev_runinfo.json").write_text(json.dumps(info), encoding="utf-8")
    return path


def test_report_groups_by_variant_and_keeps_failed_reranks_visible(tmp_path, capsys) -> None:
    base = _eval_dir(tmp_path / "base", {"g01r": (2, 1, "applied"), "g01p": ("", "", "applied"), "g01i": (9, 4, "applied")}, 1.0)
    low = _eval_dir(tmp_path / "low", {"g01r": (6, 3, "applied"), "g01p": (8, 2, "applied"), "g01i": (9, "", "fallback")}, 0.0)
    out = tmp_path / "report"
    tool.main(["report", "--queries", str(QUERIES), "--runs", f"base={base}", f"low={low}", "--output-dir", str(out)])
    with (out / "rerank_summary.csv").open(encoding="utf-8") as f:
        summary = {(r["run"], r["variant"]): r for r in csv.DictReader(f)}
    assert summary[("base", "perceived")]["hit@10"] == "0" and summary[("low", "perceived")]["hit@10"] == "1"
    assert summary[("low", "registered")]["mrr@10"] == "0.3333"
    assert summary[("low", "identity_sound")]["rerank_failed"] == "1", "리랭킹이 실패한 실행은 숨기지 않는다"
    # g01i는 low에서 후보 9위인데 최종 10위 밖이다 — Hit@10은 후보 순위가 아니라 최종 순위로 센다.
    assert (summary[("low", "identity_sound")]["in_pool"], summary[("low", "identity_sound")]["hit@10"]) == ("1", "0")
    assert (summary[("low", "(전체)")]["in_pool"], summary[("low", "(전체)")]["hit@10"], summary[("low", "(전체)")]["hit@1"]) == ("3", "2", "0")
    assert summary[("base", "identity_sound")]["rerank_failed"] == "0"
    assert summary[("low", "(전체)")]["genre_match_scale"] == "0.0", "실제 적용 배율이 함께 남는다"
    printed = capsys.readouterr().out
    assert "든 ['g01p']" in printed and "빠진 ['g01i']" in printed
    compare_line = next(line for line in printed.splitlines() if line.startswith("base → low"))
    assert "주의: low 리랭킹 실패 1건" in compare_line and "base 리랭킹 실패" not in compare_line, (
        "실패가 섞인 실행과의 비교라는 것이 비교 줄에도 남는다"
    )

    # 정책 생략은 실패가 아니다. 질의 수가 다른 실행(중간에 죽은 것)은 섞지 않는다.
    skipped = _eval_dir(tmp_path / "skipped", {"g01r": (2, 1, "policy_skipped"), "g01p": ("", "", "applied"), "g01i": (9, 4, "skipped")}, 1.0)
    tool.main(["report", "--queries", str(QUERIES), "--runs", f"skipped={skipped}", "--output-dir", str(out)])
    with (out / "rerank_summary.csv").open(encoding="utf-8") as f:
        total = next(r for r in csv.DictReader(f) if r["variant"] == "(전체)")
    assert (total["rerank_failed"], total["rerank_policy_skipped"]) == ("0", "1")
    short = _eval_dir(tmp_path / "short", {"g01r": (2, 1, "applied")}, 0.0)
    with pytest.raises(ValueError):
        tool.main(["report", "--queries", str(QUERIES), "--runs", f"base={base}", f"short={short}", "--output-dir", str(out)])


def test_report_counts_a_rank_as_a_hit_only_within_ten(tmp_path, capsys) -> None:
    """평가기를 `--top-k 15`로 재면 `rerank_rank`에 11~15위도 적힌다. 순위가 적혀 있다고 Hit@10은 아니다."""
    base = _eval_dir(tmp_path / "base", {"g01r": (2, 1, "applied"), "g01p": (20, 9, "applied"), "g01i": (9, 12, "applied")},
                     1.0, top_k=15)
    low = _eval_dir(tmp_path / "low", {"g01r": (2, 10, "applied"), "g01p": (20, 11, "applied"), "g01i": (9, 4, "applied")},
                    0.0, top_k=15)
    out = tmp_path / "report"
    tool.main(["report", "--queries", str(QUERIES), "--runs", f"base={base}", f"low={low}", "--output-dir", str(out)])
    with (out / "rerank_summary.csv").open(encoding="utf-8") as f:
        summary = {(r["run"], r["variant"]): r for r in csv.DictReader(f)}
    total = summary[("base", "(전체)")]
    assert (total["in_pool"], total["hit@1"], total["hit@10"], total["mrr@10"]) == ("3", "1", "2", "0.3704"), (
        "12위는 Hit@10도 MRR@10도 아니다 — (1 + 1/9) / 3"
    )
    total = summary[("low", "(전체)")]
    assert (total["hit@10"], total["mrr@10"]) == ("2", "0.1167"), "10위는 들고 11위는 빠진다 — (1/10 + 1/4) / 3"
    assert (summary[("low", "perceived")]["hit@10"], summary[("low", "perceived")]["mrr@10"]) == ("0", "0.0")
    printed = capsys.readouterr().out
    assert "든 ['g01i']" in printed and "빠진 ['g01p']" in printed, "9위 → 11위는 빠진 것, 12위 → 4위는 든 것이다"
    with (out / "rerank_ranks.csv").open(encoding="utf-8") as f:
        ranks = {r["query_id"]: r for r in csv.DictReader(f)}
    assert (ranks["g01p"]["base:rank"], ranks["g01p"]["low:rank"]) == ("9", "11"), "순위표에는 잰 순위를 그대로 둔다"

    # 10위보다 얕게 잰 실행은 6~10위의 순위가 없다 — Hit@10이 조용히 줄어들게 두지 않는다.
    shallow = _eval_dir(tmp_path / "shallow", {"g01r": (2, 1, "applied"), "g01p": (20, "", "applied"), "g01i": (9, 4, "applied")},
                        0.0, top_k=5)
    with pytest.raises(ValueError, match="top-k 5"):
        tool.main(["report", "--queries", str(QUERIES), "--runs", f"base={base}", f"shallow={shallow}", "--output-dir", str(out)])


INDEX_NOTE = "(색인 폴더가 다르다 — 같은 내용인지는 runinfo로 알 수 없다)"


def test_report_refuses_runs_whose_pool_column_means_another_depth(tmp_path) -> None:
    """후보 깊이가 다른 실행을 나란히 놓으면 `후보` 열이 같은 뜻이 아니다 — 받지 않는다."""
    ranks = {"g01r": (2, 1, "applied"), "g01p": (20, 9, "applied"), "g01i": (9, 4, "applied")}
    base = _eval_dir(tmp_path / "base", ranks, 1.0)
    deep = _eval_dir(tmp_path / "deep", ranks, 1.0, candidate_k=50)
    out = tmp_path / "report"
    with pytest.raises(ValueError, match="후보 깊이가 다르다"):
        tool.main(["report", "--queries", str(QUERIES), "--runs", f"base={base}", f"deep={deep}", "--output-dir", str(out)])
    # 한 실행의 폴더들 사이에서 깊이가 섞여도 같다.
    part = _eval_dir(tmp_path / "part", {"g02r": (2, 1, "applied")}, 1.0, candidate_k=50)
    with pytest.raises(ValueError, match="후보 순위 열이 하나가 아니다"):
        tool.main(["report", "--queries", str(QUERIES), "--runs", f"mixed={base},{part}", "--output-dir", str(out)])
    # 몇 위까지 잰 실행인지 모르면(args.top_k 없음) Hit@10을 세지 않는다.
    unknown = _eval_dir(tmp_path / "unknown", ranks, 1.0, with_args=False)
    with pytest.raises(ValueError, match="args.top_k가 없다"):
        tool.main(["report", "--queries", str(QUERIES), "--runs", f"unknown={unknown}", "--output-dir", str(out)])


def test_report_lists_every_condition_that_differs_from_the_base_run(tmp_path, capsys) -> None:
    """조건을 하나씩 고르지 않고 runinfo를 통째로 견준다 — 실험 변수도, 뜻하지 않게 달라진 조건도 비교 줄에 보인다."""
    ranks = {"g01r": (2, 1, "applied"), "g01p": (20, 9, "applied"), "g01i": (9, 4, "applied")}
    base = _eval_dir(tmp_path / "base", ranks, 1.0)
    again = _eval_dir(tmp_path / "again", ranks, 1.0)
    low = _eval_dir(tmp_path / "low", ranks, 0.0, path_k=60, top_k=15)
    copy = _eval_dir(tmp_path / "copy", ranks, 1.0, index="artifacts/index_b")
    out = tmp_path / "report"
    tool.main(["report", "--queries", str(QUERIES), "--runs", f"base={base}", f"again={again}", f"low={low}", f"copy={copy}",
               "--output-dir", str(out)])
    with (out / "rerank_summary.csv").open(encoding="utf-8") as f:
        total = {r["run"]: r for r in csv.DictReader(f) if r["variant"] == "(전체)"}
    assert (total["base"]["conditions_vs_base"], total["again"]["conditions_vs_base"]) == ("(기준)", "같음"), (
        "결과를 쓴 폴더는 조건이 아니다"
    )
    # 색인 폴더가 다르면 숨기지 않는다 — 복사본인지 다른 색인인지 runinfo로는 알 수 없다.
    assert total["copy"]["conditions_vs_base"] == (
        'corpus.qdrant_path: "artifacts/index" → "artifacts/index_b"' + INDEX_NOTE
    )
    assert total["low"]["conditions_vs_base"] == (
        "args.top_k: 10 → 15; args.path_k: null → 60; ranking_switches.genre_match_scale: 1.0 → 0.0"
    )
    assert (total["low"]["top_k"], total["low"]["candidate_k"], total["low"]["path_k"]) == ("15", "30", "60")
    assert (total["base"]["top_k"], total["base"]["path_k"], total["base"]["analysis_cache"]) == ("10", "", "cache.json")
    printed = capsys.readouterr().out
    line = next(text for text in printed.splitlines() if text.startswith("base → low"))
    assert "다른 조건: args.top_k: 10 → 15; args.path_k: null → 60; ranking_switches.genre_match_scale: 1.0 → 0.0" in line
    assert next(text for text in printed.splitlines() if text.startswith("base → again")).endswith("다른 조건: 없음")
    assert next(text for text in printed.splitlines() if text.startswith("base → copy")).endswith(
        '다른 조건: corpus.qdrant_path: "artifacts/index" → "artifacts/index_b"' + INDEX_NOTE
    )


# ---------------------------------------------------------------------------
# summarize_clarify — 재질문 하네스 실행 간 비교 (experiments/genre_memory_gap의 스크립트)
# ---------------------------------------------------------------------------

def _clarify_script():
    import importlib.util
    import sys

    path = ROOT / "experiments/genre_memory_gap/summarize_clarify.py"
    spec = importlib.util.spec_from_file_location("genre_gap_summarize_clarify", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clarify_dir(path: Path, hits: dict, statuses, *, top_k=10, candidate_k=30, gemini: str = "api_key",
                 answer_scale: float = 1.0, analysis_cache: str = "cache_a.json", extra_args=None,
                 with_args: bool = True, index: str = "artifacts/index") -> Path:
    """hits: 질의 → {정책: 값}. 값은 hit10 하나, 목표 곡마다의 hit10 목록, 또는 (hit10, ran) — ran 0은 그 정책이 돌지 않은 행이다.

    하네스처럼 `reject_only`가 **돈**(ran 1) 질의가 그 실행의 개입 대상이다. statuses가 None이면 상태 집계 칸을 아예 쓰지 않는다.
    """
    path.mkdir(parents=True)
    detail = []
    for qid, by_policy in hits.items():
        for policy, value in by_policy.items():
            pairs = [value] if isinstance(value, tuple) else [(v, 1) for v in (value if isinstance(value, list) else [value])]
            detail.extend({"query_id": qid, "policy": policy, "ran": str(ran), "hit10": hit, "candidate_rank@30": "5"}
                          for hit, ran in pairs)
    with (path / "clarify_detail.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    with (path / "clarify_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "n_executed", "full_hit@10", "int_hit@10", "int_candidate_recall@30"])
        writer.writeheader()
        for policy in dict.fromkeys(row["policy"] for row in detail):
            rows = [row for row in detail if row["policy"] == policy]
            share = sum(int(row["hit10"]) for row in rows) / len(rows)
            writer.writerow({"policy": policy, "n_executed": len({r["query_id"] for r in rows if r["ran"] == "1"}),
                             "full_hit@10": share, "int_hit@10": share, "int_candidate_recall@30": 1.0})
    info = {
        # output_dir · 걸린 시간은 실행마다 다르다 — 재는 조건이 아니다. 색인 폴더(qdrant_path)는 조건이다.
        "args": {"top_k": top_k, "candidate_k": candidate_k, "answer_multiplier": None, "query_ids": "",
                 "output_dir": str(path), **(extra_args or {})},
        "analysis": {"path": analysis_cache}, "corpus": {"qdrant_path": index, "point_count": 60},
        "ranking_switches": {"genre_match_scale": 1.0, "genre_answer_scale": answer_scale},
        "gemini": {"backend": gemini, "setting": gemini},
        "clarify": {"elapsed_seconds": len(str(path))},
    }
    if statuses is not None:
        info["clarify"]["rerank_status_counts"] = statuses
    if not with_args:
        del info["args"]
    (path / "clarify_runinfo.json").write_text(json.dumps(info), encoding="utf-8")
    return path


def _compare(script, tmp_path, runs: dict):
    out = tmp_path / "compare.csv"
    script.main([*(f"{name}={folder}" for name, folder in runs.items()), "--output", str(out)])
    with out.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


CLARIFY_OK = {"applied": 6, "policy_skipped:image_dominant": 2}
TWO_QUERIES = {"a": {"reject_only": 0, "oracle:genre": 1}, "b": {"reject_only": 0, "oracle:genre": 0}}


def test_clarify_comparison_leaves_failed_runs_out_of_the_common_table(tmp_path, capsys) -> None:
    """리랭킹 실패가 섞인 실행은 공통 개입 대상 비교에서 뺀다 — 공통 대상을 정할 때도 쓰지 않고, 뺐다는 사실을 남긴다."""
    script = _clarify_script()
    first = _clarify_dir(tmp_path / "first", TWO_QUERIES, CLARIFY_OK)
    second = _clarify_dir(tmp_path / "second", {"a": {"reject_only": 0, "oracle:genre": 1}, "b": {"reject_only": 1, "oracle:genre": 1}},
                          CLARIFY_OK)
    # 실패한 실행은 폴백 순서 때문에 b가 개입 대상에서 빠졌다. 섞으면 정상 실행의 비교 대상이 2건 → 1건으로 준다.
    broken = _clarify_dir(tmp_path / "broken", {"a": {"reject_only": 0, "oracle:genre": 1}, "b": {"reject_only": (1, 0)}},
                          {"applied": 2, "failed": 2})
    lines = _compare(script, tmp_path, {"first": first, "second": second, "broken": broken})

    common = {(r["run"], r["policy"]): r for r in lines if r["scope"] == script.SCOPE_COMMON}
    assert {run for run, _ in common} == {"first", "second"}, "실패한 실행은 공통 표에 정상 실행처럼 서지 않는다"
    assert all(r["n_intervention"] == "2" and r["n_executed"] == "2" for r in common.values()), "공통 대상은 뺀 실행 없이 정한다"
    assert (common[("first", "oracle:genre")]["hit_sum"], common[("second", "oracle:genre")]["hit_sum"]) == ("1.0", "2.0")
    excluded = [r for r in lines if r["scope"] == script.SCOPE_EXCLUDED]
    assert [(r["run"], r["excluded_reason"]) for r in excluded] == [("broken!", "리랭킹 실패 2건")]
    per_run = [r for r in lines if r["scope"] == script.SCOPE_PER_RUN and r["run"].startswith("broken")]
    assert per_run and all(r["run"] == "broken!" and r["rerank_failed"] == "2" for r in per_run), "실행별 표에는 표시를 달고 남긴다"
    printed = capsys.readouterr().out
    assert "비교에서 뺀 실행: broken — 리랭킹 실패 2건" in printed
    assert "공통 개입 대상 2건(비교에 쓴 실행 2개 기준): a, b" in printed
    assert "기준 실행(first)과 다른 조건: second — 없음" in printed, "결과 폴더 · 걸린 시간은 조건이 아니다"
    header = next(line for line in printed.splitlines() if line.startswith("정책"))
    assert "first" in header and "second" in header and "broken" not in header

    # 비교에 쓸 수 있는 실행이 하나도 없으면 공통 표를 만들지 않는다.
    assert not [r for r in _compare(script, tmp_path, {"broken": broken}) if r["scope"] == script.SCOPE_COMMON]
    assert "공통 개입 대상 표를 만들지 않았다" in capsys.readouterr().out


def test_clarify_comparison_keeps_only_queries_every_run_intervened_on(tmp_path, capsys) -> None:
    """공통 개입 대상은 교집합이다. 개입 대상은 reject_only가 **돈** 질의고, 목표 곡이 여럿인 질의는 질의 안에서 먼저 평균한다."""
    script = _clarify_script()
    first = _clarify_dir(tmp_path / "first", {
        "a": {"reject_only": 0, "flow:oracle": [1, 0], "oracle:genre": 1},
        "b": {"reject_only": 0, "flow:oracle": [1], "oracle:genre": (0, 0)},    # 장르 질문이 나오지 않아 그 정책은 돌지 않았다
        "c": {"reject_only": 0, "flow:oracle": [1]},
        "d": {"reject_only": (1, 0)},                                           # 처음부터 찾은 질의 — 개입 대상이 아니다
    }, CLARIFY_OK)
    second = _clarify_dir(tmp_path / "second", {
        "a": {"reject_only": 0, "flow:oracle": [1, 1], "oracle:genre": 1},
        "b": {"reject_only": 1, "flow:oracle": [0], "oracle:genre": 0},
        "c": {"reject_only": (1, 0)},                                           # 이 실행에서는 처음부터 찾았다
        "d": {"reject_only": (1, 0)},
    }, CLARIFY_OK)
    lines = _compare(script, tmp_path, {"first": first, "second": second})
    common = {(r["run"], r["policy"]): r for r in lines if r["scope"] == script.SCOPE_COMMON}
    printed = capsys.readouterr().out
    assert "실행별 개입 대상: first 3 · second 2" in printed, "ran 0인 reject_only 행은 개입 대상이 아니다"
    assert "공통 개입 대상 2건(비교에 쓴 실행 2개 기준): a, b" in printed, "한쪽에서만 개입 대상인 c는 빠진다"
    assert all(r["n_intervention"] == "2" for r in common.values())
    assert (common[("first", "flow:oracle")]["hit_sum"], common[("second", "flow:oracle")]["hit_sum"]) == ("1.5", "1.0"), (
        "목표 곡 둘 중 하나만 찾은 질의는 0.5다"
    )
    assert (common[("first", "reject_only")]["hit_sum"], common[("second", "reject_only")]["hit_sum"]) == ("0.0", "1.0")
    # n_executed는 하네스와 같은 뜻 — 그 정책이 실제로 돈 질의 수. 평균은 행이 있는 질의 전부로 낸다.
    genre = common[("first", "oracle:genre")]
    assert (genre["n_executed"], genre["n_queries"], genre["hit_sum"]) == ("1", "2", "1.0")


def test_clarify_comparison_refuses_runs_measured_at_another_depth(tmp_path) -> None:
    """`hit10`은 보여 준 top_k곡 안의 값이다. top_k가 10이 아니면 "10위 안"이 아니게 된다 — 받지 않는다."""
    script = _clarify_script()
    base = _clarify_dir(tmp_path / "base", TWO_QUERIES, CLARIFY_OK)
    shallow = _clarify_dir(tmp_path / "shallow", TWO_QUERIES, CLARIFY_OK, top_k=5)
    wide = _clarify_dir(tmp_path / "wide", TWO_QUERIES, CLARIFY_OK, candidate_k=50)
    unknown = _clarify_dir(tmp_path / "unknown", TWO_QUERIES, CLARIFY_OK, with_args=False)
    for name, folder, wanted in (("shallow", shallow, "args.top_k가 5"), ("wide", wide, "args.candidate_k가 50"),
                                 ("unknown", unknown, "args.top_k가 None")):
        with pytest.raises(ValueError, match=wanted):
            _compare(script, tmp_path, {"base": base, name: folder})


def test_clarify_comparison_drops_runs_it_cannot_vouch_for(tmp_path, capsys) -> None:
    """상태를 모르는 실행 · 이름을 모르는 실패 · Gemini 호출 경로가 다른 실행은 공통 비교에 넣지 않는다."""
    script = _clarify_script()
    runs = {
        "base": _clarify_dir(tmp_path / "base", TWO_QUERIES, CLARIFY_OK),
        "nostatus": _clarify_dir(tmp_path / "nostatus", {"a": {"reject_only": 0, "oracle:genre": 1}}, None),
        "empty": _clarify_dir(tmp_path / "empty", {"a": {"reject_only": 0, "oracle:genre": 1}}, {}),
        "odd": _clarify_dir(tmp_path / "odd", {"a": {"reject_only": 0, "oracle:genre": 1}}, {"applied": 3, "fallback": 2}),
        "vertex": _clarify_dir(tmp_path / "vertex", {"a": {"reject_only": 0, "oracle:genre": 1}}, CLARIFY_OK, gemini="vertex"),
    }
    lines = _compare(script, tmp_path, runs)
    reasons = {r["run"]: r["excluded_reason"] for r in lines if r["scope"] == script.SCOPE_EXCLUDED}
    assert reasons["nostatus!"] == reasons["empty!"] == "리랭커 상태 집계 없음(모름)"
    assert reasons["odd!"] == "리랭킹 실패 2건", "정상으로 아는 상태가 아니면 실패로 센다"
    assert reasons["vertex!"] == (
        'Gemini 호출 경로가 기준 실행과 다름(gemini.backend "api_key" → "vertex", gemini.setting "api_key" → "vertex")'
    )
    common = [r for r in lines if r["scope"] == script.SCOPE_COMMON]
    assert {r["run"] for r in common} == {"base"} and all(r["n_intervention"] == "2" for r in common), (
        "뺀 실행들은 a만 개입 대상이다 — 섞였다면 공통 대상이 1건으로 줄었다"
    )
    nostatus = next(r for r in lines if r["scope"] == script.SCOPE_PER_RUN and r["run"] == "nostatus!")
    assert (nostatus["rerank_failed"], nostatus["rerank_statuses"]) == ("", "(집계 없음)"), "모르는 것을 0건으로 적지 않는다"
    printed = capsys.readouterr().out
    assert "비교에서 뺀 실행: nostatus — 리랭커 상태 집계 없음(모름)" in printed
    assert "공통 개입 대상 2건(비교에 쓴 실행 1개 기준): a, b" in printed


def test_clarify_comparison_lists_every_condition_that_differs_from_the_base(tmp_path, capsys) -> None:
    """실험 변수일 수 있는 조건은 막지 않는다. 대신 기준 실행과 다른 칸을 전부 적는다."""
    script = _clarify_script()
    runs = {
        "base": _clarify_dir(tmp_path / "base", TWO_QUERIES, CLARIFY_OK),
        "scale": _clarify_dir(tmp_path / "scale", TWO_QUERIES, CLARIFY_OK, answer_scale=0.0),
        "mult": _clarify_dir(tmp_path / "mult", TWO_QUERIES, CLARIFY_OK, extra_args={"answer_multiplier": 5}),
        "cache": _clarify_dir(tmp_path / "cache", TWO_QUERIES, CLARIFY_OK, analysis_cache="cache_b.json"),
        "partial": _clarify_dir(tmp_path / "partial", {"a": TWO_QUERIES["a"]}, CLARIFY_OK, extra_args={"query_ids": "a"}),
        "copy": _clarify_dir(tmp_path / "copy", TWO_QUERIES, CLARIFY_OK, index="artifacts/index_b"),
    }
    lines = _compare(script, tmp_path, runs)
    printed = capsys.readouterr().out
    assert "기준 실행(base)과 다른 조건: scale — ranking_switches.genre_answer_scale: 1.0 → 0.0" in printed
    assert "기준 실행(base)과 다른 조건: mult — args.answer_multiplier: null → 5" in printed
    assert 'cache — analysis.path: "cache_a.json" → "cache_b.json"' in printed
    assert 'partial — args.query_ids: "" → "a"; 잰 질의: 2건 → 1건' in printed
    assert 'copy — corpus.qdrant_path: "artifacts/index" → "artifacts/index_b"' + INDEX_NOTE in printed, (
        "색인 폴더가 다르면 숨기지 않는다"
    )
    assert not [r for r in lines if r["scope"] == script.SCOPE_EXCLUDED], "조건이 다르다고 빼지는 않는다"
    versus = {r["run"]: r["conditions_vs_base"] for r in lines if r["scope"] == script.SCOPE_COMMON}
    assert (versus["base"], versus["scale"]) == ("(기준)", "ranking_switches.genre_answer_scale: 1.0 → 0.0")
    per_run = {r["run"]: r for r in lines if r["scope"] == script.SCOPE_PER_RUN and r["policy"] == "reject_only"}
    assert (per_run["mult"]["answer_multiplier"], per_run["partial"]["query_ids"], per_run["partial"]["n_measured_queries"]) == (
        "5", "a", "1"
    )
    assert (per_run["base"]["top_k"], per_run["base"]["candidate_k"], per_run["cache"]["analysis_cache"]) == ("10", "30", "cache_b.json")
    assert (per_run["base"]["conditions_vs_base"], per_run["mult"]["conditions_vs_base"]) == (
        "(기준)", "args.answer_multiplier: null → 5"
    ), "실행별 행에도 다른 조건이 남는다"
    assert "공통 개입 대상 1건(비교에 쓴 실행 6개 기준): a" in printed, "일부만 잰 실행이 끼면 공통 대상이 준다 — 왜 줄었는지는 위 줄에 있다"


# ---------------------------------------------------------------------------
# run_conditions — 실행 조건 비교 · 리랭킹 상태 (두 집계 도구가 함께 쓴다)
# ---------------------------------------------------------------------------

def test_rerank_failures_never_count_an_unknown_state_as_ok() -> None:
    from src.retrieval.run_conditions import rerank_failures, rerank_state_ok

    assert rerank_failures(None) is None and rerank_failures({}) is None, "집계가 없으면 0건이 아니라 모름이다"
    assert rerank_failures({"applied": 405, "policy_skipped:image_dominant": 59, "skipped": 4, "policy_skipped": 1}) == 0
    assert rerank_failures({"applied": 3, "failed": 1, "exception": 1, "no_status": 1, "fallback": 2, "unknown": 1, "": 1}) == 7
    assert not rerank_state_ok("policy_skippedX") and not rerank_state_ok("partial_failure")


def test_run_conditions_compare_whole_runinfo_blocks() -> None:
    from src.retrieval.run_conditions import condition_differences, flatten, run_conditions

    info = {
        "created_at": "지금", "args": {"top_k": 10, "output_dir": "여기"}, "corpus": {"qdrant_path": "색인", "point_count": 60},
        "bm25": {"sha256_12": "bm25지문"}, "lyrics_source": {"content_sha256_16": "가사지문"},
        "reranker": {"backend": "gemini_listwise", "config": {"model_name": "m", "passes": 1}},
        "clarify": {"elapsed_seconds": 3.0, "rerank_status_counts": {"applied": 2}, "question_policy": "p"},
        "search_reference_year_env": "2026", "genre_gap_notes": ["메모"],
    }
    assert set(flatten(info)) == {
        "args.top_k", "corpus.qdrant_path", "corpus.point_count", "bm25.sha256_12", "lyrics_source.content_sha256_16",
        "reranker.backend", "reranker.config.model_name", "reranker.config.passes", "clarify.question_policy",
        "search_reference_year_env",
    }, "결과 폴더 · 걸린 시간 · 상태 집계 · 블록 밖의 칸만 조건이 아니다"
    other = json.loads(json.dumps(info))
    other["args"]["output_dir"] = "저기"
    other["clarify"].update(elapsed_seconds=9.0, rerank_status_counts={"applied": 1, "failed": 1})
    assert condition_differences(run_conditions([info]), run_conditions([other])) == []
    # 코퍼스가 같은지를 가르는 칸(색인 폴더 · BM25 지문 · 가사 지문)과 중첩된 설정, 한쪽에만 있는 칸까지 모두 나온다.
    other["corpus"]["qdrant_path"] = "다른 색인"
    other["bm25"]["sha256_12"] = "다른 bm25"
    other["lyrics_source"]["content_sha256_16"] = "다른 가사"
    other["reranker"]["config"]["model_name"] = "n"
    other["gemini"] = {"backend": "vertex"}
    assert condition_differences(run_conditions([info]), run_conditions([other])) == [
        'corpus.qdrant_path: "색인" → "다른 색인"' + INDEX_NOTE,
        'bm25.sha256_12: "bm25지문" → "다른 bm25"', 'lyrics_source.content_sha256_16: "가사지문" → "다른 가사"',
        'reranker.config.model_name: "m" → "n"', 'gemini.backend: (없음) → "vertex"',
    ]
    # 폴더가 여럿인 실행은 폴더 자리마다 견준다. 값의 종류가 같고 자리만 다르면 자리대로 전부 적는다.
    x, y = {"args": {"input": "x"}}, {"args": {"input": "y"}}
    assert condition_differences(run_conditions([x, y, x]), run_conditions([x, y, y])) == [
        'args.input: "x" | "y" | "x" → "x" | "y" | "y"'
    ]
    assert condition_differences(run_conditions([x, x]), run_conditions([x, y])) == ['args.input: "x" → "x" | "y"']
    assert condition_differences(run_conditions([x, y]), run_conditions([x, y])) == []


# ---------------------------------------------------------------------------
# trace · answers — 실제 라우터 흐름을 대역 색인으로 끝까지 (명령 한 번이 통째로 도는지)
# ---------------------------------------------------------------------------

def _bench(tmp_path, monkeypatch):
    """텍스트 경로 60곡으로 만든 상대적 탈락의 축소판.

    1~30위는 솔로(1~10위 댄스 · 11~30위 발라드), 31~60위는 그룹·발라드다. 정답 T는 35위의 그룹 곡이고 등록 장르만
    R&B/Soul이다. "여자 그룹"이라고 말하면 형태 단서가 텍스트 31위 밖의 그룹 곡 15곡(T 포함)을 후보 경쟁에 올린다.
    거기에 "발라드"까지 말하면 등록 장르가 발라드인 34곡이 장르 가산(+1/61)을 받아 T를 넘는다 — T는 감점 없이
    후보 30 밖으로 밀린다. 장르 가산은 융합에서 살아남은 곡들 사이의 순서만 바꾸므로, 이런 탈락은 보조 경로가
    후보를 30곡보다 많이 올렸을 때만 생긴다.
    """
    from src.backend.schemas.query import ArtistTypeClue
    from src.retrieval.analysis_cache import new_cache
    from tests.test_search_explain import _analysis, _FakeTextService, _router, _track

    def song(rank: int):
        if rank <= 30:
            return _track(f"s{rank}", title=f"곡{rank}", genre="댄스" if rank <= 10 else "발라드",
                          vocal_gender="여성", artist_types=["솔로"])
        return _track(f"s{rank}", title=f"곡{rank}", genre="발라드", vocal_gender="여성", artist_types=["그룹"])

    hits = [song(rank) for rank in range(1, 61)]
    hits[34] = _track("T", title="정답", genre="R&B/Soul", vocal_gender="여성", artist_types=["그룹"])

    class TextService(_FakeTextService):
        def fetch_tracks_by_ids(self, song_ids):
            by_id = {t.id: t for t in self._hits}
            return {sid: by_id[sid] for sid in song_ids if sid in by_id}

    router = _router(hits)
    router._text_svc = TextService(hits)
    monkeypatch.setattr("src.backend.api.dependencies.get_search_router", lambda: router)
    monkeypatch.setattr("src.retrieval.evaluate_search_accuracy.collect_run_info", lambda *a, **k: {})

    group = ArtistTypeClue(values=["그룹"], confidence=0.85)
    specs = [
        ("x1r", "search", "여자 그룹의 알앤비 노래", "registered", "알앤비", dict(genre="R&B/Soul", artist_type=group)),
        ("x1p", "search", "여자 그룹의 발라드 노래", "perceived", "발라드",
         dict(genre="발라드", artist_type=group, korean_tags=["발라드", "여자그룹"])),
        ("x1i", "search", "여자 아이돌의 잔잔한 노래", "identity_sound", "", dict()),
        ("xb1", "broad", "여자 그룹의 잔잔한 발라드", "broad", "발라드", dict(genre="발라드", artist_type=group)),
    ]
    queries = tmp_path / "queries.csv"
    cache = new_cache(source=str(queries), split="dev")
    rows = []
    for qid, kind, query, variant, stated, fields in specs:
        broad = kind == "broad"
        rows.append({
            **{c: "" for c in tool.QUERY_COLUMNS}, "split": "dev", "query_id": qid, "query_type": kind, "query": query,
            "relevant_ids": "" if broad else "T", "pair_id": "" if broad else "x1", "group": "broad" if broad else "gap",
            "variant": variant, "target_scope": "broad" if broad else "specific", "stated_genre": stated,
            "perceived_genre": "" if broad else "발라드",
            "conditions": json.dumps({"vocal_gender": "여성", "type_any": ["그룹"]}) if broad else "",
        })
        cache.put(qid, query, _analysis(original_query=query, **fields))
    tool.write_csv(queries, rows, tool.QUERY_COLUMNS)
    cache_path = tmp_path / "cache.json"
    cache.save(cache_path)
    return queries, cache_path, router


def _read(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_trace_runs_end_to_end_and_shows_the_relative_drop(tmp_path, monkeypatch) -> None:
    import os

    queries, cache_path, router = _bench(tmp_path, monkeypatch)
    out = tmp_path / "trace"
    monkeypatch.setenv("GENRE_MATCH_SCALE", "0.5")     # 들어올 때의 값 — 도는 동안 덮어쓰고 끝나면 되돌린다
    try:
        tool.main(["trace", "--input", str(queries), "--analysis-cache", str(cache_path), "--scales", "1,0",
                   "--deep-k", "100", "--output-dir", str(out)])
    finally:
        router.shutdown()
    assert os.environ["GENRE_MATCH_SCALE"] == "0.5", "배율을 바꿔 가며 돈 뒤 들어올 때의 환경으로 되돌린다"

    stages = {r["query_id"]: r for r in _read(out / "stage_summary.csv")}
    # 체감 장르: 융합까지는 살아 있었는데(보정 뒤 절단) 등록 장르가 발라드인 곡들의 가산에 밀려 후보 밖 → 끄면 5위.
    assert (stages["x1p"]["stage@baseline"], stages["x1p"]["pool_rank@baseline"]) == (tool.FINAL_CUT, "")
    assert stages["x1p"]["pool_rank@0"] == "5" and stages["x1p"]["genre_scale_effect"] == "가산을 낮추면 후보에 듦"
    assert stages["x1p"]["target_genre_match"] == "0" and stages["x1p"]["target_type_match"] == "1"
    # 정답은 텍스트 35위라 주 경로(상위 30)에는 없고, 가수 형태 단서가 여는 보조 신호(metadata)의 다섯째로 들어온다.
    assert stages["x1p"]["target_paths@baseline"] == stages["x1p"]["target_paths@0"] == "metadata:5"
    assert stages["x1r"]["target_paths@baseline"] == "metadata:5" and stages["x1i"]["target_paths@baseline"] == ""
    assert stages["x1p"]["performance_search_rank"] == "-", "보컬·편성 전용 검색이 돌지 않는 질의"
    assert stages["x1p"]["pool_genre_matched@baseline"] == "30", "현행 후보 30곡이 전부 등록 장르가 발라드인 곡이다"
    assert int(stages["x1p"]["boosted_rank_deep@baseline"]) > 30 >= int(stages["x1p"]["boosted_rank_deep@0"])
    # 등록 장르 표현: 가산 덕에 1위였다가 끄면 5위 — 대조군의 손실도 같은 표에 나온다.
    assert (stages["x1r"]["pool_rank@baseline"], stages["x1r"]["pool_rank@0"]) == ("1", "5")
    assert stages["x1r"]["genre_scale_effect"] == "후보 안에서 내림(1→5)"
    # 장르도 형태도 말하지 않은 변형: 텍스트 35위라 어느 경로에도 없다. 가산을 낮춰도 그대로다 — 장르 가산의 문제가 아니다.
    assert (stages["x1i"]["stage@baseline"], stages["x1i"]["text_rank"]) == (tool.NO_PATH, "35")
    assert stages["x1i"]["genre_scale_effect"] == "어느 배율에서도 후보 밖"
    assert {r["analysis_note"] for r in stages.values()} == {tool.RAW_UNAVAILABLE}, "원본 파일을 주지 않았다"

    scales = {(r["query_id"], r["genre_match_scale"]): r for r in _read(out / "diagnosis.csv")}
    assert (scales[("x1p", "1.0")]["pool_genre_boosted"], scales[("x1p", "0.0")]["pool_genre_boosted"]) == ("30", "0")
    assert scales[("x1p", "0.0")]["pool_genre_matched"] == "19", "가산이 없어도 등록 장르가 맞는 후보 수는 사실대로 센다"
    assert scales[("x1r", "1.0")]["target_genre_delta"] == str(round(1 / 61, 6))
    assert not any(qid == "xb1" for qid, _ in scales), "넓은 질의는 단일 정답 표에 넣지 않는다"

    broad = {(r["genre_match_scale"], r["depth"]): r for r in _read(out / "broad_fit.csv")}
    assert (broad[("1.0", "10")]["registered_genre_matches_stated"], broad[("1.0", "10")]["distinct_first_genres"]) == ("10", "1")
    # 가산을 끄면 앞 10곡이 한 장르로만 채워지지 않는다(그룹 곡 9곡 가운데 R&B 1곡 + 텍스트 1위의 댄스 곡).
    assert (broad[("0.0", "10")]["registered_genre_matches_stated"], broad[("0.0", "10")]["distinct_first_genres"]) == ("8", "3")
    assert (broad[("1.0", "10")]["structural_fit"], broad[("0.0", "10")]["structural_fit"]) == ("10", "9")

    sweep = {(r["variant"], r["genre_match_scale"]): r for r in _read(out / "sweep_summary.csv")}
    assert (sweep[("perceived", "1.0")]["in_pool"], sweep[("perceived", "0.0")]["in_pool"]) == ("0", "1")


def test_answers_runs_end_to_end(tmp_path, monkeypatch) -> None:
    import os

    queries, cache_path, router = _bench(tmp_path, monkeypatch)
    out = tmp_path / "answers"
    monkeypatch.delenv("GENRE_MATCH_SCALE", raising=False)
    monkeypatch.delenv("GENRE_ANSWER_SCALE", raising=False)
    try:
        tool.main(["answers", "--input", str(queries), "--analysis-cache", str(cache_path), "--scales", "1,0",
                   "--output-dir", str(out)])
    finally:
        router.shutdown()
    assert "GENRE_MATCH_SCALE" not in os.environ and "GENRE_ANSWER_SCALE" not in os.environ
    rows = {(r["query_id"], r["answer_kind"], r["genre_answer_scale"]): r for r in _read(out / "answers.csv")}
    assert rows[("x1r", "(개입 없음)", "")]["intervention"] == "0", "처음부터 10위 안이면 재질문으로 가지 않는다"
    # 장르를 말하지 않은 질의: 정답은 텍스트 35위 → 앞 10곡을 거절하면 후보 25위로 들어온다.
    assert rows[("x1i", "reject_only", "")]["pool_rank_after"] == "25"
    # 맞는 답(등록 장르)은 정답만 올린다.
    assert rows[("x1i", "genre_oracle", "1.0")]["pool_rank_after"] == "1"
    assert rows[("x1i", "genre_oracle", "0.0")]["pool_rank_after"] == "25", "가산을 끄면 맞는 답의 이득도 사라진다"
    # 체감 장르로 답하면 등록 장르가 발라드인 곡들이 ×3 가산을 받아 정답이 후보 끝으로 밀린다. 끄면 거절만 한 것과 같다.
    assert rows[("x1i", "genre_perceived", "1.0")]["pool_rank_after"] == "30"
    assert rows[("x1i", "genre_perceived", "1.0")]["pool_answer_matched"] == "29"
    assert rows[("x1i", "genre_perceived", "0.0")]["pool_rank_after"] == "25"
    assert rows[("x1i", "genre_perceived", "0.0")]["target_answer_delta"] == "0"
    summary = {(r["variant"], r["answer_kind"], r["genre_answer_scale"]): r for r in _read(out / "answers_summary.csv")}
    assert summary[("identity_sound", "genre_oracle", "1.0")]["in_top_k"] == "1"
    assert summary[("identity_sound", "genre_perceived", "1.0")]["in_top_k"] == "0"


def test_report_merges_per_set_folders_into_one_run_and_refuses_mixed_conditions(tmp_path) -> None:
    """세트마다 따로 잰 폴더를 한 실행으로 합친다. 폴더끼리 적용 배율이 다르면 한 실행이 아니다."""
    first = _eval_dir(tmp_path / "a" / "set1", {"g01r": (2, 1, "applied")}, 1.0)
    second = _eval_dir(tmp_path / "a" / "set2", {"g01p": (8, 3, "applied")}, 1.0)
    out = tmp_path / "report"
    tool.main(["report", "--queries", str(QUERIES), "--runs", f"base={first},{second}", "--output-dir", str(out)])
    with (out / "rerank_summary.csv").open(encoding="utf-8") as f:
        total = next(r for r in csv.DictReader(f) if r["variant"] == "(전체)")
    assert (total["n"], total["hit@10"], total["genre_match_scale"]) == ("2", "2", "1.0")

    other = _eval_dir(tmp_path / "b" / "set2", {"g01p": (8, 3, "applied")}, 0.0)
    with pytest.raises(ValueError):
        tool.main(["report", "--queries", str(QUERIES), "--runs", f"mixed={first},{other}", "--output-dir", str(out)])


def test_scales_are_restored_even_when_a_run_fails(monkeypatch) -> None:
    import os

    monkeypatch.setenv("GENRE_MATCH_SCALE", "0.25")
    monkeypatch.delenv("GENRE_ANSWER_SCALE", raising=False)
    with pytest.raises(RuntimeError):
        with tool._restoring("GENRE_MATCH_SCALE", "GENRE_ANSWER_SCALE"):
            tool._set_scale("GENRE_MATCH_SCALE", 0.0)
            tool._set_scale("GENRE_ANSWER_SCALE", 0.5)
            raise RuntimeError("검색이 중간에 죽었다")
    assert os.environ["GENRE_MATCH_SCALE"] == "0.25" and "GENRE_ANSWER_SCALE" not in os.environ


def test_trace_blank_genre_drops_only_the_genre_field(tmp_path, monkeypatch) -> None:
    """--blank-genre는 분석의 genre 칸만 비운다. 질의·가수 형태 단서는 그대로라 보조 신호는 여전히 정답을 올린다."""
    queries, cache_path, router = _bench(tmp_path, monkeypatch)
    out = tmp_path / "blank"
    try:
        tool.main(["trace", "--input", str(queries), "--analysis-cache", str(cache_path), "--scales", "1",
                   "--blank-genre", "--output-dir", str(out)])
    finally:
        router.shutdown()
    rows = {r["query_id"]: r for r in _read(out / "diagnosis.csv")}
    assert (rows["x1p"]["analysis_genre"], rows["x1p"]["blanked_genre"]) == ("", "발라드")
    assert (rows["x1p"]["analysis_tags"], rows["x1p"]["analysis_artist_type"]) == ("발라드|여자그룹", "그룹"), (
        "genre 칸 밖의 분석(태그에 든 장르 낱말 · 가수 형태)은 그대로여야 한다"
    )
    assert (rows["x1p"]["pool_rank"], rows["x1p"]["target_paths"]) == ("5", "metadata:5"), "배율 1인데 장르 가산이 없다"
    assert rows["x1p"]["pool_genre_boosted"] == "0" and rows["x1p"]["target_adjustments"].startswith("artist_type:")
    assert rows["x1i"]["blanked_genre"] == "", "비울 장르가 없던 질의"
    stage = {r["query_id"]: r for r in _read(out / "stage_summary.csv")}
    assert stage["x1p"]["analysis_note"] == "진단용으로 genre를 비움(원래 '발라드')"
    assert json.loads((out / "runinfo.json").read_text(encoding="utf-8"))["genre_gap"]["blank_genre"] is True


def test_trace_refuses_rule_fallback_analyses_before_loading_anything(tmp_path, monkeypatch) -> None:
    """규칙 폴백(Gemini 실패)으로 저장된 분석은 모델 분석이 아니다 — 섞인 채로 재지 않는다."""
    from src.retrieval.analysis_cache import AnalysisCacheError, new_cache
    from src.retrieval.query_analyzer import rule_fallback

    def boom():
        raise AssertionError("캐시 검사 전에 검색기를 만들면 안 된다")

    monkeypatch.setattr("src.backend.api.dependencies.get_search_router", boom)
    queries = _query_file(tmp_path / "q.csv", {"a": "잔잔한 노래 찾아줘"})
    cache = new_cache(source=str(queries), split="dev")
    cache.put("a", "잔잔한 노래 찾아줘", rule_fallback("잔잔한 노래 찾아줘"))
    cache_path = tmp_path / "cache.json"
    cache.save(cache_path)
    with pytest.raises(AnalysisCacheError):
        tool.main(["trace", "--input", str(queries), "--analysis-cache", str(cache_path), "--output-dir", str(tmp_path / "o")])


def test_trace_records_the_rank_in_the_vocal_ensemble_search_when_that_search_runs(tmp_path, monkeypatch) -> None:
    """보컬·편성 단서가 둘 이상이면 전용 보조 검색이 돈다. 정답이 그 검색의 몇 위였는지를 깊게 받아 남긴다 —
    서비스는 상위 30곡만 읽으므로, 31위 밖이면 주 경로 순위가 같아도 후보에 못 든다."""
    from src.backend.schemas.query import PerformanceClues
    from src.retrieval.analysis_cache import new_cache
    from tests.test_search_explain import _analysis

    queries, _, router = _bench(tmp_path, monkeypatch)
    cache = new_cache(source=str(queries), split="dev")
    clues = PerformanceClues(vocal_count="multiple", vocal_roles=["여성노래"], confidence=0.9)
    for row in tool.read_rows(queries):
        fields = {"performance_clues": clues} if row["query_id"] == "x1i" else {}
        cache.put(row["query_id"], row["query"], _analysis(original_query=row["query"], **fields))
    cache_path = tmp_path / "cache_perf.json"
    cache.save(cache_path)
    out = tmp_path / "perf"
    try:
        tool.main(["trace", "--input", str(queries), "--analysis-cache", str(cache_path), "--scales", "1",
                   "--deep-k", "100", "--output-dir", str(out)])
    finally:
        router.shutdown()
    rows = {r["query_id"]: r for r in _read(out / "diagnosis.csv")}
    assert rows["x1i"]["performance_search_rank"] == "35", "대역 검색은 텍스트 순서를 그대로 돌려준다 — 정답은 35번째"
    assert rows["x1r"]["performance_search_rank"] == "-", "전용 검색이 돌지 않은 질의는 순위가 없다(빈칸과 구분)"
