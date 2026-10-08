"""단계 2 — 평가 세트 통합(SSOT) + dev/test 분할.

여기서 고정하는 것
1. queries.json이 자기 스키마를 통과한다. v0.4에서는 통과하지 못해
   `src/eval/evaluate.py`가 아예 실행되지 않았고, 아무도 그걸 몰랐다.
2. 모든 질의에 split이 있고, 재질문 개입 대상(Top-10 밖)이 한쪽에 몰리지 않는다.
3. 측정용 CSV는 항상 queries.json에서 파생된다 — 라벨이 두 곳에 생기지 않게.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.eval.export_csv import COLUMNS, select, to_rows
from src.eval.loader import cross_validate, load_eval_set
from src.eval.schema import EvalQuery, EvalSet

CORPUS = Path("data/all_songs.jsonl")
RANKS = Path("experiments/reranking/results_v05/search_eval_dev_ranks.csv")


@pytest.fixture(scope="module")
def eval_set() -> EvalSet:
    return load_eval_set()


# ---------------------------------------------------------------------------
# 스키마 — v0.4에서 깨져 있던 회귀
# ---------------------------------------------------------------------------

def test_eval_set_loads(eval_set) -> None:
    """이게 깨지면 src/eval/evaluate.py가 통째로 못 돈다."""
    assert len(eval_set.queries) == 89


def test_labeled_queries_have_positives(eval_set) -> None:
    for q in eval_set.queries:
        if q.label_status == "labeled":
            assert q.positives, f"{q.query_id}: labeled인데 정답이 없다"


def test_no_target_queries_have_no_positives(eval_set) -> None:
    """정답이 없는 것이 설계인 질의에 정답이 붙으면 둘 중 하나가 틀렸다."""
    for q in eval_set.queries:
        if q.label_status == "no_target":
            assert not q.positives, f"{q.query_id}: no_target인데 정답이 있다"


def test_partial_labels_are_still_scorable() -> None:
    """표지 확인 전이라도 오디오·메타 축이 검증된 부분 라벨은 집계에 넣는다.

    질의 세트에서 찾지 않고 **여기서 만든다.** 2026-09-24에 표지 8건을 실물로
    확인해 전부 labeled가 되면서 `pending_cover` 질의가 하나도 남지 않았다.
    그렇다고 이 규칙이 사라진 것은 아니다 — 코퍼스가 늘면 또 생긴다.
    """
    q = EvalQuery(
        query_id="x9", query="표지가 파란 노래", category="scene",
        tier="vague", modality_focus="image", split="dev",
        query_set="modality_v1", label_status="pending_cover", positives=["s1"],
    )
    assert q.is_scorable, "부분 라벨은 집계에 들어가야 한다"
    assert not q.has_verified_label, "확정 라벨로 세면 안 된다"


def test_negatives_require_reason(eval_set) -> None:
    for q in eval_set.queries:
        if q.negatives:
            assert q.negative_reason, f"{q.query_id}: 함정 이유가 없다"


def test_scorable_requires_positives() -> None:
    q = EvalQuery(
        query_id="x1", query="아무 노래", category="mood",
        split="dev", query_set="v04", label_status="no_target",
    )
    assert not q.is_scorable


# ---------------------------------------------------------------------------
# split — 홀드아웃이 실제로 존재하는가
# ---------------------------------------------------------------------------

def test_every_query_has_a_split(eval_set) -> None:
    assert set(eval_set.split_distribution()) == {"dev", "test"}


def test_test_split_is_roughly_30_percent(eval_set) -> None:
    dist = eval_set.split_distribution()
    ratio = dist["test"] / sum(dist.values())
    assert 0.25 <= ratio <= 0.35, f"test 비율 {ratio:.2f}"


def test_recovery_targets_exist_in_both_splits(eval_set) -> None:
    """Top-10 밖 질의가 한쪽 split에만 있으면 재질문 효과를 검증할 수 없다.

    이 8개가 단계 4 질문 선택의 실질 대상이다. test에 0개면 홀드아웃이
    있으나 마나다.
    """
    if not RANKS.exists():
        pytest.skip(f"{RANKS} 없음 — 측정 산출물이 필요한 테스트")
    with open(RANKS, encoding="utf-8-sig") as f:
        outside = {r["query_id"] for r in csv.DictReader(f)
                   if not (r["rerank_rank"] or "").strip()}
    split_of = {q.query_id: q.split for q in eval_set.queries}
    per_split = {s: sum(1 for q in outside if split_of.get(q) == s)
                 for s in ("dev", "test")}
    assert per_split["dev"] > 0 and per_split["test"] > 0, per_split


def test_query_set_provenance_is_preserved(eval_set) -> None:
    """v0.5 기준선 비교는 v04 부분집합으로만 한다 — 그 경계가 유지돼야 한다."""
    dist = eval_set.query_set_distribution()
    assert dist["v04"] == 60
    assert dist["modality_v1"] == 14
    assert dist["clarify_v1"] == 15


# ---------------------------------------------------------------------------
# 라벨 정합성
# ---------------------------------------------------------------------------

def test_all_song_ids_exist_in_corpus(eval_set) -> None:
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} 없음 — data/는 git 추적 대상이 아니다")
    catalog = {}
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                catalog[json.loads(line)["song_id"]] = {}
    errors = cross_validate(eval_set, catalog)
    assert not errors, errors[:5]


def test_misinformation_pair_keeps_the_same_target(eval_set) -> None:
    """q111(정확한 기억)과 q118(성별을 잘못 기억)은 같은 곡을 가리켜야 한다.

    이건 중복 라벨이 아니라 설계다. q118의 '여자가 부르는'이 바로 오정보이고,
    성별 단서가 틀려도 나머지(피아노/한자제목/담백)로 찾아내는지를 본다.
    RUN_INFO.md가 이를 라벨 오류로 적어 두어 실제로 한 번 잘못 고칠 뻔했다.
    """
    q111 = next(q for q in eval_set.queries if q.query_id == "q111")
    q118 = next(q for q in eval_set.queries if q.query_id == "q118")
    assert q118.tier == "misinformation"
    assert q111.positives == q118.positives == ["30461396"]

    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} 없음")
    with open(CORPUS, encoding="utf-8") as f:
        gender = {}
        for line in f:
            if line.strip():
                song = json.loads(line)
                gender[song["song_id"]] = song["metadata"].get("vocal_gender")
    # 질의가 '여자'라고 말하는데 정답은 남성 — 그 어긋남이 이 문항의 전부다.
    assert gender[q118.positives[0]] == "남성"


# ---------------------------------------------------------------------------
# 난도 — 재질문용 질의가 표면 매칭으로 풀려서는 안 된다
# ---------------------------------------------------------------------------

def test_clarify_queries_have_no_revealing_tokens(eval_set) -> None:
    """clarify_v1 질의에 정답을 지목하는 희소 토큰이 있으면 안 된다.

    회귀 이력: 초안 26개가 정답의 sound_tags 토큰을 그대로 썼다
    ("색소폰" DF=5, "타악기" DF=3). v0.6 측정에서 18/18이 Top-10 안에 들어와
    재질문이 개입할 여지가 0이 됐고 전부 폐기했다. 축을 겹치는 것은 난도를
    올리지 않는다 — 정답을 확정한다.
    """
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} 없음 — data/는 git 추적 대상이 아니다")
    from src.eval.query_difficulty import indexed_text, load_corpus, revealing_tokens

    songs = load_corpus(CORPUS)
    texts = [indexed_text(s) for s in songs.values()]

    offenders = {}
    for q in eval_set.by_query_set("clarify_v1"):
        if not q.positives:
            continue
        found = revealing_tokens(q.query, songs[q.positives[0]], texts)
        if found:
            offenders[q.query_id] = found
    assert not offenders, offenders


def test_clarify_set_leans_on_external_context_and_misinformation(eval_set) -> None:
    """어렵다고 확인된 두 유형에 집중했는지.

    v0.6에서 어려웠던 질의는 전부 (a) 코퍼스에 없는 외부 맥락이거나
    (b) 오정보였다. 사운드·제목 구조 유형은 전멸했다.
    """
    clarify = eval_set.by_query_set("clarify_v1")
    misinformation = [q for q in clarify if q.tier == "misinformation"]
    external = [q for q in clarify if q.clue_type == "external_context"]
    assert len(misinformation) >= 5
    assert len(external) >= 8


# ---------------------------------------------------------------------------
# CSV 내보내기 — 라벨이 두 곳에 생기지 않게
# ---------------------------------------------------------------------------

def test_export_excludes_queries_without_positives(eval_set) -> None:
    picked = select(eval_set.queries)
    assert all(q.positives for q in picked)
    assert len(picked) < len(eval_set.queries)


def test_export_can_reproduce_a_single_query_set(eval_set) -> None:
    """세트별 분리 집계가 되는가, 그리고 크기가 조용히 변하지 않는가.

    **v05 기준선(53개)과 다른 수다.** 2026-09-24에 q116·q117의 표지 라벨을
    채우면서 두 질의가 집계 대상이 됐다. v05 CSV는 그대로 얼려 두었으므로
    (v11~v18 숫자가 그 위에서 나왔다) 새 세트는 `eval_queries_v06.csv`다.
    여기 숫자를 고칠 일이 생기면 **어느 기준선이 바뀌는지부터** 적어야 한다.
    """
    v04 = select(eval_set.queries, query_sets=["v04"])
    assert all(q.query_set == "v04" for q in v04)
    assert len(v04) == 55


def test_export_rows_match_runner_columns(eval_set) -> None:
    rows = to_rows(select(eval_set.queries)[:3])
    for row in rows:
        assert list(row) == COLUMNS
        assert row["query_type"] == "search"
        assert row["split"] in ("dev", "test")
        assert "|" in row["relevant_ids"] or row["relevant_ids"]


# ---------------------------------------------------------------------------
# sparse passage — #52에서 넣은 앨범 필드가 다시 빠지지 않게 고정
# ---------------------------------------------------------------------------

def test_sparse_passage_keeps_album_context_but_not_namuwiki():
    """앨범명·앨범 요약·시청자 반응은 sparse에 들어가고, 나무위키는 들어가지 않는다.

    회귀 이력: #52가 넣은 세 필드를 #58이 나무위키를 되돌리며 함께 지웠다.
    그 상태로 벡터를 다시 만들면 dev Hit@10이 0.830 → 0.774로 떨어졌다
    (2026-09-17 A/B 측정). 나무위키 여담은 곡 벡터를 흐리므로 계속 제외한다.
    """
    from src.embedding.text.passage_builder import build_sparse_passage

    song = {
        "metadata": {
            "title": "곡제목", "artist": ["가수"], "album": "앨범제목",
            "album_summary": "앨범요약문장", "genre": ["발라드"],
        },
        "community_feedback": {"sentiment_summary": "시청자반응문장"},
        "external_context": {
            "namuwiki": {
                "context_usable": "yes",
                "context_tags": ["나무위키태그"],
                "fact_summary": "나무위키여담문장",
            }
        },
    }
    passage = build_sparse_passage(song)

    assert "앨범제목" in passage
    assert "앨범요약문장" in passage
    assert "시청자반응문장" in passage
    assert "나무위키여담문장" not in passage
    assert "나무위키태그" not in passage


# ---------------------------------------------------------------------------
# 범주형 표시 — 표지 질의 진단(results_v25_cover_diag)에서 정한 8건
# ---------------------------------------------------------------------------

def test_categorical_queries_are_the_general_cover_queries(eval_set) -> None:
    """질의만으로 원래 타깃을 특정할 수 없는 일반 속성 표지 질의. 바꾸면 범주형 집계 숫자가 바뀐다."""
    categorical = {q.query_id for q in eval_set.queries if q.target_scope == "categorical"}
    assert categorical == {"q116", "q117", "m301", "m302", "m303", "m401", "m402", "m403"}
    assert all(q.is_scorable for q in eval_set.queries if q.query_id in categorical)


def test_target_scope_defaults_to_specific() -> None:
    q = EvalQuery(query_id="x", query="질의", category="mixed", split="dev", query_set="v04", positives=["1"])
    assert q.target_scope == "specific"
    with pytest.raises(ValueError):
        EvalQuery(query_id="x", query="질의", category="mixed", split="dev", query_set="v04",
                  positives=["1"], target_scope="vague")
