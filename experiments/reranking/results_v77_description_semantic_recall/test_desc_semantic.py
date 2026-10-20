"""v77 실험 규칙 계약 시험(합성 자료 — 품질 증거 아님)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import desc_semantic as ds  # noqa: E402


def test_normalize_description_missing_and_whitespace():
    assert ds.normalize_description(None) is None
    assert ds.normalize_description(123) is None
    assert ds.normalize_description("  \n\t ") is None
    assert ds.normalize_description(" 가\n\n나  다 ") == "가 나 다"


def test_group_id_same_text_same_group():
    assert ds.group_id("같은 소개") == ds.group_id("같은 소개")
    assert ds.group_id("같은 소개") != ds.group_id("다른 소개")
    assert len(ds.group_id("x")) == 16


def test_split_chunks_packs_sentences_under_limit():
    text = "첫 문장이다. 둘째 문장이다! 셋째 문장인가? 넷째."
    chunks, forced = ds.split_chunks(text, limit=20)
    assert forced == 0
    assert all(len(c) <= 20 for c in chunks)
    assert " ".join(chunks) == text


def test_split_chunks_hard_split_long_sentence_at_space():
    long = " ".join(f"곡{i:03d}" for i in range(30))  # 마침표 없는 긴 나열(149자, 조각마다 글자가 다름)
    chunks, forced = ds.split_chunks(long, limit=50)
    assert forced == 1
    assert all(0 < len(c) <= 50 for c in chunks)
    assert " ".join(chunks) == long          # 공백에서 끊었으므로 이어 붙이면 원문


def test_split_chunks_hard_split_without_space():
    text = "".join(chr(0xAC00 + i) for i in range(95))  # 공백 없는 95자(반복 없음)
    chunks, forced = ds.split_chunks(text, limit=40)
    assert forced == 1
    assert [len(c) for c in chunks] == [40, 40, 15]
    assert "".join(chunks) == text


def test_split_chunks_same_piece_within_description_kept_once():
    chunks, _ = ds.split_chunks("가" * 80, limit=40)   # 같은 40자 조각 둘 → 첫 것만(§2-2)
    assert chunks == ["가" * 40]


def test_split_chunks_drops_duplicate_and_empty():
    chunks, _ = ds.split_chunks("반복. 반복. 반복.", limit=3)
    assert chunks == ["반복."]
    assert ds.split_chunks("", limit=10) == ([], 0)


def test_rank_groups_max_and_ties():
    groups = ["b", "a", "a", "c", "b"]
    idx = [0, 0, 1, 0, 1]
    scores = [0.5, 0.2, 0.5, 0.1, 0.5]
    ranked = ds.rank_groups(groups, idx, scores, depth=10)
    # a·b 동점 0.5 → 묶음 id 순, b의 근거 청크는 번호가 작은 0
    assert ranked[0] == ("a", 0.5, 1)
    assert ranked[1] == ("b", 0.5, 0)
    assert ranked[2][0] == "c"
    assert len(ds.rank_groups(groups, idx, scores, depth=2)) == 2


def test_rank_groups_require_positive_drops_zero():
    ranked = ds.rank_groups(["a", "b"], [0, 0], [0.0, 1.0], depth=10, require_positive=True)
    assert [g for g, _, _ in ranked] == ["b"]


def _songs():
    return {
        "10": ds.Song("10", "봄날 (Spring Day)", "g"),
        "2": ds.Song("2", "겨울", "g"),
        "3": ds.Song("3", "A", "g"),        # 1글자 제목 — 일치로 치지 않는다
    }


def test_representative_title_unique():
    group = ds.Group("g", ["2", "3", "10"])
    sid, rule = ds.pick_representative(group, "타이틀곡 '봄날'은 ...", _songs(), {"2": 0.9, "10": 0.1})
    assert (sid, rule) == ("10", "title_unique")


def test_representative_title_tie_uses_dense():
    group = ds.Group("g", ["2", "3", "10"])
    sid, rule = ds.pick_representative(group, "봄날 그리고 겨울", _songs(), {"2": 0.2, "10": 0.3})
    assert (sid, rule) == ("10", "title_tie_dense")


def test_representative_no_title_dense_then_song_id():
    group = ds.Group("g", ["2", "3", "10"])
    sid, rule = ds.pick_representative(group, "A 앨범 소개", _songs(), {"2": 0.5, "3": 0.5, "10": 0.1})
    assert (sid, rule) == ("2", "dense")   # 동점 0.5 → song_id 2 < 3 (숫자 순)


def test_representative_missing_dense_excluded():
    group = ds.Group("g", ["2"])
    assert ds.pick_representative(group, "소개", _songs(), {}) == (None, "none")


def test_bm25_scores_only_matching_docs():
    bm = ds.BM25([["오디션", "프로그램"], ["사랑"], []])
    sc = bm.scores(["오디션"])
    assert sc[0] > 0 and sc[1] == 0 and sc[2] == 0
