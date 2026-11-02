"""v80 파일럿 공용 코드. 평가 오염 방지와 태그 검증을 고정한다."""
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_lib as lib  # noqa: E402


def _song(**kw):
    base = {"metadata": {"title": "제목", "artist": ["가수"], "album_description": "앨범 소개"},
            "lyrics_data": {"full_lyrics": "가사"}, "comments": {"melon": ["멜론댓글"], "youtube": ["유튜브댓글"]}}
    base.update(kw)
    return base


def test_prompt_takes_only_the_song_so_dev_clues_cannot_leak():
    assert list(inspect.signature(lib.build_prompt).parameters) == ["song"]
    prompt = lib.build_prompt(_song())
    assert all(word in prompt for word in ("제목", "가수", "앨범 소개", "가사", "멜론댓글", "유튜브댓글"))
    assert "query" not in prompt.lower().replace("queries", "")


def test_prompt_is_the_same_regardless_of_group_or_extra_fields():
    assert lib.build_prompt(_song()) == lib.build_prompt(_song(group="F", query="고음 많은 노래", song_id="1"))


def test_prompt_limits_long_material():
    prompt = lib.build_prompt(_song(lyrics_data={"full_lyrics": "가" * 9000}))
    assert prompt.count("가" * 4000) == 1 and "가" * 4001 not in prompt


def test_sample_is_deterministic_disjoint_and_hash_ordered():
    ids = [str(i) for i in range(200)]
    failed = [str(i) for i in range(40)]
    a = lib.choose_sample(ids, failed)
    assert a == lib.choose_sample(list(reversed(ids)), list(reversed(failed)))
    assert len(a["F"]) == 25 and len(a["R"]) == 25 and not set(a["F"]) & set(a["R"])
    assert set(a["F"]) <= set(failed)


def test_clean_tags_rejects_wrong_shapes_and_drops_bad_items():
    with pytest.raises(ValueError):
        lib.clean_tags(["고음"])
    with pytest.raises(ValueError):
        lib.clean_tags({"sound_tags": "고음"})
    tags, dropped = lib.clean_tags({"sound_tags": ["고음", "고음", "비 오는 날", 3, "R&B", "가" * 16]})
    assert tags["sound_tags"] == ["고음", "R&B"] and dropped == 4
    assert tags["mood_tags"] == []


def test_clean_tags_enforces_the_limit():
    tags, dropped = lib.clean_tags({"emotion_tags": [f"감정{i}" for i in range(9)]})
    assert len(tags["emotion_tags"]) == 6 and dropped == 3


@pytest.mark.parametrize("query,tags,expected", [
    ("앨범 커버에 사진 없이 글씨만 크게 박혀 있었어. 되게 심플했는데", ["심플"], []),
    ("표지에 사람 얼굴이 크게 클로즈업되어 있던 곡이야", ["클로즈업"], []),
    ("커버가 어둡고 차가운 느낌이었는데, 노래는 현악기 깔린 남자 발라드였어", ["차가운", "현악기", "발라드"], ["현악기"]),
    ("누런 배경에 크레파스로 뭔가 써져있는표지였고, 노래는 호구 노래로 유명해", ["호구"], ["호구"]),
    ("술 마실 때 듣던 고음 많은 노래", ["고음", "남성보컬"], ["고음"]),
])
def test_evaluation_clues_drop_cover_descriptions_and_attributes(query, tags, expected):
    assert lib.evaluation_clues(query, {"korean_tags": tags}) == expected
