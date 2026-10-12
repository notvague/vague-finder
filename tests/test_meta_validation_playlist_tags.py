"""Optional playlist metadata must not exclude otherwise usable songs."""

import pytest

from src.embedding.fixtures.meta_validation import validate_meta_document


def _sample_song():
    """An invented record: no real song metadata or media enters the test."""
    return {
        "song_id": "1",
        "metadata": {
            "title": "sample song",
            "artist": ["sample artist"],
            "album": "sample album",
            "release_date": "2026.01.01",
            "genre": ["pop"],
            "type": ["solo"],
            "vocal_gender": "female",
        },
        "lyrics_data": {
            "full_lyrics": "sample lyric text",
            "lyrics_highlight": "sample lyric",
            "lyrics_summary": "sample summary",
        },
        "semantic_analysis": {
            "search_style_summary": "quiet evening",
            "mood_tags": ["calm"],
            "time_weather_tags": ["evening"],
            "place_activity_tags": ["walking"],
            "emotion_tags": ["nostalgia"],
            "vibe_tags": ["warm"],
            "relation_context_tags": ["parting"],
            "color_tags": ["blue"],
            "sound_tags": ["piano"],
            "melon_playlist_tags": [],
            "visual_imagery": ["city lights"],
        },
        "community_feedback": {
            "sentiment_summary": "listener reaction",
            "fans_tags": ["favorite"],
            "major_emotion": "sadness",
            "popularity": {"fame": "moderate"},
        },
        "links": {
            "cover_url": "https://example.org/cover.jpg",
            "melon_url": "https://example.org/song",
        },
        "comments": {"melon": ["sample comment"], "youtube": []},
    }


def _issues(record):
    return validate_meta_document(record, require_media_files=False)


def test_empty_playlist_tags_are_valid_without_changing_other_requirements():
    record = _sample_song()
    assert _issues(record) == []
    record["semantic_analysis"]["melon_playlist_tags"] = ["evening playlist"]
    assert _issues(record) == []


@pytest.mark.parametrize(
    ("value", "expected_path"),
    [
        (None, "semantic_analysis.melon_playlist_tags"),
        ("playlist", "semantic_analysis.melon_playlist_tags"),
        ([123], "semantic_analysis.melon_playlist_tags[0]"),
        (["unknown"], "semantic_analysis.melon_playlist_tags[0]"),
    ],
)
def test_playlist_tags_still_reject_bad_data(value, expected_path):
    record = _sample_song()
    record["semantic_analysis"]["melon_playlist_tags"] = value
    assert expected_path in {issue.path for issue in _issues(record)}


def test_missing_playlist_field_still_rejected():
    record = _sample_song()
    del record["semantic_analysis"]["melon_playlist_tags"]
    assert "semantic_analysis.melon_playlist_tags" in {
        issue.path for issue in _issues(record)
    }


def test_unrelated_empty_tag_and_placeholder_still_rejected():
    record = _sample_song()
    record["semantic_analysis"]["mood_tags"] = []
    record["semantic_analysis"]["relation_context_tags"] = ["unknown"]
    paths = {issue.path for issue in _issues(record)}
    assert "semantic_analysis.mood_tags" in paths
    assert "semantic_analysis.relation_context_tags[0]" in paths
