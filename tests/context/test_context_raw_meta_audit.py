"""The reviewer audit must separate valid optional nulls from broken metadata."""

import json

from experiments.namuwiki.audit_raw_namuwiki_meta import audit


def test_raw_meta_audit_counts_valid_optional_nulls_and_invalid_required_fields(tmp_path):
    values = [
        {"schema_version": "namuwiki_v3", "status": "not_found", "facts": [],
         "source_url": None, "error_code": None},
        {"schema_version": "namuwiki_v3", "status": "ok",
         "source_url": "https://namu.wiki/w/example", "error_code": None,
         "facts": [{"category": "media_usage", "section": "여담", "text": "예시 배경음악이다."}]},
        {"schema_version": "namuwiki_v3", "status": None, "facts": []},
        {"schema_version": "namuwiki_v3", "status": "unknown", "facts": []},
    ]
    for index, value in enumerate(values):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "meta.json").write_text(
            json.dumps({"namuwiki": value}, ensure_ascii=False), encoding="utf-8",
        )
    (tmp_path / "missing").mkdir()

    report = audit(tmp_path)
    assert report["raw_song_directories"] == 5
    assert report["schema_valid_namuwiki_objects"] == 2
    assert report["issue_counts"]["schema_invalid"] == 2
    assert report["issue_counts"]["required_status_missing_or_null"] == 1
    assert report["issue_counts"]["required_status_unknown"] == 1
    assert report["issue_counts"]["missing_or_unsafe_meta"] == 1
    assert "source_url" not in str(report["issue_counts"])
    assert "예시 배경음악" not in str(report)
