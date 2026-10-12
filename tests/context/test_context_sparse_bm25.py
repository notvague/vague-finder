from __future__ import annotations

import json
import hashlib
import math
import zlib
from pathlib import Path

import numpy as np
import pytest

from src.embedding.context_artifacts import ContextArtifactStore
from src.embedding.context_sparse import (
    build_context_bm25,
    iter_context_sparse_embeddings,
    load_context_bm25_query_encoder,
    load_context_sparse_input,
)
from src.embedding.models.context_bm25 import ContextBM25Encoder


def _hash(term: str) -> int:
    """Dependency-free deterministic hash used only by unit tests."""
    return zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF


def _song(song_id: str, clue: str, *, status: str = "ok") -> dict:
    facts = []
    if status == "ok":
        facts = [{
            "category": "media_usage" if song_id == "101" else "meme",
            "section": "여담 > 삽입곡" if song_id == "101" else "여담 > 밈",
            "text": clue,
        }]
    return {
        "id": song_id,
        "metadata": {"title": f"테스트곡 {song_id}", "artist": [f"가수 {song_id}"]},
        "namuwiki": {
            "schema_version": "namuwiki_v3",
            "status": status,
            "source_url": f"https://namu.wiki/w/test-{song_id}",
            "collected_at": "2026-09-21T00:00:00+00:00",
            "facts": facts,
            "error_code": None,
        },
    }


SONGS = (
    _song("101", "짱구는 못말려 6기 15화의 배경음악으로 사용되었다."),
    _song("202", "ROX Tigers가 리그 오브 레전드 영상에서 패러디했다."),
)


def _publish(root: Path, songs=SONGS, *, pending: int = 0) -> Path:
    store = ContextArtifactStore(root)
    record_count = 0
    status_counts: dict[str, int] = {}
    with store.writer():
        for song in songs:
            result = store.sync_song(song)
            record_count += int(result["record_count"])
            status = str(result["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        store.publish_manifest(
            coverage={
                "scope_total": len(songs) + pending,
                "terminal_meta": len(songs),
                "artifact_ready": len(songs),
                "pending": pending,
                "completion_percent": round(
                    len(songs) / (len(songs) + pending) * 100.0, 1
                ),
                "context_record_count": record_count,
                "status_counts": status_counts,
                "coverage_error_count": 0,
                "coverage_errors": [],
            },
            active_song_ids={str(song["id"]) for song in songs},
        )
    return root


def _score(query: dict, document: dict) -> float:
    query_map = dict(zip(query["indices"], query["values"]))
    return sum(
        float(value) * query_map.get(int(index), 0.0)
        for index, value in zip(document["indices"], document["values"])
    )


def test_partial_scope_allows_pilot_but_final_gate_refuses(tmp_path: Path):
    context = _publish(tmp_path / "context", pending=3)
    output = tmp_path / "sparse"

    result = build_context_bm25(
        context_dir=context,
        output_dir=output,
        dry_run=True,
        hash_fn=_hash,
    )

    assert result["status"] == "dry_run"
    assert result["source_scope_complete"] is False
    assert result["source_profile_count"] == 2
    assert result["documents_to_fit"] == 2
    assert not output.exists()

    with pytest.raises(ValueError, match="catalogue is incomplete"):
        load_context_sparse_input(context, require_complete_scope=True)


def test_fit_publishes_one_corpus_and_yields_song_vectors(tmp_path: Path):
    context = _publish(tmp_path / "context")
    output = tmp_path / "sparse"

    result = build_context_bm25(
        context_dir=context,
        output_dir=output,
        require_complete_scope=True,
        hash_fn=_hash,
    )

    assert result["action"] == "build"
    assert result["fitted_document_count"] == 2
    assert result["encoded_document_count"] == 2
    assert result["cache_rejection_count"] == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_count"] == 2
    assert manifest["complete_for_source_manifest"] is True
    assert (output / manifest["params_ref"]).is_file()
    assert (output / manifest["documents_ref"]).is_file()

    rows = list(iter_context_sparse_embeddings(
        context_dir=context,
        sparse_dir=output,
    ))
    assert [row["song_id"] for row in rows] == ["101", "202"]
    assert all(row["profile_id"].startswith("nws:") for row in rows)
    assert all(row["sparse_values"]["indices"] for row in rows)
    assert all(all(value > 0 for value in row["sparse_values"]["values"]) for row in rows)

    query_encoder = load_context_bm25_query_encoder(output, hash_fn=_hash)
    query = query_encoder.encode("짱구는 못말려 OST 배경음악")
    scores = [_score(query, row["sparse_values"]) for row in rows]
    assert rows[int(np.argmax(scores))]["song_id"] == "101"
    assert "삽입곡" in query_encoder.tokenize("애니메이션 OST")


def test_second_run_reuses_the_whole_valid_corpus(tmp_path: Path):
    context = _publish(tmp_path / "context")
    output = tmp_path / "sparse"
    build_context_bm25(context_dir=context, output_dir=output, hash_fn=_hash)

    def forbidden(_term: str) -> int:
        raise AssertionError("a valid published corpus must not be refit or rehashed")

    result = build_context_bm25(
        context_dir=context,
        output_dir=output,
        hash_fn=forbidden,
    )

    assert result["action"] == "reuse"
    assert result["documents_to_fit"] == 0
    assert result["reused_corpus"] is True


def test_same_profiles_republished_context_manifest_refreshes_sparse_publication(tmp_path: Path):
    context = _publish(tmp_path / "context")
    output = tmp_path / "sparse"
    build_context_bm25(context_dir=context, output_dir=output, hash_fn=_hash)
    first = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    # A no-op artifact batch writes a fresh context manifest. The profile
    # vocabulary and vectors have not changed, but Qdrant checks its file hash.
    _publish(context)
    current_hash = hashlib.sha256((context / "manifest.json").read_bytes()).hexdigest()
    assert first["source_manifest_sha256"] != current_hash

    dry = build_context_bm25(
        context_dir=context, output_dir=output, dry_run=True, hash_fn=_hash,
    )
    assert dry["action"] == "build"
    assert dry["documents_to_fit"] == len(SONGS)
    assert "another context manifest" in dry["cache_rejection_reason"]

    rebuilt = build_context_bm25(context_dir=context, output_dir=output, hash_fn=_hash)
    assert rebuilt["reused_corpus"] is False
    final = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert final["source_manifest_sha256"] == current_hash


def test_any_corpus_change_refits_all_current_profiles(tmp_path: Path):
    context = _publish(tmp_path / "context", songs=SONGS[:1])
    output = tmp_path / "sparse"
    build_context_bm25(context_dir=context, output_dir=output, hash_fn=_hash)
    _publish(context, songs=SONGS)

    result = build_context_bm25(
        context_dir=context,
        output_dir=output,
        hash_fn=_hash,
    )

    assert result["action"] == "build"
    assert result["documents_to_fit"] == 2
    assert result["fitted_document_count"] == 2
    assert result["cache_rejection_count"] == 1


def test_catalogue_scope_change_cannot_reuse_a_pilot_publication(tmp_path: Path):
    context = _publish(tmp_path / "context", pending=3)
    output = tmp_path / "sparse"
    first = build_context_bm25(
        context_dir=context,
        output_dir=output,
        hash_fn=_hash,
    )
    assert first["source_scope_complete"] is False

    # Searchable profiles are unchanged, but this exact scope is now final.
    # Refresh publication metadata instead of retaining a pilot manifest.
    _publish(context, pending=0)
    final = build_context_bm25(
        context_dir=context,
        output_dir=output,
        require_complete_scope=True,
        hash_fn=_hash,
    )

    assert final["action"] == "build"
    assert final["source_scope_complete"] is True
    assert final["cache_rejection_count"] == 1
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_scope_complete"] is True


def test_corrupt_bundle_is_rebuilt_before_manifest_publish(tmp_path: Path):
    context = _publish(tmp_path / "context")
    output = tmp_path / "sparse"
    build_context_bm25(context_dir=context, output_dir=output, hash_fn=_hash)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    (output / manifest["documents_ref"]).write_bytes(b"not-an-npz")

    result = build_context_bm25(
        context_dir=context,
        output_dir=output,
        hash_fn=_hash,
    )

    assert result["action"] == "build"
    assert result["cache_rejection_count"] == 1
    assert len(list(iter_context_sparse_embeddings(
        context_dir=context,
        sparse_dir=output,
    ))) == 2


def test_tampered_source_artifact_is_rejected_before_output(tmp_path: Path):
    context = _publish(tmp_path / "context")
    output = tmp_path / "sparse"
    path = context / "songs/101.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["retrieval"]["sparse_profile"]["terms"].append("변조")
    artifact["retrieval"]["sparse_profile"]["term_count"] += 1
    path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError):
        build_context_bm25(
            context_dir=context,
            output_dir=output,
            dry_run=True,
            hash_fn=_hash,
        )
    assert not output.exists()


def test_empty_terminal_context_publishes_without_loading_hash_backend(tmp_path: Path):
    songs = (_song("303", "", status="no_trivia"),)
    context = _publish(tmp_path / "context", songs=songs)
    output = tmp_path / "sparse"

    def forbidden(_term: str) -> int:
        raise AssertionError("an empty corpus must not initialize the hash backend")

    result = build_context_bm25(
        context_dir=context,
        output_dir=output,
        require_complete_scope=True,
        hash_fn=forbidden,
    )

    assert result["action"] == "empty"
    assert result["source_profile_count"] == 0
    assert list(iter_context_sparse_embeddings(
        context_dir=context,
        sparse_dir=output,
    )) == []


def test_common_term_is_outweighed_by_a_distinctive_context_term():
    encoder = ContextBM25Encoder(hash_fn=_hash).fit([
        ["한국", "짱구는_못말려"],
        ["한국", "rox_tigers"],
        ["한국", "2옥타브"],
    ])
    documents = encoder.encode_documents([
        ["한국", "짱구는_못말려"],
        ["한국", "rox_tigers"],
        ["한국", "2옥타브"],
    ])
    query = encoder.encode_query_terms(["한국", "rox_tigers"])

    scores = [_score(query, document) for document in documents]
    assert int(np.argmax(scores)) == 1
    assert scores[1] > scores[0]


def test_exact_compound_tokens_are_not_split_or_retokenized():
    encoder = ContextBM25Encoder(hash_fn=_hash).fit([
        ["d-e-f#m", "짱구는_못말려"],
        ["일반", "문서"],
    ])
    vector = encoder.encode_document_terms(["d-e-f#m", "짱구는_못말려"])

    assert sorted(vector["indices"]) == sorted([
        _hash("d-e-f#m"),
        _hash("짱구는_못말려"),
    ])


def test_document_and_query_weights_follow_the_published_bm25_contract():
    documents = [
        ["고유단서", "고유단서", "공통"],
        ["공통", "다른단서"],
    ]
    encoder = ContextBM25Encoder(b=0.75, k1=1.2, hash_fn=_hash).fit(documents)

    document = encoder.encode_document_terms(documents[0])
    document_weights = dict(zip(document["indices"], document["values"]))
    average_length = 2.5
    denominator_base = 1.2 * (
        1.0 - 0.75 + 0.75 * (len(documents[0]) / average_length)
    )
    assert document_weights[_hash("고유단서")] == pytest.approx(
        2.0 / (denominator_base + 2.0)
    )
    assert document_weights[_hash("공통")] == pytest.approx(
        1.0 / (denominator_base + 1.0)
    )

    query = encoder.encode_query_terms(["고유단서", "공통"])
    query_weights = dict(zip(query["indices"], query["values"]))
    unique_idf = math.log((2.0 + 1.0) / (1.0 + 0.5))
    common_idf = math.log((2.0 + 1.0) / (2.0 + 0.5))
    total_idf = unique_idf + common_idf
    assert query_weights[_hash("고유단서")] == pytest.approx(unique_idf / total_idf)
    assert query_weights[_hash("공통")] == pytest.approx(common_idf / total_idf)

    restored = ContextBM25Encoder.from_params(
        encoder.get_params(),
        hash_fn=_hash,
    )
    assert restored.encode_document_terms(documents[0]) == pytest.approx(document)
    assert restored.encode_query_terms(["고유단서", "공통"]) == pytest.approx(query)
