from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.embedding.context_artifacts import ContextArtifactStore
from src.embedding.context_dense import (
    embed_context_dense,
    iter_context_dense_embeddings,
    load_context_dense_input,
)
from src.embedding.models.text_koe5 import KoE5Embedder


class FakeEmbedder:
    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls = 0
        self.passages: list[str] = []

    def embed_passages(self, passages, **_kwargs):
        self.calls += 1
        self.passages.extend(passages)
        rows = []
        for index, text in enumerate(passages):
            base = float(sum(text.encode("utf-8")) % 97 + index + 1)
            vector = np.arange(base, base + self.dim, dtype=np.float32)
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.stack(rows).astype(np.float32)


class ForbiddenEmbedder:
    def embed_passages(self, *_args, **_kwargs):
        raise AssertionError("embedder must not be called")


def _song(song_id: str, *, status: str = "ok") -> dict:
    namuwiki = {
        "schema_version": "namuwiki_v3",
        "status": status,
        "source_url": f"https://namu.wiki/w/test-{song_id}",
        "collected_at": "2026-09-20T00:00:00+00:00",
        "facts": [],
        "error_code": None,
    }
    if status == "ok":
        namuwiki["facts"] = [
            {
                "category": "media_usage",
                "section": "여담 > 삽입곡",
                "text": f"테스트 프로그램 {song_id}화의 배경음악으로 사용되었다.",
            },
            {
                "category": "production",
                "section": "여담",
                "text": f"프로듀서 {song_id}이 데모를 다시 녹음해 완성했다.",
            },
        ]
    return {
        "id": song_id,
        "metadata": {"title": f"테스트곡 {song_id}", "artist": [f"가수 {song_id}"]},
        "namuwiki": namuwiki,
    }


def _context(tmp_path: Path, *, pending: int = 0, status: str = "ok") -> Path:
    root = tmp_path / "context"
    store = ContextArtifactStore(root)
    song = _song("101", status=status)
    with store.writer():
        store.sync_song(song)
        store.publish_manifest(
            coverage={
                "scope_total": 1 + pending,
                "terminal_meta": 1,
                "artifact_ready": 1,
                "pending": pending,
                "completion_percent": round(100 / (1 + pending), 1),
                "context_record_count": 2 if status == "ok" else 0,
                "status_counts": {status: 1},
                "coverage_error_count": 0,
                "coverage_errors": [],
            },
            active_song_ids={"101"},
        )
    return root


def test_every_dense_fact_has_the_same_compact_song_identity(tmp_path: Path):
    context = _context(tmp_path)
    artifact = json.loads(
        (context / "songs/101.json").read_text(encoding="utf-8")
    )

    assert artifact["artifact_schema_version"] == "context_artifact_v6"
    assert artifact["retrieval"]["record_schema_version"] == "context_record_v6"
    assert (
        artifact["retrieval"]["dense_text_format"]
        == "artist_title_category_fact_v1"
    )
    passages = [
        record["dense_text"] for record in artifact["retrieval"]["records"]
    ]
    assert passages
    assert all(
        passage.startswith("가수 101 - 테스트곡 101 | ")
        for passage in passages
    )
    assert passages[0].startswith(
        "가수 101 - 테스트곡 101 | 매체 사용: "
    )
    assert passages[1].startswith(
        "가수 101 - 테스트곡 101 | 제작 배경: "
    )


def test_previous_artifact_schema_is_marked_for_rebuild(tmp_path: Path):
    context = _context(tmp_path)
    artifact_path = context / "songs/101.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["artifact_schema_version"] = "context_artifact_v5"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
    )

    state = ContextArtifactStore(context).inspect(_song("101"))

    assert state.applicable is True
    assert state.ready is False
    assert state.reason.startswith("artifact_invalid:")


def test_partial_scope_allows_pilot_but_final_gate_refuses(tmp_path: Path):
    context = _context(tmp_path, pending=3)
    output = tmp_path / "dense"

    result = embed_context_dense(
        context_dir=context,
        output_dir=output,
        expected_dim=4,
        dry_run=True,
        embedder=ForbiddenEmbedder(),
    )

    assert result["status"] == "dry_run"
    assert result["source_scope_complete"] is False
    assert result["source_record_count"] == 2
    assert result["records_to_embed"] == 2
    assert not output.exists()

    with pytest.raises(ValueError, match="catalogue is incomplete"):
        load_context_dense_input(context, require_complete_scope=True)


def test_inconsistent_partial_coverage_is_always_rejected(tmp_path: Path):
    context = _context(tmp_path, pending=3)
    manifest_path = context / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"]["context_record_count"] = 999
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="record count"):
        embed_context_dense(
            context_dir=context,
            output_dir=tmp_path / "dense",
            expected_dim=4,
            dry_run=True,
            embedder=ForbiddenEmbedder(),
        )


def test_writes_one_song_bundle_and_yields_fact_vectors(tmp_path: Path):
    context = _context(tmp_path)
    output = tmp_path / "dense"
    fake = FakeEmbedder(dim=4)

    result = embed_context_dense(
        context_dir=context,
        output_dir=output,
        expected_dim=4,
        batch_size=1,
        embedder=fake,
        require_complete_scope=True,
    )

    assert result["embedded_song_count"] == 1
    assert result["source_record_count"] == 2
    assert result["cache_rejection_count"] == 0
    assert fake.calls == 2
    assert all(
        passage.startswith("가수 101 - 테스트곡 101 | ")
        for passage in fake.passages
    )
    assert (output / "songs/101.npz").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete_for_source_manifest"] is True
    assert manifest["record_count"] == 2

    records = list(iter_context_dense_embeddings(
        context_dir=context,
        dense_dir=output,
    ))
    assert len(records) == 2
    assert all(record["song_id"] == "101" for record in records)
    assert all(record["values"].shape == (4,) for record in records)
    assert all(np.isclose(np.linalg.norm(record["values"]), 1.0) for record in records)


def test_second_run_reuses_valid_song_bundle(tmp_path: Path):
    context = _context(tmp_path)
    output = tmp_path / "dense"
    embed_context_dense(
        context_dir=context,
        output_dir=output,
        expected_dim=4,
        embedder=FakeEmbedder(dim=4),
    )

    result = embed_context_dense(
        context_dir=context,
        output_dir=output,
        expected_dim=4,
        embedder=ForbiddenEmbedder(),
    )

    assert result["embedded_song_count"] == 0
    assert result["reused_song_count"] == 1
    assert result["records_to_embed"] == 0


def test_embedding_revision_change_invalidates_song_bundle(tmp_path: Path):
    context = _context(tmp_path)
    output = tmp_path / "dense"
    embed_context_dense(
        context_dir=context,
        output_dir=output,
        model_revision="commit-a",
        expected_dim=4,
        embedder=FakeEmbedder(dim=4),
    )
    fake = FakeEmbedder(dim=4)

    result = embed_context_dense(
        context_dir=context,
        output_dir=output,
        model_revision="commit-b",
        expected_dim=4,
        embedder=fake,
    )

    assert result["embedded_song_count"] == 1
    assert result["cache_rejection_count"] == 1
    assert fake.calls == 1


def test_corrupt_bundle_is_reembedded_before_manifest_publish(tmp_path: Path):
    context = _context(tmp_path)
    output = tmp_path / "dense"
    embed_context_dense(
        context_dir=context,
        output_dir=output,
        expected_dim=4,
        embedder=FakeEmbedder(dim=4),
    )
    (output / "songs/101.npz").write_bytes(b"not-an-npz")
    fake = FakeEmbedder(dim=4)

    result = embed_context_dense(
        context_dir=context,
        output_dir=output,
        expected_dim=4,
        embedder=fake,
    )

    assert result["embedded_song_count"] == 1
    assert result["cache_rejection_count"] == 1
    assert fake.calls == 1
    assert len(list(iter_context_dense_embeddings(
        context_dir=context,
        dense_dir=output,
    ))) == 2


def test_tampered_artifact_is_rejected_before_output(tmp_path: Path):
    context = _context(tmp_path)
    output = tmp_path / "dense"
    path = context / "songs/101.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["retrieval"]["records"][0]["dense_text"] += " 변조"
    path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError):
        embed_context_dense(
            context_dir=context,
            output_dir=output,
            expected_dim=4,
            dry_run=True,
            embedder=ForbiddenEmbedder(),
        )
    assert not output.exists()


def test_terminal_song_without_records_does_not_load_model(tmp_path: Path):
    context = _context(tmp_path, status="no_trivia")
    output = tmp_path / "dense"

    result = embed_context_dense(
        context_dir=context,
        output_dir=output,
        expected_dim=4,
        embedder=ForbiddenEmbedder(),
        require_complete_scope=True,
    )

    assert result["source_record_count"] == 0
    assert result["empty_song_count"] == 1
    assert result["embedded_song_count"] == 0
    assert list(iter_context_dense_embeddings(
        context_dir=context,
        dense_dir=output,
    )) == []


def test_koe5_wrapper_passes_optional_revision(monkeypatch):
    captured = {}

    class SentenceTransformerStub:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformerStub
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedder = KoE5Embedder(model_name="test/model", revision="deadbeef").load()

    assert embedder.model_name == "test/model"
    assert captured == {
        "model_name": "test/model",
        "kwargs": {"revision": "deadbeef"},
    }
