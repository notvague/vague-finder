"""Final-generation read-back guards (with a fake active Qdrant)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from experiments.namuwiki.verify_context_step9 import verify


class FakeQdrant:
    def __init__(self, *args, **kwargs):
        self.client = self
        self.closed = False

    def get_aliases(self):
        return SimpleNamespace(aliases=[
            SimpleNamespace(alias_name="context_dense", collection_name="dense_v2"),
            SimpleNamespace(alias_name="context_sparse", collection_name="sparse_v2"),
        ])

    def count(self, collection_name, exact=False):
        assert exact
        return SimpleNamespace(count={"dense_v2": 5, "sparse_v2": 2}[collection_name])

    def close(self):
        self.closed = True


def _write(tmp_path, *, pending=0, sparse_alias="context_sparse"):
    context = tmp_path / "context"
    context.mkdir()
    (context / "manifest.json").write_text(json.dumps({
        "coverage": {"scope_total": 3, "artifact_ready": 3 - pending,
                     "pending": pending, "coverage_error_count": 0},
        "song_count": 3 - pending, "record_count": 5,
        "sparse_profile_count": 2, "invalid_artifact_count": 0,
        "orphan_artifact_count": 0, "status_counts": {"ok": 2, "not_found": 1},
    }), encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "status": "ok", "source_scope_complete": True, "source_song_count": 3,
        "dense_record_count": 5, "sparse_profile_count": 2, "context_build_id": "valid",
        "qdrant": {"dense_alias": "context_dense", "sparse_alias": sparse_alias,
                   "dense_collection": "dense_v2", "sparse_collection": "sparse_v2"},
    }), encoding="utf-8")
    return context, state


def test_read_back_verifies_active_aliases_and_exact_counts(tmp_path):
    context, state = _write(tmp_path)
    with patch("experiments.namuwiki.verify_context_step9.QdrantVectorClient", FakeQdrant):
        report = verify(context_dir=context, state_manifest=state)
    assert (report["status"], report["dense_record_count"], report["sparse_profile_count"]) == (
        "verified", 5, 2,
    )


def test_never_open_qdrant_when_context_is_incomplete(tmp_path):
    context, state = _write(tmp_path, pending=1)
    with patch("experiments.namuwiki.verify_context_step9.QdrantVectorClient") as client:
        with pytest.raises(ValueError, match="incomplete"):
            verify(context_dir=context, state_manifest=state)
        client.assert_not_called()


def test_wrong_active_alias_is_rejected(tmp_path):
    context, state = _write(tmp_path, sparse_alias="old_sparse")
    with patch("experiments.namuwiki.verify_context_step9.QdrantVectorClient", FakeQdrant):
        with pytest.raises(ValueError, match="active sparse alias"):
            verify(context_dir=context, state_manifest=state)
