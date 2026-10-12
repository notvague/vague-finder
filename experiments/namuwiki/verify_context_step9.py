"""Read-only verification of the active, complete Context Qdrant generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.vector_db.qdrant_backend import QdrantVectorClient
from src.vector_db.settings import NAMESPACE


def verify(*, context_dir: Path, state_manifest: Path) -> dict:
    context = json.loads((context_dir / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads(state_manifest.read_text(encoding="utf-8"))
    coverage = context["coverage"]
    scope = int(coverage["scope_total"])
    ready = int(coverage["artifact_ready"])
    if (scope != ready or int(coverage["pending"]) != 0
            or int(coverage.get("coverage_error_count", 0)) != 0
            or int(context["song_count"]) != scope
            or int(context["invalid_artifact_count"]) != 0
            or int(context["orphan_artifact_count"]) != 0):
        raise ValueError("Context catalogue is incomplete; active Qdrant must not be declared final")
    if state["status"] != "ok" or state["source_scope_complete"] is not True:
        raise ValueError("Qdrant state manifest is not a complete publication")
    if int(state["source_song_count"]) != scope:
        raise ValueError("Qdrant state manifest song count differs from Context catalogue")
    if (int(state["dense_record_count"]) != int(context["record_count"])
            or int(state["sparse_profile_count"]) != int(context["sparse_profile_count"])):
        raise ValueError("Qdrant state manifest point counts differ from Context catalogue")
    q = state["qdrant"]
    client = QdrantVectorClient()
    try:
        aliases = {alias.alias_name: alias.collection_name for alias in client.client.get_aliases().aliases}
        counts = {}
        for label, expected in (("dense", state["dense_record_count"]),
                                ("sparse", state["sparse_profile_count"])):
            alias = q[f"{label}_alias"]
            collection = q[f"{label}_collection"]
            if aliases.get(alias) != collection:
                raise ValueError(f"active {label} alias does not point to the recorded generation")
            count = int(client.client.count(collection_name=collection, exact=True).count)
            if count != int(expected):
                raise ValueError(f"active {label} collection point count differs from manifest")
            counts[label] = count
    finally:
        client.close()
    return {"status": "verified", "source_scope_complete": True, "source_song_count": scope,
            "context_build_id": state["context_build_id"], "status_counts": context["status_counts"],
            "dense_record_count": counts["dense"], "sparse_profile_count": counts["sparse"],
            "dense_collection": q["dense_collection"], "sparse_collection": q["sparse_collection"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, default=Path("artifacts/context"))
    parser.add_argument("--state-manifest", type=Path,
                        default=Path("artifacts/vector_db/context_qdrant") / NAMESPACE / "manifest.json")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(context_dir=args.context_dir, state_manifest=args.state_manifest),
                         ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
