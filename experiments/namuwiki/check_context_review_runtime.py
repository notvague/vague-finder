"""Check effective production settings and the active complete Context generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval.context_search_settings import assert_fixed_context_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(override=False)
    from src.backend.api.dependencies import close_vector_client, get_search_router
    result = dict(status="running")
    try:
        router = get_search_router()
        result["router_arguments"] = assert_fixed_context_settings(router)
        context = router._context_search
        snapshot = context._snapshot()
        if snapshot is None:
            raise ValueError("Context collections are not published")
        state = json.loads(context.state_manifest.read_text(encoding="utf-8-sig"))
        if state.get("source_scope_complete") is not True:
            raise ValueError("The active Context generation is not complete")
        result.update(status="passed_effective_settings", scope_complete=True,
                      context_build_id=snapshot.build_id,
                      dense_collection=snapshot.dense_collection, sparse_collection=snapshot.sparse_collection)
        print("[PASS] Actual router uses .5 / 2 / 100 / 100 / 30; complete Context generation active", flush=True)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        print(f"Runtime settings check stopped: {type(exc).__name__}: {exc}", flush=True)
    finally:
        close_vector_client()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "runtime_settings.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed_effective_settings" else 2


if __name__ == "__main__":
    raise SystemExit(main())
