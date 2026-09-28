"""Load validated Namuwiki context dense/sparse vectors into Qdrant.

Pilot (partial context catalogue):

    python -m src.vector_db.cli.qdrant_load_context --dry-run
    python -m src.vector_db.cli.qdrant_load_context

Final full-catalogue publication must add ``--require-complete-context``.
When local-file Qdrant is used, stop the backend first because the storage path
can be opened by only one process at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

load_dotenv()

from src.embedding.context_dense import safe_model_tag  # noqa: E402
from src.embedding.context_sparse import DEFAULT_CONTEXT_SPARSE_OUTPUT  # noqa: E402
from src.embedding.models.text_koe5 import DEFAULT_KOE5_MODEL  # noqa: E402
from src.vector_db.context_qdrant import load_context_qdrant  # noqa: E402
from src.vector_db.settings import (  # noqa: E402
    CONTEXT_DENSE_DIM,
    CONTEXT_QDRANT_BATCH_SIZE,
    NAMESPACE,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    project = _project_root()
    model_name = os.getenv("CONTEXT_KOE5_MODEL", DEFAULT_KOE5_MODEL)
    model_tag = safe_model_tag(os.getenv("CONTEXT_KOE5_MODEL_TAG") or model_name)
    parser = argparse.ArgumentParser(
        description=(
            "Validate context artifacts and publish fact-dense/song-sparse "
            "generation collections to Qdrant"
        )
    )
    parser.add_argument(
        "--context-dir",
        type=Path,
        default=Path(os.getenv("NAMUWIKI_ARTIFACT_DIR", project / "artifacts/context")),
    )
    parser.add_argument(
        "--dense-dir",
        type=Path,
        default=Path(os.getenv(
            "CONTEXT_DENSE_OUTPUT_DIR",
            project / "artifacts/embeddings/context_dense" / model_tag,
        )),
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=Path(os.getenv(
            "CONTEXT_BM25_OUTPUT_DIR",
            project / DEFAULT_CONTEXT_SPARSE_OUTPUT,
        )),
    )
    parser.add_argument(
        "--state-manifest",
        type=Path,
        default=None,
        help="default: artifacts/vector_db/context_qdrant/<namespace>/manifest.json",
    )
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument(
        "--qdrant-path",
        default=None,
        help="default: QDRANT_PATH or artifacts/qdrant (ignored when QDRANT_URL is set)",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=CONTEXT_DENSE_DIM,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONTEXT_QDRANT_BATCH_SIZE,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate all source files and plan only; do not open or mutate Qdrant",
    )
    parser.add_argument(
        "--require-complete-context",
        action="store_true",
        help="final-run gate: refuse a partial context catalogue",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="publish a fresh repair generation even when this exact build is active",
    )
    parser.add_argument(
        "--prune-old-generations",
        action="store_true",
        help="after a successful alias switch, delete inactive context generations",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state_manifest = args.state_manifest or (
        _project_root()
        / "artifacts/vector_db/context_qdrant"
        / str(args.namespace)
        / "manifest.json"
    )

    def progress(message: str) -> None:
        print(f"[CONTEXT_QDRANT] {message}", file=sys.stderr, flush=True)

    try:
        result = load_context_qdrant(
            context_dir=args.context_dir,
            dense_dir=args.dense_dir,
            sparse_dir=args.sparse_dir,
            state_manifest=state_manifest,
            namespace=args.namespace,
            qdrant_path=args.qdrant_path,
            expected_dense_dim=args.expected_dim,
            batch_size=args.batch_size,
            require_complete_scope=args.require_complete_context,
            dry_run=args.dry_run,
            force=args.force,
            prune_old_generations=args.prune_old_generations,
            progress=progress,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        ResponseHandlingException,
        UnexpectedResponse,
    ) as exc:
        print(json.dumps({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
