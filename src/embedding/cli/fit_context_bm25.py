"""Fit and encode the song-level Namuwiki context BM25 corpus.

Examples (inside the backend container):

    python -m src.embedding.cli.fit_context_bm25 --dry-run
    python -m src.embedding.cli.fit_context_bm25

Use ``--require-complete-context`` for the final full-catalogue build.  A pilot
build is allowed while the remaining songs are still awaiting context.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.embedding.context_sparse import (  # noqa: E402
    DEFAULT_CONTEXT_SPARSE_OUTPUT,
    build_context_bm25,
)
from src.embedding.models.context_bm25 import (  # noqa: E402
    DEFAULT_CONTEXT_BM25_B,
    DEFAULT_CONTEXT_BM25_K1,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    project = _project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Validate all context sparse profiles, fit one corpus-wide BM25 model, "
            "and publish song-level sparse vectors"
        )
    )
    parser.add_argument(
        "--context-dir",
        type=Path,
        default=Path(os.getenv("NAMUWIKI_ARTIFACT_DIR", project / "artifacts/context")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv(
            "CONTEXT_BM25_OUTPUT_DIR",
            project / DEFAULT_CONTEXT_SPARSE_OUTPUT,
        )),
    )
    parser.add_argument(
        "--b",
        type=float,
        default=float(os.getenv("CONTEXT_BM25_B", str(DEFAULT_CONTEXT_BM25_B))),
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=float(os.getenv("CONTEXT_BM25_K1", str(DEFAULT_CONTEXT_BM25_K1))),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="refit the complete current corpus even when the published build is valid",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and plan only; do not fit BM25 or write files",
    )
    parser.add_argument(
        "--require-complete-context",
        action="store_true",
        help="final-run gate: refuse pending catalogue or orphan artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(message: str) -> None:
        print(f"[CONTEXT_BM25] {message}", file=sys.stderr, flush=True)

    try:
        result = build_context_bm25(
            context_dir=args.context_dir,
            output_dir=args.output_dir,
            b=args.b,
            k1=args.k1,
            force=args.force,
            dry_run=args.dry_run,
            require_complete_scope=args.require_complete_context,
            progress=progress,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
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
