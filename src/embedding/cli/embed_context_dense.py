"""CLI for fact-level KoE5 embeddings from ``artifacts/context``.

Examples (inside the backend container):

    python -m src.embedding.cli.embed_context_dense --dry-run
    python -m src.embedding.cli.embed_context_dense

The final catalogue run should add ``--require-complete-context``.  That flag
is intentionally omitted for a small pilot while the remaining songs have not
yet received Namuwiki artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.embedding.context_dense import (  # noqa: E402
    DEFAULT_CONTEXT_DENSE_BATCH_SIZE,
    DEFAULT_CONTEXT_DENSE_DIM,
    embed_context_dense,
    safe_model_tag,
)
from src.embedding.models.text_koe5 import DEFAULT_KOE5_MODEL  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    project = _project_root()
    parser = argparse.ArgumentParser(
        description="Validate context artifacts and create fact-level KoE5 embeddings"
    )
    parser.add_argument(
        "--context-dir",
        type=Path,
        default=Path(os.getenv("NAMUWIKI_ARTIFACT_DIR", project / "artifacts/context")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: artifacts/embeddings/context_dense/<model-tag>",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("CONTEXT_KOE5_MODEL", DEFAULT_KOE5_MODEL),
    )
    parser.add_argument(
        "--model-revision",
        default=os.getenv("CONTEXT_KOE5_REVISION") or None,
        help="optional Hugging Face revision/commit pin",
    )
    parser.add_argument(
        "--model-tag",
        default=os.getenv("CONTEXT_KOE5_MODEL_TAG") or None,
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=int(os.getenv("TEXT_DENSE_DIM", str(DEFAULT_CONTEXT_DENSE_DIM))),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv(
            "CONTEXT_DENSE_BATCH_SIZE", str(DEFAULT_CONTEXT_DENSE_BATCH_SIZE)
        )),
    )
    parser.add_argument("--force", action="store_true", help="re-embed every current song")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and plan only; do not load KoE5 or write files",
    )
    parser.add_argument(
        "--require-complete-context",
        action="store_true",
        help="final-run gate: refuse pending catalogue or orphan artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tag = safe_model_tag(args.model_tag or args.model_name)
    output_dir = args.output_dir or (
        _project_root() / "artifacts/embeddings/context_dense" / tag
    )

    def progress(message: str) -> None:
        print(f"[CONTEXT_DENSE] {message}", file=sys.stderr, flush=True)

    try:
        result = embed_context_dense(
            context_dir=args.context_dir,
            output_dir=output_dir,
            model_name=args.model_name,
            model_revision=args.model_revision,
            model_tag=tag,
            expected_dim=args.expected_dim,
            batch_size=args.batch_size,
            force=args.force,
            dry_run=args.dry_run,
            require_complete_scope=args.require_complete_context,
            progress=progress,
        )
    except (OSError, ValueError, RuntimeError) as exc:
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
