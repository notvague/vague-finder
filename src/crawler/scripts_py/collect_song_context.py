"""Deprecated compatibility alias for the current Namuwiki backfill CLI."""
from __future__ import annotations

import warnings

from src.crawler.scripts_py.backfill_namuwiki_context import main as backfill_main


def main(argv=None) -> int:
    warnings.warn(
        "collect_song_context is deprecated; "
        "use src.crawler.scripts_py.backfill_namuwiki_context.",
        FutureWarning,
        stacklevel=2,
    )
    return backfill_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())