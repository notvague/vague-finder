"""Deprecated flat matcher entrypoint. Use collect_song_context subcommands.

Automatic page-wide co-occurrence matching and truncated context excerpts have
been removed. This module never returns legacy context_usable/fact_summary data.
"""
import warnings


def collect_namuwiki_data(*args, **kwargs):
    raise RuntimeError(
        "Flat Namuwiki collection was retired. Use "
        "'python -m src.crawler.scripts_py.backfill_namuwiki_context --help'; "
        "results belong in data/context, not meta.json."
    )


if __name__ == "__main__":
    # CLI는 이 파일을 직접 실행할 때만 가져온다. 모듈 최상단에서 가져오면 크롤러
    # (src.crawler.main -> collect_namuwiki_data)가 CLI의 임포트 오류까지 함께 짊어진다.
    from src.crawler.scripts_py.collect_song_context import main

    warnings.warn("Use src.crawler.scripts_py.collect_song_context; CLI syntax has changed.", FutureWarning)
    raise SystemExit(main())
