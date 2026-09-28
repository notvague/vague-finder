from __future__ import annotations

import pandas as pd
from pathlib import Path
from urllib.parse import quote


INPUT_PATH = Path("experiments/namuwiki/sample_30.csv")# prepare_namuwiki_sample 코드에서 만든 파일
OUTPUT_PATH = Path("experiments/namuwiki/sample_30_match.csv")

NAMUWIKI_SEARCH_BASE = "https://namu.wiki/Search?q="


def make_search_query(row) -> str:
    artist = str(row.get("artist", "")).strip()
    title = str(row.get("title", "")).strip()

    return f"{artist} {title}".strip()


def make_search_url(query: str) -> str:
    return NAMUWIKI_SEARCH_BASE + quote(query)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"표본 CSV를 찾을 수 없습니다: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    df["search_query"] = df.apply(
        make_search_query,
        axis=1,
    )

    df["search_url"] = df["search_query"].apply(
        make_search_url
    )

    # 사람이 검증해서 채울 필드
    if "match_status" not in df.columns:
        df["match_status"] = ""

    if "match_type" not in df.columns:
        df["match_type"] = ""

    if "page_title" not in df.columns:
        df["page_title"] = ""

    if "page_url" not in df.columns:
        df["page_url"] = ""

    if "reason" not in df.columns:
        df["reason"] = ""

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[OK] 매칭 검증용 CSV 생성: {OUTPUT_PATH}")
    print(f"[OK] 총 {len(df)}곡")

    print()
    print(
        df[
            [
                "fame",
                "artist",
                "title",
                "search_query",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()