"""기존 Pinecone 텍스트 벡터에 제목 문자 존재 메타데이터만 추가한다.

임베딩과 sparse vector를 다시 만들지 않고 MongoDB의 곡 제목을 읽어 Pinecone
``set_metadata`` 업데이트만 수행한다.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from src.common.mongodb import get_collection
from src.common.title_features import analyze_title_structure
from src.vector_db.pinecone_client import get_pinecone_client
from src.vector_db.settings import NAMESPACE, TEXT_HYBRID_INDEX_NAME


def _song_id(document: Dict[str, Any]) -> str:
    return str(
        document.get("song_id")
        or document.get("id")
        or document.get("_id")
        or ""
    ).strip()


def _title(document: Dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    return str(metadata.get("title") or document.get("title") or "").strip()


def _presence_metadata(title: str) -> Dict[str, bool]:
    features = analyze_title_structure(title)
    return {
        "title_has_latin_anywhere": features["has_latin_anywhere"],
        "title_has_hangul_anywhere": features["has_hangul_anywhere"],
        "title_has_hanja_anywhere": features["has_hanja_anywhere"],
        "title_has_number_anywhere": features["has_number_anywhere"],
    }


def backfill(*, dry_run: bool = False, limit: int | None = None) -> int:
    cursor = get_collection("songs").find(
        {},
        {
            "song_id": 1,
            "id": 1,
            "title": 1,
            "metadata.title": 1,
        },
    )
    if limit is not None:
        cursor = cursor.limit(limit)

    index = None
    if not dry_run:
        index = get_pinecone_client().Index(TEXT_HYBRID_INDEX_NAME)

    scanned = 0
    updated = 0
    skipped = 0
    failed = 0
    for document in cursor:
        scanned += 1
        song_id = _song_id(document)
        title = _title(document)
        if not song_id or not title:
            skipped += 1
            continue

        metadata = _presence_metadata(title)
        if dry_run:
            if updated < 10:
                print(song_id, repr(title), metadata)
            updated += 1
            continue

        try:
            index.update(
                id=song_id,
                set_metadata=metadata,
                namespace=NAMESPACE,
            )
            updated += 1
        except Exception as exc:
            failed += 1
            print(
                f"[FAIL] id={song_id} title={title!r}: {exc}",
                file=sys.stderr,
            )

        if scanned % 100 == 0:
            print(
                f"[PROGRESS] scanned={scanned} updated={updated} "
                f"skipped={skipped} failed={failed}"
            )

    print(
        f"[DONE] index={TEXT_HYBRID_INDEX_NAME} namespace={NAMESPACE} "
        f"scanned={scanned} updated={updated} skipped={skipped} failed={failed} "
        f"dry_run={dry_run}"
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pinecone을 수정하지 않고 계산 결과 10개만 미리 본다.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="개발 확인용 최대 처리 문서 수",
    )
    args = parser.parse_args()
    return backfill(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
