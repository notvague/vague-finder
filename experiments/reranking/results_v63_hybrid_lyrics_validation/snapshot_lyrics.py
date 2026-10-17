"""v63 — 서비스 가사 표면 검색이 읽는 Mongo 자료를 **읽기 전용으로 한 번** 떠서 고정한다.

서비스와 같은 읽기를 쓴다: `LyricsExactSearchService()._source_documents()`(필터 `{}` · 서비스 projection ·
가사 전용 클라이언트 socket 제한 60초 · 배치 200). 검색 알고리즘은 건드리지 않고, A/B 재생 때는 이 파일을
같은 메서드 자리에 다시 꽂는다(`v63_runtime.install_lyrics_snapshot`).

원본 변경 감지: 전체 읽기 **앞뒤로** 가벼운 내용 지문(평가기 `_lyrics_fingerprint`와 같은 방식: song_id·full_lyrics
정렬 해시)을 한 번씩 더 읽어, 앞 = 전체 읽기에서 계산한 값 = 뒤 일 때만 게시한다. 세 값이 같아도 읽는 사이
바뀌었다가 되돌아간 경우는 배제하지 못한다(한계로 기록).

게시: 임시 폴더에 문서·manifest를 다 쓴 뒤 폴더 이름을 한 번에 바꾸고, 마지막에 `COMPLETE`를 쓴다.
읽기 시간 제한(`--max-seconds`, 기본 900초)·연결 실패·중간 예외면 임시 폴더를 `FAILED_*`로 남기고 exit 1 —
부분 자료는 게시하지 않는다. URI·계정·가사 원문은 출력하지 않는다(manifest에도 없음).

    MONGO_URI=… MONGO_DB_NAME=… venv/bin/python <이 파일> [--out artifacts/lyrics_snapshot_v63]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def content_digest(pairs) -> dict:
    """평가기 `_lyrics_fingerprint`와 같은 계산(song_id 정렬 → id\\0가사\\1)."""
    rows = sorted(pairs)
    digest = hashlib.sha256()
    total = 0
    with_lyrics = 0
    for song_id, text in rows:
        digest.update(song_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x01")
        total += len(text)
        with_lyrics += 1 if text else 0
    return {
        "document_count": len(rows),
        "full_lyrics_songs": with_lyrics,
        "full_lyrics_total_chars": total,
        "content_sha256_16": digest.hexdigest()[:16],
    }


def light_fingerprint(client, db_name: str, deadline: float) -> dict:
    started = time.perf_counter()
    pairs = []
    cursor = client[db_name]["songs"].find({}, {"song_id": 1, "lyrics_data.full_lyrics": 1}, batch_size=500)
    try:
        for document in cursor:
            if time.monotonic() > deadline:
                raise TimeoutError("가벼운 지문 읽기가 시간 제한을 넘었다")
            text = (document.get("lyrics_data") or {}).get("full_lyrics") or ""
            pairs.append((str(document.get("song_id") or ""), text))
    finally:
        cursor.close()
    out = content_digest(pairs)
    out["seconds"] = round(time.perf_counter() - started, 2)
    return out


def full_document_digest(documents) -> str:
    """서비스 projection 전체(메타·태그·링크 포함)의 지문 — 곡 정보가 바뀌어도 잡는다."""
    from bson import json_util

    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda d: (str(d.get("song_id") or ""), str(d.get("_id")))):
        digest.update(json_util.dumps(document, sort_keys=True).encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/lyrics_snapshot_v63")
    parser.add_argument("--max-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if not os.getenv("MONGO_URI"):
        raise SystemExit("MONGO_URI가 없다(값은 출력하지 않는다)")

    out_root = ROOT / args.out
    final_dir = out_root / "snapshot"
    if (final_dir / "COMPLETE").exists():
        raise SystemExit(f"이미 게시된 스냅샷이 있다: {final_dir} — 덮어쓰지 않는다")
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    tmp_dir = out_root / f"tmp_{stamp}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict = {"started_at": now(), "status": "running", "max_seconds": args.max_seconds}
    deadline = time.monotonic() + args.max_seconds

    from src.retrieval.lyrics_exact_search import (  # noqa: E402
        DEFAULT_MONGO_BATCH_SIZE,
        DEFAULT_MONGO_SOCKET_TIMEOUT_SECONDS,
        LyricsExactSearchService,
        _make_lyrics_mongo_client,
    )
    from src.common.mongodb import DB_NAME  # noqa: E402

    service = LyricsExactSearchService()  # 서비스 기본값(가사 전용 클라이언트·배치)
    probe = None
    try:
        probe = _make_lyrics_mongo_client(DEFAULT_MONGO_SOCKET_TIMEOUT_SECONDS)
        manifest["db_name"] = DB_NAME
        manifest["collection"] = "songs"
        manifest["read"] = {
            "method": "LyricsExactSearchService._source_documents (서비스와 같은 find({}, projection))",
            "socket_timeout_seconds": DEFAULT_MONGO_SOCKET_TIMEOUT_SECONDS,
            "batch_size": DEFAULT_MONGO_BATCH_SIZE,
        }
        manifest["count_documents_before"] = probe[DB_NAME]["songs"].count_documents({})
        manifest["light_fingerprint_before"] = light_fingerprint(probe, DB_NAME, deadline)
        print(f"[{now()}] 앞 지문 {manifest['light_fingerprint_before']}", flush=True)

        full_started = time.perf_counter()
        manifest["full_read_started_at"] = now()
        documents = []
        source = service._source_documents()
        try:
            for raw in source:
                if time.monotonic() > deadline:
                    raise TimeoutError("전체 읽기가 시간 제한을 넘었다")
                documents.append(raw)
                if len(documents) % 1000 == 0:
                    print(f"[{now()}] {len(documents)}건", flush=True)
        finally:
            source.close()
        manifest["full_read_finished_at"] = now()
        manifest["full_read_seconds"] = round(time.perf_counter() - full_started, 2)

        manifest["light_fingerprint_after"] = light_fingerprint(probe, DB_NAME, deadline)
        manifest["count_documents_after"] = probe[DB_NAME]["songs"].count_documents({})
        print(f"[{now()}] 뒤 지문 {manifest['light_fingerprint_after']}", flush=True)
    except BaseException as exc:  # noqa: BLE001 - 어떤 실패든 게시하지 않는다
        manifest["status"] = "failed"
        manifest["error"] = type(exc).__name__  # 메시지에는 호스트가 섞일 수 있어 적지 않는다
        manifest["finished_at"] = now()
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_dir.rename(out_root / f"FAILED_{stamp}")
        print(f"실패 — 게시하지 않음: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if probe is not None:
            probe.close()
        service.close(timeout=5)

    pairs = [
        (str(d.get("song_id") or ""), str((d.get("lyrics_data") or {}).get("full_lyrics") or ""))
        for d in documents
    ]
    from_full = content_digest(pairs)
    ids = [str(d.get("song_id") or "") for d in documents]
    counts = Counter(ids)
    manifest["documents"] = len(documents)
    manifest["content_from_full_read"] = from_full
    manifest["full_document_sha256_16"] = full_document_digest(documents)
    manifest["distinct_song_ids"] = len(counts)
    manifest["duplicate_song_ids"] = sorted(k for k, v in counts.items() if v > 1)
    manifest["missing_song_id_docs"] = counts.get("", 0)
    manifest["empty_full_lyrics_song_ids"] = sorted(
        sid for sid, text in pairs if not text.strip()
    )
    before = manifest["light_fingerprint_before"]["content_sha256_16"]
    after = manifest["light_fingerprint_after"]["content_sha256_16"]
    consistent = (
        before == after == from_full["content_sha256_16"]
        and manifest["count_documents_before"] == manifest["count_documents_after"] == len(documents)
    )
    manifest["consistent_before_full_after"] = consistent
    if not consistent:
        manifest["status"] = "failed"
        manifest["error"] = "source_changed_or_partial"
        manifest["finished_at"] = now()
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_dir.rename(out_root / f"FAILED_{stamp}")
        print("앞·전체·뒤 지문 또는 건수가 다르다 — 게시하지 않음", file=sys.stderr)
        return 1

    data_path = tmp_dir / "documents.pkl.gz"
    with gzip.open(data_path, "wb") as handle:
        pickle.dump(documents, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest["file"] = {
        "name": data_path.name,
        "bytes": data_path.stat().st_size,
        "sha256_16": hashlib.sha256(data_path.read_bytes()).hexdigest()[:16],
    }
    manifest["status"] = "complete"
    manifest["finished_at"] = now()
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_dir.rename(final_dir)
    (final_dir / "COMPLETE").write_text(manifest["finished_at"] + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "documents", "distinct_song_ids", "missing_song_id_docs", "consistent_before_full_after",
        "content_from_full_read", "full_document_sha256_16", "full_read_seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
