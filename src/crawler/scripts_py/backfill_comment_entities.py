"""
[Backfill] 저장된 멜론 댓글의 HTML 엔티티 정제

설명: 멜론 API는 댓글 본문을 HTML 이스케이프해서 준다. 수집기가 태그만 지우고 엔티티를
      풀지 않아 수집분 425곡·640개 댓글에 `&hellip;` 500회(382개 댓글), `&#39;` 144회,
      `&quot;` 127회 등의 엔티티가 남았다. 수집기는 고쳤지만(collect_melon_data) 이미 저장된
      댓글은 그대로라 따로 정제한다.

      주의: **수집기를 고치기 전에 모은 데이터에 한 번만** 실행한다. 레코드에 수집 시각이 없어
      고친 수집기로 모은 댓글과 구분할 수 없다. 고친 수집기로 새로 수집(append)한 뒤 --write를
      돌리면, 사용자가 일부러 쓴 '&lt;3' 같은 글자가 한 번 더 풀려 '<3'이 된다. 새 수집 전에
      적용하거나, 수정 전 스냅샷에만 적용할 것.

      원문 댓글은 색인에 들어가지 않는다(passage_builder·메타데이터 허용목록 모두 미사용).
      이 정제는 검색 결과를 바꾸지 않고, 댓글을 다시 LLM에 넣거나 사람이 볼 때의 품질만
      바로잡는다.

사용:
    # 미리보기 (기본) — 아무것도 쓰지 않는다
    venv/bin/python -m src.crawler.scripts_py.backfill_comment_entities

    # 적용 — 원본을 .bak으로 남기고 덮어쓴다
    venv/bin/python -m src.crawler.scripts_py.backfill_comment_entities --write

    # 개별 meta.json 폴더(드라이브 raw)도 함께
    venv/bin/python -m src.crawler.scripts_py.backfill_comment_entities --raw-dir <경로> --write

작성자: 황찬혁 (Full)
생성일: 2026-09-17
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);")


def clean_comments(record: Dict) -> int:
    """레코드의 멜론 댓글에서 엔티티를 푼다. 바뀐 댓글 수를 돌려준다."""
    comments = (record.get("comments") or {}).get("melon") or []
    changed = 0
    for i, text in enumerate(comments):
        if not isinstance(text, str) or not ENTITY.search(text):
            continue
        # 수집기(preprocess_melon_comment_candidates)와 같은 정책으로 한 번만 푼다.
        # 고치기 전 수집기가 저장한 댓글에 한 번 적용하면 고친 수집기가 저장했을 결과와 같다.
        # 여러 번 풀면 사용자가 일부러 쓴 '&lt;3'까지 '<3'이 된다. 수집분 640개는 한 번에
        # 엔티티가 남지 않는다. (이미 고친 수집기로 저장된 댓글에 다시 돌리면 안 된다 — 모듈 설명 참고)
        fixed = html.unescape(text)
        if fixed != text:
            comments[i] = fixed
            changed += 1
    return changed


def process_jsonl(path: Path, write: bool) -> Tuple[int, int, List[Tuple[str, str, str]]]:
    # 줄바꿈(\n)으로만 나눈다. str.splitlines()는 U+2028·U+0085에서도 끊는데, json.dumps(
    # ensure_ascii=False)는 이 문자를 이스케이프하지 않아 레코드 중간이 잘린다.
    with path.open(encoding="utf-8", newline="") as f:
        lines = [line[:-1] if line.endswith("\n") else line for line in f]
    out: List[str] = []
    songs = total = 0
    samples: List[Tuple[str, str, str]] = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        record = json.loads(line)
        before = list((record.get("comments") or {}).get("melon") or [])
        n = clean_comments(record)
        if n:
            songs += 1
            total += n
            if len(samples) < 5:
                after = record["comments"]["melon"]
                for b, a in zip(before, after):
                    if b != a:
                        samples.append((record.get("song_id", ""), b[:60], a[:60]))
                        break
        out.append(json.dumps(record, ensure_ascii=False))

    if write and total:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return songs, total, samples


def process_raw_dir(raw_dir: Path, write: bool) -> Tuple[int, int]:
    songs = total = 0
    for meta in sorted(raw_dir.glob("*/meta.json")):
        record = json.loads(meta.read_text(encoding="utf-8"))
        n = clean_comments(record)
        if not n:
            continue
        songs += 1
        total += n
        if write:
            shutil.copy2(meta, meta.with_suffix(".json.bak"))
            meta.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return songs, total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="저장된 멜론 댓글의 HTML 엔티티를 푼다. 수집기 수정 전 데이터에 한 번만 실행 "
                    "(고친 수집기로 새로 모은 댓글에 다시 돌리면 이중으로 풀린다)"
    )
    parser.add_argument("--jsonl", type=Path, default=Path("data/all_songs.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=None, help="곡별 meta.json 폴더 (선택)")
    parser.add_argument("--write", action="store_true", help="실제로 덮어쓴다 (없으면 미리보기)")
    args = parser.parse_args()

    mode = "적용" if args.write else "미리보기"
    if args.jsonl.exists():
        songs, total, samples = process_jsonl(args.jsonl, args.write)
        print(f"[{mode}] {args.jsonl}: {songs}곡 · 댓글 {total}개")
        for sid, before, after in samples:
            print(f"   {sid}  {before!r}")
            print(f"   {'':{len(sid)}}  -> {after!r}")
    else:
        print(f"[건너뜀] {args.jsonl} 없음")

    if args.raw_dir:
        songs, total = process_raw_dir(args.raw_dir, args.write)
        print(f"[{mode}] {args.raw_dir}/*/meta.json: {songs}곡 · 댓글 {total}개")

    if not args.write:
        print("\n미리보기입니다. 적용하려면 --write (원본은 .bak으로 남습니다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
