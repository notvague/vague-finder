#!/usr/bin/env python
"""분석 캐시가 지금 후처리 규칙과 맞는지 — 질의 원문만으로 정해지는 규칙 출력을 뽑는다.

분석 캐시에는 후처리를 거친 최종 분석만 있고 모델 원본이 없어, 후처리가 바뀐 뒤(postprocess_sha drift)에는
"같은 모델 응답을 지금 규칙으로 다시 처리하면 달라지는가"를 캐시만으로 답할 수 없다. 대신 **질의 원문만으로
정해지는** 규칙 출력(가사 가리기 · 표지/소리 근거 판정 · 규칙 가사 단서 · 규칙 폴백 분석)을 두 판의 코드에서
각각 뽑아 견준다. 출력이 같은 질의는 바뀐 규칙이 그 질의의 원문에서 다르게 동작하지 않는다는 뜻이다.
모델 응답에 따라 갈리는 분기까지 같다는 증명은 아니다.

    # 지금 코드에서
    venv/bin/python experiments/genre_memory_gap/cache_compat_probe.py dump --out new.json <질의.csv> …
    # 옛 코드 체크아웃에서(같은 스크립트를 그 폴더를 cwd로 실행)
    (cd <옛 체크아웃> && venv/bin/python <이 파일> dump --out old.json <질의.csv> …)
    venv/bin/python experiments/genre_memory_gap/cache_compat_probe.py diff old.json new.json

dev 분할만 읽는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.getcwd())


def _queries(paths: List[str]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("split") or "").strip().lower() == "dev" and (row.get("query") or "").strip():
                    found[f"{Path(path).stem}:{row['query_id'].strip()}"] = row["query"].strip()
    return found


def dump(args: argparse.Namespace) -> None:
    from src.retrieval import modality_queries as mq
    from src.retrieval import query_analyzer as qa
    from src.retrieval.lyrics_query import extract_lyric_clues

    out: Dict[str, Any] = {}
    for key, query in _queries(args.inputs).items():
        out[key] = {
            "masked": mq._mask_lyric_literals(query),
            "visual": mq.has_explicit_visual_clue(query),
            "audio_evidence": mq.extract_audio_evidence_text(query),
            "rule_lyric_clues": [getattr(c, "model_dump", lambda c=c: c)() for c in extract_lyric_clues(query)],
            "rule_fallback": qa._fallback(query).model_dump(mode="json"),
        }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
    print(f"{len(out)}건 → {args.out}")


def diff(args: argparse.Namespace) -> None:
    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    if set(old) != set(new):
        raise SystemExit(f"질의 집합이 다르다: {sorted(set(old) ^ set(new))[:5]}")
    changed = {key: [field for field in old[key] if old[key][field] != new[key][field]] for key in sorted(old)}
    changed = {key: fields for key, fields in changed.items() if fields}
    print(f"질의 {len(old)}건 중 규칙 출력이 달라진 것 {len(changed)}건")
    for key, fields in changed.items():
        print(f"  {key}: {', '.join(fields)}")
    if args.out:
        Path(args.out).write_text(json.dumps(changed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--out", required=True)
    p.add_argument("inputs", nargs="+")
    p.set_defaults(func=dump)
    p = sub.add_parser("diff")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--out", default="")
    p.set_defaults(func=diff)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
