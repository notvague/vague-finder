"""
Eval set / song catalog 로더.

- load_eval_set(path): EvalSet 로드 + 스키마 검증
- load_song_catalog(path): song_id → 메타 dict 로드
- cross_validate(eval_set, catalog): positives/negatives의 모든 song_id가
  실제 카탈로그에 존재하는지 확인 (오타/존재하지 않는 ID 차단)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

from src.eval.schema import EvalSet


DEFAULT_EVAL_PATH = Path("docs/eval/queries.json")
DEFAULT_CATALOG_PATH = Path("docs/eval/song_catalog.json")


def load_eval_set(path: Path | str = DEFAULT_EVAL_PATH) -> EvalSet:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval set 파일 없음: {p}")
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return EvalSet(**raw)


def load_song_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> Dict[str, dict]:
    """song_id → {title, artist, genre, vocal_gender, mood_tags} dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"카탈로그 없음: {p}\n"
            "→ `python -m src.eval.build_catalog` 로 재생성하세요."
        )
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {entry["song_id"]: entry for entry in raw["songs"]}


def cross_validate(eval_set: EvalSet, catalog: Dict[str, dict]) -> List[str]:
    """Eval set 안의 모든 song_id가 카탈로그에 존재하는지 검증.
    오류 메시지 리스트를 반환 (빈 리스트면 통과)."""
    errors: List[str] = []
    known: Set[str] = set(catalog.keys())
    for q in eval_set.queries:
        for sid in q.positives:
            if sid not in known:
                errors.append(f"[{q.query_id}] positive '{sid}' 카탈로그에 없음")
        for sid in q.negatives:
            if sid not in known:
                errors.append(f"[{q.query_id}] negative '{sid}' 카탈로그에 없음")
    return errors
