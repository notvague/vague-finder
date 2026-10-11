"""평가 질의 전체에 모달리티 규칙을 돌려, 어느 가지·어느 글자로 판정이 났는지 적는다(검색·Gemini 호출 없음).

  PYTHONPATH=. venv/bin/python experiments/reranking/results_v40_modality_token_fix/scan_rules.py <출력 CSV>
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from src.retrieval import modality_queries as m

SETS = {"v06": "experiments/reranking/eval_queries_v06.csv", "v09": "experiments/reranking/eval_queries_v09.csv",
        "v10": "experiments/reranking/eval_queries_v10.csv"}
HANGUL_BEFORE_ROCK = re.compile(r"[가-힣](?:록|락)")


def visual_branch(q: str) -> str:
    text = m._NEGATED_COVER_CONTEXT_RE.sub("", q)
    if m._COVER_CONTEXT_RE.search(text):
        return "cover_noun"
    if "앨범" in text and m._VISUAL_DETAIL_RE.search(text):
        return "album+detail:" + m._VISUAL_DETAIL_RE.search(text).group(0)
    if m._has_implicit_artwork_description(text):
        return "implicit"
    return ""


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    for name, path in SETS.items():
        cache = {}
        for split in ("dev", "test"):
            p = Path(f"experiments/reranking/analysis_cache_{name}_{split}.json")
            if p.exists():
                cache.update(json.load(open(p, encoding="utf-8"))["entries"])
        for r in csv.DictReader(open(path, encoding="utf-8")):
            q = r["query"]
            entry = cache.get(r["query_id"])
            perf = (entry or {}).get("analysis", {}).get("performance_clues")
            strong = m._STRONG_AUDIO_RE.search(m.extract_audio_evidence_text(q))
            rows.append({
                "set": name, "split": r["split"], "query_id": r["query_id"],
                "visual": int(m.has_explicit_visual_clue(q)), "visual_branch": visual_branch(q),
                "audio": int(m.has_explicit_audio_clue(q, perf)), "audio_strong_token": strong.group(0) if strong else "",
                "audio_evidence_text": m.extract_audio_evidence_text(q),
                "detailed_audio": int(m.has_detailed_audio_description(q, perf)),
                "rock_inside_word": "|".join(HANGUL_BEFORE_ROCK.findall(q)),
                "cached_fallback": "" if entry is None else int(bool(entry.get("fallback"))),
                "query": q,
            })
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "행 →", out)


if __name__ == "__main__":
    main()
