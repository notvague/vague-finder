"""두 분석 캐시의 항목을 대조한다 — 질의별로 어떤 필드가 다른지. 레포 루트에서 실행.
    venv/bin/python experiments/reranking/results_v36_life_stage_prompt/cache_diff.py <A.json> <B.json> [--verbose]
"""
import json
import sys
from collections import Counter

FIELDS = ("intent_type", "release_era", "korean_tags", "vocal_gender", "genre", "artist_type", "text_alpha",
          "lyric_semantic_query", "modality_weights", "lyric_keywords", "song_title", "artist_name", "performance_clues")


def ent(p):
    c = json.load(open(p, encoding="utf-8"))
    e = c.get("entries") or c.get("queries")
    return {k: v.get("analysis", v) for k, v in e.items()}


def diff(a_path, b_path):
    a, b = ent(a_path), ent(b_path)
    out = {}
    for q in a:
        if q in b:
            d = {f for f in FIELDS if a[q].get(f) != b[q].get(f)}
            if d:
                out[q] = d
    return len(a), out


if __name__ == "__main__":
    n, d = diff(sys.argv[1], sys.argv[2])
    print(f"분석 달라진 질의 {len(d)}/{n} · 필드별 {dict(Counter(f for x in d.values() for f in x))}")
    core = {q: {f for f in x if f not in ('korean_tags', 'lyric_semantic_query', 'text_alpha', 'modality_weights')} for q, x in d.items()}
    core = {q: x for q, x in core.items() if x}
    print(f"  검색 조건이 바뀌는 필드(intent·era·gender·genre·type·lyric_keywords·title·artist)가 다른 질의 {len(core)}: {core}")
    if "--verbose" in sys.argv:
        a, b = ent(sys.argv[1]), ent(sys.argv[2])
        for q, x in d.items():
            print(f"  {q}: " + " / ".join(f"{f}: {a[q].get(f)} → {b[q].get(f)}" for f in sorted(x) if f != 'korean_tags'))
