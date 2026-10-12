"""v36 — 질의 분석 프롬프트에 생애 단계 지침 한 줄(#35)을 넣은 효과. v06 dev 56(q115 제외) · test 25. 레포 루트에서 실행.

세 캐시를 같은 코드·같은 날·같은 경로(AI Studio)로 잰다.
- old : 9/22 기준 캐시(`analysis_cache_v06_{split}.json`, v35까지의 기준선이 쓴 분석) — 프롬프트 판·몇 주의 모델 변동이 섞여 있다
- ctrl: 옛 프롬프트(main)로 **오늘** 만든 캐시 — 프롬프트 한 줄의 효과는 ctrl vs new로만 가른다
- new : 새 프롬프트로 오늘 만든 캐시
같은 프롬프트로 두 번 만든 캐시(ctrl vs ctrl2, new vs new2)는 0/57 같았다 — AI Studio 분석은 결정적이고, 차이는 전부 프롬프트 문구에서 온다.
품질은 커밋되는 경량본 `search_eval_{split}_top10.csv`(곡 ID만)로 계산한다. listwise 리랭킹은 실행마다 ±1 흔들린다.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_diff import diff as cache_diff  # noqa: E402

ROOT = Path("experiments/reranking")
OUT = ROOT / "results_v36_life_stage_prompt"
METRICS = ["Hit@1", "Hit@5", "Hit@10", "MRR@10", "nDCG@10"]
VARIANTS = [("old", "9/22 캐시"), ("ctrl", "옛 프롬프트·오늘"), ("new", "새 프롬프트·오늘")]


def summary(d: Path, split: str) -> dict[str, dict]:
    return {r["metric"]: r for r in csv.DictReader(open(d / f"search_eval_{split}_summary.csv", encoding="utf-8-sig"))}


def ranks(d: Path, split: str) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for row in csv.DictReader(open(d / f"search_eval_{split}_top10.csv", encoding="utf-8-sig")):
        relevant = {x for x in row["relevant_ids"].split("|") if x}
        top = [x for x in row["rerank_top_ids"].split("|") if x]
        hit = [i for i, s in enumerate(top, start=1) if s in relevant]
        out[row["query_id"]] = hit[0] if hit else None
    return out


def route(d: Path, split: str) -> str:
    p = d / f"search_eval_{split}_runinfo.json"
    if not p.exists():
        return "?"
    info = json.load(open(p, encoding="utf-8"))
    g = info.get("gemini") or {}
    return f"{g.get('backend', '?')}/{g.get('setting', '?')}"


def cache_path(tag: str, split: str) -> Path:
    return ROOT / f"analysis_cache_v06_{split}.json" if tag == "old" else OUT / f"analysis_cache_v06_{split}_{tag}.json"


for split in ("dev", "test"):
    present = [(t, label) for t, label in VARIANTS if (OUT / f"{t}_{split}" / f"search_eval_{split}_top10.csv").exists()]
    if len(present) < 2:
        print(f"\n## v06 {split} — 결과 {len(present)}개뿐, 비교 없음")
        continue
    rk = {t: ranks(OUT / f"{t}_{split}", split) for t, _ in present}
    n = len(next(iter(rk.values())))
    print(f"\n## v06 {split} {n}건  경로: " + " · ".join(f"{t} {route(OUT / f'{t}_{split}', split)}" for t, _ in present))
    sm = {t: summary(OUT / f"{t}_{split}", split) for t, _ in present}
    print("| 지표 (리랭킹 뒤) | " + " | ".join(f"{t} ({label})" for t, label in present) + " |")
    print("|---|" + "---|" * len(present))
    for m in METRICS:
        cells = []
        for t, _ in present:
            v = float(sm[t][m]["rerank"])
            cells.append(f"{v:.3f} ({round(v * n)})" if m.startswith("Hit") else f"{v:.3f}")
        print(f"| {m} | " + " | ".join(cells) + " |")
    print("| 검색만 Hit@10 | " + " | ".join(f"{float(sm[t]['Hit@10']['baseline']):.3f} ({round(float(sm[t]['Hit@10']['baseline']) * n)})" for t, _ in present) + " |")

    pairs = [(a, b) for a, _ in present for b, _ in present if a < b and (a, b) != ("new", "old")]
    for a, b in [("old", "new"), ("ctrl", "new"), ("old", "ctrl")]:
        if a not in rk or b not in rk:
            continue
        changed = [(q, rk[a][q], rk[b][q]) for q in rk[a] if rk[a][q] != rk[b].get(q)]
        gain = [q for q, x, y in changed if x is None and y is not None]
        loss = [q for q, x, y in changed if x is not None and y is None]
        inside = [(q, x, y) for q, x, y in changed if x and y]
        _, d = cache_diff(cache_path(a, split), cache_path(b, split))
        moved = {q for q, _, _ in changed}
        print(f"\n{a} → {b}: Top-10 순위가 다른 질의 {len(changed)}/{n} — 회복 {gain} · 손실 {loss} · 안에서 이동 {inside}")
        print(f"  분석이 다른 질의 {len(d)}/{n + (1 if split == 'dev' else 0)}. 순위가 바뀐 질의 중 분석도 다른 것 {len(moved & set(d))} / 분석은 같은데 순위만 바뀐 것(listwise 변동) {len(moved - set(d))}")
