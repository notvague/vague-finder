"""v10 초안(data/eval_drafts/queries_v10_draft.csv, git 제외) → 측정용 CSV + 분할표.

- 뺀 것: 검토 '제외' 6건, 정답 미지정(no_target) 3건 → 102건
- 층: (정답 곡이 기존 3,010곡인가 · 확장 크롤링 곡인가) × modality_focus 8칸
- 칸마다 dev 몫을 최대잉여법으로 정하고(합 60), 칸 안에서는 tier·target_scope별로
  묶어 섞은 뒤 고르게 뽑는다 — 틀린 단서·범주형 질의가 한쪽에 몰리지 않게.
- 난수 씨앗 고정(20261011). 같은 초안이면 같은 분할이 나온다.
- query_id는 초안 번호 그대로 t### (cc017 → t017). queries.json 0.7.0 병합 때 그대로 쓴다.

  venv/bin/python experiments/reranking/results_v38_v10_dev_first/make_split.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

DRAFT = Path("data/eval_drafts/queries_v10_draft.csv")
CORPUS = Path("data/all_songs.jsonl")
OLD_CORPUS = Path("data/all_songs_3010_backup.jsonl")
OUT_CSV = Path("experiments/reranking/eval_queries_v10.csv")
OUT_DIR = Path(__file__).resolve().parent
SEED = 20261011
DEV_TOTAL = 60


def _song_ids(path: Path) -> set:
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(str(json.loads(line)["song_id"]))
    return ids


def main() -> None:
    rows = list(csv.DictReader(DRAFT.open(encoding="utf-8-sig")))
    corpus, old = _song_ids(CORPUS), _song_ids(OLD_CORPUS)
    kept: List[Dict[str, str]] = []
    dropped: List[Dict[str, str]] = []
    for r in rows:
        pid = r["positive_id"].strip()
        if r["review"].strip() == "제외":
            dropped.append({"tmp_id": r["tmp_id"], "reason": "검토 제외"})
        elif r["label_status"] != "labeled" or not pid:
            dropped.append({"tmp_id": r["tmp_id"], "reason": "정답 미지정(no_target)"})
        elif pid not in corpus:
            dropped.append({"tmp_id": r["tmp_id"], "reason": "정답 곡이 코퍼스에 없음"})
        else:
            r["corpus"] = "old3010" if pid in old else "new"
            kept.append(r)

    cells: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for r in kept:
        cells[(r["corpus"], r["modality_focus"])].append(r)
    # 최대잉여법
    exact = {k: len(v) * DEV_TOTAL / len(kept) for k, v in cells.items()}
    quota = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[: DEV_TOTAL - sum(quota.values())]:
        quota[k] += 1

    rng = random.Random(SEED)
    for key in sorted(cells):
        groups: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
        for r in sorted(cells[key], key=lambda r: r["tmp_id"]):
            groups[(r["tier"], r["target_scope"])].append(r)
        ordered: List[Dict[str, str]] = []
        for g in sorted(groups):
            rng.shuffle(groups[g])
            ordered.extend(groups[g])
        n, d = len(ordered), quota[key]
        offset = rng.random()
        for i, r in enumerate(ordered):
            r["split"] = "dev" if int((i + 1) * d / n + offset) > int(i * d / n + offset) else "test"
        got = sum(r["split"] == "dev" for r in ordered)
        assert got == d, (key, got, d)

    kept.sort(key=lambda r: r["tmp_id"])
    for r in kept:
        r["query_id"] = "t" + r["tmp_id"][2:]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "split", "query_type", "query", "relevant_ids"])
        for r in kept:
            w.writerow([r["query_id"], r["split"], "search", r["query"].strip(), r["positive_id"].strip()])
    # 분할표(정답 제목·가수 없이 속성만 — 질의문·정답은 측정 CSV에 있다)
    with (OUT_DIR / "split_v10.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["query_id", "split", "corpus", "modality_focus", "tier", "category", "clue_type", "target_scope"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    with (OUT_DIR / "split_v10_dropped.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tmp_id", "reason"])
        w.writeheader()
        w.writerows(dropped)

    print("초안 sha256", hashlib.sha256(DRAFT.read_bytes()).hexdigest()[:16], "· 행", len(rows))
    print("남김", len(kept), "· 뺌", Counter(d["reason"] for d in dropped))
    print("split", Counter(r["split"] for r in kept))
    for col in ("corpus", "modality_focus", "tier", "target_scope", "category"):
        c = Counter((r[col], r["split"]) for r in kept)
        print(col, {v: f"{c[(v,'dev')]}/{c[(v,'test')]}" for v in sorted({r[col] for r in kept})})
    c = Counter(((r["corpus"], r["modality_focus"]), r["split"]) for r in kept)
    print("칸(dev/test)", {f"{k[0]}·{k[1]}": f"{c[(k,'dev')]}/{c[(k,'test')]}" for k in sorted(cells)})


if __name__ == "__main__":
    main()
