"""새 평가 자료 점검·분할 — 제출 CSV(`collection_template.csv` 형식) → 형식·코퍼스 포함·기존 질의 누출·자료 안 중복 검사 + 곡 단위 분할. 모델·검색 호출 없음.

**검색 시스템을 부르지 않는다**(작성자가 결과를 보고 표현을 고치는 일을 막는다). 읽는 것: 제출 CSV, 코퍼스 메타(`data/all_songs.jsonl`의 song_id·title·artist만),
기존 질의(`docs/eval/queries.json` + `experiments/reranking/eval_queries_*.csv` — test 포함, 비교에만 쓰고 원문을 출력하지 않는다).

검사:
- 필수 열·값(memory_source ∈ own_memory|helped_recall|synthetic, written_before_lookup ∈ yes|no, track ∈ general|diag_sound|diag_work) ·
  tmp_id 중복 · 빈 질의
- 정답: positive_id가 코퍼스에 있는가(없으면 `out_of_corpus` — 평가에서 빼고 따로 센다), 정답 미확인(`unresolved`)
- **기존 질의 누출**: 정답 곡이 기존 질의의 정답과 같다(`song_overlap`) · 질의 문장이 기존 질의와 비슷하다(글자 3-gram Jaccard ≥ 0.45, `near_duplicate`)
- 자료 안 중복: 같은 곡 여러 질의는 **같은 분할 묶음**으로 간다(곡 단위 분할) · 서로 비슷한 문장 표시
- 분할: 곡 ID(정답 미확인이면 tmp_id)의 salted sha256 → `policy_dev` / `sealed`. salt는 튜닝하지 않는 보관자가 정하고 이 파일·저장소에 적지 않는다.
  같은 salt면 같은 배정이 나온다. track(전체 분포 / 집중 진단)마다 따로 비율을 맞춘다.

    venv/bin/python new_data_check.py --input <제출.csv> --salt-file <보관자 파일> --sealed-ratio 0.6 --out-dir <보관자 폴더>

출력: `<out-dir>/checked.csv`(분할·플래그 포함 전체 — **보관자만**), `<out-dir>/policy_dev.csv`, `<out-dir>/sealed.csv`(보관자만),
`<out-dir>/report.json`(건수·플래그 수만, 질의 원문 없음 — 공유 가능). 화면에는 건수만 찍는다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REQUIRED = ("tmp_id", "writer_id", "memory_source", "written_before_lookup", "query", "track", "tier", "positive_id",
            "corpus_status")
MEMORY = {"own_memory", "helped_recall", "synthetic"}
TRACKS = {"general", "diag_sound", "diag_work"}
NEAR_DUP = 0.45


def grams(text: str, n: int = 3) -> Set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def existing_queries(root: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    qj = root / "docs/eval/queries.json"
    if qj.exists():
        for q in json.loads(qj.read_text(encoding="utf-8"))["queries"]:
            out.append({"text": q["query"], "ids": set(q.get("positives") or [])})
    for f in sorted((root / "experiments/reranking").glob("eval_queries_*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            out.append({"text": r["query"], "ids": {x.strip() for x in (r.get("relevant_ids") or "").split("|") if x.strip()}})
    return out


def corpus_ids(path: Path) -> Set[str]:
    ids = set()
    if not path.exists():
        raise SystemExit(f"코퍼스 파일이 없다: {path} (--corpus로 메인 체크아웃의 data/all_songs.jsonl을 준다)")
    for line in open(path, encoding="utf-8"):
        ids.add(str(json.loads(line)["song_id"]))
    return ids


def split_of(key: str, salt: str, sealed_ratio: float) -> str:
    h = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return "sealed" if h < sealed_ratio else "policy_dev"


def check(rows: List[Dict[str, str]], corpus: Set[str], existing: List[Dict[str, object]], salt: str,
          sealed_ratio: float) -> Dict[str, object]:
    problems: List[str] = []
    for col in REQUIRED:
        if rows and col not in rows[0]:
            problems.append(f"열 없음: {col}")
    ids = Counter(r["tmp_id"] for r in rows)
    problems += [f"tmp_id 중복: {k}" for k, n in ids.items() if n > 1]
    existing_ids = set().union(*[e["ids"] for e in existing]) if existing else set()
    existing_grams = [grams(str(e["text"])) for e in existing]
    new_grams = {r["tmp_id"]: grams(r["query"]) for r in rows}
    for r in rows:
        flags = []
        if not r["query"].strip():
            problems.append(f"{r['tmp_id']}: 빈 질의")
        if r["memory_source"] not in MEMORY:
            problems.append(f"{r['tmp_id']}: memory_source '{r['memory_source']}'")
        if r["written_before_lookup"] not in ("yes", "no"):
            problems.append(f"{r['tmp_id']}: written_before_lookup '{r['written_before_lookup']}'")
        if r["track"] not in TRACKS:
            problems.append(f"{r['tmp_id']}: track '{r['track']}'")
        pid = r["positive_id"].strip()
        if not pid:
            flags.append("unresolved")
        elif pid not in corpus:
            flags.append("out_of_corpus")
        if pid and pid in existing_ids:
            flags.append("song_overlap")
        best = max((jaccard(new_grams[r["tmp_id"]], g) for g in existing_grams), default=0.0)
        if best >= NEAR_DUP:
            flags.append("near_duplicate")
        if r["memory_source"] == "synthetic":
            flags.append("synthetic")  # 계약 시험용 — 독립 품질 증거로 쓰지 않는다
        if r["written_before_lookup"] == "no":
            flags.append("written_after_lookup")
        if len(re.sub(r"\s+", "", r["query"])) <= 15:
            flags.append("short")
        r["flags"] = "|".join(flags)
        r["max_sim_existing"] = f"{best:.3f}"
    # 자료 안: 같은 곡은 한 묶음, 비슷한 문장 표시
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if jaccard(new_grams[a["tmp_id"]], new_grams[b["tmp_id"]]) >= NEAR_DUP:
                a["flags"] = "|".join(filter(None, [a["flags"], f"near_dup_in_set:{b['tmp_id']}"]))
    for r in rows:
        key = r["positive_id"].strip() or f"tmp:{r['tmp_id']}"
        excluded = any(f in r["flags"].split("|") for f in ("out_of_corpus", "song_overlap", "near_duplicate", "synthetic"))
        r["split"] = "excluded" if excluded else split_of(f"{r['track']}:{key}", salt, sealed_ratio)
    by_song: Dict[str, Set[str]] = {}
    for r in rows:
        if r["split"] != "excluded" and r["positive_id"].strip():
            by_song.setdefault(r["positive_id"].strip(), set()).add(r["split"])
    split_leaks = [s for s, v in by_song.items() if len(v) > 1]
    if split_leaks:
        problems.append(f"같은 곡이 두 분할에 걸침: {len(split_leaks)}곡")
    report = {
        "rows": len(rows), "problems": problems,
        "split_counts": dict(Counter((r["track"], r["split"]) for r in rows).most_common()),
        "flag_counts": dict(Counter(f.split(":")[0] for r in rows for f in r["flags"].split("|") if f)),
        "memory_source": dict(Counter(r["memory_source"] for r in rows)),
        "writers": dict(Counter(r["writer_id"] for r in rows)),
        "sealed_ratio": sealed_ratio, "near_dup_threshold": NEAR_DUP,
    }
    return report


def write(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--salt-file", type=Path, required=True, help="보관자만 가진 파일(한 줄). 저장소·튜닝 세션에 두지 않는다")
    ap.add_argument("--sealed-ratio", type=float, default=0.6)
    ap.add_argument("--root", type=Path, default=ROOT, help="기존 질의(docs/eval·experiments/reranking)를 읽을 저장소 루트")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/all_songs.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    salt = args.salt_file.read_text(encoding="utf-8").strip()
    if len(salt) < 16:
        raise SystemExit("salt가 너무 짧다(16자 이상)")
    rows = [r for r in csv.DictReader(open(args.input, encoding="utf-8-sig")) if not r["tmp_id"].startswith("ex")]
    report = check(rows, corpus_ids(args.corpus), existing_queries(args.root), salt, args.sealed_ratio)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / "checked.csv", rows)
    write(args.out_dir / "policy_dev.csv", (r for r in rows if r["split"] == "policy_dev"))
    write(args.out_dir / "sealed.csv", (r for r in rows if r["split"] == "sealed"))
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "split_counts", "flag_counts")}, ensure_ascii=False))
    if report["problems"]:
        print(f"문제 {len(report['problems'])}건 — report.json 참조", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
