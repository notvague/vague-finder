"""(a) 보너스 전 순서 고정 + 위약 — m402·c603 정답 순위. 레포 루트에서 실행.

읽는 파일: 각 폴더의 clarify_detail.csv(rank · candidate_rank@30 · found_turn)와 clarify_runinfo.json(리랭커 상태 · 호출 경로).
flow 칸은 m402의 후보 안 대화(67169)만. 후보 순위(괄호)는 보너스 순서의 후보 목록 기준이다 — (a)·위약은 리랭커 입력만 바꾼다.
위약(placebo:<seed>)은 보너스가 바꾼 바로 그 자리들의 곡을 고정 시드(+후보 id)로 섞고 답은 프롬프트에 넣지 않은 대조다.

**읽기 전 규칙**: 열마다 먼저 찍히는 리랭커 상태와 호출 경로를 본다. failed·exception·상태 없음이 있는 열은 비교에서 뺀다.
호출 경로(gemini.backend)가 다른 열끼리는 비교하지 않는다 — v10 열(AI Studio)은 참고로만 둔다.
"""
import csv
from pathlib import Path

ROOT = Path("experiments/reranking")
RUNS = [
    ("v10 base (AI Studio · 참고)", ROOT / "results_clarify_v10_corrections/base_r1"),
    ("v10 ①+② (AI Studio · 참고)", ROOT / "results_clarify_v10_corrections/both_r1"),
    ("base", ROOT / "results_clarify_v11_prebonus/base_rerun_r1"),
    ("(a) r1", ROOT / "results_clarify_v11_prebonus/prebonus_r1"),
    ("(a) r2", ROOT / "results_clarify_v11_prebonus/prebonus_r2"),
    ("(a)+①+② r1", ROOT / "results_clarify_v11_prebonus/prebonus_both_r1"),
    ("(a)+①+② r2", ROOT / "results_clarify_v11_prebonus/prebonus_both_r2"),
    ("위약 s1", ROOT / "results_clarify_v11_prebonus/placebo_s1_r1"),
    ("위약 s2", ROOT / "results_clarify_v11_prebonus/placebo_s2_r1"),
    ("위약 s3", ROOT / "results_clarify_v11_prebonus/placebo_s3_r1"),
    ("위약 s4", ROOT / "results_clarify_v11_prebonus/placebo_s4_r1"),
]
POLICIES = ["reject_only", "oracle:vocal_gender", "oracle:genre", "oracle:type", "oracle:release_era", "rule:oracle",
            "noisy:vocal_gender", "noisy:genre", "noisy:type", "noisy:release_era",
            "flow:reject", "flow:oracle", "flow:noisy"]
QUERIES = ["m402", "c603"]


def cells(d: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for r in csv.DictReader(open(d / "clarify_detail.csv", encoding="utf-8-sig")):
        if r["policy"] not in POLICIES or r["query_id"] not in QUERIES:
            continue
        key = (r["query_id"], r["policy"])
        if key in out:  # m402 flow 대화 4개 중 첫 행(후보 안 67169)만
            continue
        rank = r["rank"] or "-"
        cand = r["candidate_rank@30"] or "-"
        turn = f" t{r['found_turn']}" if r["policy"].startswith("flow:") and r["found_turn"] not in ("", "None") else ""
        out[key] = f"{rank} ({cand}){turn}"
    return out


def run_meta(d: Path) -> tuple[dict, str]:
    """runinfo의 리랭커 상태 집계와 호출 경로. 실패가 있는 실행은 결과가 검색 순서 폴백이라 비교에서 빼야 한다 —
    10/9 위약 s2~s4가 AI Studio 크레딧 소진(402)으로 실패했는데 '시간 변동'으로, 18:38 base 변화는 Vertex 전환인데 시간 변동으로 잘못 읽었다."""
    path = d / "clarify_runinfo.json"
    if not path.exists():
        return {}, "?"
    import json
    info = json.load(open(path, encoding="utf-8"))
    counts = info.get("clarify", {}).get("rerank_status_counts") or {"(기록 없음)": ""}
    g = info.get("gemini") or {}
    # backend 실제 기록값은 api_key(AI Studio 키) / vertex / none — 문서 표기를 이 값에 맞춘다
    route = f"{g.get('backend', '?')}:{g.get('location', '')}".rstrip(":") if g else "api_key(기록 없음·v10 시기)"
    return counts, route


def excluded(counts: dict) -> str:
    if "(기록 없음)" in counts:
        return "  ← 상태 기록 없음(v10 로그상 실패 0) · 참고"
    bad = [k for k in ("failed", "exception", "no_status") if counts.get(k) not in (None, 0)]
    return f"  ← {'·'.join(bad)} 있음, 비교 제외" if bad else ""


def intervened(d: Path, q: str) -> bool:
    """첫 검색에서 정답을 못 찾아 재질문까지 간 질의인지(ran=1 행이 있는지)."""
    for r in csv.DictReader(open(d / "clarify_detail.csv", encoding="utf-8-sig")):
        if r["query_id"] == q and r["policy"] == "reject_only":
            return r["ran"] in ("1", "1.0", "True")
    return False


data = {label: (cells(d) if (d / "clarify_detail.csv").exists() else {}) for label, d in RUNS}
print("## 리랭커 상태 · 호출 경로 (runinfo) — failed·exception·상태 없음이 있으면 비교 제외, 경로가 다르면 비교하지 않는다")
for label, d in RUNS:
    sc, route = run_meta(d)
    print(f"  {label}: {route} {sc}{excluded(sc)}")
for q in QUERIES:
    print(f"\n## {q}  — 칸: 리랭킹 뒤 정답 순위 (리랭커 입력에서의 후보 순위), flow는 찾은 턴")
    print("| 정책 | " + " | ".join(l for l, _ in RUNS) + " |")
    print("|---|" + "---|" * len(RUNS))
    for pol in POLICIES:
        print(f"| {pol} | " + " | ".join(data[l].get((q, pol), "?") for l, _ in RUNS) + " |")

def losses(label: str, folder: Path, prefix: str) -> str:
    d = data[label]
    out, skipped = [], []
    for q in QUERIES:
        if not intervened(folder, q):
            skipped.append(q)  # 첫 검색에서 찾아 재질문이 없던 질의 — 0으로 세지 않고 따로 표시
            continue
        out += [(q, p) for p in POLICIES if p.startswith(prefix)
                and not d.get((q, "reject_only"), "-").startswith("-") and d.get((q, p), "-").startswith("-")]
    note = f"  (개입 없음: {skipped})" if skipped else ""
    return f"{len(out)} {out}{note}"


print("\n맞는 답(oracle:*)으로 reject_only 대비 Top-10을 잃은 (질의, 정책) — 위약에서는 '답을 받은 행이 순서만 흔들려 잃은' 수:")
for label, folder in RUNS:
    print(f"  {label}: {losses(label, folder, 'oracle:')}")

print("\n틀린 답(noisy:*) 행이 reject_only 대비 Top-10을 잃은 수 (같은 기준):")
for label, folder in RUNS:
    print(f"  {label}: {losses(label, folder, 'noisy:')}")
