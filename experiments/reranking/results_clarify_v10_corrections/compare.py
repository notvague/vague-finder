"""m402·c603 답변 변형 비교 — 변형별 2회, 정책별 정답 순위(리랭킹 뒤 Top-10 안 순위, 밖이면 '-').

읽는 파일: 각 폴더의 clarify_detail.csv(rank · candidate_rank@30), clarify_turns.csv(flow 턴별). 레포 루트에서 실행.
"""
import csv
from pathlib import Path

ROOT = Path("experiments/reranking/results_clarify_v10_corrections")
VARIANTS = ["base", "new_only", "type_label", "both"]
REPS = (1, 2)
POLICIES = ["reject_only", "oracle:vocal_gender", "oracle:genre", "oracle:type", "oracle:release_era", "rule:oracle",
            "flow:reject", "flow:oracle", "flow:noisy"]
QUERIES = ["m402", "c603"]


def ranks(d: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], list[str]] = {}
    for r in csv.DictReader(open(d / "clarify_detail.csv", encoding="utf-8-sig")):
        if r["policy"] in POLICIES and r["query_id"] in QUERIES:
            rank = r["rank"] or "-"
            if r["policy"].startswith("flow:"):
                rank += f"(t{r['found_turn']})" if r["found_turn"] not in ("", "None") else ""
            out.setdefault((r["query_id"], r["policy"]), []).append(rank)
    # m402는 정답이 4곡이라 flow 행이 4개다(detail 순서대로 '/'로 잇는다. 첫 칸이 후보 안의 67169 대화). c603은 1곡
    return {k: "/".join(v) for k, v in out.items()}


cols = [f"{v} r{rep}" for v in VARIANTS for rep in REPS]
data = {}
for v in VARIANTS:
    for rep in REPS:
        d = ROOT / f"{v}_r{rep}"
        data[f"{v} r{rep}"] = ranks(d) if (d / "clarify_detail.csv").exists() else {}

for q in QUERIES:
    print(f"\n## {q}  (정답 순위 — Top-10 밖이면 '-', flow는 (찾은 턴))")
    print("| 정책 | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for pol in POLICIES:
        print(f"| {pol} | " + " | ".join(data[c].get((q, pol), "?") for c in cols) + " |")

# 손실 수: reject_only에서 Top-10 안인데 oracle:*에서 밖
print("\n맞는 답(oracle:*)으로 reject_only 대비 Top-10을 잃은 (질의, 정책) 수:")
for c in cols:
    d = data[c]
    lost = [(q, p) for q in QUERIES for p in POLICIES if p.startswith("oracle:")
            and d.get((q, "reject_only"), "-") != "-" and d.get((q, p), "-") == "-"]
    print(f"  {c}: {len(lost)} {lost}")
