"""재질문 평가 비교. 기본값은 CE 기준선(v08 dev · test_v05) vs Gemini listwise 기본(v09 dev · test_v06)이고 --dev/--test/--labels로 쌍을 바꾼다.

커밋되는 clarify_summary.csv(정책별 집계)와 clarify_detail.csv(목표 곡별 결과)를 읽는다.
후보 순위(candidate_rank@30)는 보너스 순서의 후보 목록 기준이다 — (a) 스위치는 리랭커 입력만 바꾸므로 측정 간 기준이 같다. 레포 루트에서 실행.
비교는 **질의 단위**(복수 정답 질의는 목표 곡별 평균)로 한다 — (질의, 정책, 목표 곡) 키로 맞추면 개입 여부가
두 측정 사이에 바뀐 질의(목표 곡 행이 없는 쪽)가 조용히 빠진다(리뷰). 개입 대상 수가 다르므로
답변의 효과는 **양쪽 모두 개입 대상인 같은 질의**에서 따로 센다.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path("experiments/reranking")
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--dev", nargs=2, metavar=("BASE", "NEW"), default=("results_clarify_v08", "results_clarify_v09"),
                 help="dev 비교 쌍 폴더 이름(experiments/reranking 아래). 기본 CE v08 → listwise v09")
_ap.add_argument("--test", nargs=2, metavar=("BASE", "NEW"), default=("results_clarify_test_v05", "results_clarify_test_v06"),
                 help="test 비교 쌍 폴더 이름. 기본 CE test_v05 → listwise test_v06")
_ap.add_argument("--labels", default="CE,LW", help="두 측정의 표시 이름(쉼표). 예: v09,new_only")
_args = _ap.parse_args()
LABEL_B, LABEL_N = [x.strip() for x in _args.labels.split(",", 1)]
PAIRS = [
    ("dev 57", ROOT / _args.dev[0], ROOT / _args.dev[1]),
    ("test 25", ROOT / _args.test[0], ROOT / _args.test[1]),
]
COLS = ["n_executed", "n_conversations", "full_hit@1", "full_hit@10", "int_hit@1", "int_hit@10", "int_candidate_recall@30", "int_found_by_turn2", "n_conv_out_of_candidates"]
FLOWS = ("flow:reject", "flow:oracle", "flow:noisy")


def summary(d: Path) -> dict[str, dict]:
    return {r["policy"]: r for r in csv.DictReader(open(d / "clarify_summary.csv", encoding="utf-8-sig"))}


def hit10_by_query(d: Path) -> dict[tuple[str, str], float]:
    """(query_id, policy) → Hit@10, 목표 곡별 평균."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in csv.DictReader(open(d / "clarify_detail.csv", encoding="utf-8-sig")):
        v = r["hit10"]
        acc[(r["query_id"], r["policy"])].append(1.0 if v in ("True", "1", "1.0") else 0.0)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def rank_by_query(d: Path, policy: str) -> dict[str, str]:
    """(query_id) → 리랭킹 뒤 정답 순위(목표 곡별 '/' 연결, 밖이면 '-').

    개입 대상이 아닌 질의(첫 검색에서 찾음)는 detail에 ran=0 행이 들어가고 첫 검색 결과가 복사된다.
    그걸 거절 뒤 순위처럼 읽으면 안 되므로 '첫 검색 N'으로 표시한다 (리뷰).
    """
    out: dict[str, list[str]] = defaultdict(list)
    for r in csv.DictReader(open(d / "clarify_detail.csv", encoding="utf-8-sig")):
        if r["policy"] == policy:
            rank = r["rank"] or "-"
            out[r["query_id"]].append(rank if r["ran"] in ("1", "1.0", "True") else f"첫 검색 {rank}")
    return {q: "/".join(v) for q, v in out.items()}


def fmt(v: str) -> str:
    if v in ("", None):
        return "-"
    try:
        f = float(v)
    except ValueError:
        return v
    return f"{f:.3f}" if f != int(f) else str(int(f))


for name, base_dir, new_dir in PAIRS:
    if not (new_dir / "clarify_summary.csv").exists():
        print(f"{name}: 아직 없음")
        continue
    b, n = summary(base_dir), summary(new_dir)
    print(f"\n## {name}  (개입 대상 {LABEL_B} {b['initial']['n_intervention']} → {LABEL_N} {n['initial']['n_intervention']})")
    print("| 정책 | " + " | ".join(f"{c} {LABEL_B}→{LABEL_N}" for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for pol in n:
        if pol in b:
            print(f"| {pol} | " + " | ".join(f"{fmt(b[pol][c])}→{fmt(n[pol][c])}" for c in COLS) + " |")

    hb, hn = hit10_by_query(base_dir), hit10_by_query(new_dir)
    queries = sorted({q for q, _ in hn})
    # 정책 행이 빠진 질의는 조용히 0으로 세지 않고 멈춘다 (리뷰)
    for pol in ("initial", *FLOWS):
        missing = [q for q in queries if (q, pol) not in hb or (q, pol) not in hn]
        assert not missing, f"{name} {pol}: 정책 행이 없는 질의 {missing}"
    int_b = {q for q in queries if hb[(q, "initial")] < 1.0}
    int_n = {q for q in queries if hn[(q, "initial")] < 1.0}
    common = sorted(int_b & int_n)
    print(f"\n개입 대상: {LABEL_B} {len(int_b)} · {LABEL_N} {len(int_n)} · 공통 {len(common)} · {LABEL_B}만 {sorted(int_b - int_n)} · {LABEL_N}만 {sorted(int_n - int_b)}")

    print("\n질의 단위 Hit@10 변화 (전체 질의, 목표 곡별 평균):")
    for pol in FLOWS:
        gain = [q for q in queries if hb[(q, pol)] < hn[(q, pol)]]
        loss = [q for q in queries if hb[(q, pol)] > hn[(q, pol)]]
        print(f"  {pol:12s} 회복 {len(gain)} {gain} / 손실 {len(loss)} {loss}")

    # Hit@10만 보면 Top-10 안에서의 순위 변화(q205 6→2위 같은 확인용 답의 이득)가 안 보인다 (리뷰).
    print(f"\n개입 대상 질의별 순위 (reject_only → oracle:vocal_gender → oracle:genre → flow:reject → flow:oracle), {LABEL_B} | {LABEL_N}:")
    cols_r = ("reject_only", "oracle:vocal_gender", "oracle:genre", "flow:reject", "flow:oracle")
    rb = {pol: rank_by_query(base_dir, pol) for pol in cols_r}
    rn = {pol: rank_by_query(new_dir, pol) for pol in cols_r}
    for q in sorted(int_b | int_n):
        print(f"  {q}: " + " → ".join(rb[pol].get(q, "?") for pol in cols_r) + " | " + " → ".join(rn[pol].get(q, "?") for pol in cols_r))
    print(f"\n같은 질의만 비교 (양쪽 모두 개입 대상인 {len(common)}건, 3턴 안에 찾은 질의 수):")
    print(f"| 정책 | {LABEL_B} | {LABEL_N} |\n|---|---|---|")
    for pol in FLOWS:
        print(f"| {pol} | {sum(hb[(q, pol)] for q in common):.2f} | {sum(hn[(q, pol)] for q in common):.2f} |")
    for label, pol in (("맞는 답의 이득 (oracle − reject)", "flow:oracle"), ("틀린 답의 손해 (noisy − reject)", "flow:noisy")):
        db = sum(hb[(q, pol)] - hb[(q, "flow:reject")] for q in common)
        dn = sum(hn[(q, pol)] - hn[(q, "flow:reject")] for q in common)
        who_b = [q for q in common if hb[(q, pol)] != hb[(q, "flow:reject")]]
        who_n = [q for q in common if hn[(q, pol)] != hn[(q, "flow:reject")]]
        print(f"| {label} | {db:+.2f} {who_b} | {dn:+.2f} {who_n} |")

    # 맞는 답이 정답을 Top-10 밖으로 밀어낸 경우 (reject_only 대비 oracle:*)
    for label, d in ((LABEL_B, base_dir), (LABEL_N, new_dir)):
        rows = list(csv.DictReader(open(d / "clarify_detail.csv", encoding="utf-8-sig")))
        base = {(r["query_id"], r["target_id"]): r["hit10"] for r in rows if r["policy"] == "reject_only"}
        hurt = sorted({(r["query_id"], r["policy"]) for r in rows if r["policy"].startswith("oracle:")
                       and base.get((r["query_id"], r["target_id"])) in ("True", "1", "1.0") and r["hit10"] in ("False", "0", "0.0")})
        print(f"맞는 답(oracle:*)이 거절만(reject_only)보다 정답을 Top-10 밖으로 보낸 경우 — {label}: {len(hurt)} {hurt}")
