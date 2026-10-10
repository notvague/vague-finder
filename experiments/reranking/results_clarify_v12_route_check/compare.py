"""호출 경로 확인 — AI Studio 재측정(base_r1·r2)을 v10 base(AI Studio)·v11 base_rerun(Vertex)과 행 단위로 비교. 레포 루트에서 실행.

읽는 파일: 각 폴더의 clarify_detail.csv(shown_ids · rank)와 clarify_runinfo.json(호출 경로 · 리랭커 상태 · 색인 해시).
비교 기준은 v10 base_r1(10/9 13:34, AI Studio). 행 키는 (질의, 정책, 목표 곡) — 52행.
**읽기 전 규칙**: 열마다 먼저 찍히는 호출 경로와 리랭커 상태를 본다. failed가 있으면 그 실행은 검색 순서 폴백이라 비교에서 뺀다.
v10은 경로·상태 기록 전이라 비어 있다(로그상 실패 0, AI Studio).
"""
import csv
import json
from pathlib import Path

R = Path("experiments/reranking")
REF = "v10 base_r1 (AI Studio, 10/9 13:34)"
RUNS = {
    REF: R / "results_clarify_v10_corrections/base_r1",
    "v10 base_r2 (AI Studio)": R / "results_clarify_v10_corrections/base_r2",
    "v11 base_rerun (Vertex, 10/9 18:38)": R / "results_clarify_v11_prebonus/base_rerun_r1",
    "v12 r1 (AI Studio, 10/10 00:02)": R / "results_clarify_v12_route_check/base_r1",
    "v12 r2 (AI Studio, 10/10 00:07)": R / "results_clarify_v12_route_check/base_r2",
}


def load(d: Path) -> dict:
    return {(r["query_id"], r["policy"], r["target_id"]): r
            for r in csv.DictReader(open(d / "clarify_detail.csv", encoding="utf-8-sig"))}


def meta(d: Path) -> str:
    j = json.load(open(d / "clarify_runinfo.json", encoding="utf-8"))
    g = j.get("gemini") or {}
    route = f"{g.get('backend')}·{g.get('setting', '-')}" if g else "기록 없음(v10 시기 — AI Studio)"
    lyr = j.get("lyrics_source", {})
    return (f"경로 {route} · 상태 {j.get('clarify', {}).get('rerank_status_counts') or '기록 없음'} · "
            f"색인 {j.get('corpus', {}).get('point_count')} · bm25 {j.get('bm25', {}).get('sha256_12')} · "
            f"가사 {lyr.get('document_count')}/{lyr.get('content_sha256_16')}")


def diff(a: dict, b: dict) -> tuple[list, list]:
    keys = sorted(set(a) | set(b))
    shown = [k for k in keys if (a.get(k) or {}).get("shown_ids") != (b.get(k) or {}).get("shown_ids")]
    rank = [k for k in keys if (a.get(k) or {}).get("rank") != (b.get(k) or {}).get("rank")]
    return shown, rank


avail = {k: d for k, d in RUNS.items() if (d / "clarify_detail.csv").exists()}
print("## 실행 조건 (runinfo)")
for k, d in avail.items():
    print(f"  {k}: {meta(d)}")

ref = load(avail[REF])
print(f"\n## {REF} 대비 — Top-10(shown_ids)이 다른 행 · 정답 순위가 다른 행 (52행)")
for k, d in avail.items():
    if k == REF:
        continue
    x = load(d)
    shown, rank = diff(ref, x)
    print(f"  {k}: Top-10 {len(shown)}/{len(set(ref) | set(x))} · 순위 {len(rank)}")
    if len(shown) <= 5:  # 몇 행뿐이면 어느 자리가 바뀌었는지까지
        for key in shown:
            a, b = ref[key]["shown_ids"].split("|"), x[key]["shown_ids"].split("|")
            places = [i + 1 for i, (p, q) in enumerate(zip(a, b)) if p != q]
            print(f"      {key[0]} {key[1]}: 정답 순위 {ref[key]['rank'] or '-'} → {x[key]['rank'] or '-'}, 바뀐 자리 {places}")

r1, r2 = "v12 r1 (AI Studio, 10/10 00:02)", "v12 r2 (AI Studio, 10/10 00:07)"
if r1 in avail and r2 in avail:
    shown, rank = diff(load(avail[r1]), load(avail[r2]))
    print(f"\n## v12 r1 vs r2: Top-10 {len(shown)}/52 · 순위 {len(rank)} {[k[:2] for k in shown]}")
