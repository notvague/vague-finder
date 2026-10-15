"""재사용 검증 — 이 폴더의 실행과 v53 실행이 같은 조건·같은 후보 지문인지(모델·DB 없음).
질의마다 detail CSV의 candidate_digest · candidate_rank@30 · rerank_top_ids를 대조하고, runinfo의 코퍼스·BM25·Mongo·캐시·스위치를 비교한다.

    venv/bin/python <이 파일> cand_attr0_eraplus results_v53_attr_filter_no_genre/cand_attr0_eraplus [--sets v06,v09,v10]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent


def folder_of(tag: str) -> Path:
    if tag.startswith("/"):
        return Path(tag)  # 다른 워크트리의 실행(detail·explain은 git 제외라 그 워크트리에만 있다)
    return (HERE.parent / tag) if "/" in tag else (HERE / tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mine")
    ap.add_argument("theirs")
    ap.add_argument("--sets", default="v06,v09,v10")
    args = ap.parse_args()
    for s in args.sets.split(","):
        a, b = folder_of(args.mine) / f"{s}_dev", folder_of(args.theirs) / f"{s}_dev"
        if not (a / "search_eval_dev_detail.csv").exists() or not (b / "search_eval_dev_detail.csv").exists():
            print(f"{s}: detail 없음 (건너뜀)")
            continue
        da = {r["query_id"]: r for r in csv.DictReader(open(a / "search_eval_dev_detail.csv", encoding="utf-8-sig"))}
        db = {r["query_id"]: r for r in csv.DictReader(open(b / "search_eval_dev_detail.csv", encoding="utf-8-sig"))}
        same_digest = sum(1 for q in da if da[q]["candidate_digest"] == db.get(q, {}).get("candidate_digest"))
        same_top = sum(1 for q in da if da[q].get("rerank_top_ids") == db.get(q, {}).get("rerank_top_ids"))
        same_cand = sum(1 for q in da if da[q]["candidate_rank@30"] == db.get(q, {}).get("candidate_rank@30"))
        diff = [q for q in da if da[q]["candidate_digest"] != db.get(q, {}).get("candidate_digest")]
        ia = json.loads((a / "search_eval_dev_runinfo.json").read_text(encoding="utf-8"))
        ib = json.loads((b / "search_eval_dev_runinfo.json").read_text(encoding="utf-8"))
        keys = {}
        for k in ("corpus", "bm25", "lyrics_source", "analysis", "gemini"):
            keys[k] = ia.get(k) == ib.get(k)
        swa, swb = ia.get("ranking_switches", {}), ib.get("ranking_switches", {})
        sw_diff = {k: (swa.get(k), swb.get(k)) for k in set(swa) | set(swb) if swa.get(k) != swb.get(k)}
        print(f"{s}: 질의 {len(da)}/{len(db)} · 후보 지문 같음 {same_digest} · top-10 같음 {same_top} · 후보 순위 같음 {same_cand}"
              f" · 다른 지문 {diff or '없음'}")
        print(f"    runinfo 동일(corpus/bm25/lyrics_source/analysis/gemini): {keys} · 스위치 차이: {sw_diff or '없음'}")
        print(f"    corpus={ia.get('corpus')} bm25={ia.get("bm25", {}).get("sha256_12")} lyrics={ia.get("lyrics_source", {}).get("content_sha256_16")}")


if __name__ == "__main__":
    main()
