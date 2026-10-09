"""재질문 측정 결과의 행(질의 × 정책 × 목표 곡)을 **변형이 리랭커 입력을 바꾸는지**로 셋으로 나눈다.

dev 57에서 ①(`GEMINI_RERANK_CORRECTIONS=new_only`)+②(`GEMINI_RERANK_TYPE_SLOT_LABEL`)를 함께 재더라도
효과를 따로 세고, 입력이 그대로인 행의 차이는 실행 변동으로 분리하기 위한 것이다(리뷰).

- `corrections`: 그 행의 답(맞는 답이든 틀린 답이든) 중 하나라도 원래 질의 분석과 같은 값 → ①이 프롬프트에서 뺀다.
  **틀린 답도 빠진다** — 분석이 틀렸고 틀린 답이 그 분석과 같으면 "확인용"이다(dev c701·c705 남성, test c704 여성·q118 여성·발라드).
  그런 행에서 flow:noisy가 바뀌면 변동이 아니라 ①의 효과다.
- `type_label`: 답 중 type 슬롯이 있음 → ②가 문구를 바꾼다. 단 그 type 답이 분석과 같은 값이면 ①+② 측정에서는 ①이 먼저
  빼서 ②의 문구가 닿지 않으므로 `corrections`로만 센다(q205 솔로, q110·q112·q210 그룹).
- `both` / `unchanged` / `not_run`: ran=0 행(개입 대상이 아니거나 슬롯 값이 없어 다른 결과가 복사된 행)은 답을 보지 않고
  `not_run`이다 — 거기서 생기는 차이는 첫 검색의 실행 변동뿐이다.
- `oracle:best`는 따로 검색하지 않고 그 질의의 oracle:* 중 가장 좋은 것을 고르므로(하네스 add_oracle_best), 같은 질의에서
  실행된 oracle:* 행들의 분류를 합쳐 붙인다.
- flow 3턴에 무엇을 물을지는 2턴 결과에 따라 정해져 두 측정에서 답이 다를 수 있다. `--other-dir`로 다른 측정도 함께 분류하면
  **양쪽 모두 unchanged/not_run인 행**만 노이즈 기준으로 센다.

답의 출처: flow 정책은 `clarify_turns.csv`(answer_kind·slot·answer_value), 한 번 답변 정책(oracle:*·noisy:*·rule:*)은 하네스와 같은
`oracle_value`/`noisy_value`(정답 곡 메타데이터, 로컬 코퍼스 필요). 코퍼스가 없으면 flow 행만 분류한다.

실행 (레포 루트):
    venv/bin/python experiments/reranking/results_clarify_v10_corrections/classify_rows_by_switch.py \\
        --result-dir experiments/reranking/results_clarify_v09 --analysis-cache experiments/reranking/analysis_cache_v06_dev.json
출력: 이 폴더의 rows_by_switch_<결과 폴더 이름>.csv(--out으로 변경)와 요약. 커밋된 결과 폴더에는 쓰지 않는다.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.backend.schemas.query import QueryAnalysis  # noqa: E402
from src.backend.schemas.search import ClarifyAnswer  # noqa: E402
from src.eval.loader import DEFAULT_EVAL_PATH, load_eval_set  # noqa: E402
from src.retrieval.clarify import answer_confirms_analysis  # noqa: E402
from src.retrieval.evaluate_clarification import DEFAULT_CORPUS, load_corpus, noisy_value, oracle_value  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--analysis-cache", required=True)
    p.add_argument("--queries", default=str(DEFAULT_EVAL_PATH))
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--other-dir", default="", help="비교 상대 측정 폴더. 주면 양쪽 모두 바뀌지 않은 행 수를 함께 센다")
    p.add_argument("--out", default="", help="출력 CSV. 기본은 이 스크립트 폴더의 rows_by_switch_<결과 폴더 이름>.csv — 커밋된 결과 폴더를 건드리지 않는다")
    return p.parse_args()


def _one_shot_answer(policy: str, song: dict | None, picked_slot: str) -> ClarifyAnswer | None:
    """한 번 답변 정책의 답 — 하네스 evaluate_query와 같은 규칙."""
    if song is None:
        return None
    if policy.startswith("oracle:") and policy != "oracle:best":
        slot = policy.split(":", 1)[1]
        v = oracle_value(song, slot)
        return ClarifyAnswer(slot=slot, value=v) if v else None
    if policy.startswith("noisy:"):
        slot = policy.split(":", 1)[1]
        v = noisy_value(song, slot)
        return ClarifyAnswer(slot=slot, value=v) if v else None
    if policy in ("rule:oracle", "rule:noisy") and picked_slot:
        v = oracle_value(song, picked_slot) if policy == "rule:oracle" else noisy_value(song, picked_slot)
        return ClarifyAnswer(slot=picked_slot, value=v) if v else None
    return None


def classify(rdir: Path, entries: dict, positives: dict, corpus: dict) -> list[dict]:
    flow_answers: dict[tuple[str, str, str], list[ClarifyAnswer]] = defaultdict(list)
    for t in csv.DictReader(open(rdir / "clarify_turns.csv", encoding="utf-8-sig")):
        if t["answer_kind"] in ("oracle", "noisy") and t["answer_value"]:
            flow_answers[(t["query_id"], t["policy"], t["target_id"])].append(
                ClarifyAnswer(slot=t["slot"], value=t["answer_value"])
            )

    rows_out: list[dict] = []
    for r in csv.DictReader(open(rdir / "clarify_detail.csv", encoding="utf-8-sig")):
        qid, policy = r["query_id"], r["policy"]
        ran = r["ran"] in ("1", "1.0", "True")
        row = {"query_id": qid, "policy": policy, "target_id": r["target_id"], "ran": int(ran), "answers": "", "confirming": "", "group": "not_run"}
        if ran and policy != "oracle:best":
            entry = entries.get(qid)
            analysis = QueryAnalysis(**(entry.get("analysis", entry))) if entry else None
            if policy.startswith("flow:"):
                answers = flow_answers.get((qid, policy, r["target_id"]), [])
            else:
                target = positives.get(qid, [None])[0]
                one = _one_shot_answer(policy, corpus.get(target) if corpus and target else None, r["picked_slot"])
                answers = [one] if one else []
            confirming = [a for a in answers if analysis is not None and answer_confirms_analysis(analysis, a)]
            typed = [a for a in answers if a.slot in ("type", "artist_type") and a not in confirming]
            row["answers"] = "|".join(f"{a.slot}={a.value}" for a in answers)
            row["confirming"] = "|".join(f"{a.slot}={a.value}" for a in confirming)
            row["group"] = "both" if confirming and typed else "corrections" if confirming else "type_label" if typed else "unchanged"
        rows_out.append(row)

    # oracle:best = 같은 질의의 실행된 oracle:* 분류의 합
    by_query: dict[str, set[str]] = defaultdict(set)
    for x in rows_out:
        if x["policy"].startswith("oracle:") and x["policy"] != "oracle:best" and x["ran"]:
            by_query[x["query_id"]].add(x["group"])
    for x in rows_out:
        if x["policy"] == "oracle:best" and x["ran"]:
            g = by_query.get(x["query_id"], set())
            has_c = bool(g & {"corrections", "both"}); has_t = bool(g & {"type_label", "both"})
            x["group"] = "both" if has_c and has_t else "corrections" if has_c else "type_label" if has_t else "unchanged"
    return rows_out


def main() -> None:
    args = _parse()
    rdir = Path(args.result_dir)
    cache = json.load(open(args.analysis_cache, encoding="utf-8"))
    entries = cache.get("entries") or cache.get("queries")
    positives = {q.query_id: q.positives for q in load_eval_set(Path(args.queries)).queries}
    corpus = load_corpus(Path(args.corpus)) if Path(args.corpus).exists() else {}

    rows_out = classify(rdir, entries, positives, corpus)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / f"rows_by_switch_{rdir.name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    counts = Counter(x["group"] for x in rows_out)
    executed = [x for x in rows_out if x["ran"]]
    changed = [x for x in executed if x["group"] in ("corrections", "type_label", "both")]
    print(f"저장: {out} ({len(rows_out)}행)  {dict(counts)}")
    print(f"  실행된 행 {len(executed)} 중 ①·②가 입력을 바꾸는 행 {len(changed)} "
          f"(① {sum(x['group']=='corrections' for x in changed)} · ② {sum(x['group']=='type_label' for x in changed)} · 둘 다 {sum(x['group']=='both' for x in changed)})")
    for kind in ("oracle", "noisy"):
        hit = sorted({f"{x['query_id']} {x['confirming']}" for x in rows_out
                      if x["confirming"] and x["policy"].startswith("flow:") and x["policy"].endswith(kind)})
        print(f"  flow:{kind}에서 ①이 빼는 답: {hit}")
    if args.other_dir:
        other = {(x["query_id"], x["policy"], x["target_id"]): x["group"] for x in classify(Path(args.other_dir), entries, positives, corpus)}
        quiet = [x for x in rows_out if x["group"] in ("unchanged", "not_run")
                 and other.get((x["query_id"], x["policy"], x["target_id"])) in ("unchanged", "not_run")]
        print(f"  양쪽({rdir.name} · {Path(args.other_dir).name}) 모두 바뀌지 않은 행(노이즈 기준): {len(quiet)} / {len(rows_out)}")
    if not corpus:
        print("  [경고] 코퍼스 없음 — 한 번 답변 정책은 답을 몰라 unchanged로 분류됐다")


if __name__ == "__main__":
    main()
