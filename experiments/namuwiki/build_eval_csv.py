"""Context 평가 원본 JSON에서 기존 검색 평가기와 호환되는 CSV를 만든다.

프로젝트 루트에서:
    python experiments/namuwiki/build_eval_csv.py
    python experiments/namuwiki/build_eval_csv.py --check

--check는 현재 Context artifact에 정답 곡과 근거가 남아 있는지도 검사한다.
재크롤링/재생성 후 근거가 바뀌었다면 라벨을 재검토하고 JSON을 수정할 것.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "queries_context_v01.json"
OUTPUT = HERE / "eval_queries_context_v01.csv"
ARTIFACTS = ROOT / "artifacts/context/songs"
# src.eval.export_csv.COLUMNS 및 eval_queries_v05.csv와 같은 열과 순서.
COLUMNS = ["query_id", "split", "query_type", "query", "relevant_ids"]


def build_rows() -> tuple[list[dict[str, str]], Counter[str], Counter[str]]:
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    if raw.get("version") != "context_v01":
        raise ValueError("평가 원본 버전이 다릅니다")
    cases = raw["queries"]
    if len(cases) != 30:
        raise ValueError(f"평가 질의는 30개여야 합니다: {len(cases)}")

    seen_ids: set[str] = set()
    seen_songs: set[str] = set()
    rows: list[dict[str, str]] = []
    splits: Counter[str] = Counter()
    segments: Counter[str] = Counter()
    for case in cases:
        query_id = case["query_id"]
        song_id = str(case["song_id"])
        split = case["split"]
        segment = case["segment"]
        if query_id in seen_ids or song_id in seen_songs:
            raise ValueError(f"중복 query_id 또는 곡: {query_id}, {song_id}")
        if not query_id.startswith("nw") or not query_id[2:].isdigit():
            raise ValueError(f"잘못된 query_id: {query_id}")
        if split not in {"dev", "test"} or segment not in {"context", "mixed", "control"}:
            raise ValueError(f"잘못된 split/segment: {query_id}")
        if not song_id.isdigit() or not case["query"].strip():
            raise ValueError(f"잘못된 정답/질의: {query_id}")

        path = ARTIFACTS / f"{song_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"[{query_id}] 정답 곡 artifact 없음: {path}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if str(artifact["song"]["song_id"]) != song_id:
            raise ValueError(f"[{query_id}] artifact 곡 ID 불일치")

        record_id = case["evidence_record_id"]
        additional_ids = case.get("additional_evidence_record_ids", [])
        if segment == "control":
            if record_id is not None or additional_ids or case["clue_category"] != "no_context":
                raise ValueError(f"[{query_id}] 대조군에 Context 근거가 붙었습니다")
            if artifact["status"] not in {"no_trivia", "not_found"}:
                raise ValueError(f"[{query_id}] 대조군 상태가 바뀌었습니다: {artifact['status']}")
            if artifact["retrieval"]["records"]:
                raise ValueError(f"[{query_id}] 대조군에 Context 사실이 생겼습니다")
        else:
            if artifact["status"] != "ok" or not artifact["source"].get("url"):
                raise ValueError(f"[{query_id}] Context 원천이 없습니다")
            evidence_ids = [record_id, *additional_ids]
            if not record_id or len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError(f"[{query_id}] 근거가 없거나 중복입니다")
            by_id = {r["record_id"]: r for r in artifact["retrieval"]["records"]}
            if record_id not in by_id or by_id[record_id]["category"] != case["clue_category"]:
                raise ValueError(f"[{query_id}] 해당 곡의 사실 ID/분류가 맞지 않습니다: {record_id}")
            for evidence_id in evidence_ids:
                if evidence_id not in by_id or not by_id[evidence_id]["evidence_text"].strip():
                    raise ValueError(f"[{query_id}] 추가 사실 근거가 없습니다: {evidence_id}")

        seen_ids.add(query_id)
        seen_songs.add(song_id)
        splits[split] += 1
        segments[segment] += 1
        rows.append({
            "query_id": query_id,
            "split": split,
            "query_type": "search",
            "query": case["query"],
            "relevant_ids": song_id,
        })
    if splits != {"dev": 21, "test": 9} or segments != {"context": 20, "mixed": 6, "control": 4}:
        raise ValueError(f"평가 분할이 예상과 다릅니다: {splits}, {segments}")
    return rows, splits, segments


def render_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="정답 근거와 생성된 CSV의 일치 여부만 확인")
    args = parser.parse_args()
    rows, splits, segments = build_rows()
    expected = render_csv(rows)
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes().decode("utf-8") != expected:
            raise SystemExit(f"CSV가 원본 JSON과 다릅니다: {OUTPUT}\n재생성: python experiments/namuwiki/build_eval_csv.py")
        action = "검증 완료"
    else:
        OUTPUT.write_text(expected, encoding="utf-8", newline="")
        action = "CSV 생성 완료"
    print(f"{action}: {OUTPUT.relative_to(ROOT)} — {len(rows)}개, split={dict(splits)}, segment={dict(segments)}")


if __name__ == "__main__":
    main()
