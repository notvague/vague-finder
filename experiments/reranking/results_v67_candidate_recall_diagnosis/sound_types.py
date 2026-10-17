"""소리 단서 유형 — RUN_INFO §5에 결과 전 적은 정의를 질의 **원문**에 적용한다(정답·검색 결과를 보지 않는다).

대상: A(on_rep1)에서 오디오 조회가 실행된 질의. 여러 유형이면 앞 번호 우선.
  ① named_event  이름 붙은 소리 사건(사이렌·물소리·박수·휘파람·경적·효과음·긁는 소리 …)
  ② instrument   악기·편성(피아노·기타·현악·드럼·베이스·전자음·반주 …)
  ③ vocal        음색·창법·보컬(목소리·고음·가성·허스키·콧소리·랩 …)
  ④ genre_mood   위에 해당하지 않음(장르·분위기·템포만)

    venv/bin/python sound_types.py --run runs/on_rep1 --out analysis/sound_types.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETS = ("v06", "v09", "v10")
RULES = (
    ("named_event", ("사이렌", "물소리", "물 흐르", "물흐르", "빗소리", "박수", "휘파람", "전화벨", "벨소리", "경적", "새소리",
                     "파도 소리", "파도소리", "종소리", "효과음", "긁는 소리", "총소리", "발소리", "웃음소리", "알람")),
    ("instrument", ("피아노", "기타", "드럼", "베이스", "현악", "스트링", "오케스트라", "브라스", "신스", "전자음", "반주", "악기",
                    "바이올린", "첼로", "하모니카", "색소폰", "관악", "퍼커션", "밴드", "리프", "비트", "뿅뿅")),
    ("vocal", ("목소리", "보컬", "음색", "창법", "고음", "가성", "생목", "허스키", "미성", "콧소리", "랩", "래퍼", "떼창",
               "쇳소리", "절규", "중저음", "내지르", "지르", "부르는", "불러",
               # 유형별 효과를 보기 전 표본 점검에서 빠진 것을 더함(RUN_INFO §6 앞 기록)
               "바이브레이션", "기교", "소프라노", "성악", "화음", "높은 음", "높아서", "음을 길게")),
)


def classify(text: str) -> str:
    t = re.sub(r"\s+", " ", text)
    for name, words in RULES:
        if any(w in t for w in words):
            return name
    return "genre_mood"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    queries = {}
    for s in SETS:
        for r in csv.DictReader(open(ROOT / f"experiments/reranking/eval_queries_{s}.csv", encoding="utf-8-sig")):
            if r["split"].strip() == "dev":
                queries[r["query_id"].strip()] = r["query"].strip()
    run = HERE / args.run if not args.run.is_absolute() else args.run
    rows = []
    for s in SETS:
        for line in open(run / f"{s}_dev" / "v67_queries.jsonl", encoding="utf-8"):
            d = json.loads(line)
            rows.append({"query_id": d["query_id"], "audio_ran": int(d["audio_queries"] > 0),
                         "sound_type": classify(queries[d["query_id"]]) if d["audio_queries"] > 0 else ""})
    if len(rows) != 176 or len({r["query_id"] for r in rows}) != 176:
        raise SystemExit("질의 수 이상")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(Counter(r["sound_type"] for r in rows if r["audio_ran"]))


if __name__ == "__main__":
    main()
