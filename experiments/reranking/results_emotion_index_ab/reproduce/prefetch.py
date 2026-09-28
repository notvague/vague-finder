"""평가 질의 분석을 한 번만 받아 저장한다. A/B 두 조건이 같은 분석을 쓰게 하기 위함."""
import csv, json, os, sys, time
from pathlib import Path
from src.backend.api.dependencies import get_query_analyzer
from src.backend.schemas.query import QueryAnalysis

rows = list(csv.DictReader(open("experiments/reranking/eval_queries_v05.csv", encoding="utf-8-sig")))
out_path = Path(sys.argv[1])
cache = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
# 키가 없으면 분석기가 예외 없이 fallback(confidence=0.0)을 돌려줘서, 그대로 저장하면
# 두 조건에 Gemini 분석이 아닌 값이 주입된다. 먼저 막는다.
if not os.getenv("GEMINI_API_KEY"):
    raise SystemExit("GEMINI_API_KEY 미설정: $REPO/.env를 읽은 뒤 실행할 것 (fallback 분석 저장 방지)")
analyzer = get_query_analyzer()
for i, r in enumerate(rows, 1):
    q = r["query"]
    if q in cache: continue
    for attempt in range(3):
        try:
            a = analyzer.analyze(q)
            if a.confidence == 0.0:
                raise RuntimeError("fallback 분석(confidence=0.0)")
            break
        except Exception as e:
            print("재시도", r["query_id"], e); time.sleep(5)
    else:
        raise SystemExit(f"분석 실패: {r['query_id']}")
    dumped = a.model_dump(mode="json")
    assert QueryAnalysis.model_validate(dumped).model_dump(mode="json") == dumped, "왕복 불일치"
    cache[q] = dumped
    out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{i}/{len(rows)} {r['query_id']} {a.intent_type}", flush=True)
print("완료", len(cache))
