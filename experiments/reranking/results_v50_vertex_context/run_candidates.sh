#!/bin/zsh
# v50 후보 단계(리랭킹 끔 · Gemini 호출 없음) — 완성 Context 색인 복사본에서 Context 경로 끔(A)/켬(B)을 dev 176건으로 잰다.
#   CTX=0   : A — CONTEXT_WEIGHT=0 → 라우터가 Context 조회 자체를 하지 않는다(search_router.use_context_path, 융합 절단도 기본 폭)
#   CTX=0.5 : B — 현행 기본값(가중 0.5 · 작품명 배율 2.0 · 폭 100/100 · 후보 30), 튜닝 없음
#   COPY=a|b : 색인 복사본(artifacts/qdrant_eval_a · _b, 서비스 색인 artifacts/qdrant의 파일 복사 — 한 프로세스가 하나만 연다)
#   SETS="v06 v09 v10"
# 분석은 v49의 Vertex 캐시(analysis_cache_*_dev_vertex.json)를 그대로 읽는다 — 지문이 현행 코드와 같음을 확인했다(RUN_INFO §1).
# 사용량 탭(usage_tap.py)을 끼워 "모델 호출 0회"를 기록 파일로 증명한다(파일이 없거나 비어 있으면 호출 0).
#   예) CTX=0 COPY=a zsh run_candidates.sh → cand_ctx0/{v06,v09,v10}_dev
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v50_vertex_context
CACHES=experiments/reranking/results_v49_vertex_baseline
PY=${PY:-venv/bin/python}
CTX=${CTX:-0.5}
COPY=${COPY:-a}
SETS=(${=SETS:-v06 v09 v10})
export QDRANT_PATH=artifacts/qdrant_eval_$COPY
export CONTEXT_WEIGHT=$CTX AUDIO_FUSION_SCALE=1
export RERANKER_ENABLED=false RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false GEMINI_RETRIEVAL_BACKEND=vertex
unset CLAUDE_EFFORT CLAUDE_API
name=cand_ctx${CTX}
for set in $SETS; do
  GEMINI_USAGE_TAP=$OUT/usage_${name}_$set.jsonl GEMINI_USAGE_TAG=$name \
  $PY $OUT/usage_tap.py src.retrieval.evaluate_search_accuracy \
    --input experiments/reranking/eval_queries_$set.csv --split dev \
    --analysis-cache $CACHES/analysis_cache_${set}_dev_vertex.json \
    --output-dir $OUT/$name/${set}_dev > $OUT/${name}_$set.log 2>&1
  echo "done $name $set exit=$? $(date +%T)" >> $OUT/progress.txt
  # 실행 조건 기록 — runinfo에는 Context 스위치·Context 컬렉션 세대가 없다
  { echo "CONTEXT_WEIGHT=$CONTEXT_WEIGHT"; echo "QDRANT_PATH=$QDRANT_PATH"; echo "AUDIO_FUSION_SCALE=$AUDIO_FUSION_SCALE"; echo "RERANKER_ENABLED=${RERANKER_ENABLED:-}"
    echo "git_sha=$(git rev-parse --short HEAD)"; ls $QDRANT_PATH/collection | sed 's/^/collection=/'
    echo "context_build_id=$(python3 -c "import json;print(json.load(open('artifacts/vector_db/context_qdrant/dev/manifest.json'))['context_build_id'][:16])")"
  } > $OUT/$name/conditions_$set.txt
done
