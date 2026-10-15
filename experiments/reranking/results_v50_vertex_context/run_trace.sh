#!/bin/zsh
# v50 후보 탈락 단계 추적(리랭킹 없음 · Gemini 호출 없음) — trace_candidate_drop을 Context 끔(A)/켬(B)으로.
#   CTX=0|0.5  COPY=a|b  SETS="v06 v09 v10"
# --deep-k 5143: 기본값 3010은 3,010곡 코퍼스 시절 값이라 지금 색인(5,143곡)에서는 전수가 아니다. 5,143 이상이면 전수.
# 결과: trace_ctx$CTX/candidate_drop_$set.csv — stage(후보 안/경로 밖/융합 절단/보정 뒤 절단) · 경로별 순위 · 보조 경로(context 포함) · 가산.
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
name=trace_ctx${CTX}
mkdir -p $OUT/$name
for set in $SETS; do
  GEMINI_USAGE_TAP=$OUT/usage_${name}_$set.jsonl GEMINI_USAGE_TAG=$name \
  $PY $OUT/usage_tap.py src.retrieval.trace_candidate_drop \
    --input experiments/reranking/eval_queries_$set.csv --split dev --deep-k 5143 \
    --analysis-cache $CACHES/analysis_cache_${set}_dev_vertex.json \
    --output $OUT/$name/candidate_drop_$set.csv > $OUT/${name}_$set.log 2>&1
  echo "done $name $set exit=$? $(date +%T)" >> $OUT/progress.txt
  # 실행 조건 기록 — runinfo에는 Context 스위치·Context 컬렉션 세대가 없다
  { echo "CONTEXT_WEIGHT=$CONTEXT_WEIGHT"; echo "QDRANT_PATH=$QDRANT_PATH"; echo "AUDIO_FUSION_SCALE=$AUDIO_FUSION_SCALE"; echo "RERANKER_ENABLED=${RERANKER_ENABLED:-}"
    echo "git_sha=$(git rev-parse --short HEAD)"; ls $QDRANT_PATH/collection | sed 's/^/collection=/'
    echo "context_build_id=$(python3 -c "import json;print(json.load(open('artifacts/vector_db/context_qdrant/dev/manifest.json'))['context_build_id'][:16])")"
  } > $OUT/$name/conditions_$set.txt
done
