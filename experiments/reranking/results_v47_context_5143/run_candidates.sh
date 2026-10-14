#!/bin/zsh
# 5,143곡 색인에 Context 컬렉션을 올린 뒤의 후보 단계 측정 — dev 176건(v06 57 · v09 59 · v10 60), 리랭킹 끔(Gemini 호출 없음).
#   CTX   : CONTEXT_WEIGHT (0 = Context 경로 끔, 0.5 = #12 기본값)
#   SCALE : AUDIO_FUSION_SCALE (1 = 현행, 0 = 소리 경로 끔)
#   예) CTX=0.5 SCALE=0 zsh run_candidates.sh  → ctx0.5_scale0/
# 색인은 v46과 같은 기준 색인의 복사본에 Context 컬렉션 2개를 더한 것(artifacts/qdrant_context).
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v47_context_5143
CACHES=experiments/reranking/results_v46_main_after_context
PY=${PY:-venv/bin/python}
CTX=${CTX:-0.5}
SCALE=${SCALE:-1}
SETS=(${=SETS:-v06 v09 v10})
export QDRANT_PATH=artifacts/qdrant_context
export CONTEXT_WEIGHT=$CTX AUDIO_FUSION_SCALE=$SCALE
export RERANKER_ENABLED=false RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
name=ctx${CTX}_scale${SCALE}
for set in $SETS; do
  $PY -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_$set.csv --split dev \
    --analysis-cache $CACHES/analysis_cache_${set}_dev_ctx.json --output-dir $OUT/$name/${set}_dev > $OUT/${name}_$set.log 2>&1
  echo "done $name $set exit=$? $(date +%T)" >> $OUT/progress.txt
done
