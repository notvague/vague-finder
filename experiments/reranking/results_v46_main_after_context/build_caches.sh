#!/bin/zsh
# 지금 main의 분석기(#12에서 프롬프트 · 후처리가 바뀜)로 dev 176건의 분석 캐시를 새로 만든다. test 분할은 건드리지 않는다.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v46_main_after_context
PY=${PY:-venv/bin/python}
SETS=(${=SETS:-v06 v09 v10})
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
for set in $SETS; do
  $PY -m src.retrieval.build_analysis_cache --input experiments/reranking/eval_queries_$set.csv --split dev \
    --output $OUT/analysis_cache_${set}_dev_ctx.json > $OUT/build_cache_$set.log 2>&1
  echo "done cache $set exit=$? $(date +%T)" >> $OUT/progress.txt
done
