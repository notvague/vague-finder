#!/bin/zsh
# Vertex 분석기로 dev 176건의 분석 캐시를 **새로** 만든다(v46의 AI Studio 캐시는 건드리지 않는다). test 분할은 만들지 않는다.
#   SETS="v06 v09 v10"  USAGE_TAP=<sitecustomize 폴더>(선택)
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v49_vertex_baseline
PY=${PY:-venv/bin/python}
SETS=(${=SETS:-v06 v09 v10})
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false GEMINI_RETRIEVAL_BACKEND=vertex
export PYTHONPATH=${USAGE_TAP:+$USAGE_TAP:}.
for set in $SETS; do
  GEMINI_USAGE_TAP=$OUT/usage_build_cache_$set.jsonl \
  $PY -m src.retrieval.build_analysis_cache --input experiments/reranking/eval_queries_$set.csv --split dev \
    --output $OUT/analysis_cache_${set}_dev_vertex.json > $OUT/build_cache_$set.log 2>&1
  echo "done cache $set exit=$? $(date +%T)" >> $OUT/progress.txt
done
