#!/bin/zsh
# 대조용 캐시(ctrl · off)별 후보 단계 추적 — 리랭킹·Gemini 없음. dev 176건(v06 57 · v09 59 · v10 60) 전부.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v43_audio_path_weight
PY=${PY:-venv/bin/python}
TAGS=(${=TAGS:-ctrl off})
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
for tag in $TAGS; do
  for set in v06 v09 v10; do
    $PY -m src.retrieval.trace_candidate_drop --input experiments/reranking/eval_queries_$set.csv --split dev \
      --analysis-cache $OUT/analysis_cache_${set}_dev_$tag.json --deep-k 5200 \
      --output $OUT/candidate_drop_${set}_dev_$tag.csv > $OUT/trace_${set}_$tag.log 2>&1
    echo "done $tag $set $(date +%T)" >> $OUT/progress.txt
  done
done
