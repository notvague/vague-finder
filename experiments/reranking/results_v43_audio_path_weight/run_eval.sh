#!/bin/zsh
# 소리 융합 배율(AUDIO_FUSION_SCALE)별 검색·리랭킹 측정 — dev 176건 전부(v06 57 · v09 59 · v10 60). 분석 캐시는 그대로.
#   SCALE=0 REPS="1 2" zsh run_eval.sh      → scale0_rep1 · scale0_rep2
#   SCALE=1 REPS="1" zsh run_eval.sh        → scale1_rep1 (현행, 같은 코드·같은 날 대조)
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v43_audio_path_weight
PY=${PY:-venv/bin/python}
SCALE=${SCALE:-0}
REPS=(${=REPS:-1 2})
SETS=(${=SETS:-v06 v09 v10})
export AUDIO_FUSION_SCALE=$SCALE
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
typeset -A CACHE
CACHE=(
  v06 experiments/reranking/analysis_cache_v06_dev.json
  v09 experiments/reranking/results_v42_modality_degrade/analysis_cache_v09_dev_degrade.json
  v10 experiments/reranking/results_v41_cover_description_image_noun/analysis_cache_v10_dev_imagenoun.json
)
for rep in $REPS; do
  for set in $SETS; do
    dir=$OUT/scale${SCALE}_rep$rep/${set}_dev
    $PY -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_$set.csv --split dev \
      --analysis-cache $CACHE[$set] --output-dir $dir > $OUT/scale${SCALE}_rep${rep}_$set.log 2>&1
    echo "done scale$SCALE rep$rep $set $(date +%T)" >> $OUT/progress.txt
  done
done
