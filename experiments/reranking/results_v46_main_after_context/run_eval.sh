#!/bin/zsh
# #12(나무위키 Context 경로 · 분석 프롬프트 변경) 병합 뒤의 main에서 dev 176건(v06 57 · v09 59 · v10 60)을 다시 잰다.
#   CACHES=old  : #12 전의 분석 캐시 그대로(v43과 같은 것) — 검색 코드만 바뀐 효과
#   CACHES=new  : build_caches.sh로 지금 분석기에서 새로 만든 캐시 — 지금 서비스가 하는 분석
#   SCALE       : AUDIO_FUSION_SCALE (1 = 현행, 0 = 소리 경로 끔)
#   예) CACHES=new SCALE=0 REPS="1 2" zsh run_eval.sh  → new_scale0_rep1 · new_scale0_rep2
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v46_main_after_context
PY=${PY:-venv/bin/python}
CACHES=${CACHES:-new}
SCALE=${SCALE:-1}
REPS=(${=REPS:-1})
SETS=(${=SETS:-v06 v09 v10})
export AUDIO_FUSION_SCALE=$SCALE
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
typeset -A CACHE
if [[ $CACHES == old ]]; then
  CACHE=(
    v06 experiments/reranking/analysis_cache_v06_dev.json
    v09 experiments/reranking/results_v42_modality_degrade/analysis_cache_v09_dev_degrade.json
    v10 experiments/reranking/results_v41_cover_description_image_noun/analysis_cache_v10_dev_imagenoun.json
  )
else
  CACHE=(
    v06 $OUT/analysis_cache_v06_dev_ctx.json
    v09 $OUT/analysis_cache_v09_dev_ctx.json
    v10 $OUT/analysis_cache_v10_dev_ctx.json
  )
fi
for rep in $REPS; do
  for set in $SETS; do
    name=${CACHES}_scale${SCALE}_rep$rep
    $PY -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_$set.csv --split dev \
      --analysis-cache $CACHE[$set] ${=EXTRA:-} --output-dir $OUT/$name/${set}_dev > $OUT/${name}_$set.log 2>&1
    echo "done $name $set exit=$? $(date +%T)" >> $OUT/progress.txt
  done
done
