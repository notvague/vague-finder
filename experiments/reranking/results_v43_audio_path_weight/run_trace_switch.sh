#!/bin/zsh
# 실험 스위치를 켠 채 후보 단계 추적 — 분석 캐시는 그대로, 리랭킹·Gemini 없음. dev 176건 전부.
# 스위치는 환경 변수로 넘긴다.
#   TAG=sparse_query TEXT_SPARSE_SOURCE=query zsh run_trace_switch.sh
#   TAG=sparse_both_scale0 TEXT_SPARSE_SOURCE=both AUDIO_FUSION_SCALE=0 zsh run_trace_switch.sh
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v43_audio_path_weight
PY=${PY:-venv/bin/python}
TAG=${TAG:?TAG가 필요하다}
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
for set in v06 v09 v10; do
  $PY -m src.retrieval.trace_candidate_drop --input experiments/reranking/eval_queries_$set.csv --split dev \
    --analysis-cache $OUT/analysis_cache_${set}_dev_ctrl.json --deep-k 5200 \
    --output $OUT/candidate_drop_${set}_dev_$TAG.csv > $OUT/trace_${set}_$TAG.log 2>&1
  echo "done trace $TAG $set (TEXT_SPARSE_SOURCE=${TEXT_SPARSE_SOURCE:-} AUDIO_FUSION_SCALE=${AUDIO_FUSION_SCALE:-}) $(date +%T)" >> $OUT/progress.txt
done
