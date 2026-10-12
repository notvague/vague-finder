#!/bin/zsh
# 소리 융합 배율(AUDIO_FUSION_SCALE)별 후보 단계 추적 — 분석 캐시는 그대로, 리랭킹·Gemini 없음. dev 176건 전부.
#   SCALES="0.5 0.25" zsh run_trace_scale.sh   → candidate_drop_{v06,v09,v10}_dev_scale0.5.csv …
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v43_audio_path_weight
PY=${PY:-venv/bin/python}
SCALES=(${=SCALES:-0.5 0.25})
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
for scale in $SCALES; do
  for set in v06 v09 v10; do
    AUDIO_FUSION_SCALE=$scale $PY -m src.retrieval.trace_candidate_drop --input experiments/reranking/eval_queries_$set.csv --split dev \
      --analysis-cache $OUT/analysis_cache_${set}_dev_ctrl.json --deep-k 5200 \
      --output $OUT/candidate_drop_${set}_dev_scale$scale.csv > $OUT/trace_${set}_scale$scale.log 2>&1
    echo "done trace scale$scale $set $(date +%T)" >> $OUT/progress.txt
  done
done
