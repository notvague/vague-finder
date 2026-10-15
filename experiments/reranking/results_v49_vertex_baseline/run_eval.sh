#!/bin/zsh
# Vertex AI 기준선(2026-10-15) — Gemini 호출 경로를 AI Studio에서 Vertex로 옮긴 뒤 dev 176건(v06 57 · v09 59 · v10 60)을 다시 잰다.
#   CACHES=aistudio : 분석은 v46의 AI Studio 캐시 그대로(고정 후보) — **리랭커만** Vertex. 저장된 Gemini(AI Studio) 결과와 비교
#   CACHES=vertex   : build_caches.sh로 Vertex 분석기에서 새로 만든 캐시 — 분석·리랭킹 모두 Vertex(전체 스택)
#   PILOT=1         : 시범 30건(세트당 10건, experiments/haiku_comparison/pilot_queries.csv와 같은 질의)만
#   REPS="1 2"  SETS="v06 v09 v10"  QDRANT=artifacts/qdrant_eval(색인 복사본 — 서비스 색인은 열지 않는다)  USAGE_TAP=<sitecustomize 폴더>(선택)
#   예) CACHES=aistudio PILOT=1 REPS=1 zsh run_eval.sh → aistudio_cache_pilot_rep1/{v06,v09,v10}_dev
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v49_vertex_baseline
PY=${PY:-venv/bin/python}
CACHES=${CACHES:-aistudio}
REPS=(${=REPS:-1})
SETS=(${=SETS:-v06 v09 v10})
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
export GEMINI_RETRIEVAL_BACKEND=vertex
export QDRANT_PATH=${QDRANT:-artifacts/qdrant_eval}
# 사용량 기록(토큰·지연·응답 모델 버전) — 저장소 밖의 sitecustomize를 PYTHONPATH로 끼운다. 없으면 기록하지 않는다.
export PYTHONPATH=${USAGE_TAP:+$USAGE_TAP:}.
typeset -A CACHE PILOT_IDS
PILOT_IDS=(
  v06 m302,q101,q112,q204,q205,q207,q209,q312,q314,q318
  v09 n010,n031,n038,n057,n061,n071,n073,n082,n086,n096
  v10 t024,t025,t029,t042,t075,t083,t086,t096,t097,t104
)
if [[ $CACHES == aistudio ]]; then
  CACHE=(
    v06 experiments/reranking/results_v46_main_after_context/analysis_cache_v06_dev_ctx.json
    v09 experiments/reranking/results_v46_main_after_context/analysis_cache_v09_dev_ctx.json
    v10 experiments/reranking/results_v46_main_after_context/analysis_cache_v10_dev_ctx.json
  )
else
  CACHE=(
    v06 $OUT/analysis_cache_v06_dev_vertex.json
    v09 $OUT/analysis_cache_v09_dev_vertex.json
    v10 $OUT/analysis_cache_v10_dev_vertex.json
  )
fi
for rep in $REPS; do
  for set in $SETS; do
    name=${CACHES}_cache${PILOT:+_pilot}_rep$rep
    ids=()
    [[ -n ${PILOT:-} ]] && ids=(--query-ids $PILOT_IDS[$set])
    GEMINI_USAGE_TAP=$OUT/usage_${name}_$set.jsonl \
    $PY -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_$set.csv --split dev \
      --analysis-cache $CACHE[$set] $ids --output-dir $OUT/$name/${set}_dev > $OUT/${name}_$set.log 2>&1
    echo "done $name $set exit=$? $(date +%T)" >> $OUT/progress.txt
  done
done
