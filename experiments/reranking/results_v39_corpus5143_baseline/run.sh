#!/bin/zsh
# 5,143곡 기준선 — 기존 dev 116건(v06 57 + v09 59)을 현재 기본 설정으로. 후보 단계 추적 1회 + 측정 2회.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v39_corpus5143_baseline
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
for v in v06 v09; do
  echo "trace $v $(date +%T)" >> $OUT/progress.txt
  venv/bin/python -m src.retrieval.trace_candidate_drop --input experiments/reranking/eval_queries_$v.csv --split dev \
    --analysis-cache experiments/reranking/analysis_cache_${v}_dev.json --deep-k 5200 \
    --output $OUT/candidate_drop_${v}_dev.csv > $OUT/trace_$v.log 2>&1
  echo "  rc=$? $(date +%T)" >> $OUT/progress.txt
done
for rep in ${=REPS:-1 2}; do
  for v in v06 v09; do
    echo "eval rep$rep $v $(date +%T)" >> $OUT/progress.txt
    venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_$v.csv --split dev \
      --analysis-cache experiments/reranking/analysis_cache_${v}_dev.json --allow-fallback-analysis \
      --output-dir $OUT/rep$rep/${v}_dev > $OUT/rep${rep}_$v.log 2>&1
    echo "  rc=$? $(date +%T)" >> $OUT/progress.txt
  done
done
echo done >> $OUT/progress.txt
