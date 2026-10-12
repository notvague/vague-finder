#!/bin/zsh
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v35_rare_verify_off
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
echo "start v06 $(date +%T)" > $OUT/progress.txt
venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v06.csv --split dev \
  --analysis-cache experiments/reranking/analysis_cache_v06_dev.json --output-dir $OUT/v06_dev > $OUT/v06_dev.log 2>&1
echo "v06 rc=$? $(date +%T)" >> $OUT/progress.txt
venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v09.csv --split dev \
  --analysis-cache experiments/reranking/analysis_cache_v09_dev.json --allow-fallback-analysis --output-dir $OUT/v09_dev > $OUT/v09_dev.log 2>&1
echo "v09 rc=$? $(date +%T)" >> $OUT/progress.txt
echo done >> $OUT/progress.txt
