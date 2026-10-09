#!/bin/zsh
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking
P=$OUT/clarify_v09_progress.txt
echo "start dev $(date +%T)" > $P
venv/bin/python -m src.retrieval.evaluate_clarification --split dev \
  --analysis-cache experiments/reranking/analysis_cache_v06_dev.json \
  --output-dir $OUT/results_clarify_v09 > $OUT/results_clarify_v09.log 2>&1
echo "dev rc=$? $(date +%T)" >> $P
venv/bin/python -m src.retrieval.evaluate_clarification --split test \
  --analysis-cache experiments/reranking/analysis_cache_v06_test.json \
  --output-dir $OUT/results_clarify_test_v06 > $OUT/results_clarify_test_v06.log 2>&1
echo "test rc=$? $(date +%T)" >> $P
echo done >> $P
