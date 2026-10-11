#!/bin/zsh
# 토큰 경계 수정으로 분석이 바뀐 5건(v10 t009·t071·t086·t099, v09 n002)만 다시 잰다. 나머지는 분석이 같다(reapply_safeguards.py).
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v40_modality_token_fix
TAG=${TAG:-tokens}
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
Q10=${Q10:-t009,t071,t086,t099}; Q09=${Q09:-n002}
venv/bin/python -m src.retrieval.trace_candidate_drop --input experiments/reranking/eval_queries_v10.csv --split dev \
  --analysis-cache $OUT/analysis_cache_v10_dev_$TAG.json --deep-k 5200 --output $OUT/candidate_drop_v10_dev_$TAG.csv > $OUT/trace_v10_$TAG.log 2>&1
venv/bin/python -m src.retrieval.trace_candidate_drop --input experiments/reranking/eval_queries_v09.csv --split dev \
  --analysis-cache $OUT/analysis_cache_v09_dev_$TAG.json --deep-k 5200 --output $OUT/candidate_drop_v09_dev_$TAG.csv > $OUT/trace_v09_$TAG.log 2>&1
for rep in 1 2; do
  venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v10.csv --split dev --query-ids $Q10 \
    --analysis-cache $OUT/analysis_cache_v10_dev_$TAG.json --allow-fallback-analysis --output-dir $OUT/${TAG}_rep$rep/v10_dev > $OUT/${TAG}_rep${rep}_v10.log 2>&1
  venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v09.csv --split dev --query-ids $Q09 \
    --analysis-cache $OUT/analysis_cache_v09_dev_$TAG.json --allow-fallback-analysis --output-dir $OUT/${TAG}_rep$rep/v09_dev > $OUT/${TAG}_rep${rep}_v09.log 2>&1
done
echo "done $TAG $(date +%T)" >> $OUT/progress.txt
