#!/bin/zsh
# '…이미지'로 맺은 표지 묘사 규칙 — 분석이 바뀐 v10 dev 2건(t086·t099)만 다시 잰다. 나머지는 분석이 같다(rules_before/after.csv).
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v41_cover_description_image_noun
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
venv/bin/python -m src.retrieval.trace_candidate_drop --input experiments/reranking/eval_queries_v10.csv --split dev \
  --analysis-cache $OUT/analysis_cache_v10_dev_imagenoun.json --deep-k 5200 --output $OUT/candidate_drop_v10_dev.csv > $OUT/trace.log 2>&1
for rep in 1 2; do
  venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v10.csv --split dev --query-ids t086,t099 \
    --analysis-cache $OUT/analysis_cache_v10_dev_imagenoun.json --output-dir $OUT/rep$rep > $OUT/rep$rep.log 2>&1
done
echo "done $(date +%T)" >> $OUT/progress.txt
