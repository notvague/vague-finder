#!/bin/zsh
# 가정 실험 — 소리 묘사가 중심인 q115의 소리 프롬프트가 누출 검사에 걸렸을 때 네 가지 처리(make_q115_leak_variants.py).
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v42_modality_degrade
PY=${PY:-venv/bin/python}
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
for tag in stored fallback degrade audio_off; do
  $PY -m src.retrieval.trace_candidate_drop --input $OUT/eval_queries_v06_q115.csv --split dev \
    --analysis-cache $OUT/analysis_cache_q115_$tag.json --deep-k 5200 --output $OUT/candidate_drop_q115_$tag.csv > $OUT/trace_q115_$tag.log 2>&1
  for rep in 1 2; do
    $PY -m src.retrieval.evaluate_search_accuracy --input $OUT/eval_queries_v06_q115.csv --split dev \
      --analysis-cache $OUT/analysis_cache_q115_$tag.json --allow-fallback-analysis --output-dir $OUT/q115_${tag}_rep$rep > $OUT/q115_${tag}_rep$rep.log 2>&1
  done
  echo "done q115 $tag $(date +%T)" >> $OUT/progress.txt
done
