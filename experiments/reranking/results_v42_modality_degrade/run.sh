#!/bin/zsh
# 모달리티만 내려놓기 — 분석이 바뀌는 질의는 dev 176건 중 폴백으로 남아 있던 n025 하나다(reapply_safeguards.py).
# 그 한 건을 세 가지 분석으로 잰다: before(규칙 폴백) · degrade(채택: 누출된 소리 프롬프트 → 규칙 프롬프트) · audio_off(대조: 소리 끔)
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v42_modality_degrade
PY=${PY:-venv/bin/python}
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
for tag in before degrade audio_off; do
  $PY -m src.retrieval.trace_candidate_drop --input $OUT/eval_queries_v09_n025.csv --split dev \
    --analysis-cache $OUT/analysis_cache_v09_dev_$tag.json --deep-k 5200 --output $OUT/candidate_drop_n025_$tag.csv > $OUT/trace_$tag.log 2>&1
  for rep in 1 2; do
    $PY -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v09.csv --split dev --query-ids n025 \
      --analysis-cache $OUT/analysis_cache_v09_dev_$tag.json --allow-fallback-analysis --output-dir $OUT/${tag}_rep$rep > $OUT/${tag}_rep$rep.log 2>&1
  done
  echo "done $tag $(date +%T)" >> $OUT/progress.txt
done
