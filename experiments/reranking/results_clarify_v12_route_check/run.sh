#!/bin/zsh
# 호출 경로 확인: 검색(질의 분석·리랭커)만 AI Studio로 고정하고(#31) v10 base(m402·c603)와 같은 조건으로 2회 잰다.
# v10 base_r1·r2(10/9 13:34, AI Studio)와 행 단위로 같으면, v11에서 본 차이는 시간이 아니라 Vertex 경로 탓이다.
# 실제 측정(10/10 00:02~00:12)은 임시 폴더에 쓴 뒤 이 폴더로 옮겼다 — runinfo output_dir은 옮긴 위치로 고쳤다.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_clarify_v12_route_check
for rep in 1 2; do
  echo "start r$rep $(date +%T)"
  env GEMINI_RETRIEVAL_BACKEND=api_key GEMINI_RERANK_CORRECTIONS=all \
    venv/bin/python -m src.retrieval.evaluate_clarification --split dev --query-ids m402,c603 \
    --analysis-cache experiments/reranking/analysis_cache_v06_dev.json \
    --output-dir $OUT/base_r$rep > $OUT/base_r$rep.log 2>&1
  echo "r$rep rc=$? $(date +%T)"
done
