#!/bin/zsh
# v55 봉인 평가 실제 실행 순서: test 분석 캐시 1회 생성 → A → B → D (각 1회, 순차 단일 프로세스). 앞 단계가 실패하면 멈춘다.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v55_attr_final_validation
if [[ ! -f $OUT/analysis_cache_v10_test_vertex.json ]]; then
  $OUT/build_cache.sh || { echo "캐시 생성 실패 — 중단" >&2; exit 1; }
fi
for cfg in A B D; do
  CONFIG=$cfg $OUT/run_eval.sh || { echo "$cfg 실행 실패(exit) — 중단" >&2; exit 1; }
done
echo "pipeline done $(date +%T)" >> $OUT/progress.txt
