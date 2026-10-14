#!/bin/zsh
# 후보 단계 추적 — 장르 일치 가산 배율 1 / 0.5 / 0.25 / 0 (리랭킹 없음, Gemini 호출 없음).
#   새 비교 질의(87건)는 전체 깊이까지, 기존 dev 176건(v06 57 · v09 59 · v10 60)은 지금 깊이만 본다.
#   genre_gap_blank는 진단용 — 새 비교 질의를 분석의 genre 칸만 비운 사본으로 한 번 더 돈다(--blank-genre).
#   기존 dev의 분석 캐시는 results_v46(#12 병합 뒤 새로 분석한 것)이다 — RUN_INFO.md §0 참고.
# 로컬 Qdrant는 한 프로세스만 열 수 있어 차례로 돈다.
set -u
cd "$(git rev-parse --show-toplevel)"
source experiments/genre_memory_gap/env.sh
D=experiments/genre_memory_gap
OUT=$D/${RESULTS:-results_v01}
V46=experiments/reranking/results_v46_main_after_context
SETS=(${=SETS:-genre_gap genre_gap_blank v06 v09 v10})
mkdir -p $OUT
for set in $SETS; do
  if [[ $set == genre_gap ]]; then
    $PY -m src.retrieval.diagnose_genre_gap trace \
      --analysis-cache $D/analysis_cache_genre_gap_dev.json --model-raw $D/model_raw_genre_gap_dev.jsonl \
      --deep-k 5143 --output-dir $OUT/trace_genre_gap > $OUT/trace_genre_gap.log 2>&1
  elif [[ $set == genre_gap_blank ]]; then
    $PY -m src.retrieval.diagnose_genre_gap trace --blank-genre --scales 1 \
      --analysis-cache $D/analysis_cache_genre_gap_dev.json --model-raw $D/model_raw_genre_gap_dev.jsonl \
      --deep-k 5143 --output-dir $OUT/trace_genre_gap_blank > $OUT/trace_genre_gap_blank.log 2>&1
  else
    $PY -m src.retrieval.diagnose_genre_gap trace --input experiments/reranking/eval_queries_$set.csv \
      --analysis-cache $V46/analysis_cache_${set}_dev_ctx.json \
      --output-dir $OUT/trace_$set > $OUT/trace_$set.log 2>&1
  fi
  echo "done trace $set exit=$? $(date +%T)" >> $OUT/progress.txt
done
