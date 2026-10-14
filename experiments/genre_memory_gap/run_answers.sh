#!/bin/zsh
# 재질문 장르 답변 가산 — 후보 단계 비교(리랭킹 없음, Gemini 호출 없음). GENRE_ANSWER_SCALE 1 / 0.5 / 0.
#   정답이 후보 순서의 10위 밖인 질의에서, 거절 뒤 장르를 답하면 정답의 후보 순위가 어떻게 되는지 본다.
#   답은 맞는 답(정답의 첫 등록 장르) · 체감 장르 답(새 비교 질의에만 있다) · 답 없음(거절만).
set -u
cd "$(git rev-parse --show-toplevel)"
source experiments/genre_memory_gap/env.sh
D=experiments/genre_memory_gap
OUT=$D/${RESULTS:-results_v01}
V46=experiments/reranking/results_v46_main_after_context
SETS=(${=SETS:-genre_gap v06 v09 v10})
mkdir -p $OUT
for set in $SETS; do
  if [[ $set == genre_gap ]]; then
    input=$D/queries_dev.csv; cache=$D/analysis_cache_genre_gap_dev.json
  else
    input=experiments/reranking/eval_queries_$set.csv; cache=$V46/analysis_cache_${set}_dev_ctx.json
  fi
  $PY -m src.retrieval.diagnose_genre_gap answers --input $input --analysis-cache $cache \
    --output-dir $OUT/answers_$set > $OUT/answers_$set.log 2>&1
  echo "done answers $set exit=$? $(date +%T)" >> $OUT/progress.txt
done
