#!/bin/zsh
# 재질문 2턴 하네스(기존 evaluate_clarification · 리랭커 켬 · v06 dev) — 두 배율을 준 값으로 한 번 잰다.
#   MATCH=1 ANSWER=0 REP=1 zsh experiments/genre_memory_gap/run_clarify.sh  → clarify_m1_a0_rep1
# 결과를 읽기 전에 clarify_runinfo.json의 clarify.rerank_status_counts와 ranking_switches를 본다.
# 같은 이름의 실행이 이미 시작됐으면 건너뛴다(색인 복사본 여럿으로 나눠 돌릴 때 겹치지 않게). 다시 재려면 폴더를 지운다.
set -u
cd "$(git rev-parse --show-toplevel)"
source experiments/genre_memory_gap/env.sh
D=experiments/genre_memory_gap
OUT=$D/${RESULTS:-results_v01}
V46=experiments/reranking/results_v46_main_after_context
MATCH=${MATCH:-1}; ANSWER=${ANSWER:-1}; REP=${REP:-1}
name=clarify_m${MATCH}_a${ANSWER}_rep$REP
mkdir -p $OUT
if ! mkdir $OUT/$name 2>/dev/null; then
  echo "skip $name (이미 시작됨) $(date +%T)" >> $OUT/progress.txt
  exit 0
fi
GENRE_MATCH_SCALE=$MATCH GENRE_ANSWER_SCALE=$ANSWER $PY -m src.retrieval.evaluate_clarification --split dev \
  --analysis-cache $V46/analysis_cache_v06_dev_ctx.json --corpus "$CORPUS" \
  --output-dir $OUT/$name > $OUT/$name.log 2>&1
echo "done $name exit=$? $(date +%T)" >> $OUT/progress.txt
