#!/bin/zsh
# 새 캐시로 현행(배율 1)과 소리 경로 끔(배율 0)을 번갈아 2회씩 잰다. 로컬 Qdrant는 한 프로세스만 열 수 있어 차례로 돈다.
set -u
cd "$(git rev-parse --show-toplevel)"
RUN=experiments/reranking/results_v46_main_after_context/run_eval.sh
for rep in 1 2; do
  CACHES=new SCALE=1 REPS=$rep zsh $RUN
  CACHES=new SCALE=0 REPS=$rep zsh $RUN
done
