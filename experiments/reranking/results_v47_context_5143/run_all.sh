#!/bin/zsh
# Context 켬/끔 × 소리 경로 켬/끔 네 조합을 차례로 잰다(로컬 Qdrant는 한 프로세스만 연다).
set -u
cd "$(git rev-parse --show-toplevel)"
RUN=experiments/reranking/results_v47_context_5143/run_candidates.sh
for ctx in 0 0.5; do
  for scale in 1 0; do
    CTX=$ctx SCALE=$scale zsh $RUN
  done
done
