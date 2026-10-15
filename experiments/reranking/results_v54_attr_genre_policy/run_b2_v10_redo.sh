#!/bin/zsh
# B2 v10 재실행 — 1차 실행이 첫 질의(모델 적재)에서 exit 139(SIGSEGV)로 죽어(Gemini generate 호출 0) D2가 끝난 뒤 같은 조건으로 다시 돈다
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v54_attr_genre_policy
until grep -q "rerank chain done" $OUT/progress.txt; do sleep 20; done
SETS=v10 POLICY=always REP=2 COPY=b RERANK=1 zsh $OUT/run_eval.sh
echo "B2 v10 redo done $(date +%T)" >> $OUT/progress.txt
