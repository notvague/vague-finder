#!/bin/zsh
# D2 v06 재실행 — 1차 실행이 첫 질의(모델 적재)에서 41분간 멈춰(Gemini generate 호출 0) 죽였다. B2 v10 재실행 뒤 같은 조건으로 다시 돈다
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v54_attr_genre_policy
until grep -q "B2 v10 redo done" $OUT/progress.txt; do sleep 20; done
rm -f $OUT/usage_rerank_attr0.5_eraplus_condgenre_rep2_v06.jsonl
SETS=v06 POLICY=conditional REP=2 COPY=c RERANK=1 zsh $OUT/run_eval.sh
echo "D2 v06 redo done $(date +%T)" >> $OUT/progress.txt
