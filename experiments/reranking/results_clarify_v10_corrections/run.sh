#!/bin/zsh
# m402·c603 두 질의로 리랭커 답변 변형을 잰다. 변형당 2회(실행 변동 확인).
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_clarify_v10_corrections
P=$OUT/progress.txt
: > $P
run() {  # name, env...
  local name=$1; shift
  for rep in 1 2; do
    echo "start $name r$rep $(date +%T)" >> $P
    env "$@" venv/bin/python -m src.retrieval.evaluate_clarification --split dev --query-ids m402,c603 \
      --analysis-cache experiments/reranking/analysis_cache_v06_dev.json \
      --output-dir $OUT/${name}_r$rep > $OUT/${name}_r$rep.log 2>&1
    echo "$name r$rep rc=$? $(date +%T)" >> $P
  done
}
run base            GEMINI_RERANK_CORRECTIONS=all
run new_only        GEMINI_RERANK_CORRECTIONS=new_only
run type_label      GEMINI_RERANK_CORRECTIONS=all      GEMINI_RERANK_TYPE_SLOT_LABEL="artist type"
run both            GEMINI_RERANK_CORRECTIONS=new_only GEMINI_RERANK_TYPE_SLOT_LABEL="artist type"
echo done >> $P
