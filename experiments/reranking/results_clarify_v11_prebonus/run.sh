#!/bin/zsh
# (a) 보너스 전 순서 고정 스위치를 m402·c603으로 잰다. 변형당 2회. 기준선은 같은 경로·시간대의 base_rerun(v10은 AI Studio라 참고만).
# 위약 대조(placebo:<seed>): 보너스가 바꾼 자리 수만큼 보너스 전 순서를 고정 시드로 섞고 답은 프롬프트에 안 넣는다. 시드당 1회.
# 사용: run.sh [all|base|prebonus|placebo]  (기본 all). base는 같은 호출 경로·시간대의 기준선(v10 base는 AI Studio라 직접 비교하지 않는다)
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_clarify_v11_prebonus
P=$OUT/progress.txt
WHAT=${1:-all}
run() {
  local name=$1; local reps=$2; shift 2
  for rep in $(seq 1 $reps); do
    echo "start $name r$rep $(date +%T)" >> $P
    env "$@" venv/bin/python -m src.retrieval.evaluate_clarification --split dev --query-ids m402,c603 \
      --analysis-cache experiments/reranking/analysis_cache_v06_dev.json \
      --output-dir $OUT/${name}_r$rep > $OUT/${name}_r$rep.log 2>&1
    echo "$name r$rep rc=$? $(date +%T)" >> $P
  done
}
if [[ $WHAT == all || $WHAT == base ]]; then
  run base_rerun   1 CLARIFY_RERANK_INPUT_ORDER=bonus
fi
if [[ $WHAT == all || $WHAT == prebonus ]]; then
  run prebonus      2 CLARIFY_RERANK_INPUT_ORDER=pre_bonus
  run prebonus_both 2 CLARIFY_RERANK_INPUT_ORDER=pre_bonus GEMINI_RERANK_CORRECTIONS=new_only GEMINI_RERANK_TYPE_SLOT_LABEL="artist type"
fi
if [[ $WHAT == all || $WHAT == placebo ]]; then
  # 위약: 보너스가 바꾼 자리만, 시드에 후보 id를 섞어 행마다 독립 (리뷰 반영 뒤 재측정)
  for seed in 1 2 3 4; do
    run placebo_s$seed 1 CLARIFY_RERANK_INPUT_ORDER=placebo:$seed
  done
fi
echo "done $WHAT" >> $P
