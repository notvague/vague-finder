#!/bin/zsh
# 리랭커(Gemini listwise · AI Studio)를 켠 비교 — 기준선(GENRE_MATCH_SCALE=1)과 한 설정을 번갈아 잰다.
#   SCALE=0.5 REPS="1 2" zsh experiments/genre_memory_gap/run_rerank.sh
#   → rerank_scale1_rep1 · rerank_scale0.5_rep1 · rerank_scale1_rep2 · rerank_scale0.5_rep2 (세트마다 하위 폴더)
# 결과를 읽기 전에 각 runinfo의 ranking_switches.genre_match_scale(실제 적용값)과 리랭킹 미적용 건수를 본다.
set -u
cd "$(git rev-parse --show-toplevel)"
source experiments/genre_memory_gap/env.sh
D=experiments/genre_memory_gap
OUT=$D/${RESULTS:-results_v01}
V46=experiments/reranking/results_v46_main_after_context
SCALE=${SCALE:?비교할 GENRE_MATCH_SCALE을 준다(예: SCALE=0.5)}
REPS=(${=REPS:-1 2})
SETS=(${=SETS:-genre_gap v06 v09 v10})
ARMS=(${=ARMS:-1 $SCALE})
mkdir -p $OUT
for rep in $REPS; do
  for scale in $ARMS; do
    for set in $SETS; do
      if [[ $set == genre_gap ]]; then
        input=$D/queries_dev.csv; cache=$D/analysis_cache_genre_gap_dev.json
      else
        input=experiments/reranking/eval_queries_$set.csv; cache=$V46/analysis_cache_${set}_dev_ctx.json
      fi
      name=rerank_scale${scale}_rep$rep
      GENRE_MATCH_SCALE=$scale $PY -m src.retrieval.evaluate_search_accuracy --input $input --split dev \
        --analysis-cache $cache --output-dir $OUT/$name/${set}_dev > $OUT/${name}_$set.log 2>&1
      echo "done $name $set exit=$? $(date +%T)" >> $OUT/progress.txt
    done
  done
done
