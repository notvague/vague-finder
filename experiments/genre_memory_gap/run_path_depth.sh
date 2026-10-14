#!/bin/zsh
# 탐색 — 경로 깊이를 넓혔을 때 장르 일치 가산의 영향 (후보 단계, 리랭킹 없음, Gemini 호출 없음).
#   기본 경로 깊이(path_k)를 60 · 100으로 넓히고 GENRE_MATCH_SCALE 1 / 0만 본다. 최종 후보 수는 30 그대로다.
#   results_v23(경로 깊이 40은 가산이 먼 곡을 끌어올려 얻는 만큼 잃는다)의 연장선 — 그 가산 가운데 장르 몫을 뺀 경우.
set -u
cd "$(git rev-parse --show-toplevel)"
source experiments/genre_memory_gap/env.sh
D=experiments/genre_memory_gap
OUT=$D/${RESULTS:-results_v01}
V46=experiments/reranking/results_v46_main_after_context
SETS=(${=SETS:-genre_gap v06 v09 v10})
DEPTHS=(${=DEPTHS:-60 100})
mkdir -p $OUT
for depth in $DEPTHS; do
  for set in $SETS; do
    if [[ $set == genre_gap ]]; then
      input=$D/queries_dev.csv; cache=$D/analysis_cache_genre_gap_dev.json
    else
      input=experiments/reranking/eval_queries_$set.csv; cache=$V46/analysis_cache_${set}_dev_ctx.json
    fi
    $PY -m src.retrieval.diagnose_genre_gap trace --input $input --analysis-cache $cache --scales 1,0 \
      --path-k $depth --output-dir $OUT/depth${depth}_$set > $OUT/depth${depth}_$set.log 2>&1
    echo "done depth$depth $set exit=$? $(date +%T)" >> $OUT/progress.txt
  done
done
