#!/bin/zsh
# 리랭커를 켠 네 실행(기준선 2 · 한 설정 2)을 변형별·세트별로 묶는다 (모델·DB 없음).
#   SCALE=0 zsh experiments/genre_memory_gap/run_report.sh
set -u
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-venv/bin/python}
D=experiments/genre_memory_gap
OUT=$D/${RESULTS:-results_v01}
SCALE=${SCALE:?비교한 GENRE_MATCH_SCALE}
runs() {   # runs <세트…> → "이름=폴더,폴더" 네 개
  local out=()
  for rep in 1 2; do
    for scale in 1 $SCALE; do
      local dirs=()
      for set in "$@"; do dirs+=($OUT/rerank_scale${scale}_rep$rep/${set}_dev); done
      out+=("scale${scale}_rep$rep=${(j:,:)dirs}")
    done
  done
  print -r -- $out
}
$PY -m src.retrieval.diagnose_genre_gap report --queries $D/queries_dev.csv \
  --runs $(runs genre_gap) --output-dir $OUT/report_genre_gap
$PY -m src.retrieval.diagnose_genre_gap report \
  --queries experiments/reranking/eval_queries_v06.csv experiments/reranking/eval_queries_v09.csv experiments/reranking/eval_queries_v10.csv \
  --runs $(runs v06 v09 v10) --output-dir $OUT/report_dev176
