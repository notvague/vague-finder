#!/bin/zsh
# v54 모델 평가 순서(한 번에 띄우는 드라이버) — ① 모델 없는 2턴 흐름 B/D(영향 18건, 동시 2프로세스) → ② 리랭킹 B1 → D1 → B2 → D2 순차(단일 프로세스)
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v54_attr_genre_policy
IDS_V06="m103,q101,q104,q112,q115,q206,q308"
IDS_V09="n002,n011,n014,n016,n030,n033,n040,n042,n081,n096,n097"
echo "pipeline start $(date +%T)" >> $OUT/progress.txt
( for SET in v06 v09; do IDS=$([[ $SET == v06 ]] && echo $IDS_V06 || echo $IDS_V09); POLICY=always RERANK=0 COPY=a SET=$SET IDS=$IDS zsh $OUT/run_clarify.sh; done ) &
( for SET in v06 v09; do IDS=$([[ $SET == v06 ]] && echo $IDS_V06 || echo $IDS_V09); POLICY=conditional RERANK=0 COPY=b SET=$SET IDS=$IDS zsh $OUT/run_clarify.sh; done ) &
wait
echo "norerank clarify done $(date +%T)" >> $OUT/progress.txt
POLICY=always REP=1 COPY=b RERANK=1 zsh $OUT/run_eval.sh
POLICY=conditional REP=1 COPY=c RERANK=1 zsh $OUT/run_eval.sh
POLICY=always REP=2 COPY=b RERANK=1 zsh $OUT/run_eval.sh
POLICY=conditional REP=2 COPY=c RERANK=1 zsh $OUT/run_eval.sh
echo "rerank chain done $(date +%T)" >> $OUT/progress.txt
