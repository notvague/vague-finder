#!/bin/zsh
# 재질문 영향 18건 B/D 리랭킹 포함 2회차(동시 2프로세스, 복사본 a·b) — 1회차 실측($0.42)으로 예산 안에 들어 돌린다
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v54_attr_genre_policy
IDS_V06="m103,q101,q104,q112,q115,q206,q308"
IDS_V09="n002,n011,n014,n016,n030,n033,n040,n042,n081,n096,n097"
( for SET in v06 v09; do IDS=$([[ $SET == v06 ]] && echo $IDS_V06 || echo $IDS_V09); POLICY=always RERANK=1 REP=2 COPY=a SET=$SET IDS=$IDS zsh $OUT/run_clarify.sh; done ) &
( for SET in v06 v09; do IDS=$([[ $SET == v06 ]] && echo $IDS_V06 || echo $IDS_V09); POLICY=conditional RERANK=1 REP=2 COPY=b SET=$SET IDS=$IDS zsh $OUT/run_clarify.sh; done ) &
wait
echo "clarify rep2 done $(date +%T)" >> $OUT/progress.txt
