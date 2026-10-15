#!/bin/zsh
# 꼬리 드라이버 — D2 v06 재실행이 끝나면 ① B2 v10 3차 실행(2차는 t018의 2번째 호출 실패로 partial_failure) ② 재질문 영향 18건 B/D 리랭킹 포함 1회차(동시 2프로세스, 복사본 a·b)
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v54_attr_genre_policy
until grep -q "D2 v06 redo done" $OUT/progress.txt; do sleep 20; done
rm -f $OUT/usage_rerank_attr0.5_eraplus_rep2_v10.jsonl
SETS=v10 POLICY=always REP=2 COPY=b RERANK=1 zsh $OUT/run_eval.sh
echo "B2 v10 try3 done $(date +%T)" >> $OUT/progress.txt
IDS_V06="m103,q101,q104,q112,q115,q206,q308"
IDS_V09="n002,n011,n014,n016,n030,n033,n040,n042,n081,n096,n097"
( for SET in v06 v09; do IDS=$([[ $SET == v06 ]] && echo $IDS_V06 || echo $IDS_V09); POLICY=always RERANK=1 REP=1 COPY=a SET=$SET IDS=$IDS zsh $OUT/run_clarify.sh; done ) &
( for SET in v06 v09; do IDS=$([[ $SET == v06 ]] && echo $IDS_V06 || echo $IDS_V09); POLICY=conditional RERANK=1 REP=1 COPY=b SET=$SET IDS=$IDS zsh $OUT/run_clarify.sh; done ) &
wait
echo "clarify rep1 done $(date +%T)" >> $OUT/progress.txt
