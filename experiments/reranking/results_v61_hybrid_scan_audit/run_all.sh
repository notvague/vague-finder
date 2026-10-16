#!/bin/zsh
# v61 — dev 176건(v06 57 · v09 59 · v10 60)을 A → B → C 순서로 세트마다 돈다. 로컬 Qdrant 복사본은 한 프로세스만 열 수 있어 **순차**다.
# 마지막에 A를 v06에서 한 번 더(rep2) 돌려 같은 입력의 반복 재현성을 본다. 로그는 폴더의 *.log(커밋 제외).
# 이 스크립트는 **기준선을 다시 만드는** 용도라 --overwrite를 준다. 수정 코드 검증은 run_candidates.py를 다른 --out-root로 직접 돌린다(RUN_INFO §8-4).
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v61_hybrid_scan_audit
PY=${PY:-venv/bin/python}
unset CLAUDE_EFFORT CLAUDE_API
: > $OUT/progress.txt
for set in v06 v09 v10; do
  for arm in A B C; do
    $PY $OUT/run_candidates.py --arm $arm --set $set --overwrite > $OUT/run_${arm}_${set}.log 2>&1
    echo "done $arm $set exit=$? $(date +%T)" >> $OUT/progress.txt
  done
done
$PY $OUT/run_candidates.py --arm A --set v06 --rep 2 --overwrite > $OUT/run_A_rep2_v06.log 2>&1
echo "done A_rep2 v06 exit=$? $(date +%T)" >> $OUT/progress.txt
echo "ALL DONE $(date +%T)" >> $OUT/progress.txt
