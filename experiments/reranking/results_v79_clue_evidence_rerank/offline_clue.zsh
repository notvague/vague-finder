#!/bin/zsh
# 단서별 검색의 후보 변화. 리랭킹을 꺼서 모델 호출이 없다. 기준은 offline_syn_off(같은 조건, 스위치 꺼짐).
R=${0:A:h}; source $R/env.zsh; export RERANKER_ENABLED=false CLUE_SPARSE_SYNONYMS=0 CLUE_PATHS_WEIGHT=0.5; cd "$MAIN_DIR"
for s in v06 v09 v10; do
  rm -f $QDRANT_PATH/.lock
  venv/bin/python $R/one_eval.py --set $s --out $R/runs/offline_clue_on/$s > $R/logs/offline_clue_$s.log 2>&1 || { echo "FAIL $s"; exit 1; }
  venv/bin/python $R/check_run.py $R/runs/offline_clue_on/$s $R/logs/offline_clue_$s.log >> $R/logs/offline_clue.out 2>&1 || { echo "FAIL check $s"; exit 1; }
done
echo DONE
