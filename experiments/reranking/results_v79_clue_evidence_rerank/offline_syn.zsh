#!/bin/zsh
# 동의어 확장의 후보 변화. 리랭킹을 꺼서 모델 호출이 없다.
R=${0:A:h}; source $R/env.zsh; export RERANKER_ENABLED=false; cd "$MAIN_DIR"
for arm in off on; do
  [[ $arm == on ]] && export CLUE_SPARSE_SYNONYMS=1 || export CLUE_SPARSE_SYNONYMS=0
  for s in v06 v09 v10; do
    rm -f $QDRANT_PATH/.lock
    venv/bin/python $R/one_eval.py --set $s --out $R/runs/offline_syn_$arm/$s > $R/logs/offline_syn_${arm}_$s.log 2>&1 || { echo "FAIL $arm $s"; exit 1; }
    venv/bin/python $R/check_run.py $R/runs/offline_syn_$arm/$s $R/logs/offline_syn_${arm}_$s.log >> $R/logs/offline_syn.out 2>&1 || { echo "FAIL check $arm $s"; exit 1; }
  done
done
echo DONE
