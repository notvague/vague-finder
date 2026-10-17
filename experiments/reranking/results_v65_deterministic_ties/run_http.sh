#!/bin/zsh
# v65 HTTP 실행 묶음 — 실행마다 서버 프로세스를 새로 띄우고(run_http.py로 dev 176) 끝나면 내린다. 순차, 동시 실행 없음.
#   ./run_http.sh after:server:2 after:local:1 after:local:2 before:local:1 before:server:1
set -u
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v65_deterministic_ties
for spec in "$@"; do
  code=${spec%%:*}; rest=${spec#*:}; mode=${rest%%:*}; rep=${rest##*:}
  run=${code}_${mode}_${rep}
  port=$([[ $mode == local ]] && echo 8066 || echo 8067)
  $R/start_http.sh $code $mode $port $run
  venv/bin/python $R/run_http.py --run $run --port $port
  rc=$?
  kill $(cat $R/ctl/server_$run.pid) 2>/dev/null
  while lsof -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; do sleep 1; done
  sleep 2
  echo "done $run exit=$rc $(date +%T)" >> $R/logs/progress_http.txt
  (( rc != 0 )) && { echo "FAILED $run" >&2; exit $rc; }
done
echo "HTTP DONE"
