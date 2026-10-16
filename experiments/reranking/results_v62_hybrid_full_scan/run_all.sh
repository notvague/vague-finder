#!/bin/zsh
# v62 — 전체 재현 순서(저장소 루트에서). 모델·Mongo·운영 Qdrant·멜론 호출 없음. 로컬 복사본은 한 프로세스만 열 수 있어 **순차**다.
#   준비(한 번): v61 워크트리의 색인 복사본을 다시 복사 — cp -R /private/tmp/vague-finder-hybrid-scan-audit/artifacts/qdrant_v61_audit artifacts/ && rm -f artifacts/qdrant_v61_audit/.lock
#   v61 저장 질의 벡터는 V61_SOURCE_ROOT(기본 v61 워크트리 폴더)에서 읽기만 한다.
# 실패 처리: **어느 단계라도 실패하면 즉시 그 종료 코드로 끝난다**(다음 단계가 남아 있는 옛 결과를 비교하지 않게). 단, 6에서 띄운 검증용 Docker
# 컨테이너는 실패해도 EXIT 트랩으로 멈춘다. 후보 출력 루트는 CAND_ROOT(기본 cand_fixed/), 같은 곳에 다시 쓰려면 OVERWRITE=1.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
OUT=experiments/reranking/results_v62_hybrid_full_scan
PY=${PY:-venv/bin/python}
export CAND_ROOT=${CAND_ROOT:-$OUT/cand_fixed}
unset CLAUDE_EFFORT CLAUDE_API QDRANT_HYBRID_SCAN_LIMIT
mkdir -p $OUT/logs
step() { echo "[$(date +%T)] $*" | tee -a $OUT/logs/run_all.log; }
SERVER_STARTED=0
cleanup() {
  if [ $SERVER_STARTED -eq 1 ]; then
    step "정리: 검증용 컨테이너 종료(저장소는 artifacts/qdrant_server_v62/storage에 보존)"
    $PY $OUT/server_setup_v62.py --stop >> $OUT/logs/server_setup.log 2>&1 || step "  경고: 컨테이너 종료 실패 — docker stop vague-finder-qdrant-v62"
  fi
}
trap cleanup EXIT
run() {  # run <이름> <명령...> — 실패하면 종료 코드를 그대로 전달하며 끝낸다
  local name=$1; shift
  step "$name"
  "$@"
  local rc=$?
  step "  exit=$rc"
  if [ $rc -ne 0 ]; then step "FAILED at '$name' (exit=$rc) — 중단"; exit $rc; fi
}

run "1 회귀 테스트" zsh -c "$PY -m pytest tests/test_v62_hybrid_full_scan.py tests/test_v61_hybrid_scan_audit.py tests/test_qdrant_backend.py -q -p no:cacheprovider > $OUT/logs/pytest.log 2>&1"
run "1b 회귀 테스트(QDRANT_HYBRID_SCAN_LIMIT=6000)" zsh -c "QDRANT_HYBRID_SCAN_LIMIT=6000 $PY -m pytest tests/test_v62_hybrid_full_scan.py tests/test_v61_hybrid_scan_audit.py -q -p no:cacheprovider > $OUT/logs/pytest_env6000.log 2>&1"
run "2 저장 호출 311건 재생(수정 구현 ↔ 독립 기준, 불일치·건수 부족·limit 검증 실패면 비정상 종료)" zsh -c "$PY $OUT/replay_fixed.py --label fixed --expected-calls 311 > $OUT/logs/replay_fixed.log 2>&1"
run "3 수정 구현으로 후보 30개 재생성(dev 176) → $CAND_ROOT" zsh $OUT/run_candidates_fixed.sh
run "4 후보 비교(v61 A·B ↔ 수정 구현)" zsh -c "$PY $OUT/compare_fixed.py --fixed-root $CAND_ROOT > $OUT/logs/compare_fixed.log 2>&1"
run "5 지연 — 로컬 파일 모드(수정 전 fc20e86 ↔ 수정 후, 교대)" zsh -c "$PY $OUT/bench_latency_v62.py --mode local --set v06 --max-calls 60 --filter-repeats 5 > $OUT/logs/bench_local.log 2>&1"
SERVER_STARTED=1
run "6 서버 모드 — 전용 Docker Qdrant 기동·텍스트 컬렉션 이전·지문 확인" zsh -c "$PY $OUT/server_setup_v62.py > $OUT/logs/server_setup.log 2>&1"
run "7 지연 — 서버 모드(exact=True, 교대)" zsh -c "$PY $OUT/bench_latency_v62.py --mode server --set v06 --max-calls 60 --filter-repeats 5 > $OUT/logs/bench_server.log 2>&1"
step "ALL DONE"
