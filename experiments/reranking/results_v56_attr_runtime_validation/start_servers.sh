#!/bin/zsh
# v56 검증 서버 2개(A: 8056 · D: 8057)를 띄운다 — 같은 전용 Qdrant 서버(127.0.0.1:6356)에 붙는다. 운영 .env·기본값은 건드리지 않는다.
# 운영 .env 와 다른 점(검증 환경에만 적용): QDRANT_URL(서버 모드) · FEEDBACK_COLLECTION_ENABLED=0(검증 검색을 Mongo 피드백 기록에 남기지 않음) ·
# SEARCH_TIMING_LOG(구간 기록) · V56_*(가짜 리랭커 제어) · GCP_PROJECT_ID 는 더미(실제 호출은 코드에서 차단, 인증 파일 unset).
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v56_attr_runtime_validation
mkdir -p $OUT/logs $OUT/ctl
export QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6356}
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
export RERANKER_BACKEND=gemini_listwise GEMINI_RETRIEVAL_BACKEND=vertex GCP_PROJECT_ID=v56-blocked-project GCP_LOCATION=global
export FEEDBACK_COLLECTION_ENABLED=0
unset GOOGLE_APPLICATION_CREDENTIALS GEMINI_API_KEY CLAUDE_API CLAUDE_EFFORT RERANKER_ENABLED CONTEXT_WEIGHT

start_one() {
  local name=$1 port=$2 weight=$3 policy=$4
  (
    export ATTR_FILTER_WEIGHT=$weight ATTR_FILTER_GATE=era_plus ATTR_FILTER_K=30 ATTR_FILTER_USE_GENRE=1 ATTR_FILTER_GENRE_POLICY=$policy
    export SEARCH_TIMING_LOG=$OUT/timing_$name.jsonl V56_FAKE_MODE_FILE=$OUT/ctl/fake_mode_$name.json V56_FAKE_LOG=$OUT/fake_calls_$name.jsonl
    echo '{"mode":"normal"}' > $V56_FAKE_MODE_FILE
    nohup venv/bin/uvicorn v56_app:app --app-dir $OUT --host 127.0.0.1 --port $port --log-level info \
      > $OUT/logs/server_$name.log 2>&1 &
    echo $! > $OUT/ctl/server_$name.pid
    echo "started $name port=$port pid=$! weight=$weight policy=$policy $(date +%T)" | tee -a $OUT/progress.txt
  )
}

case "${1:-both}" in
  A) start_one A 8056 0 always ;;
  D) start_one D 8057 0.5 conditional ;;
  both) start_one A 8056 0 always; start_one D 8057 0.5 conditional ;;
esac
