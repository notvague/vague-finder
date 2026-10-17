#!/bin/zsh
# v64 검증 서버 2개(A: 8064 · D: 8065)를 띄운다 — 같은 전용 Qdrant 서버(127.0.0.1:6364)에 붙는다. 운영 .env·기본값은 건드리지 않는다.
# 워크트리에는 .env가 없다. 아래 값만 검증 프로세스 환경에 준다. 외부 클라이언트·네트워크는 v64_runtime이 막는다.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v64_service_readiness
RUN=${V64_RUN:-main}
LOG=$OUT/logs/$RUN
mkdir -p $LOG $OUT/ctl
for key in MONGO_URI MONGO_DB_NAME GOOGLE_APPLICATION_CREDENTIALS GEMINI_API_KEY GOOGLE_API_KEY CLAUDE_API CLAUDE_EFFORT \
  RERANKER_ENABLED QDRANT_PATH QDRANT_HYBRID_SCAN_LIMIT LYRICS_CACHE_TTL_SECONDS LYRICS_REFRESH_RETRY_SECONDS \
  GEMINI_RERANK_ENABLED GEMINI_RERANK_MODEL_NAME GEMINI_RERANK_TIMEOUT_SECONDS GEMINI_RERANK_BUDGET_SECONDS \
  GEMINI_RERANK_MAX_RETRIES GEMINI_RERANK_PASSES GEMINI_RERANK_USE_SEARCH VF_CLAUDE_EFFORT VF_CLAUDE_THINKING; do
  unset $key
done
export QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6364}
export CONTEXT_WEIGHT=0.5 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RERANKER_BACKEND=gemini_listwise GEMINI_RETRIEVAL_BACKEND=vertex GCP_PROJECT_ID=v64-blocked-project GCP_LOCATION=global
export GEMINI_MODEL_NAME=gemini-3.1-flash-lite GEMINI_RERANK_RARE_FACT_VERIFY=0
export FEEDBACK_COLLECTION_ENABLED=0 FEEDBACK_INDEX_WORKER_ENABLED=0 FEEDBACK_RETRIEVAL_ENABLED=0

start_one() {
  local name=$1 port=$2 weight=$3 policy=$4
  (
    export ATTR_FILTER_WEIGHT=$weight ATTR_FILTER_GATE=era_plus ATTR_FILTER_K=30 ATTR_FILTER_USE_GENRE=1 ATTR_FILTER_GENRE_POLICY=$policy
    export SEARCH_TIMING_LOG=$LOG/timing_$name.jsonl V64_FAKE_MODE_FILE=$OUT/ctl/fake_mode_$name.json
    export V64_FAKE_LOG=$LOG/fake_calls_$name.jsonl V64_NET_LOG=$LOG/net_attempts_$name.jsonl
    echo '{"mode":"normal"}' > $V64_FAKE_MODE_FILE
    nohup venv/bin/uvicorn v64_app:app --app-dir $OUT --host 127.0.0.1 --port $port --log-level info \
      > $LOG/server_$name.log 2>&1 &
    echo $! > $OUT/ctl/server_$name.pid
    echo "started run=$RUN $name port=$port pid=$! weight=$weight policy=$policy $(date +%T)" | tee -a $OUT/progress.txt
  )
}

case "${1:-both}" in
  A) start_one A 8064 0 always ;;
  D) start_one D 8065 0.5 conditional ;;
  both) start_one A 8064 0 always; start_one D 8065 0.5 conditional ;;
esac
