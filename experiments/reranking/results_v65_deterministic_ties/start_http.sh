#!/bin/zsh
# v65 HTTP 검증 서버 하나를 띄운다 — 실제 FastAPI 앱(v65_app) · 코드 판(before|after) · Qdrant 모드(local|server).
#   ./start_http.sh <code> <mode> <port> <run>
# 로컬 모드는 복사본(artifacts/qdrant_v65_src)을 이 서버 프로세스만 연다 — 같은 때 다른 로컬 프로세스를 띄우지 않는다.
# 운영 .env·기본값은 건드리지 않는다(워크트리에 .env 없음). 아래 값만 이 프로세스 환경에 준다. 외부 호출은 v65_runtime이 막는다.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v65_deterministic_ties
CODE=$1 MODE=$2 PORT=$3 RUN=$4
LOG=$OUT/logs/http_$RUN
mkdir -p $LOG $OUT/ctl
for key in MONGO_URI MONGO_DB_NAME GOOGLE_APPLICATION_CREDENTIALS GEMINI_API_KEY GOOGLE_API_KEY CLAUDE_API CLAUDE_EFFORT \
  RERANKER_ENABLED QDRANT_PATH QDRANT_URL QDRANT_HYBRID_SCAN_LIMIT LYRICS_CACHE_TTL_SECONDS LYRICS_REFRESH_RETRY_SECONDS \
  GEMINI_RERANK_ENABLED GEMINI_RERANK_MODEL_NAME GEMINI_RERANK_TIMEOUT_SECONDS GEMINI_RERANK_BUDGET_SECONDS \
  GEMINI_RERANK_MAX_RETRIES GEMINI_RERANK_PASSES GEMINI_RERANK_USE_SEARCH VF_CLAUDE_EFFORT VF_CLAUDE_THINKING \
  ATTR_FILTER_GATE ATTR_FILTER_K ATTR_FILTER_USE_GENRE ATTR_FILTER_GENRE_POLICY; do
  unset $key
done
if [[ $MODE == local ]]; then export QDRANT_PATH=artifacts/qdrant_v65_src; else export QDRANT_URL=http://127.0.0.1:6366; fi
export V65_CODE=$CODE ATTR_FILTER_WEIGHT=0 CONTEXT_WEIGHT=0.5 SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RERANKER_BACKEND=gemini_listwise GEMINI_RETRIEVAL_BACKEND=vertex \
  GCP_PROJECT_ID=v65-blocked-project GCP_LOCATION=global GEMINI_MODEL_NAME=gemini-3.1-flash-lite GEMINI_RERANK_RARE_FACT_VERIFY=0 \
  FEEDBACK_COLLECTION_ENABLED=0 FEEDBACK_INDEX_WORKER_ENABLED=0 FEEDBACK_RETRIEVAL_ENABLED=0 \
  SEARCH_TIMING_LOG=$LOG/timing.jsonl V65_FAKE_MODE_FILE=$OUT/ctl/fake_mode_$RUN.json V65_FAKE_LOG=$LOG/fake_calls.jsonl \
  V65_NET_LOG=$LOG/net_attempts.jsonl
echo '{"mode":"identity"}' > $V65_FAKE_MODE_FILE
nohup venv/bin/uvicorn v65_app:app --app-dir $OUT --host 127.0.0.1 --port $PORT --log-level warning > $LOG/server.log 2>&1 &
echo $! > $OUT/ctl/server_$RUN.pid
echo "started $RUN code=$CODE mode=$MODE port=$PORT pid=$! $(date +%T)" | tee -a $OUT/logs/progress_http.txt
