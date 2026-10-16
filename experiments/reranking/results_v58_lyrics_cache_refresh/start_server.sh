#!/bin/zsh
# v58 검증 서버를 띄운다 — v56 전용 Qdrant 서버(127.0.0.1:6356, 별칭 포함)에 붙는다. 운영 .env·기본값은 건드리지 않는다.
# 운영 .env와 다른 점(검증 환경에만 적용): QDRANT_URL(서버 모드) · FEEDBACK_COLLECTION_ENABLED=0 · SEARCH_TIMING_LOG ·
# LYRICS_CACHE_TTL_SECONDS/LYRICS_REFRESH_RETRY_SECONDS(짧게 — 350초를 기다리지 않기 위한 검증값) · V58_*(가짜 자료원 제어) ·
# GCP_PROJECT_ID 는 더미(실제 호출은 코드에서 차단, 인증 파일 unset). Mongo는 코드에서 차단(MONGO_URI 무관).
# 사용: start_server.sh <name> [old]   예) start_server.sh new  /  start_server.sh old old
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=experiments/reranking/results_v58_lyrics_cache_refresh
NAME=${1:-new}
PORT=${PORT:-8058}
mkdir -p $OUT/logs $OUT/ctl
export QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6356}
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
export RERANKER_BACKEND=gemini_listwise GEMINI_RETRIEVAL_BACKEND=vertex GCP_PROJECT_ID=v58-blocked-project GCP_LOCATION=global
export FEEDBACK_COLLECTION_ENABLED=0 FEEDBACK_RETRIEVAL_ENABLED=0
export LYRICS_CACHE_TTL_SECONDS=${LYRICS_CACHE_TTL_SECONDS:-20} LYRICS_REFRESH_RETRY_SECONDS=${LYRICS_REFRESH_RETRY_SECONDS:-15}
export V58_LYRICS_CTL=$OUT/ctl/lyrics_ctl_$NAME.json V58_SOURCE_LOG=$OUT/source_calls_$NAME.jsonl
export V56_FAKE_MODE_FILE=$OUT/ctl/fake_mode_$NAME.json V56_FAKE_LOG=$OUT/fake_calls_$NAME.jsonl
export SEARCH_TIMING_LOG=$OUT/timing_$NAME.jsonl
unset GOOGLE_APPLICATION_CREDENTIALS GEMINI_API_KEY CLAUDE_API CLAUDE_EFFORT RERANKER_ENABLED CONTEXT_WEIGHT MONGO_URI
if [[ "${2:-}" == "old" ]]; then export V58_OLD_SERVICE=1 V58_OLD_REF=${V58_OLD_REF:-origin/main}; else export V58_OLD_SERVICE=0; fi
echo '{"mode":"normal","version":"v1"}' > $V58_LYRICS_CTL
echo '{"mode":"normal"}' > $V56_FAKE_MODE_FILE
nohup venv/bin/uvicorn v58_app:app --app-dir $OUT --host 127.0.0.1 --port $PORT --log-level info \
  > $OUT/logs/server_$NAME.log 2>&1 &
echo $! > $OUT/ctl/server_$NAME.pid
echo "started $NAME port=$PORT pid=$! old=${V58_OLD_SERVICE} ttl=$LYRICS_CACHE_TTL_SECONDS retry=$LYRICS_REFRESH_RETRY_SECONDS $(date +%T)" | tee -a $OUT/progress.txt
