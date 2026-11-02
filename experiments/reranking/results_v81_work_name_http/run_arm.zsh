#!/bin/zsh
# 사용: run_arm.zsh off|on — 서버를 띄워 요청을 보내고 내린다. 메인 체크아웃의 .env에서 지정한 키만 받는다.
arm=$1; R=${0:A:h}; WT=${R:h:h:h}; cd "$WT"
PY=$MAIN_DIR/venv/bin/python; mkdir -p $R/budget $R/logs
export QDRANT_PATH="$WT/artifacts/qdrant_v81" GOOGLE_APPLICATION_CREDENTIALS="$MAIN_DIR/secrets/vertex-sa.json" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FEEDBACK_COLLECTION_ENABLED=0 FEEDBACK_INDEX_WORKER_ENABLED=0 FEEDBACK_RETRIEVAL_ENABLED=0 FEEDBACK_EVIDENCE_ENABLED=0 SEARCH_REFERENCE_YEAR=2026
unset CLAUDE_EFFORT GEMINI_RERANK_TIMEOUT_SECONDS GEMINI_RERANK_BUDGET_SECONDS
if [[ $arm == on ]]; then export WORK_NAME_RESOLUTION_ENABLED=1; else export WORK_NAME_RESOLUTION_ENABLED=0; fi
rm -f $QDRANT_PATH/.lock
$PY experiments/reranking/results_v75_work_audio_live_ab/with_env.py GCP_PROJECT_ID,GCP_LOCATION,GEMINI_RETRIEVAL_BACKEND,GEMINI_MODEL_NAME,RERANKER_BACKEND,VECTOR_BACKEND,MONGO_URI,MONGO_DB_NAME,PINECONE_NAMESPACE -- $PY $R/serve.py $arm > $R/logs/server_$arm.log 2>&1 &
pid=$!
$PY $R/drive.py $arm > $R/logs/drive_$arm.log 2>&1; rc=$?
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "ARM_DONE $arm rc=$rc" >> $R/logs/progress.txt
