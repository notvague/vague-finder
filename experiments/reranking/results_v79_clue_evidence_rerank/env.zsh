# 구성 공통 조건 = v75 A(작품명 ON · 소리 ON · Context 0.5 · 속성 보조 0 · 피드백 끔 · 평가 전용 60/120초). 구성 차이는 스위치 둘뿐이다.
export V79_WORKTREE="${${(%):-%x}:A:h:h:h:h}"   # 이 파일 위치에서 워크트리 루트를 찾는다
ARCHIVE="$HOME/vague-finder-archive/vague-finder-v75-work-audio-live-ab"   # 해석 캐시·언급 저장소(Git 제외 원기록)
export WORK_NAME_RESOLUTION_ENABLED=1 WORK_NAME_RESOLVER="cache:$ARCHIVE/experiments/reranking/results_v75_work_audio_live_ab/resolution_cache/v71_rep1.json" WORK_MENTIONS_PATH="$ARCHIVE/artifacts/work_names/work_mentions_v1.json.gz"
export AUDIO_FUSION_SCALE=1 ATTR_FILTER_WEIGHT=0 CONTEXT_WEIGHT=0.5 RERANKER_BACKEND=gemini_listwise SEARCH_REFERENCE_YEAR=2026
export FEEDBACK_EVIDENCE_ENABLED=0 FEEDBACK_COLLECTION_ENABLED=0 FEEDBACK_INDEX_WORKER_ENABLED=0 FEEDBACK_RETRIEVAL_ENABLED=0 SEARCH_WARMUP=0 TOKENIZERS_PARALLELISM=false
export GEMINI_RERANK_TIMEOUT_SECONDS=60 GEMINI_RERANK_BUDGET_SECONDS=120
export GOOGLE_APPLICATION_CREDENTIALS="$MAIN_DIR/secrets/vertex-sa.json" QDRANT_PATH="$V79_WORKTREE/artifacts/qdrant_v79"   # MAIN_DIR = 메인 체크아웃
unset CLAUDE_EFFORT QDRANT_HYBRID_SCAN_LIMIT
# 워크트리에는 .env가 없다. 비밀값이 아닌 설정은 여기에 고정하고, GCP_PROJECT_ID는 with_env.py로 값 기록 없이 넘긴다.
export VECTOR_BACKEND=qdrant GEMINI_RETRIEVAL_BACKEND=vertex GCP_LOCATION=global GEMINI_MODEL_NAME=gemini-3.1-flash-lite
export V79_LYRICS_SNAPSHOT="$HOME/vague-finder-archive/vague-finder-hybrid-lyrics-validation/artifacts/lyrics_snapshot_v63/snapshot"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GEMINI_RERANK_RARE_FACT_VERIFY=0
