# 이 실험의 모든 실행이 같은 색인·호출 경로·스위치를 쓰게 하는 공통 환경. `source`로 읽는다.
#   색인: artifacts/qdrant_5143_baseline_backup(10/11 적재, #45의 5곡 수정 전)을 복사한 것.
#         v39·v41·v43·v46 기준선이 쓴 색인과 같다. 서비스 색인(artifacts/qdrant)은 열지 않는다.
#   별도 워크트리에서 돌릴 때는 원본 체크아웃의 .env · venv · data/all_songs.jsonl을 연결(symlink)해 둔다
#   (셋 다 git 제외 대상이다). 여기서 export한 값이 .env보다 우선한다.
export QDRANT_PATH=${QDRANT_PATH:-artifacts/qdrant_snapshot_5143_pre45}
export RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
export GEMINI_RETRIEVAL_BACKEND=api_key
export SEARCH_REFERENCE_YEAR=2026 TOKENIZERS_PARALLELISM=false
# 피드백 수집·보충·색인 작업은 이 비교에서 끈다(최초 장르 비교의 고정 조건).
export FEEDBACK_COLLECTION_ENABLED=0 FEEDBACK_RETRIEVAL_ENABLED=0 FEEDBACK_INDEX_WORKER_ENABLED=0
PY=${PY:-venv/bin/python}
# 재질문 하네스가 정답 곡의 등록 값을 읽는 코퍼스 파일(읽기만 한다).
CORPUS=${CORPUS:-data/all_songs.jsonl}
