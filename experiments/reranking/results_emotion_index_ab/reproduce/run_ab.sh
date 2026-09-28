#!/bin/zsh
# A/B 평가 실행기. 사용: reproduce/run_ab.sh <작업폴더> <레포>   (상대경로도 된다)
#
# <작업폴더>에는 다음이 있어야 한다 (RUN_INFO.md의 "재현" 순서로 만든다).
#   wA/, wB/            feat/qdrant-backend@6053bdd worktree (wB에는 arm_B.patch 적용)
#   analyses.json       두 조건에 주입할 질의 분석
#   params_A.json, params_B.json   fit_params.py 결과
#   qdrant_A/, qdrant_B/           qdrant_load 결과
# 결과: <작업폴더>/result_A, result_B, log_A.txt, log_B.txt
set -u
SCR="${1:A}"; REPO="${2:A}"; HERE="${0:A:h}"   # cd 뒤에도 쓰므로 절대경로로 바꾼다

# 가사 정확일치 경로가 MongoDB(MONGO_URI·MONGO_DB_NAME)를 읽는다. .env가 없으면 그 경로가 비어
# 수치가 달라지므로 멈춘다.
if [ ! -f "$REPO/.env" ]; then
  echo "[중단] $REPO/.env 없음 — MONGO_URI 등을 읽을 수 없다" >&2
  exit 1
fi
set -a; . "$REPO/.env"; set +a

# 측정 당시 리랭커 설정으로 고정한다. .env에 없거나 값이 달라도 같은 조건이 되게 한다.
# SPREAD_REF는 코드 기본값이 0인데 측정은 0.02로 했다(상세 결과의 rerank_spread/rerank_confidence).
export RERANKER_ENABLED=true RERANKER_WEIGHT=0.90 RERANKER_SPREAD_REF=0.02 \
       RERANKER_LOW_CONF_TOP_N=5 RERANKER_MAX_LENGTH=512 \
       RERANKER_MODEL_NAME=dragonkue/bge-reranker-v2-m3-ko

for arm in A B; do
  echo "===== $arm 시작 $(date +%H:%M:%S) ====="
  cd "$SCR/w$arm" || exit 1
  VECTOR_BACKEND=qdrant QDRANT_PATH="$SCR/qdrant_$arm" BM25_PARAMS_PATH="$SCR/params_$arm.json" \
  PYTHONPATH=. "$REPO/venv/bin/python" "$HERE/run_eval.py" "$SCR/analyses.json" \
    --input experiments/reranking/eval_queries_v05.csv --output-dir "$SCR/result_$arm" \
    --top-k 10 --candidate-k 30 > "$SCR/log_$arm.txt" 2>&1
  rc=$?   # 바로 저장한다. echo 안의 $(date)가 먼저 돌면 $?가 date의 종료코드로 바뀐다.
  echo "===== $arm 종료 $(date +%H:%M:%S) exit=$rc ====="
  [ "$rc" -eq 0 ] || exit "$rc"
done
