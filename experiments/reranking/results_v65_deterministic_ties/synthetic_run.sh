#!/bin/zsh
# v65 합성 백엔드 순서: 적재(정순·역순, 로컬 파일 정순·역순·다시 열기) → 두 코드 판 조회 → 전용 서버 재시작 → 조회 → 셔플 재적재 → 조회.
set -eu
cd "$(git rev-parse --show-toplevel)"
R=experiments/reranking/results_v65_deterministic_ties
PY="env -u QDRANT_URL -u QDRANT_PATH venv/bin/python"
V65_CODE=after  ${=PY} $R/synthetic_backends.py load
V65_CODE=before ${=PY} $R/synthetic_backends.py load
docker restart vague-finder-qdrant-v65 >/dev/null
until curl -sf http://127.0.0.1:6366/readyz >/dev/null; do sleep 1; done
V65_CODE=after  ${=PY} $R/synthetic_backends.py query restart
V65_CODE=before ${=PY} $R/synthetic_backends.py query restart
V65_CODE=after  ${=PY} $R/synthetic_backends.py reload reload
V65_CODE=before ${=PY} $R/synthetic_backends.py query reload
echo SYNTHETIC DONE
