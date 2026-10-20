# 8단계: 부분 코퍼스 Context 검색 평가

이 평가는 같은 `QueryAnalysis`와 같은 Text/Image/Audio/가사 색인에서 `use_context=False`와
`True`를 순서대로 실행합니다. 두 조건 모두 `use_rerank=False`, `candidate_k=30`,
`top_k=10`을 명시합니다. Gemini listwise가 `.env`에 켜져 있어도 이 측정에는 적용되지
않습니다. Context 가중치나 검색 폭을 이 단계에서 조정하지 않습니다.

## 입력과 평가 대상

- `experiments/namuwiki/eval_queries_context_v01.csv` 30건(개발 21, 시험 9).
  원본 `queries_context_v01.json`이 있으면 정답·분할·질의·segment 일치를 확인합니다.
  `context`, `mixed`, `control`별로도 집계합니다.
- `experiments/reranking/eval_queries_v05.csv` 76건(개발 53, 시험 23).
  **76건 모두** 전후 비교하며 가사 기억, 분위기·소리, 앨범아트 회귀군도 따로 보여 줍니다.
  그룹을 나눈 ID는 평가기 상단 `REGRESSION_GROUP_IDS`에 고정되어 있습니다. `other`에도
  이전 Context성 질의가 들어갈 수 있으므로 전체 v0.5와 무단서 질의 결과를 함께 읽습니다.
- 두 CSV와 JSON은 로컬/팀 드라이브에서 가져옵니다. 공개 레포나 배포 ZIP에 넣지
  않습니다. 결과와 분석 캐시도 `artifacts/context_eval_step8/` 아래에만 저장합니다.

현재 Text Qdrant에 없는 정답은 `unavailable_ids`로 기록하며 성능 집계 분모에서
제외합니다. 지표마다 `scored`와 전체 `queries`를 함께 보아야 합니다. 일부 정답만
있는 복수 정답 질의는 존재하는 정답을 기준으로 계산합니다. 이 결과는 **부분 코퍼스**의
측정이며 최종 전체 곡 성능으로 일반화하지 않습니다.

## 설치 및 입력 검사 (VS Code PowerShell, 프로젝트 루트)

ZIP을 프로젝트 루트에 풀었다면 아래 명령을 실행합니다. `--dry-run`은 모델이나
Qdrant를 열지 않습니다.

```powershell
docker compose build backend
docker compose run --rm --no-deps -T backend python -m pytest -q tests/context/test_context_step8_evaluation.py tests/context/test_context_router_step6.py
docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.evaluate_context_step8 --split dev --dry-run
```

## 눈으로 확인하는 짧은 실검색

로컬 Qdrant는 한 프로세스에서만 열 수 있습니다. 백엔드를 중지한 뒤 테스트 컨테이너를
실행하고, `finally`에서 백엔드를 다시 켭니다. MongoDB와 외부 분석 서비스가 설정된
환경에서 실행하세요. 모델 분석이 한 번 완료된 질의는 다음 실행에서 캐시를 재사용합니다.

```powershell
docker compose up -d mongodb
docker compose stop backend
if ($LASTEXITCODE -ne 0) { throw 'backend 중지 실패' }
try {
    docker compose run --rm --no-deps -T -e SEARCH_REFERENCE_YEAR=2026 backend python -u -m experiments.namuwiki.evaluate_context_step8 `
        --split dev --query-ids nw001,q203,q211,q215,q204 --show-top `
        --output-dir artifacts/context_eval_step8/smoke_dev
    if ($LASTEXITCODE -ne 0) { throw '8단계 짧은 실검색 실패' }
} finally {
    docker compose up -d backend
}
```

출력의 `C@30 없음→18`, `Top10 없음→7`은 해당 **정답**이 Context를 켰을 때
후보 30곡 안에 18위, 최종 10곡 안에 7위로 들어왔음을 뜻합니다. 실제 수치는
실행 시점의 색인과 분석 결과에 따라 달라집니다. `indexed=False`인 문항은 현재
Text 색인에 정답이 없어 점수에 넣지 않았습니다. 화면의 제목은 로컬에서만 확인하세요.

## 전체 개발 분할, 이후 시험 분할

짧은 확인이 정상이고 입력 CSV가 그대로라면 개발 분할 전체를 측정합니다. 처음에는
분석 API 호출이 많을 수 있으며, 각 문항의 분석은 캐시에 즉시 저장됩니다. 도중에
끊기면 같은 명령으로 재실행합니다. **완료 여부는** `paired_summary.json`의
`complete: true`와 `run_info.json`의 `status: complete` 둘 다 확인합니다.

```powershell
docker compose stop backend
if ($LASTEXITCODE -ne 0) { throw 'backend 중지 실패' }
try {
    docker compose run --rm --no-deps -T -e SEARCH_REFERENCE_YEAR=2026 backend python -u -m experiments.namuwiki.evaluate_context_step8 --split dev
    if ($LASTEXITCODE -ne 0) { throw '개발 분할 평가 실패' }
} finally {
    docker compose up -d backend
}
```

개발 분할의 손실 문항과 분석 누락을 먼저 검토하고 설정을 고정한 뒤 `--split test`로
같은 명령을 한 번 더 실행합니다. `test` 분할은 가중치나 top-k 조정에 쓰지 않습니다.

```powershell
docker compose stop backend
if ($LASTEXITCODE -ne 0) { throw 'backend 중지 실패' }
try {
    docker compose run --rm --no-deps -T -e SEARCH_REFERENCE_YEAR=2026 backend python -u -m experiments.namuwiki.evaluate_context_step8 --split test
    if ($LASTEXITCODE -ne 0) { throw '시험 분할 평가 실패' }
} finally {
    docker compose up -d backend
}
```

## 결과 읽기

`artifacts/context_eval_step8/results_dev/paired_summary.json`의 그룹마다
`candidate_hit30`, `candidate_recall30`, `hit1`, `hit5`, `hit10`, `mrr10`의
OFF/ON/차이가 있습니다. `gained`와 `lost`에는 각 문항 ID를 남깁니다. 평균 차이가
0이라도 손실 문항이 있을 수 있습니다. 결과 표 `paired_detail.csv`에는 정답의 두
조건 후보 순위, Top-10 순위, 두 결과의 ID, 색인에 없는 정답, Context 후보 수가 있습니다.

```powershell
$out = '.\artifacts\context_eval_step8\results_dev'
$summary = Get-Content "$out\paired_summary.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$summary.groups.context_all.metrics.hit10
$summary.groups.regression_lyrics.metrics.hit10
$summary.groups.regression_mood_sound.metrics.hit10
$summary.groups.regression_image.metrics.hit10
$summary.issues
Import-Csv "$out\paired_detail.csv" | Where-Object { $_.query_id -in @('nw001','q203','q211','q215','q204') } | Format-Table query_id,off_candidate_rank,on_candidate_rank,off_top10_rank,on_top10_rank,context_candidate_count
```

`issues.missing_context_clue`는 Context 질의 분석이 단서를 누락한 경우입니다.
`unexpected_control_clue`와 `unexpected_regression_clue`는 비Context 단서를
외부 사실로 오인했을 가능성이므로 원문과 함께 수기로 확인합니다. 무단서 질의의
두 후보 목록이 달라지거나 실행 중 Text/Context 인덱스 상태가 바뀌면 평가기가 실패합니다.
`context_hit_queries=0`인 전체 Context 결과도 성공으로 처리하지 않습니다.

이 단계의 Candidate@30 유입은 최종 Top-10 성공을 뜻하지 않습니다. 사실 근거의
진위·적합성, 전체 수집 후 최종 가중치와 top-k 조정은 별도로 확인해야 합니다.
