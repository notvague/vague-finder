# 8단계 평가 중 nw008 분석 폴백 복구

## 원인과 영향

`nw008`의 "불후의 명곡에서 **커버해 우승**"은 공연·버전 관련 Context 단서입니다.
시각 단서 판정이 `커버해` 안의 `커버`를 앨범아트로 읽으면서 빈
`image_english_query`를 거부했고, 세 차례 재시도 후 규칙 폴백을 반환했습니다.
평가기의 폴백 중단은 의도한 보호 장치입니다. `nw008` 이후의 순위는 측정하지
않았으므로 이전 부분 결과를 전체 dev 성능으로 인용하면 안 됩니다.

같은 30문항 사전 검사에서 `nw010`의 "뮤직비디오에서 다음 앨범을 암시"도
앨범아트로 잘못 인식될 수 있었습니다. 수정한 판정은 **노래를 커버한 행위**를
표지에서 제외하고, "다음 앨범"과 별개의 뮤직비디오 시각 표현을 합쳐 표지로
판단하지 않습니다. "앨범이 파란색", "앨범 커버 해상도" 같은 실제 표지 설명은
계속 이미지 단서입니다.

## 적용과 검증 (PowerShell, 프로젝트 루트)

수정 ZIP을 프로젝트 루트에 풀고 아래 명령으로 모델 호출 없이 회귀를 확인합니다.
ZIP에는 실제 질의 CSV, 음원, 이미지, meta.json, Qdrant 산출물이 없습니다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Step8_Visual_Guard_Fix.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 프로젝트 폴더 또는 Downloads에 넣어 주세요.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
git diff --check
git diff -- src/retrieval/modality_queries.py
docker compose build backend
docker compose run --rm --no-deps -T backend python -m pytest -q tests/context/test_context_step8_modality_boundary.py tests/context/test_context_query_analyzer.py tests/context/test_context_step8_evaluation.py
if ($LASTEXITCODE -ne 0) { throw '회귀 테스트 실패' }
```

기존 캐시는 실패한 `nw008` 규칙 폴백과 **수정 전 분석기 지문**을 담고 있습니다.
평가기가 지문 불일치를 거부하므로, 삭제 대신 이름을 바꿔 보관합니다. 이 작업 후
분석 비용이 다시 들지만 같은 분석기 조건으로 74문항을 비교할 수 있습니다.

```powershell
$cache = '.\artifacts\context_eval_step8\analysis_cache.json'
if (Test-Path -LiteralPath $cache) {
    $backup = '.\artifacts\context_eval_step8\analysis_cache_before_visual_fix_{0}.json' -f (Get-Date -Format 'yyyyMMdd_HHmmss')
    Move-Item -LiteralPath $cache -Destination $backup
    Write-Host "기존 분석 캐시 보관: $backup"
}
```

먼저 오류 문항과 뒤따를 문항만 실제 색인에서 확인합니다. 둘 다 `dev`입니다.
Docker 로컬 Qdrant를 여는 동안 백엔드가 동시에 접근하지 않게 합니다.

```powershell
docker compose stop backend
if ($LASTEXITCODE -ne 0) { throw 'backend 중지 실패' }
try {
    docker compose run --rm --no-deps -T -e SEARCH_REFERENCE_YEAR=2026 backend python -u -m experiments.namuwiki.evaluate_context_step8 `
        --split dev --query-ids nw008,nw010 --show-top `
        --output-dir artifacts/context_eval_step8/smoke_visual_fix
    if ($LASTEXITCODE -ne 0) { throw 'nw008/nw010 실검색 실패' }
} finally {
    docker compose up -d backend
}
```

실검색이 완료되면 `--query-ids`와 `--output-dir` 없이 `--split dev` 전체를
재실행합니다. 처음 다섯 문항도 새 캐시와 동일한 분석 코드로 다시 측정됩니다.
완료 조건은 `artifacts/context_eval_step8/results_dev/run_info.json`의
`status=complete`와 `paired_summary.json`의 `complete=true`입니다. 이후 손실
문항을 확인한 뒤 test 분할을 실행합니다. `--allow-fallback-analysis`나 결과
CSV에서 실패 문항을 삭제하는 방식으로 진행하지 마세요.

```powershell
docker compose stop backend
if ($LASTEXITCODE -ne 0) { throw 'backend 중지 실패' }
try {
    docker compose run --rm --no-deps -T -e SEARCH_REFERENCE_YEAR=2026 backend python -u -m experiments.namuwiki.evaluate_context_step8 --split dev
    if ($LASTEXITCODE -ne 0) { throw '전체 dev 평가 실패' }
} finally {
    docker compose up -d backend
}

$out = '.\artifacts\context_eval_step8\results_dev'
$run = Get-Content "$out\run_info.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$summary = Get-Content "$out\paired_summary.json" -Raw -Encoding UTF8 | ConvertFrom-Json
if ($run.status -ne 'complete' -or -not $summary.complete -or $summary.evaluated -ne 74) {
    throw '전체 dev 결과가 아직 완료되지 않았습니다.'
}
$summary.groups.context_all.metrics.hit10 | Select-Object off, on, gained, lost
$summary.groups.regression_all.metrics.hit10 | Select-Object off, on, gained, lost
$summary.issues
```

`context_all`의 개선과 `regression_all`의 손실을 함께 검토합니다. Top-10
손실이 없더라도 `candidate_hit30`의 `lost`와 `issues`를 살펴봐야 합니다.
