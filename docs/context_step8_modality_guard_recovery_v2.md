# Context 8단계 전체 dev 평가 복구: nw013

## 결과 해석

이전 수정으로 `nw008`과 `nw010`은 실데이터 검색에서 정상 완료됐습니다.
`nw008`은 정답 후보·Top-10 모두 Context를 켠 경우 1위로 진입했고,
`nw010`은 양쪽 모두 1위였습니다. 이 두 문항만의 결과를 74문항 전체의
성능으로 해석하지 않습니다.

전체 평가에서는 `nw013`에서 Gemini의 텍스트 전용 분석이 세 차례 거부돼
규칙 폴백이 나왔습니다. 정확한 오판 원인은 **`음원`이 아니라 `탈락`에 들어 있는
`락`**입니다. 장르 검사가 단어의 일부에 매칭되어 `audio_english_query`를
요구했습니다. 기존의 성공한 항목은 분석 캐시에 있지만, 이 평가 시도는
`run_info.json`의 완료 상태를 만들지 못했으므로 전체 결과는 미완료입니다.

수정은 한국어 록·락 장르의 단독 표현과 흔한 합성어를 보존하면서
`탈락`·`연락`·`기록` 내부 매칭을 막습니다. `사운드트랙`처럼 수록 이력을
가리키는 단어, `후렴의 영어 가사`처럼 가사 내용만 가리키는 표현,
`힙합 경연 프로그램`·`무대에서 노래하는 장면`·과거에 `부르던` 버전 같은
외부 이력은 CLAP용 청각 단서로 취급하지 않습니다. 록 밴드, 악기 소리,
보컬 인원·역할, 리듬 변화 등 실제 청각 단서는 계속 검사합니다.

## 적용과 단위 검증 (PowerShell, 프로젝트 루트)

이 ZIP은 앞서 받은 visual fix를 **포함한 전체 파일**입니다. 이전 ZIP을 다시
풀 필요는 없습니다. ZIP에 평가 CSV, 수집 데이터, 음원, Qdrant 산출물은
포함되지 않습니다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Step8_Modality_Guard_v2.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw 'v2 ZIP을 프로젝트 폴더 또는 Downloads에 넣어 주세요.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
git diff --check
docker compose build backend
if ($LASTEXITCODE -ne 0) { throw 'backend 빌드 실패' }
docker compose run --rm --no-deps -T backend python -m pytest -q tests/context/test_context_step8_modality_boundary.py tests/context/test_context_query_analyzer.py tests/context/test_context_step8_evaluation.py
if ($LASTEXITCODE -ne 0) { throw '회귀 테스트 실패' }
```

## 캐시 보관과 실제 검색

`analysis_cache.json`에는 수정 전 모달리티 후처리 지문과 `nw013` 폴백이
포함되어 있으므로, 해당 캐시는 그대로 재사용할 수 없습니다. 삭제 대신 보관하고
새 지문으로 재분석합니다. 기존 `smoke_visual_fix` 결과도 해당 시점의 기록으로
남기되, 새 결과와 단순 합산하지 않습니다.

```powershell
$cache = '.\artifacts\context_eval_step8\analysis_cache.json'
if (Test-Path -LiteralPath $cache) {
    $backup = '.\artifacts\context_eval_step8\analysis_cache_before_audio_fix_{0}.json' -f (Get-Date -Format 'yyyyMMdd_HHmmss')
    Move-Item -LiteralPath $cache -Destination $backup
    Write-Host "기존 분석 캐시 보관: $backup"
}

docker compose stop backend
if ($LASTEXITCODE -ne 0) { throw 'backend 중지 실패' }
try {
    docker compose run --rm --no-deps -T -e SEARCH_REFERENCE_YEAR=2026 backend python -u -m experiments.namuwiki.evaluate_context_step8 `
        --split dev --query-ids nw013,nw017,c602,c605,c607 --show-top `
        --output-dir artifacts/context_eval_step8/smoke_audio_fix
    if ($LASTEXITCODE -ne 0) { throw '오분류 경계 실검색 실패' }

    docker compose run --rm --no-deps -T -e SEARCH_REFERENCE_YEAR=2026 backend python -u -m experiments.namuwiki.evaluate_context_step8 --split dev
    if ($LASTEXITCODE -ne 0) { throw '전체 dev 평가 실패' }
} finally {
    docker compose up -d backend
}
```

중간의 실패는 `$LASTEXITCODE`로 확인합니다. 정상이라면 새 `analysis_cache.json`에
해당 다섯 질의가 먼저 저장되어 전체 평가에서도 같은 분석을 재사용합니다.

```powershell
$out = '.\artifacts\context_eval_step8\results_dev'
$run = Get-Content "$out\run_info.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$summary = Get-Content "$out\paired_summary.json" -Raw -Encoding UTF8 | ConvertFrom-Json
if ($run.status -ne 'complete' -or -not $summary.complete -or $summary.evaluated -ne 74) {
    throw '전체 dev 결과가 아직 완료되지 않았습니다.'
}
$summary.groups.context_all.metrics.candidate_hit30 | Select-Object off, on, gained, lost
$summary.groups.context_all.metrics.hit10 | Select-Object off, on, gained, lost
$summary.groups.regression_all.metrics.candidate_hit30 | Select-Object off, on, gained, lost
$summary.groups.regression_all.metrics.hit10 | Select-Object off, on, gained, lost
$summary.issues
```

`run_info.json`의 `status=complete`, `paired_summary.json`의 `complete=true`,
`evaluated=74`가 함께 확인되어야 dev 평가 완료입니다. 이후 Context 개선과
기존 가사·분위기·이미지 질의의 손실 목록을 비교하고 test 분할로 진행합니다.
