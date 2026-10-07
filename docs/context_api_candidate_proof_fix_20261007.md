# Context API 최종 검증 보완 — 2026-10-07

이번 피드백의 성능 비교는 이미 통과했다. 중단 지점은 실제 API 검사다.
기존 검사는 `results`의 화면 Top-10만 읽으면서 Context 후보의 점수 기여를
확인했다. 정답이 후보 11–30위에 있으면 이 방식으로는 기여를 확인할 수 없다.
실제 저장된 `q203` 응답은 HTTP 200, 정답 후보 22위, Context 경로 실행을
보여 준다. 그러나 정답에 붙은 Context 기여는 해당 Top-10 응답에 들어 있지 않다.

이 보완은 실험용 검증 코드와 그 테스트만 수정한다. 검색 관련 등록 코드
142개의 바이트가 보내준 평가 기록과 일치함을 확인했다. 검색 라우터,
Qdrant 백엔드, QueryAnalyzer, 가중치, 라벨, 임베딩, `.env`를 수정하지 않는다.

## 보내준 실제 환경의 결과

`context_lyric_scope_20261007_040024_004464`의 `validation_report.json`은
`passed_observed_tests`다. dev 74개, test 32개, 추가 60개를 모두 측정했고
10개 반복 분석 검사도 통과했다. 위반과 미검토 표시 근거는 모두 0개다.

| 같은 분석으로 Context OFF → ON | 문항 수 | Candidate@30 | Hit@10 |
|---|---:|---:|---:|
| dev Context 질의·혼합 질의 | 18 | 6/18 → 15/18 | 6/18 → 13/18 |
| test Context 질의·혼합 질의 | 8 | 8/8 → 8/8 | 6/8 → 8/8 |
| 추가 Context 질의·혼합 질의 | 36 | 26/36 → 31/36 | 23/36 → 29/36 |
| dev 기존 회귀 질의 | 53 | 46/53 → 47/53 | 36/53 → 36/53 |
| test 기존 회귀 질의 | 23 | 17/23 → 17/23 | 14/23 → 14/23 |
| 추가 기존 회귀 질의 | 18 | 18/18 → 18/18 | 18/18 → 18/18 |

Context 무관 대조 질의는 각 단계에서 결과를 유지했다. 위 표의 Context 행은
대조 질의를 제외한다. 화면 Hit@10과 후보 Candidate@30을 별도로 계산한다.
리랭킹은 OFF이고 설정은 `0.5 / 2.0 / 100 / 100 / 30`, 화면 Top-10이다.
코퍼스는 3,016곡, Context Dense 3,995개, Sparse 478개다.

이 결과는 이미 관측·수정에 사용한 질의의 재분석이다. 처음 보는 블라인드
표본으로 재표현하지 않는다. 모든 가능한 자연어 질의의 정답이나 사실의
진위를 보증하지 않는다. `q203`의 후보 진입과 Top-10 진입도 서로 다른 결과다.

## 새 API 검사에서 확인하는 것

1. 기존 17개 질의와 거절 후 후보 보충 1건을 동일하게 검사한다.
2. 처음 요청은 새 QueryAnalyzer 분석으로 화면 Top-10을 조회한다.
3. Context 검사 3건은 같은 응답의 `analysis`를 `prior_analysis`로 보내고,
   `candidate_k=30`, `use_rerank=false` 및 나머지 요청 상태를 그대로 유지하며
   `top_k=30`으로 한 번 더 조회한다. 추가 Gemini 분석을 하지 않는다.
4. 두 HTTP 요청 각각의 응답 헤더와 새 타이밍 기록을 연결한다.
5. 분석, 후보 30개의 ID·순서, 기존 Top-10의 곡·점수·기여·근거가 모두 같아야 한다.
6. 정답 곡에 실제 양수 Context 점수 기여가 있어야 한다. 후보에 ID가 있거나
   `path.context` 로그만 있는 것으로는 통과시키지 않는다.
7. 거절 검사에서는 거절 ID와 이전 후보 상태를 보존하고, 거절 곡 재등장을 차단한다.
8. 이미지·오디오는 전용 검색 문장, 양수 가중치, 같은 요청 안의 임베딩과
   인덱스 조회 실행을 확인한다. 관련 대조 질의의 Context 기여와 잘못된 근거도 차단한다.

전체 흐름은 기본 검사 18건과 후보 확인 요청 3건, 총 21개 HTTP 요청이다.
`required_path_in_top10`은 검색 경로가 Top-10에 기여했는지를 뜻한다.
정답 자체의 화면 진입 여부는 `target_top10_ranks`와
`context_candidate_proof.target_in_top10`을 확인한다.

## 적용 및 실행 — 프로젝트 루트의 PowerShell

변경 ZIP을 프로젝트 폴더 또는 Downloads에 저장한 뒤 실행한다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_API_Candidate_Proof_Fix_20261007*.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw 'API 검증 수정 ZIP을 찾지 못했습니다.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
.\experiments\namuwiki\run_context_api_candidate_check.ps1 -OutputDir 'artifacts/context_lyric_scope_20261007_040024_004464'
```

기존 평가 출력 폴더를 지정해야 한다. 이 경로는 이번 피드백에 실제 기록된 경로다.
새 평가 폴더를 만들거나 기존 전체 평가 스크립트를 다시 실행할 필요가 없다.
설치기는 기존 실험 코드가 알려진 버전인지 먼저 확인하고 원본을 보관한다.
다른 팀원의 변경이 있으면 덮어쓰지 않고 중단한다.

스크립트는 빌드 → 회귀 테스트 → 기존 환경·코퍼스·유효 설정 확인 → 서버
준비 대기 → 실제 API 검사 → 평소 설정으로 백엔드 복구 → 피드백 ZIP 저장을 수행한다.
공용 MongoDB를 시작하거나 중지하지 않는다. `.env`도 편집하지 않는다.
이미 끝난 비교 평가와 캐시, 원래 `runner_status.json`은 바이트 그대로 보존한다.
새 API 상태는 `api_resume_status.json`에 남긴다.

```powershell
$out = '.\artifacts\context_lyric_scope_20261007_040024_004464'
Get-Content "$out\api_resume_status.json" -Raw -Encoding UTF8 | ConvertFrom-Json |
    Select-Object status, performance_verified, api_verified, completed_evaluation_preserved, backend_restore_exit
$api = Get-Content "$out\activation\api_smoke_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$api.rows | Where-Object required_path -eq 'context' |
    Select-Object query_id, status, target_candidate_ranks, target_top10_ranks, required_path_in_top10, context_candidate_proof
```

완료 기준은 `status=passed_api_resume`, `api_verified=true`,
`completed_evaluation_preserved=true`, `backend_restore_exit=0`이다.
기존 `runner_status.json`은 마지막 실패 당시 기록이므로 그대로 `stopped`로 남는다.
다시 실패하면 콘솔의 마지막 `Send this file:`에 나온 새 ZIP 하나를 보내면 된다.
매번 새 파일명으로 저장하므로 기존 피드백 ZIP이 열려 있어도 덮어쓰지 않는다.

## 로컬 검증과 한계

검색 관련 회귀 테스트, 합성 HTTP 요청, 실제 FastAPI 응답 경로, 후보 1/10/11/16/
22/24/30위, 잘못된 요청·분석·타이밍·기여·후보·거절 상태, 실행 실패 시 복구,
보고서 보존, 설치 전 원본 확인을 검사했다. 상세 결과는 동봉한 로컬 검사 JSON에 있다.
로컬 합성 HTTP 검증은 실제 곡 성능 측정을 대체하지 않는다. 변경 후의 Gemini와
실제 로컬 Qdrant를 이용한 API 결과는 위 명령을 사용자 환경에서 실행해 확인한다.

공개 레포에는 실험 코드·합성 테스트·문서만 올린다. `artifacts`의 피드백, 질의,
라벨, 분석 캐시, `.env`, 임베딩, Qdrant 데이터와 ZIP은 팀의 비공개 경로로 공유한다.
