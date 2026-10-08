# Context 분석 경계 보완 및 실제 검색 적용 검증

현재 Context는 SearchRouter에 연결되어 있다. 이번 작업은 반복된 분석 오류를 보완하고, 고정 평가가 완료되면 측정한 설정을 서버 생성 코드에 적용하여 실제 HTTP 검색까지 확인하는 절차를 추가한다. 첨부된 최신 실행은 독립 60문항의 분석 중 fresh032에서 중단되었으므로 최종 평가 완료 기록은 아직 없다.

## 수정한 동작

- 단독 `커버`는 의미가 모호하다. 가수의 재해석·커버 공개·화제·역주행은 Context 단서이며, 표지의 색·글씨·그림·질감·분위기 묘사는 Image 단서다. 의미가 불명확하면 원문을 Text 검색에 남긴다.
- `앨범 커버`·`표지`와 실제 표지 묘사는 기존 이미지 경로를 유지한다. 커버 공연과 표지가 한 문장에 함께 나오면 각 단서가 살아남으며, 공연의 Context 검색 문장에 표지 묘사를 덧붙이지 않는다.
- 소리의 `background vocals` 같은 적절한 공간 표현은 기존 규칙으로 처리한다. 실제 사진·그림·커버 이미지가 오디오 프롬프트로 들어가면 계속 거부한다. 이번 로그의 첫 `background` 재시도에는 실패한 원본 영문 프롬프트가 없으므로 그 거부가 오판이었다고 단정하지 않았다.
- 가사·제목에 들어 있는 사건 표현, 앞으로 커버할 곡을 추천해 달라는 요청을 외부 사건으로 만들지 않는다. 불확실한 회상은 Context 신뢰도를 낮게 유지한다.
- 평가의 가중치·라벨·근거 검토 기준·판정 기준은 그대로 사용한다. 크롤링·임베딩·Qdrant 재적재는 이번 수정에 필요하지 않다.

## 여기서 완료한 검증

| 검증 | 결과 | 의미 |
|---|---:|---|
| 서로 다른 합성 질의 조합 | 1,851개 통과 | 9종 사건 × 표지·소리 단서 × 단서 순서 × 마침표·쉼표·개행 |
| 새 분석 경계 테스트 | 1,921개 통과 | 위 조합과 활용형·한 문장 혼합·모호함·가사·예정 활동·음성 프롬프트 오염 |
| 적용·HTTP 검사 절차 테스트 | 70개 통과 | 미완료·회귀·미검토 근거·코드 변경 차단, 설정 편집, 백업, 응답 검증 |
| 관련 기존 테스트 포함 | 2,413개 통과 | 앞서 수정한 공연·대회·방송·소리 공간 표현 경계 포함 |
| 관측된 분석 재처리 | 137개, 오류·채널·Context 문장·가중치 변경 0개 | dev 74 + test 32 + 이미 관측한 독립 분석 31개 |
| 보고된 실패 재현 | 이전 코드 거부 → 수정 코드 Context+Audio 허용 | 실제 원문과 합성 모델 응답으로 파서 동작 확인 |
| 문법 | Python 3.11, PowerShell 정적 문법 통과 | 로컬 테스트 런타임은 Python 3.12이며 Windows 자체 실행은 여기서 하지 않음 |

HTTP 응답 테스트에는 가짜 전송을 사용했다. 새 Gemini 분석·사용자 Qdrant를 사용하는 실제 검색 재평가와 HTTP 검증은 아래 명령으로 사용자 환경에서 실행한다. 합성 조합의 수는 모든 임의 질의를 커버했다는 의미가 아니다.

## 업로드한 실행에서 관측된 성능

이 수치는 이번 수정 이후의 새 실행 결과가 아니다. 사용자가 보낸 최신 ZIP에 있는 고정 설정·리랭킹 OFF 결과다.

| 평가군 | 문항 | Candidate@30 OFF→ON | Hit@10 OFF→ON | MRR@10 OFF→ON |
|---|---:|---:|---:|---:|
| dev Context 양성 | 18 | 7/18 → 16/18 | 6/18 → 14/18 | 0.202 → 0.531 |
| test Context 양성 | 8 | 8/8 → 8/8 | 6/8 → 8/8 | 0.575 → 0.906 |
| dev 기존 검색 회귀 | 53 | 44/53 → 45/53 | 34/53 → 34/53 | 0.447 → 0.447 |
| test 기존 검색 회귀 | 23 | 16/23 → 16/23 | 13/23 → 13/23 | 0.392 → 0.392 |

검사한 이미지 필수 경로는 dev 9개, test 3개에서 OFF/ON 모두 임베딩·DB 조회 구간까지 확인되었다. 기존 분석 137개가 이번 수정에서도 같은 경로를 유지하지만, 성능 수치를 새 코드의 실제 실행 결과로 대신하지 않는다. 이미 관측한 dev/test는 회귀 자료이며, 새 독립 세트도 코퍼스 근거로 작성한 질의다. 실제 사용자 분포의 무작위 표본은 아니다.

## 1. ZIP 적용 및 빠른 경계 검사

VSCode에서 새 Org 프로젝트의 PowerShell 터미널에 입력한다. ZIP에는 수정 코드·합성 검사·실행 스크립트·이 문서만 들어 있다. 원래 데이터·평가 CSV·캐시·실행 로그를 덮어쓰지 않는다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Cover_Meaning_Fix_20261006*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '이번 수정 ZIP을 프로젝트 폴더 또는 Downloads에 저장해 주세요.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\experiments\namuwiki\run_context_cover_meaning_fix.ps1 -CheckOnly
```

빠른 검사에서는 Gemini 호출·Qdrant 적재 없이 실제 QueryAnalyzer 파서의 분리 결과를 볼 수 있다. `rendition_with_sound`는 Context=True, Image=False, Audio=True이고, 실제 표지는 Image=True로 유지되어야 한다.

## 2. 고정 평가를 끝까지 실행하고 실제 서버에 적용

```powershell
.\experiments\namuwiki\run_context_cover_meaning_fix.ps1 -ApplyAfterPass
```

기본 출력은 `artifacts/context_fixed_followup_20261006_cover_meaning_fix`다. 변경된 소스는 이전 평가의 코드 해시와 다르므로 새 출력 폴더에서 새 등록·새 분석을 생성한다. 기존 캐시를 이 폴더로 옮기지 않는다.

실행 순서:

1. Docker 빌드와 전체 Context·재질문·설명 테스트.
2. 백엔드 중지 후 dev 74, test 32, 독립 60문항의 OFF/ON 검색 비교. 각 질의는 동일 분석으로 양쪽을 검색하고, 리랭킹은 OFF다.
3. 별도 10문항 재분석 검사와 표시 근거 검사. 독립 60문항의 분모에 재분석을 추가하지 않는다.
4. 모든 단계의 `passed_observed_tests`·회귀 문제 없음·미검토 근거 0개를 확인한 뒤 호스트에서 `get_search_router`의 설정 인자만 편집한다. 원본은 출력 폴더의 `activation/backups/`에 남긴다.
5. 백엔드를 다시 빌드·재생성하고 `/health`의 모델 준비 완료를 기다린다.
6. 실제 `/api/v1/search`로 새 분석을 수행하여 알려진 Context 두 질의의 Candidate@30, Image·Audio·대조 질의, 원문 보존, 실제 경로 기여, 정식 제목, 리랭킹 미실행, 근거 필드·출처 URL을 확인한다. 같은 분석을 돌려보내 거절 ID 5개를 제외하고 후보 30개를 다시 채우는지도 확인한다.
7. 콘솔 로그와 평가·적용·API 보고서를 새 통합 피드백 ZIP으로 저장한다. 기존 ZIP은 보존한다.

서버 생성 코드에 적용할 값:

| 설정 | 값 |
|---|---:|
| `context_weight` | 0.5 |
| `context_fact_k` | 100개 사실 |
| `context_sparse_k` | 100개 곡 프로필 |
| `context_named_media_multiplier` | 2.0 |
| `default_candidate_k` | 30개 곡 |
| API 확인 `top_k` | 10개 |
| API 확인 `use_rerank` | False |

이 과정은 서버 전체 리랭커 설정이나 프런트엔드의 리랭킹 선택을 바꾸지 않는다. 여기서 증명하는 검색 성능은 리랭킹 OFF 조건이다. 프런트엔드가 `use_rerank=True`를 보내면 다른 정렬 조건이므로 같은 수치를 적용했다고 설명해서는 안 된다.

## 3. 완료 여부를 짧게 확인

```powershell
$out = '.\artifacts\context_fixed_followup_20261006_cover_meaning_fix'
$evaluation = Get-Content "$out\fresh\validation_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$evaluation | Select-Object status, completed_phases
$evaluation.reports.independent | Select-Object evaluated, status, violations
$evaluation.analyzer_repeat_check | Select-Object question_count, violations
if (Test-Path "$out\activation\activation_report.json") {
    Get-Content "$out\activation\activation_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json | Select-Object status, source_changed, api_verified
}
if (Test-Path "$out\activation\api_smoke_report.json") {
    $api = Get-Content "$out\activation\api_smoke_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    $api | Select-Object status, error_type, error
    $api.rows | Format-Table query_id, candidate_count, result_count, positive_paths, client_ms
}
```

완료 기준은 평가 `passed_observed_tests`, 독립 문항 60개, 재분석 문항 10개, 적용 `settings_applied`·`api_verified=True`, API `passed_api_smoke`다. 실제 서버의 Context 설정과 알려진 HTTP 검색 경로를 확인했다는 뜻이다. 모든 미래 질의의 정답률이나 제3자 원문의 진위를 보장하는 판정은 아니다.

## 중단·재시도와 보낼 파일

- 평가가 중단되면 최종 설정 적용으로 넘어가지 않는다. 콘솔 마지막의 **정확한 `Send this file:` 통합 ZIP 경로**를 보내면 원래 실패·캐시·전체 콘솔 로그를 함께 확인할 수 있다. 가중치나 정답·검토 라벨을 통과시키기 위해 바꾸지 않는다.
- 코드·데이터·환경이 그대로이고 일시적인 API 장애라면 같은 명령·폴더로 이어서 실행한다. 코드나 코퍼스를 변경하면 새 출력 폴더가 필요하다.
- 평가가 완전히 통과한 뒤 빌드·서버 준비·HTTP 확인만 실패했다면 다음 명령으로 적용/API 단계만 재시도할 수 있다. 평가를 다시 분석하지 않는다.

```powershell
.\experiments\namuwiki\run_context_cover_meaning_fix.ps1 -ActivateOnly
```

- 설정 적용 후 `dependencies.py`가 변경되므로 같은 등록 폴더로 전체 평가를 다시 실행하면 변경 감지가 정상적으로 중단한다. 전체 재측정이 필요하면 새 `-OutputDir artifacts/원하는새이름`을 사용한다.
- 추가 근거 검토가 필요한 상태는 자동 최종 적용 대상이 아니다. 보고서의 검토 항목을 확인해야 한다. 이번 스크립트는 검토 결과를 임의로 긍정으로 바꾸지 않는다.
- API 검사에서 새 분석이 필수 후보를 누락하거나 오류가 나면 적용 보고서에 `api_verified=False`가 남는다. 성공으로 발표하지 않고 출력된 통합 피드백 ZIP을 보낸다. 서버와 원본 백업은 남아 있다.

## 공개 PR 범위

ZIP에서 추가한 코드·합성 테스트·실행 스크립트·이 문서와 최종 적용 과정에서 바뀐 `src/backend/api/dependencies.py`를 검토한다. `data/`, `artifacts/`, 평가 CSV·리뷰·캐시·로그·피드백 ZIP·음원·원본 meta·수집 목록은 공개 커밋에 포함하지 않는다. ZIP 자체도 커밋하지 않는다.

리뷰 설명은 세 부분으로 구분하면 된다: (1) 커버 다의어를 실제 원문의 표지·공연 증거로 분리한 파서 수정, (2) 공개 합성 경계 테스트와 실제 OFF/ON 평가의 구분, (3) 통과한 평가와 서버 설정·HTTP 실행 결과를 각각 기록한 적용 검증. 측정하지 않은 독립 평가 결과를 기존 회귀 수치로 대신하지 않는다.
