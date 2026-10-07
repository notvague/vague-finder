# 상세 Context 단서가 일반 OST 검색으로 축약되는 회귀 수정

## 실제 실패와 원인

2026-10-06 사용자가 실행한 `context_review_cleanup`의 피드백을 기준으로 한다.
코드 테스트는 **3,478개 통과**했고, 실제 라우터 설정과 완료 코퍼스 검사도 통과했다.
dev/test를 끝낸 뒤 test의 한 문항에서 회귀 게이트가 막혔다. 추가 60문항의 이번
재분석과 실제 HTTP 최종 검사는 실행되지 않았다. backend 복구는 성공했다.

| 분할·질의 그룹 | n | Candidate@30 OFF → ON | Hit@10 OFF → ON |
|---|---:|---:|---:|
| dev Context 긍정 질의 | 18 | 7/18 → 16/18 | 5/18 → 14/18 |
| dev 기존 검색 회귀 질의 | 53 | 44/53 → 45/53 | 34/53 → 34/53 |
| test Context 긍정 질의 | 8 | 8/8 → 8/8 | 6/8 → 8/8 |
| test 기존 검색 회귀 질의 | 23 | 18/23 → 18/23 | 14/23 → 13/23 |

손실 문항 `c703`은 정답이 OFF 10위, ON 13위였다. 작품명이 없는 상세한 기억을
모델이 일반적인 관계명만 남긴 검색어로 축약했다. 기존 관련성 검사는 상세한
미상 작품 단서에 적용되므로, 이 축약된 두 단어는 단순 OST 목록 조회로 처리됐다.
그 결과 정답을 뒷받침하지 않는 Context 후보가 가산됐다. 위 표는 **수정 전 실제
실행의 결과**이며, 이번 패키지 적용 후의 성능 수치로 사용하면 안 된다.

## 수정 범위

- 원문에 근거한 규칙 단서를 모델 검색어로 교체하는 **한 지점**에 검사를 추가한다.
  상세한 미상 작품 기억이 일반 OST/BGM 검색으로 축약되면 기존 원문 단서를 보존한다.
- 사용자가 명시한 작품명이 일반 관계명으로 사라지는 경우에도 원문 단서를 보존한다.
  작품명이 남아 있는 간결한 검색어와 진짜 일반 OST 추천은 계속 허용한다.
- 기존 Dense 사실 관련성 검사를 사용한다. 뒷받침하는 사실이 없는 상세한 미상 작품
  후보는 Context 가산에서 제외한다. 기존 Text/Image/Audio 후보의 점수는 유지한다.
- 프롬프트에도 사용자가 말한 장면·시기·가수 조건을 유지하라는 일반 지시를 넣는다.
  특정 평가 문항의 작품·인물·정답을 프롬프트에 넣지 않는다.

가중치 **0.5**, 명시 작품 배수 **2.0**, Dense 사실 조회 **100**, Sparse 프로필 조회
**100**, 후보 수 **30**, 표시 수 **10**으로 비교한다. 평가 입력·정답 라벨·코퍼스를
바꾸지 않는다. `search_router.py`와 `qdrant_backend.py`는 패키지에 포함하지 않으며,
병합한 파일의 본문도 설치기가 변경하지 않는다.

## 실행 — Windows PowerShell

현재 프로젝트 루트에서 다운로드한 ZIP을 적용하고 아래 명령을 한 번 실행한다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Media_Specificity_Fix_20261006*.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 찾지 못했습니다.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
.\experiments\namuwiki\run_context_media_specificity_fix.ps1 -Mode All
```

실행은 backend 중지 → 설치 대상 구조 확인 → 필요한 부분만 수정·백업 → 빌드 →
전체 Context 및 Qdrant 수명주기·재질문·설명 테스트 → 실효 설정/완료 코퍼스 확인 →
새 분석 OFF/ON 비교 → 실제 HTTP 검사 → 일반 서버 설정으로 복구 순서다.

현재 피드백에서는 `.env`의 다섯 설정이 이미 비교 조건과 일치했다. 새 환경변수를
수동으로 추가할 필요 없다. 기존 공용 Gemini 키와 MongoDB 설정은 보존한다.
Qdrant·임베딩·artifact·BM25를 다시 생성하지 않으며, 공용 MongoDB를 위한 실행에서
로컬 MongoDB 서비스를 올리지 않는다. 실제 분석에는 기존 Gemini 키의 사용량이 발생한다.

새 실행은 `artifacts/context_media_specificity_<시각>/`과 새 분석 캐시를 만든다.
기존 실패 폴더에 `-Resume`으로 이번 코드 변경을 이어 붙이지 않는다.
dev 74 + test 32를 통과해야 추가 60문항과 10문항 별도 재분석, 실제 이미지·오디오
경로 및 HTTP 검사가 이어진다. 회귀·fallback·미실행 경로·근거 미검토가 남으면
중단하고 피드백 ZIP을 생성한다. 반환값만 0인 근거 미검토 상태를 통과로 처리하지 않는다.

동일 코드·설정·입력·코퍼스에서 API 통신 오류로 재시도할 때만 새 출력 폴더명을 넣는다.

```powershell
.\experiments\namuwiki\run_context_media_specificity_fix.ps1 -Mode All `
    -OutputDir 'artifacts/실제로_출력된_새_폴더명' -Resume
```

## 검증 근거와 한계

로컬 검증 기록은 `context_media_specificity_local_checks.json`에 있다. 경계 테스트에는
일반 OST 추천, 상세 미상 작품 기억, 명시 작품명, 서로 다른 외부 사건, 가사·앨범아트·
소리의 비유, 복합 모달리티, 근거의 곡 일치, 원문 불변, 설치 반복·줄바꿈·백업·오류 복구를
포함한다. 실제 라우터의 순위 결합과 rejected-ID 보충을 검사하는 추가 네 테스트도 준비했다.
이 네 테스트의 최종 실행은 사용자의 Docker 환경에서 진행한다.

사용자가 보낸 코드 해시와 일치하는 Context 추출 규칙을 기준으로 기존 분석 106개를
재생했을 때 Context 단서가 달라진 것은 `c703` 한 건이었다. 나머지 분석 필드는 유지됐다.
공유 artifact의 3,016곡·3,995개 사실을 검사했을 때, 복원한 해당 단서를 지지하는 사실은
없었다. 따라서 기존 관련성 검사는 그 고정 분석의 근거 없는 OST 가산을 차단한다.
이는 **이미 공개된 분석의 원인 진단**이며, 새 Gemini 분석·벡터 검색·OFF/ON 성능 재측정을
대체하지 않는다. 특히 새 분석에서의 정답 순위를 미리 10위라고 보증하지 않는다.

dev/test와 추가 60문항은 이미 관찰한 질문이다. 새 분석을 사용해도 새 블라인드 표본이
되지 않는다. 이번 실행은 고정 설정의 회귀 재검증이며, 모든 사용자 질의의 성공이나
독립 표본에서의 통계적 일반화를 입증하는 실험으로 설명하지 않는다.

## 결과 확인과 공유

끝에 출력되는 **정확한 `Send this file:` 경로**의 피드백 ZIP을 전달한다.
그 ZIP은 설치 보고서, 실효 설정, 분할별 OFF/ON 상세 결과, 분석 캐시, 실제 콘솔 로그,
최종 HTTP 검사 결과를 묶는다. 공개 저장소에는 코드·테스트·문서와 비밀 값 없는 설정만
올리고, 개인 실행 자료·평가 입력·라벨·캐시는 올리지 않는다.

최종 완료 판정은 아래 값을 함께 확인한다.

- `runner_status.json`: `passed_review_cleanup_validation`, `performance_verified=true`,
  `api_verified=true`, `backend_restore_exit=0`
- `fresh/validation_report.json`: `passed_observed_tests`
- `activation/api_smoke_report.json`: `passed_api_smoke`

`-Mode Tests`는 코드 검사만 완료한 상태다. 가중치나 정답을 바꿔 게이트를 통과시키지 않는다.
설치한 소스의 변경 내역은 `git --no-pager diff -- src/retrieval/context_query.py
src/retrieval/query_analyzer.py`로 확인한다. Context 규칙의 원본 백업은 새 실행 폴더의
`specificity/backups/`에 보관한다.
