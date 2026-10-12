# Context 코드 리뷰 보완과 새 분석 재검증

이 회귀 수정 릴리스의 실행 명령과 실패 분석은
[상세 Context 단서 보존 수정](context_media_specificity_fix_20261006.md)을 먼저 참고한다.

이 패키지는 현재 프로젝트의 병합된 파일을 읽고 필요한 부분만 수정한다.
기존 `search_router.py`, `qdrant_backend.py`를 옛 버전 파일로 교체하지 않는다.
실행 전에 변경 대상 전체를 검사하고, 수정 전 파일은 실행 폴더의
`installation/backups/`에 보관한다. 알려진 구조와 다르면 파일을 덮어쓰지 않고 중단한다.

## 변경 내용

1. QueryAnalyzer의 Context 예시를 가상 작품으로 바꾼다. 평가 질의의 작품·인물·장면을
   그대로 예시로 사용하지 않으며, 사용자가 말하지 않은 인물이나 장면을 추가하지
   말라는 지시를 넣는다. 실제 단서 추출 규칙과 이미지·오디오·가사 분리 규칙은 유지한다.
2. 라우터 팩토리가 환경변수 다섯 개를 실제로 읽는다. 빠진 값은 아래 기본값을
   사용하고, 잘못된 숫자는 환경변수 이름을 포함한 오류로 알려 준다.
3. `docs/data_policy.md`에서 나무위키 검색을 연결 전이라고 설명하던 문장을 갱신하고,
   API의 근거 필드와 표시 조건, 데이터 공개 범위, 평가의 한계를 추가한다.
4. 비교 평가가 라우터 설정을 몰래 고정값으로 덮어쓰던 동작을 제거한다.
   실제 설정이 고정 비교 조건과 다르면 중단한다. 기존 평가 테스트의 빈 가짜 라우터에는
   실제 기본값을 넣어 테스트 목적을 유지한다. 활성화 스크립트가 환경변수 연결을
   다시 숫자 리터럴로 되돌리지 않도록 보호한다.

| 환경변수 | 기본값 | 의미 |
|---|---:|---|
| CONTEXT_WEIGHT | 0.5 | 기존 검색과 결합할 Context 경로 가중치 |
| CONTEXT_NAMED_MEDIA_MULTIPLIER | 2.0 | 명시된 작품 단서의 Context 가중치 배수 |
| CONTEXT_FACT_K | 100 | Dense에서 조회할 사실 수; 곡 수와 다름 |
| CONTEXT_SPARSE_K | 100 | Sparse에서 조회할 곡 프로필 수 |
| SEARCH_DEFAULT_CANDIDATE_K | 30 | 요청에서 따로 정하지 않은 기본 후보 수 |

`.env`에 없어도 기본값은 적용된다. 팀의 실행 조건을 명확히 하기 위해 명시적으로
기록하는 편이 좋다. 기존 예제의 `1.0 / 1.0`을 복사해 둔 경우 이 실행은 그 두 값을
기존 고정 재검증값인 `0.5 / 2.0`으로 정리한다. 임의의 새 가중치 최적화를 수행하지 않는다.
Gemini API 키, 공용 MongoDB URI와 다른 `.env` 항목은 보존한다.

## 한 번에 적용하고 검증하기 — Windows PowerShell

다운로드한 ZIP을 프로젝트 폴더 또는 Downloads에 두고 프로젝트 루트에서 실행한다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Review_Cleanup_20261006*.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 찾지 못했습니다.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
.\experiments\namuwiki\run_context_review_cleanup.ps1 -Mode All
```

Windows PowerShell이 스크립트 실행 정책으로 차단하면 현재 터미널에서만 다음을
실행한 뒤 다시 실행한다. 이미 실행 가능한 환경에서는 필요 없다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

기본 실행은 다음을 순서대로 수행한다.

- backend 중지; 프롬프트·팩토리·환경변수·문서·평가 코드의 필요한 부분만 적용
- `.env`의 다섯 항목 정리; 기존 공용 키와 MongoDB 설정 보존
- Docker 이미지 빌드와 `tests/context`, Qdrant 수명주기, 재질문, 설명 테스트
- 실제 라우터의 설정과 전체 Context 세대 활성 상태 검사
- 새 실행 폴더의 새 캐시로 dev 74문항, test 32문항, 추가 60문항 OFF/ON 비교
- 10문항 별도 재분석; 이미지·오디오 embed/query 실행과 표시 근거 검사
- 비교 보고서가 통과한 경우에만 해당 실행과 활성화 기록을 연결
- 리랭킹 OFF인 실제 HTTP 검색으로 API·이미지·오디오·재질문 후보 보충 검사
- 정상 서버 설정으로 backend 재생성; 성공·실패 모두 고유한 피드백 ZIP 생성

공용 MongoDB를 사용하므로 MongoDB 컨테이너를 새로 올리지 않는다. Qdrant를 재적재하거나
artifact·BM25·임베딩을 다시 만들지 않는다. 평가 중에는 backend를 중지해 로컬 Qdrant 잠금
충돌을 피한다. 마지막 HTTP 검증은 실행 중인 서버를 호출하며 두 번째 Qdrant 클라이언트를
열지 않는다. Gemini를 호출하므로 실제 평가에는 공용 API 키의 사용량이 발생한다.

`compose.context-test.yml`이 있으면 함께 사용하며, 다섯 설정만 정의한
`compose.context-ranking.yml`을 마지막에 적용한다. 평가는 임시 Compose 파일로
리랭킹·기준 연도·타이밍 로그만 바꾸고, Context 설정은 실제 서버 설정을 그대로 검증한다.
PowerShell 세션에 같은 이름의 환경변수가 있으면 Compose가 `.env`보다 우선한다.
예상과 다른 값이면 실제 설정 검사에서 중단하므로 해당 값을 확인한다.

실행마다 `artifacts/context_review_cleanup_<시각>/`을 만든다. 기존 평가 폴더나 캐시를
삭제하지 않는다. 끝에 출력되는 정확한 `Send this file:` 경로의 ZIP을 전달하면 된다.
ZIP에는 보고서·분석 캐시·실제 콘솔 로그만 들어가며 `.env`, 소스 백업과 곡 원본은
포함하지 않는다. 피드백 ZIP은 개인 실행 자료이므로 공개 저장소에 커밋하지 않는다.

## 결과 판정

최종 `runner_status.json`의 `status=passed_review_cleanup_validation`,
`performance_verified=true`, `api_verified=true`, `backend_restore_exit=0`을 확인한다.
그와 함께 `fresh/validation_report.json`은 `passed_observed_tests`,
`activation/api_smoke_report.json`은 `passed_api_smoke`여야 한다.

Candidate@30·Hit@10·MRR@10의 OFF/ON 결과는 `fresh/validation_report.md`와 각 분할의
`*_detail.csv`에서 확인한다. 기존 보고서 수치로 이번 수정 후 성능을 대신 주장하지 않는다.
`evidence_review_required`는 프로세스 반환값이 0이어도 완료로 처리하지 않는다.
분석 fallback, 회귀 손실, 경로 미실행, 검토하지 않은 근거는 중단 사유로 남긴다.
가중치나 라벨을 바꿔 통과시키지 말고 피드백 ZIP으로 원인을 확인한다.

dev/test와 기존 추가 60문항은 이전에 본 질의다. 새 Gemini 분석을 사용해도 새 블라인드
표본이 되지는 않는다. 이번 검증은 프롬프트 예시를 정리한 뒤 같은 코퍼스와 고정 설정에서
실행하는 회귀 재검증이다. 독립적인 일반화 주장을 하려면 이번 결과를 보고 수정하지 않은
새 질의셋을 따로 등록해야 한다. Top-10 표시 수의 최적값이나 모든 사용자 질의의 성공을
입증하는 실험은 아니다.

## 중간에 API 오류가 나면

같은 코드·입력·설정·코퍼스로만 재개할 수 있다. 실제 출력된 실행 폴더를 넣는다.

```powershell
.\experiments\namuwiki\run_context_review_cleanup.ps1 -Mode All `
    -OutputDir 'artifacts/실제_실행_폴더명' -Resume
```

코드를 다시 고치거나 모델·데이터·설정을 바꿨다면 `-Resume` 대신 기본 명령으로
새 실행을 만든다. `-Mode Tests`는 코드 테스트까지만 수행하며 성능 측정 완료가 아니다.
기존 `.env`를 수정하지 않으려면 `-NoUpdateEnv`를 사용할 수 있으나, 실효값이 비교 조건과
다르면 평가를 중단한다.

## 커밋할 범위

이번 ZIP에서 추가된 코드·테스트·문서·비밀 값 없는 Compose 파일과 설치 후 수정된
`src/backend/api/dependencies.py`, `src/retrieval/query_analyzer.py`, `.env.example`,
`docs/data_policy.md`, 기존 평가·활성화 스크립트, 평가 테스트의 가짜 라우터 수정만
검토하여 커밋한다. `.env`, `artifacts/`, 피드백 ZIP, 평가 입력·라벨·캐시는 제외한다.
최종 커밋 전 `git diff --check`, `git status --short`, 변경 파일 diff를 확인한다.
