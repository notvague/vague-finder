# Context 후보 절단과 사실 근거 검증 수정

## 제출된 보고서의 결론

이전 단서 수정 진단에서 Context 양성 질의 18개 모두 단서가 생겼지만,
Candidate@30과 Hit@10은 OFF/ON 모두 12/18이었다. 순위가 오른 사례는
있었으나, 새 정답을 후보에 포함시킨 사례는 없었다. 대조군 3개와 기존
가사·분위기·이미지 질의 9개는 유지됐다. 이 수치는 성공 판정으로 바꾸지 않는다.

원시 Dense/Sparse 검색이 정답을 찾아도 기존 라우터의 RRF 상위 후보 절단이
가수·제목 등 정식 메타데이터에 대한 점수 보정보다 먼저 실행됐다. 예를 들어
Text 가중치가 1, Context 가중치가 0.5, 단서 신뢰도가 0.8이면 Context의
1위 기여는 `0.4 / 61`, Text 30위 기여는 `1 / 90`이다. 이 조건에서
Context 전용 후보는 기존 가수 일치 보정을 받을 기회를 잃는다.

## 코드 변경

- Context 후보가 실제로 있을 때에는 최초 RRF 결과를 먼저 자르지 않고,
  정식 Text 메타데이터에 대한 기존 보정을 적용한 뒤 최종 후보를 자른다.
  Context 내부 Dense/Sparse는 여전히 합친 한 경로로만 기여한다.
  가중치, 신뢰도, 후보 수, 화면 결과 수, 부스트 계수는 변경하지 않는다.
- Context가 없는 경우, 비활성화된 경우, 조회 실패 또는 빈 결과이면 기존
  절단 방식을 유지한다. 거절한 곡 제외와 후보 보충도 기존 순서로 수행한다.
  Context 후보의 자리를 별도로 예약하지 않으며, 보정 뒤에도 낮은 점수면 탈락한다.
- 사실 문장에서 숫자는 독립적인 수치와 단위로 확인한다. `1위`를 `2013년`의
  숫자 일부로 충족시키거나 `9개`를 `9월`로 충족시키지 않는다.
- 군 복무·위문열차·올킬·디자인 등 구체적인 사건 조건은 같은 사실 문장에
  있어야 한다. 일반 무대 설명이나 출연자 이름의 일치로 대체하지 않는다.
- 제작·발매·버전 사실이 다른 작품들의 제목만 명시하면, 현재 곡의 제목 또는
  현재 곡을 가리키는 표현이 없을 때 인용을 보류한다. 곡 문서에 있다는 것만으로
  그 문서가 언급한 다른 곡의 사건을 현재 곡의 근거로 사용하지 않는다.

관련성이 불충분하면 곡 후보는 유지하고 근거 표시만 보류한다.
단일 사실로 복합 단서 전체를 뒷받침하지 못하는 경우도 보수적으로 보류한다.
이 검증은 인용의 질의 관련성 검증이며 외부 문서의 사실성을 보증하지 않는다.

## 검증 범위

추가된 회귀 테스트는 실제 SearchRouter의 RRF, 메타데이터 보정, 거절한 ID
제외, 최종 후보 절단을 실행한다. 임베딩 서비스는 제어된 후보를 반환한다.
낮은 가중치의 Context 전용 정답, 보정 없는 탈락, 비활성 Context, Text/Image/Audio
합산, 후보 보충, 여러 Context 후보, Sparse 전용 근거 보류를 확인한다.
사실 검증은 숫자 부분 일치·단위·날짜·시간·다른 곡 언급·같은 문장 내 사건 조건을
확인한다. 진단 도구는 코퍼스·코드·이전 캐시·OFF 기준선 변경을 거부하고
체크포인트를 검증한다. 모든 가능한 자연어 표현을 보장하는 테스트는 아니다.

수정 전 라우터에서 낮은 가중치 후보가 먼저 탈락하는 테스트를 재현했고,
수정 후 같은 테스트가 통과했다. 제출된 보고서의 무관한 세 인용도 원본 사실과
현재 단서 규칙으로 확인했을 때 보류됐다. 실제 Qdrant의 최종 성능은 아래 실행
결과로 판단한다. 로컬 코드 테스트만으로 실제 코퍼스 성능을 주장하지 않는다.

제공된 소스 기준 로컬 Context 테스트는 `420 passed, 1 skipped`였다.
로컬 환경의 모델 클래스를 호출 불가능한 대체 클래스로 두었고,
FastAPI가 없는 환경이라 API 테스트 파일은 제외했다. 아래 Docker 스크립트는
프로젝트 의존성이 설치된 환경에서 API 및 재질문·설명 테스트도 실행한다.

## PowerShell 실행

프로젝트 루트에서 새 ZIP을 푼 뒤 한 번 실행한다. 빌드와 테스트가 통과하면
backend를 중지하고 30문항을 비교하며, 진단 실패 시에도 backend를 다시 시작한다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Candidate_Cut_Fix_20261003*.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 프로젝트 폴더 또는 Downloads에 저장해 주세요.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
Unblock-File -LiteralPath .\experiments\namuwiki\run_context_candidate_fix.ps1
.\experiments\namuwiki\run_context_candidate_fix.ps1
```

스크립트 실행 정책에 의해 차단될 때에는 현재 터미널에만 다음 설정을 적용하고
마지막 실행 명령을 다시 실행한다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\experiments\namuwiki\run_context_candidate_fix.ps1
```

기본 입력은 다음 파일들이다. 이전 결과와 캐시는 이동하거나 삭제하지 않는다.

```text
artifacts/context_eval_step10_v2_blind_new/blind_report.json
artifacts/context_eval_step10_v2_blind_new/blind_checkpoint.json
artifacts/context_eval_step10_v2_blind_new/blind_analysis_cache.json
artifacts/context_eval_step10_after_fix/diagnostic_report.json
data/context/eval_blind_step10.csv
```

이 진단은 기존 LLM 분석의 Context 단서만 현재 규칙으로 재구성한다. 다른 분석
필드, 코퍼스, 기존 가중치 및 검색 폭을 고정하고 리랭킹을 끈다. 새 LLM 분석을
요청하지 않는다. 이전 단서 수정과 이번 후보 절단 수정의 효과도 구분해 기록한다.
이미 열람한 30문항의 수정 진단이므로 새 독립 평가의 성공으로 표시하지 않는다.

모델 로딩 뒤 질의별 Candidate@30·Top10 순위가 출력된다. 재실행 시 같은
코드·입력·코퍼스의 완료된 질의는 체크포인트를 재사용한다. 진단 코드나 입력이
변경되면 중단하며, 이전 결과를 덮어쓰는 대신 다른 출력 경로를 지정해야 한다.

## 확인할 결과

다음 파일 하나로 정답 순위, 단서 원문, 최종 후보에 들어오기 전의 Context 기여와
메타데이터 보정, 표시한 사실 및 출처를 검토할 수 있다.

```text
artifacts/context_eval_step10_candidate_fix/diagnostic_report.json
```

`groups.context_positive`에서 새 정답 진입과 Hit@10·MRR@10 변화를 확인한다.
`context_control` 및 `regression_all`에서는 손실과 예기치 않은 단서를 확인한다.
`target_scores_before_cut`은 최종 탈락한 정답의 Context 기여도도 보관한다.
`target_had_context_hit`은 최종 후보 안에 살아남은 Context 결과를 뜻하므로,
그 값만 보고 원시 Context 조회가 정답을 못 찾았다고 판단하지 않는다.
인용 라벨 일치는 별도로 확인하고 비정답 곡의 인용도 원문과 대조한다.

전체 진단 명령의 종료 코드 0은 일관된 조건의 실행 완료를 뜻한다.
성능 개선·배포 승인·새 블라인드 통과를 뜻하지 않는다. 이 수정 결과를 확인한 뒤
기존 dev/test를 새 코드로 다시 측정하고 별도의 독립 질의셋으로 검증한다.
새 질의와 검토 노트는 기존 실패 사례를 그대로 변형해 성공률을 높이지 않도록
별도로 작성하고, 평가 전 코드와 설정을 고정한다.

평가 질의, 곡 ID·제목, 사실 문장, 캐시, 보고서는 로컬 `data/`·`artifacts/`에만
보관한다. ZIP에는 소스·테스트·문서만 포함한다. 공개 레포에는 이 파일들을
포괄하는 `git add .` 대신 변경할 코드 경로만 명시해 추가한다.
