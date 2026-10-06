# Context 평가 로그 첨부 복구

## 현재 기록에서 확인된 진행 상황

- dev 74문항: `passed`, 위반 항목 없음.
- test 32문항: `passed`, 위반 항목 없음.
- 이미지 경로가 필수인 dev 9문항, test 3문항은 Context OFF/ON 양쪽에서 실제 경로 실행이 확인됨.
- 새 독립 질의 60문항: `fresh001`부터 `fresh021`까지 분석 저장. `fresh022`에서 대체 분석(fallback)이 반환되어 중단.
- 독립 질의의 검색 성능 측정과 마지막 반복 분석 검사는 아직 완료되지 않았음.
- 이전 실패 문항 `fresh015`는 이번에 정상 분석됨. Context 단서와 실제 베이스 소리 단서가 보존되고, 이미지 질의는 비어 있음.

`failure_report.json`의 “retry prepare after service recovery”는 평가 도구의 일반 오류 문구다. 이 문구만으로 API 장애였다고 판단할 수 없다. 정확한 원인은 저장된 `console_run.log`의 QueryAnalyzer 오류에서 확인해야 한다.

## 수정 내용

기존 로그 첨부 도구는 새 ZIP을 만든 뒤 `os.replace()`로 원래 피드백 ZIP을 교체했다. 사용자의 Windows 실행에서 이 교체 작업이 `WinError 5`로 실패했다. 어떤 프로세스 또는 권한 상태가 교체를 막았는지는 현재 자료로 확정할 수 없다.

이번에는 원래 ZIP을 읽기만 하고, 고유한 이름의 `*_feedback_with_log_*.zip`을 새로 만든다. 원래 ZIP과 이전에 만든 로그 포함 ZIP을 덮어쓰지 않는다. 원래 보고서와 캐시의 바이트를 보존하며, 원래 ZIP의 SHA256과 저장된 자식 프로세스 종료 코드를 함께 기록한다. 종료 코드가 로그에 없으면 추정하지 않고 `null`로 기록한다.

`-PackageOnly`는 이미 저장된 로그와 ZIP만 묶는다. Docker, QueryAnalyzer, Qdrant, 평가를 실행하지 않는다. 성공 시 종료 코드 0은 **로그 묶기 성공**을 뜻하며, 평가의 실패 또는 중단 판정은 그대로 유지된다.

향후 실행의 로그는 기존 파일에 이어 쓰므로 이전 실패 로그를 보존한다. 실행 실패 메시지는 로그 첨부 성공을 미리 주장하지 않고, 실제 출력된 `Send this file:` 경로를 안내한다.

## 지금 실행할 명령

수정 ZIP을 프로젝트 폴더 또는 Downloads에 저장한 뒤, 프로젝트 루트의 PowerShell에서 실행한다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Feedback_Capture_Fix_20261005.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 프로젝트 폴더 또는 Downloads에 저장해 주세요.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
.\experiments\namuwiki\run_context_cover_dance_fix.ps1 -PackageOnly
```

콘솔에 출력되는 `Send this file:`의 **새 ZIP**을 보내면 된다. 파일 이름 끝의 고유 문자는 실행마다 달라진다. 이 작업을 위해 백엔드를 중지하거나 이미지를 다시 빌드할 필요가 없다.

묶기가 다시 실패하면 다음 파일만 보내면 된다.

```text
artifacts/context_fixed_followup_20261005_cover_dance_fix/console_run.log
```

## 선택: 수정 도구 테스트

아래 검사는 사용자 환경에 pytest가 설치되어 있을 때 사용할 수 있다. 실제 평가나 모델 API를 호출하지 않는다.

```powershell
.\venv\Scripts\python.exe -X utf8 -m pytest -q tests/context/test_context_cover_dance_capture.py
git diff --check
```

## 평가 기록의 보존

수정 대상은 로그 캡처 Python, 실행 PowerShell, 해당 테스트뿐이다. 검색 코드, QueryAnalyzer, 평가 코드, 가중치, 라벨, 코퍼스를 변경하지 않는다. 이 세 파일은 전달받은 등록 기록의 `code_sha256` 대상에도 포함되지 않는다. 기존 실행 폴더 및 캐시는 지우지 않는다.

로그 첨부 문제만 고쳐서는 `fresh022` 분석 실패가 해결되지 않는다. 콘솔 로그를 확인한 뒤, 일시적 서비스 문제라면 기존 실행을 재개하고, 분석 코드 수정이 필요하면 새 등록과 새 분석으로 평가해야 한다. 결과를 확인하기 전에 대체 분석 허용, 라벨 변경 또는 가중치 조정으로 통과시키지 않는다.

이 패키지는 평가 CSV나 사용자 피드백 로그를 포함하지 않는다. 사용자 환경에서 생성하는 피드백 ZIP은 기존처럼 비공개 평가 자료로 취급한다.
