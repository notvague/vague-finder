# Step 10: 최초 블라인드 실패 이후 진단

첫 30문항은 모두 실행되었고 `blind_report.json`에 `not_recommended`가 기록되었다.
종료 코드 4는 평가기의 품질 게이트가 실패했다는 뜻이다. 사용하던 backend는
PowerShell 스크립트의 `finally` 절에서 재시작된다. 기존 평가 폴더와 CSV는
삭제하거나 덮어쓰지 않는다.

로그에는 Context 단서가 없는 양성 문항 12개와, OFF@30에 비해 Context Hit@10의
순개선이 없다는 위반이 기록되었다. 상위 10위 안의 위치가 개선된 몇 문항은
있지만 Hit@10이 개선되었다고 볼 수는 없다. 전체 그룹의 정확한 분모와
Candidate@30 수치는 `blind_report.json`으로 확인한다.

## 이 패치의 범위

- 차트 순위, 타이틀 변경, 공연, 방송 장면 사용, 이름이 명시된 코러스, 특정
  녹음 일화처럼 원문의 **외부 사건**을 검색 단서로 인식한다.
- 가사 인용, 커버 이미지와 소리의 비유에는 Context 단서를 만들지 않는다.
  특정 발음과 시간대별 소리의 경우에도 원문에서 `여담`이나 `일화`라고
  기억했을 때만 Context로 보조 검색한다.
- 사실을 표시할 때는 곡의 ID, 안전한 출처 URL, 사실 안의 사건과 고유 세부사항을
  계속 검사한다. 공식 곡/가수명이 사실 문장에 다시 나오지 않아도 검증을
  수행한다. 부정문은 표시를 보류한다.

이 패치는 질의 분석과 근거의 일부 누락을 고친 것이다. 새로운 Qdrant 점수나
전체 검색 성능을 아직 측정한 결과가 아니다. 최초 블라인드 CSV를 보고
코드를 고쳤으므로 **동일한 CSV 재실행은 개발 진단**이며 독립적인 블라인드
성능 증거가 아니다. 처음의 실패 판정은 그대로 보고해야 한다.

## VS Code PowerShell에서 확인

프로젝트 루트에 ZIP을 푼 뒤, 최초 결과를 읽기 전용으로 요약한다.

```powershell
docker compose build backend
docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.inspect_step10_blind_report artifacts/context_eval_step10_v2_blind_new
docker compose run --rm --no-deps -T backend python -m pytest -q tests/context/test_context_external_event_variants.py tests/context/test_context_query_analyzer.py tests/context/test_context_relevance_step10_v2.py tests/context/test_context_evidence_step7.py
git diff --check
git status --short
```

기존 스크립트는 `context_query.py`와 `context_evidence.py`의 해시를 잠금에
포함한다. 따라서 코드 수정 뒤 기존 결과 폴더에 `-Mode Blind`만 실행하면
소스 지문이 바뀌어 실패한다. 동일한 30문항에서 수정 효과를 **개발용**으로
보려면 새 출력 폴더에서 `-Mode All`로 dev/test를 다시 잠근 다음 Blind를 실행한다.
아래의 첫 명령은 준비·개발 설정 탐색·기존 시험까지 다시 실행하므로 오래 걸린다.

```powershell
.\experiments\namuwiki\run_context_step10.ps1 -Mode All -OutputDir artifacts/context_eval_step10_after_failure_diagnostic
.\experiments\namuwiki\run_context_step10.ps1 -Mode Blind -OutputDir artifacts/context_eval_step10_after_failure_diagnostic
```

두 번째 명령이 또 코드 4로 끝나더라도 새 보고서를 읽고 원인을 확인한다.
`not_recommended`이면 제품의 Context 가중치를 최종 확정하거나 성능 개선을
주장하지 않는다. 어떤 설정을 새로 택할지는 새 개발 자료에서 결정한다.

새로운 최종 블라인드 검증은 이 수정 코드와 선택 설정을 **동결한 뒤**,
이전 106문항과 최초 블라인드 30문항의 정답 곡을 재사용하지 않는 별도의
검토된 질의셋을 실행해야 한다. 실제 음악·표지 판단은 그 매체를 확인한
라벨에 한정한다. 기존 평가의 `regression_image`가 텍스트상의 장면 묘사만
기반으로 만들어졌다면 실제 앨범아트 품질을 입증한 것으로 쓰지 않는다.
