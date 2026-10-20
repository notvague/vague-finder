# Context 10단계: 첫 블라인드 실패 이후 원인 분리

## 확인된 것

- 원래 블라인드 30문항은 `not_recommended`로 실패했다. 이 결과는 보존한다.
- 수정된 Context 규칙으로 여섯 실패 질의의 정답은 Dense 곡 순위 1위,
  Sparse 프로필 순위 1~4위였고, Dense의 선택된 `record_id`도 라벨과 일치했다.
- 이는 원시 검색의 재현율을 확인한 것이며, 전체 검색의 Candidate@30,
  Hit@10, 인용 사실의 적합도를 증명하지는 않는다.

## 이 변경의 범위

- `analysis_cache.analyzer_fingerprint()`에 `context_query.py`를 포함한다.
  이전에는 해당 규칙을 바꿔도 이전 분석 캐시가 현재 분석으로 취급될 수 있었다.
- 진단 스크립트는 **이미 공개된 같은 30문항**에서 이전 실패 보고서,
  기존 QueryAnalyzer 출력, 같은 3016곡 코퍼스를 고정한다. 저장된 분석의
  Context 단서만 현재 규칙으로 다시 계산한다. 동일한 분석으로 Context OFF와
  기존에 선택했던 설정의 ON을 각각 검색하고 리랭킹은 끈다.
- 로그에는 질의별 단서 수, Candidate@30 및 Hit@10 순위, 정답 근거 라벨 일치를
  출력한다. 전체 근거 문장과 출처는 `artifacts/`의 로컬 보고서에만 보관한다.
- 기존 OFF@30 순서가 원래 실패 실행과 달라지면 원인 비교를 중단한다.
  Qdrant 세대, 기존 평가 소스, 구 코드 지문, 원래 캐시 해시도 검사한다.

## PowerShell 실행

이전 코드 패치 `Vague-Finder_Context_Step10_Blind_Fix_20261003.zip`이 적용된
프로젝트에서 실행한다. 새 ZIP은 덮어쓴 다음 빌드하고 테스트한다.

```powershell
Expand-Archive -LiteralPath .\Vague-Finder_Context_Step10_After_Fix_Diagnostic.zip -DestinationPath . -Force
docker compose build backend
if ($LASTEXITCODE -ne 0) { throw 'backend 빌드 실패' }
docker compose run --rm --no-deps -T -e SEARCH_REFERENCE_YEAR=2026 backend python -m pytest -q tests/context/test_context_analysis_cache_fingerprint.py tests/context/test_context_external_event_variants.py
if ($LASTEXITCODE -ne 0) { throw '검증 실패' }
docker compose stop backend
if ($LASTEXITCODE -ne 0) { throw 'backend 중지 실패' }
try {
    docker compose run --rm --no-deps -T -e SEARCH_REFERENCE_YEAR=2026 backend python -u -m experiments.namuwiki.diagnose_context_step10_after_fix --all
    if ($LASTEXITCODE -ne 0) { throw '수정 후 실검색 진단 실패' }
} finally {
    docker compose up -d backend
}
```

기본값은 이전 결과를 `artifacts/context_eval_step10_v2_blind_new/`에서 읽고,
보고서를 `artifacts/context_eval_step10_after_fix/diagnostic_report.json`에
저장한다. 처음 여섯 문항만 빠르게 확인하려면 마지막 명령에서 `--all`을
생략한다. 스크립트는 완료된 질의를 체크포인트로 보관하므로 나중에 `--all`을
실행하면 나머지만 측정한다.

진단 보고서만 전송하려면 다음 파일을 보내면 된다.

```text
artifacts/context_eval_step10_after_fix/diagnostic_report.json
```

## 해석과 다음 평가

이 보고서는 **이미 열람한 질의셋의 수정 후 회귀 진단**이다. 결과가 좋아져도
이를 새 블라인드 검증 또는 최종 설정 승인으로 부르지 않는다. Context 질의의
Candidate@30과 Hit@10 변화 및 가사·분위기·이미지 질의의 손실을 함께
검토한다. 근거 표시의 라벨 일치와 비정답에 표시된 사실은 사람이 원문과
대조한다. 이후 새 코드에 맞는 분석 캐시로 dev/test를 다시 측정하고,
별도의 새로운 독립 질의셋을 준비해 한 번만 평가해야 한다. 기존 Step 10
`blind_checkpoint.json`/`blind_report.json`은 덮어쓰지 않는다.
