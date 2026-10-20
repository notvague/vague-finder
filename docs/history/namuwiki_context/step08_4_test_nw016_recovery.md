# Context 8단계 test 평가 복구: nw016

## 중단 원인과 수정 범위

test 32문항의 다섯 번째 질의 `nw016`에서 Gemini는 `audio_english_query`를
비워 둔 정상 분석을 반환했습니다. 후처리가 `초기 기획 때 ... 둘이 부르는 안도
있었다`를 실제 음원의 듀엣으로 해석해서 오디오 프롬프트를 세 차례 요구했고,
평가기는 폴백 분석을 거부해 중단했습니다. 따라서 **현재 test 32문항 평가는
완료되지 않았습니다.** 앞서 완료된 dev 74문항 결과도 이전 분석기 지문입니다.

`extract_audio_evidence_text`는 기획 당시의 **실현되지 않은 공연/편곡 계획**만
청각 단서에서 뺍니다. 전체 질의는 Text/Context에 그대로 전달되므로 제작
비화 검색문이 보존됩니다. 문장 뒤에서 실제 녹음의 악기·보컬을 설명하면
그 부분은 CLAP 단서로 남습니다. 실제 듀엣 설명도 여전히 오디오 검색문을
요구합니다. 기존 visual guard와 `nw013` 장르 경계 수정이 포함된 전체 파일입니다.

## 로컬 적용과 재평가

프로젝트 루트 `Vague-Finder-Org`의 PowerShell에서 ZIP을 풀고 스크립트를
실행하세요. `data/`, Qdrant, 개인 평가 CSV 등은 ZIP에 포함되지 않습니다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Step8_Unrealized_Performance_v3.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw 'v3 ZIP을 프로젝트 폴더 또는 Downloads에 넣어 주세요.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
git diff --check
if ($LASTEXITCODE -ne 0) { throw '소스 줄바꿈이나 공백 오류가 있습니다.' }
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\experiments\namuwiki\run_context_step8_v3.ps1
```

스크립트는 빌드와 3개 테스트 파일의 회귀 검사, Qdrant 충돌 방지를 위한
backend 정지, `nw016` 실데이터 단일 평가, **dev 74문항과 test 32문항의
순차 평가**, 상태·분석기 지문·코퍼스 비교, backend 재시작까지 수행합니다.
기존 dev/test 출력과 분석 캐시는 `artifacts/context_eval_step8/backup_before_v3_*`
폴더에 보관합니다. 기존 캐시에는 이전 후처리 지문과 test 폴백이 들어 있어
새 코드로 그대로 재사용하면 안 됩니다. `SEARCH_REFERENCE_YEAR=2026`,
Context OFF/ON의 동일 분석, Candidate@30, Top-10, 리랭킹 OFF를 유지합니다.

마지막 `PASS`는 두 실행의 완료 상태와 측정 조건이 일치한다는 뜻입니다.
성능을 판정하려면 함께 출력되는 `context_all`, `regression_all`, 가사·음향·이미지
그룹의 `C@30`, `Top10`, `lost`, `issues`를 확인해야 합니다. 관련 곡이
30위에는 진입했어도 Top-10에 없을 수 있습니다. 폴백 또는 다른 질의의
모달리티 오류로 멈춘 경우에는 완료라고 보고하지 말고 해당 `query_id`와
오류 로그를 확인하세요. 이전 결과는 백업되며 backend는 `finally`에서
다시 켜집니다.

이 ZIP에는 비공개 평가 질의 CSV, 음원, 가사, 수집 데이터, 벡터 산출물을
담지 않았습니다. 공개 레포에는 변경한 소스·테스트·가이드만 올리세요.
