# Context 단서·근거 세부 조건 수정 및 최종 실행

이 패키지는 이미 적용한 Cover Meaning Fix 위에 덮어쓰는 변경 파일입니다.
이전 평가 결과, 정답 CSV, 리뷰 기록, 가중치 및 통과 조건을 수정하지 않습니다.

## 받은 결과의 정확한 의미

받은 두 ZIP은 CRC와 모든 내보낸 파일의 SHA256을 확인했으며 공통 내용이 같습니다.
이전 코드의 Docker 테스트는 3,025개 및 subtest 32개가 통과했습니다.
dev 74개, test 32개, 후속 질의 60개까지 검색 측정이 모두 끝났습니다.
최종 상태는 `blocked`이며 이번 중단은 분석 서버 장애나 ZIP 저장 실패가 아닙니다.

| 남은 항목 | 확인한 원인 | 이번 변경 |
|---|---|---|
| fresh033의 Context 경로 미실행 | 광고에서 '가사를 바꿔 부른'을 '개사한'과 같은 외부 사건으로 인식하지 못함 | 광고·응원가·로고송의 완료된 가사 변경 사건을 인식 |
| fresh014의 추가 근거 | 삭제됐다는 문장을 판권 만료의 근거로도 표시 | 사용자가 말한 판권·라이선스 만료 조건을 해당 사실 문장에서 확인 |
| fresh017의 추가 근거 | 같은 콘서트 이름을 발언 내용의 근거로 표시 | 기억한 발언의 내용 단서를 확인하고 공연장 이름만 겹치는 문장을 제외 |

근거 2건을 정답 라벨에 추가해 통과시키지 않았습니다. 해당 문장이 질의의 핵심 조건을 뒷받침하지 못하므로 근거 판정을 수정했습니다.
단순 삭제 사실과 가사 수정 사실도 일반 OST 사용의 증거로 바뀌지 않도록 별도 경계를 검사합니다.
Sparse 프로필만 있는 후보에는 사실 근거를 표시하지 않습니다.
같은 요청에서 검색된 다른 Dense 사실이 조건을 충족하면 그 사실을 선택할 수 있습니다.
결과의 `context_evidence` 필드와 출처·record_id 전달 형태는 유지합니다.

## 이전 코드에서 실제 측정된 지표

아래 값은 첨부된 원래 보고서에서 읽은 값입니다. 이번 수정 후 새 검색의 결과가 아닙니다.
Context 양성 질의만 계산하며 쉬운 음성 대조 문항은 효과 분모에서 제외했습니다.

| 질의군 | n | Candidate@30 OFF→ON | Hit@10 OFF→ON | MRR@10 OFF→ON |
|---|---:|---:|---:|---:|
| dev Context 양성 | 18 | 7/18→16/18 | 6/18→14/18 | 0.202→0.531 |
| test Context 양성 | 8 | 8/8→8/8 | 6/8→8/8 | 0.575→0.906 |
| 후속 Context 양성 | 36 | 26/36→31/36 | 23/36→29/36 | 0.325→0.580 |
| 후속 가사·오디오·이미지 회귀 | 18 | 18/18→18/18 | 18/18→18/18 | 0.950→0.950 |

후속 양성군의 Hit@10은 6건 개선, 손실 0건입니다. 표본 내 정확 대응 검정 p=0.03125,
질의 단위 부트스트랩의 Hit@10 차이 95% 구간은 약 +5.6~+30.6%p였습니다.
이는 사람이 자연스럽게 입력한 모든 질의의 성능 보장이 아니라, corpus 근거로 작성한 이 표본의 관측 결과입니다.
후속 이미지 6개·오디오 6개 모두 OFF/ON에서 실제 임베딩 및 인덱스 조회 실행이 확인됐습니다.
가사·분위기·이미지 회귀군의 점수와 Hit@10 손실도 별도로 확인합니다.

이번에 해당 결과를 보고 수정했으므로 같은 60문항의 다음 측정은 공개된 질의의 후속 회귀 검증입니다.
새로운 블라인드 평가라고 부르지 않습니다. 가중치를 이 문항들에 맞춰 올리지 않습니다.

## 여기서 완료한 검증과 남은 검증

- 새 사건·근거 경계 테스트 66개를 포함한 관련 테스트 2,678개가 통과했습니다.
- 사건 10종, 앨범 그림 3종, 실제 들리는 소리 3종, 절 순서·쉼표·문장·줄바꿈을 교차한 서로 다른 합성 질의 2,050개가 포함됩니다.
- 사용자 원문, 출처, 곡 ID, 숫자, 부정문, 잘못된 모델 정답, 미래 추천 요청, 가사·표지에 쓰인 문구를 검사합니다.
- 저장된 166개 분석을 재파싱한 결과 오류 0건이며 fresh033의 Context 단서만 0→1로 바뀝니다. 다른 분석 필드는 유지됩니다.
- 저장된 근거 14개를 저장 문장과 리뷰 메타데이터로 재검사하여 적절한 12개는 유지하고 문제의 2개는 차단했습니다.
- Python 3.11 문법 및 PowerShell 정적 문법 검사를 통과했습니다.

여기에는 사용자의 Qdrant와 원문 corpus가 모두 있지 않아 새 Gemini 분석·실제 검색·Windows PowerShell 실행은 수행하지 않았습니다.
로컬의 일부 프로젝트에는 평가 모듈이 없어 전체 Docker 테스트도 여기서 실행했다고 주장하지 않습니다.
전체 Docker 테스트와 새 검색 지표는 아래 실행으로 사용자 환경에서 확인합니다.
합성 질의 테스트 수를 검색 정확도 표본 수로 합산하지 않습니다.

## 프로젝트에 적용

VSCode PowerShell에서 `Vague-Finder-Org` 프로젝트 루트에 있습니다.
ZIP을 프로젝트 폴더 또는 Downloads에 저장한 뒤 실행합니다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Event_Detail_Fix_20261006.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 프로젝트 폴더 또는 Downloads에 저장해 주세요.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
```

이전 단계에서 설치한 `apply_validated_context_settings.py`, `verify_context_api_ready.py` 및 기존 평가 프로그램을 사용합니다.
나무위키 재수집·임베딩 재생성·Qdrant 재적재는 이번 수정에 필요하지 않습니다.

## 빠르게 눈으로 확인

```powershell
.\experiments\namuwiki\run_context_event_detail_fix.ps1 -CheckOnly
```

이 모드는 관련 테스트와 검색 엔진을 열지 않는 확인 프로그램을 실행합니다.
`completed_adaptation=True`, `literal_lyric=False`, `future_adaptation=False`를 확인할 수 있습니다.
`license_condition_present=True`, `only_removal_no_reason=False`,
`remembered_speech_present=True`, `only_shared_concert=False`, `sparse_only=False`가 출력됩니다.
이 결과는 파서·근거 선택의 검사이며 실제 곡의 최종 순위를 보장하지 않습니다.

선택적으로 이전 ZIP의 166개 분석을 실제 파서로 재검사합니다. ZIP과 캐시는 읽기만 합니다.

```powershell
$prior = Get-ChildItem -LiteralPath .\artifacts -File -Filter 'context_fixed_followup_20261006_cover_meaning_fix_feedback*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $prior) { throw '이전 Cover Meaning 평가 ZIP을 artifacts에서 찾지 못했습니다.' }
$priorPath = 'artifacts/' + $prior.Name
docker compose run --rm --no-deps -T -e PYTHONUTF8=1 -e SEARCH_REFERENCE_YEAR=2026 backend python -u -m experiments.namuwiki.inspect_context_event_detail_fix --prior-feedback $priorPath
if ($LASTEXITCODE -ne 0) { throw '저장 분석 재검사 실패' }
```

`Registered prior outputs reparsed: 166; changed=1` 및 `fresh033: context_clues 0->1`을 확인합니다.
이 재검사에도 Gemini와 Qdrant는 사용하지 않습니다.

## 새 평가부터 통과한 설정 적용과 API 확인까지

```powershell
.\experiments\namuwiki\run_context_event_detail_fix.ps1 -ApplyAfterPass
```

기본 저장 위치는 `artifacts/context_fixed_followup_20261006_event_detail_fix`입니다.
수정 전 평가 폴더는 보존하고 새 등록·캐시로 시작합니다. 과거 캐시를 이 폴더에 복사하지 않습니다.
동일 코드·설정·입력·corpus에서 중단됐다면 같은 명령으로 재개할 수 있습니다.
추가 코드 수정 후에는 새 `-OutputDir artifacts/다른이름`을 사용합니다.

실행 순서는 전체 관련 Docker 테스트, dev/test/후속 166문항 OFF/ON 평가,
10문항의 추가 재분석, 통과 조건 검사, 통과한 설정의 호스트 소스 반영,
백엔드 재빌드·워밍업 대기, 실제 HTTP Context/Image/Audio/거절 후보 재충원 확인입니다.
로컬 Qdrant를 사용하는 실제 평가 동안은 백엔드를 중지하고 끝나면 다시 시작합니다.
가중치 0.5·Dense 사실 100·Sparse 100·후보 30·결과 10·작품 단서 배수 2·기준 연도 2026·리랭킹 OFF를 유지합니다.
해결하지 않은 근거 검토나 회귀 손실이 남으면 설정 적용을 차단합니다. 보고서 값이나 정답을 바꾸어 통과시키지 않습니다.

## 최종 성공 여부 확인

```powershell
$out = '.\artifacts\context_fixed_followup_20261006_event_detail_fix'
$report = Get-Content "$out\fresh\validation_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$report | Select-Object status, completed_phases
$report.analyzer_repeat_check | Select-Object question_count, violations
$activation = Get-Content "$out\activation\activation_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$activation | Select-Object status, api_verified, source_changed
$api = Get-Content "$out\activation\api_smoke_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$api | Select-Object status
```

필요한 값은 `passed_observed_tests`, 재분석 10문항·violations 없음,
`api_verified=True`, `passed_api_smoke`입니다.
기존 설정으로도 Context 경로가 연결돼 있었지만, 이 마지막 값들이 있어야 이번에 측정한 설정과 실제 API까지 확인됐다고 말할 수 있습니다.
이 지표는 리랭킹 OFF 조건입니다. 프런트엔드에서 리랭킹을 켜는 요청의 순위·지연까지 같은 결과라고 주장하지 않습니다.

중단됐다면 콘솔 마지막의 정확한 `Send this file:` 경로에 있는 ZIP을 보내면 됩니다.
이번 출력은 원래 ZIP을 보존하고 콘솔을 포함한 별도 ZIP을 만듭니다.
모든 질의 형태를 유한한 시험으로 보장할 수는 없으며, 보고서는 실제 실행한 경로·문항·손실·근거와 제한을 함께 남깁니다.

## 공개 PR에 포함할 범위

변경은 source의 일반 표현·사건 조건 검사와 합성 테스트·실행 도구입니다.
실제 곡 목록, 정답 CSV, 리뷰 데이터, 분석 캐시, raw meta, 음원, embedding, 피드백 ZIP은 포함하지 않습니다.
`git diff --check` 후 파일을 명시하여 추가합니다. `git add .`로 private 산출물을 한꺼번에 추가하지 않습니다.

PR 설명에는 광고용 가사 변경의 평문 표현을 처리하고,
삭제 사유·발언 내용이 없는 문장을 근거로 표시하지 않게 했다는 결과를 적을 수 있습니다.
실제 검색 지표와 API 성공 여부는 새 평가 완료 후 해당 보고서 값으로 갱신합니다.
