# Context API 확인 마무리 — 2026-10-06

## 이번 피드백의 결론

첨부한 event_detail_fix 보고서에서 dev 74문항, test 32문항, 작성 질의셋 60문항과 새 QueryAnalyzer 재분석 10건이 모두 통과했습니다. Docker의 기존 전체 테스트도 **3290 passed, 32 subtests passed**였습니다. 평가 상태는 `passed_observed_tests`이며, 설정 적용 상태는 `settings_applied`입니다.

아직 실제 API 확인 상태는 `failed_api_smoke`, `api_verified=false`입니다. 이 ZIP은 그 API 확인 절차를 보완합니다. 검색 코드, 가중치, 정답, 사실 관련성 기준, 완료된 OFF/ON 평가를 변경하지 않습니다.

## 성능 변화

Context를 요구하는 질의만 아래 효과 분모에 포함했습니다. Context가 필요 없는 대조 질의는 제외했습니다. 두 조건은 같은 분석 결과와 같은 완성 코퍼스를 사용했고, 리랭킹을 껐습니다.

| 배경지식 질의 집합 | 문항 수 | Candidate@30 OFF→ON | Hit@10 OFF→ON | MRR@10 OFF→ON |
|---|---:|---:|---:|---:|
| dev | 18 | 7/18→16/18 | 6/18→14/18 | 0.202→0.531 |
| test | 8 | 8/8→8/8 | 6/8→8/8 | 0.575→0.906 |
| 작성한 새 질의셋의 배경지식 부분 | 36 | 26/36→31/36 | 23/36→29/36 | 0.325→0.580 |

- dev의 기존 검색 53문항은 Hit@10과 MRR@10이 유지됐고, Candidate@30은 44/53→45/53으로 개선됐습니다.
- test의 기존 검색 23문항, 작성 질의셋의 기존 검색 18문항에서는 보호한 지표의 하락이 없었습니다.
- 작성 질의셋에서 이미지 필수 6문항과 오디오 필수 6문항은 OFF/ON 모두 실제 임베딩·DB 조회가 확인됐습니다. 가중치가 양수라는 사실만으로 통과시키지 않았습니다.
- 표시한 근거는 dev 9건, test 5건, 작성 질의셋 14건 모두 등록된 사실 라벨과 일치했고, 미검토 항목은 0건입니다. 재분석에서도 위반 및 미검토 항목이 0건입니다.

새 질의셋도 이번 수정 과정에서 결과를 확인하고 다시 검사한 집합입니다. 이번 재실행은 **후속 회귀 검증**이며, 새 블라인드 평가나 실제 사용자 표본으로 표현하면 안 됩니다. 36개 배경지식 질의 중 ON에서도 7개는 Top-10에 정답이 없었습니다. 모든 자연어 표현을 완벽히 처리했다는 주장은 이 결과로 입증되지 않습니다. Top-10은 고정한 화면 크기이고, 최적 크기로 증명한 값은 아닙니다.

## API 검사가 중단된 이유와 보완

직전 스크립트는 이미지·오디오 필수 경로를 확인할 때 **최종 Top-10에 양수 기여가 있어야 한다**고 판정했습니다. 첫 오디오 검사는 가사와 신스 소리를 함께 묻는 기존 dev 혼합 질의였습니다. 이 질의에서 오디오 기여가 최종 결과에 없다고 해서 오디오 경로가 실행되지 않았다고 결론 내릴 수 없습니다. 실제 실행의 다른 Context 혼합 질의에서는 오디오 양수 기여가 이미 관찰됐습니다.

기존 실패 보고서는 중단된 혼합 질의의 응답과 시간 기록을 저장하지 않았으므로, 그 요청의 오디오 실행 여부 자체는 아직 확인되지 않았습니다. 이를 임의로 성공 처리하지 않고 다음 증거를 요구합니다.

1. 원래 질의로 새 HTTP 검색을 수행합니다. 전용 영어 질의나 가중치를 스크립트에서 강제로 주입하지 않습니다.
2. 응답의 `X-Request-Id`와 그 요청 이후 서버 시간 로그에 추가된 레코드를 대조합니다. ID, 원문 질의, HTTP 200이 모두 맞아야 합니다. 이전 로그나 같은 문장의 다른 요청은 사용할 수 없습니다.
3. 이미지·오디오 질의는 새 분석의 전용 질의와 양수 가중치, 실제 `path.image`/`path.audio`, 같은 작업의 `.embed`와 `.query` 구간, 오류 없음까지 확인합니다.
4. Top-10의 양수 기여 여부는 `required_path_in_top10`으로 별도 기록합니다. 실행 증거가 없으면 최종 기여가 있어도 실패합니다. 실행이 확인됐지만 Top-10에 남지 않은 혼합 질의는 실행 검사를 통과할 수 있습니다.
   전체 API 검사에서는 이미지·오디오 각각 최소 한 요청에서 **실행 증거와 Top-10 양수 기여를 함께** 확인해야 합니다. 모든 요청에서 해당 검색 결과가 사라지는 경우에는 최종 통과하지 않습니다.
5. 기존 혼합 오디오 질의를 유지하고, 등록된 이미지 필수 6개와 오디오 필수 6개를 전부 추가합니다. 문항 선택에 정답 순위·성공률·새 API 결과를 사용하지 않습니다.
6. Context 두 질의는 양수 Context 기여와 알려진 정답의 Candidate@30 진입을 계속 요구합니다. 중복 후보, 정식 제목, fallback, 경로 오류, 리랭킹 OFF, 불필요한 Context 단서·근거, 이전 5곡 거절 후 후보 보충 검사도 유지합니다.
7. 한 질의가 실패해도 나머지를 검사하고, 실제 응답·분석·요청 ID·시간 구간·오류를 비공개 보고서에 저장합니다. 한 건이라도 실패하면 `api_verified`는 false입니다.

## 적용 및 재개

새 ZIP을 프로젝트 루트에 풀고 PowerShell에서 실행합니다.

```powershell
.\experiments\namuwiki\run_context_api_finish.ps1
```

기본값은 기존에 통과한 출력 폴더입니다.

```text
artifacts/context_fixed_followup_20261006_event_detail_fix
```

기존 출력 폴더를 지우거나 새 평가 폴더로 옮기지 마세요. 완료된 166문항 평가를 다시 수행하지 않습니다. 이 실행은 이미지 빌드, 관련 테스트, 기존 완료 평가·소스 해시 검사, 설정의 동일 값 확인, 백엔드 재시작, 실제 HTTP 검사, 기존 형식의 피드백 묶음을 수행합니다. 별도 Qdrant 클라이언트를 열지 않습니다.

기본값과 다른 폴더에서 완료했다면 `-OutputDir`에 이미 통과한 출력 폴더의 상대 경로를 지정합니다. 경로는 기존 실행기와 같은 영문·숫자·밑줄·대시 형식을 사용합니다.

관련 테스트를 따로 실행하는 명령은 다음과 같습니다.

```powershell
docker compose build backend
docker compose run --rm --no-deps -T -e PYTHONUTF8=1 backend python -m pytest -q tests/context/test_context_validated_activation.py tests/context/test_context_api_execution_proof.py
git diff --check
```

로컬 검증은 **154 passed**입니다. 실제 HTTP 연결/UTF-8/요청 ID 검사는 임시 로컬 서버로 검증했고, 검색 단계의 성공·누락·오류·다른 요청·중첩 작업은 합성 fixture로 검증했습니다. 사용자 컴퓨터의 실제 Qdrant·Gemini를 여기서 실행한 결과로 해석하면 안 됩니다. 그것은 위 재개 명령이 확인합니다.

## 성공과 실패 확인

성공하면 실제 HTTP **18건**이 검사됩니다: 기존 5건 + 이미지 6건 + 오디오 6건 + 후보 거절 후 보충 1건. 새 API 보고서 schema는 `context_api_ready_smoke_v2`입니다.

```powershell
$root = '.\artifacts\context_fixed_followup_20261006_event_detail_fix'
$api = Get-Content "$root\activation\api_smoke_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$activation = Get-Content "$root\activation\activation_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$api | Select-Object schema, status
$activation | Select-Object status, api_verified
$api.rows | Select-Object query_id, status, required_path, required_path_in_top10, client_ms
$api.media_coverage
```

완료 기준은 다음 두 값입니다.

```text
api.status = passed_api_smoke
activation.api_verified = True
```

`audio_embed_query=verified, in_top10=False`는 오디오 임베딩과 조회는 확인됐지만 Top-10 기여가 남지 않았다는 뜻입니다. 둘을 구분하기 위한 의도한 출력입니다. 오디오 실행 없이 이 메시지가 출력되지 않도록 회귀 검사했습니다.

실패하면 `api.requests`에서 해당 요청의 분석과 응답, `api.failures`에서 실패 사유를 확인할 수 있습니다. `activation/history`에 이전 API 보고서를 보관하고, 원래 평가 ZIP은 그대로 두고 새 결합 ZIP을 만듭니다. 콘솔 마지막의 **Send this file** 경로의 ZIP을 보내 주세요. 파일 이름은 실행마다 고유합니다.

이 실행에서 `SEARCH_TIMING_LOG`를 변경하지 않습니다. 기존 등록 경로는 `artifacts/timing/context_review.jsonl`입니다. 실행 중인 서버가 새 시간 레코드를 기록하지 않으면 성공으로 처리하지 않고 명확하게 실패합니다.

## 공개 PR에서의 설명과 데이터

공개 변경물은 API 검증 스크립트, 실행기, 테스트, 이 문서입니다. 새 ZIP에는 실제 곡 목록·질의 원문·정답 CSV·평가 캐시·근거 라벨·로그·음원·임베딩·API 응답이 없습니다. 피드백 ZIP과 `artifacts`의 보고서/로그는 공개 커밋하지 마세요.

PR에서 설명할 핵심은 **혼합 질의의 Top-10 기여와 검색 실행을 구분하고, 같은 HTTP 요청의 임베딩 및 Qdrant 조회 증거로 실제 이미지·오디오 실행을 확인한다**는 것입니다. API 스모크와 검색 정확도 비교는 서로 다른 검사이며, 이번 수정으로 OFF/ON 성능 수치를 바꾸지 않았습니다. 최종 적용 확인은 `passed_api_smoke`와 `api_verified=true`가 확인된 뒤 기록합니다. 검증된 요청에서는 리랭킹을 껐으며, 평소 UI의 리랭킹 ON 경로 성능까지 이번 결과로 확대해서 설명하지 않습니다.
