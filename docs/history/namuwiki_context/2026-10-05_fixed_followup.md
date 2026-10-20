# Context 고정 설정 후속 검증

제공된 context_fixed_validation_20261005_feedback.zip을 기준으로 검색과 근거 표시를 보완했다. 이전 dev/test는 공개된 회귀 평가이고, 아직 검색하지 않은 새 60문항은 별도 독립 평가로 유지한다. 질의와 리뷰 노트는 앞서 작성된 파일을 그대로 사용하므로 사용자가 새로 작성할 필요는 없다.

**확인된 이전 결과**

| 군 | n | Candidate@30 OFF→ON | Hit@10 OFF→ON | 해석 |
|---|---:|---:|---:|---|
| dev Context 양성 | 18 | 7→16 | 6→14 | Context 개선이 관찰됐다 |
| test Context 양성 | 8 | 8→8 | 6→8 | Top-10 개선이 관찰됐다 |
| dev 기존 검색 | 53 | 44→42 | 34→34 | 정답 후보 2건이 탈락했다 |
| test 기존 검색 | 23 | 16→16 | 13→12 | Top-10 정답 1건이 탈락했다 |

c607·c707은 정답 후보가 빠졌고, c703은 정답이 8위에서 12위로 밀렸다. q203은 양쪽 모두 후보에 없었다. m103에는 작품 사용을 입증하지 않는 OST 인기 조사 문장이 표시됐다. 실제 이미지 검색은 dev 9문항·test 3문항 모두 양쪽에서 embed/query가 실행됐다. 이 결과로 최종 적용을 선언할 수는 없다.

**검색 변경**

상세한 장면·줄거리·시기 묘사로 찾는 미확인 작품은 실제 Dense 사실에서 사용 관계와 해당 단서의 구체적인 내용이 함께 확인될 때 Context로 승격한다. OST라는 단어, 곡 프로필의 Sparse 일치, 별개의 제작 일화만으로 상세한 작품 묘사를 뒷받침하지 않는다. 사실이 여러 문장에 흩어져 있거나 부정된 사용 관계이면 승격과 근거 표시를 보류한다.

이 규칙은 정확도를 우선한다. 자료가 장면 정보를 생략하거나 같은 사건을 다른 표현으로 기술하면 유용한 Context 후보도 제외할 수 있다. 새 독립 평가에서 이 손실도 측정해야 한다. 광범위한 OST 목록 질의, 이름이 주어진 작품, 제작·공연·뮤직비디오 같은 다른 관계에서는 Sparse 후보를 계속 사용할 수 있다.

작품명이 사용자 원문에 있으면 원래 문장 조회에 더해 작품명·관계만으로 조회한다. 알려진 작품명 등가 표현만 사용하며, 줄거리로 작품명이나 정답 곡을 추측하지 않는다. 추가 Dense 조회는 해당 작품명이 사실 문장에 포함된 포인트 ID로 범위를 제한한다. 한국어 조사·띄어쓰기를 정규화하고, 포인트 목록은 활성 Dense 세대별로 한 번 읽어 캐시한다. 벡터 재생성이나 Qdrant 재적재는 필요하지 않다.

가중치 0.5, Dense 사실 검색 폭 100, Sparse 프로필 검색 폭 100, 후보 30, 화면 10, 작품 단서 배수 2.0, 리랭킹 OFF를 유지한다. 검색 폭은 조회 한 번의 제한이다. 작품명 단서는 조회가 최대 두 번이므로 총 계산량이 이전과 같다는 뜻은 아니다. 새 보고서에서 검색 구간 지연도 확인한다. 여러 조회·단서는 곡당 가장 강한 순위만 선택하며, 외부 RRF에는 Context 한 경로만 기여한다. 별도 정답 슬롯이나 점수 보너스를 추가하지 않는다.

정식 메타데이터 조회, 최종 후보 절단 순서, 제외 곡의 후보 보충은 기존 흐름을 유지한다. 인기 조사·차트 순위는 그 곡이 사용자가 기억한 작품에 쓰였다는 근거로 표시하지 않는다. 라벨 외 표시 근거는 자동으로 오답이라 단정하지 않고 검토 목록에 남긴다.

**이번 환경에서 확인한 범위**

| 항목 | 결과 | 범위 |
|---|---|---|
| 테스트 | 695 passed, 32 subtests passed | 검색·API·후보 절단·근거·평가 도구; 실제 모델 추론은 미실행 |
| Qdrant 통합 테스트 | 통과 | 실제 Qdrant, 합성 4차원 벡터; 필터·Sparse 유지·세대 갱신 |
| 이전 분석 보존 | 106개 검증 | dev 74개, test 32개; 캐시·보고서·평가 원본 해시 확인 |
| 실제 사실 검사 | 3,995개 확인 | c607/c707/c703/m103의 미지원 작품 단서 승격 차단; m103 조사 근거 거절 |
| q203 작품명 필터 | 사실 5개·곡 3개 | 이 중 정답 곡 사실 3개 포함; 최종 순위 측정이 아니다 |
| PowerShell | 문법 검사 통과 | Windows/Docker 실행은 사용자 환경에서 수행 |

테스트 환경은 Python 3.12와 Qdrant client 1.19.1이며, 모델 호출을 금지한 import용 대체 모듈을 테스트 환경에만 두었다. 대체 모듈이나 가상환경은 전달 ZIP에 포함하지 않는다. 합성 임베딩 테스트의 통과는 실제 KoE5 순위 개선을 입증하지 않는다. 데스크탑의 Python 3.11 Docker에서는 실제 설치된 모델과 기존 데이터로 전체 평가를 실행한다. 테스트 수는 사용자의 추가 테스트에 따라 달라질 수 있다.

**적용과 한 번 실행**

변경 ZIP을 다운로드하고 VS Code에서 새 Org 프로젝트 루트의 PowerShell을 사용한다. 아래 명령은 프로젝트 폴더와 Downloads에서 최신 ZIP을 찾아 적용한다.

~~~powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Fixed_Followup_20261005*.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 먼저 다운로드해 주세요.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
.\experiments\namuwiki\run_context_fixed_followup.ps1
~~~

실행 정책으로 ps1 실행이 차단될 때는 해당 스크립트에 대해 Unblock-File을 실행한다. 현재 터미널에서만 설정이 필요한 경우 기존에 사용하던 Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned를 적용한다.

스크립트는 다음 순서로 진행한다.

1. Docker 이미지 빌드와 전체 테스트.
2. 이전 feedback ZIP에서 dev/test의 분석 106개를 그대로 읽어 수정 효과 진단. QueryAnalyzer 호출 없이 OFF/ON을 비교한다.
3. OFF 결과가 이전 기준선과 달라졌는지 검사. 코드·코퍼스·입력·BM25 내용이 바뀐 재개는 거절한다.
4. 진단의 회귀·경로 검사를 통과하면 새 캐시로 dev/test 106문항 분석·검색.
5. 새 dev/test도 통과하면 독립 60문항 분석·검색, 이어서 사전 지정 10문항을 한 번 더 분석. 재분석 10건은 독립 60문항의 분모에 더하지 않는다.
6. 결과 ZIP 생성과 backend 재시작. 평가 실패도 결과에 남긴다.

새 평가 루트는 artifacts/context_fixed_followup_20261005다. 원래 artifacts/context_fixed_validation_20261005 기록과 캐시는 덮어쓰지 않는다. 처음 실행하는 새 루트의 fresh 캐시가 새 분석을 보장한다. 같은 명령의 재개는 해당 실행에서 이미 완료된 분석과 측정만 재사용한다.

기본 이전 입력은 artifacts/context_fixed_validation_20261005_feedback.zip이다. 다른 곳에 보관했다면 아래처럼 지정한다.

~~~powershell
.\experiments\namuwiki\run_context_fixed_followup.ps1 -PriorFeedback "$env:USERPROFILE\Downloads\context_fixed_validation_20261005_feedback.zip"
~~~

진단만 실행하려면 -Mode Diagnostic, 진단을 별도로 완료한 후 새 평가만 실행하려면 -Mode Fresh를 사용한다. 코드를 수정한 후에는 -OutputDir artifacts/context_fixed_followup_20261005_retry2처럼 새 루트를 사용한다. 동일 코드·데이터에서 API 장애만 복구했다면 같은 명령으로 재개한다. 중단된 평가를 통과시키기 위해 가중치나 라벨을 바꾸지 않는다.

**결과 확인과 전달**

~~~powershell
Get-Content .\artifacts\context_fixed_followup_20261005\diagnostic\diagnostic_report.md -Encoding UTF8
if (Test-Path .\artifacts\context_fixed_followup_20261005\fresh\validation_report.md) {
    Get-Content .\artifacts\context_fixed_followup_20261005\fresh\validation_report.md -Encoding UTF8
}
~~~

최종 전달 파일은 artifacts/context_fixed_followup_20261005_feedback.zip 하나다. 콘솔 전체를 복사할 필요가 없다. diagnostic에는 이전 ON→수정 ON의 질의별 순위와 원시 Context 순위가, fresh에는 새 분석의 OFF/ON 지표·근거·실제 이미지/오디오 실행·재분석 결과가 들어간다.

Exit 4는 평가 기준 미충족, Exit 2는 분석/API/입력 등의 실행 오류다. 이 경우에도 ZIP을 보내면 실패 지점부터 확인할 수 있다. 테스트나 파일 사전 검사에서 먼저 중단됐을 때는 평가 ZIP이 아직 생기지 않을 수 있다.

Candidate@30은 정답이 후보에 들어온 비율, Hit@10은 화면 10개에 들어온 비율, MRR@10은 정답 순위의 역수를 평균한 값이다. 양성 Context 문항과 무관한 대조 문항을 분리하고, 각 질의의 OFF/ON에 동일한 분석을 사용한다. 새 60문항은 실제 사용자 무작위 표본이 아닌 코퍼스 근거로 작성한 테스트다. 통과는 관찰한 시험 범위의 근거이며 모든 질문 형태의 성공이나 최적 화면 개수를 입증하지 않는다.

전달 ZIP은 기존 코드 네 파일과 새 테스트·도구·안내만 담는다. 평가 CSV, 리뷰 노트, raw 데이터, 임베딩, Qdrant 데이터, 모델 대체 모듈은 들어 있지 않다. 평가 결과 ZIP과 private CSV는 공개 레포에 올리지 않는다. 운영 .env의 최종 가중치 적용은 이번 새 평가 결과를 확인한 뒤 판단한다.
