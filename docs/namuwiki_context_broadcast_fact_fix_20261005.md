# 방송명과 외부 사실 단서 경계 수정

## 이번 결과 해석

이전 실행은 dev 74문항과 test 32문항의 검사를 통과한 뒤, 60문항 확장 평가의
27번째 질의 분석에서 멈췄다. 26번째까지 저장된 분석은 검색 결과가 아니다.
해당 단계는 전체 분석을 준비한 다음 OFF/ON 검색을 실행하므로, 이 실행의
60문항 검색 성능이나 최종 적용 여부는 아직 판정할 수 없다.

반복된 `album-art visual clue requires image_english_query` 오류는 방송명의
‘스케치’와 별도 문장의 보컬 성별을 합쳐 이미지 단서로 판정한 코드 문제였다.
서버 장애가 확인된 것은 아니다. `analyzer fallback; retry prepare after service
recovery`는 원인을 구분하지 않는 기존 안내 문구다.

로그 포함 ZIP은 정상 생성됐다. 원래 보고서 ZIP과 로그 포함 ZIP의 기존 보고서
16개는 바이트 단위로 같다. 보고서의 평가 종료 코드 2와 바깥 PowerShell 실행의
종료 코드 1은 서로 다른 단계의 상태이며, 새 ZIP 생성 성공이 평가 통과를 뜻하지 않는다.

## 변경한 동작

1. `이름에서 부른/연주한/커버한`이라는 과거 공연 문맥의 이름을 암묵적 이미지
   판정에서만 제외한다. 방송명 목록이나 특정 가수·정답 곡을 하드코딩하지 않는다.
   원래 질의는 보존하고 실제 그림, 사진, 손글씨, 표지 설명은 계속 검사한다.
2. `뮤비`, 작사·작곡 비화, 프로듀서 데뷔, 코러스 녹음 참여, 공연 중 발언,
   방송사고, 광고·응원가 개사, 판권 만료와 같은 외부 사실 표현을 규칙으로 보완한다.
   모델이 `context_clues=[]`를 반환해도 원문의 관계가 분명하면 단서를 보존한다.
3. `그 뮤비의 노래가 뭐였지?` 같은 후속 질문은 새 사건으로 중복 추출하지 않는다.
   가사·곡명 안에 쓰인 사건 단어, 실제 표지의 글자, 앞으로 할 활동과 단순히
   들리는 코러스는 외부 사실로 추가하지 않는다. 실제 가사 수정 비화는 구별한다.
4. 모델이 보탠 작품명·정답·검색 단어를 원문으로 검증하는 기존 보호 규칙을 유지한다.

고정 검색 설정은 그대로다: Context 가중치 0.5, Dense 사실 100개, Sparse 곡
프로필 100개, 후보 30개, 화면 10개, 작품명 배수 2, 리랭킹 OFF, 기준 연도 2026.
QueryAnalyzer의 프롬프트, 기존 모달리티 가중치 정책, 평가 정답·리뷰 파일,
Context 인덱스와 평가 합격 기준은 수정하지 않았다. 임베딩 재생성은 필요하지 않다.

## 로컬에서 완료한 검증

- 새 경계 검사 93개와 기존 관련 검사 227개: 총 320개 통과.
- 프로그램 이름 5종 × 과거 공연 동사 4종, 따옴표·띄어쓰기·조사 변형,
  실제 그림과 독립 표지 단서, 소리·표지·외부 사실의 혼합을 검사했다.
- 외부 사실 유형, 후속 질문 중복, 가사·곡명 속 문구, 미래 활동,
  모델이 지어낸 작품·정답, 동기/비동기 Analyzer의 첫 시도 정상 수용을 검사했다.
- 이미 관찰한 분석 26개를 다시 파싱했다. 14개의 누락된 Context 단서가 보존됐고,
  파싱 오류가 없었다. 이미 관찰한 27번째 실패 질의의 이미지 오인도 재현 후 해소했다.
- 기존 dev/test 분석 106개를 다시 파싱해 오류 0건, Context 단서 유무 변경 0건,
  이미지·오디오 검색 문장 변경 0건을 확인했다. 이것은 캐시된 응답의 파싱 검사이며,
  새 모델 호출이나 실제 OFF/ON 검색 성능 재측정은 아니다.
- 눈으로 보는 합성 응답 검사 12종은 실제 `QueryAnalyzer._parse`를 사용한다.
  Gemini API와 Qdrant를 호출하지 않는다.

여기서는 팀원의 Windows Docker 환경, 원격 모델과 현재 Qdrant를 직접 실행하지
않았다. 전체 서비스·검색 회귀 검사는 아래 실행으로 확인해야 한다. 일부가 이미
관찰된 60문항은 통과해도 완전히 미노출된 블라인드 결과로 표현하지 않는다.

## 적용과 눈으로 확인하기

VSCode PowerShell에서 새 프로젝트 루트 `Vague-Finder-Org`를 연다. 다운로드한
ZIP을 찾고 압축을 풀어야 새 코드가 적용된다. ZIP에 사설 데이터나 평가 정답은 없다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Broadcast_Fact_Fix_20261005.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 찾지 못했습니다.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force

$out = 'artifacts/context_fixed_followup_20261005_broadcast_fact_fix'
.\experiments\namuwiki\run_context_broadcast_fact_fix.ps1 -OutputDir $out -CheckOnly
```

검사 종료 후 `12 parser cases` PASS를 확인한다. 대표 출력은 다음과 같다.

| 사례 | Context | Image | Audio |
|---|---:|---:|---:|
| 방송명 + 실제 피아노·남자 보컬 | True | False | True |
| 코러스 녹음 참여 여담 + 실제 코러스 | True | False | True |
| 뮤비의 촬영·등장 장면 | True | False | False |
| 광고 개사·게임 판권 만료 | True | False | False |
| 가사 속 ‘작사 비화’라는 단어 | False | False | False |
| 앞으로 방송에서 부를 곡 추천 | False | False | False |
| 실제 앨범 표지만 묘사 | False | True | False |
| 공연 사실 + 실제 표지 + 실제 코러스 | True | True | True |

마지막 누출 사례의 `blocked_as_expected`는 의도한 통과다. 오디오 검색 문장에
앨범 그림 묘사가 섞이면 계속 차단한다. 전체 테스트 수는 사설 평가셋 설치 여부에
따라 달라질 수 있다. 새 경계 검사 파일의 93개는 사설 데이터를 요구하지 않는다.

## 설정을 고정한 전체 재측정

위 검사가 통과하면 다음 명령을 실행한다. 이미 존재하는 과거 평가 폴더로 바꾸지 않는다.
새 코드의 등록 해시로 dev/test, 60문항, 재분석 반복 검사를 다시 수행한다.

```powershell
.\experiments\namuwiki\run_context_broadcast_fact_fix.ps1 -OutputDir $out
```

이 실행은 기존 전체 Context·API·설명 테스트를 수행하고, 로컬 Qdrant의 동시
접근을 막기 위해 backend를 중지한 다음 평가하며, 종료 시 backend를 다시 올린다.
분석·검색·회귀·근거 검사에 실패하면 기존 기준에 따라 멈춘다. 실패를 없애려고
설정이나 정답을 바꾸지 않는다. 같은 코드·입력·환경에서 중단된 이 새 폴더로
재실행하면 등록된 정상 캐시를 이어서 사용한다. 코드를 다시 수정한 뒤에는 또 다른
출력 폴더로 등록해야 한다. 과거 캐시를 새 폴더에 복사하지 않는다.

`-CheckOnly`는 원격 모델을 호출하지 않는다. 전체 재측정은 새로운 분석 API를
호출하므로 기존 분석 26개를 그대로 가져오는 방식보다 시간이 더 걸린다.

## 보내줄 결과 파일

실행 맨 마지막에 출력되는 **로그 포함 ZIP의 실제 경로**를 사용한다.

```text
Send this file: ..._feedback_with_log_....zip
```

새 ZIP에는 평가 보고서와 콘솔 로그가 함께 들어간다. 원래 ZIP은 보존된다.
로그 첨부만 다시 하고 싶다면 검색·분석 없이 아래 명령을 사용한다.

```powershell
.\experiments\namuwiki\run_context_broadcast_fact_fix.ps1 -OutputDir $out -PackageOnly
```

패키징에 실패하면 출력 폴더의 `console_run.log`를 보내면 된다. 결과 CSV, 분석 캐시,
리뷰·정답 파일, Qdrant 데이터와 피드백 ZIP은 사설 데이터이므로 공개 PR에는
넣지 않는다. 수정 코드와 일반 예시를 사용하는 테스트·가이드를 검토 대상으로 올린다.
