# 커버 경연·영상과 앨범 표지의 경계 수정

## 피드백 ZIP에서 확인한 결과

이전 Broadcast Fact 수정은 적용되어 있었다. 등록된 `modality_queries.py`,
`context_query.py`, `query_analyzer.py`의 해시가 전달한 코드와 일치했다.

| 구간 | 이번 업로드에서 확인한 상태 |
|---|---|
| 전체 Docker 테스트 | 932 passed, 32 subtests passed |
| dev 74문항 | passed, 위반 0건, 근거 수기 검토 대기 0건 |
| test 32문항 | passed, 위반 0건, 근거 수기 검토 대기 0건 |
| 필요한 실제 이미지 경로 | dev 9개, test 3개 모두 OFF/ON 실행 확인 |
| 확장 평가 60문항 | 앞선 30개의 분석 저장, 31번째 분석에서 중단 |

앞선 30개의 분석에는 모두 Context 단서가 있었다. 확장 평가는 60개의 분석을
준비한 다음 검색하므로, 이번 실행의 60문항 OFF/ON 검색 결과는 아직 없다.
실제 이미지 경로를 실행한 dev/test 결과는 확인되었지만 전체 최종 적용 판정은
확장 평가와 반복 검사까지 완료된 뒤에 해야 한다.

실패 질의는 ‘커버 서바이벌이라는 패러디 공모전을 열었던 노래’였다. 코드가
공연을 뜻하는 ‘커버’를 앨범 표지로 해석해 이미지 문장을 요구했다. 같은 이미지
검증 오류가 3회 반복되어 fallback이 반환됐고 평가가 종료 코드 2로 중단됐다.
`retry prepare after service recovery` 문구가 서버 장애를 증명하지는 않는다.

원래 ZIP의 보고서 16개와 로그 포함 ZIP의 같은 보고서는 바이트 단위로 같았다.
두 ZIP 모두 CRC·내보내기 파일 해시 검사를 통과했다. 로그 첨부도 정상이다.
평가 종료 코드 2와 바깥 스크립트 종료 코드 1은 서로 다른 단계의 상태다.

## 수정한 동작

- 커버 서바이벌·공모전·경연·대회·콘테스트, 커버 영상·라이브·연주·공연을
  공연 표현으로 구별한다. 특정 질의 ID, 플랫폼, 가수나 정답 곡은 하드코딩하지 않는다.
- 앨범 커버 공모전·앨범의 커버 영상처럼 실제 표지에 연결된 표현은 계속
  이미지 단서다. 별도의 실제 표지 설명과 들리는 반주·코러스도 유지한다.
- 과거 개최·우승·참가·사용 사건과 과거 커버 영상 공개·촬영 사건을 Context로
  추출한다. 모델이 Context를 비워도 원문이 관계를 제공하면 규칙으로 보존한다.
- 커버 표현 내부의 탭·줄바꿈은 Context 분리 전에 정리한다. 원래 사용자 질의는
  보존하며 별도 사건 사이의 줄바꿈을 전부 합치지는 않는다.
- 같은 절의 ‘커버 대회에서 우승한 곡인데 앨범 커버에는 꽃이 있었다’는
  Context와 Image를 모두 유지한다. Context 문장에는 공연 사건 앞부분만 보관한다.
  표지에 그려진 참가자의 사진을 같은 사건으로 추출하지 않는다.
- 가사·곡명 속 행사 문구와 앞으로 할 커버 활동은 과거 배경 사실로 추가하지
  않는다. 모델이 그 미래 활동을 Context로 제안한 경우도 검사한다.

QueryAnalyzer 프롬프트와 기존 모달리티 가중치 정책, 고정 검색 설정, 평가 정답·
리뷰·합격 기준은 유지했다. 인덱스 적재나 임베딩 재생성은 필요하지 않다.

고정 설정: Context 가중치 0.5, Dense 사실 100개, Sparse 프로필 100개,
Candidate 30개, 화면 Top-10, 작품명 배수 2, 리랭킹 OFF, 기준 연도 2026.

## 이번 수정의 검증 범위

- 최초 경계 검사 91개 중 수정 전 67개 실패로 문제를 재현했다.
- 새 검사 102개 + 기존 관련 검사 320개 = 422개 통과했다. 새 검사는
  공모전 유형, 띄어쓰기·탭·줄바꿈·따옴표, 공연 영상과 실제 표지,
  같은 문장의 혼합 단서, 미래 활동, 가사·제목, 모델의 허위 추가,
  입력 보존, 동기·비동기 Analyzer의 첫 시도 수용을 검사한다.
- 기존 dev/test 분석 106개와 앞선 확장 분석 30개를 수정 전후 다시 파싱했다.
  파싱 오류 0건, Context 유무·개수·문장 변경 0건, 이미지/오디오 문장·
  모달리티 가중치 변경 0건이었다. 원래 질의도 보존됐다.
- 실제 실패 질의는 수정 전 이미지 요구 오류가 발생했고, 수정 후 이미지/오디오
  요구 없이 Context 1개로 파싱됐다. 여기에는 합성 모델 응답을 사용했다.
- 기존 확인 스크립트 3개와 새 16사례 확인 스크립트가 모두 통과했다.
  실제 `QueryAnalyzer._parse`를 사용하며 Gemini API나 Qdrant는 호출하지 않는다.
- Python 3.11 문법과 변경한 PowerShell 스크립트의 정적 문법을 확인했다.
  로컬 pytest는 Python 3.12 환경에서 실행했다. 실제 Windows PowerShell·
  Docker Python 3.11 검사는 아래 명령으로 실행한다.

저장된 응답의 파싱 비교는 새 모델 호출·OFF/ON 검색 성능 재측정이 아니다.
여기서 사용자의 현재 모델 API와 Qdrant로 재평가하지는 않았다. 60문항 중
앞부분이 이미 관찰되었으므로 완료해도 전부 미노출된 블라인드 평가라고 부르지 않는다.

## 적용과 눈으로 확인하기

프로젝트 루트 `Vague-Finder-Org`의 VSCode PowerShell에서 실행한다.
압축을 푸는 단계가 실제 코드 적용이다. 이 ZIP은 이전 Broadcast Fact 수정이
적용된 현재 환경을 기준으로 하며, 사설 평가 입력·캐시·정답을 포함하지 않는다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Cover_Contest_Fix_20261006.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 찾지 못했습니다.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force

$out = 'artifacts/context_fixed_followup_20261006_cover_contest_fix'
.\experiments\namuwiki\run_context_cover_contest_fix.ps1 -OutputDir $out -CheckOnly
if ($LASTEXITCODE -ne 0) { throw '경계 검사 실패. console_check.log를 확인하세요.' }
git diff --check
```

마지막에 `[PASS] 16 cover parser cases`와 다음과 같은 사례별 출력을 확인한다.

| 예시 | Context | Image | Audio |
|---|---:|---:|---:|
| 커버 서바이벌 공모전 개최 사실 | True | False | False |
| 다른 가수의 커버 영상 공개 | True | False | False |
| 앞으로 커버 대회에 참가할 곡 추천 | False | False | False |
| 가사·제목에 나온 행사 문구 | False | False | False |
| 앨범 표지 디자인 공모전의 꽃 그림 | False | True | False |
| 공연 사건과 별도 앨범 표지 | True | True | False |
| 공연 사건과 실제 남자 코러스 | True | False | True |
| 공연 사건·표지·코러스의 혼합 | True | True | True |

마지막 `blocked_as_expected`는 오디오에 앨범 그림 묘사가 섞인 문장을 의도대로
차단했다는 뜻이다. 새 경계 검사 파일은 사설 데이터 없이 102개를 검사한다.
묶음 검사 전체 수는 사설 평가 입력 설치 여부에 따라 달라질 수 있다.

## 설정을 고정한 전체 재측정

위 검사가 통과하면 같은 `$out`으로 실행한다.

```powershell
.\experiments\namuwiki\run_context_cover_contest_fix.ps1 -OutputDir $out
```

전체 Context/API/설명 테스트 후 backend를 중지하고, dev 74문항·test 32문항·
확장 60문항·독립 재분석 반복 검사를 기존 러너로 진행한다. 종료 시 backend를
다시 올리고 보고서와 콘솔 로그를 묶는다. 가중치·정답·판정 기준을 바꾸지 않는다.

코드가 바뀌어 평가 등록 해시가 달라지므로 이전 `broadcast_fact_fix` 출력 폴더를
재사용하지 않는다. 이전 보고서와 캐시는 보존하고 새 출력 폴더로 등록한다.
이 실행은 새 모델 분석을 호출하므로 시간이 걸린다. 새 코드로 시작한 이 폴더에서
다시 중단되면 동일 코드·입력·설정을 유지한 재실행으로 정상 캐시를 이어 사용한다.
코드를 다시 고친 경우에는 또 다른 출력 폴더가 필요하다.

실행 마지막의 실제 경로를 사용해 로그 포함 ZIP을 보낸다.

```text
Send this file: ..._feedback_with_log_....zip
```

로그 첨부만 다시 필요하면 아래 명령은 검색·모델 분석 없이 패키징만 수행한다.

```powershell
.\experiments\namuwiki\run_context_cover_contest_fix.ps1 -OutputDir $out -PackageOnly
```

공개 PR에는 수정 코드·일반 예시 테스트·가이드만 포함한다. 평가 정답·CSV·
리뷰·캐시·콘솔 로그·피드백 ZIP과 Qdrant 데이터를 커밋하지 않는다.
