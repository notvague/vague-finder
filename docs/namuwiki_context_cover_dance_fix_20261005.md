# Context 커버댄스 분류 오류 수정과 재평가

2026-10-05 실행 로그에서 `fresh015`가 세 번 모두 `album-art visual clue requires image_english_query`로 실패했다. 실행 코드와 같은 SHA-256의 `modality_queries.py`로 재현한 결과, `커버댄스` 안의 `커버`가 표지 단서로 잡혔다. API 장애를 나타내는 결과가 아니었다.

`context_query.py`에도 별도의 표지 판정이 있어서 공연 단서를 제거할 수 있었다. 두 경로가 같은 명시적 표지 판정을 사용하도록 맞췄다. 특정 query_id, song_id, 정답 제목을 검색 코드에 넣지 않았다.

| 사용자가 제공한 단서 | 처리 |
|---|---|
| 커버댄스·댄스 커버·춤 커버를 했다는 사건 | Context 공연 이력 |
| 커버댄스라는 단어 자체 | 앨범 표지나 음원의 댄스 장르로 해석하지 않음 |
| 베이스 소리·반주·템포 등 실제 음향 묘사 | 기존 Audio 검증 유지 |
| 독립적으로 언급한 앨범 표지·커버 그림 | 기존 Image 검증 유지 |
| 커버댄스할 곡을 고르는 계획, 가사·표지에 적힌 사건 | 외부 공연 사실을 만들어내지 않음 |

실제 표지가 있는데 이미지 검색 문장이 빠졌거나, 음향 단서가 있는데 오디오 검색 문장이 빠진 모델 출력은 계속 거부한다. fallback을 정상 분석으로 바꾸거나 평가에서 제외하는 수정이 아니다.

변경한 검색 소스는 `src/retrieval/modality_queries.py`와 `src/retrieval/context_query.py`다. Context 가중치 0.5, Dense 사실 100, Sparse 프로필 100, 후보 30, 표시 10, 작품 단서 배수 2.0 및 리랭킹 OFF 조건은 기존 평가 도구에서 그대로 사용한다. 평가 CSV·정답·Dense/Sparse·Qdrant 적재 내용을 변경하지 않았다.

로컬 Python 환경에서 새 경계·로그 캡처 테스트와 기존 Context 분석·모달리티 경계 테스트 158개가 통과했다. 실제 Gemini 응답을 생성하거나 검색 성능을 다시 측정한 결과는 아니다. PowerShell 파일은 구문 파서로 검사했으며 Windows/Docker 실행 검증은 아래 명령으로 진행한다.

프로젝트 루트 PowerShell에서 ZIP을 적용하고 짧은 검사를 실행한다.

```powershell
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File `
    -Filter 'Vague-Finder_Context_Cover_Dance_Guard_Fix_20261005.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 찾지 못했습니다.' }

docker compose stop backend
if ($LASTEXITCODE -ne 0) { throw 'backend 중지 실패' }
try {
    Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force
    .\experiments\namuwiki\run_context_cover_dance_fix.ps1 -CheckOnly
} finally {
    docker compose up -d backend
}
```

`-CheckOnly`는 Docker 이미지 빌드, 관련 테스트, 실제 QueryAnalyzer 파서의 오프라인 검사를 실행한다. Gemini API와 Qdrant에는 접근하지 않는다. 콘솔에 아래와 같은 분리 결과가 보여야 한다.

```text
radio_and_bass:    image=false audio=true  context=true
dance_history:    image=false audio=false context=true
cover_and_history: image=true audio=false context=true
dance_genre:      image=false audio=true  context=false
[PASS] Offline analyzer parse; no Gemini call and no Qdrant access.
```

통과하면 같은 프로젝트 루트에서 전체 평가를 실행한다.

```powershell
.\experiments\namuwiki\run_context_cover_dance_fix.ps1
```

이 명령은 기존 실행 도구를 `-Mode Fresh`로 호출한다. dev 74, test 32, 새 작성 질의 60문항을 동일한 분석의 OFF/ON으로 비교하고, 기존 도구가 정한 반복 분석 검사도 진행한다. 각 단계의 기존 중단 기준을 적용한다. 실패가 발생해도 다른 단계의 결과와 중단 이유는 보존한다.

분석기 후처리가 바뀌었으므로 출력 폴더는 `artifacts/context_fixed_followup_20261005_cover_dance_fix`로 분리한다. 이전 `artifacts/context_fixed_followup_20261005`와 그 안의 14개 독립 질의 분석 캐시는 보관한다. 새 평가로 복사해서 재사용하지 않는다. 새 출력 폴더의 새 분석 캐시와 코드 해시를 등록한다. 이후 코드·설정·입력·코퍼스를 바꾸지 않은 재시도는 동일 명령으로 재개할 수 있다.

새 질의셋의 검색 결과는 아직 관찰되지 않았으나 `fresh015`의 분석 실패는 수정 과정에서 확인했다. 따라서 결과를 설명할 때는 “새 작성 질의셋으로 분류 오류 수정 후 재검증”이라고 적고, 완전히 미관찰된 블라인드 검증이라고 표현하지 않는다. 기존 dev/test는 이미 관찰한 회귀 평가다.

Windows PowerShell 5.1의 `Start-Transcript`는 이번 로그에서 Docker 출력을 대부분 누락했다. 새 실행기는 자식 프로세스의 실제 stdout/stderr를 함께 수집하고 UTF-8 파일로 저장한다. 기존 평가 ZIP의 보고서 바이트와 해시를 검증한 뒤 전체 실행 로그를 추가한다. 보고서의 판정은 수정하지 않는다.

성공하거나 중단되면 다음 ZIP 한 개를 보내면 된다.

```text
artifacts/context_fixed_followup_20261005_cover_dance_fix_feedback.zip
```

이 ZIP에는 `fresh/validation_report.json`, 단계별 보고서·체크포인트·분석 캐시와 `console_run.log`가 들어간다. 이미지·오디오 경로 실행, 후보·Top-10 손실, 근거 라벨 비교, 분석 반복 변동을 함께 확인할 수 있다. 빌드나 테스트 단계에서 중단돼 보고서 ZIP이 생성되지 않은 경우에는 `artifacts/context_fixed_followup_20261005_cover_dance_fix/console_run.log`를 보낸다.

평가 결과와 로그에는 비공개 질의·정답·코퍼스 정보가 들어갈 수 있으므로 공개 레포에는 올리지 않는다. 수정 ZIP에는 평가 CSV·음원·가사·곡 메타데이터·임베딩을 포함하지 않았다.

