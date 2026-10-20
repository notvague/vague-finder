# Context 9단계: 전체 범위 확정, 재생성, Qdrant 최종 적재

## 완료의 뜻

Context의 `scope_complete=True`는 **코어 메타데이터와 미디어가 유효한 모든 곡**에
`ok`, `no_trivia`, `not_found` 가운데 검증된 상태가 있고, 그 상태에 맞는 최신 artifact가
있다는 뜻이다. 모든 곡에 나무위키 사실이나 검색 벡터가 생긴다는 뜻은 아니다.
찾지 못한 문서에 대한 사실이나 검색 프로필을 만들어서는 안 된다.

이전에 언급한 444곡은 오래된 범위의 숫자다. 새 raw가 들어온 뒤 관측된 상태는
3017 디렉터리 중 코어 유효 3016곡, `needs_review` 481곡, artifact 2535곡이었다.
실행 시점의 파일을 다시 읽어 정확한 수를 결정한다. 코어에서 거절된 1곡
(`relation_context_tags = unknown`)은 수정하여 유효하게 만들기 전까지 범위에 들어가지 않는다.
실행서는 이전에 확인된 유효곡 3016곡보다 범위가 작아지면 적재를 거부한다.
의도적으로 다른 범위에서 실행하는 경우에만 `-MinimumScope`를 해당 범위의
검증된 최솟값으로 지정한다.

## 1. 비공개 검토표 생성

ZIP을 프로젝트 루트에 풀고 PowerShell에서 실행한다.

```powershell
.\experiments\namuwiki\run_context_step9.ps1 -Mode Prepare
```

`data/context/review_decisions_step9.csv`를 만든다. **`data/context`는 Git에서
제외되며, 이 CSV에는 곡명·가수·실패 사유·시도한 주소가 담긴다. 커밋·PR·공유 ZIP에
넣지 않는다.** 파일이 이미 있으면 덮어쓰지 않는다. `manual_review.csv`는 감사용 표본만
담는 파일이므로 최종 확정용 입력으로 사용할 수 없다.

이미 `needs_review=0`이고 artifact가 `3016/3016`이면 검토표는 헤더만 있으며,
새로 상태를 바꿀 곡이 없다. 이 경우 CSV를 채우지 않고 곧바로 `-Mode Publish`를
실행한다. 배치 상태 출력은 곡명·폴더 경로를 생략한 ASCII JSON이라 Windows
PowerShell에서도 출력 인코딩 때문에 파싱이 깨지지 않는다.

검토표를 살펴보는 명령:

```powershell
$csv = '.\data\context\review_decisions_step9.csv'
$rows = @(Import-Csv -LiteralPath $csv)
$rows.Count
$rows | Group-Object previous_error_code | Sort-Object Count -Descending |
    Select-Object Count, Name
$rows | Select-Object -First 10 song_id, title, artists, previous_error_code, attempted_url
```

이미 수동으로 **현재 검토표에 나온 모든 곡의 곡명·가수와 문서 일치 여부를
확인했을 때에만**, 그 결론을 기록한다. 해당하지 않는 곡은 일괄 표시하지 말고
개별 확인 또는 수집 재시도 대상에 둔다. 예전에 확인한 444개 ID와 현재 파일의
ID가 다르면 새로운 ID는 새로 판단해야 한다.

```powershell
$csv = '.\data\context\review_decisions_step9.csv'
$rows = @(Import-Csv -LiteralPath $csv)
foreach ($row in $rows) {
    $row.decision = 'not_found'
    $row.review_note = '곡명과 가수를 수동 대조했으며 일치하는 나무위키 곡 문서를 찾지 못함'
}
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
```

`decision=not_found`는 **일치하는 곡 문서를 검증 과정에서 찾지 못했다**는 결정이다.
존재하지 않는다는 수학적 증명이 아니다. 접근 제한·임시 네트워크 오류만 있었거나,
일치하는 문서가 확인된 곡은 이 결정을 해서는 안 된다. 그런 곡은 정확한 URL을
확인하고 기존 `--retry-needs-review` 수집 흐름으로 재처리한 뒤 검토표를 새로
만든다. 검토표의 `meta_sha256`은 수정하지 않는다.

## 2. 한 번에 최종화

```powershell
.\experiments\namuwiki\run_context_step9.ps1 -Mode Publish
```

실행서의 순서:

1. Docker 이미지와 변경 파일을 확인하고 대상 테스트를 실행한다.
2. 검토표의 **모든 현재 `needs_review` 곡**에 대해 ID, 해시, 명시적 결정,
   검토 메모를 검사한다. 다른 사람이 raw를 바꿨으면 수정 전에 실패한다.
3. 백엔드를 멈추고 각 `meta.json`을 바이트 그대로 백업한다. 검토된 곡만
   `not_found`로 바꾸고 `source_url`과 사실 목록을 비운다. 결정 메모는
   `data/context/manual_resolutions/`에 보관한다. 중간에 끊겨도 같은 검토표로
   검사 후 재개할 수 있다.
4. `--artifacts-only`로 현재 메타에 맞는 artifact를 배치별로 갱신한다.
   진척이 없거나 invalid/orphan artifact가 나오면 즉시 멈춘다. 크롤링이나
   `--force`에 의한 다른 곡의 재수집은 하지 않는다.
5. `--require-complete-context`로 Dense 사전 검사 → 생성, BM25 사전 검사 →
   전체 코퍼스 재적합, Qdrant 사전 검사 → 세대 적재 순서로 실행한다.
6. 저장된 manifest와 **실제 활성 alias 및 정확한 포인트 수**가 일치하는지
   읽어 확인한 뒤 백엔드를 재시작한다. 오류가 나도 백엔드 재시작을 시도한다.

최종 결과는 `artifacts/context/step9_final_report.json`에 저장된다. 확인 기준은
`[PASS] Complete Context generation active`, `source_scope_complete: true`,
`source_song_count = artifact의 scope_total`, `artifact pending = 0`,
활성 Dense/Sparse alias의 점 수가 manifest와 일치하는 것이다. 사실이 없는
`not_found` 곡은 Dense/Sparse 포인트를 만들지 않는다. 기존 약 3995 사실과
478 프로필이 그대로여도, 범위 확정 및 새 artifact/manifest를 반영한 최종 적재일 수 있다.

추가 확인을 원하면 백엔드를 중지한 상태에서만:

```powershell
docker compose stop backend
try {
    docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.verify_context_step9
    if ($LASTEXITCODE -ne 0) { throw 'Qdrant 활성 세대 검증 실패' }
} finally {
    docker compose up -d backend
}
```

실패했다면 `artifacts/context/unresolved.json`, `artifacts/context/manifest.json`,
검토표에 나온 ID/해시, 터미널의 첫 오류를 확인한다. `--require-complete-context`를
빼거나 `--force`, `--prune-old-generations`로 우회하지 않는다.

기존 코드에서 artifact 배치가 변경 없이 Context manifest만 다시 발행하면,
BM25가 같은 프로필이라는 이유로 이전 manifest를 재사용하여 Qdrant 사전 검사에서
`context sparse publication targets another context manifest`가 발생할 수 있었다.
현재 BM25 재사용 검사는 **Context manifest 파일 해시**도 대조한다. 이 오류에서
중단된 경우 수정 ZIP을 다시 적용한 뒤 `-Mode Publish`를 재실행하면 원본 곡의
Dense 벡터는 재사용하면서 BM25 전체 프로필을 다시 적합하고, 동일한 Context
manifest를 대상으로 최종 적재한다. 검토표가 비어 있다면 수동 판정을 다시 만들지 않는다.

## 3. 이후 평가

8단계 dev/test 측정값은 **부분 코퍼스·리랭킹 OFF의 기준선**으로 보존한다.
최종 코퍼스를 적재한 뒤 10단계에서 같은 절차로 새 기준선을 측정한다. test의
`c703`은 Context ON에서 Top-10 밖으로 밀렸던 회귀 사례이므로 함께 추적하되,
해당 한 문항을 기준으로 test 가중치를 맞추지 않는다. 새로운 코퍼스에서의
후보 수와 최종 Top-10은 재측정 전까지 보장되지 않는다.
