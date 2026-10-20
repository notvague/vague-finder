# Context 평가: 오디오 background/foreground 검증 수정

## 로그에서 확인한 원인

전달받은 콘솔 로그에서 `fresh022`는 세 번 모두 다음 검증 오류로 재시도한 뒤 fallback으로 종료됐다.

```text
audio_english_query contains visual-only wording: 'background'
```

이는 모달리티 검증 오류다. 이 문항의 해당 실패 로그에는 모델 서버 연결 오류가 없다. 기존 정규식은 `background`와 `foreground`라는 단어 자체를 시각 정보로 취급했다. 실제 분석기에서 합성 모델 응답을 사용하면 `male background vocals` 및 `guitar playing in the background` 같은 정상적인 소리 표현도 같은 오류로 거부된다.

로그에는 원래 모델 응답 전체가 없으므로 원래 영문 문장이 정확히 `background vocals`였다고 단정하지 않는다. 이번 수정은 재현된 공통 규칙 오류를 바로잡으며, 실제 `fresh022`의 재분석 성공 여부는 새 평가 결과에서 확인해야 한다.

이전 실행에서는 dev 74문항과 test 32문항이 passed였고, 독립 평가 60문항의 분석 중 21문항이 저장됐다. 독립 평가 검색 및 반복 분석 검사는 완료되지 않았다. 이전 커버댄스 실패 문항은 이번 로그에서 정상 분석됐다.

## 수정 범위

- 오디오 영문 질의의 `background/foreground`를 짧고 명시적인 소리 구절 안에서만 허용한다. 예: `background vocals`, `background instrumental layers`, `piano softly playing in the background`.
- 보컬·코러스·악기·소리 등 오디오 명사와 제한된 수식어·서술어를 사용한다. 임의 길이의 문장이나 다른 절까지 허용 범위를 확장하지 않는다.
- 모든 시각 단서를 끝까지 검사한다. 정상적인 소리 구절 뒤에 앨범 배경색, 이미지, 사진, 타이포그래피가 섞이면 여전히 거부한다. 이미지·사진의 일반적인 복수형과 명시적인 artwork도 확인한다.
- 원래 사용자가 오디오 단서를 주지 않은 경우에는 생성된 `background vocals`를 계속 폐기한다. 모델의 영문 표현만으로 오디오 경로를 열지 않는다.
- 영문 질의에서 단어를 지우거나 다른 문장으로 바꾸지 않는다. Context 단서와 기존 가중치 정규화·오디오 관계 보충 동작을 보존한다.
- 실제 모달리티 위반은 기존처럼 재시도 후 fallback을 반환한다. 평가에서는 fallback을 계속 거부한다.
- 직전 로그 첨부 수정도 포함한다. 원본 피드백 ZIP을 교체하지 않고 로그 포함 ZIP을 고유한 이름으로 생성한다. 로그를 이어 써 이전 실패 기록도 보존한다.

검색 가중치, 후보 폭, 코퍼스, 라벨, 평가 코드 및 질문 내용은 바꾸지 않는다.

## 로컬 검증

관련 다섯 테스트 파일에서 **총 227개 테스트 통과**. 새 공간 표현 경계 테스트는 **63 passed**다. 최초 실행에서는 226개 통과, 비공개 pilot 평가 JSON 미설치로 1개 스킵이 있었다. 이후 등록 기록의 SHA256과 정확히 일치하는 기존 평가 파일을 복원해 해당 테스트를 단독 재실행했고 통과했다. 평가 파일은 수정하지 않았으며 전달 ZIP에도 포함하지 않는다.

정상 소리 표현, 실제 시각 오염, 혼합 앨범/코러스 질의, 오디오 단서가 없는 경우, Context 보존, 동기·비동기 분석의 첫 시도 성공, 실제 위반의 재시도와 fallback, 캐시 지문 변경, 로그와 ZIP 보존을 검증했다.

Python 3.11 문법과 PowerShell 구문을 검사했다. 테스트는 로컬 Python 환경에서 실행했으며, 실제 Windows Docker 전체 테스트 및 실제 모델 API/코퍼스 평가는 아래 명령으로 실행한다. 새 코드의 실검색 지표는 아직 측정되지 않았다.

## 적용과 눈으로 확인하기

프로젝트 루트의 PowerShell에서 실행한다. 기존 커버댄스 실행 도구를 재사용하되, **새 결과 폴더**를 명시한다.

```powershell
$ErrorActionPreference = 'Stop'
$zip = Get-ChildItem -Path @('.', "$env:USERPROFILE\Downloads") -File -Filter 'Vague-Finder_Context_Audio_Spatial_Guard_Fix_20261005*.zip' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $zip) { throw '수정 ZIP을 프로젝트 폴더 또는 Downloads에 저장해 주세요.' }
Expand-Archive -LiteralPath $zip.FullName -DestinationPath . -Force

$out = 'artifacts/context_fixed_followup_20261005_audio_spatial_fix'
.\experiments\namuwiki\run_context_cover_dance_fix.ps1 -CheckOnly -OutputDir $out
```

`-CheckOnly`는 빌드, 관련 단위 테스트, 두 오프라인 분석 확인 도구를 실행한다. 실제 모델 API나 Qdrant 검색을 호출하지 않는다. 새 도구의 확인 결과는 다음과 같다.

| case | image | audio | Context | 판정 |
|---|---|---|---|---|
| background_chorus_and_fact | false | true | true | 소리와 제작 비화를 함께 보존 |
| guitar_behind_lead | false | true | false | 악기·보컬의 소리 관계 보존 |
| artwork_and_chorus | true | true | false | 앨범 묘사와 소리를 각각 보존 |
| artwork_only | true | false | false | 모델이 추가한 오디오를 폐기 |
| real_visual_leak | — | — | — | 실제 시각 오염을 거부 |

마지막에 다음 문구가 나와야 한다.

```text
[PASS] Spatial sound accepted; artwork leakage blocked; no model API or Qdrant call.
```

## 새 등록으로 전체 평가 실행

위 검사가 통과하면 같은 PowerShell에서 실행한다.

```powershell
.\experiments\namuwiki\run_context_cover_dance_fix.ps1 -OutputDir $out
```

이 명령은 전체 관련 테스트 후 백엔드를 중지하고, dev 74 → test 32 → 독립 평가 60 → 반복 분석 검사를 실행한다. 필요한 라벨·리뷰 노트는 기존 등록 입력을 그대로 사용한다. 종료 시 보고서를 내보내고 백엔드를 다시 시작한다. 실패한 경우에도 원래 종료 판정과 로그를 보존한다.

고정 설정은 Context weight 0.5, Dense fact_k 100, Sparse song_k 100, Candidate@30, Top-10, named_media_multiplier 2.0, 리랭킹 OFF, 기준 연도 2026이다. Dense의 100은 사실 개수이며 곡 개수가 아니다.

분석 규칙의 코드 해시가 바뀌므로 이전 `context_fixed_followup_20261005_cover_dance_fix` 폴더의 21개 분석 또는 이전 dev/test 결과를 새 실행에 가져오지 않는다. 이전 폴더는 그대로 보관한다. 새 폴더에서는 모든 문항을 새 분석으로 시작하고, OFF/ON은 문항마다 같은 분석을 공유한다.

새 실행이 일시적으로 중단되면 코드·환경·코퍼스·설정·라벨을 유지한 채 **같은 새 폴더**로 명령을 다시 실행한다. 등록 검사를 건너뛰거나 fallback을 허용하지 않는다. 코드 수정이 다시 필요하면 별도의 새 등록 폴더를 사용한다.

## 보낼 파일과 해석 범위

캡처 도구가 마지막에 출력하는 `Send this file:` 경로의 **로그 포함 새 ZIP**을 보내면 된다. 파일명은 `context_fixed_followup_20261005_audio_spatial_fix_feedback_with_log_*.zip`이며 끝의 고유 문자는 실행마다 다르다.

이미 생성된 로그를 나중에 묶으려면 평가 없이 다음 명령을 실행할 수 있다.

```powershell
.\experiments\namuwiki\run_context_cover_dance_fix.ps1 -PackageOnly -OutputDir $out
```

이번 소스 수정의 단위 검증과 실검색 성능 검증은 구분한다. 실제 지표와 근거 적합성은 새 보고서에서 판단한다. 앞서 실패 문항을 보고 공통 규칙을 수정했으므로 기존 60문항을 완전히 처음 보는 블라인드 평가라고 부르지 않는다. 라벨 및 설정을 고정한 회귀·독립 질의 검사로 기록한다.

패키지는 평가 CSV, 사용자 로그, 음원, 임베딩 또는 곡 목록을 포함하지 않는다. 사용자 환경에서 생성하는 피드백 ZIP은 비공개 평가 자료다.
