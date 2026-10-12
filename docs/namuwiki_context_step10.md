# 10단계: 전체 코퍼스 검색 평가와 설정 동결

## 평가 계약

이 절차는 9단계에서 활성화한 완전한 Context 세대와 같은 Text Qdrant에서 기존 검색
(`use_context=False`)과 Context 추가 검색(`True`)을 **같은 저장된 QueryAnalysis**로
비교한다. 두 조건 모두 `use_rerank=False`이며 `top_k=10`이다. 평가기가
`artifacts/context/step9_final_report.json`, 전체 Context manifest, Dense/Sparse
Qdrant alias, 곡 수와 Text point 수 및 모든 정답의 Text 색인 존재를 확인한다.
중간에 세대가 바뀌면 중단한다. `candidate_k=50`을 실험해도 Candidate@30은 첫
30곡만 세고, 전체 후보 50곡의 적중 여부는 별도 열로 기록한다.

**Context OFF는 과거의 적은 곡을 대상으로 검색하는 시스템이 아니다.** 똑같은 현재
3016곡 Text/Image/Audio/Qdrant와 같은 분석에서 Context 경로 하나만 끈 조건이다.
이것이 나무위키 검색 경로의 추가 효과를 비교하는 올바른 기준선이다.

9단계 최종 보고서상 현재 적격 범위는 3016곡이고 Context 사실은 3995건,
프로필은 478곡이다. 나머지 2538곡은 `not_found` 또는 `no_trivia`로 종료되었다.
결과를 보지 않고 모든 곡에서 Context 개선을 기대하지 않는다.

### 설정 선택과 안전장치

평가 코드는 새 실행 전에 고정된 24개 설정을 dev에서 비교한다:

| 항목 | 값 |
| --- | --- |
| 외부 Context RRF 가중치 | 0.25, 0.5, 1.0 |
| 추가 가중치 후보 | 0.6, 0.7, 0.8, 0.9 (Dense/Sparse 100/100, 후보 30) |
| 명시된 작품 사용 단서만의 배율 | 1.5, 2.0 (기본 0.5 × 이 배율, Dense/Sparse 100/100, 후보 30) |
| Dense 사실 수 / Sparse 곡 수 | 100/100, 160/100, 100/160 |
| 리랭킹 전 후보 폭 | 30, 50 |
| 화면 결과 수 | 검색은 10 고정, 동일한 후보 순서로 5/10/20 표시 폭의 적중률 별도 계산 |

기존 `.env` 값과 무관하게 각 실험에서 라우터 설정만 바꾼다. 기존 30곡 후보의
Context OFF 결과를 고정 기준으로 사용하며, 가중치나 검색 폭만 바꾼 Context ON과
비교한다. Context 무관 질의의 OFF/ON 목록은 정확히 같아야 한다.

가사·분위기/소리·이미지를 포함한 **기존 v0.5 76문항**은 전부 포함한다.
dev에서 기존 질의 또는 Context 없는 대조군의 Candidate@30·Hit@1/5/10 중
한 문항이라도 손실이 있거나 Context 질의의 Hit@10이 떨어지면 해당 설정을
제외한다. 조건을 통과한 설정 중 Context Hit@10, Candidate@30, MRR@10을 순서대로
높이는 것을 고르고 동점이면 가중치·검색 폭·후보 폭이 작은 것을 선택한다.
`nw001`·`q203`의 정답이 Candidate@30에 모두 포함되어야 하며, 검토된
`nw005`·`nw021`·`nw024`에서 다른 사건의 사실을 표시하면 설정을 탈락시킨다.
명시된 작품 사용 배율은 대상 작품이 원문에 있고, 단일 Context 사용 관계인
질의에만 적용한다. `m102` 같은 공연 참여 단서에는 적용하지 않는다.
적격 설정이 하나도 없으면 `blocked`로 끝나며 배포값을 추측하지 않는다.

개발 74문항에서 설정을 **고정한 뒤** 기존 시험 32문항을 한 설정으로 재검증한다.
시험 질의 `c703`의 손실은 이전 대화에서 이미 공개됐다. 따라서 이 시험은
엄밀한 새 블라인드 평가가 아니며 통과해도 `provisional`이다. 시험 손실을 본 뒤
다른 가중치를 선택하지 않는다. 추가 확신이 필요하면 기존 평가의 질의·곡 ID와
겹치지 않는 독립 질의셋을 *실행 전에* 검토·기록한다.

## 실행 (프로젝트 루트의 VS Code PowerShell)

Step10 v2 변경 ZIP을 새 org 프로젝트 루트에 풀고, 기존 `data/`, `artifacts/`, `.env`
등은 유지한다. 프로젝트에는 개인 평가 CSV/JSON 두 종류와 기존 v0.5 CSV,
완료된 9단계 보고서가 필요하다. ZIP에는 개인 질의·가사·음원·곡 메타를 넣지 않는다.

```powershell
Expand-Archive -LiteralPath .\Vague-Finder_Context_Step10_v2.zip -DestinationPath . -Force
.\experiments\namuwiki\run_context_step10.ps1 -Mode All
```

이 명령은 빌드와 회귀 테스트를 먼저 실행하고, 백엔드를 멈춰 embedded Qdrant의
파일 잠금을 풀고, 분석 캐시 준비 → dev 24설정 → 기존 test 한 설정을 순서대로
실행한다. 중간에 실패해도 `finally`에서 백엔드를 다시 시작한다. 평가 중에는
`artifacts/context_eval_step10_v2/{dev,test}_checkpoint.json`을 문항마다 기록하므로
같은 명령으로 재시도할 수 있다. 단, 데이터/모델/코드/분석 캐시가 바뀌면 오래된
체크포인트 재사용을 거부한다. 그 경우 결과를 백업한 후
`-OutputDir artifacts/context_eval_step10_v3`처럼 새 출력 경로로 평가한다.
검색 시간이 길 수 있다. 실험 중 백엔드가 정지해 있으니 업무 시간 외에 실행한다.

과거 분석 캐시에 폴백이나 혼합 지문이 남아 있으면 `prepare`에서 멈춘다. 해당
캐시를 **복사해 백업**하고, 분석 API가 정상인지 확인한 후 캐시를 새로 만들 수 있다:

```powershell
$cache = '.\artifacts\context_eval_step8\analysis_cache.json'
if (Test-Path -LiteralPath $cache) {
    Copy-Item -LiteralPath $cache -Destination "$cache.before_step10"
    Remove-Item -LiteralPath $cache
}
.\experiments\namuwiki\run_context_step10.ps1 -Mode All
```

이전 dev/test 평가 파일은 덮어쓰지 않는다. 새 결과는
`artifacts/context_eval_step10_v2/`에만 저장한다:

- `dev_selection.json`: 24설정의 적격 여부, 손실 ID, 선택 규칙
- `final_report.md`, `final_report.json`: 기존 test 재검증과 잠정 설정
- `dev/*.csv`, `test/*.csv`: 질의별 OFF/ON/기준선 순위, 구간별 지표, 표시 근거
- `*_checkpoint.json`: 재시작을 위한 코퍼스/소스/코드/분석 해시와 진행 상태

마지막에 `inspect_context_step10_v2`가 `nw001`·`q203`의 후보 순위와 세
문제 사실의 표시 여부·실제 출처를 출력한다. 두 곡이 후보 30개에 없거나 잘못된
사실이 재등장하면 명령이 실패한다. 사실이 검색 범위에 없으면 표시를 생략한다.

예전 8단계 숫자는 이전 Context 세대에서 나온 *참고치*다. 이번 원인 비교는 새
세대 안에서 짝지은 OFF/ON 지표를 사용한다. `candidate_hit30`, `candidate_recall30`,
Hit@1/5/10, MRR@10, 맥락/복합/대조·가사·분위기/소리·이미지별 분모와 손실 문항을
확인한다. bootstrap 구간은 2000회 재표본의 설명적 불확실성 범위이며 30개
Context 문항을 모든 사용자 질의로 일반화하지 않는다.

## 독립 질의셋: 강한 품질 주장 전 확인

새 곡을 대상으로 팀원이 **이번 dev 결과를 열어 보기 전에** 최소 30문항을 수동 작성해
`data/context/eval_blind_step10.csv`에 둔다. UTF-8 CSV 헤더는 다음과 같다:

```text
query_id,query,relevant_ids,group,clue_category,evidence_record_ids,review_note
```

30곡을 처음부터 찾기 어렵다면 공개 저장소의 샘플러로 **미검토 초안만** 만든다:

```powershell
docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.prepare_context_blind_draft
Copy-Item .\data\context\eval_blind_step10_draft.csv .\data\context\eval_blind_step10.csv
```

초안은 기존 정답 곡을 제외하고 여섯 사실 유형에서 세 곡씩, 사실 없는 곡에서
대조·가사·소리·앨범아트용 세 곡씩 결정론적으로 고른다. `query`, `review_note`는
의도적으로 비워 둔다. 해당 곡의 실제 가사·앨범아트를 확인해 새로운 구어체
질의와 정답 근거를 직접 쓰고, 검토 뒤 `reference_fact_for_author` 열은 제거한다.
재실행 때 기존 CSV를 덮어쓰지 않는다. 초안 그대로는 등록에 실패한다.
이번 dev 결과나 실패 문항을 보고 질의 유형을 바꾸지 않는다.

각 질의에 정답 song_id 하나와 구체적인 검토 메모를 넣는다. Context 문항은
`artifacts/context/songs/<song_id>.json`의 사실 `record_id`와 실제 category를
기록한다. 필요한 사실이 둘이라면 `id1|id2`처럼 둘 다 기록한다.
대조군과 일반 회귀 질의의 `clue_category`는 `no_context`, 사실 ID는 빈 칸이다.
이 CSV는 공개 레포에 커밋하거나 ZIP으로 공유하지 않는다.

| `group` | 검증할 질의 예 |
| --- | --- |
| `context_context` | 작품/인물·매체에서 쓰임, 제작/뮤비/밈/버전/방송 사실, 동의어·별칭·모호한 회상 |
| `context_mixed` | 사실 + 가수/장르/소리/이미지, 두 사실의 결합, 잘못 기억한 세부사항 |
| `context_control` | 사실 없는 곡의 일반 제목/가수 단서 |
| `regression_lyrics` | 부분/틀린/의미 가사 |
| `regression_mood_sound` | 분위기·악기·보컬 질감 |
| `regression_image` | 앨범아트 색·형상 (뮤비 장면과 구별) |
| `regression_other` | 제목/시기/가수, 혼합 구문 |

여섯 필수 그룹은 각각 세 문항 이상, 사실 category는 여섯 종류 이상 포함해야
한다. 이전 106문항과 질의 ID·정답 곡을 공유하지 못하게 검사한다. 작품 이름
오타·별칭, 한영 혼용, 부정/오정보, 짧은 회상, 복수 사실, 맥락 없는 곡 등은
`review_note`에 유형과 정답 선정 이유를 남긴다. `-Mode All`은 이 파일이 있으면
**dev 선택 전에 파일 해시를 등록**한다. 평가 후 파일을 바꾸거나 나중에 등록하면
기존 dev/test 체크포인트를 재사용할 수 없다. 새 결과 디렉터리에서 처음부터
실행해야 한다. 문장 형태는 무한하므로 이 표는
공백을 드러내는 표본 설계일 뿐 모든 경우의 수를 보장하지 않는다.

dev/test가 `provisional`이고 이 CSV가 준비되면 고정된 설정을 **한 번만** 검증한다:

```powershell
.\experiments\namuwiki\run_context_step10.ps1 -Mode Blind
```

`artifacts/context_eval_step10_v2/blind_report.md`에서 어떤 그룹의 Hit@10 또는
Candidate@30이 개선됐고 기존 질의 손실이 있는지 읽는다. 손실이 있으면
`not_recommended`이며 결과를 보면서 해당 CSV에 맞춰 재조정하지 않는다.
`recommended`이더라도 `displayed_evidence`의 사실 문장과 출처가 질의와 실제로
부합하는지 사람이 확인해야 한다. 코드가 검사하는 정답 fact ID 일치는 출처의
진실성을 보증하지 않는다.

권고안이 최종 확정되면 보고서의 Context 가중치·명시 작품 배율·검색 폭·후보 폭을 로컬 `.env`에 반영하고
`docker compose up -d --force-recreate backend` 후 `/health`가 준비됐는지
확인한다. 기존 `.env` 값이나 팀 전체 설정을 평가 스크립트가 자동 변경하지 않는다.
화면 `top_k=10`은 기존 기본값으로 유지한다. 보고서에서 5/10/20곡을 표시했을 때의
정답 포함 비율을 비교하지만, 후보의 앞 20곡을 계산한 값만으로 화면 이용성과
실제 응답 지연을 증명할 수는 없다. 최적 화면 폭은 독립 사용자 테스트 뒤 결정한다.
