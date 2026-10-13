# Context 고정 설정 재평가와 새 독립 평가

이 절차는 후보 절단·근거 차단 수정 후의 검증이다. 이미 결과를 확인한 실패 문항에 맞춰 가중치를 조정하지 않는다. 이전 단계의 grid 탐색이나 dev 선택 함수를 호출하지 않으며 `.env`를 수정하지 않는다.

| 설정 | 고정값 |
|---|---:|
| Context 외부 RRF 가중치 | 0.5 |
| Dense 사실 검색 폭 / Sparse 프로필 검색 폭 | 100 / 100 |
| 최종 후보 / 화면 표시 | 30 / 10 |
| 기존 작품 단서 배수 | 2.0 |
| 리랭킹 | OFF |

Text/Image/Audio의 질의별 가중치와 전용 문장은 새 QueryAnalyzer 결과를 사용한다. 이미지 경로 실행을 통과시키려고 가중치나 문장을 강제로 덮어쓰지 않는다. 각 문항의 Context OFF/ON에는 동일한 분석 객체를 전달하고, 라우터가 그 객체를 변경하면 중단한다.

## 평가 순서와 새 문항

1. 기존 dev 74문항과 test 32문항을 각각 새 분석으로 재측정한다. 둘 다 같은 코드·코퍼스·설정을 사용한다. 기존 분할은 공개된 회귀 평가다.
2. 기존 평가가 통과하면 새 독립 질의 60문항을 측정한다. 이전 Context/v0.5/공개된 blind 평가의 정답 곡 126개를 제외했으며, 새 문항끼리도 정답 곡을 중복하지 않았다. 새 문항을 선택할 때 검색 결과를 조회하지 않았다.
3. 사전 지정한 10문항에 추가 분석을 한 번씩 수행한다. CSV 순서상 처음 두 개의 Context, mixed, 가사, 소리, 이미지 문항이다. 분석·후보 순위 변화와 추가 근거를 기록하며, 좋은 결과만 고르거나 독립 평가의 분모에 더하지 않는다.

| 새 문항 종류 | 수 | 라벨 작성 근거 |
|---|---:|---|
| 배경지식 | 24 | 현재 곡에 귀속된 artifact의 사건·record_id·원문·출처 |
| 배경지식과 다른 단서 혼합 | 12 | 같은 사건과 제공된 곡 메타데이터의 추가 단서 |
| 배경 사실 없는 대조군 | 6 | 사실 0개인 not_found/no_trivia 곡의 정식 제목·가수 |
| 가사 회귀 | 6 | 제공된 Text payload의 실제 가사 구절 |
| 소리 회귀 | 6 | 정식 곡 식별 정보와 sound_tags의 명시적 청각 단서 |
| 이미지 회귀 | 6 | 프로젝트가 연결한 Melon 앨범 표지를 직접 열어 확인한 묘사 |

배경 사건은 작품 사용·제작 비화·뮤직비디오·공연·버전/편곡·음악적 특징·영향·기록·밈의 9종류를 포함한다. 소리 문항은 실제 Audio 실행과 결과 유지 검사이며, 곡 식별 정보도 포함하므로 소리만으로 곡을 맞히는 능력을 입증하지 않는다. 소리 태그의 파형 청취 검증은 하지 않았다. 표지는 가사에 등장하는 장면을 묘사한 `visual_imagery` 대신 실제 이미지를 확인했다. 문항·정답·검토 노트·표지 출처와 이미지 바이트 해시는 ZIP의 `data/context/`에 제공된다.

## 실행

변경 ZIP을 새 프로젝트 루트에 풀고 PowerShell에서 아래 명령을 실행한다. 기존 코드에는 최신 후보 절단·근거 차단 수정이 적용되어 있어야 한다. 기존 평가 CSV/JSON, 공개된 blind CSV, 최종 Qdrant 적재 기록과 실제 Text/Image/Audio/Context 인덱스가 필요하다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
Unblock-File .\experiments\namuwiki\run_context_fixed_validation.ps1
.\experiments\namuwiki\run_context_fixed_validation.ps1
```

스크립트가 backend를 빌드하고 `tests/context`, 재질문 상태, 검색 설명 테스트를 실행한다. 라벨·설정 등록 후 backend를 중지하여 로컬 Qdrant 잠금을 해제하고 평가를 실행한다. 평가 성공·실패 여부와 관계없이 결과를 내보내고 backend를 다시 시작한다. 시작 직후에는 기존 warmup 시간이 필요할 수 있다.

순서는 dev → test → 새 60문항 → 추가 분석 10문항이다. 기존 평가에서 회귀나 실제 이미지 경로 검증 공백이 나오면 새 60문항 검색 전 중단한다. 각 단계의 분석은 이 실험에 등록된 새 캐시에서만 생성한다. 이전 실험의 캐시를 이름만 바꿔 넣으면 거부한다. API fallback도 정상 분석으로 집계하지 않는다.

중간 API 오류라면 설정·코드·데이터를 변경하지 않고 같은 명령을 재실행할 수 있다. 완료된 질의 쌍과 분석을 보존하여 이어서 진행한다. 측정 후 문제를 수정했다면 그 결과는 공개된 진단으로 남기고, 다른 `-OutputDir artifacts/새이름`에서 회귀 재측정한다. 그때 이미 확인한 60문항을 새로운 독립 평가라고 부르면 안 된다.

## 측정과 판정

`Candidate@30`, `Hit@1/5/10`, `MRR@10`의 OFF/ON 값과 질문별 순위를 기록한다. 배경지식 개선의 주 분모는 positive 36문항이며, 쉬운 대조군 6문항은 따로 보고한다. Context 사건 종류별 결과, 개선·악화 ID, 질의별 분석, 대상 곡의 검색 경로 기여, 실제 경로 구간과 오류, 표시된 fact_text/출처/라벨 일치도 함께 남긴다.

이미지와 소리는 `path.image/audio` 진입에 더해 실제 `image/audio.embed`와 `image/audio.query`가 존재하고 오류가 없어야 한다. 기존 이미지 그룹과 새 이미지·소리 6문항 각각의 OFF/ON 양쪽을 검사한다. 실행 검증은 해당 정답이 높은 순위에 있다는 뜻이 아니다. 실제 순위와 Hit@10은 별도 지표다.

Context 단서 누락, 대조군·회귀 질의의 잘못된 Context 단서, OFF에서의 Context 실행, 누락된 정답, 리랭킹 실행, 검색 경로 오류, Top-10 악화는 통과하지 못한다. 대조군·회귀 문항은 Candidate@30, Hit@1/5와 MRR@10 악화도 검사한다. `nw001`, `q203`의 Candidate@30 포함을 계속 요구한다. 새 60문항에서는 positive Hit@10의 양의 변화와 Candidate@30 비악화를 요구한다.

표시 근거가 검토된 정답 사건 라벨과 일치하지 않으면 `evidence_review_required`로 남긴다. 이를 자동으로 거짓 사실이라 단정하거나 근거 정확도 100%라고 계산하지 않는다. 추가 분석 10문항에서 새로 나타난 근거도 같은 검토 대상으로 포함한다.

| 최종 상태 | 해석 |
|---|---|
| blocked_before_independent | 기존 회귀·경로 검사 실패. 새 독립 검색 미실행 |
| interrupted | API/분석/인덱스/상태 문제로 중단. 완료분은 보관 |
| blocked | 지표·경로·단서·재분석 검사 실패 |
| evidence_review_required | 측정 완료, 라벨 밖 표시 근거의 추가 검토 필요 |
| passed_observed_tests | 이 고정 설정과 이번 질의에서 사전 지정 검사가 통과 |
| partial | 요청한 일부 단계만 완료 |

paired exact p와 질문 단위 bootstrap 95% 변화 구간도 탐색적으로 제공한다. 소수 개선이면 `positive_observed_change_with_limited_statistical_support`로 표시한다. 통과 상태가 유의한 개선이나 실제 사용자의 모든 질문에서의 성능을 자동으로 입증하지 않는다. 새 문항은 코퍼스에 근거해 작성한 테스트이며 사용자 질문의 무작위 표본이 아니다. 원문 내용의 진위나 화면 표시 수 10의 최적값을 검증한 실험도 아니다. 검색 시간은 QueryAnalyzer/API 시간을 제외하며, 초기 모델 적재가 포함될 수 있다.

실행 전후 active Context build/alias/point 수/manifest를 검사하고, 로컬 Qdrant의 Text·Image·Audio·Context SQLite 바이트 해시를 대조한다. 기존 Text BM25가 정상 적재되어 있는지도 검사하며 파라미터 파일 해시와 `n_docs`를 기록한다. 환경 설정, Python·핵심 패키지 버전, 전체 `src/` 코드와 입력 해시를 등록하여 다른 실행을 섞지 않는다. 원격 Qdrant에서는 파일 해시를 사용할 수 없어 alias·point 수·manifest 확인 범위로 제한된다.

## 결과를 보내는 방법

콘솔 전체를 복사할 필요 없이 아래 파일 하나를 보내면 된다.

```text
artifacts/context_fixed_validation_20261005_feedback.zip
```

ZIP에는 등록 정보, 새 질의와 검토 노트, 완료된 분할의 JSON/CSV, 종합 Markdown/JSON, 추가 분석 결과와 중단 오류가 들어간다. 이 실행의 분석 캐시와 체크포인트도 포함하여 완료분과 중단 원인을 확인할 수 있다. 임베딩·음원·모델 파일·`.env`는 내보내지 않는다. `validation_report.md`로 지표와 상태를 직접 확인할 수 있다. 보고서와 `data/context/`의 실제 질의·정답은 기존처럼 개인 데이터로 보관하고 public PR에는 코드·테스트·이 문서만 추가한다.
