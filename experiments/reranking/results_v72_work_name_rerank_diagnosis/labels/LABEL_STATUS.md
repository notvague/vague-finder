# 작품 라벨 검토 상태와 대기 항목(v72, 2026-10-18 13:59 KST 확인)

## 상태

| 항목 | 값 |
|---|---|
| 사람 라벨 | **0/22** — v71 대기표(`results_v71_work_name_resolution_live/labels/work_label_queue.csv`, 2026-10-18 13:09 KST 작성)의 라벨 열이 모두 비어 있다. 원문판(`work_label_queue_private.csv`, Git 제외)도 0/22 |
| 라벨러·확인자 | **미지정**(PM이 정한다) |
| 해석 정확도(정답·오답·보류율) | **미측정**. 자동으로 셀 수 있는 것은 보류 5/22 · 검증 거부 1/22뿐 |
| 에이전트·모델 라벨 | 쓰지 않는다. 에이전트는 v71·v72 세션에서 모델 출력을 봤으므로 라벨러가 될 수 없다 |

v72 리랭킹 결과는 라벨을 기다리지 않고 진행했다(승인 범위). 라벨이 없으므로 v72 결론에는 "해석 정확도 미측정"이 붙는다.

## 라벨러에게 줄 자료(모델 출력을 보기 전)

1. v71 `labels/LABEL_GUIDE.md` — 열 정의와 규칙. 질의 의도 · 작품명 · 여러 작품 가능 · 판단 불가를 **먼저** 적는다.
2. v71 `labels/work_label_queue_private.csv` — 질의 원문이 든 22행(Git 제외 파일이라 v71 워크트리 `/private/tmp/vague-finder-v71-work-name-live/…/labels/`에 있다. 보관자가 직접 전달한다).

주지 말 것: v71 `live/`·`resolution_cache/`·`analysis_private/`, v72 `resolution_cache/`·`labels_private/`·`live/`·`analysis/`·이 폴더의 `review_after_labels.csv`, RUN_INFO의 결과 절. v72 결과(어느 질의가 10위 안에 들었는지)도 라벨을 끌어당길 수 있으므로 라벨이 끝날 때까지 보이지 않는다.

- 정답 곡이 그 작품에 쓰였다는 이유만으로 사용자가 그 작품을 뜻했다고 단정하지 않는다(질의 설명과 작품이 맞는지를 본다).
- 이미 모델 출력을 본 사람은 `saw_model_output_before=yes` — 독립 라벨로 세지 않는다.

## 라벨 뒤 대조(라벨을 다 채운 다음에만 연다)

- `review_after_labels.csv`(추적, 질의 id·상태·표시만): 22행. 라벨 열을 대기표에서 옮겨 적고(`label_*_copied`), 검토자가 `model_works_match_label`(yes · no · partial · undeterminable) · `wrong_work_link`(yes · no)를 채운다.
- `../labels_private/review_after_labels_private.csv`(Git 제외): 같은 22행 + v71 반복 1의 모델 작품 후보·채택 여부·거부 사유. 대조는 이 파일로 한다.
- 표시(`review_flags`) — 빠뜨리지 않을 사례:
  - `name_in_query`: **q105 · n031** — 질의 낱말과 같은 이름이 채택됐다(v3 위험). q105는 채택이지만 리랭커 입력이 A와 같다(보조 곡이 후보 30 밖). n031은 입력이 바뀐 13건에 든다.
  - `v72_rerank_input_changed`: 리랭킹 결과가 달라질 수 있는 13건. 이 중 이득·손실 질의는 v72 결과와 함께 본다.
  - `applied_input_identical`: n045 · q105 · t024 — 채택이지만 리랭커 입력이 같다(효과 0).
  - `work_name_equals_song_title`: n096 — 작품명이 곡 제목과 같다. 이득이 있어도 작품 의도 해석의 근거로 따로 본다.
  - `abstained`(c603 · n010 · t036 · t041 · t098) · `validation_rejected`(t043).

## 라벨이 오면 v72 판단에 쓰는 방법(실행 전 RUN_INFO §4-6 그대로)

- 안정 이득 질의에서 잘못된 작품 연결(`wrong_work_link=yes`)이 확인되면 그 이득은 의도 해석의 근거로 세지 않는다(검색 결과 이득 자체는 기록대로 둔다).
- 라벨러가 판단 불가로 적은 질의는 정답·오답 어느 쪽으로도 세지 않는다.
- 알려진 10건 밖의 12건을 오답·음성으로 치지 않는다.
