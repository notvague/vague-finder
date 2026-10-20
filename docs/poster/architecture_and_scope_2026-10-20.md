# 문제 정의 · 현재 구조 · 검증 범위 — 포스터 P0-2 (2026-10-20)

기준: main `26dec55`(PR #95 병합), 워크트리 `/private/tmp/vague-finder-poster-evidence`. 아래 내용은 **현재 코드와 산출물 설정에서 직접 확인**한 것이며 `.claude/PROJECT_BRIEF.md`의 옛 계획(e5-large-instruct, Cross-Encoder 기본, 50곡·20질의 평가)은 가져오지 않았다. 수치는 [근거표](evidence_table_2026-10-20.md)만 쓴다. 새 검색·모델 호출 없음.

산출물: 이 문서 + 구조도 `figures/fig3_architecture.svg`(`make_architecture_figure.py`).

갱신 2026-10-20(README 정리): main `7bfba98`(#96, 문서 정리, `src`·`tests`는 `26dec55`와 같음). Context 코드 기본값을 0.5로 정정하고 피드백·v76/v78 행을 §3에 더했다. 구조도의 "서비스 기본 구성" 표기는 코드 기본값 기준이다. 후속 통합 검토에서 질의당 리랭킹 1회라는 단정을 묶음별 호출로 정정하고 생성 스크립트와 그림을 함께 갱신했다.

## 1. 문제 정의 — 범위와 피치(확정 문안, 2026-10-17 사용자 결정)

**피치 문장(그대로 쓴다):** "흥얼거릴 수 있는 노래는 이미 찾을 수 있습니다. Vague-Finder는 멜로디조차 떠오르지 않고 뮤비 장면, 들었던 시기, 분위기 같은 조각만 남았을 때를 위한 검색입니다."

| 범위 안 | 범위 밖(한계가 아니라 기획의 출발점) |
|---|---|
| 입력: 자연어로 쓴 기억 단서 한 문장(가사 조각·음차, 분위기·상황, 표지 인상, 소리 인상, 시기, 보컬 성별, 장르, 제목 형태, 드라마·사건 같은 배경) | 허밍·멜로디·음원 입력(기존 음악 인식 서비스 영역) |
| 대상: 한국 대중음악 5,143곡(2026-10-11 확장 뒤) | 해외 곡, 5,143곡 밖 |
| 출력: 후보 15곡(화면) + 선정 근거 + 노래 맵 위치, 없으면 후보 상태·대화 한도에 따라 재질문(보컬 성별 또는 장르) 뒤 재검색 | 정답 하나를 단정하는 답변 |

**한계 세 가지(확정 문안)와 코드 대조.** 포스터에는 성능 한계(dev 재사용·독립 test 없음·70% 미달·평가기 조건)와 별개로 함께 싣는다.

| 확정 문안 | 코드·산출물 대조 결과 |
|---|---|
| ① **(2026-10-20 교체 확정)** "소리 단서는 음원 앞 120초의 CLAP 임베딩과, 가사·앨범 소개·댓글에서 뽑은 소리 태그 두 자료로 맞춘다. 음원 경로의 기여는 작고 불안정해 보류 구성이 생겼다." — 옛 문안: "소리 단서는 음원을 들은 결과가 아니다. 가사·앨범 소개·댓글에서 뽑은 간접 자료다." | **옛 문안은 사실과 어긋나며, 교체한 문안은 코드와 일치한다.** 소리 경로(`_search_audio`, `search_router.py:3704`)는 CLAP(`laion/clap-htsat-fused`)으로 **실제 음원 `audio.m4a` 앞 120초**(`audio_io.py:26`, `AUDIO_MAX_SECONDS=120`)의 구간 평균 벡터 5,143곡(`artifacts/audio_mean_manifest.json` `num_tracks: 5143`)과 질의의 소리 영어 문장을 대조한다. 간접 자료(LLM 소리 태그 `sound_tags`)는 텍스트 경로가 쓴다. 다만 음원 경로의 기여는 작고 방향이 불안정하다(끄면 Hit@10 +2 · Hit@1 −5, v75). **사용자 결정(2026-10-20): 왼쪽의 새 문안으로 교체. NEXT_WORK §2 확정 문안도 같이 고쳤다.** |
| ② "표지 검색은 한계가 있다. 표지 묘사만으로 찾는 질의는 리랭킹을 건너뛰고 후보 순서를 그대로 낸다(표지 지배 정책)." | 일치. `rerank_policy.py` `RERANK_SKIP_IMAGE_DOMINANT`. dev 176에서 정책 생략(표지 지배 + MV 장면) 21건, 그대로 후보 순서. 표지 경로는 `has_visual_clue`가 참일 때만 돈다(`schemas/query.py` `has_visual_clue`, `_search_image:3677`) |
| ③ "작품·사건 같은 배경 사실(Context, 나무위키)은 5,143곡 중 639곡에만 있다." | 일치. `artifacts/context/manifest.json`: 범위 5,139곡(메타 불완전 4곡 제외) 중 `ok` 639 · `no_trivia` 1,082 · `not_found` 3,418, 사실 5,054개. 문장은 "5,143곡 중 639곡"으로 두어도 된다(차이 4곡은 범위 밖) |

## 2. 현재 구조 — 단계표(그림 3의 근거)

요청 하나의 경로. 생성형 호출은 Vertex AI Gemini `gemini-3.1-flash-lite` 2회(질의 분석 1 · 리랭킹 1, 정책 생략이면 1)뿐이다. 모델명·치수는 `artifacts/qdrant/meta.json`과 `artifacts/embeddings/` 폴더명으로 확인했다(클래스 이름이 아니라 산출물 설정).

| 단계 | 무엇을 하나 | 모델 · 저장소 · 설정값 | 코드 | 확인한 산출물 |
|---|---|---|---|---|
| ① 입력 | 질의 원문 + 선택: 재질문 답, 거절 곡 id, 출생 연도, `top_k`(API 기본 10, 화면 15), `candidate_k`(화면 30 고정) | FastAPI `POST /api/v1/search`, `/api/v1/search/clarify`, `/api/v1/feedback` | `src/backend/api/routes/search.py:284`, `schemas/search.py:106`, `frontend/static/map.js:757-760` | — |
| ② 질의 분석 | 원문을 단서 구조(JSON 스키마)로 나눈다: 의도, 한국어 태그, 가사 단서(verbatim/partial/phonetic/semantic), 제목·가수·제목 구조, 보컬 성별·장르·시기, 표지 영어 질의, 소리 영어 질의, 배경 사실 단서, 모달리티 가중. 실패하면 규칙 폴백(`confidence=0`) | Gemini 3.1 flash-lite(Vertex), 전체 20초 | `src/retrieval/query_analyzer.py`(폴백 :814), 스키마 `src/backend/schemas/query.py:245` | 평가는 고정 분석 캐시 사용 |
| ③-a 가사 표면 일치 | 기억한 구절을 가사 원문에서 문자 순서 보존 exact/fuzzy로 찾는다. exact 1위는 보호 대상 | MongoDB `full_lyrics`, 메모리 판(TTL·백그라운드 갱신) | `lyrics_exact_search.py`, `_search_lyrics:3660`, `select_protected_lyric_ids:1285` | — |
| ③-b 텍스트 하이브리드 | BM25(Kiwi 형태소, 입력은 모델이 뽑은 한국어 태그) + KoE5 dense, `text_alpha`로 비중 | `nlpai-lab/KoE5` 1024(Dot) + BM25 sparse, Qdrant `vaguefinder-text-1024-koe5` | `_search_text:3325`, `_text_sparse_source:1225` | `artifacts/embeddings/text_dense/nlpai-lab__KoE5`, `text_sparse` |
| ③-c 앨범 표지 | 표지 영어 질의를 텍스트→이미지 공간으로 임베딩해 표지 벡터와 대조. 시각 단서가 없으면 경로 0 | `google/siglip2-base-patch16-224` 768(Cosine) | `_search_image:3677` | `artifacts/embeddings/image/google__siglip2-base-patch16-224`, 컬렉션 `vaguefinder-image-768` |
| ③-d 소리 | 소리 영어 질의를 텍스트→오디오 공간으로 임베딩해 **음원 앞 120초 구간 평균** 벡터(센터링)와 대조. 제목·가사·가수·표지 단서가 전혀 없고 소리만 상세하면 깊은 융합(deep audio fusion) | `laion/clap-htsat-fused` 512(Cosine), `AUDIO_FUSION_SCALE=1` | `_search_audio:3704`, 조건 `use_deep_audio_fusion:1948`, `src/embedding/audio/audio_io.py:26` | `audio_mean_manifest.json` `num_tracks 5143`, `audio_mean.npy` |
| ③-e 배경 사실 Context | 배경 단서(드라마·사건·수상 등)를 사실 단위 dense + 곡 단위 sparse로 찾아 곡으로 묶는다(곡별 max, RRF) | KoE5 dense(사실 5,054) + BM25 sparse(639곡), `CONTEXT_WEIGHT=0.5`(코드 기본값 = 평가값, `context_search_settings.py:14`) | `_search_context:3268`, `context_route.py`, `context_ranking.py`, `context_startup.py` | `artifacts/context/manifest.json`, 별칭 세대 `f6424b94621a28cf` |
| ③-f 제목·속성 보조 | 제목 포함/구조/의미 3경로, 수행 단서(라이브·피처링 등), 성별·장르·시기 메타 필터 경로 | Qdrant payload 필터 | `_search_title_*:3489-3660`, `_search_performance_*:3404-3488` | — |
| ③-x 기본 OFF 실험 경로 | 작품명 해석(간접 지칭 → Vertex 해석 → 검증 → 보조 후보 ≤10곡) · 속성 조건 보조 · 정답 피드백 보충. 미채택: 앨범 소개 낱말/의미 보조(v76·v78, 패치로만 보존) | `WORK_NAME_RESOLUTION_ENABLED` 기본 꺼짐, `ATTR_FILTER_WEIGHT=0`, `FEEDBACK_*=0` | `work_name_resolution.py:121`, `_apply_work_name_aux:2789`, `_apply_feedback_supplement:2882` | 평가에서 작품명 ON = 개발 비교 기준 |
| ④ 융합·보정 | 경로별 검색 결과를 RRF(k=60)로 합치고 가산(가사 표면·제목·성별·장르·시기·가수 유형 일치) → 거절 곡 제외 → 후보 30 절단 → 재질문 답 가산(질의는 바꾸지 않음) → 가사 exact 보호 지정 | `_RRF_K=60`, `candidate_k = max(top_k×3, 30)` | `_rrf_fuse:2124`, `_fuse_add:2175-2368`, `_reject_and_trim:2420`, 답 가산 `:2440/2463` | — |
| ⑤ 리랭킹 | 최대 30곡의 후보를 비교해 순서를 다시 정한다(listwise 1패스, Google 검색 끔). 가사 보호 후보는 제약 점수별 묶음과 나머지 후보를 나누어 재정렬할 수 있어 질의당 호출 수는 고정 1회가 아니다(`search_router.py:2595-2655`). 정책 생략 2종(표지 지배·자료 없는 MV 장면)은 모델을 부르지 않는다. 실패·시간 초과면 검색 순서 유지(`rerank_outcome=failed`) | Gemini 3.1 flash-lite, `GEMINI_RERANK_MAX_CANDIDATES=30`, 운영 호출 20초·예산 30초(평가 60/120). 선택 백엔드 Cross-Encoder `dragonkue/bge-reranker-v2-m3-ko` | `rerank_policy.py`, `call_reranker:940`, `gemini_listwise_reranker.py:256,297`, `place_protected_lyrics_first:1320` | dev 176: applied 155 · policy_skipped 21 · failed 0 |
| ⑥ 응답 | 상위 곡 + 선정 근거 패널(템플릿, 모델·DB 호출 없음: 단서별 일치·순위 반영 기록·가사 인용) + 노래 맵(CLAP·KoE5 2D) 표시, YouTube 임베드, 정답 피드백 버튼 | — | `user_explanation.py`, `explain.py`, `frontend/static/map.js` | `map_data.json`(Git 제외) |
| 재질문(2턴) | "이 중에 없어요" → 현재 후보로 보컬 성별·장르 중 하나를 고른다(출생 연도는 시기 창으로 1회). 답은 분석에 병합, 거절 곡 제외 뒤 같은 경로로 재검색 | — | `clarify.py` | 측정 `results_clarify_*` |

**자료층.** 곡 레코드 `data/all_songs.jsonl` 5,143줄(멜론 메타·앨범 소개·플레이리스트 태그, 가사 원문, LLM 태그, 댓글 요약·인기도, 나무위키 사실). 저장소에는 원본·벡터·곡 목록이 없다(저작권, `docs/data_policy.md`). 수집 정제·댓글 선별·Context 추출은 `gemini-3.5-flash-lite`로 오프라인에서 했다.

**PROJECT_BRIEF·README와 다른 점(포스터에 옛 값을 쓰지 않기 위한 목록).**

| 항목 | 옛 문서 | 현재 |
|---|---|---|
| 텍스트 의미 모델 | `multilingual-e5-large-instruct` | `nlpai-lab/KoE5` |
| 리랭킹 | Cross-Encoder 기본 | Gemini listwise 기본, CE는 선택 백엔드 |
| 평가 규모 | 50곡 평가셋 + 질의 20개 | 5,143곡 · dev 176(+ 사용한 test 105) |
| 코퍼스 | README 배지 3,010곡 | 5,143곡(README는 2026-10-20 같은 워크트리에서 갱신, §5) |
| 생성형 호출 | AI Studio 키 | Vertex AI 전용(2026-10-15) |
| 작품명 해석·Context | 없음 | Context는 기본 경로(가중 0.5), 작품명은 기본 OFF 실험 경로 |

## 3. 구현 / 평가 / 미검증 범위

| 항목 | 구현 | 평가한 것(어디서 · 얼마나) | 미검증·모름 |
|---|---|---|---|
| 질의 분석 | Vertex Gemini, 스키마 검증, 규칙 폴백 | 측정은 **고정 분석 캐시**로 했다(분석 변동을 뺀 비교). 프롬프트 한 줄 변경이 dev 49/57을 바꾼 관측(v36) | 분석 자체의 정확도(단서 추출 정답률) 미측정 |
| 후보 회수(③+④) | 경로 6종 + RRF + 가산 | dev 176 후보 회수@30: 122(A) · 125(B) · 130(C). 후보 밖 실패 51의 원인 분류(v74) | 경로별 ablation은 소리(v67~v69·v75)·Context 후보(v50)·표지 진단(v25)만. 텍스트·가사 경로 단독 기여 미측정 |
| 리랭킹 | Gemini listwise + 정책 생략 + 실패 폴백 + 예산 공유 | dev 176 실제 Vertex, 구성 3종 × 2회(v73·v75), 실패 0. 리랭킹 효율 121/125 | 운영 20/30초 조건의 품질·실패율(측정은 60/120초). CE 대비는 3,010곡·AI Studio 시기(v22 vs v32~v35)라 현재 판에서 재비교 없음 |
| 정책 생략 | 표지 지배 · MV 장면 미검증 | dev 176에서 21건 생략(표지 지배). | MV 장면 생략은 dev 대상 0건이라 효과 미측정. 낱말 경계 수정 ①·②만 반영(③ 보류) |
| 재질문 2턴 | 성별·장르 슬롯, 출생 연도, 거절 재검색 | AI Studio 시기 dev(`results_clarify_v09`)·test(`results_clarify_test_v06`). 속성 실험에서 Vertex 2턴 18건 +2/−0(v54) | **Vertex 전환 뒤 전체 재질문 재측정 없음**. 현재 구성 B에서의 2턴 수치 없음 |
| 작품명 해석(기본 OFF) | 판정·해석·검증·보조 후보 | dev 176 +4/−0(v73), 입력 변경 13질의 5→9(v72), 해석 라벨 정답 13/16(PM 단독+AI 보조) | 실시간 해석 품질·cold 시간 초과(기동 직후), 운영 시간 제한에서의 동작 |
| 소리 경로 생략 | 스위치만 | dev 176 +3/−1, Hit@1 −5 → `hold_mixed`(v75) | 채택 여부는 제품 지표 우선순위 결정 뒤, 새 자료에서 |
| 정답 피드백(수집 · 승인 사례 색인 · 후보 보충) | 세 스위치 모두 코드 기본 꺼짐(`FEEDBACK_COLLECTION_ENABLED` · `FEEDBACK_INDEX_WORKER_ENABLED` · `FEEDBACK_RETRIEVAL_ENABLED`). 이 PC `.env`는 수집만 1 | off/on 비교 도구 · 합성 자료 시험(`docs/feedback.md` §8) | 실제 승인 자료로 잰 품질, 기존 dev에서의 off/on 손실·지연(`docs/feedback.md` "하지 않은 것") |
| 앨범 소개 원문 보조(v76 낱말 · v78 의미) | **main에 코드 없음** — 결과 폴더와 재현 패치만(PR #91 · #95) | 후보 단계 오프라인 A/B(v76 125→124, v78 125→129·n019 탈락 → G2 거짓) | 실제 리랭킹 최종 순위 미측정 · 보류 |
| 선정 근거 패널 | 템플릿 기반, 모델 호출 없음(PR #52) | 단위 시험 | 브라우저 실확인·사용자 이해도 평가 없음 |
| 노래 맵·프론트 | 2D 지도, 결과 표시, 재생 임베드 | — | 사용성 평가 없음 |
| 서버·배포 | FastAPI, Qdrant 서버/로컬, Docker, 기동 검사(Vertex·Context) | HTTP 경로 격리 검증(v64, 모델 0회) · 서버 모드 Vertex 기준선 117(v66) · 로컬↔서버 후보 174/176 일치(v65) | HTTP 전체 응답 시간(현재 구성), CPU/Docker 성능, 무중단 적재. 실행 중인 서버 프로세스의 실제 스위치 값(`.env`·코드 기본값으로 단정하지 않음, 근거표 §2) |
| 평가 자료 | dev 176(v06 57·v09 59·v10 60), test 105 | test는 모두 사용됨: v06 test 25(CE, 3,010곡), v09 test 38(listwise, 3,010곡·AI Studio), v10 test 42(속성 실험 A=B=D 22/42, Vertex) | **현재 구성 B의 봉인 test 수치는 없다.** 독립 자료 v11은 준비만(수집 0건) |
| 일반화·사용자 | — | — | 실사용자 정확도, 통계적 유의성, dev 밖 일반화 모두 주장하지 않음 |

## 4. 포스터에 넣을 구조 설명 문안(초안)

- 한 줄: "질의 한 문장을 Gemini가 단서 구조로 나누고, 가사·텍스트·표지·소리·배경 사실 여섯 경로를 병렬로 돌려 RRF로 합친 뒤, 최대 30곡의 후보를 Gemini가 다시 정렬한다(가사 보호 시 묶음별). 결과에는 어떤 단서가 어디서 확인됐는지 근거를 붙이고, 없으면 후보 상태와 대화 한도에 따라 성별·장르를 되묻는다."
- 모델 한 줄: "KoE5 · BM25 · SigLIP2 · CLAP · Gemini 3.1 flash-lite(Vertex AI) · Qdrant · MongoDB".
- 수치는 근거표 §4의 표와 그림 1·2만 쓴다.

## 5. 정합성 확인에서 나온 결정 요청(사용자)

1. ~~한계 ① 문안 수정 여부~~ — **결정됨(2026-10-20): §1의 새 문안으로 교체.**
2. ~~README 배지·본문의 3,010곡 표기와 성능 표(v32~v35, AI Studio)는 옛 값이다. 갱신할지.~~ — **반영(2026-10-20):** 같은 워크트리의 `README.md`를 5,143곡·dev 176·Vertex 최신 측정(v73·v75)으로 바꾸고, 옛 표는 "과거 측정 — 3,010곡 · AI Studio"로 분리했다. 기능 적용 상태 표를 더했다.
3. ~~이 PC `.env`에 `CONTEXT_WEIGHT=0.5` 추가(시연 전).~~ — **철회(2026-10-20):** 코드 기본값이 0.5라 필수 조치가 아니다(근거표 §2 정정). 시연 때는 `/health`의 `context`로 기동 상태만 확인한다.
