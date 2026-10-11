# results_v38 — v10(3차) 질의 세트 개발용 60건 첫 측정 (2026-10-11)

추가한 v10 질의로 **첫 검색에서 정답을 얼마나 찾는지** 한 번 잰 진단이다. 설정 비교가 아니고, 고친 것도 없다.

- 워크트리 `../vague-finder-v10-eval` · 브랜치 `eval/v10-first-run` (main `1ea2bd5`에서 분기, 색인은 `artifacts/` 복사본)
- 코퍼스 5,143곡 — Qdrant 5,143 · BM25 `bf606b19ab02` · 가사 DB 5,143(`e0d58f28f1f4a563`)
- 리랭커 Gemini listwise 1패스(`gemini-3.1-flash-lite`, Search 끔, 희소 사실 검증 끔) · 호출 경로 `api_key`(AI Studio) · `SEARCH_REFERENCE_YEAR=2026`
- 분석 캐시 `analysis_cache_v10_dev.json` — prompt `08f0b3bc4c37` · postprocess `8ef688eceb3d`
- **1회 측정**이다. listwise는 실행마다 ±1 흔들린다(반복 측정은 아직 안 했다)

## 1. 질의 세트와 분할

`data/eval_drafts/queries_v10_draft.csv`(111건, sha256 `7cbb1b227090ca15`, git 제외) → `make_split.py`

| | 건수 |
|---|---|
| 초안 | 111 |
| 검토 '제외' | −6 (cc012·015·017·019·034·064) |
| 정답 미지정(no_target) | −3 (cc055·056·057) |
| **측정 대상** | **102** — 정답 곡 전부 5,143곡 안에 있음 |
| 개발용(dev) / 최종 확인용(test) | **60 / 42** |

층: (정답이 기존 3,010곡 · 확장 크롤링 곡) × modality_focus 8칸에 dev 몫을 최대잉여법으로 배정, 칸 안에서 tier·target_scope가 한쪽에 몰리지 않게 뽑았다(씨앗 20261011).

| | dev / test |
|---|---|
| 확장 곡 · 기존 3,010곡 | 40/28 · 20/14 |
| text · multimodal · audio · image | 30/21 · 11/8 · 10/6 · 9/7 |
| 틀린 단서(misinformation) | 3/2 |
| 범주형(categorical) | 2/2 |

- 측정 CSV `experiments/reranking/eval_queries_v10.csv`(102행). query_id는 초안 번호 그대로 `t###`(cc017 → t017)
- 분할표 `split_v10.csv`(속성만) · 뺀 목록 `split_v10_dropped.csv`
- **test 42건은 재지 않았다.** 최종 설정 하나로 한 번만 잰다(v09 test 38과 같은 규칙)
- `docs/eval/queries.json`(0.6.0)에는 **아직 병합하지 않았다** — 이 CSV는 초안에서 바로 만든 것이다. 0.7.0 병합 때 같은 ID·분할을 쓰면 캐시를 그대로 쓴다

## 2. 결과 (dev 60, `run1/`)

| | 건수 | 비율 |
|---|---|---|
| 정답이 후보 30 안 | **32 / 60** | 53.3% |
| 최종 Hit@1 | **16 / 60** | 26.7% |
| 최종 Hit@5 | 29 / 60 | 48.3% |
| 최종 Hit@10 | **32 / 60** | 53.3% |
| MRR@10 | 0.354 | |
| (리랭킹 전 Hit@1 · Hit@10) | 11 · 25 | |
| **허용 정답 포함** Hit@1 · Hit@10 · MRR@10 | **17** · 32 · 0.362 | t003 한 건(§4) |

- **후보 30에 든 32건은 전부 최종 10위 안에 들었다.** 실패 28건은 모두 후보 단계에서 정답을 놓친 것이다
- 리랭킹: 적용 54 · 생략 6 · 실패 0. 순위 오름 14 · 내림 3(t004 2→4, t086 2→7, t108 1→2)
- 후보 밖 28건의 단계(`candidate_drop_dev.csv`): 경로 밖 23 · 융합 절단 4(t077·t100·t103·t111) · 보정 뒤 절단 1(t058)

| 층 | n | 후보 30 | Hit@1 | Hit@10 |
|---|---|---|---|---|
| 확장 곡 | 40 | 22 | 9 | 22 |
| 기존 3,010곡 | 20 | 10 | 7 | 10 |
| text | 30 | 17 | 10 | 17 |
| multimodal | 11 | 7 | 3 | 7 |
| image | 9 | 7 | 3 | 7 |
| **audio** | 10 | **1** | 0 | 1 |
| 틀린 단서 | 3 | 0 | 0 | 0 |
| clue: sound_texture | 7 | 0 | 0 | 0 |
| clue: external_context | 13 | 7 | 5 | 7 |
| clue: lyric_fragment | 9 | 7 | 4 | 7 |
| clue: image_only | 9 | 7 | 3 | 7 |
| clue: mixed_av | 11 | 7 | 3 | 7 |
| clue: title_structure | 5 | 3 | 1 | 3 |
| clue: era_group | 3 | 0 | 0 | 0 |

층별 건수가 작다(3~13건). 한두 건으로 비율이 크게 움직이므로 순서만 본다.

**기존 세트와 나란히 (같은 5,143곡 · 같은 설정, `results_v39_corpus5143_baseline`).** 기존 dev 116건은 후보 30 안 80(69.0%) · Hit@1 56(48.3%) · Hit@10 76(65.5%)이고 2회가 완전히 같았다.
v10 dev는 후보 53.3% · Hit@1 26.7% · Hit@10 53.3% — 같은 코퍼스에서 후보 16%p · Hit@10 12%p · Hit@1 22%p 낮다. **v10이 기존 세트보다 어렵다.**
곡이 3,010 → 5,143으로 늘어난 영향은 기존 세트에서 후보 89 → 80, Hit@10 84 → 76이다. 3,010곡 기준선 숫자와는 나란히 놓지 않는다.

## 3. 실패 28건 (`dev_first_results.csv`에 전체 60건)

T·I·A는 텍스트·표지·소리 경로에서 정답의 순위(전체 깊이), `-`는 그 경로를 쓰지 않음. "깊이"는 경로를 끝까지 넓혔을 때의 후보 순위(30 이하면 들어옴).

### A. 질의 분석이 규칙 폴백으로 떨어짐 — 3건 (분석기 결함)

| ID | 질의(요약) | 정답 | 위치 |
|---|---|---|---|
| t009 | 작곡가 객원 보컬 프로젝트, 이별 노래인데 밝음, 2000년대 후반 | 뜨거운 안녕 - 토이 | T648 |
| t071 | FIFA 07에 한국어 곡 최초로 수록된 노래 | Fly - 에픽하이 | T89 |
| t099 | 빨간 드레스 여자가 초록 풀밭에 엎드린 이미지 | 꽃 - 지수 | T276, 표지 경로 꺼짐 |

Gemini 응답은 왔는데 `apply_modality_query_safeguards`(`src/retrieval/modality_queries.py`)가 3회 모두 거부해 규칙 폴백이 반환됐다. 두 번 돌려도 같다(`build_cache.log`·`build_cache_retry1.log`) — 일시 오류가 아니라 **서비스에서도 그대로 일어나는 동작**이다. 그래서 `--allow-fallback-analysis`로 그대로 쟀다.

- 글자 하나로 걸린다: `_STRONG_AUDIO_RE`·`_AUDIO_GENRE_RE`가 "초**록**"·"수**록**된"의 `록`을 록 장르로 읽어 "소리 단서가 있는데 `audio_english_query`가 없다"고 거부. t009는 "이**별**" + "앨범"이 표지 묘사로 읽힘
- 폴백의 결과: t086·t099(표지만 말한 질의)가 **표지 가중 0 · 소리 프롬프트 "An audio track with a rock style."** 로 검색됐다. t086은 텍스트 경로로 후보 2위에 들어 최종 7위, t099는 밖
- dev 60 중 "초록"이 든 질의 4건(t086·t099·t101·t110) 가운데 소리 단서가 따로 없는 2건이 걸렸다. 표지 색으로 흔한 단어다

### B. 소리 묘사 질의 — 8건 (소리 경로가 정답을 못 찾음)

| ID | 질의(요약) | 정답 | 위치 |
|---|---|---|---|
| t058 | 레게 리듬, 엇박 기타, 콧소리 섞인 능청스러운 남자 | 핑계 - 김건모 | **T1** · A782 · 보정 뒤 절단 |
| t111 | 도입부에 사이렌 소리 | 품행제로 - 블락비 바스타즈 | T25 · A751 · 융합 절단 |
| t077 | 혼성 아이돌, 묵직한 베이스, 자신감 생기는 | FAMOUS - ALLDAY PROJECT | T82 · A2033 · 깊이 21 |
| t073 | 콧소리 남자, 피아노, 후렴에서 짧은 말 반복 | 양화대교 - Zion.T | T3530 · A36 |
| t078 | 드럼·피아노가 질주하는 록 | 하늘을 달리다 - 이적 | T120 · A1252 |
| t075 | 피아노·현악, 중저음 남자 발라드 | 너의 바다에 머무네 - 토이 | T859 · A381 |
| t079 | 가성 없이 생목 고음 발라드 | 가질 수 없는 너 - 뱅크 | T632 · A784 |
| t080 | 통통 튀는 기타, 남자 랩 + 여자 후렴 | 한여름밤의 꿀 - San E | T1489 · A3392 |

소리 경로 순위가 36~3,392위다. 분석이 소리 가중을 0.5~0.7로 주기 때문에, 소리 경로에서 못 찾으면 텍스트에서 1위여도 밀린다 — t058은 텍스트 1위(가중 0.3)인데 소리 경로 상위 30곡(가중 0.7)에 밀려 후보 밖.

### C. 방송·영화·사연 같은 외부 맥락 — 5건

| ID | 질의(요약) | 정답 | 위치 |
|---|---|---|---|
| t044 | 무한도전에서 나온 미국 서부 느낌 | 멋진헛간 - 오대천왕 | T58 |
| t043 | 넷플릭스 애니에서 제일 유명한 노래, 높은 고음 | Golden - HUNTR/X | T79 (1위는 같은 작품 Takedown) |
| t038 | 오디션에서 남자 둘이 부른 옛 록 명곡 리메이크 | 걱정말아요 그대 - 곽진언 | T490 |
| t040 | 첫 여자 비행사 영화 주제가, 원곡 | 서쪽 하늘 - 이승철 | T810 |
| t042 | 군 복무 중 역주행한 남자 솔로 록발라드 | Drowning - WOODZ | T4216 · A102 |

### D. 틀린 단서(misinformation) — 3건 전부

| ID | 틀린 부분 | 정답 | 위치 |
|---|---|---|---|
| t029 | "남자 가수가" | 김철수 씨 이야기 - 허회경 | T62 |
| t016 | "남자 셋이 하던 그룹" | 비행기 - 거북이 | T616 (1위는 같은 가수 다른 곡) |
| t026 | "남자 아이돌 힙합 곡" | Zombie - DAY6 | T1760 |

### E. 가수를 돌려 말했거나 가수명 + 분위기 — 3건

| ID | 질의(요약) | 정답 | 위치 |
|---|---|---|---|
| t031 | 아이유, 핑크핑크한 밝은 노래 | strawberry moon | T557 · 깊이 27 (1위 Blueming) |
| t068 | 박재범, 여자 보컬·남자 래퍼가 한 파트씩 | All I Wanna Do (K) | T96 (1위 GANADARA) |
| t001 | 공연 묘사로 가수를 가리킴, 90년대 후반 발라드 | 나와 같다면 - 김장훈 | T863 · 깊이 24 |

### F. 제목 모양·곡 구성 — 2건

| ID | 질의(요약) | 정답 | 위치 |
|---|---|---|---|
| t011 | 힙합 듀오, 후렴은 R&B 가수, 제목이 줄임말, 2007년쯤 | 출첵 - 다이나믹 듀오 | T154 · **깊이 2** |
| t062 | 분위기가 갑자기 바뀌는 걸그룹 노래 | O.O - NMIXX | T33 · 깊이 24 (1위 같은 가수 DICE) |

### G. 표지 · 표지+소리 — 4건

| ID | 질의(요약) | 정답 | 위치 |
|---|---|---|---|
| t100 | 털모자 쓴 여자 표지 + 허스키한 여자 R&B | 장마 - 정인 | T46 · I28 · 깊이 15 · 융합 절단 |
| t103 | 누런 배경 크레파스 글씨 표지 + "호구 노래" | 사랑의 바보 - 더 넛츠 | I13 · T525 · 융합 절단 |
| t092 | 흑백 사진, 남자 여럿 V자, 빨간 붓글씨 | 뛰뛰빵빵 - 비투비 | I82 |
| t060 | 디스코, "에헤레이아" 가사, 롤리폴리 비슷한 안무 | 나팔바지 - 싸이 | T244 (1위 Roly-Poly) |

경로를 끝까지 넓히면 후보에 드는 것은 28건 중 6건(t001·t011·t031·t062·t077·t100)이다. 경로 깊이 확대는 v23에서 채택하지 않았다(얻는 만큼 잃음) — 여기서는 위치만 적는다.

## 4. 라벨 확인이 필요한 사례

정답 곡과 제목이 같은 다른 곡이 코퍼스에 있는지(17건), 그 곡이 top-10에 들었는지만 기계적으로 봤다. 질의의 사실 관계는 확인하지 않았다.

| ID | 내용 | 할 일 |
|---|---|---|
| t003 | 정답 고해 - 임재범(1998)이 2위, **1위가 고해 - 황치열(2006, 리메이크)** | **결정(10/11, 잠정): 성공으로 둔다.** 황치열 판 `1484943`을 허용 정답으로 `allowed_v10.csv`에 적었다. 엄격 지표(원래 정답만)는 그대로 두고 확장 지표로 따로 낸다 |
| t038 | 정답 `5402092`는 **곽진언·김필 듀엣**(앨범 "슈퍼스타K6 곽진언 vs 김필", 2014)이다 — 질의의 "오디션 프로에서 남자 둘이 듀엣"과 맞는다. 코퍼스의 다른 판은 **이적 솔로**(`7921208`, 응답하라 1988 OST, 2015)이고, 곽진언과 이적이 함께 부른 판은 코퍼스에 없다. 질의의 "나중에 드라마 OST"는 이적 판 이야기. 둘 다 top-10 밖 | 라벨 유지. 이적 판은 허용 정답에 넣지 않았다. 곡 메타데이터의 가수 칸에 김필이 빠져 있다(곽진언만) |
| t018 | 정답 밤이 깊었네 - 크라잉넛 1위, 싸이 판(2022) 2위 | 참고(문제 없음) |

그 밖의 같은 제목 15건은 top-10에 들지 않았다. 라벨이 틀렸다고 볼 사례는 찾지 못했다.

## 5. 이 측정으로 말할 수 없는 것

- ~~v10이 기존 세트보다 어려운지~~ → `results_v39_corpus5143_baseline`에서 확인(위 §2). 왜 어려운지(질의 구성 · 초안을 태그·요약 안 보고 쓴 것)는 가르지 않았다
- 수치의 흔들림 — 1회 측정
- 폴백 3건을 고치면 몇 건이 살아나는지 — 고쳐서 재지 않았다
- test 42건의 성적 — 재지 않았다

## 6. 보존

| 파일 | git |
|---|---|
| `eval_queries_v10.csv` · `analysis_cache_v10_dev.json` (`experiments/reranking/`) | 추적 대상 |
| `make_split.py` · `summarize.py` · `RUN_INFO.md` | 추적 대상 |
| `split_v10.csv` · `split_v10_dropped.csv` · `allowed_v10.csv` · `dev_first_results.csv` · `candidate_drop_dev.csv` | 추적 대상(`.gitignore` 예외 추가) |
| `run1/search_eval_dev_summary.csv` · `_top10.csv` · `_runinfo.json` | 추적 대상 |
| `run1/search_eval_dev_detail.csv` · `_explain.jsonl` · `_lyric_boosts.csv` · `*.log` | 제외(기존 규칙) |

## 7. 다시 돌리기

```bash
export TOKENIZERS_PARALLELISM=false SEARCH_REFERENCE_YEAR=2026 RERANKER_BACKEND=gemini_listwise GEMINI_RERANK_RARE_FACT_VERIFY=0
OUT=experiments/reranking/results_v38_v10_dev_first
venv/bin/python $OUT/make_split.py
venv/bin/python -m src.retrieval.build_analysis_cache --input experiments/reranking/eval_queries_v10.csv --split dev \
  --output experiments/reranking/analysis_cache_v10_dev.json
venv/bin/python -m src.retrieval.trace_candidate_drop --input experiments/reranking/eval_queries_v10.csv --split dev \
  --analysis-cache experiments/reranking/analysis_cache_v10_dev.json --deep-k 5200 --output $OUT/candidate_drop_dev.csv
venv/bin/python -m src.retrieval.evaluate_search_accuracy --input experiments/reranking/eval_queries_v10.csv --split dev \
  --analysis-cache experiments/reranking/analysis_cache_v10_dev.json --allow-fallback-analysis --output-dir $OUT/run1
venv/bin/python $OUT/summarize.py
```

`build_analysis_cache`는 폴백 4건(t009·t071·t086·t099) 때문에 종료 코드 1로 끝난다. 분석기를 고치기 전에는 정상이다.
`TOKENIZERS_PARALLELISM=false`를 빼면 HF tokenizers가 fork 뒤 멈추거나 죽는다(results_v37에서 확인).
