# results_v46 — #12 병합 뒤 main의 dev 기준선 · 소리 경로를 끄면 (2026-10-13)

#12(나무위키 Context 경로)가 main에 들어오면서 **질의 분석 프롬프트와 모달리티 규칙이 바뀌었다.** 그래서 기존 분석 캐시(v39 · v41 · v43 기준선이 쓴 것)는 지금 서비스가 하는 분석이 아니다. dev 176건(v06 57 · v09 59 · v10 60)을 다시 재고, 소리 경로를 끄는 안(`AUDIO_FUSION_SCALE=0`, results_v43)이 새 main에서도 통하는지 봤다.
**코드 변경 없음. 봉인 test(v10 test 42)는 재지 않았다.**

- 워크트리 `.claude/worktrees/eval-main-after-context` · 브랜치 `eval/main-after-context-baseline` (main `5e08e68`)
- 5,143곡 · BM25 `bf606b19ab02` · 색인은 v39 · v41 · v43과 같은 것(`artifacts/qdrant_5143_baseline_backup`을 `artifacts/qdrant`로 연결)
- listwise 기본(`RERANKER_BACKEND=gemini_listwise` · `GEMINI_RERANK_RARE_FACT_VERIFY=0`) · AI Studio(`api_key`) · `SEARCH_REFERENCE_YEAR=2026`
- **이 로컬에는 Context 색인이 없다.** Context 경로는 빈 결과를 내고, 그 효과는 아래 숫자에 들어 있지 않다(§4)

## 결론

1. **#12의 검색 코드는 기존 순위를 바꾸지 않는다.** 예전 분석 캐시 그대로 재면 176건이 v43 현행과 순위까지 같다
2. **분석을 지금 분석기로 다시 하면 기준선이 움직인다.** 후보 113 → 118 · Hit@10 109 → 113 · Hit@1 73 → 72. 10위 안에 10건이 새로 들고 6건이 빠진다
3. **소리 경로를 끄는 효과는 새 기준선에서 더 크다.** Hit@1 72 → **79** · Hit@10 113 → **115** · MRR 0.492 → **0.522**(2회 동일). v43에서는 Hit@1 +1 · Hit@10 +2였다
4. 끄면 빠지는 질의가 1건 있다 — n018 "인트로에 물 흐르는 소리". 이름 붙은 소리를 묻는 질의로, results_v45의 t111(사이렌)과 같은 갈래다

## 1. 측정표 (엄격 지표)

| 분석 캐시 | 소리 배율 | 횟수 | 후보 30 안 | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| #12 전 (v43 현행과 같은 것) | 1 | 1 | 113 | 73 | 102 | 109 | 0.485 |
| 지금 분석기로 새로 | 1 (현행) | 2 | **118** | **72** | **103** | **113** | **0.492** |
| 지금 분석기로 새로 | 0 (소리 끔) | 2 | **119** | **79** | **105** | **115** | **0.522** |

세트별(새 캐시):

| 세트 | n | 현행: 후보 · Hit@1 · Hit@10 · MRR | 소리 끔: 후보 · Hit@1 · Hit@10 · MRR |
|---|---|---|---|
| v06 | 57 | 44 · 28 · 41 · 0.563 | 44 · 31 · 41 · 0.596 |
| v09 | 59 | 39 · 26 · 38 · 0.512 | 39 · 28 · 38 · 0.534 |
| v10 | 60 | 35 · 18 · 34 · 0.405 | 36 · 20 · 36 · 0.440 |

- 같은 설정 2회는 176건의 최종 순위가 모두 같았다(현행 · 소리 끔 둘 다)
- 리랭킹이 적용되지 않은 질의는 없다
- 허용 정답을 넣은 확장 지표는 재지 않았다

## 2. 예전 캐시 → 새 캐시 (현행끼리)

`old_scale1_rep1`은 v43 `scale1_rep1`과 176건 모두 같다 — 그러니 아래 변화는 전부 **분석이 달라진 것**이다.

- 10위 안에 새로 든 10: v06 c601 · m303 · q106 · q208 · q210 / v09 n018 · n076 · n089 / v10 t026 · t044
- 빠진 6: v06 m104(1위 → 밖) · q205 / v09 n040(1위 → 밖) · n057 · n091(1위 → 밖) / v10 t008
- 10위 안에서 오름 14 · 내림 13

분석이 어떻게 달라졌나(`compare_caches.py` · `cache_diff.csv`):

| | 176건 중 |
|---|---|
| 규칙 폴백 | 0 → 0 |
| 모달리티를 내려놓은 분석 | 1 (q115, audio — "background" 누출 검사) |
| 경로 구성이 바뀐 질의(가중이 0 ↔ 양수) | 8 |
| `context_clues`가 붙은 질의 | 37 |
| intent가 바뀐 질의 | 10 |

- 경로 구성이 바뀐 것은 8건뿐인데 순위는 그보다 많이 움직였다. 태그 · 영어 프롬프트 같은 나머지 분석 내용이 다시 쓰였기 때문이다
- **이 변화가 #12의 프롬프트 변경 때문인지, 분석을 다시 돌린 것 자체의 흔들림인지는 가리지 않았다.** 예전 캐시는 여러 차례의 프롬프트 변경을 거치며 일부 항목만 갱신해 온 것이라(캐시 meta의 `drift_notes`), #12 직전 코드로 전부 다시 분석한 결과와도 같지 않다

## 3. 소리 경로를 끄면 (새 캐시 · 현행 → 배율 0)

- 든 3: n057(밖 → 3위) · t008(밖 → 1위) · t058(밖 → 2위)
- 빠진 1: **n018(3위 → 밖)** — 후보 10위 → 후보 밖
- 10위 안에서 오름 12 · 내림 5. 1위로 올라온 것: c707(2 → 1) · q210(2 → 1) · q312(10 → 1) · n010 · n031 · n089(5 → 1) · t101 · t104

n018은 "인트로에 물흐르는 소리가 나오는 옛날 민요풍 트로트?"다(소리 가중 0.5). 새 분석에서는 소리 경로가 켜져 있을 때만 정답이 후보(10위)에 든다 — 예전 분석에서는 켜도 후보 밖이었다. results_v45에서 본 대로 CLAP은 사이렌 · 물소리처럼 **이름 붙은 소리**는 잡는다. 소리 경로를 끄면 이 갈래를 잃는다 — dev 176건에서 확인된 것은 이 한 건이다.

v43과 달라진 것: 거기서 빠졌던 c707은 이번에는 2 → 1위로 오른다. 거기서 들었던 t077과 후보에 들던 t111(사이렌)은 이번에는 어느 설정에서도 후보 밖이다.

## 4. 이 숫자가 말하지 못하는 것

- **Context 경로의 효과.** 새 분석은 37건에 `context_clues`를 붙였지만 이 로컬에는 Context 색인이 없어 경로가 돌지 않았다. #12의 Context 색인은 3,016곡 코퍼스(검색 가능한 Context 478곡) 기준으로 만들어졌고 5,143곡 색인에는 없다. Context 색인을 올리면 이 37건의 후보가 달라질 수 있어 **기준선을 한 번 더 재야 한다**
- 봉인 test. 최종 설정이 정해지지 않았다(위 Context 색인 문제 포함)
- 5곡 메타데이터 수정(#45) 뒤의 서비스 색인(`artifacts/qdrant`)이 아니라 수정 전 색인으로 쟀다 — v43과 견주기 위해서다

## 파일

| | |
|---|---|
| `build_caches.sh` · `analysis_cache_{v06,v09,v10}_dev_ctx.json` | 지금 분석기(prompt `80c464515d9c` · postprocess `203a9248e607`)로 만든 dev 분석 캐시 |
| `compare_caches.py` · `cache_diff.csv` | 예전 캐시와 질의별 비교 |
| `run_eval.sh` · `run_new_all.sh` | 측정 |
| `old_scale1_rep1/` | 예전 캐시 · 현행 — v43 `scale1_rep1`과 같음을 확인 |
| `new_scale1_rep{1,2}/` · `new_scale0_rep{1,2}/` | 새 캐시 · 현행 / 소리 끔 |
| `summarize.py` | 위 표와 질의별 변화. detail CSV는 커밋하지 않아, 커밋된 파일만으로는 "후보 30 안" 칸이 `?`로 나온다 |

```
PY=venv/bin/python zsh experiments/reranking/results_v46_main_after_context/build_caches.sh      # Gemini 분석 176회
CACHES=old SCALE=1 zsh experiments/reranking/results_v46_main_after_context/run_eval.sh
zsh experiments/reranking/results_v46_main_after_context/run_new_all.sh                            # 약 65분
venv/bin/python experiments/reranking/results_v46_main_after_context/summarize.py results_v43_audio_path_weight/scale1_rep1 old_scale1_rep1 new_scale1_rep1 new_scale1_rep2 new_scale0_rep1 new_scale0_rep2
```
