# 자료 · 모델 · 색인 명세 — v77 `desc_semantic_v1`

## 1. 입력

| 입력 | 위치 | 지문(sha256) | 쓰는 것 |
|---|---|---|---|
| 코퍼스 | 메인 체크아웃 `data/all_songs.jsonl`(읽기만, Git 제외) | `63891fd3fa6155f73e2f7496ec5fdff414937c67ea2373c5e0ff18470568d9de` | `song_id` · `metadata.title` · `metadata.album_description` |
| 기존 텍스트 dense 벡터 | 메인 체크아웃 `artifacts/embeddings/text_dense/nlpai-lab__KoE5/<song_id>.npy`(운영 색인 적재 원본, 읽기만) | 5,143/5,143 있음(파일별 지문은 기록하지 않음) | 대표곡 규칙 ②(§3)에만 |
| 질의 분석 | `experiments/reranking/results_v49_vertex_baseline/analysis_cache_{v06,v09,v10}_dev_vertex.json` | v06 `36eab91c…` · v09 `e1048ecf…` · v10 `c6c2d9d8…` | `search_text`(S) · `korean_tags`(L·V) · 가사 단서(정확 가사 표시만) |
| v75 A 후보 | `inputs/v75A_candidates.json`(v76 A1 기록에서 id만 뽑음, 원기록 sha 포함) | — | 채점만 |
| v75 A Hit@10 | `inputs/v75A_hit10.json`(v76 `analysis/per_query.csv`에서) | — | 채점만 |
| 정답 | `experiments/reranking/eval_queries_{v06,v09,v10}.csv`의 dev `relevant_ids` | — | **`score.py`에서만** |
| v76 비교 | v76 워크트리 코드 사본 `artifacts/src_v76_f3c2e1b737479bdc/src/retrieval/description_aux.py`(`f05d4556…`) · 산출물 `description_residual_v1.json.gz`(`11d72680…`) | | V 순위만(읽기 전용) |

## 2. 모델

- 모델: `nlpai-lab/KoE5`, revision `237b580aea358e3042fcf5baf78124c33c3afd16`. 로컬 HF 캐시만 썼고 새로 내려받지 않았다(`HF_HUB_OFFLINE=1` · `TRANSFORMERS_OFFLINE=1`). `model.safetensors` blob `97693a2aeaeae9ec…`.
- 구조: XLM-R large(24층) · 1024차원 · 평균 풀링 · L2 정규화 모듈 · `max_seq_length` 512.
- 입력:
  - 청크 `"passage: " + 청크`
  - 질의 `"query: " + analysis.search_text`(운영 `SearchService.search_text`와 같은 형식)
- `normalize_embeddings=True` · 배치 32 · 장치 `mps:0`(sentence-transformers 기본 선택).
- 실행 환경: Python 3.14.7 · torch 2.10.0 · sentence-transformers 3.3.1 · transformers 4.50.3 · numpy 2.5.3 · kiwipiepy 0.23.2 · Apple M4 Pro(arm64, 24GB).

## 3. 규칙(`desc_semantic.py`)

1. 소개 정규화: 연속 공백(줄바꿈 포함) → 공백 하나, 앞뒤 공백 제거. 문자열이 아니거나 비면 자료 없음.
2. 묶음 id = 정규화 원문 sha256 앞 16자. v76은 `strip()`만 한 원문이라 id가 다를 수 있다. V 순위는 v76 자신의 곡→묶음 표로 채점했다.
3. 문장 분할 `(?<=[.!?。])\s+`. 400자 이하로 이어 붙인다(겹침 없음). 400자 초과 문장은 400자 안의 마지막 공백에서, 공백이 없으면 400자에서 끊는다. 같은 묶음 안 같은 청크는 첫 것만 남기고 빈 청크는 버린다.
4. 점수 = 정규화 벡터 내적(float32, 전수 비교). 묶음 점수는 청크 최댓값이다. 동점은 (점수 내림, 묶음 id 오름) 순이고, 묶음 안에서는 청크 번호가 작은 것이 앞선다.
5. 대표곡:
   - ① 근거 청크에 제목(괄호 제거 · NFKC · 소문자 · 글자/숫자만, 2글자 이상)이 나오는 곡이 하나면 그 곡.
   - ② 그 밖에는 (①이 여럿이면 그중에서) 기존 텍스트 dense 내적이 가장 큰 곡. 동점이면 song_id 숫자가 작은 곡.
6. L(대조군 — 문서는 S와 같고 질의 표현은 다름): 같은 청크를 `normalize_for_bm25`(`af0c6afc…`)로 토큰화한다. 질의는 `korean_tags` 토큰이다. Okapi BM25(k1 1.2 · b 0.75 · Lucene idf, 문서 = 묶음별 청크 행)이고 점수 0은 순위에서 뺀다.

## 4. 산출물(`artifacts/description_semantic/desc_semantic_v1/`, Git 제외)

| 파일 | 크기 | sha256 |
|---|---:|---|
| `vectors.npy`(고유 청크 13,344 × 1024 float32) | 54.66MB | `50fb9fd3964d7fb06ef03fd4e5fee29d9ae3b3db2fbe256528745f6ef0b113e5` |
| `chunks.jsonl.gz`(청크 행 13,372 — **원문 포함**) | 3.25MB | `a61a34a25552553729e2c8ddfc7ca5601e7ad86c316dffaf1528ede1d36311dc` |
| `groups.jsonl.gz` | 70kB | `2fc1ae9394f2bb67301531827180d96364b27526bf750c0381cd41599d8b3f2b` |
| `bm25_tokens.jsonl.gz`(원문 토큰) | 2.7MB | `5660385fbc1ed42084aa7b8e8c38a52fb48f68a10d2e007fbe18bad78e8db75e` |
| 청크 목록 지문(묶음·번호·청크 sha) | — | `6d68a05277bb6a0af606e66349d279acddeaebbf2a3e34fa1d02b8bf4a11437f` |
| `manifest.json`(사본 `inputs/index_manifest.json`) | 2.7kB | `1e49ccec2f20a8c7…` |

## 5. 규모

| 항목 | 값 |
|---|---:|
| 곡 · 소개 없는 곡 | 5,143 · 270 |
| 소개 묶음 · 여러 곡 공유 묶음(곡 수) · 최대 묶음 크기 | 3,904 · 514(1,483) · 11 |
| 청크 행 · 고유 청크 · 여러 묶음에 같은 글자로 나온 청크 | 13,372 · 13,344 · 24 |
| 묶음당 청크 중앙값 | 3 |
| 400자 초과로 강제 분할한 문장 | 607 |
| 토큰 수(접두어·특수 토큰 포함) 중앙값 · p90 · p99 · 최대 | 170 · 228 · 251 · 295 |
| 512토큰 초과(잘림) | **0** |

## 6. 생성 비용 · 재현

| 항목 | 1차(run0) | 2차(경로 정리 뒤 재생성) |
|---|---:|---:|
| 임베딩 시간 | 448.6초 | 444.9초 |
| 전체(토큰화·적재 포함) | 483.8초 | 476.4초 |
| 최대 RSS | 1,479MB | 1,489MB |
| 산출물 sha | — | **1차와 4파일 모두 같음** |

- 재현 표본: 고유 청크 200개(seed 77)를 다른 배치 구성으로 다시 임베딩했다. 최대 절댓값 차 8.2e-7, 최소 코사인 0.9999999로 순위 영향은 없다(검색 R1 = R2 = 1차 R1 = 1차 R2).
- 1차 생성 때 `/usr/bin/time -l`의 `peak memory footprint`는 31.6GB로 나왔다. MPS 통합 메모리 할당을 세는 값으로 보이며 원인은 확인하지 않았다. 스왑은 0이었다.
- 외부 호출 0 · 비용 $0.

## 7. 코드 지문(2차 실행 기준)

| 파일 | sha256 앞 16자 |
|---|---|
| `desc_semantic.py` | `48a4efba6d91e78d` |
| `build_index.py` | `84b0fc32b1806d20` |
| `retrieve.py` | `6ec973ec1534abd9` |
| `score.py` | `4a91dc2921e31c73`(리뷰 반영 — 고유 곡 수 집계 추가. 측정 때 판 `b2f2d09118517b61`) |
| `analyze.py` | `f33f2427718c8533` |
| `test_desc_semantic.py` | `41169f4104ffcd32` |
| `run_all.sh` | `91dd1631dd865a52` |

재현: 워크트리 루트에서 `bash experiments/reranking/results_v77_description_semantic_recall/run_all.sh`를 실행한다. 메인 체크아웃은 git 공용 폴더에서 찾고 `VF_MAIN_CHECKOUT`으로 바꿀 수 있다. v76 워크트리는 `V76_WORKTREE`로 바꾼다. 색인·실행 폴더가 있으면 멈춘다.
