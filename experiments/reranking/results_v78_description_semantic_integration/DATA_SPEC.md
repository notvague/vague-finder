# 자료 · 모델 · 코드 지문 — v78 소개 의미 보조 오프라인 통합

## 1. 입력(읽기만)

| 입력 | 위치 | 지문(sha256 앞 16자) | 쓰는 것 |
|---|---|---|---|
| v77 의미 검색 색인 `desc_semantic_v1` | v77 워크트리 `artifacts/description_semantic/desc_semantic_v1/`(PR #92 OPEN — 미병합, Git 제외)에서 이 워크트리로 바이트 복사 | vectors `50fb9fd3964d7fb0` · chunks `a61a34a255525537` · groups `2fc1ae9394f2bb67` · manifest `1e49ccec2f20a8c7` | 벡터(런타임이 직접 읽음) · 청크 행·묶음(빌드 때만) |
| 코퍼스 | 메인 체크아웃 `data/all_songs.jsonl`(Git 제외) | `63891fd3fa6155f7`(v77 색인 manifest의 코퍼스 sha와 같아야 빌드된다) | 제목(빌드 때 제목 일치 계산만) |
| 기존 텍스트 dense 벡터 | `artifacts/embeddings/text_dense/nlpai-lab__KoE5/<id>.npy`(메인 연결) | 4,873/4,873(소개 있는 곡 전부) | 대표곡 규칙 ② |
| 질의 분석 | `results_v49_vertex_baseline/analysis_cache_{v06,v09,v10}_dev_vertex.json` | v75·v76과 같은 파일 | 검색 입력(`search_text` 등) |
| 작품명 해석 캐시 | v76 `resolution_cache/v71_rep1.json` 사본(Git 제외) | `03ad094834c986e5` | A·B 공통 |
| 언급 저장소 | `artifacts/work_names/work_mentions_v1.json.gz`(v76 사본) | `bfd522ebe9da7590` | A·B 공통 |
| Qdrant 저장소 | 정지된 v76 컨테이너 저장소의 파일 복사본 `artifacts/qdrant_server_v78/storage`(828파일) | sha1 목록 지문 `b3aa0a81041b6667`(원본 = 복사본) · 텍스트 색인 내용 지문 `238299e6c8c17f2a` | 서버 `vague-finder-qdrant-v78`(127.0.0.1:6392, `qdrant/qdrant:v1.19.1`) |
| 가사 | v63 스냅샷 | 내용 지문 `e0d58f28f1f4a563` | A·B 공통 |
| 정답 | `eval_queries_{v06,v09,v10}.csv` dev `relevant_ids` | — | **`analyze_v78.py`에서만** |
| v75 A 성공 121 | main `results_v76_description_evidence_recall/analysis/per_query.csv`의 `v75A_hit10` | — | 분석만 |
| v77 R1 순위(대조용) | v77 워크트리 `runs/R1/rankings.jsonl.gz`(Git 제외) | `5c6f729be780ce87` | 단독·실행 중 대조만 |

## 2. 모델

- `nlpai-lab/KoE5`, 로컬 HF 캐시의 revision `237b580aea358e3042fcf5baf78124c33c3afd16`(캐시에 이 판 하나). `HF_HUB_OFFLINE=1` · `TRANSFORMERS_OFFLINE=1`. 새로 내려받지 않았다.
- 질의 입력 `"query: " + analysis.search_text`, `normalize=True`. 제품 임베더(`KoE5Embedder()`, revision 미지정 — 운영과 같음)와 v77 방식(`SentenceTransformer(revision=237b580a)`)의 질의 벡터 최대 차 0.0(dev 176).
- 장치 `mps:0`(sentence-transformers 기본 선택). Python 3.14.7 · torch 2.10.0 · sentence-transformers 3.3.1 · numpy 2.5.3 · Apple M4 Pro.

## 3. 런타임 산출물 `artifacts/description_semantic/desc_semantic_v1_runtime/`(Git 제외)

| 파일 | 크기 | sha256 |
|---|---:|---|
| `structure.json.gz`(묶음 3,904 · 묶음 곡 · 청크 행 13,372 = (묶음 번호, 청크 번호, 벡터 행, 제목이 나오는 곡) · dense 곡 목록) — 원문 없음 | 170kB | `466bf366afafc86225a4c4a2ef9daf6a458342856577aad4a29160e9132028bc` |
| `member_dense.npy`(4,873 × 1024 float32) | 20MB | `01a1ef1970e69b38dcb102bac246b2570eb7921c6a628067b23a0cec346ca93c` |
| `manifest.json` | 1.4kB | (위 두 sha · v77 manifest sha · 벡터 상대 경로와 sha · 코퍼스 sha · 모델 · 상수) |

- 빌드: `python -m src.retrieval.description_semantic_aux build --index-dir artifacts/description_semantic/desc_semantic_v1 --songs <main>/data/all_songs.jsonl --text-dense-dir artifacts/embeddings/text_dense/nlpai-lab__KoE5 --out artifacts/description_semantic/desc_semantic_v1_runtime` — 1.5초 · 최대 RSS 127MB. 같은 자료면 같은 바이트(계약 시험). 있으면 덮어쓰지 않는다.
- 적재: 0.04초(sha 검사 포함) · RSS +81MB.

## 4. 코드

| 대상 | 지문 |
|---|---|
| 기준 커밋 | `3660083038f666aa2b69892e76db7d07ca6d8024` |
| 패치 `patch/v78_description_semantic_aux.patch`(10파일) | sha256 앞 16자 `449842372f3f6d09` |
| 실행 코드 사본 `artifacts/src_v78_b4131c16124861eb` | `src` `.py` 트리 지문 `b4131c16124861eb`(바뀐 7 · 새 1, `inputs/code_manifest.json`) = 깨끗한 기준 커밋 + 패치 |
| 하네스 출처(v76, main `3660083`) | `v76_eval_runtime.py` `01f319c85c669be8` · `run_eval.py` `1580de573f7cbf11` · `compare_inputs.py` `9725650a46b0f6f5` · `server_state.py` `7b8bb791bd5096f6` · `qdrant_reference.py` `4b8bb92205257666`(그대로 복사) |

## 5. 재현 절차

1. `git worktree add <폴더> 3660083 && git -C <폴더> apply <이 폴더>/patch/v78_description_semantic_aux.patch`
2. `artifacts/`를 연결·복사한다(embeddings · context · map · timing · vector_db · audio_* · cover_stats는 메인 연결, `bm25_params.json` 복사, `work_names/`는 v76 사본, `src/frontend/static/map_data.json` 복사). v77 색인을 `artifacts/description_semantic/desc_semantic_v1/`로 복사하고 §1 sha를 확인한다.
3. §3 빌드 → sha 확인. `python make_code_copy.py`(트리 지문 `b4131c16124861eb`이어야 한다).
4. 해석 캐시를 `resolution_cache/v71_rep1.json`으로 복사(sha `03ad0948…`).
5. 정지된 v76 저장소를 복사해 컨테이너를 띄우고 `server_state.py --wait-green 600 --out server_state_baseline.json`.
6. `parity_v77.py --v77-rankings …` → `run_offline.sh` → `compare_all.sh` → `analyze_v78.py --v77-rankings …` → `measure_load.py`.
