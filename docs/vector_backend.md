# 벡터 DB 백엔드 (Pinecone / Qdrant)

검색은 두 백엔드 중 하나를 쓴다. `.env`의 `VECTOR_BACKEND`로 고르며, **검색·재질문·
리랭킹 코드는 어느 쪽이든 동일하다** — Qdrant 어댑터가 Pinecone과 같은
`Index(name).query(...)` 응답(`matches`의 `id / score / metadata`)을 돌려준다.

```bash
# 기본값 (관리형 Pinecone)
VECTOR_BACKEND=pinecone

# 로컬 Qdrant — 사용 한도 없음, 네트워크 왕복 없음
VECTOR_BACKEND=qdrant
QDRANT_PATH=artifacts/qdrant          # 생략 가능 (기본값)
# QDRANT_URL=http://localhost:6333    # 서버/클라우드를 쓸 때만
# QDRANT_API_KEY=...
```

## 처음 받았을 때 (팀원용)

**임베딩 벡터와 곡 데이터는 레포에 없다.** 가사·앨범 커버·음원에서 뽑은 것이라 커밋하지
않는다. 아래 네 단계를 거치면 검색이 돈다.

**1. 패키지** — `qdrant-client`가 새로 들어갔다

```bash
venv/bin/pip install -r requirements.txt
```

**2. 드라이브에서 받을 것**

| 받는 것 | 놓을 위치 |
|---|---|
| `embeddings.zip` (905곡, 약 14MB) | `artifacts/embeddings/` 아래에 풀기 |
| `all_songs.jsonl` | `data/all_songs.jsonl` |

푼 모양은 이렇다. 모델 이름 폴더는 자동으로 찾으므로 이름이 달라도 된다.

```
artifacts/embeddings/text_dense/<모델>/<song_id>.npy
artifacts/embeddings/image/<모델>/<song_id>.npy
artifacts/embeddings/audio/<모델>/<song_id>.npy
```

**3. `.env` 설정** — `.env.example`을 복사해 채운다

```bash
cp .env.example .env     # GEMINI_API_KEY와 VECTOR_BACKEND=qdrant 확인
```

`VECTOR_BACKEND`를 적지 않으면 **기본값이 pinecone이라 한도가 막힌 Pinecone으로 간다.**

**4. 적재**

```bash
venv/bin/python -m src.vector_db.cli.qdrant_load --recreate
```

디렉터리가 하나라도 없으면 명령이 **실패로 끝난다**(예전에는 조용히 건너뛰고 성공으로
끝나 빈 컬렉션이 만들어졌다). BM25 sparse는 파일이 아니라 이 단계에서 만들어진다.

이제 `uvicorn src.backend.main:app --reload`로 서버를 띄우거나 측정을 돌릴 수 있다.
가사 정확 일치 검색만 MongoDB가 따로 필요하고, 없으면 그 경로만 비고 나머지는 동작한다.

만들어지는 것 — `artifacts/qdrant/` (git 추적 안 함)

| 컬렉션 | 내용 |
|---|---|
| `vaguefinder-text-1024-koe5__dev` | KoE5 dense 1024 + BM25 sparse |
| `vaguefinder-image-768__dev` | SigLIP2 768 |
| `vaguefinder-audio-512__dev` | CLAP 512 |

Pinecone의 (인덱스, 네임스페이스)가 Qdrant 컬렉션 하나(`<인덱스>__<네임스페이스>`)에 대응한다.

나무위키 배경지식은 기존 곡 단위 컬렉션에 섞지 않는다. 사실 단위 Dense와 곡 단위
Sparse를 별도 세대 컬렉션에 안전하게 게시하는 절차는
[`namuwiki_context_vector_db.md`](namuwiki_context_vector_db.md)를 따른다.

## 왜 옮겼나

- Pinecone Starter는 **월 전송 1GB**다. 2026-09-16 측정 도중 소진돼 모든 읽기가 429로
  막혔고, 검색 결과가 0개가 됐다. 검색 1회당 약 0.3MB라 1GB ≈ 3,000회뿐이다.
- 이 프로젝트는 측정이 중심이다(하네스 한 번에 수백 회 검색). 질의 수를 줄이는 대신
  한도가 없는 쪽으로 옮겼다.
- 곡이 905~2,095곡이라 로컬 전수 검색이 빠르고 정확하다.
- Starter는 리전이 us-east-1 고정, 사용자 2명 제한이기도 하다.

## 점수를 같게 맞춘 방법

| 경로 | Pinecone | Qdrant |
|---|---|---|
| 텍스트 | alpha로 스케일한 dense·sparse를 한 번의 dotproduct 질의로 합산 | dense·sparse를 따로 조회해 **코드에서 합산**. 두 조회 모두 컬렉션 전수를 훑어 잘림이 없다 |
| 이미지·오디오 | cosine | cosine (저장 벡터가 전부 L2 정규화라 dot과 동일) |
| 메타데이터 필터 | `$eq` `$and` `$gte`(문장형 제목) `$or`(의성어 제목) | 같은 뜻의 Qdrant 필터로 변환(범위는 Range, OR는 should). **모르는 연산자는 예외를 던진다** |
| 검색 방식 | 근사 | 로컬은 전수 비교, 서버 모드는 `exact=True` |

## 주의

- `artifacts/qdrant/`는 **적재 산출물**이라 커밋하지 않는다. 벡터가 바뀌면 다시 적재한다.
- 평가 실행기(`src/eval/evaluate.py`, `src/retrieval/evaluate_*.py`)도 `VECTOR_BACKEND`를 따른다.
- 로컬 모드는 한 프로세스만 디렉터리를 연다. 서버와 측정 스크립트를 동시에 돌리려면
  각각 다른 `QDRANT_PATH`를 쓰거나 서버 모드(`QDRANT_URL`)를 쓴다.
- BM25 sparse는 `artifacts/bm25_params.json`과 짝이다. 한쪽만 바꾸면 점수가 어긋난다.
