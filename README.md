# Vague-Finder

**기억의 파편을 잇는 멀티 모달 RAG 검색 엔진**
(모호한 자연어와 시각 데이터 기반 지능형 시맨틱 검색 서비스)

## Team Roles

| 이름              | 메인 역할                                    | 서브 역할                                |
| ----------------- | -------------------------------------------- | ---------------------------------------- |
| **황찬혁 (팀장)** | FastAPI 백엔드 + Next.js 프론트엔드 + 배포   | 팀원 최적화 서브 지원                    |
| **이연우**        | 데이터 파이프라인 (yt-dlp, GitHub Actions)   | Pinecone 인덱스 관리, SigLIP Int8 양자화 |
| **최정현**        | 멀티모달 임베딩 (SigLIP, Ko-E5) + Re-ranking | RAG 프롬프트 설계                        |

## Project Structure

```bash
.
├── configs/                     # 글로벌 환경 설정 (Hyperparams, Path 등)
├── data/                        # 데이터 저장소 (Git 추적 제외)
│   ├── raw/                     # 크롤링된 원본 데이터 (보존용)
│   ├── processed/               # 전처리 완료된 데이터 (학습/서비스용)
│   └── cache/                   # 임시 캐시 및 중간 파일
├── docs/                        # 팀 협업 문서
│   ├── api_specs/               # API 명세서 (백엔드 <-> 프론트/AI)
│   └── data_schemas/            # 데이터 포맷 및 스키마 정의
├── notebooks/                   # 실험 및 연구용 샌드박스 (Jupyter)
│   ├── data_experiments/        # 데이터 정제 및 양자화 실험 (이연우)
│   └── model_experiments/       # 모델 성능 테스트 및 RAG 프롬프트 실험 (최정현)
├── scripts/                     # 실행 및 배포 스크립트
│   └── deployment/              # CI/CD 및 배포 스크립트 (황찬혁)
├── src/                         # 프로덕션 레벨 소스 코드
│   ├── backend/                 # FastAPI 백엔드 서버 로직 (황찬혁)
│   ├── common/                  # 공통 유틸리티 및 상수
│   ├── crawler/                 # 유튜브 메타데이터 및 콘텐츠 수집기 (이연우)
│   ├── embedding/               # SigLIP 2 & Ko-E5 임베딩 모델 (최정현)
│   ├── frontend/                # React/Next.js 웹 프론트엔드 (황찬혁)
│   ├── quantization/            # 모델 양자화 및 최적화 로직 (이연우)
│   ├── retrieval/               # 하이브리드 검색 및 Re-ranking 로직 (최정현)
│   └── vector_db/               # Pinecone Vector DB 연동 및 인덱스 관리
└── tests/                       # 유닛 테스트 및 통합 테스트
```

## Data (저장소에 포함하지 않음)

수집한 곡 데이터는 저작권이 있는 원문(가사·앨범 소개문·댓글)과 음원·앨범 커버, 그리고 거기서 뽑은
파생물이라 이 저장소에 올리지 않는다. 팀원은 팀 드라이브에서 받는다.

| 무엇 | 위치 | 받는 법 |
| --- | --- | --- |
| 곡 폴더 (`meta.json` · `audio.m4a` · `cover.jpg`) | `VAGUEFINDER_DATA_DIR` | 팀 드라이브 (`.env.example` 참고) |
| 임베딩 벡터 | `artifacts/embeddings/` | `docs/vector_backend.md` 절차 |
| 지도 화면 데이터 | `src/frontend/static/map_data.json` | 팀 드라이브 `vague-finder/map_data.json`, 또는 `venv/bin/python -m src.pipelines.build_map` (`data/all_songs.jsonl`과 임베딩 필요) |

`map_data.json`이 없으면 `/map` 화면이 곡을 그리지 못한다.

## Getting Started

1. **Environment Setup**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   ```

   ```

2. **For Data Engineers (Crawler)**
   - **Input**: Edit `songs.csv` (artist, title)
   - **Run**:
     ```bash
     python src/crawler/main.py
     ```
   - **Output**: `data/raw/{artist}__{title}_{video_id}/meta.json` & `cover.jpg`
   - **Features**:
     - Melon Metadata & Lyrics
     - YouTube Reaction & Vibe Check (Emoji Filter Applied)
     - LLM-based Vague Summary (Gemini)

3. **For AI Researchers**
   - Check `src/embedding/` and `notebooks/model_experiments/`
