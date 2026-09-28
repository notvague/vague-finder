# Python 3.11 Slim 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수 설정
# PYTHONDONTWRITEBYTECODE: 파이썬 .pyc 파일 생성 방지
# PYTHONUNBUFFERED: 버퍼링 없이 즉시 로그 출력
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# 시스템 의존성 설치 (필요한 경우)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Browser runtime outside /app so the Compose bind mount cannot hide it.
RUN python -m playwright install --with-deps chromium

# NLTK 검색 데이터 설치 — GitHub raw 429를 피하기 위해 CDN 사용
RUN python -c "import io,urllib.request,zipfile; from pathlib import Path; root=Path('/usr/local/share/nltk_data'); packages=[('tokenizers','punkt_tab'),('corpora','stopwords')]; [(root/category).mkdir(parents=True,exist_ok=True) or zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(f'https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/{category}/{name}.zip',timeout=60).read())).extractall(root/category) for category,name in packages]"

# 오디오 디코딩은 위에서 설치한 ffmpeg가 한다(src/embedding/audio/audio_io.py).
# torchcodec은 설치하지 않는다 — FFmpeg 4~8용 바이너리만 들고 오므로 그보다 새 ffmpeg가
# 깔린 환경에서는 로드되지 않고, torchaudio.load가 곡마다 실패하며 로그만 쌓인다.

# 소스 코드 복사
COPY . .

# 실행 명령 (docker-compose에서 오버라이딩 가능)
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
