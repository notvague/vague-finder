"""
src/vector_db/settings.py

- Vector DB(Qdrant) 설정 모음. 컬렉션 이름은 `<INDEX_NAME>__<NAMESPACE>`다
  (qdrant_backend.collection_name).

- 기존 멀티모달 인덱스 3개
  1) TEXT_HYBRID_INDEX_NAME: KoE5 dense + BM25 sparse 함께 저장(하이브리드용)
  2) IMAGE_INDEX_NAME: SigLIP2 벡터 저장
  3) AUDIO_INDEX_NAME: CLAP 벡터 저장

- 나무위키 context 전용 alias 2개
  - CONTEXT_DENSE_INDEX_NAME: 사실 단위 KoE5
  - CONTEXT_SPARSE_INDEX_NAME: 곡 단위 BM25 profile
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Index names / namespace
# 기본값을 바꾸면 이미 적재한 로컬 컬렉션을 찾지 못한다 — 다시 적재해야 한다.
# -----------------------------
TEXT_HYBRID_INDEX_NAME = os.getenv("QDRANT_TEXT_COLLECTION", "vaguefinder-text-1024-koe5")
IMAGE_INDEX_NAME = os.getenv("QDRANT_IMAGE_COLLECTION", "vaguefinder-image-768")
AUDIO_INDEX_NAME = os.getenv("QDRANT_AUDIO_COLLECTION", "vaguefinder-audio-512")

# Namuwiki context uses different retrieval units from the existing song index:
# dense is one point per fact, while sparse is one point per song profile.  Keep
# both in dedicated collections so fact counts cannot bias the existing music
# search or the song-level BM25 score.
CONTEXT_DENSE_INDEX_NAME = os.getenv(
    "CONTEXT_DENSE_INDEX_NAME", "vaguefinder-context-dense-1024-koe5"
)
CONTEXT_SPARSE_INDEX_NAME = os.getenv(
    "CONTEXT_SPARSE_INDEX_NAME", "vaguefinder-context-sparse-bm25"
)

NAMESPACE = os.getenv("QDRANT_NAMESPACE", "dev")

# -----------------------------
# Vector dimensions (must match embedder outputs)
# KoE5: 1024, SigLIP2: 768 (프로젝트 기본값)
# -----------------------------
TEXT_DENSE_DIM = int(os.getenv("TEXT_DENSE_DIM", "1024"))
IMAGE_DIM = int(os.getenv("IMAGE_DIM", "768"))
AUDIO_DIM = int(os.getenv("AUDIO_DIM", "512"))
CONTEXT_DENSE_DIM = int(os.getenv("CONTEXT_DENSE_DIM", str(TEXT_DENSE_DIM)))

# A moderate default bounds memory for the full catalogue while still avoiding
# one-request-per-fact overhead.
CONTEXT_QDRANT_BATCH_SIZE = int(os.getenv("CONTEXT_QDRANT_BATCH_SIZE", "256"))

# -----------------------------
# Metadata selection
# -----------------------------
METADATA_ALLOWLIST = {
    # basic metadata
    "title",
    "artist",
    "release_date",
    "genre",
    "type",
    "vocal_gender",

    # derived title structure
    "title_script",
    "title_char_count",
    "title_word_count",
    "title_contains_number",
    "title_repeated_char",
    "title_repeated_word",
    "title_has_latin_anywhere",
    "title_has_hangul_anywhere",
    "title_has_hanja_anywhere",
    "title_has_number_anywhere",

    # lyrics
    "lyrics_highlight",
    "lyrics_summary",

    # semantic_analysis
    "search_style_summary",
    "mood_tags",
    "time_weather_tags",
    "place_activity_tags",
    "emotion_tags",
    "vibe_tags",
    "relation_context_tags",
    "color_tags",
    "sound_tags",
    "melon_playlist_tags",
    "visual_imagery",

    # community_feedback
    "sentiment_summary",
    "fans_tags",
    "major_emotion",
}

# 문자열 길이 제한 (긴 summary/comment_context 대비)
MAX_METADATA_STR_LEN = int(os.getenv("METADATA_MAX_STR_LEN", "2000"))
