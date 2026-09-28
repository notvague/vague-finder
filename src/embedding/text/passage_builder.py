"""
text/passage_builder.py
- 곡 메타데이터를 Ko-E5 입력용 passage 텍스트로 표준화하는 모듈

[변경사항]
- Dense(Ko-E5)용 passage와 Sparse(BM25)용 passage를 분리한다.

Dense:
- search_style_summary, lyrics_summary를 backbone으로 사용
- 태그/이미지/장르 정보는 "라벨: 값" 형태의 반정형 블록으로 붙인다
- 원문 손실을 최소화하는 방향으로 구성

Sparse(BM25):
- 키워드 중심 블록으로 구성
- 문장형 요약은 제외하고, 검색될 가능성이 높은 표면 단서를 최대한 보존
- title/artist/genre/type/vocal_gender/mood_tags/time_weather_tags/place_activity_tags/
  emotion_tags/vibe_tags/relation_context_tags/color_tags/sound_tags/melon_playlist_tags/
  visual_imagery/fans_tags/major_emotion 을 반영
"""
from __future__ import annotations

from typing import Dict, Any, List


def _join_tags(tags: Any, *, sep: str = " ") -> str:
    """tags가 list[str]면 join, str이면 그대로, None이면 빈 문자열."""
    if tags is None:
        return ""
    if isinstance(tags, list):
        return sep.join([str(t).strip() for t in tags if t is not None and str(t).strip()])
    return str(tags).strip()


def _safe_text(x: Any) -> str:
    """None/비문자 입력을 안전하게 문자열로 변환."""
    if x is None:
        return ""
    return str(x).strip()

def _as_list(x: Any) -> List[str]:
    """값을 BM25용 토큰 리스트로 정규화."""
    if x is None:
        return []

    if isinstance(x, list):
        return [str(v).strip() for v in x if v is not None and str(v).strip()]

    s = str(x).strip()
    if not s:
        return []

    # 문자열 하나로 들어오는 경우 그대로 1개 항목으로 취급
    return [s]

def _get_first_nonempty(dicts: List[Dict[str, Any]], key: str) -> Any:
    """여러 dict를 순서대로 보며 가장 먼저 나오는 non-empty 값을 반환."""
    for d in dicts:
        if not isinstance(d, dict):
            continue
        value = d.get(key)
        if value is None:
            continue
        if isinstance(value, list) and value:
            return value
        if str(value).strip():
            return value
    return None

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        token = str(item).strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out

def _keyword_block(x: Any) -> str:
    """
    BM25용 키워드 블록 문자열 생성.
    - list면 공백 join
    - str이면 그대로
    - 중복 제거
    """
    items = _dedupe_keep_order(_as_list(x))
    return " ".join(items)

# -----------------------------
# Dense passage (Ko-E5)
# -----------------------------
def build_dense_passage(song: Dict) -> str:
    """
    Ko-E5 입력용 dense passage 생성.

    구성:
    - passage: {search_style_summary} {lyrics_summary}
    - 가사 : {lyrics_highlight}
    - 종합 무드 : {mood_tags}
    - 감정: {emotion_tags}
    - 시청자 반응: {sentiment_summary}
    - 관계 맥락: {relation_context_tags}
    - 시간/날씨: {time_weather_tags}
    - 장소/활동: {place_activity_tags}
    - 분위기: {vibe_tags}
    - 사운드: {sound_tags}
    - 시각 이미지: {visual_imagery}
    - 보컬/형태: {vocal_gender} {type}
    - 장르: {genre}

    원칙:
    - search_style_summary, lyrics_summary는 그대로 사용
    - 태그는 해석/요약하지 않고 라벨만 붙여 반정형으로 유지
    - 빈 값인 섹션은 포함하지 않음
    """

    md = song.get("metadata", {})
    lyrics_data = song.get("lyrics_data", {})
    semantic = song.get("semantic_analysis", {})
    community = song.get("community_feedback", {})

    search_style_summary = _safe_text(semantic.get("search_style_summary"))
    lyrics_summary = _safe_text(lyrics_data.get("lyrics_summary"))

    lyrics_highlight = _safe_text(lyrics_data.get("lyrics_highlight"))

    mood_tags = _join_tags(semantic.get("mood_tags"))
    sentiment_summary = _safe_text(community.get("sentiment_summary"))
    emotion_tags = _join_tags(semantic.get("emotion_tags"))
    relation_context_tags = _join_tags(semantic.get("relation_context_tags"))
    time_weather_tags = _join_tags(semantic.get("time_weather_tags"))
    place_activity_tags = _join_tags(semantic.get("place_activity_tags"))
    vibe_tags = _join_tags(semantic.get("vibe_tags"))
    sound_tags = _join_tags(semantic.get("sound_tags"))
    visual_imagery = _join_tags(semantic.get("visual_imagery"))
    
    vocal_gender = _safe_text(md.get("vocal_gender"))
    song_type = _join_tags(md.get("type"))
    genre = _join_tags(md.get("genre"))

    vocal_type_parts: List[str] = []
    if vocal_gender:
        vocal_type_parts.append(vocal_gender)
    if song_type:
        vocal_type_parts.append(song_type)
    vocal_type_text = " ".join(vocal_type_parts)

    # 최종 passage 생성
    parts: List[str] = []

    # backbone: search_style_summary + lyrics_summary
    summary_parts: List[str] = []
    if search_style_summary:
        summary_parts.append(search_style_summary)
    if lyrics_summary:
        summary_parts.append(lyrics_summary)

    if summary_parts:
        parts.append(" ".join(summary_parts))

    # 반정형 라벨 블록
    if lyrics_highlight:
        parts.append(f"가사: {lyrics_highlight}")

    if mood_tags:
        parts.append(f"종합 무드: {mood_tags}")

    if emotion_tags:
        parts.append(f"감정: {emotion_tags}")

    if sentiment_summary:
        parts.append(f"사용자 반응: {sentiment_summary}")

    if relation_context_tags:
        parts.append(f"관계 맥락: {relation_context_tags}")

    if time_weather_tags:
        parts.append(f"시간/날씨: {time_weather_tags}")

    if place_activity_tags:
        parts.append(f"장소/활동: {place_activity_tags}")

    if vibe_tags:
        parts.append(f"분위기: {vibe_tags}")

    if sound_tags:
        parts.append(f"사운드: {sound_tags}")

    if visual_imagery:
        parts.append(f"시각 이미지: {visual_imagery}")

    if vocal_type_text:
        parts.append(f"보컬/형태: {vocal_type_text}")

    if genre:
        parts.append(f"장르: {genre}")

    # 모든 필드가 비었을 때의 최후 fallback
    if not parts:
        parts.append("[EMPTY]\n(no semantic/lyrics data)")

    body = "\n".join(parts)
    return f"passage: {body}"


# -----------------------------
# Sparse passage (BM25)
# -----------------------------
def build_sparse_passage(song: Dict) -> str:
    """
    BM25 입력용 sparse passage 생성(키워드 중심).

    목표:
    - 문장형 설명이 아니라 '실제로 검색될 법한 표면 단서'를 최대한 보존
    - 짧은 키워드 블록 위주로 구성
    - 라벨은 제거하고 줄바꿈 기반 블록으로 정리

    포함 필드:
    - title, artist, genre, type, vocal_gender
    - mood_tags, time_weather_tags, place_activity_tags
    - emotion_tags, vibe_tags, relation_context_tags
    - color_tags, sound_tags, melon_playlist_tags
    - visual_imagery, fans_tags, major_emotion
    """
    md = song.get("metadata", {})
    lyrics_data = song.get("lyrics_data", {})
    semantic = song.get("semantic_analysis", {})
    community = song.get("community_feedback", {})
    search_inputs = song.get("search_vector_inputs", {})

    sources = [md, semantic, community, search_inputs]

    title = _keyword_block(_get_first_nonempty(sources, "title"))
    artist = _keyword_block(_get_first_nonempty(sources, "artist"))
    # 앨범명·앨범 요약·시청자 반응은 #52에서 sparse에 넣어 성능을 올린 필드다.
    # #58이 나무위키를 되돌리면서 함께 빠졌고, 그 상태로 벡터를 다시 만들면
    # dev Hit@10이 0.830 → 0.774로 떨어진다(2026-09-17 A/B 측정).
    # 나무위키(external_context)는 여전히 넣지 않는다.
    album = _keyword_block(_get_first_nonempty(sources, "album"))
    album_summary = _safe_text(_get_first_nonempty(sources, "album_summary"))
    sentiment_summary = _safe_text(_get_first_nonempty(sources, "sentiment_summary"))
    genre = _keyword_block(_get_first_nonempty(sources, "genre"))
    song_type = _keyword_block(_get_first_nonempty(sources, "type"))
    vocal_gender = _keyword_block(_get_first_nonempty(sources, "vocal_gender"))
    full_lyrics = _safe_text(lyrics_data.get("full_lyrics"))
    mood_tags = _keyword_block(_get_first_nonempty(sources, "mood_tags"))
    time_weather_tags = _keyword_block(_get_first_nonempty(sources, "time_weather_tags"))
    place_activity_tags = _keyword_block(_get_first_nonempty(sources, "place_activity_tags"))
    emotion_tags = _keyword_block(_get_first_nonempty(sources, "emotion_tags"))
    vibe_tags = _keyword_block(_get_first_nonempty(sources, "vibe_tags"))
    relation_context_tags = _keyword_block(_get_first_nonempty(sources, "relation_context_tags"))
    color_tags = _keyword_block(_get_first_nonempty(sources, "color_tags"))
    sound_tags = _keyword_block(_get_first_nonempty(sources, "sound_tags"))
    melon_playlist_tags = _keyword_block(_get_first_nonempty(sources, "melon_playlist_tags"))
    visual_imagery = _keyword_block(_get_first_nonempty(sources, "visual_imagery"))
    fans_tags = _keyword_block(_get_first_nonempty(sources, "fans_tags"))
    major_emotion = _keyword_block(_get_first_nonempty(sources, "major_emotion"))

    parts: List[str] = []
    
    # 1) 식별 블록: 제목/가수는 가장 앞
    identity_block: List[str] = []
    if title:
        identity_block.append(title)
    if artist:
        identity_block.append(artist)
    if identity_block:
        parts.append("\n".join(identity_block))

    # 1-1) 앨범·시청자 반응: 곡이 놓인 맥락. 제목/가수만으로 안 걸리는 질의를 잡는다
    context_block: List[str] = []
    if album:
        context_block.append(album)
    if album_summary:
        context_block.append(album_summary)
    if sentiment_summary:
        context_block.append(sentiment_summary)
    if context_block:
        parts.append("\n".join(context_block))

    # 2) 장르/유형/보컬 성별: 비교적 자주 검색될 수 있는 기본 메타
    meta_block: List[str] = []
    if genre:
        meta_block.append(genre)
    if song_type:
        meta_block.append(song_type)
    if vocal_gender:
        meta_block.append(vocal_gender)
    if meta_block:
        parts.append("\n".join(meta_block))

    # 3) 대표 감정. BM25 점수 계산에는 위치 가중치가 없다. 다만 passage 전체를 한 번에
    #    Kiwi로 분석하므로, 이 줄의 위치나 내용이 이 줄과 인접 줄의 토큰을 바꿀 수 있다.
    #    dense passage에는 들어가지 않는다.
    if major_emotion:
        parts.append(major_emotion)

    # 4) 감정/분위기/관계: 기억 기반 질의에서 핵심
    semantic_core_block: List[str] = []
    if mood_tags:
        semantic_core_block.append(mood_tags)
    if emotion_tags:
        semantic_core_block.append(emotion_tags)
    if vibe_tags:
        semantic_core_block.append(vibe_tags)
    if relation_context_tags:
        semantic_core_block.append(relation_context_tags)
    if semantic_core_block:
        parts.append("\n".join(semantic_core_block))

    # 5) 시간/장소: 상황 기억형 질의 대응
    scene_block: List[str] = []
    if time_weather_tags:
        scene_block.append(time_weather_tags)
    if place_activity_tags:
        scene_block.append(place_activity_tags)
    if scene_block:
        parts.append("\n".join(scene_block))

    # 6) 청각/시각 단서
    sensory_block: List[str] = []
    if sound_tags:
        sensory_block.append(sound_tags)
    if color_tags:
        sensory_block.append(color_tags)
    if visual_imagery:
        sensory_block.append(visual_imagery)
    if sensory_block:
        parts.append("\n".join(sensory_block))

    # 7) 사용자/플레이리스트 태그: recall 보강용
    social_block: List[str] = []
    if melon_playlist_tags:
        social_block.append(melon_playlist_tags)
    if fans_tags:
        social_block.append(fans_tags)
    if social_block:
        parts.append("\n".join(social_block))

    # 8) 전체 가사
    if full_lyrics:
        parts.append(full_lyrics)

    if not parts:
        parts.append("EMPTY")

    return "\n\n".join(parts)