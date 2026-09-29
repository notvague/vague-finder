"""
src/vector_db/metadata_loader.py

메타데이터를 다양한 소스에서 로드해 id -> metadata dict로 만든다.

data_dir 스캔:
   - data_dir 아래에 meta.json 파일들을 찾아 로드
   - 파일 내용에서 id를 찾아 key로 사용

메타데이터 정제:
- allowlist 기반 필터링 + 문자열 길이 제한을 upsert.py와 동일 정책으로 적용.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
import json

import numpy as np
from src.common.title_features import analyze_title_structure
from src.vector_db.settings import METADATA_ALLOWLIST, MAX_METADATA_STR_LEN


def _sanitize_metadata_value(v: Any) -> Any:
    """
    src/vector_db/upsert.py 와 동일한 정책으로 metadata value 정제.
    """
    if v is None:
        return None

    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)

    if isinstance(v, str):
        s = v.strip()
        return s[:MAX_METADATA_STR_LEN] if len(s) > MAX_METADATA_STR_LEN else s

    if isinstance(v, (int, float, bool)):
        return v

    if isinstance(v, list):
        return [_sanitize_metadata_value(x) for x in v if x is not None]

    if isinstance(v, dict):
        out: Dict[str, Any] = {}
        for kk, vv in v.items():
            if kk is None:
                continue
            out[str(kk)] = _sanitize_metadata_value(vv)
        return out

    s = str(v)
    return s[:MAX_METADATA_STR_LEN] if len(s) > MAX_METADATA_STR_LEN else s


def _filter_allowlist(md: Dict[str, Any]) -> Dict[str, Any]:
    """
    allowlist 필드만 남기고 정제한 metadata 반환.
    """
    out: Dict[str, Any] = {}
    for k in METADATA_ALLOWLIST:
        if k in md and md[k] is not None:
            sv = _sanitize_metadata_value(md[k])
            if sv is not None:
                out[k] = sv
    return out


def _with_title_features(md: Dict[str, Any]) -> Dict[str, Any]:
    """사전 생성 metadata JSON에도 최신 제목 파생 필드를 보강한다."""
    enriched = dict(md)
    features = analyze_title_structure(enriched.get("title"))
    mapping = {
        "title_script": "script",
        "title_char_count": "char_count",
        "title_word_count": "word_count",
        "title_contains_number": "contains_number",
        "title_repeated_char": "repeated_char",
        "title_repeated_word": "repeated_word",
        "title_has_latin_anywhere": "has_latin_anywhere",
        "title_has_hangul_anywhere": "has_hangul_anywhere",
        "title_has_hanja_anywhere": "has_hanja_anywhere",
        "title_has_number_anywhere": "has_number_anywhere",
    }
    for metadata_key, feature_key in mapping.items():
        enriched[metadata_key] = features[feature_key]
    return enriched


def load_metadata_from_json(metadata_json: Path) -> Dict[str, Dict[str, Any]]:
    """
    단일 JSON 파일에서 id -> metadata dict 로드.
    """
    p = Path(metadata_json)
    if not p.exists():
        raise FileNotFoundError(f"metadata_json not found: {p}")

    obj = json.loads(p.read_text(encoding="utf-8"))

    out: Dict[str, Dict[str, Any]] = {}

    # (A) {"id": {...}, ...}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, dict):
                out[str(k)] = _filter_allowlist(_with_title_features(v))
        return out

    # (B) [{"id": "...", "metadata": {...}}, ...]
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            tid = item.get("id")
            md = item.get("metadata") or item
            if tid is None or not isinstance(md, dict):
                continue
            out[str(tid)] = _filter_allowlist(_with_title_features(md))
        return out

    raise ValueError("Unsupported metadata_json format. Use dict{id->metadata} or list of dicts.")


def metadata_from_record(obj: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """meta.json / all_songs.jsonl 한 건을 (id, 벡터 DB 메타데이터)로 바꾼다.

    meta.json 스캔과 jsonl 적재가 **같은 필드**를 만들어야 한다. 두 경로가 갈리면
    벡터 DB를 바꿀 때 검색 결과의 메타데이터가 달라진다.
    """
    if not isinstance(obj, dict):
        return None

    # 현재 meta.json은 id가 아니라 song_id를 사용함
    tid = obj.get("id") or obj.get("song_id")
    if tid is None:
        return None
    tid = str(tid).strip()
    if not tid:
        return None

    base_md = obj.get("metadata") or {}
    lyrics = obj.get("lyrics_data") or {}
    semantic = obj.get("semantic_analysis") or {}
    community = obj.get("community_feedback") or {}

    if not isinstance(base_md, dict):
        base_md = {}
    if not isinstance(lyrics, dict):
        lyrics = {}
    if not isinstance(semantic, dict):
        semantic = {}
    if not isinstance(community, dict):
        community = {}

    title = base_md.get("title")
    title_features = analyze_title_structure(title)

    md_raw = {
        # metadata
        "title": base_md.get("title"),
        "artist": base_md.get("artist"),
        "release_date": base_md.get("release_date"),
        "genre": base_md.get("genre"),
        "type": base_md.get("type"),
        "vocal_gender": base_md.get("vocal_gender"),

        # derived title structure
        "title_script": title_features["script"],
        "title_char_count": title_features["char_count"],
        "title_word_count": title_features["word_count"],
        "title_contains_number": title_features["contains_number"],
        "title_repeated_char": title_features["repeated_char"],
        "title_repeated_word": title_features["repeated_word"],
        "title_has_latin_anywhere": title_features["has_latin_anywhere"],
        "title_has_hangul_anywhere": title_features["has_hangul_anywhere"],
        "title_has_hanja_anywhere": title_features["has_hanja_anywhere"],
        "title_has_number_anywhere": title_features["has_number_anywhere"],

        # lyrics_data
        "lyrics_highlight": lyrics.get("lyrics_highlight"),
        "lyrics_summary": lyrics.get("lyrics_summary"),

        # semantic_analysis
        "search_style_summary": semantic.get("search_style_summary"),
        "mood_tags": semantic.get("mood_tags"),
        "time_weather_tags": semantic.get("time_weather_tags"),
        "place_activity_tags": semantic.get("place_activity_tags"),
        "emotion_tags": semantic.get("emotion_tags"),
        "vibe_tags": semantic.get("vibe_tags"),
        "relation_context_tags": semantic.get("relation_context_tags"),
        "color_tags": semantic.get("color_tags"),
        "sound_tags": semantic.get("sound_tags"),
        "melon_playlist_tags": semantic.get("melon_playlist_tags"),
        "visual_imagery": semantic.get("visual_imagery"),

        # community_feedback
        "sentiment_summary": community.get("sentiment_summary"),
        "fans_tags": community.get("fans_tags"),
        "major_emotion": community.get("major_emotion"),
    }

    return tid, _filter_allowlist(md_raw)


def load_metadata_from_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    """all_songs.jsonl에서 id -> metadata dict 로드 (meta.json 스캔과 동일 필드)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl not found: {p}")

    out: Dict[str, Dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = metadata_from_record(json.loads(line))
            except Exception as e:
                print(f"[META][SKIP] failed to parse line: {e}")
                continue
            if parsed:
                out[parsed[0]] = parsed[1]

    print(f"[META] jsonl={p} loaded_records={len(out)}")
    return out


def load_metadata_from_data_dir(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    data_dir 아래의 **/meta.json 을 전부 스캔하여 id -> metadata dict로 만든다.

    벡터 DB payload에는 지정한 일부 필드만 저장한다(metadata_from_record).
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"data_dir not found: {base}")

    out: Dict[str, Dict[str, Any]] = {}
    meta_paths = list(base.rglob("meta.json"))

    for meta_path in meta_paths:
        try:
            obj = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[META][SKIP] failed to read {meta_path}: {e}")
            continue

        parsed = metadata_from_record(obj)
        if parsed is None:
            print(f"[META][SKIP] no id/song_id: {meta_path}")
            continue
        out[parsed[0]] = parsed[1]

    print(f"[META] scanned meta.json files={len(meta_paths)} loaded_records={len(out)}")
    return out


def build_metadata_map(
    *,
    data_dir: Optional[Path] = None,
    metadata_json: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    metadata_map 생성.
    우선순위:
      1) metadata_json 제공 시 그걸 사용
      2) 아니면 data_dir 제공 시 스캔
      3) 아니면 빈 dict 반환(메타데이터 없이 upsert)

    NOTE:
    - 둘 다 주면 metadata_json 우선.
    """
    if metadata_json is not None:
        return load_metadata_from_json(metadata_json)

    if data_dir is not None:
        return load_metadata_from_data_dir(data_dir)

    return {}
