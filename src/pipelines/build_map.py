"""
src/pipelines/build_map.py

노래 맵 좌표 생성 파이프라인입니다

맵은 두 종류를 만든다. 두 임베딩의 이웃 집합이 3.3%밖에 겹치지 않아
(소리가 닮은 곡과 정서가 닮은 곡이 서로 다른 집합) 한 장에 합치면
양쪽 다 망가진다. 그래서 탭으로 분리한다.

  sound: CLAP 오디오 512차원 → 소리가 닮은 곡이 모인다
  mood : KoE5 텍스트 1024차원 → 가사·정서가 닮은 곡이 모인다

단계:
  1) data/all_songs.jsonl에서 곡 메타데이터 로드
  2) artifacts/embeddings에서 곡 벡터 로드
     — 크롤링·임베딩이 끝나면 이 둘이 곧 최신이라 맵도 추가 적재 없이 따라온다.
       예전 경로(MongoDB + Pinecone)는 둘 다 수동 단계라 맵만 옛 곡 수에 멈춰 있었고,
       2026-09-29에 뺐다.
  3) UMAP으로 2D 투영
  4) 축 정렬 — 아래 align_sound_axes()·align_mood_axes() 참조
  5) 캔버스 픽셀 좌표로 스케일링 (곡 수에 비례해 면적 자동 확장 → 3000곡 대응)
  6) 라벨 충돌 완화(겹치는 제목 밀어내기)
  7) 프론트가 읽는 정적 JSON 산출: src/frontend/static/map_data.json

실행:
  venv/bin/python -m src.pipelines.build_map
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_JSON = PROJECT_ROOT / "src" / "frontend" / "static" / "map_data.json"

# 크롤링·임베딩이 끝나면 이 둘이 곧 최신이라, 맵도 추가 적재 없이 따라온다.
# (예전 MongoDB + Pinecone 경로에서는 검색은 952곡인데 맵은 905곡에 멈춰 있었다.)
LOCAL_CORPUS = PROJECT_ROOT / "data" / "all_songs.jsonl"
LOCAL_EMBEDDINGS = PROJECT_ROOT / "artifacts" / "embeddings"
LOCAL_MODALITY_DIR = {"audio": "audio", "text": "text_dense"}


# 축 정렬에 쓰는 태그. 이 태그를 가진 곡들의 벡터 차이로 축 방향을 학습한다.
ENERGY_POS = ("신나는", "활기찬", "역동적", "중독성", "강렬함")
ENERGY_NEG = ("잔잔함", "차분함", "먹먹함", "애절함", "서정적")
# mood 맵은 UMAP 축이 이미 해석 가능해서 방향(부호)만 맞추면 된다.
# 각 축의 한쪽 끝에 실제로 몰리는 태그로 고른다 (952곡·3,010곡 둘 다 확인).
MOOD_X_LEFT = ("두근거림", "설렘", "풋풋함", "상큼함", "달콤함")
MOOD_Y_TOP = ("자신감", "카리스마", "쾌감", "흥분", "해방감")

CHAR_WIDTH_PX = 13.0
LABEL_HEIGHT_PX = 20.0
CANVAS_WIDTH = 1800
AREA_PER_SONG = 6000  
MAX_DISPLACEMENT = 60.0  # 충돌 완화 시 원좌표에서 벗어날 수 있는 최대 px
MARGIN = 80  # 캔버스 가장자리 여백


def load_songs_from_jsonl(corpus: Path = LOCAL_CORPUS) -> list[dict[str, Any]]:
    """all_songs.jsonl에서 곡 리스트를 만든다."""
    if not corpus.is_file():
        raise RuntimeError(
            f"코퍼스를 찾지 못했습니다: {corpus}\n"
            f"팀 드라이브에서 all_songs.jsonl을 받아 data/ 아래에 두세요."
        )

    tag_fields = ("sound_tags", "mood_tags", "vibe_tags",
                  "time_weather_tags", "emotion_tags")
    songs: list[dict[str, Any]] = []
    with corpus.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            md = doc.get("metadata") or {}
            title = md.get("title")
            song_id = str(doc.get("song_id") or doc.get("id") or "").strip()
            if not title or not song_id:
                print(f"[load] 경고: title/song_id 없는 레코드 제외: {song_id or '?'}")
                continue
            links = doc.get("links") or {}
            sa = doc.get("semantic_analysis") or {}
            tags: set[str] = set()
            for field in tag_fields:
                tags.update(sa.get(field) or [])
            artist = md.get("artist") or []
            songs.append(
                {
                    "song_id": song_id,
                    "title": title,
                    "artist": ", ".join(artist) if isinstance(artist, list) else str(artist),
                    "genre": (md.get("genre") or ["기타"])[0],
                    # 폴더명이 없으면 정렬 키로 song_id를 쓴다(재현성만 필요하다).
                    "folder": doc.get("folder_name") or song_id,
                    "youtube_id": youtube_id(links.get("youtube_url")),
                    "cover_url": links.get("cover_url") or "",
                    "tags": sorted(tags),
                    "release_date": release_date(md.get("release_date")),
                }
            )
    if not songs:
        raise RuntimeError(f"{corpus}에서 곡을 찾지 못했습니다.")
    # 곡 순서는 UMAP 결과를 좌우하므로 고정해 재현성을 보장한다
    songs.sort(key=lambda s: s["folder"])
    return songs


def load_vectors_from_artifacts(
    song_ids: list[str],
    kind: str = "audio",
    base: Path = LOCAL_EMBEDDINGS,
) -> dict[str, np.ndarray]:
    """artifacts/embeddings에서 벡터를 읽는다 (기본 소스).

    모델 이름 폴더는 자동으로 찾는다 — 내보낸 환경에 따라 이름이 달라도 막히지 않게.
    """
    modality = base / LOCAL_MODALITY_DIR[kind]
    if not modality.is_dir():
        raise RuntimeError(
            f"임베딩 폴더가 없습니다: {modality}\n"
            f"팀 드라이브의 embeddings.zip을 artifacts/ 아래에 풀거나 임베딩을 먼저 돌리세요."
        )
    candidates = [d for d in sorted(modality.iterdir()) if d.is_dir() and any(d.glob("*.npy"))]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{modality} 아래 벡터 폴더가 {len(candidates)}개입니다"
            + (f": {[d.name for d in candidates]} — 쓸 폴더만 남겨주세요." if candidates else ".")
        )
    folder = candidates[0]

    wanted = set(song_ids)
    vectors: dict[str, np.ndarray] = {}
    for path in sorted(folder.glob("*.npy")):
        if path.stem in wanted:
            vectors[path.stem] = np.load(path).astype(np.float32).reshape(-1)
    missing = [sid for sid in song_ids if sid not in vectors]
    if missing:
        print(f"[load:{kind}] 경고: 벡터 없는 곡 {len(missing)}개 (예: {missing[:5]})")
    print(f"[load:{kind}] {folder.name}에서 {len(vectors)}곡")
    return vectors



def youtube_id(url: str | None) -> str:
    """YouTube URL에서 video id만 추출. 프론트가 임베드 주소를 조립한다."""
    if not url:
        return ""
    m = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else ""


def release_date(raw: Any) -> str:
    """발매일을 정렬 가능한 YYYY-MM-DD로. 파싱 실패 시 빈 문자열."""
    if not raw:
        return ""
    m = re.match(r"(\d{4})[.\-/]?(\d{2})?[.\-/]?(\d{2})?", str(raw).strip())
    if not m:
        return ""
    year, month, day = m.group(1), m.group(2) or "01", m.group(3) or "01"
    return f"{year}-{month}-{day}"


def display_title(title: str) -> str:
    """표시용 제목: 꼬리표 괄호 제거. 괄호로 시작하는 제목은 그대로 둔다."""
    t = unicodedata.normalize("NFC", title).strip()
    cut = re.split(r"\s*[(\[]", t, maxsplit=1)[0].strip()
    return cut if cut else t



def project_umap(matrix: np.ndarray, seed: int = 42) -> np.ndarray:
    """임베딩을 UMAP으로 2D 투영한다."""
    import umap

    n_neighbors = min(30, max(5, matrix.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=0.25,
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(matrix), dtype=np.float64)


def tag_matrix(songs: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    """곡별 태그를 멀티핫 행렬로. 축 방향 학습에 쓴다."""
    tag_idx: dict[str, int] = {}
    rows: list[set[str]] = []
    for s in songs:
        tags = set(s.get("tags") or [])
        rows.append(tags)
        for t in tags:
            tag_idx.setdefault(t, len(tag_idx))
    m = np.zeros((len(songs), len(tag_idx)), np.float32)
    for i, tags in enumerate(rows):
        for t in tags:
            m[i, tag_idx[t]] = 1.0
    return m, tag_idx


def _tag_score(axis: np.ndarray, m: np.ndarray, tag_idx: dict[str, int],
               words: tuple[str, ...]) -> float:
    """주어진 태그를 가진 곡들의 축 평균 위치."""
    vals = [axis[m[:, tag_idx[w]] > 0].mean() for w in words
            if w in tag_idx and (m[:, tag_idx[w]] > 0).any()]
    return float(np.mean(vals)) if vals else 0.0


def align_sound_axes(
    coords: np.ndarray,
    vectors: np.ndarray,
    m: np.ndarray,
    tag_idx: dict[str, int],
) -> np.ndarray:
    """sound 맵의 세로축을 '느린·어쿠스틱(위) → 빠른·전자음(아래)' 방향에 맞춰 돌린다.

    세로에 두는 이유: 캔버스는 폭을 고정하고 곡 수만큼 아래로 늘어난다
    (3,010곡이면 1800x10,033px). 가장 긴 방향이 곧 스크롤 방향이므로,
    아래로 내려갈수록 신나는 곡이 나오게 맞춘다.

    회전·뒤집기는 점 사이 거리를 바꾸지 않는 변환이라, UMAP이 만든 이웃 구조를
    그대로 둔 채 축 의미만 얻는다 (측정: 이웃보존 0.323 → 0.323).
    축을 직접 정의하는 방식은 해석력이 더 높지만 이웃보존이 0.09까지
    떨어져서 쓰지 않는다.
    """
    from sklearn.linear_model import Ridge

    # 태그로 '에너지' 방향 벡터를 학습
    y = np.zeros(len(vectors))
    for words, sign in ((ENERGY_POS, 1.0), (ENERGY_NEG, -1.0)):
        for w in words:
            if w in tag_idx:
                y[m[:, tag_idx[w]] > 0] += sign
    y = np.clip(y, -1.0, 1.0)
    labeled = y != 0
    if labeled.sum() < 30:
        print("[align] 경고: 에너지 태그가 부족해 회전을 건너뜁니다")
        return coords

    unit = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    w = Ridge(alpha=1.0).fit(unit[labeled], y[labeled]).coef_
    target = unit @ (w / (np.linalg.norm(w) + 1e-9))
    target = (target - target.mean()) / (target.std() + 1e-9)

    # 정규화한 UMAP 좌표를 돌려가며 target과 가장 잘 맞는 각도를 찾는다
    z = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)
    best_angle, best_corr = 0.0, 0.0
    for angle in np.linspace(0.0, np.pi, 721):
        proj = z @ np.array([np.cos(angle), np.sin(angle)])
        corr = float(np.corrcoef(proj, target)[0, 1])
        if abs(corr) > abs(best_corr):
            best_angle, best_corr = angle, corr

    # 찾은 방향으로 투영한 값을 그대로 y로 쓰고, x는 그와 직교하는 방향으로 둔다
    cos_a, sin_a = np.cos(best_angle), np.sin(best_angle)
    energy = z @ np.array([cos_a, sin_a])
    if best_corr < 0:
        energy = -energy  # 화면 y는 아래로 커진다 → 빠른·전자음 쪽이 항상 아래
    rotated = np.column_stack([z @ np.array([-sin_a, cos_a]), energy])
    print(f"[align] sound: 에너지 방향을 세로축에 맞춤 (상관 {abs(best_corr):.3f})")
    return rotated


def align_mood_axes(
    coords: np.ndarray,
    m: np.ndarray,
    tag_idx: dict[str, int],
) -> np.ndarray:
    """mood 맵은 UMAP 축이 이미 해석 가능하다. 방향(부호)만 맞춘다.

    측정된 축 의미 (3,010곡, 축 끝에 몰린 태그. 빌드할 때 report_axis_tags()가 다시 찍는다):
      x: 두근거림·상큼함·달콤함 ↔ 체념·비통함·우울
      y: 자신감·쾌감·해방감 ↔ 슬픔·그리움·연민
    """
    out = coords.copy()
    if _tag_score(out[:, 0], m, tag_idx, MOOD_X_LEFT) > np.median(out[:, 0]):
        out[:, 0] *= -1.0  # 들뜬 가사를 왼쪽으로
    if _tag_score(out[:, 1], m, tag_idx, MOOD_Y_TOP) > np.median(out[:, 1]):
        out[:, 1] *= -1.0  # 당당한 가사를 위로
    return out


def report_axis_tags(
    name: str,
    coords: np.ndarray,
    m: np.ndarray,
    tag_idx: dict[str, int],
    min_songs: int = 20,
    top: int = 5,
) -> None:
    """축 양 끝에 몰린 태그를 찍는다.

    축 라벨(payload의 axes)은 사람이 붙인 이름이라 곡이 바뀌어도 그대로 남는다.
    952곡에서 붙인 mood 세로축 라벨이 실제 배치와 달랐던 적이 있어,
    다시 빌드할 때마다 이 출력으로 라벨이 아직 맞는지 확인한다.
    """
    z = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)
    names = list(tag_idx)
    counts = m.sum(axis=0)
    keep = [j for j in range(len(names)) if counts[j] >= min_songs]
    for ax, (lo_side, hi_side) in enumerate((("왼", "오른"), ("위", "아래"))):
        means = sorted((float(z[m[:, j] > 0, ax].mean()), names[j]) for j in keep)
        lo = ", ".join(t for _, t in means[:top])
        hi = ", ".join(t for _, t in means[-top:][::-1])
        print(f"[axis] {name} {lo_side}: {lo}  |  {hi_side}: {hi}")


def scale_to_canvas(coords: np.ndarray, n_songs: int) -> tuple[np.ndarray, int, int]:
    """UMAP 좌표를 픽셀 좌표로 스케일. 곡 수에 비례해 캔버스 면적 확장."""
    width = CANVAS_WIDTH
    height = max(1200, int(n_songs * AREA_PER_SONG / width))
    xy = coords - coords.min(axis=0)
    span = xy.max(axis=0)
    span[span == 0] = 1.0
    xy[:, 0] = xy[:, 0] / span[0] * (width - 2 * MARGIN) + MARGIN
    xy[:, 1] = xy[:, 1] / span[1] * (height - 2 * MARGIN) + MARGIN
    return xy, width, height


def relax_collisions(
    xy: np.ndarray,
    widths: np.ndarray,
    iterations: int = 80,
) -> np.ndarray:
    """겹치는 라벨 박스를 서로 밀어낸다. 원좌표에서 MAX_DISPLACEMENT 이상 벗어나지 않음."""
    pos = xy.copy()
    origin = xy.copy()
    n = len(pos)
    cell = 160.0

    for it in range(iterations):
        moved = 0.0
        grid: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            key = (int(pos[i, 0] // cell), int(pos[i, 1] // cell))
            grid.setdefault(key, []).append(i)

        for (cx, cy), members in grid.items():
            neighbors: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbors.extend(grid.get((cx + dx, cy + dy), []))
            for i in members:
                for j in neighbors:
                    if j <= i:
                        continue
                    ox = (widths[i] + widths[j]) / 2 - abs(pos[i, 0] - pos[j, 0])
                    oy = LABEL_HEIGHT_PX - abs(pos[i, 1] - pos[j, 1])
                    if ox <= 0 or oy <= 0:
                        continue
                    # 세로로 밀어내는 게 가로로 미는 것보다 왜곡이 적다 (라벨이 납작해서)
                    if oy * 3 < ox:
                        push, axis = oy / 2 + 0.5, 1
                    else:
                        push, axis = min(ox, oy) / 2 + 0.5, 1 if oy < ox else 0
                    sign = 1.0 if pos[i, axis] >= pos[j, axis] else -1.0
                    pos[i, axis] += sign * push
                    pos[j, axis] -= sign * push
                    moved += push

        # 원좌표 이탈 제한
        delta = pos - origin
        dist = np.linalg.norm(delta, axis=1)
        over = dist > MAX_DISPLACEMENT
        if over.any():
            pos[over] = origin[over] + delta[over] * (MAX_DISPLACEMENT / dist[over])[:, None]

        if moved < 1.0:
            print(f"[relax] {it + 1}회 반복 후 수렴")
            break
    return pos


def layout(
    coords: np.ndarray,
    labels: list[str],
    n_songs: int,
) -> tuple[np.ndarray, int, int]:
    """정렬된 2D 좌표를 픽셀로 옮기고 라벨 충돌을 완화한다."""
    xy, width, height = scale_to_canvas(coords, n_songs)
    widths = np.array([max(3, len(t)) * CHAR_WIDTH_PX for t in labels])
    return relax_collisions(xy, widths), width, height


def build() -> None:
    songs = load_songs_from_jsonl()
    print(f"[load] {LOCAL_CORPUS.name}에서 {len(songs)}곡 로드")

    song_ids = [s["song_id"] for s in songs]
    audio = load_vectors_from_artifacts(song_ids, kind="audio")
    text = load_vectors_from_artifacts(song_ids, kind="text")

    # 두 맵 모두에 그릴 수 있는 곡만 남긴다 (탭 전환 시 곡이 사라지지 않도록)
    songs = [s for s in songs if s["song_id"] in audio and s["song_id"] in text]
    if not songs:
        raise RuntimeError("오디오·텍스트 벡터를 모두 가진 곡이 없습니다.")
    print(f"[embed] 두 벡터를 모두 가진 곡: {len(songs)}곡")

    labels = [display_title(s["title"]) for s in songs]
    tags, tag_idx = tag_matrix(songs)

    va = np.stack([audio[s["song_id"]] for s in songs])
    vt = np.stack([text[s["song_id"]] for s in songs])
    print(f"[embed] sound {va.shape}  mood {vt.shape}")

    # sound: UMAP 후 회전 정렬 (이웃 구조 보존)
    sound = align_sound_axes(project_umap(va), va, tags, tag_idx)
    report_axis_tags("sound", sound, tags, tag_idx)
    sound_xy, width, height = layout(sound, labels, len(songs))

    # mood: UMAP 축이 이미 해석 가능하므로 방향만 정리
    mood = align_mood_axes(project_umap(vt), tags, tag_idx)
    report_axis_tags("mood", mood, tags, tag_idx)
    mood_xy, _, _ = layout(mood, labels, len(songs))
    print(f"[canvas] {width}x{height}px")

    items = [
        {
            "id": s["song_id"],
            "t": labels[i],
            "ft": s["title"],
            "a": s["artist"],
            "g": s["genre"],
            "x": round(float(sound_xy[i, 0]), 1),
            "y": round(float(sound_xy[i, 1]), 1),
            "mx": round(float(mood_xy[i, 0]), 1),
            "my": round(float(mood_xy[i, 1]), 1),
            "d": s.get("release_date", ""),
            "yt": s.get("youtube_id", ""),
            "c": s.get("cover_url", ""),
        }
        for i, s in enumerate(songs)
    ]
    payload = {
        "canvas": {"width": width, "height": height},
        "count": len(items),
        # 프론트가 축 라벨을 그릴 때 쓴다. 라벨이 맞는지는 빌드 로그의 [axis] 줄로 확인한다
        # lines: 어느 방향의 기준선을 그릴지. "x"=가로선, "y"=세로선
        "axes": {
            "sound": {"x": ["", ""], "y": ["느린 · 어쿠스틱", "빠른 · 전자음"],
                      "lines": ["x"]},
            "mood": {"x": ["들뜬 가사", "먹먹한 가사"],
                     "y": ["당당한", "애절한"],
                     "lines": ["x", "y"]},
        },
        "songs": items,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[out] {OUTPUT_JSON} ({len(items)}곡, sound+mood)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="노래 맵 좌표 생성 (data/all_songs.jsonl + artifacts/embeddings)"
    )
    parser.parse_args()
    build()
