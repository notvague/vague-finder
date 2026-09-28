"""앨범 표지를 **직접 보고** 라벨을 붙이기 위한 도구.

왜 필요한가. 표지 질의의 정답은 곡의 분위기가 아니라 **실제 cover.jpg**로 정해야
한다. 2026-05 재라벨링에서 "어두운/실루엣" 정답 3곡이 전부 어둡지 않았고
(STATUS.md S-5), 2026-09-24 확인에서도 m402의 "어둡고 차가운 커버" 정답이
겨울연가 OST — 밝고 따뜻한 사진이었다. 곡을 듣고 상상한 표지를 적으면 이렇게 된다.

`meta.json`의 `color_tags`·`visual_imagery`도 **쓰면 안 된다.** LLM이 곡을 읽고
상상한 값이라 표지와 무관하다(STATUS.md B-4).

그래서 952장을 사람이 봐야 하는데, 그대로 보기에는 많다. 이 도구는 둘을 한다.

1. `scan` — 표지마다 모델 없이 픽셀 통계를 낸다(채도·명도·색상 분포·알록달록함).
   검색 시스템을 쓰지 않으므로 **평가할 대상으로 평가 라벨을 만드는 순환**이 없다.
2. `sheet` — 고른 곡들을 번호 붙인 한 장의 대조표로 묶는다. 한 번에 수십 장을 본다.

    venv/bin/python -m src.eval.cover_survey scan
    venv/bin/python -m src.eval.cover_survey sheet --ids 523274,4097003 --out /tmp/a.png

통계는 후보를 **좁히는 데만** 쓴다. 정답 여부는 대조표를 눈으로 보고 정한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# 곡 폴더는 드라이브 마운트에 있고 경로는 .env의 VAGUEFINDER_DATA_DIR이다.
load_dotenv()

DEFAULT_RAW = Path(
    os.getenv("VAGUEFINDER_DATA_DIR", "data/raw")
).expanduser()
DEFAULT_STATS = Path("artifacts/cover_stats.json")

# 통계를 낼 때 줄이는 크기. 색 분포만 보므로 원본 해상도가 필요 없다.
THUMB = 96


def _song_dirs(raw: Path) -> Dict[str, Path]:
    """폴더 이름 끝의 숫자가 곡 id다."""
    found: Dict[str, Path] = {}
    for entry in sorted(raw.iterdir()):
        if not entry.is_dir():
            continue
        match = re.search(r"_(\d+)$", entry.name)
        if match:
            found[match.group(1)] = entry
    return found


def _stats(path: Path) -> Dict[str, float]:
    """표지 한 장의 픽셀 통계. 모델을 쓰지 않는다."""
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((THUMB, THUMB), Image.BILINEAR)
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 색이 거의 없는 픽셀의 색상값은 잡음이다. 채도가 있는 픽셀만 색상을 센다.
    colored = sat > 0.25
    share: Dict[str, float] = {}
    if colored.any():
        h = hue[colored]
        buckets = {
            "red": ((h < 0.042) | (h >= 0.958)),
            "orange": ((h >= 0.042) & (h < 0.11)),
            "yellow": ((h >= 0.11) & (h < 0.19)),
            "green": ((h >= 0.19) & (h < 0.44)),
            "blue": ((h >= 0.44) & (h < 0.75)),
            "purple": ((h >= 0.75) & (h < 0.88)),
            "pink": ((h >= 0.88) & (h < 0.958)),
        }
        total = float(colored.sum())
        share = {name: float(mask.sum()) / total for name, mask in buckets.items()}

    # Hasler-Süsstrunk colorfulness — "알록달록한가"의 흔한 근사식
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorfulness = float(
        math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
    )

    return {
        "sat_mean": float(sat.mean()),
        "sat_p90": float(np.percentile(sat, 90)),
        "val_mean": float(val.mean()),
        "val_std": float(val.std()),
        "colored_share": float(colored.mean()),
        "colorfulness": colorfulness,
        # 가장 많은 색과 그 비중. "온통 빨간색" 같은 질의를 좁히는 값이다.
        "top_hue": max(share, key=share.get) if share else "",
        "top_hue_share": max(share.values()) if share else 0.0,
        **{f"hue_{k}": v for k, v in share.items()},
    }


def _meta(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    meta = raw.get("metadata", {})
    sem = raw.get("semantic_analysis", {})
    return {
        "title": meta.get("title", ""),
        "artist": meta.get("artist", ""),
        "album": meta.get("album", ""),
        "genre": meta.get("genre", ""),
        "vocal_gender": meta.get("vocal_gender", ""),
        "release_date": meta.get("release_date", ""),
        # 소리 쪽 단서만 쓴다. color_tags·visual_imagery는 LLM 상상이라 제외한다.
        "sound_tags": sem.get("sound_tags", []),
        "mood_tags": sem.get("mood_tags", []),
        "vibe_tags": sem.get("vibe_tags", []),
    }


def scan(raw: Path, out: Path) -> int:
    dirs = _song_dirs(raw)
    if not dirs:
        print(f"곡 폴더가 없다: {raw} (마운트를 확인하세요)", file=sys.stderr)
        return 2
    rows: List[Dict[str, Any]] = []
    missing = 0
    for i, (song_id, folder) in enumerate(dirs.items(), start=1):
        cover = folder / "cover.jpg"
        if not cover.exists():
            missing += 1
            continue
        try:
            row = {"song_id": song_id, "folder": folder.name, **_stats(cover)}
        except Exception as exc:  # 깨진 파일 하나가 전체를 멈추면 안 된다
            print(f"  [건너뜀] {folder.name}: {exc}", file=sys.stderr)
            missing += 1
            continue
        row.update(_meta(folder / "meta.json"))
        rows.append(row)
        if i % 200 == 0:
            print(f"  {i}/{len(dirs)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"표지 {len(rows)}장 · 표지 없음 {missing}곡 → {out}")
    return 0


def sheet(raw: Path, ids: List[str], out: Path, cols: int = 6, cell: int = 220) -> int:
    """번호 붙인 대조표. 번호로 곡을 지목할 수 있어야 눈으로 고를 수 있다."""
    dirs = _song_dirs(raw)
    picked = [(sid, dirs[sid]) for sid in ids if sid in dirs]
    if not picked:
        print("대조표에 넣을 곡이 없다", file=sys.stderr)
        return 2
    rows = math.ceil(len(picked) / cols)
    label_h = 22
    canvas = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (song_id, folder) in enumerate(picked):
        x, y = (i % cols) * cell, (i // cols) * (cell + label_h)
        try:
            with Image.open(folder / "cover.jpg") as im:
                canvas.paste(im.convert("RGB").resize((cell, cell), Image.BILINEAR),
                             (x, y))
        except Exception:
            draw.rectangle([x, y, x + cell, y + cell], fill="#eee")
        draw.text((x + 4, y + cell + 4), f"{i + 1}. {song_id}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"{len(picked)}장 → {out}")
    for i, (song_id, folder) in enumerate(picked, start=1):
        print(f"  {i}. {song_id}  {folder.name}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="곡 폴더 루트")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="표지 통계를 만든다")
    p_scan.add_argument("--out", type=Path, default=DEFAULT_STATS)

    p_sheet = sub.add_parser("sheet", help="고른 곡들을 한 장의 대조표로")
    p_sheet.add_argument("--ids", required=True, help="쉼표로 구분한 곡 id")
    p_sheet.add_argument("--out", type=Path, required=True)
    p_sheet.add_argument("--cols", type=int, default=6)

    args = ap.parse_args(argv)
    if args.cmd == "scan":
        return scan(args.raw, args.out)
    return sheet(args.raw, [s.strip() for s in args.ids.split(",") if s.strip()],
                 args.out, cols=args.cols)


if __name__ == "__main__":
    raise SystemExit(main())
