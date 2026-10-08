"""Vague Finder 로고 컨셉 A '흩어지는 V' — 심볼과 락업 SVG를 만든다.

꼭짓점은 또렷한 획이고, 위로 갈수록 점으로 흩어지는 V.
락업에서는 이 V가 'Vague'의 V 자리에 들어가고 아래에 자간을 넓힌 FINDER가 붙는다.

필요한 것 (프로젝트 requirements에는 없다):
  pip install fonttools uharfbuzz brotli
  Jost 가변 서체(OFL): https://github.com/google/fonts/raw/main/ofl/jost/Jost%5Bwght%5D.ttf
  → fontTools.varLib.instancer로 wght 400·800 인스턴스를 Jost-400.ttf, Jost-800.ttf로 만든다

실행:
  python build.py --fonts <Jost-400/800.ttf가 있는 폴더> --out docs/brand/concept-a
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

Dot = tuple[float, float, float]
BBox = tuple[float, float, float, float]

# ---------- 심볼 형태 ----------
ALPHA = 25            # 팔 기울기(수직 기준, 도)
STROKE = 40           # 또렷한 획 두께
SOLID = 100           # 꼭짓점에서 획이 이어지는 길이
RADII = (17, 13.5, 10.5, 7.5)   # 흩어지는 점의 반지름 — 위로 갈수록 작아진다
GAPS = (8, 10, 12, 14)          # 점 사이 간격 — 위로 갈수록 벌어진다

# ---------- 락업 글자 ----------
MAIN, SUB = 132.0, 34.0         # 'ague'와 'FINDER'의 글자 크기 (캔버스 단위)


def f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def capsule(p0: tuple[float, float], p1: tuple[float, float], r: float) -> str:
    """양 끝이 둥근 막대 하나를 닫힌 패스로."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length * r, dx / length * r
    a = (p0[0] + nx, p0[1] + ny)
    b = (p1[0] + nx, p1[1] + ny)
    c = (p1[0] - nx, p1[1] - ny)
    d = (p0[0] - nx, p0[1] - ny)
    return (f"M{f(a[0])} {f(a[1])} L{f(b[0])} {f(b[1])} A{f(r)} {f(r)} 0 0 0 {f(c[0])} {f(c[1])} "
            f"L{f(d[0])} {f(d[1])} A{f(r)} {f(r)} 0 0 0 {f(a[0])} {f(a[1])} Z")


def v_shapes() -> tuple[list[str], list[Dot], BBox]:
    """꼭짓점 중심 (0,0) 기준 로컬 좌표. 두 팔 = 둥근 획 + 점 네 개."""
    r = STROKE / 2
    paths: list[str] = []
    dots: list[Dot] = []
    for side in (-1, 1):
        ux = side * math.sin(math.radians(ALPHA))
        uy = -math.cos(math.radians(ALPHA))
        paths.append(capsule((0, 0), (ux * SOLID, uy * SOLID), r))
        t = SOLID + r
        for rr, gap in zip(RADII, GAPS):
            t += gap + rr
            dots.append((ux * t, uy * t, rr))
            t += rr
    xs0 = [x - rr for x, _, rr in dots] + [-r]
    xs1 = [x + rr for x, _, rr in dots] + [r]
    ys0 = [y - rr for _, y, rr in dots]
    ys1 = [y + rr for _, y, rr in dots] + [r]
    return paths, dots, (min(xs0), min(ys0), max(xs1), max(ys1))


def placed(paths: list[str], dots: list[Dot], bbox: BBox, box: float, cx: float, cy: float) -> list[str]:
    """bbox의 긴 변을 box에 맞추고 중심을 (cx, cy)에 둔다."""
    bx0, by0, bx1, by1 = bbox
    s = box / max(bx1 - bx0, by1 - by0)
    tx, ty = cx - s * (bx0 + bx1) / 2, cy - s * (by0 + by1) / 2
    els = [f'<path transform="matrix({f(s)} 0 0 {f(s)} {f(tx)} {f(ty)})" d="{d}"/>' for d in paths]
    els += [f'<circle cx="{f(tx + s * x)}" cy="{f(ty + s * y)}" r="{f(s * r)}"/>' for x, y, r in dots]
    return els


def svg(w: float, h: float, els: list[str], title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {f(w)} {f(h)}" width="{f(w)}" height="{f(h)}" '
            f'role="img" aria-labelledby="t"><title id="t">{title}</title>\n'
            f'<g fill="#000">\n' + "\n".join(els) + "\n</g>\n</svg>\n")


# ---------- 글자 → 패스 (라이브 텍스트 없음) ----------
class Face:
    def __init__(self, path: Path) -> None:
        self.tt = TTFont(path)
        self.hb = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
        self.upm = self.tt["head"].unitsPerEm
        self.glyphs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()

    def run(self, text: str, size: float, x: float, base: float, track_em: float = 0.0) -> tuple[str, float, float]:
        """한 줄을 패스로. 커닝 적용. (패스, 잉크 왼쪽, 잉크 오른쪽)을 돌려준다."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": False})
        k = size / self.upm
        pen_x, parts = x, []
        left = right = None
        items = list(zip(buf.glyph_infos, buf.glyph_positions))
        for i, (info, pos) in enumerate(items):
            name = self.order[info.codepoint]
            gx = pen_x + pos.x_offset * k
            bp = BoundsPen(self.glyphs)
            self.glyphs[name].draw(bp)
            if bp.bounds:
                lo, hi = gx + bp.bounds[0] * k, gx + bp.bounds[2] * k
                left = lo if left is None else min(left, lo)
                right = hi if right is None else max(right, hi)
            pen = SVGPathPen(self.glyphs, ntos=f)
            self.glyphs[name].draw(TransformPen(pen, (k, 0, 0, -k, gx, base)))
            parts.append(pen.getCommands())
            pen_x += pos.x_advance * k
            if i < len(items) - 1:
                pen_x += track_em * size
        return " ".join(parts), left, right

    def fit_tracking(self, text: str, size: float, target: float) -> float:
        _, lo, hi = self.run(text, size, 0, 0)
        return (target - (hi - lo)) / (len(text) - 1) / size


def build(fonts: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    paths, dots, bbox = v_shapes()

    # 심볼: 256 캔버스, 긴 변 208, 광학 중심을 살짝 위(126)로
    (out / "a-symbol.svg").write_text(svg(256, 256, placed(paths, dots, bbox, 208, 128, 126),
                                          "Vague Finder symbol"))

    # 락업: 흩어지는 V가 'Vague'의 V를 대신한다
    heavy, regular = Face(fonts / "Jost-800.ttf"), Face(fonts / "Jost-400.ttf")
    cap, sub_cap, gap = 0.70 * MAIN, 0.70 * SUB, 0.36 * MAIN
    b1 = (256 - (cap + gap + sub_cap)) / 2 + cap          # 'ague' 기준선
    b2 = b1 + gap + sub_cap                               # 'FINDER' 기준선
    over = 0.015 * MAIN                                   # 둥근 꼭짓점의 기준선 아래 오버슈트
    hgt = cap + over
    bx0, by0, bx1, by1 = bbox
    w = (bx1 - bx0) * hgt / (by1 - by0)
    x0 = 16
    els = placed(paths, dots, bbox, max(w, hgt), x0 + w / 2, b1 + over - hgt / 2)
    _, lo, _ = heavy.run("ague", MAIN, 0, b1, -0.03)
    d1, _, right1 = heavy.run("ague", MAIN, x0 + w - lo - 0.02 * MAIN, b1, -0.03)
    track = regular.fit_tracking("FINDER", SUB, right1 - x0)
    _, lo2, _ = regular.run("FINDER", SUB, 0, b2, track)
    d2, _, right2 = regular.run("FINDER", SUB, x0 - lo2, b2, track)
    els += [f'<path d="{d1}"/>', f'<path d="{d2}"/>']
    (out / "a-lockup.svg").write_text(svg(max(right1, right2) + x0, 256, els, "Vague Finder"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    build(a.fonts, a.out)
    print("wrote", a.out / "a-symbol.svg", a.out / "a-lockup.svg")
