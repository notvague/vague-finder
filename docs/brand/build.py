"""Vague Finder 로고 키트 — 마스터 SVG를 만든다.

로고: '흩어지는 V'. 꼭짓점은 또렷한 획이고, 위로 갈수록 점으로 흩어진다.
락업에서는 이 V가 'Vague'의 V 자리에 들어간다(심볼과 단어의 V가 겹치지 않는다).
락업의 V는 글자 굵기에 맞춘 별도 비율(LOCKUP)이고, 작은 크기용 심볼도 따로 그린다(SMALL).

만드는 것 (logo/):
  vaguefinder-symbol[-ink|-black|-white|-dark].svg          심볼 (기본 256 캔버스)
  vaguefinder-symbol-small[-ink|-black|-white].svg          32px 이하용 단순화 버전 (점 둘)
  vaguefinder-stacked[-ink|-black|-white|-dark].svg         두 줄 락업 (기본)
  vaguefinder-horizontal[-ink|-black|-white|-dark].svg      한 줄 락업
  기본 = V는 Ember, 글자는 Dusk Ink (밝은 바탕)
  -dark = V는 Ember, 글자는 Mist, 조금 가늘게 (어두운 바탕)
  -white / -dark 는 어두운 바탕에서 굵어 보이는 현상(irradiation) 때문에 획과 글자를 4% 가늘게 그린다

필요한 것 (프로젝트 requirements에는 없다):
  pip install fonttools uharfbuzz brotli
  Jost 가변 서체(OFL): https://github.com/google/fonts/raw/main/ofl/jost/Jost%5Bwght%5D.ttf
  → fontTools.varLib.instancer로 wght 380·400·770·800 인스턴스를 Jost-<wght>.ttf로 만든다

실행:
  python docs/brand/build.py --fonts <Jost-*.ttf 폴더>
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

Dot = tuple[float, float, float]
BBox = tuple[float, float, float, float]

INK = "#261D3B"      # Dusk Ink — 글자, 한 가지 색 버전
EMBER = "#E66A3A"    # Ember — 심볼
MIST = "#F7F2EC"     # Mist — 어두운 바탕 위 글자, 밝은 보조 바탕
THIN = 0.96          # 어두운 바탕용 가늘게 그리기 비율


@dataclass(frozen=True)
class VSpec:
    alpha: float = 25                                   # 팔 기울기(수직 기준, 도)
    stroke: float = 40                                  # 또렷한 획 두께
    solid: float = 100                                  # 꼭짓점에서 획이 이어지는 길이
    radii: tuple[float, ...] = (17, 13.6, 10.9, 8.7)    # 흩어지는 점 — 0.8배씩 작아진다
    gaps: tuple[float, ...] = (8, 10, 12, 14)           # 점 사이 간격 — 2씩 벌어진다
    thin: float = 1.0                                   # 획·점만 줄이고 중심은 그대로


FULL = VSpec()
# 16~32px: 점 둘, 굵은 획. 획 길이가 두께의 2.3배보다 짧으면 V 안쪽 홈이 얕아져 하트처럼 보인다
SMALL = VSpec(stroke=52, solid=124, radii=(22, 16), gaps=(18, 20))
# 락업 안의 V: Jost 800 기둥 굵기에 맞춰 획을 굵게, 점은 셋
LOCKUP = VSpec(stroke=64, solid=150, radii=(23, 18.4, 14.7), gaps=(11, 13, 15))


def f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---------------- 심볼 형태 ----------------
def capsule(p0: tuple[float, float], p1: tuple[float, float], r: float) -> str:
    """양 끝이 둥근 막대 하나를 닫힌 패스로."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length * r, dx / length * r
    a, b = (p0[0] + nx, p0[1] + ny), (p1[0] + nx, p1[1] + ny)
    c, d = (p1[0] - nx, p1[1] - ny), (p0[0] - nx, p0[1] - ny)
    return (f"M{f(a[0])} {f(a[1])} L{f(b[0])} {f(b[1])} A{f(r)} {f(r)} 0 0 0 {f(c[0])} {f(c[1])} "
            f"L{f(d[0])} {f(d[1])} A{f(r)} {f(r)} 0 0 0 {f(a[0])} {f(a[1])} Z")


def v_shapes(spec: VSpec) -> tuple[list[str], list[Dot], BBox]:
    """꼭짓점 중심 (0,0) 기준. bbox는 가늘게 하기 전 형태로 잡아 버전끼리 위치가 같다."""
    r = spec.stroke / 2
    paths: list[str] = []
    dots: list[Dot] = []
    for side in (-1, 1):
        ux = side * math.sin(math.radians(spec.alpha))
        uy = -math.cos(math.radians(spec.alpha))
        paths.append(capsule((0, 0), (ux * spec.solid, uy * spec.solid), r * spec.thin))
        t = spec.solid + r
        for rr, gap in zip(spec.radii, spec.gaps):
            t += gap + rr
            dots.append((ux * t, uy * t, rr))
            t += rr
    bbox = (min([x - rr for x, _, rr in dots] + [-r]), min(y - rr for _, y, rr in dots),
            max([x + rr for x, _, rr in dots] + [r]), max([y + rr for _, y, rr in dots] + [r]))
    return paths, [(x, y, rr * spec.thin) for x, y, rr in dots], bbox


def placed(spec: VSpec, box: float, cx: float, cy: float) -> list[str]:
    """bbox 긴 변을 box에 맞추고 중심을 (cx, cy)에 둔다. 획 둘을 한 패스로 합친다."""
    paths, dots, (bx0, by0, bx1, by1) = v_shapes(spec)
    s = box / max(bx1 - bx0, by1 - by0)
    tx, ty = cx - s * (bx0 + bx1) / 2, cy - s * (by0 + by1) / 2
    els = [f'<path transform="matrix({f(s)} 0 0 {f(s)} {f(tx)} {f(ty)})" d="{" ".join(paths)}"/>']
    els += [f'<circle cx="{f(tx + s * x)}" cy="{f(ty + s * y)}" r="{f(s * r)}"/>' for x, y, r in dots]
    return els


def v_width(spec: VSpec, height: float) -> float:
    _, _, (bx0, by0, bx1, by1) = v_shapes(spec)
    return (bx1 - bx0) * height / (by1 - by0)


# ---------------- 글자 → 패스 (라이브 텍스트 없음) ----------------
class Face:
    def __init__(self, path: Path) -> None:
        self.tt = TTFont(path)
        self.hb = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
        self.upm = self.tt["head"].unitsPerEm
        self.glyphs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()

    def run(self, text: str, size: float, x: float, base: float, track_em: float = 0.0) -> tuple[str, float, float]:
        """한 줄을 패스로. 커닝 적용. (패스, 잉크 왼쪽, 잉크 오른쪽)."""
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

    def at(self, text: str, size: float, left: float, base: float, track_em: float = 0.0) -> tuple[str, float]:
        """잉크 왼쪽 끝을 left에 맞춰 놓는다. (패스, 잉크 오른쪽)."""
        _, lo, _ = self.run(text, size, 0, base, track_em)
        d, _, hi = self.run(text, size, left - lo, base, track_em)
        return d, hi

    def fit_tracking(self, text: str, size: float, target: float) -> float:
        _, lo, hi = self.run(text, size, 0, 0)
        return (target - (hi - lo)) / (len(text) - 1) / size


# ---------------- 출력 ----------------
@dataclass(frozen=True)
class Colorway:
    suffix: str
    v: str
    text: str
    thin: bool = False


COLORWAYS = [
    Colorway("", EMBER, INK),
    Colorway("-dark", EMBER, MIST, True),
    Colorway("-ink", INK, INK),
    Colorway("-black", "#000000", "#000000"),
    Colorway("-white", "#FFFFFF", "#FFFFFF", True),
]


def svg(w: float, h: float, groups: list[tuple[str, str, list[str]]], title: str) -> str:
    body = "".join(f'<g id="{gid}" fill="{fill}">\n' + "\n".join(els) + "\n</g>\n" for gid, fill, els in groups)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {f(w)} {f(h)}" width="{f(w)}" height="{f(h)}" '
            f'role="img" aria-labelledby="t"><title id="t">{title}</title>\n{body}</svg>\n')


class Kit:
    PAD = 24            # 락업 상하좌우 여백 (높이 256 기준)

    def __init__(self, fonts: Path, out: Path) -> None:
        self.out = out
        self.faces = {w: Face(fonts / f"Jost-{w}.ttf") for w in (380, 400, 770, 800)}
        out.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, w: float, h: float, groups: list[tuple[str, str, list[str]]], title: str) -> None:
        (self.out / f"{name}.svg").write_text(svg(w, h, groups, title), encoding="utf-8")

    # 심볼 — 256 캔버스, 긴 변 208, 광학 중심을 2 위로
    def symbols(self) -> None:
        for cw in COLORWAYS:
            spec = replace(FULL, thin=THIN if cw.thin else 1.0)
            self.write(f"vaguefinder-symbol{cw.suffix}", 256, 256, [("symbol", cw.v, placed(spec, 208, 128, 126))],
                       "Vague Finder")
            if cw.suffix != "-dark":
                small = replace(SMALL, thin=THIN if cw.thin else 1.0)
                self.write(f"vaguefinder-symbol-small{cw.suffix}", 256, 256,
                           [("symbol", cw.v, placed(small, 224, 128, 127))], "Vague Finder")

    def _v_in_word(self, spec: VSpec, main: float, x0: float, base: float) -> tuple[list[str], float]:
        """'Vague'의 V 자리에 흩어지는 V를 놓는다. 꼭짓점은 기준선 아래로 살짝 내려간다(오버슈트)."""
        cap, over = 0.70 * main, 0.015 * main
        hgt = cap + over
        w = v_width(spec, hgt)
        return placed(spec, max(w, hgt), x0 + w / 2, base + over - hgt / 2), x0 + w - 0.02 * main

    # 두 줄 락업 — 'Vague' / 자간을 넓혀 폭을 맞춘 'FINDER' (지금 사이트 헤더 구조)
    def stacked(self) -> None:
        main = (256 - 2 * self.PAD) / (0.70 + 0.36 + 0.70 * 34 / 132)
        sub = main * 34 / 132
        b1 = self.PAD + 0.70 * main
        b2 = b1 + 0.36 * main + 0.70 * sub
        for cw in COLORWAYS:
            heavy, regular = self.faces[770 if cw.thin else 800], self.faces[380 if cw.thin else 400]
            spec = replace(LOCKUP, thin=THIN if cw.thin else 1.0)
            v_els, ague_x = self._v_in_word(spec, main, self.PAD, b1)
            d1, right1 = heavy.at("ague", main, ague_x, b1, -0.03)
            track = regular.fit_tracking("FINDER", sub, right1 - self.PAD)
            d2, right2 = regular.at("FINDER", sub, self.PAD, b2, track)
            self.write(f"vaguefinder-stacked{cw.suffix}", max(right1, right2) + self.PAD, 256,
                       [("symbol", cw.v, v_els), ("wordmark", cw.text, [f'<path d="{d1}"/>', f'<path d="{d2}"/>'])],
                       "Vague Finder")

    # 한 줄 락업 — 'Vague'(800) + 'Finder'(400), README·좁은 높이용
    def horizontal(self) -> None:
        main = (256 - 2 * self.PAD) / (0.70 + 0.242 + 0.015)
        base = self.PAD + 0.70 * main
        for cw in COLORWAYS:
            heavy, regular = self.faces[770 if cw.thin else 800], self.faces[380 if cw.thin else 400]
            spec = replace(LOCKUP, thin=THIN if cw.thin else 1.0)
            v_els, ague_x = self._v_in_word(spec, main, self.PAD, base)
            d1, right1 = heavy.at("ague", main, ague_x, base, -0.03)
            d2, right2 = regular.at("Finder", main, right1 + 0.24 * main, base, -0.01)
            self.write(f"vaguefinder-horizontal{cw.suffix}", right2 + self.PAD, 256,
                       [("symbol", cw.v, v_els), ("wordmark", cw.text, [f'<path d="{d1}"/>', f'<path d="{d2}"/>'])],
                       "Vague Finder")


# ---------------- 가이드 도해 (guide/) — 설명용 그림이라 라벨은 라이브 텍스트 ----------------
LABEL = 'font-family="Pretendard, Apple SD Gothic Neo, Noto Sans KR, sans-serif"'


def nest(src: Path, x: float, y: float, h: float) -> tuple[str, float]:
    """마스터 SVG를 높이 h로 끼워 넣는다. (요소, 폭)."""
    raw = src.read_text(encoding="utf-8")
    vb = raw.split('viewBox="', 1)[1].split('"', 1)[0]
    vw, vh = (float(v) for v in vb.split()[2:])
    inner = raw.split(">", 1)[1].rsplit("</svg>", 1)[0]
    w = h * vw / vh
    return f'<svg x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" viewBox="{vb}">{inner}</svg>', w


def clear_zone(x0: float, y0: float, x1: float, y1: float, c: float) -> str:
    """잉크 상자 둘레에 여백 띠를 칠하고 x 표시를 단다."""
    return (f'<path fill="{EMBER}" fill-opacity="0.12" fill-rule="evenodd" d="M{f(x0 - c)} {f(y0 - c)} H{f(x1 + c)} '
            f'V{f(y1 + c)} H{f(x0 - c)} Z M{f(x0)} {f(y0)} H{f(x1)} V{f(y1)} H{f(x0)} Z"/>'
            f'<rect x="{f(x0 - c)}" y="{f(y0 - c)}" width="{f(x1 - x0 + 2 * c)}" height="{f(y1 - y0 + 2 * c)}" '
            f'fill="none" stroke="{EMBER}" stroke-dasharray="6 5" stroke-width="1.5"/>'
            f'<rect x="{f(x0)}" y="{f(y0)}" width="{f(x1 - x0)}" height="{f(y1 - y0)}" fill="none" stroke="#B9B2C6" '
            f'stroke-width="1"/>'
            f'<text x="{f((x0 + x1) / 2)}" y="{f(y0 - c / 2 + 6)}" {LABEL} font-size="18" fill="{EMBER}" '
            f'text-anchor="middle">x</text>'
            f'<text x="{f(x0 - c / 2)}" y="{f((y0 + y1) / 2 + 6)}" {LABEL} font-size="18" fill="{EMBER}" '
            f'text-anchor="middle">x</text>')


def guide(logo: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pad = Kit.PAD
    # 두 줄 락업: x = V 높이(= 대문자 높이)의 1/2
    main = (256 - 2 * pad) / (0.70 + 0.36 + 0.70 * 34 / 132)
    k = 1.1                                     # 도해 배율
    c_st = 0.5 * 0.70 * main * k
    lx, ly = 60 + c_st, 60 + c_st
    st, st_w = nest(logo / "vaguefinder-stacked.svg", lx - pad * k, ly - pad * k, 256 * k)
    ink = (lx, ly, lx + st_w - 2 * pad * k, ly + (256 - 2 * pad) * k)
    # 심볼: x = 심볼 높이의 1/4
    _, _, (bx0, by0, bx1, by1) = v_shapes(FULL)
    sym_h = 280.0
    sym_w = sym_h * (bx1 - bx0) / (by1 - by0)
    c_sy = sym_h / 4
    sx = ink[2] + c_st + 120 + c_sy
    sy = ly + (ink[3] - ink[1] - sym_h) / 2
    scale = sym_h / 208                          # 마스터에서 긴 변 208 = 높이
    sym, _ = nest(logo / "vaguefinder-symbol.svg", sx - (256 * scale - sym_w) / 2, sy - (126 - 104) * scale, 256 * scale)
    W = sx + sym_w + c_sy + 60
    H = ink[3] + c_st + 120
    notes = (f'<text x="60" y="{f(H - 50)}" {LABEL} font-size="20" fill="{INK}">락업: x = V 높이(대문자 높이)의 1/2'
             f'</text><text x="{f(sx - c_sy)}" y="{f(H - 50)}" {LABEL} font-size="20" fill="{INK}">심볼: x = 심볼 높이의 1/4'
             f'</text>')
    body = (f'<rect width="100%" height="100%" fill="#FFFFFF"/>' + clear_zone(*ink, c_st) + st
            + clear_zone(sx, sy, sx + sym_w, sy + sym_h, c_sy) + sym + notes)
    (out / "clearspace.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {f(W)} {f(H)}" width="{f(W)}" height="{f(H)}">'
        f'{body}</svg>\n', encoding="utf-8")

    # 팔레트
    def lum(hexc: str) -> float:
        ch = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        ch = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]

    def ratio(a: str, b: str) -> float:
        la, lb = sorted((lum(a), lum(b)), reverse=True)
        return (la + 0.05) / (lb + 0.05)

    def cmyk(hexc: str) -> str:
        r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
        kk = 1 - max(r, g, b)
        if kk >= 1:
            return "0 0 0 100"
        return " ".join(str(round(v * 100)) for v in ((1 - r - kk) / (1 - kk), (1 - g - kk) / (1 - kk),
                                                     (1 - b - kk) / (1 - kk), kk))

    swatches = [("Dusk Ink", INK, "글자 · 한 가지 색 로고",
                 f"흰 바탕 대비 {ratio(INK, '#FFFFFF'):.1f} : 1"),
                ("Ember", EMBER, "심볼(V) · 강조 — 본문 글자색으로는 쓰지 않는다",
                 f"흰 바탕 {ratio(EMBER, '#FFFFFF'):.1f} : 1 · Ink 바탕 {ratio(EMBER, INK):.1f} : 1"),
                ("Mist", MIST, "어두운 바탕 위 글자 · 밝은 보조 바탕",
                 f"Ink 바탕 대비 {ratio(MIST, INK):.1f} : 1")]
    cards = []
    for i, (name, hx, use, contrast) in enumerate(swatches):
        x = 40 + i * 400
        rgb = " ".join(str(int(hx[j:j + 2], 16)) for j in (1, 3, 5))
        border = ' stroke="#E2DDD6"' if hx == MIST else ""
        cards.append(
            f'<rect x="{x}" y="40" width="360" height="180" rx="14" fill="{hx}"{border}/>'
            f'<text x="{x}" y="262" {LABEL} font-size="24" font-weight="700" fill="{INK}">{name}</text>'
            f'<text x="{x}" y="294" {LABEL} font-size="17" fill="#4A4458">HEX {hx} · RGB {rgb}</text>'
            f'<text x="{x}" y="320" {LABEL} font-size="17" fill="#4A4458">CMYK {cmyk(hx)} (단순 변환값)</text>'
            f'<text x="{x}" y="346" {LABEL} font-size="17" fill="#4A4458">{contrast}</text>'
            f'<text x="{x}" y="372" {LABEL} font-size="15" fill="#7A7488">{use}</text>')
    (out / "palette.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 410" width="1240" height="410">'
        '<rect width="100%" height="100%" fill="#FFFFFF"/>' + "".join(cards) + "</svg>\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "logo")
    a = ap.parse_args()
    kit = Kit(a.fonts, a.out)
    kit.symbols()
    kit.stacked()
    kit.horizontal()
    guide(a.out, a.out.parent / "guide")
    print("wrote", len(list(a.out.glob("*.svg"))), "SVG files to", a.out, "+ guide/")
