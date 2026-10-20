"""포스터용 비교 그림 2장을 SVG로 만든다(외부 패키지 없음).

입력 숫자는 evidence_table_2026-10-20.md §1의 저장 결과(v73 3차·v75)에서 옮겨 적은 값이며,
이 스크립트는 검색·모델 호출을 하지 않는다. 실행: python docs/poster/make_poster_figures.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"
N = 176

# 색: dataviz 기준 팔레트(검증 통과: 파랑·주황 인접 ΔE 통과). 분모는 중립 회색.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
FONT = "font-family='Pretendard, Apple SD Gothic Neo, Noto Sans KR, sans-serif'"


def text(x: float, y: float, s: str, size: int = 13, fill: str = INK, anchor: str = "start", weight: str = "normal") -> str:
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' "
            f"font-weight='{weight}' {FONT}>{s}</text>")


def fig_ceiling() -> str:
    """그림 1 — 후보 회수가 Hit@10의 천장이다(dev 176, 작품명 ON·소리 ON)."""
    rows = [
        ("평가 질의 전체", N, GRAY, "176"),
        ("후보 30개 안에 정답이 든 질의", 125, BLUE, "125 (71.0%)"),
        ("최종 10위 안에 정답이 든 질의 (Hit@10)", 121, BLUE, "121 (68.8%)"),
    ]
    w, h = 760, 230
    left, right, top, bar_h, gap = 300, 110, 48, 28, 22
    scale = (w - left - right) / N
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>",
           f"<rect width='{w}' height='{h}' fill='#ffffff'/>",
           text(16, 26, "후보 회수가 Hit@10의 천장이다 — dev 176, 작품명 ON·소리 ON", 15, INK, "start", "bold")]
    y = top
    for label, v, color, vt in rows:
        out.append(text(left - 12, y + bar_h * 0.68, label, 13, INK2, "end"))
        out.append(f"<rect x='{left}' y='{y}' width='{v * scale:.1f}' height='{bar_h}' rx='4' fill='{color}'/>")
        out.append(text(left + v * scale + 8, y + bar_h * 0.68, vt, 13, INK, "start", "bold"))
        y += bar_h + gap
    # 천장 표시선
    cx = left + 125 * scale
    out.append(f"<line x1='{cx:.1f}' y1='{top + bar_h + 6}' x2='{cx:.1f}' y2='{y - gap + 4}' stroke='{INK2}' stroke-dasharray='3 3'/>")
    out.append(text(16, h - 14, "후보에 든 정답 중 10위 안: 121/125 = 96.8%. 후보 밖 51건은 리랭커가 복구할 수 없다(상위 30개만 재정렬).", 11, INK2))
    out.append("</svg>")
    return "\n".join(out)


def fig_configs() -> str:
    """그림 2 — 구성별 Hit@10·Hit@1(dev 176, 각 2회 반복 중 첫 실행)."""
    configs = [
        ("작품명 OFF · 소리 ON\n(v73 A)", 117, 81),
        ("작품명 ON · 소리 ON\n(개발 비교 기준, v73 B = v75 A)", 121, 83),
        ("작품명 ON · 소리 생략\n(보류 구성, v75 B)", 123, 78),
    ]
    w, h = 760, 330
    left, top, plot_h = 70, 60, 180
    group_w = 200
    bar_w = 56
    ymax = 140
    scale = plot_h / ymax
    base = top + plot_h
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>",
           f"<rect width='{w}' height='{h}' fill='#ffffff'/>",
           text(16, 26, "구성별 정답 건수 — dev 176 · 실제 Google Cloud(Vertex AI) 리랭킹 · 평가기 60/120초", 15, INK, "start", "bold")]
    # 목표선
    for v, lab in ((124, "70% = 124"), (141, "80% = 141")):
        yy = base - v * scale
        out.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{w - 20}' y2='{yy:.1f}' stroke='{INK2}' stroke-dasharray='4 4' stroke-width='1'/>")
        out.append(text(w - 22, yy - 4, lab, 11, INK2, "end"))
    out.append(f"<line x1='{left}' y1='{base}' x2='{w - 20}' y2='{base}' stroke='{INK2}' stroke-width='1'/>")
    for i, (name, h10, h1) in enumerate(configs):
        gx = left + 30 + i * group_w
        for j, (v, color, lab) in enumerate(((h10, BLUE, "Hit@10"), (h1, ORANGE, "Hit@1"))):
            x = gx + j * (bar_w + 6)
            y = base - v * scale
            out.append(f"<rect x='{x}' y='{y:.1f}' width='{bar_w}' height='{v * scale:.1f}' rx='4' fill='{color}'/>")
            out.append(text(x + bar_w / 2, y + 18, str(v), 13, "#ffffff", "middle", "bold"))
        for k, line in enumerate(name.split("\n")):
            out.append(text(gx + bar_w + 3, base + 18 + k * 15, line, 11, INK2, "middle"))
    # 범례
    lx = left
    out.append(f"<rect x='{lx}' y='{h - 22}' width='12' height='12' rx='2' fill='{BLUE}'/>")
    out.append(text(lx + 18, h - 12, "Hit@10 (10위 안 정답)", 11, INK2))
    out.append(f"<rect x='{lx + 170}' y='{h - 22}' width='12' height='12' rx='2' fill='{ORANGE}'/>")
    out.append(text(lx + 188, h - 12, "Hit@1 (1위 정답)", 11, INK2))
    out.append(text(w - 20, h - 12, "작품명 ON·소리 생략은 Hit@10 +2이지만 Hit@1 −5 → 채택 보류", 11, INK2, "end"))
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig1_recall_ceiling.svg").write_text(fig_ceiling(), encoding="utf-8")
    (OUT / "fig2_config_hits.svg").write_text(fig_configs(), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
