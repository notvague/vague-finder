"""현재 검색 구조도(2026-10-20, main 26dec55 기준)를 SVG로 만든다. 외부 패키지 없음.

내용 근거: docs/poster/architecture_and_scope_2026-10-20.md §2 단계표(코드 위치 포함).
실행: python docs/poster/make_architecture_figure.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures" / "fig3_architecture.svg"

INK = "#0b0b0b"
INK2 = "#52514e"
LINE = "#8a8984"
BLUE = "#2a78d6"
BLUE_BG = "#e8f0fb"
ORANGE = "#eb6834"
ORANGE_BG = "#fdeee7"
GRAY_BG = "#f1f0ec"
GRAY_BD = "#c3c2b7"
FONT = "font-family='Pretendard, Apple SD Gothic Neo, Noto Sans KR, sans-serif'"

W, H = 1240, 840
parts: list[str] = []


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    parts.append(f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}' {FONT}>{s}</text>")


def box(x, y, w, h, title, lines=(), bg=GRAY_BG, bd=GRAY_BD, title_fill=INK, dashed=False):
    dash = " stroke-dasharray='5 4'" if dashed else ""
    parts.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{bg}' stroke='{bd}' stroke-width='1.2'{dash}/>")
    t(x + 10, y + 19, title, 12.5, title_fill, "start", "bold")
    for i, line in enumerate(lines):
        t(x + 10, y + 36 + i * 15, line, 11, INK2)


def arrow(x1, y1, x2, y2, dashed=False):
    dash = " stroke-dasharray='5 4'" if dashed else ""
    parts.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{LINE}' stroke-width='1.4' marker-end='url(#ah)'{dash}/>")


parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
parts.append("<defs><marker id='ah' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'>"
             f"<path d='M0,0 L8,4 L0,8 z' fill='{LINE}'/></marker></defs>")
parts.append(f"<rect width='{W}' height='{H}' fill='#ffffff'/>")
t(16, 26, "Vague-Finder 검색 구조 — 2026-10-20 main 기준 (서비스 기본 구성; 점선은 기본 OFF 실험 경로)", 15, INK, "start", "bold")

# 열 1: 입력
box(16, 60, 150, 92, "① 입력", ["자연어 기억 질의", "+ 재질문 답(성별·장르)", "+ 거절한 곡 · 출생 연도"])

# 열 2: 분석
box(196, 60, 200, 150, "② 질의 분석", ["Gemini 3.1 flash-lite",
                                    "Google Cloud(Vertex AI)",
                                    "→ 단서 구조(JSON 스키마)",
                                    "가사 구절/음차 · 한국어 태그 ·",
                                    "표지 영어 질의 · 소리 영어 질의 ·",
                                    "성별·장르·시기·제목 구조·배경 사실",
                                    "실패 시 규칙 폴백(confidence 0)",
                                    "시간 제한 20초"], bg=BLUE_BG, bd=BLUE)
arrow(166, 100, 196, 100)

# 열 3: 후보 검색 경로
px, pw = 426, 250
paths = [
    ("가사 표면 일치", ["MongoDB full_lyrics", "exact / fuzzy 구절, 보호 대상"], BLUE_BG, BLUE),
    ("텍스트 하이브리드", ["BM25(Kiwi 형태소, 태그 입력)", "+ KoE5 dense 1024, Qdrant"], BLUE_BG, BLUE),
    ("앨범 표지", ["SigLIP2 base-patch16-224 (768)", "시각 단서가 있을 때만"], BLUE_BG, BLUE),
    ("소리", ["CLAP htsat-fused (512), 음원 120초", "텍스트→오디오 임베딩 대조"], BLUE_BG, BLUE),
    ("배경 사실 Context", ["나무위키 사실 639곡 · 5,054개", "KoE5 dense + BM25 sparse, RRF"], BLUE_BG, BLUE),
    ("제목·속성 보조", ["제목 구조/의미/포함 · 수행 단서", "성별·장르·시기 메타 필터"], BLUE_BG, BLUE),
]
py = 60
for title, lines, bg, bd in paths:
    box(px, py, pw, 58, title, lines, bg, bd)
    arrow(396, 135, px, py + 29)
    py += 64
t(px, py + 10, "단서에 맞는 경로 병렬 조회 → 후보 융합", 10.5, INK2)

# 기본 OFF 실험 경로
box(px, py + 36, pw, 94, "기본 OFF 실험 경로", ["작품명 해석(간접 지칭 → 보조 후보 ≤10곡)",
                                             "속성 조건 보조(ATTR_FILTER_WEIGHT=0)",
                                             "정답 피드백 보충(FEEDBACK_*=0)",
                                             "미채택: 앨범 소개 낱말/의미 보조(v76·v78)"],
    bg="#ffffff", bd=ORANGE, title_fill=ORANGE, dashed=True)

# 열 4: 융합·보정
fx = 716
box(fx, 60, 220, 190, "③ 융합·보정", ["RRF 통합(k=60)",
                                     "가산: 가사 표면 · 제목 · 성별 ·",
                                     "  장르 · 시기 · 가수 유형 일치",
                                     "거절한 곡 제외 → 후보 30 절단",
                                     "재질문 답 가산(질의는 안 바꿈)",
                                     "가사 exact 1위 보호 지정",
                                     "소리 전용 질의는 깊은 융합",
                                     "(deep audio fusion)"], bg=BLUE_BG, bd=BLUE)
for i in range(6):
    arrow(px + pw, 89 + i * 64, fx, 155)
arrow(px + pw, py + 83, fx, 200, dashed=True)

# 열 5: 리랭킹
rx = 966
box(rx, 60, 258, 190, "④ 리랭킹", ["Gemini 3.1 flash-lite listwise",
                                   "최대 30곡 · 보호 시 묶음별 1패스",
                                   "정책 생략(모델 안 부름):",
                                   "  표지 지배 질의 · 자료 없는 MV 장면",
                                   "실패·시간 초과 → 검색 순서 유지",
                                   "  (rerank_outcome=failed)",
                                   "호출 20초 · 예산 30초(운영)",
                                   "선택: Cross-Encoder bge-m3-ko"], bg=BLUE_BG, bd=BLUE)
arrow(fx + 220, 155, rx, 155)

# 출력
box(rx, 290, 258, 118, "⑤ 응답", ["상위 15곡(화면, API 기본 10) + 가사 보호 배치",
                                  "선정 근거 패널(템플릿, 모델 호출 없음):",
                                  "  단서별 일치 · 순위 반영 기록 · 가사 인용",
                                  "노래 맵(CLAP·KoE5 2D 지도)에 표시",
                                  "YouTube 임베드 재생 · 정답 피드백 버튼"], bg=GRAY_BG, bd=GRAY_BD)
arrow(rx + 129, 250, rx + 129, 290)

# 재질문 루프
box(fx, 290, 220, 118, "재질문(2턴)", ["\"이 중에 없어요\" → 현재 후보로",
                                     "보컬 성별 · 장르 중 하나를 고름",
                                     "(출생 연도는 시기 창으로 1회)",
                                     "답은 분석에 병합, 거절 곡 제외",
                                     "후 같은 경로로 다시 검색"], bg=GRAY_BG, bd=GRAY_BD)
arrow(rx, 349, fx + 220, 349, dashed=True)
parts.append(f"<path d='M {fx+110} 408 L {fx+110} {py+22} L 296 {py+22} L 296 212' fill='none' stroke='{LINE}' stroke-width='1.4' stroke-dasharray='5 4' marker-end='url(#ah)'/>")
t(288, py - 2, "답·거절 곡을 ② 분석에 병합해", 10.5, INK2, "end")
t(288, py + 12, "같은 경로로 다시 검색", 10.5, INK2, "end")

# 자료층
dy = py + 148
box(16, dy, 1208, 118, "자료층 — 5,143곡(2026-10-11 확장 뒤), 저장소에는 원본 없음",
    ["곡 레코드 data/all_songs.jsonl: 멜론 메타(제목·가수·앨범·발매일·장르·보컬 성별·앨범 소개·플레이리스트 태그) · 가사 원문 · LLM 태그(분위기·장소·색·소리·심상) · 댓글 요약·인기도 · 나무위키 사실",
     "임베딩 artifacts/embeddings: 텍스트 KoE5 dense + BM25 sparse(5,143) · 표지 SigLIP2(5,143) · 음원 CLAP 평균(5,143, 센터링) · Context KoE5 dense(사실 5,054) + BM25 sparse(639곡)",
     "저장소: Qdrant 컬렉션 5개(text 1024 Dot · image 768 Cos · audio 512 Cos · context dense/sparse, 별칭 세대 f6424b94) · MongoDB(가사 원문) · 평가 질의 CSV(v06·v09·v10)",
     "분석·리랭킹: Google Cloud(Vertex AI) Gemini(리랭킹은 후보 묶음별, 정책 생략 시 호출 없음). 수집 정제·댓글 선별·Context 추출은 gemini-3.5-flash-lite(오프라인)"],
    bg="#fafaf8", bd=GRAY_BD)

# 범례
ly = dy + 140
parts.append(f"<rect x='16' y='{ly}' width='14' height='14' rx='3' fill='{BLUE_BG}' stroke='{BLUE}'/>")
t(36, ly + 11, "요청 경로(서비스 기본 구성: 작품명 OFF · 소리 ON · Context 0.5 · 속성/피드백 OFF)", 11, INK2)
parts.append(f"<rect x='16' y='{ly+22}' width='14' height='14' rx='3' fill='#ffffff' stroke='{ORANGE}' stroke-dasharray='4 3'/>")
t(36, ly + 33, "기본 OFF 실험 경로 — 평가에서만 켜서 측정(작품명 ON = 개발 비교 기준 121/176)", 11, INK2)
parts.append(f"<rect x='16' y='{ly+44}' width='14' height='14' rx='3' fill='{GRAY_BG}' stroke='{GRAY_BD}'/>")
t(36, ly + 55, "모델을 부르지 않는 단계(입력 · 응답 구성 · 재질문 계산)", 11, INK2)
t(16, ly + 82, "코드 위치: src/retrieval/search_router.py(②~④ 조립) · query_analyzer.py · lyrics_exact_search.py · context_*.py · rerank_policy.py · gemini_listwise_reranker.py · clarify.py · user_explanation.py", 10.5, INK2)
t(16, ly + 98, "출처: docs/poster/architecture_and_scope_2026-10-20.md §2 (각 단계의 코드 줄·산출물 설정 확인 기록)", 10.5, INK2)

parts.append("</svg>")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print("wrote", OUT)
