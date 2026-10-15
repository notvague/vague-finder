===== v06
B(always): 리랭킹 상태 {'skipped': 15} · attr_filter_weight=0.5 · use_genre=True · gemini={'backend': 'unused', 'setting': 'vertex'}
D(conditional): 리랭킹 상태 {'skipped': 15} · attr_filter_weight=0.5 · use_genre=True · gemini={'backend': 'unused', 'setting': 'vertex'}

질의 7건: m103, q101, q104, q112, q115, q206, q308

달라진 턴 0개 (질의 · 흐름 · 턴: 순위 B(always)→D(conditional) · 후보 순위 → · 답 → · 후보 집합 같음?)

흐름별 최종(대화 끝) 10위 안 — B(always) → D(conditional)
| 흐름 | 질의 수 | 10위 안 B | 10위 안 C | 이득 | 손실 | 후보 30 안 B | C |
|---|---|---|---|---|---|---|---|
| initial | 7 | 6 | 6 | 0  | 0  | 6 | 6 |
| flow:reject | 7 | 6 | 6 | 0  | 0  | 6 | 6 |
| flow:skip | 7 | 6 | 6 | 0  | 0  | 6 | 6 |
| flow:oracle | 7 | 6 | 6 | 0  | 0  | 6 | 6 |
| flow:noisy | 7 | 6 | 6 | 0  | 0  | 6 | 6 |

질의별 최종 순위(흐름: B→C):
  m103: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  q101: initial 9→9 · reject 9→9 · skip 9→9 · oracle 9→9 · noisy 9→9
  q104: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  q112: initial 2→2 · reject 2→2 · skip 2→2 · oracle 2→2 · noisy 2→2
  q115: initial 밖→밖 · reject 밖→밖 · skip 밖→밖 · oracle 밖→밖 · noisy 밖→밖
  q206: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  q308: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
===== v09
B(always): 리랭킹 상태 {'skipped': 53} · attr_filter_weight=0.5 · use_genre=True · gemini={'backend': 'unused', 'setting': 'vertex'}
D(conditional): 리랭킹 상태 {'skipped': 49} · attr_filter_weight=0.5 · use_genre=True · gemini={'backend': 'unused', 'setting': 'vertex'}

질의 11건: n002, n011, n014, n016, n030, n033, n040, n042, n081, n096, n097

달라진 턴 32개 (질의 · 흐름 · 턴: 순위 B(always)→D(conditional) · 후보 순위 → · 답 → · 후보 집합 같음?)
  n016 flow:reject 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n016 flow:reject 턴3: 순위 밖→밖 · 후보 밖→28 · 답 :none= · 후보 집합 다름
  n016 flow:skip 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n016 flow:skip 턴3: 순위 밖→밖 · 후보 밖→28 · 답 :none= · 후보 집합 다름
  n016 flow:oracle 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n016 flow:oracle 턴3: 순위 밖→밖 · 후보 밖→28 · 답 :none= · 후보 집합 다름
  n016 flow:noisy 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n016 flow:noisy 턴3: 순위 밖→밖 · 후보 밖→28 · 답 :none= · 후보 집합 다름
  n033 flow:reject 턴1: 순위 밖→밖 · 후보 밖→20 · 답 := · 후보 집합 다름
  n033 flow:reject 턴2: 순위 밖→10 · 후보 밖→10 · 답 :none= · 후보 집합 다름
  n033 flow:reject 턴3: 순위 밖→밖 · 후보 밖→밖 · 답 :none= → := · 후보 집합 다름
  n033 flow:skip 턴1: 순위 밖→밖 · 후보 밖→20 · 답 := · 후보 집합 다름
  n033 flow:skip 턴2: 순위 밖→10 · 후보 밖→10 · 답 :none= · 후보 집합 다름
  n033 flow:skip 턴3: 순위 밖→밖 · 후보 밖→밖 · 답 :none= → := · 후보 집합 다름
  n033 flow:oracle 턴1: 순위 밖→밖 · 후보 밖→20 · 답 := · 후보 집합 다름
  n033 flow:oracle 턴2: 순위 밖→10 · 후보 밖→10 · 답 :none= · 후보 집합 다름
  n033 flow:oracle 턴3: 순위 밖→밖 · 후보 밖→밖 · 답 :none= → := · 후보 집합 다름
  n033 flow:noisy 턴1: 순위 밖→밖 · 후보 밖→20 · 답 := · 후보 집합 다름
  n033 flow:noisy 턴2: 순위 밖→10 · 후보 밖→10 · 답 :none= · 후보 집합 다름
  n033 flow:noisy 턴3: 순위 밖→밖 · 후보 밖→밖 · 답 :none= → := · 후보 집합 다름
  n096 flow:reject 턴1: 순위 밖→밖 · 후보 밖→밖 · 답 := · 후보 집합 다름
  n096 flow:reject 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n096 flow:reject 턴3: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n096 flow:skip 턴1: 순위 밖→밖 · 후보 밖→밖 · 답 := · 후보 집합 다름
  n096 flow:skip 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n096 flow:skip 턴3: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n096 flow:oracle 턴1: 순위 밖→밖 · 후보 밖→밖 · 답 := · 후보 집합 다름
  n096 flow:oracle 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n096 flow:oracle 턴3: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n096 flow:noisy 턴1: 순위 밖→밖 · 후보 밖→밖 · 답 := · 후보 집합 다름
  n096 flow:noisy 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n096 flow:noisy 턴3: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름

흐름별 최종(대화 끝) 10위 안 — B(always) → D(conditional)
| 흐름 | 질의 수 | 10위 안 B | 10위 안 C | 이득 | 손실 | 후보 30 안 B | C |
|---|---|---|---|---|---|---|---|
| initial | 11 | 5 | 5 | 0  | 0  | 7 | 8 |
| flow:reject | 11 | 7 | 8 | 1 ['n033'] | 0  | 7 | 9 |
| flow:skip | 11 | 7 | 8 | 1 ['n033'] | 0  | 7 | 9 |
| flow:oracle | 11 | 8 | 9 | 1 ['n033'] | 0  | 8 | 10 |
| flow:noisy | 11 | 6 | 7 | 1 ['n033'] | 0  | 6 | 8 |

질의별 최종 순위(흐름: B→C):
  n002: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  n011: initial 밖→밖 · reject 밖→밖 · skip 밖→밖 · oracle 5→5 · noisy 밖→밖
  n014: initial 밖→밖 · reject 5→5 · skip 5→5 · oracle 5→5 · noisy 5→5
  n016: initial 밖→밖 · reject 밖→밖 · skip 밖→밖 · oracle 밖→밖 · noisy 밖→밖
  n030: initial 밖→밖 · reject 2→2 · skip 2→2 · oracle 6→6 · noisy 밖→밖
  n033: initial 밖→밖 · reject 밖→10 · skip 밖→10 · oracle 밖→10 · noisy 밖→10
  n040: initial 5→5 · reject 5→5 · skip 5→5 · oracle 5→5 · noisy 5→5
  n042: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  n081: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  n096: initial 밖→밖 · reject 밖→밖 · skip 밖→밖 · oracle 밖→밖 · noisy 밖→밖
  n097: initial 9→9 · reject 9→9 · skip 9→9 · oracle 9→9 · noisy 9→9
