B(always): 리랭킹 상태 {'applied': 42} · attr_filter_weight=0.5 · use_genre=True · gemini={'backend': 'vertex', 'location': 'global', 'setting': 'vertex'}
D(conditional): 리랭킹 상태 {'applied': 34} · attr_filter_weight=0.5 · use_genre=True · gemini={'backend': 'vertex', 'location': 'global', 'setting': 'vertex'}

질의 11건: n002, n011, n014, n016, n030, n033, n040, n042, n081, n096, n097

달라진 턴 24개 (질의 · 흐름 · 턴: 순위 B(always)→D(conditional) · 후보 순위 → · 답 → · 후보 집합 같음?)
  n016 flow:reject 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n016 flow:reject 턴3: 순위 밖→1 · 후보 밖→28 · 답 :none= · 후보 집합 다름
  n016 flow:skip 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n016 flow:skip 턴3: 순위 밖→1 · 후보 밖→28 · 답 :none= · 후보 집합 다름
  n016 flow:oracle 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n016 flow:oracle 턴3: 순위 밖→1 · 후보 밖→28 · 답 :none= · 후보 집합 다름
  n016 flow:noisy 턴2: 순위 밖→밖 · 후보 밖→밖 · 답 :none= · 후보 집합 다름
  n016 flow:noisy 턴3: 순위 밖→1 · 후보 밖→28 · 답 :none= · 후보 집합 다름
  n033 flow:reject: 최종 순위 밖→5 · D(conditional)는 1턴에 발견 — 흐름 안 돔
  n033 flow:skip: 최종 순위 밖→5 · D(conditional)는 1턴에 발견 — 흐름 안 돔
  n033 flow:oracle: 최종 순위 밖→5 · D(conditional)는 1턴에 발견 — 흐름 안 돔
  n033 flow:noisy: 최종 순위 밖→5 · D(conditional)는 1턴에 발견 — 흐름 안 돔
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
| initial | 11 | 7 | 8 | 1 ['n033'] | 0  | 7 | 8 |
| flow:reject | 11 | 7 | 9 | 2 ['n016', 'n033'] | 0  | 7 | 9 |
| flow:skip | 11 | 7 | 9 | 2 ['n016', 'n033'] | 0  | 7 | 9 |
| flow:oracle | 11 | 8 | 10 | 2 ['n016', 'n033'] | 0  | 8 | 10 |
| flow:noisy | 11 | 7 | 9 | 2 ['n016', 'n033'] | 0  | 7 | 9 |

질의별 최종 순위(흐름: B→C):
  n002: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  n011: initial 밖→밖 · reject 밖→밖 · skip 밖→밖 · oracle 1→1 · noisy 밖→밖
  n014: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  n016: initial 밖→밖 · reject 밖→1 · skip 밖→1 · oracle 밖→1 · noisy 밖→1
  n030: initial 4→4 · reject 4→4 · skip 4→4 · oracle 4→4 · noisy 4→4
  n033: initial 밖→5 · reject 밖→5 · skip 밖→5 · oracle 밖→5 · noisy 밖→5
  n040: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  n042: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  n081: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  n096: initial 밖→밖 · reject 밖→밖 · skip 밖→밖 · oracle 밖→밖 · noisy 밖→밖
  n097: initial 3→3 · reject 3→3 · skip 3→3 · oracle 3→3 · noisy 3→3
