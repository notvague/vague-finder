B(always): 리랭킹 상태 {'applied': 15} · attr_filter_weight=0.5 · use_genre=True · gemini={'backend': 'vertex', 'location': 'global', 'setting': 'vertex'}
D(conditional): 리랭킹 상태 {'applied': 15} · attr_filter_weight=0.5 · use_genre=True · gemini={'backend': 'vertex', 'location': 'global', 'setting': 'vertex'}

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
  q101: initial 3→3 · reject 3→3 · skip 3→3 · oracle 3→3 · noisy 3→3
  q104: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  q112: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  q115: initial 밖→밖 · reject 밖→밖 · skip 밖→밖 · oracle 밖→밖 · noisy 밖→밖
  q206: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
  q308: initial 1→1 · reject 1→1 · skip 1→1 · oracle 1→1 · noisy 1→1
