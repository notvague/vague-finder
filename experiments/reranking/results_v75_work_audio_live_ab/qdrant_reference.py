"""v65 — Qdrant 조회의 **독립 전체 점수 기준**과 동점·경계 분류.

서비스 구현(`QdrantIndex`·`ContextQdrantSearch`)을 쓰지 않는다. 컬렉션의 벡터·payload 전부를 한 번 읽어 질의마다
float64로 직접 점수를 계산하고, v65 계약(점수 내림차순 → 원래 식별자 키)으로 정렬한다.

- dense: COSINE 컬렉션은 질의를 L2 정규화해 저장 벡터(적재 때 정규화됨)와 내적, DOT은 그대로 내적.
- sparse: 역색인 합(질의 인덱스와 겹치는 항만). 겹치는 항이 없는 점은 후보가 아니다(Qdrant sparse 조회와 같은 의미).
- 텍스트 하이브리드: v61 `ExactHybridReference.score`(dense 전체 + sparse 합)를 그대로 쓴다.
- 필터: `scroll(scroll_filter=…)`로 허용 점 집합을 만든다(필터 의미 해석은 Qdrant에 맡긴다, v61과 같다).
- **동점**: float64 점수가 정확히 같을 때만. 저장 벡터가 바이트까지 같으면 어떤 질의에서도 같은 값이 나온다.
  0 < |차이| ≤ `NEAR_TIE`는 "근소 차이"로 따로 센다(동점으로 치지 않는다 — 백엔드 f32 누적과 f64의 차이 범위).

분류(`classify`): 구현 결과(id 목록)를 같은 질의의 기준과 비교한다.
  identical            id·순서 모두 기준과 같다
  tie_order_only       집합이 같고, 순서 차이가 모두 정확한 동점 묶음 안에서만 난다
  boundary_tie_set     k번째 점수의 동점 묶음이 경계를 넘는데(기준에서 그 묶음 일부가 k 밖), 구현이 그 묶음에서 다른 점을 골랐다
  near_tie             차이가 근소 차이(≤ NEAR_TIE) 쌍의 순서로만 설명된다
  other                그 밖(누락·여분·점수 순서 위반) — 결함 후보
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V61 = ROOT / "experiments/reranking/results_v61_hybrid_scan_audit"
if str(V61) not in sys.path:
    sys.path.insert(0, str(V61))

NEAR_TIE = 1e-6


def song_key(song_id: str) -> Tuple[int, int, str]:
    """v65 계약 §3-2 — 서비스 구현과 따로 적는다(구현을 따라 쓰지 않는다)."""
    s = str(song_id).strip()
    if s and s.isascii() and s.isdigit():
        return (0, int(s), s)
    return (1, 0, s)


def _scroll(client: Any, name: str, *, vectors: bool, payload: Any, page: int = 512, scroll_filter: Any = None) -> List[Any]:
    out, offset, seen = [], None, set()
    while True:
        pts, offset = client.scroll(collection_name=name, limit=page, offset=offset, with_vectors=vectors,
                                    with_payload=payload, scroll_filter=scroll_filter)
        for p in pts:
            if p.id in seen:
                raise RuntimeError(f"scroll 중복 point {p.id} in {name}")
            seen.add(p.id)
        out.extend(pts)
        if offset is None:
            return out


class CollectionReference:
    """한 컬렉션의 기준. kind: 'song'(곡 컬렉션) · 'fact'(Context dense) · 'profile'(Context sparse)."""

    def __init__(self, client: Any, name: str, kind: str) -> None:
        self.client = client
        self.name = name
        self.kind = kind
        info = client.get_collection(name)
        vectors_cfg = info.config.params.vectors or {}
        dense_cfg = vectors_cfg.get("dense") if isinstance(vectors_cfg, dict) else None
        self.cosine = bool(dense_cfg and str(dense_cfg.distance.value).lower().startswith("cos"))
        points = _scroll(client, name, vectors=True, payload=True)
        self.point_ids = [p.id for p in points]
        self.row_of = {pid: i for i, pid in enumerate(self.point_ids)}
        self.keys: List[tuple] = []
        self.ident: List[str] = []
        for p in points:
            pl = p.payload or {}
            sid = str(pl.get("song_id", ""))
            if kind == "fact":
                rid = str(pl.get("record_id", ""))
                self.keys.append((song_key(sid), rid))
                self.ident.append(rid)
            elif kind == "profile":
                prof = str(pl.get("profile_id", ""))
                self.keys.append((song_key(sid), prof))
                self.ident.append(prof)
            else:
                self.keys.append((song_key(sid),))
                self.ident.append(sid)
        dense = [((p.vector or {}).get("dense")) for p in points]
        self.dense = np.asarray(dense, dtype=np.float64) if dense and dense[0] is not None else None
        self.postings: Dict[int, Tuple[List[int], List[float]]] = {}
        for row, p in enumerate(points):
            sv = (p.vector or {}).get("sparse")
            if sv is None:
                continue
            for idx, val in zip(sv.indices, sv.values):
                rows, vals = self.postings.setdefault(int(idx), ([], []))
                rows.append(row)
                vals.append(float(val))
        self._filter_cache: Dict[str, Optional[set]] = {}

    def allowed(self, qdrant_filter: Any) -> Optional[set]:
        if qdrant_filter is None:
            return None
        key = qdrant_filter.model_dump_json() if hasattr(qdrant_filter, "model_dump_json") else json.dumps(str(qdrant_filter))
        if key not in self._filter_cache:
            pts = _scroll(self.client, self.name, vectors=False, payload=False, page=2048, scroll_filter=qdrant_filter)
            self._filter_cache[key] = {p.id for p in pts}
        return self._filter_cache[key]

    def scores(self, *, using: str, query: Any, allowed: Optional[set]) -> Tuple[np.ndarray, np.ndarray]:
        """(rows, scores) — 후보가 되는 점만."""
        if using == "dense":
            q = np.asarray(list(query), dtype=np.float64)
            if self.cosine:
                q = q / np.linalg.norm(q)
            s = self.dense @ q
            rows = np.arange(len(self.point_ids))
        else:
            idx = [int(i) for i in query.indices]
            val = [float(v) for v in query.values]
            acc: Dict[int, float] = {}
            for i, v in zip(idx, val):
                post = self.postings.get(i)
                if not post:
                    continue
                for r, dv in zip(*post):
                    acc[r] = acc.get(r, 0.0) + dv * v
            rows = np.asarray(sorted(acc), dtype=np.int64)
            s = np.asarray([acc[r] for r in rows.tolist()], dtype=np.float64)
        if allowed is not None:
            keep = np.asarray([self.point_ids[r] in allowed for r in rows.tolist()], dtype=bool)
            rows, s = rows[keep], s[keep]
        return rows, s

    def ranking(self, rows: np.ndarray, s: np.ndarray) -> List[Tuple[str, float, tuple]]:
        order = sorted(range(len(rows)), key=lambda i: (-float(s[i]), self.keys[int(rows[i])]))
        return [(self.ident[int(rows[i])], float(s[i]), self.keys[int(rows[i])]) for i in order]


def classify(actual_ids: Sequence[str], ranking: List[Tuple[str, float, tuple]], k: int) -> Dict[str, Any]:
    """구현 결과 `actual_ids`(길이 ≤ k)를 기준 전체 순위 `ranking`의 top-k와 비교한다."""
    k_eff = min(k, len(ranking))
    expected = [r[0] for r in ranking[:k_eff]]
    score_of = {r[0]: r[1] for r in ranking}
    out: Dict[str, Any] = {"k": k, "k_eff": k_eff, "n_actual": len(actual_ids)}
    # 경계 동점: k번째 점수와 정확히 같은 점이 k 밖에 몇 개인가
    if k_eff and len(ranking) > k_eff:
        s_k = ranking[k_eff - 1][1]
        inside = sum(1 for r in ranking[:k_eff] if r[1] == s_k)
        beyond = 0
        for r in ranking[k_eff:]:
            if r[1] == s_k:
                beyond += 1
            else:
                break
        out["boundary_tie_inside"] = inside
        out["boundary_tie_beyond"] = beyond
        nxt = ranking[k_eff][1]
        out["boundary_gap"] = s_k - nxt
    else:
        out["boundary_tie_inside"] = out["boundary_tie_beyond"] = 0
        out["boundary_gap"] = None
    # 기준 안의 정확한 동점 묶음(top-k 범위와 겹치는 것)
    tie_pairs = sum(1 for a, b in zip(ranking[:k_eff], ranking[1:k_eff + 1]) if a[1] == b[1])
    near_pairs = sum(1 for a, b in zip(ranking[:k_eff], ranking[1:k_eff + 1]) if 0 < abs(a[1] - b[1]) <= NEAR_TIE)
    out["exact_tie_adjacent_pairs"] = tie_pairs
    out["near_tie_adjacent_pairs"] = near_pairs
    actual = list(actual_ids)
    if actual == expected:
        out["class"] = "identical"
        return out
    unknown = [a for a in actual if a not in score_of]
    if unknown or len(set(actual)) != len(actual):
        out["class"] = "other"
        out["detail"] = {"unknown_or_duplicate": unknown[:5]}
        return out
    # 점수 순서 위반(동점·근소 차이 제외)이 있는가
    act_scores = [score_of[a] for a in actual]
    violation = any(b - a > NEAR_TIE for a, b in zip(act_scores, act_scores[1:]))
    if set(actual) == set(expected):
        exact_groups_ok = all(
            sorted(x for x in actual if score_of[x] == sc) == sorted(x for x in expected if score_of[x] == sc)
            for sc in set(act_scores)
        )
        # 점수만으로 정렬했을 때 같은가(동점 안 순서만 다름)
        by_score_actual = [score_of[a] for a in actual]
        by_score_expected = [score_of[e] for e in expected]
        if by_score_actual == by_score_expected and exact_groups_ok:
            out["class"] = "tie_order_only"
        elif not violation:
            out["class"] = "near_tie"
        else:
            out["class"] = "other"
        return out
    # 집합이 다르다 — 차이가 경계 동점 묶음 안에서만인가
    s_k = ranking[k_eff - 1][1] if k_eff else None
    missing = [e for e in expected if e not in set(actual)]
    extra = [a for a in actual if a not in set(expected)]
    if s_k is not None and all(score_of[x] == s_k for x in missing + extra) and not violation:
        out["class"] = "boundary_tie_set"
    elif s_k is not None and all(abs(score_of[x] - s_k) <= NEAR_TIE for x in missing + extra) and not violation:
        out["class"] = "near_tie"
    else:
        out["class"] = "other"
    out["detail"] = {"missing": missing[:5], "extra": extra[:5]}
    return out


class ReferenceStore:
    """검증 프로세스가 연 클라이언트로 5개 컬렉션의 기준을 만든다(읽기 전용)."""

    def __init__(self, client: Any, text_collection: str) -> None:
        from exact_hybrid import ExactHybridReference

        self.client = client
        self.by_name: Dict[str, CollectionReference] = {}
        self.text_collection = text_collection
        self.hybrid = ExactHybridReference.from_client(client, text_collection)
        self.text_keys = [song_key(s) for s in self.hybrid.song_ids]
        names = sorted(c.name for c in client.get_collections().collections)
        for n in names:
            kind = "fact" if n.startswith("vaguefinder-context-dense") else "profile" if n.startswith("vaguefinder-context-sparse") else "song"
            self.by_name[n] = CollectionReference(client, n, kind)
        aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
        for alias, target in aliases.items():
            if target in self.by_name:
                self.by_name[alias] = self.by_name[target]

    def hybrid_ranking(self, vector: Sequence[float], sparse: Dict[str, Any], allowed: Optional[set]) -> List[Tuple[str, float, tuple]]:
        table = self.hybrid.score(vector, sparse, allowed_point_ids=allowed)
        order = sorted(range(len(table.song_ids)), key=lambda i: (-float(table.total[i]), song_key(table.song_ids[i])))
        return [(table.song_ids[i], float(table.total[i]), (song_key(table.song_ids[i]),)) for i in order]
