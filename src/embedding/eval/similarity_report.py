"""
eval/similarity_report.py
- 임베딩 품질을 빠르게 확인하기 위한 유사도 리포트 모듈

[추가]
- text_dense_values / text_sparse_values / image_values 실제 생성 결과를
  확인 할 수 있도록 디버그 출력 함수 추가

[확장]
- audio_values(CLAP) 디버그 출력 + 유사도 리포트 섹션 추가
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _fmt_head(vec: np.ndarray, n: int = 5) -> str:
    """벡터 앞 n개를 보기 좋게 출력."""
    head = vec[:n].tolist()
    return ", ".join([f"{x:.6f}" for x in head])

def print_embedding_values(
    songs: List[Dict],
    *,
    # 출력할 곡의 개수
    max_songs: int = 300,
    dense_head: int = 3,
    image_head: int = 3,
    audio_head: int = 3,
    sparse_topk: int = 5,
) -> None:
    print("\n==================== Embedding Values Debug ====================\n")

    for idx, s in enumerate(songs[:max_songs]):
        song_id = s.get("id", "N/A")
        title = s.get("metadata", {}).get("title", "")
        print(f"[{idx}] {song_id} | {title}")

        dense = s.get("text_dense_values")
        if dense is None:
            print("  - text_dense_values: None")
        else:
            dense = np.asarray(dense)
            print(f"  - text_dense_values: dim={dense.shape[0]} | head({dense_head})=[{_fmt_head(dense, dense_head)}]")

        sparse = s.get("text_sparse_values")
        if sparse is None:
            print("  - text_sparse_values: None")
        else:
            indices = sparse.get("indices", []) or []
            values = sparse.get("values", []) or []
            nnz = min(len(indices), len(values))
            print(f"  - text_sparse_values: nnz={nnz}")
            if nnz > 0:
                pairs = list(zip(indices, values))
                pairs.sort(key=lambda x: x[1], reverse=True)
                top = pairs[:sparse_topk]
                top_str = ", ".join([f"(idx={int(i)}, w={float(v):.4f})" for i, v in top])
                print(f"    top{min(sparse_topk, nnz)} weights: {top_str}")

        img = s.get("image_values")
        if img is None:
            print("  - image_values: None")
        else:
            img = np.asarray(img)
            print(f"  - image_values: dim={img.shape[0]} | head({image_head})=[{_fmt_head(img, image_head)}]")

        aud = s.get("audio_values")
        if aud is None:
            print("  - audio_values: None")
        else:
            aud = np.asarray(aud)
            print(f"  - audio_values: dim={aud.shape[0]} | head({audio_head})=[{_fmt_head(aud, audio_head)}]")

        print()

    print("================================================================\n")


def print_embedding_similarities(songs: List[Dict]) -> None:
    """
    songs 내에 실제로 값이 채워진 임베딩들만 골라서 유사도 출력.
    - text_dense_values가 하나라도 있으면 텍스트 dense 유사도 출력
    - text_sparse_values가 하나라도 있으면 텍스트 sparse 유사도 출력
    - image_values가 하나라도 있으면 이미지 유사도 출력
    - audio_values가 하나라도 있으면 오디오 유사도 출력
    """

    def _has_any(key: str) -> bool:
        for s in songs:
            if s.get(key) is not None:
                return True
        return False

    ids = [s.get("id") for s in songs]
    titles = [s.get("metadata", {}).get("title") for s in songs]

    # -------------------- TEXT DENSE --------------------
    if _has_any("text_dense_values"):
        print("\n================ Text Dense Vector Similarity (KO-E5) ================\n")
        vectors = [s.get("text_dense_values") for s in songs]

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                v1, v2 = vectors[i], vectors[j]
                if v1 is None or v2 is None:
                    continue
                cosine_sim = _cosine(v1, v2)
                cosine_sim = max(min(cosine_sim, 1.0), -1.0)
                angle_deg = float(np.arccos(cosine_sim) * 180.0 / np.pi)
                print(
                    f"{ids[i]} ({titles[i]})  <->  {ids[j]} ({titles[j]})\n"
                    f"  cosine similarity : {cosine_sim:.4f}\n"
                    f"  angle (degree)    : {angle_deg:.2f}°\n"
                )
        print("==========================================================================\n")

    # -------------------- TEXT SPARSE (BM25) --------------------
    if _has_any("text_sparse_values"):
        print("\n================= Text Sparse Vector Similarity (BM25) =================\n")

        def _sparse_to_dict(sv: Dict) -> Dict[int, float]:
            indices = sv.get("indices", []) or []
            values = sv.get("values", []) or []
            out: Dict[int, float] = {}
            for i, v in zip(indices, values):
                ii = int(i)
                vv = float(v)
                out[ii] = out.get(ii, 0.0) + vv
            return out

        def _sparse_dot(a: Dict, b: Dict) -> float:
            da = _sparse_to_dict(a)
            db = _sparse_to_dict(b)
            if len(da) > len(db):
                da, db = db, da
            s = 0.0
            for k, va in da.items():
                vb = db.get(k)
                if vb is not None:
                    s += va * vb
            return float(s)

        def _sparse_norm(a: Dict) -> float:
            da = _sparse_to_dict(a)
            return float(np.sqrt(sum(v * v for v in da.values())))

        def _sparse_cosine(a: Dict, b: Dict) -> float:
            dot = _sparse_dot(a, b)
            na = _sparse_norm(a)
            nb = _sparse_norm(b)
            if na == 0.0 or nb == 0.0:
                return 0.0
            return float(dot / (na * nb))

        sparse_vecs = [s.get("text_sparse_values") for s in songs]

        for i in range(len(sparse_vecs)):
            for j in range(i + 1, len(sparse_vecs)):
                a, b = sparse_vecs[i], sparse_vecs[j]
                if a is None or b is None:
                    continue
                dot = _sparse_dot(a, b)
                cos = _sparse_cosine(a, b)
                print(
                    f"{ids[i]} ({titles[i]})  <->  {ids[j]} ({titles[j]})\n"
                    f"  sparse dot        : {dot:.6f}\n"
                    f"  sparse cosine     : {cos:.6f}\n"
                )
        print("==========================================================================\n\n")

    # -------------------- IMAGE --------------------
    if _has_any("image_values"):
        print("======================= Image Vector Similarity (SigLIP2) ===================\n")
        vectors = [s.get("image_values") for s in songs]

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                v1, v2 = vectors[i], vectors[j]
                if v1 is None or v2 is None:
                    continue
                cosine_sim = _cosine(v1, v2)
                cosine_sim = max(min(cosine_sim, 1.0), -1.0)
                angle_deg = float(np.arccos(cosine_sim) * 180.0 / np.pi)
                print(
                    f"{ids[i]} ({titles[i]}) <-> {ids[j]} ({titles[j]})\n"
                    f"  cosine similarity : {cosine_sim:.4f}\n"
                    f"  angle (degree)    : {angle_deg:.2f}°\n"
                )
        print("==========================================================================\n\n")

    # -------------------- AUDIO (CLAP) --------------------
    if _has_any("audio_values"):
        print("======================= Audio Vector Similarity (CLAP) ===================\n")
        vectors = [s.get("audio_values") for s in songs]
    
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                v1, v2 = vectors[i], vectors[j]
                if v1 is None or v2 is None:
                    continue
                cosine_sim = _cosine(v1, v2)
                cosine_sim = max(min(cosine_sim, 1.0), -1.0)
                angle_deg = float(np.arccos(cosine_sim) * 180.0 / np.pi)
                print(
                    f"{ids[i]} ({titles[i]}) <-> {ids[j]} ({titles[j]})\n"
                    f"  cosine similarity : {cosine_sim:.4f}\n"
                    f"  angle (degree)    : {angle_deg:.2f}°\n"
                )
        print("==========================================================================")