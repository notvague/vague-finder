"""
artifacts/embeddings/* 를 읽어 로컬 Qdrant에 적재한다.

Pinecone 업로드(upsert_from_artifacts.py)와 같은 입력·같은 메타데이터를 쓴다.
두 백엔드가 다른 데이터를 갖고 있으면 결과를 비교할 수 없다.

    venv/bin/python -m src.vector_db.cli.qdrant_load --recreate

입력 (기본값). 모델 이름 폴더는 자동으로 찾는다.
    artifacts/embeddings/text_dense/<모델>/<song_id>.npy       1024
    artifacts/embeddings/image/<모델>/<song_id>.npy             768
    artifacts/embeddings/audio/<모델>/<song_id>.npy             512
    data/all_songs.jsonl                                       메타데이터 + BM25 sparse 생성

BM25 sparse는 파일로 두지 않고 적재할 때 지금 코드로 만든다(몇 초). 옛 벡터가
남아 코드와 어긋나는 사고를 막기 위해서다. 파일을 쓰려면 --sparse-from-artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from qdrant_client import models

from src.embedding.models.text_bm25 import BM25SparseEncoder
from src.embedding.text.passage_builder import build_sparse_passage
from src.vector_db.metadata_loader import load_metadata_from_jsonl
from src.vector_db.qdrant_backend import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    QdrantVectorClient,
    point_id,
)
from src.vector_db.settings import (
    AUDIO_DIM,
    AUDIO_INDEX_NAME,
    IMAGE_DIM,
    IMAGE_INDEX_NAME,
    NAMESPACE,
    TEXT_DENSE_DIM,
    TEXT_HYBRID_INDEX_NAME,
)

DEFAULT_ARTIFACTS = Path("artifacts/embeddings")
DEFAULT_CORPUS = Path("data/all_songs.jsonl")

# (인덱스, 모달리티 디렉터리, 차원). 모델 이름 하위 폴더는 자동으로 찾는다 —
# 내보낸 사람에 따라 폴더 이름이 달라도 적재가 막히지 않게.
MODALITIES = [
    (TEXT_HYBRID_INDEX_NAME, "text_dense", TEXT_DENSE_DIM),
    (IMAGE_INDEX_NAME, "image", IMAGE_DIM),
    (AUDIO_INDEX_NAME, "audio", AUDIO_DIM),
]
SPARSE_SUBDIR = "text_sparse/bm25"


def resolve_vector_dir(base: Path) -> Optional[Path]:
    """`text_dense/<모델이름>/*.npy` 처럼 한 단계 안쪽에 있는 벡터 폴더를 찾는다.

    모델 폴더 이름(`nlpai-lab__KoE5` 등)을 코드에 박아 두면, 내보낸 환경이 다를 때
    팀원 쪽에서 조용히 "0건 적재"가 된다.
    """
    if not base.exists():
        return None
    if any(base.glob("*.npy")):
        return base
    subdirs = sorted(d for d in base.iterdir() if d.is_dir() and any(d.glob("*.npy")))
    if len(subdirs) == 1:
        return subdirs[0]
    if len(subdirs) > 1:
        raise SystemExit(
            f"[FAIL] {base} 아래에 벡터 폴더가 여러 개입니다: "
            + ", ".join(d.name for d in subdirs)
            + " — 쓸 폴더만 남겨주세요."
        )
    return None


def load_dense(directory: Path, dim: int) -> Dict[str, List[float]]:
    """dense 벡터를 읽는다. 차원과 값이 모두 멀쩡한지 여기서 본다.

    차원만 보면 NaN·Inf가 사전 검증을 통과한다. 그러면 컬렉션을 지우고 다시 만든 뒤
    upsert에서 "Vector contains NaN values"로 죽어, 세 컬렉션이 서로 다른 상태로 갈린다
    (텍스트는 새 데이터, 이미지는 비어 있고, 오디오는 예전 데이터).
    """
    out: Dict[str, List[float]] = {}
    for path in sorted(directory.glob("*.npy")):
        vec = np.load(path).astype(np.float32).reshape(-1)
        if vec.shape[0] != dim:
            raise ValueError(f"{path}: 차원이 {vec.shape[0]}, 기대값 {dim}")
        if not np.isfinite(vec).all():
            bad = int(np.count_nonzero(~np.isfinite(vec)))
            raise ValueError(f"{path}: NaN/Inf가 {bad}개 있습니다 (임베딩을 다시 만들어야 합니다)")
        out[path.stem] = vec.tolist()
    return out


def build_sparse(corpus: Path, song_ids: Iterable[str], params: Path) -> Dict[str, models.SparseVector]:
    """BM25 sparse 벡터를 지금 코드로 그 자리에서 만든다 (기본 동작).

    파일로 들고 다니면 `passage_builder`가 바뀐 뒤에도 옛 벡터가 남는다. 실제로
    레포에 905곡 중 364곡짜리 옛 버전이 남아 있었고, #58이 sparse에서 앨범
    정보를 지운 뒤에도 그 사실이 드러나지 않았다(2026-09-17). 몇 초면 만들어지니
    저장하지 않는다.
    """
    wanted = set(song_ids)
    encoder = BM25SparseEncoder(params_path=params).load()
    out: Dict[str, models.SparseVector] = {}
    with corpus.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            song = json.loads(line)
            song_id = str(song.get("song_id") or song.get("id") or "").strip()
            if song_id not in wanted:
                continue
            encoded = encoder.encode_documents(build_sparse_passage(song))
            if isinstance(encoded, list):
                encoded = encoded[0]
            out[song_id] = models.SparseVector(
                indices=[int(i) for i in encoded["indices"]],
                values=[float(v) for v in encoded["values"]],
            )
    return out


def load_sparse(directory: Path) -> Dict[str, models.SparseVector]:
    out: Dict[str, models.SparseVector] = {}
    for path in sorted(directory.glob("*.npz")):
        data = np.load(path)
        out[path.stem] = models.SparseVector(
            indices=[int(i) for i in data["indices"]],
            values=[float(v) for v in data["values"]],
        )
    return out


class LoadPlan:
    """한 모달리티의 적재 계획. 검증이 끝나기 전에는 컬렉션을 건드리지 않는다."""

    def __init__(self, index_name: str, subdir: str, dim: int, directory: Path,
                 dense: Dict[str, List[float]],
                 sparse: Optional[Dict[str, models.SparseVector]] = None):
        self.index_name = index_name
        self.subdir = subdir
        self.dim = dim
        self.directory = directory
        self.dense = dense
        self.sparse = sparse


def _sample(ids: Iterable[str], limit: int = 5) -> str:
    listed = sorted(ids)
    head = ", ".join(listed[:limit])
    return head + (f" ... (총 {len(listed)}건)" if len(listed) > limit else "")


def validate_plans(
    plans: List["LoadPlan"],
    metadata: Dict[str, dict],
    *,
    corpus: Path,
    allow_missing_metadata: bool = False,
    allow_missing_sparse: bool = False,
) -> List[str]:
    """적재 전에 확인할 것을 모두 본다. 문제 목록을 돌려준다(빈 목록이면 통과).

    컬렉션을 하나라도 바꾸기 **전에** 세 모달리티를 다 검증한다. 예전에는 모달리티마다
    읽고 곧바로 교체해서, 뒤쪽 입력의 차원 오류를 만나기 전에 앞쪽 컬렉션이 이미 바뀌었다.
    그러면 텍스트는 새 데이터, 이미지·오디오는 예전 데이터로 남아 비교가 불가능해진다.
    """
    problems: List[str] = []

    for plan in plans:
        missing_meta = [sid for sid in plan.dense if sid not in metadata]
        if missing_meta and not allow_missing_metadata:
            problems.append(
                f"{plan.index_name}: 메타데이터가 없는 곡 {len(missing_meta)}건 "
                f"({_sample(missing_meta)}). {corpus}에 해당 song_id가 없습니다. "
                f"메타데이터 없이 적재하면 검색 결과에 제목·가수가 비어 나옵니다. "
                f"의도한 것이면 --allow-missing-metadata."
            )

        if plan.sparse is None:
            continue

        # sparse도 값이 멀쩡한지 본다. dense와 같은 이유로, 컬렉션을 건드리기 전에 걸러야 한다.
        bad_sparse = sorted(
            sid for sid, vec in plan.sparse.items()
            if not np.isfinite(np.asarray(vec.values, dtype=np.float64)).all()
        )
        if bad_sparse:
            problems.append(
                f"{plan.index_name}: sparse 값에 NaN/Inf가 있는 곡 {len(bad_sparse)}건 "
                f"({_sample(bad_sparse)}). BM25 파라미터나 sparse 파일을 다시 만들어야 합니다."
            )

        missing_sparse = [sid for sid in plan.dense if sid not in plan.sparse]
        if missing_sparse and not allow_missing_sparse:
            problems.append(
                f"{plan.index_name}: sparse 벡터가 없는 곡 {len(missing_sparse)}건 "
                f"({_sample(missing_sparse)}). 그 곡들은 하이브리드 검색에서 dense 점수만 "
                f"받아 가사·앨범 텍스트로는 찾히지 않습니다. "
                f"의도한 것이면 --allow-missing-sparse."
            )

    # 모달리티 사이 ID 차이는 막지 않는다 — 커버가 없는 곡처럼 정상인 경우가 있다.
    # 다만 조용히 넘기면 '이미지로는 안 찾히는 곡'이 생긴 줄 모르므로 세어서 알린다.
    return problems


def report_id_gaps(plans: List["LoadPlan"]) -> None:
    union = set()
    for plan in plans:
        union |= set(plan.dense)
    for plan in plans:
        missing = union - set(plan.dense)
        if missing:
            print(
                f"[주의] {plan.index_name}: 다른 모달리티에는 있고 여기엔 없는 곡 "
                f"{len(missing)}건 ({_sample(missing)})"
            )


def upsert_modality(
    client: QdrantVectorClient,
    index_name: str,
    dense: Dict[str, List[float]],
    metadata: Dict[str, dict],
    *,
    dim: int,
    namespace: str,
    recreate: bool,
    sparse: Optional[Dict[str, models.SparseVector]] = None,
    batch_size: int = 200,
) -> None:
    collection = client.ensure_collection(
        index_name, dim=dim, namespace=namespace, with_sparse=sparse is not None, recreate=recreate
    )

    points: List[models.PointStruct] = []
    missing_metadata = 0
    for song_id, vec in dense.items():
        payload = dict(metadata.get(song_id) or {})
        if not payload:
            missing_metadata += 1
        # 원본 곡 id는 payload에 보존한다 (Qdrant point id는 정수/UUID만 가능)
        payload["song_id"] = song_id
        vectors: Dict[str, object] = {DENSE_VECTOR: vec}
        if sparse is not None and song_id in sparse:
            vectors[SPARSE_VECTOR] = sparse[song_id]
        points.append(models.PointStruct(id=point_id(song_id), vector=vectors, payload=payload))

    for start in range(0, len(points), batch_size):
        client.client.upsert(collection_name=collection, points=points[start : start + batch_size])

    count = client.client.count(collection_name=collection).count
    sparse_note = f" sparse={sum(1 for s in dense if sparse and s in sparse)}" if sparse else ""
    print(
        f"[OK] {collection}: points={count} (입력 {len(points)}){sparse_note}"
        + (f" · 메타데이터 없음 {missing_metadata}건" if missing_metadata else "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="로컬 Qdrant에 임베딩 적재")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--qdrant-path", default=None, help="기본값: QDRANT_PATH 또는 artifacts/qdrant")
    parser.add_argument("--recreate", action="store_true", help="컬렉션을 지우고 다시 만든다")
    parser.add_argument(
        "--sparse-from-artifacts",
        action="store_true",
        help=f"BM25 벡터를 만들지 않고 {SPARSE_SUBDIR}의 파일을 쓴다 (기본은 즉석 생성)",
    )
    parser.add_argument("--bm25-params", type=Path, default=Path("artifacts/bm25_params.json"))
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="코퍼스에 없는 곡도 song_id만 넣어 적재한다 (검색 결과에 제목·가수가 비어 나온다)",
    )
    parser.add_argument(
        "--allow-missing-sparse",
        action="store_true",
        help="sparse 벡터가 없는 곡도 적재한다 (그 곡은 dense 점수만 받는다)",
    )
    args = parser.parse_args()

    resolved = {index: resolve_vector_dir(args.artifacts / subdir) for index, subdir, _ in MODALITIES}
    missing = [
        str(args.artifacts / subdir)
        for index, subdir, _ in MODALITIES
        if resolved[index] is None
    ]
    if missing:
        # 조용히 건너뛰면 컬렉션이 비어 있는 채로 "성공"해서, 검색이 0건을
        # 돌려줄 때까지 아무도 모른다. 벡터 파일은 git에 없고 드라이브에서
        # 받아야 하므로 이 실수가 실제로 일어난다.
        raise SystemExit(
            "[FAIL] 임베딩 디렉터리가 없습니다:\n  "
            + "\n  ".join(missing)
            + "\n팀 드라이브에서 벡터 파일을 받아 artifacts/embeddings/ 아래에 풀어주세요."
            + " (docs/vector_backend.md)"
        )

    metadata = load_metadata_from_jsonl(args.corpus)

    # --- 1단계: 읽고 검증만 한다. 컬렉션도, 저장 폴더도 아직 열지 않는다. -------------
    #
    # 예전에는 모달리티마다 읽고 곧바로 교체했다. 뒤쪽 입력에 문제가 있으면 그것을 만나기
    # 전에 앞쪽 컬렉션이 이미 바뀌어, 텍스트는 새 데이터·이미지는 예전 데이터로 갈라졌다.
    plans: List[LoadPlan] = []
    for index_name, subdir, dim in MODALITIES:
        directory = resolved[index_name]
        dense = load_dense(directory, dim)          # 차원이 어긋나면 여기서 멈춘다
        if not dense:
            raise SystemExit(f"[FAIL] {directory}에 .npy 벡터가 없습니다.")
        print(f"[IN] {index_name}: {len(dense)}건 dim={dim}")

        sparse: Optional[Dict[str, models.SparseVector]] = None
        if index_name == TEXT_HYBRID_INDEX_NAME:
            if args.sparse_from_artifacts:
                sparse_dir = args.artifacts / SPARSE_SUBDIR
                sparse = load_sparse(sparse_dir)
                print(f"[IN] sparse={len(sparse)}건 (파일: {sparse_dir})")
            else:
                sparse = build_sparse(args.corpus, dense.keys(), args.bm25_params)
                print(f"[IN] sparse={len(sparse)}건 (지금 코드로 생성)")

        plans.append(LoadPlan(index_name, subdir, dim, directory, dense, sparse))

    problems = validate_plans(
        plans,
        metadata,
        corpus=args.corpus,
        allow_missing_metadata=args.allow_missing_metadata,
        allow_missing_sparse=args.allow_missing_sparse,
    )
    if problems:
        raise SystemExit(
            "[FAIL] 적재 전 검증에서 걸렸습니다. 컬렉션은 그대로 둡니다.\n  - "
            + "\n  - ".join(problems)
        )
    report_id_gaps(plans)

    # --- 2단계: 검증을 통과한 뒤에만 적재한다. -----------------------------------------
    #
    # 남는 위험: 적재 도중(네트워크·디스크) 실패하면 앞쪽 컬렉션은 이미 새 데이터다.
    # 그것까지 막으려면 별도 컬렉션에 다 넣고 alias를 바꿔 다는 방식이 필요하다.
    client = QdrantVectorClient(path=args.qdrant_path)
    try:
        for plan in plans:
            upsert_modality(
                client,
                plan.index_name,
                plan.dense,
                metadata,
                dim=plan.dim,
                namespace=args.namespace,
                recreate=args.recreate,
                sparse=plan.sparse,
            )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
