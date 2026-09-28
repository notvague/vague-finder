"""
src/embedding/models/bm25_sparse.py

BM25 기반 sparse 벡터 생성 모듈
- BM25Encoder로 corpus에 fit한 뒤 sparse vector 생성
- 결과는 Pinecone sparse_values 포맷(dict: indices, values)

주의:
- BM25는 corpus 통계(df)가 필요하므로, 문서(곡) 전체로 fit을 먼저 해야 함.
- query 단계에서도 동일한 bm25 파라미터로 encode_queries를 해야 함.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pinecone_text.sparse import BM25Encoder  # pinecone-text 패키지

from src.embedding.text.korean_bm25_tokenizer import normalize_for_bm25


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class BM25SparseEncoder:
    """
    BM25 sparse encoder 래퍼
    - fit/encode_documents/encode_queries
    - dump/load로 파라미터 저장/재사용

    params_path:
    - 학습된 BM25 파라미터(json)를 저장/로드할 위치
    """
    params_path: Path = Path("artifacts/bm25_params.json")
    _bm25: Optional[BM25Encoder] = None
    _is_fitted: bool = False

    def bm25(self) -> BM25Encoder:
        if self._bm25 is None:
            self._bm25 = BM25Encoder(
                lower_case=False,
                remove_punctuation=False,
                remove_stopwords=False,
                stem=False,
            )
        return self._bm25

    def fit(self, corpus_texts: Sequence[str]) -> "BM25SparseEncoder":
        """
        corpus_texts: 전체 문서(곡 passage) 텍스트 목록
        - BM25는 corpus 기반 통계(df)가 필요하므로 fit이 필수.
        """
        normed = [normalize_for_bm25(t) for t in corpus_texts]
        self.bm25().fit(normed)  # corpus 통계(df) 학습
        self._is_fitted = True
        return self

    def encode_documents(self, docs: Union[str, Sequence[str]]) -> Union[Dict, List[Dict]]:
        """
        문서 텍스트 -> sparse_values 생성
        반환 포맷:
          {"indices": [...], "values": [...]}
        """
        if isinstance(docs, str):
            return self.bm25().encode_documents(normalize_for_bm25(docs))
        normed = [normalize_for_bm25(d) for d in docs]
        return self.bm25().encode_documents(normed)

    def encode_queries(self, query: Union[str, Sequence[str]]) -> Union[Dict, List[Dict]]:
        """
        쿼리 텍스트 -> sparse_vector 생성
        반환 포맷:
          {"indices": [...], "values": [...]}
        """
        if isinstance(query, str):
            return self.bm25().encode_queries(normalize_for_bm25(query))
        normed = [normalize_for_bm25(q) for q in query]
        return self.bm25().encode_queries(normed)

    def dump(self) -> None:
        """
        BM25 파라미터를 json으로 저장
        """
        _ensure_parent_dir(self.params_path)
        self.bm25().dump(str(self.params_path))

    def load(self) -> "BM25SparseEncoder":
        """
        json에서 BM25 파라미터 로드
        """
        if not self.params_path.exists():
            raise FileNotFoundError(f"BM25 params not found: {self.params_path}")
        self.bm25().load(str(self.params_path))
        self._is_fitted = True
        return self

    def load_or_fit(self, corpus_texts: Sequence[str]) -> "BM25SparseEncoder":
        """
        params 파일이 있으면 load, 없으면 fit 후 dump
        - 팀/배포 환경에서 재현성을 확보하기 위한 패턴
        """
        if self.params_path.exists():
            return self.load()
        self.fit(corpus_texts)
        self.dump()
        return self