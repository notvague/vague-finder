"""BM25 encoder for already curated Namuwiki context terms.

The existing song BM25 path starts from free-form lyrics and mood text.  A
context artifact is different: ``sparse_profile.terms`` is already tokenized,
deduplicated and capped per song.  Passing those terms through NLTK or another
tokenizer a second time can split entities such as ``짱구는_못말려`` and
``d-e-f#m``.  This encoder therefore consumes token sequences directly.

The weighting and unsigned 32-bit MurmurHash layout match the BM25 sparse
vectors used by ``pinecone-text`` (the song BM25 library) so the output can be
sent to Qdrant as ``{"indices": ..., "values": ...}``.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from typing import Callable, Mapping, Sequence

from src.embedding.text.context_bm25_tokenizer import (
    CONTEXT_BM25_TOKENIZER_VERSION,
)

CONTEXT_BM25_PARAMS_VERSION = "context_bm25_params_v1"
CONTEXT_BM25_HASH_ALGORITHM = "mmh3_32_unsigned"
CONTEXT_BM25_DOCUMENT_TOKENIZER = "pretokenized_sparse_profile_v1"
DEFAULT_CONTEXT_BM25_B = 0.75
DEFAULT_CONTEXT_BM25_K1 = 1.2

HashFunction = Callable[[str], int]


@lru_cache(maxsize=1)
def _mmh3_module():
    try:
        import mmh3
    except ImportError as exc:  # pragma: no cover - exercised in the container
        raise RuntimeError(
            "mmh3 is required for context BM25; install project requirements"
        ) from exc
    return mmh3


def _default_hash(term: str) -> int:
    return int(_mmh3_module().hash(term, signed=False))


def _terms(value: Sequence[str], *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("BM25 terms must be a sequence, not a string")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("BM25 terms must contain only strings")
        term = item.strip()
        if not term or any(character.isspace() for character in term):
            raise ValueError("BM25 terms must be non-empty whitespace-free tokens")
        result.append(term)
    if not result and not allow_empty:
        raise ValueError("BM25 documents must contain at least one term")
    return tuple(result)


class ContextBM25Encoder:
    """Static-corpus BM25 over exact context tokens.

    Document vectors carry BM25 term-frequency/length normalization.  Query
    vectors carry normalized IDF.  Their dot product is the BM25 score used by
    the existing sparse-vector backends.
    """

    def __init__(
        self,
        *,
        b: float = DEFAULT_CONTEXT_BM25_B,
        k1: float = DEFAULT_CONTEXT_BM25_K1,
        hash_fn: HashFunction | None = None,
    ):
        if not 0.0 <= float(b) <= 1.0:
            raise ValueError("BM25 b must be between 0 and 1")
        if float(k1) <= 0.0:
            raise ValueError("BM25 k1 must be positive")
        self.b = float(b)
        self.k1 = float(k1)
        self._hash_fn = hash_fn or _default_hash
        self.doc_freq: dict[int, int] | None = None
        self.n_docs: int | None = None
        self.avgdl: float | None = None

    def hash_term(self, term: str) -> int:
        value = int(self._hash_fn(term))
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("BM25 hash function must return an unsigned 32-bit integer")
        return value

    def _tf(self, terms: Sequence[str], *, allow_empty: bool) -> tuple[list[int], list[int]]:
        tokens = _terms(terms, allow_empty=allow_empty)
        counts = Counter(self.hash_term(term) for term in tokens)
        pairs = sorted(counts.items())
        return [index for index, _ in pairs], [count for _, count in pairs]

    def _require_fitted(self) -> tuple[dict[int, int], int, float]:
        if self.doc_freq is None or self.n_docs is None or self.avgdl is None:
            raise ValueError("context BM25 must be fit before encoding")
        return self.doc_freq, self.n_docs, self.avgdl

    def fit(self, documents: Sequence[Sequence[str]]) -> "ContextBM25Encoder":
        if isinstance(documents, (str, bytes)):
            raise TypeError("BM25 corpus must be a sequence of token sequences")
        if not documents:
            raise ValueError("context BM25 corpus is empty")

        document_frequency: Counter[int] = Counter()
        total_length = 0
        for document in documents:
            indices, frequencies = self._tf(document, allow_empty=False)
            document_frequency.update(indices)
            total_length += sum(frequencies)

        self.n_docs = len(documents)
        self.avgdl = total_length / self.n_docs
        self.doc_freq = dict(sorted(document_frequency.items()))
        return self

    def encode_document_terms(self, terms: Sequence[str]) -> dict[str, list]:
        _doc_freq, _n_docs, avgdl = self._require_fitted()
        indices, frequencies = self._tf(terms, allow_empty=False)
        length = float(sum(frequencies))
        values = [
            float(
                frequency
                / (
                    self.k1 * (1.0 - self.b + self.b * (length / avgdl))
                    + frequency
                )
            )
            for frequency in frequencies
        ]
        return {"indices": indices, "values": values}

    def encode_documents(
        self, documents: Sequence[Sequence[str]]
    ) -> list[dict[str, list]]:
        return [self.encode_document_terms(document) for document in documents]

    def encode_query_terms(self, terms: Sequence[str]) -> dict[str, list]:
        document_frequency, n_docs, _avgdl = self._require_fitted()
        indices, _frequencies = self._tf(terms, allow_empty=True)
        if not indices:
            return {"indices": [], "values": []}
        idf = [
            math.log((n_docs + 1.0) / (document_frequency.get(index, 1) + 0.5))
            for index in indices
        ]
        total = sum(idf)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("context BM25 query produced invalid IDF weights")
        return {
            "indices": indices,
            "values": [float(value / total) for value in idf],
        }

    def get_params(self) -> dict:
        document_frequency, n_docs, avgdl = self._require_fitted()
        pairs = sorted(document_frequency.items())
        return {
            "schema_version": CONTEXT_BM25_PARAMS_VERSION,
            "n_docs": n_docs,
            "avgdl": avgdl,
            "doc_freq": {
                "indices": [int(index) for index, _ in pairs],
                "values": [int(value) for _, value in pairs],
            },
            "b": self.b,
            "k1": self.k1,
            "hash_algorithm": CONTEXT_BM25_HASH_ALGORITHM,
            "document_tokenizer": CONTEXT_BM25_DOCUMENT_TOKENIZER,
            "query_tokenizer": CONTEXT_BM25_TOKENIZER_VERSION,
            "query_synonym_expansion": True,
        }

    @classmethod
    def from_params(
        cls,
        params: Mapping,
        *,
        hash_fn: HashFunction | None = None,
    ) -> "ContextBM25Encoder":
        if params.get("schema_version") != CONTEXT_BM25_PARAMS_VERSION:
            raise ValueError("unsupported context BM25 parameter schema")
        if params.get("hash_algorithm") != CONTEXT_BM25_HASH_ALGORITHM:
            raise ValueError("context BM25 hash algorithm mismatch")
        if params.get("document_tokenizer") != CONTEXT_BM25_DOCUMENT_TOKENIZER:
            raise ValueError("context BM25 document tokenizer mismatch")
        if params.get("query_tokenizer") != CONTEXT_BM25_TOKENIZER_VERSION:
            raise ValueError("context BM25 query tokenizer mismatch")
        if params.get("query_synonym_expansion") is not True:
            raise ValueError("context BM25 query synonym policy mismatch")

        try:
            n_docs = int(params["n_docs"])
            avgdl = float(params["avgdl"])
            b = float(params["b"])
            k1 = float(params["k1"])
            raw_df = params["doc_freq"]
            indices = list(raw_df["indices"])
            values = list(raw_df["values"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("context BM25 parameters are incomplete") from exc
        if n_docs < 1 or not math.isfinite(avgdl) or avgdl <= 0.0:
            raise ValueError("context BM25 corpus statistics are invalid")
        if len(indices) != len(values) or not indices:
            raise ValueError("context BM25 document frequencies are invalid")

        encoder = cls(b=b, k1=k1, hash_fn=hash_fn)
        document_frequency: dict[int, int] = {}
        for raw_index, raw_value in zip(indices, values):
            index, value = int(raw_index), int(raw_value)
            if (
                not 0 <= index <= 0xFFFFFFFF
                or not 1 <= value <= n_docs
                or index in document_frequency
            ):
                raise ValueError("context BM25 document frequencies are invalid")
            document_frequency[index] = value
        encoder.n_docs = n_docs
        encoder.avgdl = avgdl
        encoder.doc_freq = dict(sorted(document_frequency.items()))
        return encoder
