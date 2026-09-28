import json, sys
from pathlib import Path
from src.embedding.models.text_bm25 import BM25SparseEncoder
passages = json.load(open(sys.argv[1], encoding="utf-8"))
enc = BM25SparseEncoder(params_path=Path(sys.argv[2]))
enc.fit([passages[k] for k in sorted(passages)]).dump()
p = json.load(open(sys.argv[2]))
print(f"n_docs={p['n_docs']} avgdl={p['avgdl']:.2f} 고유토큰={len(p['doc_freq']['indices'])}")
