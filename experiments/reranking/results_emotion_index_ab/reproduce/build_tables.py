"""RUN_INFO.md가 인용하는 표 3개를 A/B 산출물에서 다시 만든다.

사용 (wB worktree 루트에서 — passage_emotion_changes는 arm_B.patch의 코드가 필요하다):
    PYTHONPATH=. python <reproduce>/build_tables.py <작업폴더> <출력폴더>

<작업폴더>에는 result_A/, result_B/, params_A.json, params_B.json이 있어야 한다.
만드는 것: rank_changes.csv, query_token_idf.csv, passage_emotion_changes.csv
"""
import collections
import csv
import glob
import json
import math
import sys
from pathlib import Path

from src.common.emotion_vocab import emotion_index_terms          # arm_B.patch에만 있다
from src.embedding.models.text_bm25 import BM25SparseEncoder
from src.embedding.text.korean_bm25_tokenizer import tokenize_for_bm25

SCR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

# 질의에 감정 표현이 들어 있는지 가늠하는 어간 (질의 원문 + 분석의 korean_tags에서 찾는다)
EMOTION_STEMS = ["슬프", "슬픈", "그리", "기쁘", "기쁜", "신나", "설레", "쓸쓸", "추억", "위로", "행복", "외로"]
IDF_EXPRESSIONS = ["슬픈", "그리운", "기쁜", "신나는", "설레는", "쓸쓸한", "추억의", "위로되는",
                   "사랑하는", "당당한", "화난", "희망찬"]


def write_csv(name, rows):
    with open(OUT / name, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows)}행")


# 1) 질의별 순위
def load(arm):
    path = SCR / f"result_{arm}" / "search_eval_all_detail.csv"
    return {r["query_id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}

A, B = load("A"), load("B")
rank_rows = []
for q in A:
    a, b = A[q], B[q]
    row = {"query_id": q, "split": a["split"],
           "has_emotion_expression": int(any(w in a["query"] + a["korean_tags"] for w in EMOTION_STEMS)),
           "same_analysis": int(a["analysis_json"] == b["analysis_json"])}
    changed = False
    for col in ("candidate_rank@30", "baseline_rank", "rerank_rank"):
        row[f"A_{col}"], row[f"B_{col}"] = a[col], b[col]
        changed |= a[col] != b[col]
    row["rank_changed"] = int(changed)
    rank_rows.append(row)
write_csv("rank_changes.csv", rank_rows)

# 2) 질의 토큰의 문서빈도·IDF (Pinecone BM25Encoder._encode_single_query와 같은 식).
#    코퍼스에 없는 토큰은 인코더가 df=1로 계산하므로 IDF도 그렇게 맞춘다. 문서빈도 열은 실제 값(0).
enc = {arm: BM25SparseEncoder(params_path=SCR / f"params_{arm}.json").load() for arm in "AB"}
n_docs = enc["B"].bm25().n_docs
idf = lambda df: math.log((n_docs + 1) / (df + 0.5))
idf_rows = []
for expression in IDF_EXPRESSIONS:
    encoded = enc["B"].encode_queries(expression)
    for token, index in zip(tokenize_for_bm25(expression), encoded["indices"]):
        df_a = int(enc["A"].bm25().doc_freq.get(index, 0))
        df_b = int(enc["B"].bm25().doc_freq.get(index, 0))
        idf_rows.append({"query_expression": expression, "token": token,
                         "A_doc_freq": df_a, "B_doc_freq": df_b,
                         "A_idf": round(idf(df_a or 1), 3), "B_idf": round(idf(df_b or 1), 3)})
write_csv("query_token_idf.csv", idf_rows)

# 3) 감정 줄이 어떻게 바뀌었나 (임베딩이 있는 곡만)
ids = {Path(p).stem for p in glob.glob("artifacts/embeddings/text_dense/*/*.npy")}
transitions = collections.Counter()
for line in open("data/all_songs.jsonl", encoding="utf-8"):
    song = json.loads(line)
    if str(song.get("song_id")) not in ids:
        continue
    raw = (song.get("community_feedback") or {}).get("major_emotion") or ""
    semantic = song.get("semantic_analysis") or {}
    tags = (semantic.get("emotion_tags") or []) + (semantic.get("mood_tags") or [])
    transitions[(raw, emotion_index_terms(raw, tags))] += 1
with open(OUT / "passage_emotion_changes.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["A_major_emotion", "B_index_terms", "songs"])
    for (a_value, b_value), count in transitions.most_common():
        writer.writerow([a_value, b_value or "(비움)", count])
print(f"passage_emotion_changes.csv: {len(transitions)}행 / {sum(transitions.values())}곡")
