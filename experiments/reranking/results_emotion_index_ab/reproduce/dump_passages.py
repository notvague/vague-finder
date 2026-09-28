import glob, json, sys
from pathlib import Path
from src.embedding.text.passage_builder import build_sparse_passage
ids = {Path(p).stem for p in glob.glob("artifacts/embeddings/text_dense/*/*.npy")}
out = {}
for line in open("data/all_songs.jsonl", encoding="utf-8"):
    song = json.loads(line); sid = str(song.get("song_id"))
    if sid in ids: out[sid] = build_sparse_passage(song)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(len(ids), "개 임베딩 /", len(out), "개 passage")
