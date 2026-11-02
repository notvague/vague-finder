"""E0 미통과 진단(사후, 판정에 쓰지 않음). 문장 비교가 약한지, 임베딩에 정보가 없는지 가른다."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]
from clap_labels import LABELS
from pilot_lib import load_songs
from src.retrieval.user_explanation import _norm
main_dir = Path(sys.argv[1]); songs = load_songs(main_dir / "data/all_songs.jsonl")
emb_dir = main_dir / "artifacts/embeddings/audio/laion__clap-htsat-fused__mean"
ids = list(songs); X = np.stack([np.load(emb_dir / f"{k}.npy") for k in ids]).astype(np.float32)
from src.embedding.models.audio_clap import CLAPAudioEmbedder
emb = CLAPAudioEmbedder().load()
def auc(pos, scores):
    pos = np.asarray(pos); p = pos.sum(); n = len(pos) - p
    r = scores.argsort().argsort() + 1
    return round(float((r[pos].sum() - p * (p + 1) / 2) / (p * n)), 3)
genre = lambda name: [name in (songs[k]["metadata"].get("genre") or []) for k in ids]
T = {s: np.asarray(emb.embed_texts([s], l2_normalize=True), dtype=np.float32)[0] for s in ("a person rapping", "a ballad song", "a dance pop song", "a rock song with loud guitars")}
out = {"sentence_vs_genre": {
    "a person rapping × 랩/힙합": auc(genre("랩/힙합"), X @ T["a person rapping"]),
    "a ballad song × 발라드": auc(genre("발라드"), X @ T["a ballad song"]),
    "a dance pop song × 댄스": auc(genre("댄스"), X @ T["a dance pop song"]),
    "a rock song with loud guitars × 록/메탈": auc(genre("록/메탈"), X @ T["a rock song with loud guitars"]),
}}
# 임베딩 자체의 정보: 5겹 교차검증 로지스틱 회귀(문장 없이 곡 임베딩만)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
def cv(y):
    y = np.asarray(y).astype(int); s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        s[te] = LogisticRegression(max_iter=400, C=10.0, class_weight="balanced").fit(X[tr], y[tr]).decision_function(X[te])
    return round(float(roc_auc_score(y, s)), 3)
out["classifier_cv_genre"] = {name: cv(genre(name)) for name in ("랩/힙합", "발라드", "댄스", "록/메탈")}
stags = {k: [_norm(t) for t in (songs[k]["semantic_analysis"].get("sound_tags") or [])] for k in ids}
has = lambda tags, words: any(any(_norm(w) == t or (len(_norm(w)) >= 2 and _norm(w) in t) for w in words) for t in tags)
out["classifier_cv_sound_tags"] = {l[0]: cv([has(stags[k], l[2]) for k in ids]) for l in LABELS if sum(has(stags[k], l[2]) for k in ids) >= 50}
(HERE / "clap_diagnosis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=0))
