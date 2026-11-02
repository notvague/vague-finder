"""저장 CLAP 임베딩으로 소리 판별(RUN_INFO §6). 유료 호출 없음. 음원을 읽지 않는다."""
import csv, json, re, statistics as st, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]
from clap_labels import LABELS, TOP_SHARE
from pilot_lib import load_songs
from src.retrieval.user_explanation import _norm
main_dir = Path(sys.argv[1])
songs = load_songs(main_dir / "data/all_songs.jsonl")
emb_dir = main_dir / "artifacts/embeddings/audio/laion__clap-htsat-fused__mean"
ids = [k for k in songs if (emb_dir / f"{k}.npy").exists()]
assert len(ids) == 5143, len(ids)
X = np.stack([np.load(emb_dir / f"{k}.npy") for k in ids]).astype(np.float32)
# 저장 임베딩은 색인 벡터와 같다(무작위 5곡 코사인 1.0, 이미 평균을 뺀 값). 또 빼면 조용히 다른 벡터가 된다.
assert float(np.linalg.norm(X.mean(axis=0))) < 0.2, "저장 임베딩이 평균을 뺀 값이 아니다"
assert float(np.abs(np.linalg.norm(X, axis=1) - 1).max()) < 1e-3, "저장 임베딩이 정규화되지 않았다"
from src.embedding.models.audio_clap import CLAPAudioEmbedder  # noqa: E402
emb = CLAPAudioEmbedder().load()
T = np.asarray(emb.embed_texts([l[1] for l in LABELS], l2_normalize=True), dtype=np.float32)
S = X @ T.T
order = S.argsort(axis=0)
ranks = np.empty_like(order); N = len(ids)
for j in range(S.shape[1]):
    ranks[order[:, j], j] = np.arange(N)
pct = ranks / (N - 1)   # 1에 가까울수록 상위
TOP = 1 - TOP_SHARE
idx = {k: i for i, k in enumerate(ids)}
stags = {k: [_norm(t) for t in (songs[k]["semantic_analysis"].get("sound_tags") or [])] for k in ids}
all_tags = {k: [_norm(t) for key, v in songs[k]["semantic_analysis"].items() if isinstance(v, list) for t in v] for k in ids}
def has(tags, words):
    w = [_norm(x) for x in words]
    return any(any(y == t or (len(y) >= 2 and y in t) for y in w) for t in tags)
def auc(pos, scores):
    pos = np.asarray(pos); p = pos.sum(); n = len(pos) - p
    if not p or not n: return None
    r = scores.argsort().argsort() + 1
    return float((r[pos].sum() - p * (p + 1) / 2) / (p * n))
rows = []
for j, (label, sentence, tag_words, _) in enumerate(LABELS):
    pos = [has(stags[k], tag_words) for k in ids]
    a = auc(pos, S[:, j]); p = np.asarray(pos); top = pct[:, j] >= TOP
    rows.append({"label": label, "sentence": sentence, "tag_songs": int(p.sum()), "auc": round(a, 3) if a else None,
                 "tag_share_in_top15": round(float(p[top].mean()), 3), "base": round(float(p.mean()), 3),
                 "lift": round(float(p[top].mean() / p.mean()), 2) if p.mean() else None})
others = [r for r in rows if r["label"] != "랩" and r["auc"] is not None]
e0 = next(r for r in rows if r["label"] == "랩")["auc"]
# dev 단서
ent = {}; gold = {}
for s in ("v06", "v09", "v10"):
    ent.update(json.load((ROOT / f"experiments/reranking/results_v49_vertex_baseline/analysis_cache_{s}_dev_vertex.json").open(encoding="utf-8"))["entries"])
    for r in csv.DictReader((ROOT / f"experiments/reranking/eval_queries_{s}.csv").open(encoding="utf-8-sig")):
        if r.get("split") == "dev":
            gold[r["query_id"]] = [x for x in re.split(r"[|,;\s]+", r["relevant_ids"]) if x in idx]
cases = []
for qid, gids in gold.items():
    q = _norm(ent[qid]["query"])
    for j, (label, _, tag_words, clue_words) in enumerate(LABELS):
        if label == "랩" or not any(_norm(w) in q for w in clue_words):
            continue
        for g in gids:
            cases.append({"query_id": qid, "label": label, "song": g, "tagged": has(all_tags[g], tag_words), "pct": round(float(pct[idx[g], j]), 3), "top15": bool(pct[idx[g], j] >= TOP)})
missing = [c for c in cases if not c["tagged"]]
hit_missing = sum(c["top15"] for c in missing)
# E3: 무작위 곡 비교 — 같은 단서를 전체 곡에 적용한 평균
old_gold = sum(c["tagged"] for c in cases); new_gold = sum(c["tagged"] or c["top15"] for c in cases)
old_all = new_all = 0.0
for c in cases:
    j = [l[0] for l in LABELS].index(c["label"]); tw = LABELS[j][2]
    t = np.asarray([has(all_tags[k], tw) for k in ids]); top = pct[:, j] >= TOP
    old_all += float(t.mean()); new_all += float((t | top).mean())
n = len(cases)
res = {"embeddings": N, "labels": rows, "E0_rap_auc": e0, "median_auc_excl_rap": round(st.median(r["auc"] for r in others), 3), "median_lift_excl_rap": round(st.median(r["lift"] for r in others), 2),
       "dev": {"cases": n, "missing_tag_cases": len(missing), "missing_in_top15": hit_missing, "missing_rate": round(hit_missing / len(missing), 3) if missing else None,
               "by_label": {l[0]: {"cases": sum(c["label"] == l[0] for c in cases), "missing": sum(c["label"] == l[0] for c in missing), "missing_top15": sum(c["label"] == l[0] and c["top15"] for c in missing)} for l in LABELS if l[0] != "랩"},
               "gold_old": round(old_gold / n, 3) if n else None, "gold_union": round(new_gold / n, 3) if n else None, "random_old": round(old_all / n, 3) if n else None, "random_union": round(new_all / n, 3) if n else None}}
d = res["dev"]
lift_old = d["gold_old"] / d["random_old"] if d["random_old"] else 0; lift_new = d["gold_union"] / d["random_union"] if d["random_union"] else 0
d["lift_old"] = round(lift_old, 2); d["lift_union"] = round(lift_new, 2)
res["gates"] = {
    "E0": e0 is not None and e0 >= 0.85,
    "E1": res["median_auc_excl_rap"] >= 0.70 and res["median_lift_excl_rap"] >= 2.5,
    "E2": ("표본 부족" if len(missing) < 10 else bool(d["missing_rate"] >= 0.30)),
    "E3": bool(d["gold_union"] - d["gold_old"] >= 0.10 and lift_new >= lift_old),
}
(HERE / "clap_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "runs").mkdir(exist_ok=True); (HERE / "runs/clap_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
