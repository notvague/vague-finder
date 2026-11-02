"""곡 임베딩 분류기의 장르 안 검증(RUN_INFO §7). 유료 호출 없음. 음원을 읽지 않는다."""
import csv, json, re, statistics as st, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]
from clap_labels import LABELS, TOP_SHARE
from pilot_lib import load_songs
from src.retrieval.user_explanation import _norm
warnings.filterwarnings("ignore")
main_dir = Path(sys.argv[1]); songs = load_songs(main_dir / "data/all_songs.jsonl")
emb_dir = main_dir / "artifacts/embeddings/audio/laion__clap-htsat-fused__mean"
ids = list(songs); idx = {k: i for i, k in enumerate(ids)}; N = len(ids)
X = np.stack([np.load(emb_dir / f"{k}.npy") for k in ids]).astype(np.float32)
groups = np.array([(songs[k]["metadata"].get("artist") or ["?"])[0] for k in ids])
genres = [songs[k]["metadata"].get("genre") or [] for k in ids]
GENRES = sorted({g for gs in genres for g in gs})
in_genre = {g: np.array([g in gs for gs in genres]) for g in GENRES}
stags = [[_norm(t) for t in (songs[k]["semantic_analysis"].get("sound_tags") or [])] for k in ids]
all_tags = [[_norm(t) for key, v in songs[k]["semantic_analysis"].items() if isinstance(v, list) for t in v] for k in ids]
def has(tags, words):
    w = [_norm(x) for x in words]
    return any(any(y == t or (len(y) >= 2 and y in t) for y in w) for t in tags)
def oof(y, grouped=True):
    y = np.asarray(y).astype(int); s = np.zeros(N)
    split = StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups) if grouped else StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)
    for tr, te in split:
        s[te] = LogisticRegression(max_iter=400, C=10.0, class_weight="balanced").fit(X[tr], y[tr]).decision_function(X[te])
    return s
def cells(y, s):
    out = {}
    y = np.asarray(y).astype(bool)
    for g, mask in in_genre.items():
        p, n = int((y & mask).sum()), int((~y & mask).sum())
        if p >= 30 and n >= 30:
            out[g] = round(float(roc_auc_score(y[mask], s[mask])), 3)
    return out
rng = np.random.default_rng(0)
labels = [l for l in LABELS if sum(has(t, l[2]) for t in stags) >= 50]
res = {"labels": {}, "control": {}, "groups": int(len(set(groups)))}
scores = {}
for label, _, tag_words, _ in labels:
    y = np.array([has(t, tag_words) for t in stags])
    s = oof(y); scores[label] = s
    res["labels"][label] = {"tag_songs": int(y.sum()), "overall_auc_artist_split": round(float(roc_auc_score(y, s)), 3),
                            "overall_auc_song_split": round(float(roc_auc_score(y, oof(y, grouped=False))), 3), "within_genre": cells(y, s)}
    # 대조군: 장르 안에서 태그를 섞는다(장르별 태그 비율은 유지)
    shuffled = y.copy()
    for mask in [np.array([(gs[0] if gs else "") == g for gs in genres]) for g in sorted({(gs[0] if gs else "") for gs in genres})]:
        shuffled[mask] = rng.permutation(y[mask])
    res["control"][label] = cells(shuffled, oof(shuffled))
all_cells = [v for r in res["labels"].values() for v in r["within_genre"].values()]
ctl_cells = [v for r in res["control"].values() for v in r.values()]
res["cells"] = len(all_cells); res["median_within"] = round(st.median(all_cells), 3) if all_cells else None
res["control_median"] = round(st.median(ctl_cells), 3) if ctl_cells else None; res["control_range"] = [min(ctl_cells), max(ctl_cells)] if ctl_cells else None
high = res["labels"].get("고음", {}).get("within_genre", {}).get("발라드")
res["high_note_in_ballad"] = high
# 장르 안 백분위: 곡이 속한 첫 장르 안에서 순위를 매긴다
def within_pct(s):
    pct = np.zeros(N); first = np.array([gs[0] if gs else "" for gs in genres])
    for g in set(first):
        m = first == g
        pct[m] = s[m].argsort().argsort() / max(1, m.sum() - 1)
    return pct
pcts = {k: within_pct(v) for k, v in scores.items()}
TOP = 1 - TOP_SHARE
ent = {}; gold = {}
for s in ("v06", "v09", "v10"):
    ent.update(json.load((ROOT / f"experiments/reranking/results_v49_vertex_baseline/analysis_cache_{s}_dev_vertex.json").open(encoding="utf-8"))["entries"])
    for r in csv.DictReader((ROOT / f"experiments/reranking/eval_queries_{s}.csv").open(encoding="utf-8-sig")):
        if r.get("split") == "dev":
            gold[r["query_id"]] = [x for x in re.split(r"[|,;\s]+", r["relevant_ids"]) if x in idx]
cases = []
names = [l[0] for l in LABELS]
for qid, gids in gold.items():
    q = _norm(ent[qid]["query"])
    for label, _, tag_words, clue_words in labels:
        if label == "랩" or not any(_norm(w) in q for w in clue_words):
            continue
        for g in gids:
            i = idx[g]
            cases.append({"query_id": qid, "label": label, "song": g, "tagged": has(all_tags[i], tag_words), "pct": round(float(pcts[label][i]), 3), "top15": bool(pcts[label][i] >= TOP)})
miss = [c for c in cases if not c["tagged"]]
hit = sum(c["top15"] for c in miss); n = len(cases)
old_gold = sum(c["tagged"] for c in cases) / n; new_gold = sum(c["tagged"] or c["top15"] for c in cases) / n
old_r = new_r = 0.0
for c in cases:
    tw = next(l[2] for l in labels if l[0] == c["label"])
    t = np.array([has(a, tw) for a in all_tags]); top = pcts[c["label"]] >= TOP
    old_r += float(t.mean()); new_r += float((t | top).mean())
old_r /= n; new_r /= n
res["dev"] = {"cases": n, "missing": len(miss), "missing_top15": hit, "missing_rate": round(hit / len(miss), 3) if miss else None, "gold_old": round(old_gold, 3), "gold_union": round(new_gold, 3),
              "random_old": round(old_r, 3), "random_union": round(new_r, 3), "lift_old": round(old_gold / old_r, 2), "lift_union": round(new_gold / new_r, 2),
              "by_label": {l[0]: [sum(c["label"] == l[0] for c in miss), sum(c["label"] == l[0] and c["top15"] for c in miss)] for l in labels if l[0] != "랩"}}
d = res["dev"]
w0 = res["control_median"] is not None and 0.45 <= res["control_median"] <= 0.55
res["gates"] = {"W0_control": w0, "W1_within_auc": bool(res["median_within"] and res["median_within"] >= 0.70), "W2_high_note": ("칸 부족" if high is None else bool(high >= 0.70)),
                "W3_missing": ("표본 부족" if len(miss) < 10 else bool(d["missing_rate"] >= 0.30)), "W4_discrimination": bool(new_gold - old_gold >= 0.10 and d["lift_union"] >= d["lift_old"])}
(HERE / "clap_within_genre.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("가수 수", res["groups"], "칸", res["cells"], "장르 안 AUC 중앙", res["median_within"], "대조군 중앙", res["control_median"], "범위", res["control_range"])
for k, r in res["labels"].items():
    print(k, "태그곡", r["tag_songs"], "곡분할", r["overall_auc_song_split"], "가수분할", r["overall_auc_artist_split"], "장르 안", r["within_genre"])
print({k: v for k, v in d.items() if k != "by_label"}); print(d["by_label"]); print(res["gates"])
