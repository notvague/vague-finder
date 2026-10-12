"""
표지(이미지) 질의 진단 — 정답 표지가 어디서 빠지는지 근거로 나눈다. **순위 계산은 바꾸지 않는다.**

대상: docs/eval/queries.json에서 modality_focus가 image·multimodal이고 정답이 있는 질의(dev·test).
정답 곡마다 세 가지를 한 줄에 남긴다.

1. 데이터 연결 — 모델을 탓하기 전에 본다
   - 이미지 벡터 파일(artifacts/embeddings/image/<모델>/<곡>.npy)이 있는가, Qdrant 이미지 컬렉션의 벡터와 같은가
   - 지금 표지(links.cover_url)를 SigLIP2로 다시 임베딩하면 저장된 벡터와 같은가 — 다르면 다른 이미지로
     만든 벡터다(잘못된 표지·오래된 임베딩). 같은 모델·같은 이미지면 코사인이 1에 가깝다. 정답이 아닌 곡
     표본(--control)으로 같은 값을 재서 기준을 함께 남긴다
2. 문장 — 같은 이미지 인덱스에서 두 문장으로 정답 표지의 순위를 잰다(이미지 단독, 전수)
   - cached: 분석 캐시에 저장된 image_english_query (실제 검색이 쓰는 문장)
   - visual: 질의 원문에 적힌 시각 조건만 담은 문장(VISUAL_ONLY). **정답 표지와 순위를 보기 전에 질의
     원문만 보고 썼다** — 정답 표지에 맞춰 쓰면 대조가 성립하지 않는다
3. 단계 연결 — 이미지 경로 실행 여부 → 이미지 단독 순위 → 깊이 무제한일 때 융합·보정 순위 → 최종 후보 30
   → 리랭킹 전·후 순위와 재정렬 정책 → 최종 Top-10. 이미지 단독 순위 말고는 이미 있는 기록에서 읽는다
   (v23 `candidate_drop_*.csv`, v22 detail·explain). 검색을 다시 돌리지 않는다

**현재 인덱스와 과거 기록을 합치므로 연결부터 검증한다.** 필요한 질의·정답 행이 기록에 모두 있어야 하고,
지금 캐시의 분석이 v22 detail의 analysis_json과 내용까지 같아야 하며, 지금 인덱스로 잰 이미지 순위가 v23 추적의
이미지 경로 순위와, v23의 후보 순위가 v22 실행 기록의 리랭킹 전 순위와 곡마다 같아야 한다. v22 측정의 코퍼스 곡 수·
BM25 지문도 지금과 같아야 한다. 하나라도 다르면 멈춘다.
순위가 없는 칸은 빈칸이 아니라 `out`(후보·Top-10 밖)으로 적는다. 쓴 파일·캐시·인덱스의 지문은
`image_diag_runinfo.json`에 남긴다.

  python -m src.retrieval.diagnose_image_queries \\
      --output-dir experiments/reranking/results_v25_cover_diag

곡 ID·순위·점수와 질의에서 나온 문장만 남긴다. 표지 이미지·표지 주소·곡 제목은 남기지 않는다 — 같은 표지를
한 번만 세는 데 쓰는 열(`cover_hash`)도 주소가 아니라 주소의 해시다(docs/data_policy.md: 곡 메타 파생물 비공개).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.eval.schema import V05_QUERY_SETS
from src.embedding.image.image_io import fetch_image
from src.embedding.models.image_siglip2 import SigLIP2Embedder
from src.retrieval.analysis_cache import load_cache
from src.vector_db.qdrant_backend import QdrantVectorClient, collection_name, point_id
from src.vector_db.settings import IMAGE_INDEX_NAME, NAMESPACE

QUERIES = Path("docs/eval/queries.json")
CORPUS = Path("data/all_songs.jsonl")
IMAGE_DIR = Path("artifacts/embeddings/image/google__siglip2-base-patch16-224")
CACHES = {s: Path(f"experiments/reranking/analysis_cache_v06_{s}.json") for s in ("dev", "test")}
TRACE = {s: Path(f"experiments/reranking/results_v23_path_depth/candidate_drop_{s}.csv") for s in ("dev", "test")}
V22_DIR = Path("experiments/reranking/results_v22_corpus3010")
V22 = {s: V22_DIR / f"search_eval_{s}_detail.csv" for s in ("dev", "test")}
V22_EXPLAIN = {s: V22_DIR / f"search_eval_{s}_explain.jsonl" for s in ("dev", "test")}
V22_RUNINFO = {s: V22_DIR / f"search_eval_{s}_runinfo.json" for s in ("dev", "test")}
BM25 = Path("artifacts/bm25_params.json")
OUT = "out"   # 후보·Top-10 밖 — 기록이 없어서 비운 칸과 구분한다

# 질의 원문에 적힌 시각 조건만 옮긴 문장. 음색·장르·곡 분위기 같은 비시각 설명은 뺐다.
# 2026-10-01, 정답 표지와 순위를 보기 전에 질의 원문만 보고 작성했다. 고치지 않는다.
VISUAL_ONLY: Dict[str, str] = {
    "q116": "an album cover in shades of blue with no human face",
    "q117": "an album cover with only large text and no photo, very simple",
    "q204": "an illustration on a white background of five women each wearing a different animal mask",
    "q205": "an album cover image of a huge pile of colorful books",
    "q300": "an almost solid pink album cover with a black triangle like a lifted corner of paper at one lower "
            "corner and a small white label-like rectangle at the bottom",
    "q304": "an album cover with a green background and four large white geometric shapes, angular lettering, "
            "very simple",
    "q307": "a gray album cover divided diagonally, with four tiny men scattered and standing, cold and minimal",
    "q309": "a sky blue background with a long curved pink cursive title, almost no photo of people, playful "
            "toy-like lettering",
    "q312": "a large hand-drawn flower in black lines on a rough, calm paper-textured background, with a "
            "handwritten title",
    "q315": "a black and white photo of a woman's face on a light pinkish background with overlapping flower "
            "shadows or petals, and one large uppercase English letter on the left",
    "m301": "an all red album cover",
    "m302": "an album cover with a large close-up of a person's face",
    "m303": "a black and white photo album cover with almost no text",
    "m401": "a bright pastel-toned album cover",
    "m402": "a dark and cold album cover",
    "m403": "a colorful, busy and noisy album cover",
}


def cover_key(url: str) -> str:
    """같은 표지를 한 번만 세기 위한 키 — 멜론 앨범 이미지 경로(크기·최적화 꼬리 제외). 기록에는 남기지 않는다."""
    base = (url or "").split("?")[0].split("/melon/")[0]
    return re.sub(r"_\d+\.jpg$", ".jpg", base)


def cover_hash(url: str) -> str:
    """기록에 남기는 표지 키. 주소 대신 해시라 공개 저장소에 표지 주소가 남지 않는다."""
    return hashlib.sha256(cover_key(url).encode("utf-8")).hexdigest()[:12] if url else ""


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def read_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "없음"


def explain_targets(path: Path) -> Dict[tuple, dict]:
    """v22 실행 기록에서 정답 곡의 리랭킹 전·후 순위와 재정렬 정책."""
    out: Dict[tuple, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            for song in o["songs"]:
                if not song.get("relevant"):
                    continue
                e = song.get("explain") or {}
                out[(o["query_id"], song["song_id"])] = {
                    "rank_before_rerank": e.get("rank_before_rerank") or OUT,
                    "rank_after_rerank": e.get("rank_after_rerank") or OUT,
                    "rerank_strategy": (e.get("score_mix") or {}).get("strategy", ""),
                }
    return out


def check_links(queries: List[dict], trace: Dict[tuple, dict], detail: Dict[str, dict],
                explains: Dict[tuple, dict], points: int, analyses: Optional[Dict] = None) -> List[str]:
    """과거 기록이 지금 인덱스·입력과 같은 조건인지. 문제를 돌려준다(빈 목록이면 통과).

    - 분석은 **내용**으로 비교한다 — 지금 캐시의 분석과 v22 detail의 analysis_json이 질의마다 같아야 한다.
      같은 경로의 캐시를 다시 만들면 이미지 문장은 같아도 성별·가중치가 달라질 수 있다
    - 후보 순위는 **곡별로** 비교한다 — v23 추적의 후보 순위와 v22 실행 기록의 리랭킹 전 순위.
      질의의 최솟값(v22 candidate_rank@30) 비교는 추가로 한다
    analyses를 주지 않으면 분석 비교를 건너뛴다(테스트용).
    """
    problems: List[str] = []
    for q in queries:
        qid = q["query_id"]
        if qid not in detail:
            problems.append(f"{qid}: v22 detail에 없다")
            continue
        if analyses is not None:
            past = json.loads(detail[qid]["analysis_json"])
            if json.loads(analyses[qid].model_dump_json()) != past:
                problems.append(f"{qid}: 분석 캐시 내용이 v22 측정 때와 다르다")
        for sid in q["positives"]:
            if (qid, sid) not in trace:
                problems.append(f"{qid}/{sid}: v23 추적에 없다")
            if (qid, sid) not in explains:
                problems.append(f"{qid}/{sid}: v22 실행 기록에 없다")
            if (qid, sid) in trace and (qid, sid) in explains:
                pool = str(trace[(qid, sid)]["pool_rank"] or OUT)
                before = str(explains[(qid, sid)].get("rank_before_rerank") or OUT)
                if pool != before:
                    problems.append(f"{qid}/{sid}: v23 후보 순위 {pool} ≠ v22 리랭킹 전 순위 {before}")
        pools = [int(trace[(qid, s)]["pool_rank"]) for s in q["positives"]
                 if (qid, s) in trace and trace[(qid, s)]["pool_rank"]]
        want = str(min(pools)) if pools else ""
        if want != detail[qid]["candidate_rank@30"]:
            problems.append(f"{qid}: v23 후보 순위와 v22 candidate_rank@30이 다르다")
    for split, path in V22_RUNINFO.items():
        info = json.load(open(path, encoding="utf-8"))
        if Path(info["analysis"]["path"]) != CACHES[split]:
            problems.append(f"v22 {split}: 분석 캐시가 다르다 ({info['analysis']['path']})")
        if info["corpus"].get("point_count") not in (None, points):
            problems.append(f"v22 {split}: 코퍼스 곡 수 {info['corpus']['point_count']} ≠ 지금 {points}")
        if info["bm25"].get("sha256_12") != sha12(BM25):
            problems.append(f"v22 {split}: BM25 지문이 다르다")
    return problems


class Index:
    """이미지 컬렉션 한 개 — 전수 순위와 저장 벡터."""

    def __init__(self) -> None:
        self.client = QdrantVectorClient()
        self.collection = collection_name(IMAGE_INDEX_NAME, NAMESPACE)
        self.index = self.client.Index(IMAGE_INDEX_NAME)
        self.size = self.client.client.get_collection(self.collection).points_count

    def ranking(self, vector: np.ndarray) -> Dict[str, tuple]:
        res = self.index.query(vector=vector.tolist(), top_k=self.size, include_metadata=False)
        return {m["id"]: (rank, m["score"]) for rank, m in enumerate(res["matches"], start=1)}

    def stored(self, song_id: str) -> Optional[np.ndarray]:
        recs = self.client.client.retrieve(self.collection, ids=[point_id(song_id)], with_vectors=True)
        if not recs:
            return None
        vec = recs[0].vector
        vec = vec.get("dense") if isinstance(vec, dict) else vec
        return np.asarray(vec, dtype=np.float32) if vec is not None else None


def fresh_cosine(embedder: SigLIP2Embedder, url: str, stored: Optional[np.ndarray]) -> tuple:
    if not url or stored is None:
        return "", "표지 URL 또는 저장 벡터 없음"
    try:
        vec = embedder.embed_images([fetch_image(url)], l2_normalize=True)[0]
    except Exception as exc:  # noqa: BLE001 - 내려받기 실패는 기록하고 넘어간다
        return "", f"내려받기 실패: {type(exc).__name__}"
    return round(cosine(np.asarray(vec), stored), 6), ""


def main() -> None:
    p = argparse.ArgumentParser(description="표지(이미지) 질의 진단 — 순위 계산은 바꾸지 않는다")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--control", type=int, default=30, help="정답이 아닌 곡 표본 수 (새 임베딩 코사인 기준)")
    p.add_argument("--seed", type=int, default=20261001)
    args = p.parse_args()

    # v0.5 세트만 — 이 진단은 v22·v23 기록과 대조하므로 그 기록에 없는 v09(2026-10) 표지 질의는
    # 대상이 아니다. v09를 보려면 VISUAL_ONLY 문장을 쓰고 기록을 새로 만든 뒤 여기를 넓힌다.
    queries = [
        q for q in json.load(open(QUERIES, encoding="utf-8"))["queries"]
        if q.get("modality_focus") in ("image", "multimodal") and q.get("positives")
        and q.get("label_status") == "labeled" and q.get("query_set") in V05_QUERY_SETS
    ]
    missing = sorted(q["query_id"] for q in queries if q["query_id"] not in VISUAL_ONLY)
    if missing:
        raise SystemExit(f"시각 조건 문장이 없는 질의: {missing}")
    caches = {s: load_cache(path) for s, path in CACHES.items()}
    analyses = {q["query_id"]: caches[q["split"]].get(q["query_id"], q["query"]) for q in queries}

    songs = {}
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                songs[str(r["song_id"])] = (r.get("links") or {}).get("cover_url", "")
    trace = {(r["query_id"], r["song_id"]): r for s in TRACE for r in read_csv(TRACE[s])}
    detail = {r["query_id"]: r for s in V22 for r in read_csv(V22[s])}
    explains = {k: v for s in V22_EXPLAIN for k, v in explain_targets(V22_EXPLAIN[s]).items()}

    index = Index()
    # 모델을 올리기 전에 과거 기록과의 연결부터 본다
    problems = check_links(queries, trace, detail, explains, index.size, analyses)
    if problems:
        raise SystemExit("과거 기록과 연결할 수 없다:\n  " + "\n  ".join(problems[:20]))
    embedder = SigLIP2Embedder().load()

    rows: List[dict] = []
    sentences: List[dict] = []
    for q in queries:
        qid, a = q["query_id"], analyses[q["query_id"]]
        use_image = bool(a.has_visual_clue and a.image_english_query.strip() and a.modality_weights.image > 0)
        texts = {"cached": a.image_english_query, "visual": VISUAL_ONLY[qid]}
        rankings = {}
        for name, text in texts.items():
            vec = np.asarray(embedder.embed_texts([text], l2_normalize=True)[0])
            rankings[name] = index.ranking(vec)
        sentences.append({
            "split": q["split"], "query_id": qid, "modality_focus": q["modality_focus"],
            "has_visual_clue": int(bool(a.has_visual_clue)), "image_weight": round(a.modality_weights.image, 4),
            "image_path_active": int(use_image),
            "cached_sentence": texts["cached"], "visual_sentence": texts["visual"],
            "top1_score_cached": round(min(rankings["cached"].values())[1], 6),
            "top1_score_visual": round(min(rankings["visual"].values())[1], 6),
        })
        top10 = detail[qid]["rerank_top_ids"].split("|")
        for sid in q["positives"]:
            stored = index.stored(sid)
            npy = IMAGE_DIR / f"{sid}.npy"
            file_vec = np.load(npy).astype(np.float32).reshape(-1) if npy.exists() else None
            fresh, note = fresh_cosine(embedder, songs.get(sid, ""), stored)
            t, e = trace[(qid, sid)], explains[(qid, sid)]
            c_rank, c_score = rankings["cached"].get(sid, (OUT, ""))
            v_rank, v_score = rankings["visual"].get(sid, (OUT, ""))
            if use_image and str(c_rank) != t["image_rank"]:
                problems.append(f"{qid}/{sid}: 지금 인덱스의 이미지 순위 {c_rank} ≠ v23 추적 {t['image_rank']}")
            rows.append({
                "split": q["split"], "query_id": qid, "modality_focus": q["modality_focus"], "song_id": sid,
                "cover_hash": cover_hash(songs.get(sid, "")),
                # 1. 데이터 연결 — 지금 표지 URL의 이미지와 저장 벡터의 일관성 (URL이 맞는 표지인지는 아니다)
                "vector_file": int(file_vec is not None),
                "in_collection": int(stored is not None),
                "cos_file_vs_collection": (
                    round(cosine(file_vec, stored), 6) if file_vec is not None and stored is not None else ""
                ),
                "cos_fresh_vs_collection": fresh,
                "fetch_note": note,
                # 2. 문장 (이미지 단독, 전수)
                "rank_cached": c_rank, "score_cached": round(c_score, 6) if c_score != "" else "",
                "rank_visual": v_rank, "score_visual": round(v_score, 6) if v_score != "" else "",
                # 3. 단계 연결 (기존 기록)
                "image_path_active": int(use_image),
                "image_rank_v23": t["image_rank"] or OUT,
                "fused_rank_deep": t["fused_rank_deep"] or OUT,
                "boosted_rank_deep": t["boosted_rank_deep"] or OUT,
                "pool_rank_30": t["pool_rank"] or OUT,
                "stage_30": t["stage"],
                "rank_before_rerank": e["rank_before_rerank"],
                "rank_after_rerank": e["rank_after_rerank"],
                "rerank_strategy": e["rerank_strategy"],
                # 리랭커가 이 곡을 채점했나. 앞 20개만 채점하므로 후보 21~30위는 채점 자체가 없다 —
                # '저신뢰 상위 5개만 재정렬'과는 다른 제한이다
                "ce_scope": (
                    "채점" if e["rerank_strategy"]
                    else "채점 안 함(앞 20개만 채점)" if t["pool_rank"] and int(t["pool_rank"]) > 20
                    else "채점 기록 없음" if t["pool_rank"] else "후보 밖"
                ),
                "final_rank_v22": top10.index(sid) + 1 if sid in top10 else OUT,
            })
        print(f"{qid}: 정답 {len(q['positives'])}곡 · 이미지 경로 {'켜짐' if use_image else '꺼짐'}", flush=True)
    if problems:
        raise SystemExit("지금 인덱스와 v23 추적이 다르다:\n  " + "\n  ".join(problems[:20]))

    # 정답이 아닌 곡 표본 — 새 임베딩과 저장 벡터의 코사인이 보통 얼마인지
    targets = {r["song_id"] for r in rows}
    pool = sorted(s for s in songs if s not in targets)
    control: List[dict] = []
    for sid in random.Random(args.seed).sample(pool, min(args.control, len(pool))):
        fresh, note = fresh_cosine(embedder, songs.get(sid, ""), index.stored(sid))
        control.append({"song_id": sid, "cos_fresh_vs_collection": fresh, "fetch_note": note})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("image_diag_targets.csv", rows), ("image_diag_sentences.csv", sentences),
                       ("image_diag_control.csv", control)):
        with open(args.output_dir / name, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    run_info = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "image_model": IMAGE_DIR.name,
        "image_collection": {"name": index.collection, "points": index.size},
        "control": {"n": args.control, "seed": args.seed},
        "files_sha256_12": {
            str(path): sha12(path)
            for path in [QUERIES, CORPUS, BM25, *CACHES.values(), *TRACE.values(), *V22.values(),
                         *V22_EXPLAIN.values(), *V22_RUNINFO.values()]
        },
        "analysis_cache_meta": {s: caches[s].meta for s in caches},
        "links_checked": "질의·정답 행 존재, 분석 내용 = v22 analysis_json, 이미지 순위 = v23 추적, "
                         "곡별 v23 후보 순위 = v22 리랭킹 전 순위, 질의 최솟값 = v22 candidate_rank@30, "
                         "v22 분석 캐시 경로·곡 수·BM25 = 지금",
    }
    (args.output_dir / "image_diag_runinfo.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    vals = [c["cos_fresh_vs_collection"] for c in control if c["cos_fresh_vs_collection"] != ""]
    print(f"\n질의 {len(queries)} · 정답 {len(rows)}곡 · 고유 표지 {len({r['cover_hash'] for r in rows})}")
    if vals:
        print(f"표본 {len(vals)}곡 새 임베딩 코사인: 최소 {min(vals):.4f} · 중앙값 {sorted(vals)[len(vals) // 2]:.4f}")
    print(f"기록: {args.output_dir}")


if __name__ == "__main__":
    main()
