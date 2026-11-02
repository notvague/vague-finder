"""세트 하나를 공식 측정 도구(`evaluate_search_accuracy`)로 잰다. 실행 조건은 run_ab.zsh가 환경 변수로 고정한다.

`--ledger`를 주면 v75 비용 장치를 SDK HTTP 시도 단위로 건다(실제 Vertex 실행). 안 주면 장치 없이 돈다(리랭킹 끔 실행 전용).
"""
import argparse, os, sys
from pathlib import Path

WT = Path(os.environ["V79_WORKTREE"])
sys.path.insert(0, str(WT))
p = argparse.ArgumentParser()
p.add_argument("--set", required=True, choices=["v06", "v09", "v10"])
p.add_argument("--out", required=True)
p.add_argument("--run-tag", default="")
p.add_argument("--ledger", default="")
a = p.parse_args()
if a.ledger:
    sys.path.insert(0, str(WT / "experiments/reranking/results_v75_work_audio_live_ab"))
    import v75_budget
    v75_budget.install(Path(a.ledger), cost_cap_usd=float(os.environ["V79_COST_CAP"]), attempt_cap=int(os.environ["V79_ATTEMPT_CAP"]),
                       deadline_window_seconds=float(os.environ["V79_WINDOW"]), run_tag=a.run_tag, model="gemini-3.1-flash-lite")
import src  # noqa: E402
assert str(WT) in str(Path(src.__file__).resolve()), f"워크트리 코드가 아니다: {src.__file__}"
from src.retrieval import evaluate_search_accuracy as ev  # noqa: E402
R = WT / "experiments/reranking"
sys.path.insert(0, str(R / "results_v63_hybrid_lyrics_validation"))
from snapshot_lyrics import content_digest  # noqa: E402
import gzip, hashlib, json, pickle  # noqa: E402
from src.retrieval import lyrics_exact_search as les  # noqa: E402
# 가사는 v75와 같은 고정 스냅샷으로 준다(라이브 DB 변동을 측정에 섞지 않는다). 지문이 다르면 호출 전에 멈춘다.
SNAP = Path(os.environ["V79_LYRICS_SNAPSHOT"])
manifest = json.loads((SNAP / "manifest.json").read_text(encoding="utf-8"))
blob = (SNAP / manifest["file"]["name"]).read_bytes()
assert hashlib.sha256(blob).hexdigest()[:16] == manifest["file"]["sha256_16"], "가사 스냅샷 파일 sha 불일치"
documents = pickle.loads(gzip.decompress(blob))
digest = content_digest([(str(d.get("song_id") or ""), str((d.get("lyrics_data") or {}).get("full_lyrics") or "")) for d in documents])
assert digest["content_sha256_16"] == "e0d58f28f1f4a563", f"가사 지문 불일치: {digest}"
les.LyricsExactSearchService._source_documents = lambda self: iter(list(documents))
fingerprint = {"db": manifest.get("db_name"), "collection": "songs", **digest, "source": "v63_snapshot"}
ev._lyrics_fingerprint = lambda: dict(fingerprint)
sys.argv = ["evaluate_search_accuracy", "--input", str(R / f"eval_queries_{a.set}.csv"), "--split", "dev",
            "--analysis-cache", str(R / f"results_v49_vertex_baseline/analysis_cache_{a.set}_dev_vertex.json"),
            "--output-dir", a.out]
import asyncio  # noqa: E402
asyncio.run(ev.evaluate(ev.build_parser().parse_args()))
if a.ledger and v75_budget.GUARD is not None and v75_budget.GUARD.halted:
    print(f"BUDGET_HALT {v75_budget.GUARD.halt_reason}", file=sys.stderr)
    sys.exit(90)
