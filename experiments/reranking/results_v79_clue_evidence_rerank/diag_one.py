"""질의 하나의 리랭킹 응답 원문을 본다(진단 전용). 비용 장부를 함께 쓴다."""
import csv, json, os, sys
from pathlib import Path
WT = Path(os.environ["V79_WORKTREE"]); R = WT / "experiments/reranking"
sys.path.insert(0, str(WT)); sys.path.insert(0, str(R / "results_v75_work_audio_live_ab")); sys.path.insert(0, str(R / "results_v63_hybrid_lyrics_validation"))
qid, setname, ledger, out = sys.argv[1:5]
import v75_budget
v75_budget.install(Path(ledger), cost_cap_usd=float(os.environ["V79_COST_CAP"]), attempt_cap=int(os.environ["V79_ATTEMPT_CAP"]),
                   deadline_window_seconds=float(os.environ["V79_WINDOW"]), run_tag=f"diag_{qid}", model="gemini-3.1-flash-lite")
import gzip, hashlib, pickle, asyncio
from snapshot_lyrics import content_digest
from src.retrieval import lyrics_exact_search as les, evaluate_search_accuracy as ev, gemini_listwise_reranker as g
SNAP = Path(os.environ["V79_LYRICS_SNAPSHOT"]); m = json.loads((SNAP / "manifest.json").read_text(encoding="utf-8"))
docs = pickle.loads(gzip.decompress((SNAP / m["file"]["name"]).read_bytes()))
les.LyricsExactSearchService._source_documents = lambda self: iter(list(docs))
ev._lyrics_fingerprint = lambda: {"source": "v63_snapshot"}
dump = {"prompts": [], "responses": []}
orig_gen = g.GeminiListwiseReranker._generate
def gen(self, *a, **k):
    dump["prompts"].append(k.get("contents") or (a[0] if a else ""))
    resp = orig_gen(self, *a, **k)
    dump["responses"].append(getattr(resp, "text", "") or "")
    return resp
g.GeminiListwiseReranker._generate = gen
src = R / f"eval_queries_{setname}.csv"
rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
sub = Path(out) / f"{qid}.csv"; sub.parent.mkdir(parents=True, exist_ok=True)
with sub.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows([r for r in rows if r["query_id"] == qid])
sys.argv = ["e", "--input", str(sub), "--split", "dev", "--analysis-cache", str(R / f"results_v49_vertex_baseline/analysis_cache_{setname}_dev_vertex.json"), "--output-dir", str(Path(out) / qid)]
asyncio.run(ev.evaluate(ev.build_parser().parse_args()))
Path(out, f"{qid}_dump.json").write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8")
