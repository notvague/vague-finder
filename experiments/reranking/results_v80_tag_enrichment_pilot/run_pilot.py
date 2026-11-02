"""파일럿 태그 생성. 비용 장치를 건 실제 Vertex 호출. 이미 끝낸 곡은 건너뛴다."""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "experiments/reranking/results_v75_work_audio_live_ab")]
import v75_budget
MODEL = "gemini-3.5-flash-lite"
v75_budget.PRICES[MODEL] = {"input": 0.30, "output": 2.50}   # 공식 표 Standard·Global, 2026-11-03
v75_budget.install(HERE / "budget/ledger.jsonl", cost_cap_usd=1.0, attempt_cap=120, deadline_window_seconds=1800, run_tag="pilot", model=MODEL)
from src.common.gemini_client import make_genai_client
from pilot_lib import build_prompt, choose_sample, clean_tags, failed_query_gold, load_songs
songs = load_songs(Path(sys.argv[1]))
gold = failed_query_gold(ROOT / "experiments/reranking/results_v79_clue_evidence_rerank/per_query.csv", ROOT / "experiments/reranking")
sample = choose_sample(songs, [x for ids in gold.values() for x in ids])
assert len(sample["F"]) == 25 and len(sample["R"]) == 25, {k: len(v) for k, v in sample.items()}
out = HERE / "runs"; out.mkdir(exist_ok=True)
(out / "sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
done = {}
path = out / "new_tags.jsonl"
if path.exists():
    done = {json.loads(l)["song_id"]: 1 for l in path.open(encoding="utf-8") if l.strip()}
client = make_genai_client()
for group, ids in sample.items():
    for song_id in ids:
        if song_id in done:
            continue
        rec = {"song_id": song_id, "group": group, "status": "failed", "attempts": 0, "error": ""}
        start = time.time()
        for attempt in range(3):
            rec["attempts"] = attempt + 1
            try:
                resp = client.models.generate_content(model=MODEL, contents=build_prompt(songs[song_id]), config={"response_mime_type": "application/json"})
                rec["tags"], rec["dropped"] = clean_tags(json.loads(resp.text))
                rec["status"], rec["error"] = "ok", ""
                break
            except v75_budget.BudgetHalt as exc:
                rec["error"] = f"BudgetHalt: {exc}"
                print("BUDGET_HALT", exc, flush=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sys.exit(90)
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                time.sleep(2)
        rec["seconds"] = round(time.time() - start, 2)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(group, song_id, rec["status"], rec["attempts"], rec["seconds"], flush=True)
print("DONE", flush=True)
