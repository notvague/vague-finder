"""v58 단위 재현 — 수정 전(`origin/main`의 lyrics_exact_search.py)과 수정 후를 **같은 느린 가짜 자료원**으로 비교한다.

실제 Mongo·모델 없음. 자료원은 호출마다 `SLOW_SECONDS`(기본 3초) 동안 읽기 안에서 멈춘 뒤 합성 문서를 낸다
(v56의 350초짜리 Atlas 전체 읽기를 축소한 것). 재는 것:
  A. TTL 만료 뒤 첫 가사 검색의 소요 시간 — 수정 전은 재적재를 기다리고(≈SLOW), 수정 후는 지금 판으로 바로 답한다
  B. 그 사이 들어온 두 번째 검색의 소요 시간 — 수정 전은 `_load_lock`을 기다린다
  C. 만료 뒤 검색이 쓴 판으로 인용했을 때 판 해시 — 수정 전은 재적재 뒤 새 판, 수정 후는 옛 판(검색이 본 판)이고 갱신 뒤 새 요청은 새 판
  D. 자료 같을 때 결과·순위·판 해시가 전후 동일

실행: venv/bin/python experiments/reranking/results_v58_lyrics_cache_refresh/repro_cache_refresh.py [--slow 3] [--out repro_before_after.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src.backend.schemas.query import LyricClue  # noqa: E402
import v58_runtime  # noqa: E402

CLUE = LyricClue(text=v58_runtime.CLUE_TEXT, kind="verbatim", confidence=0.95)


def load_old_module(ref: str):
    text = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{ref}:src/retrieval/lyrics_exact_search.py"], text=True)
    spec = importlib.util.spec_from_loader("lyrics_exact_search_old", loader=None)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    module.__dict__["__file__"] = f"<git:{ref}>"
    sys.modules["lyrics_exact_search_old"] = module  # dataclass가 cls.__module__을 sys.modules에서 찾는다
    exec(compile(text, module.__dict__["__file__"], "exec"), module.__dict__)
    return module


class SlowSource:
    """읽기 안에서 `slow`초 멈춘 뒤 문서를 낸다. 호출 수·판 표식(version)을 기록한다."""

    def __init__(self, slow: float) -> None:
        self.slow = slow
        self.calls = 0
        self.version = "v1"
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.calls += 1
        version = self.version

        def generate():
            docs = v58_runtime.synthetic_documents(120, version=version)
            yield docs[0]
            time.sleep(self.slow)       # Mongo의 긴 getMore를 흉내 — 첫 문서 뒤에 멈춘다
            yield from docs[1:]

        return generate()


def observe(service, label: str) -> Dict[str, Any]:
    used: List[Any] = []
    started = time.perf_counter()
    tracks = service.search([CLUE], snapshot_out=used)
    elapsed = time.perf_counter() - started
    quote = service.excerpt(tracks[0].id, tracks[0].lyric_match_detail.matched_phrase, used[0]) if tracks else None
    return {
        "label": label,
        "elapsed_s": round(elapsed, 3),
        "ids": [t.id for t in tracks],
        "match_types": [t.lyric_match_type for t in tracks],
        "snapshot_songs": len(used[0].documents),
        "snapshot_version_marker": used[0].get(v58_runtime.TARGET_ID).full_lyrics.strip().split("\n")[-1],
        "quote_version": None if quote is None else quote.data_version,
        "quote_span": None if quote is None else [quote.char_start, quote.char_end],
    }


def scenario(make_service, slow: float, ttl: float) -> Dict[str, Any]:
    source = SlowSource(slow)
    service = make_service(source)
    out: Dict[str, Any] = {"ttl_seconds": ttl, "slow_seconds": slow, "steps": []}

    out["steps"].append(observe(service, "1_initial_load(blocking, expected≈slow)"))
    time.sleep(ttl + 0.2)                       # TTL 만료
    source.version = "v2"                       # 그 사이 자료가 바뀌었다

    results: Dict[str, Dict[str, Any]] = {}

    def first() -> None:
        results["first"] = observe(service, "2_first_request_after_expiry")

    def second() -> None:
        time.sleep(0.3)
        results["second"] = observe(service, "3_second_request_while_reload_in_progress")

    t1, t2 = threading.Thread(target=first), threading.Thread(target=second)
    wall = time.perf_counter()
    t1.start(); t2.start(); t1.join(); t2.join()
    out["wall_for_both_requests_s"] = round(time.perf_counter() - wall, 3)
    out["steps"] += [results["first"], results["second"]]

    # 갱신이 끝나기를 기다린다(수정 후만 백그라운드가 있다. 수정 전은 이미 끝났다)
    thread = getattr(service, "_refresh_thread", None)
    if thread is not None:
        thread.join(slow + 5)
    out["steps"].append(observe(service, "4_request_after_refresh_completed"))
    out["source_calls"] = source.calls
    status = getattr(service, "status", None)
    out["status"] = status() if callable(status) else "n/a(old code)"
    close = getattr(service, "close", None)
    if callable(close):
        close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow", type=float, default=3.0)
    parser.add_argument("--ttl", type=float, default=1.0)
    parser.add_argument("--old-ref", default="origin/main")
    parser.add_argument("--out", default=str(HERE / "repro_before_after.json"))
    args = parser.parse_args()

    old = load_old_module(args.old_ref)
    from src.retrieval import lyrics_exact_search as new

    def make_old(source):
        service = old.LyricsExactSearchService(cache_ttl_seconds=args.ttl)
        service._source_documents = source
        return service

    def make_new(source):
        return new.LyricsExactSearchService(cache_ttl_seconds=args.ttl, documents_source=source)

    report = {
        "old_ref": args.old_ref,
        "before": scenario(make_old, args.slow, args.ttl),
        "after": scenario(make_new, args.slow, args.ttl),
    }
    same_data = (
        [(s["ids"], s["match_types"], s["quote_span"]) for s in report["before"]["steps"]]
        == [(s["ids"], s["match_types"], s["quote_span"]) for s in report["after"]["steps"]]
    )
    report["same_ids_types_spans_before_vs_after"] = same_data
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("before", "after"):
        print(f"== {key}")
        for step in report[key]["steps"]:
            print(f"  {step['label']:<48} {step['elapsed_s']:>7.3f}s  판={step['snapshot_version_marker']} quote={step['quote_version']}")
        print(f"  source_calls={report[key]['source_calls']} wall(both)={report[key]['wall_for_both_requests_s']}s")
    print(f"same ids/types/spans before vs after: {same_data}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
