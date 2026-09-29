"""검색 지연시간 측정 — 실제 API로 요청을 보내고 서버 구간 기록과 맞춘다.

**왜 평가 스크립트로 재지 않는가.** `evaluate_search_accuracy`의 `retrieval_ms`는
라우터만 직접 부른 시간이다. 질의 분석은 캐시에서 읽고, CE 모델은 타이머 전에
`load()`로 올리고, 응답 직렬화와 HTTP 왕복은 애초에 범위 밖이다. 정확도를 재기엔
맞는 설계지만 **서비스 지연**은 그 바깥이 대부분일 수 있다. 여기서는 실제로
사용자가 기다리는 경로를 그대로 지난다.

**두 시계를 함께 남긴다.** 클라이언트에서 잰 벽시계 시간과 서버가 남긴 구간 기록을
`X-Request-Id`로 맞춘다. 둘의 차이가 네트워크와 프레임워크 바깥이다.

**합치지 않는다.** 병렬 경로 시간을 더해서 "전체"를 만들면 실제보다 커진다. 경로는
가장 느린 것 하나와 그 대기 시간만 본다.

**평균을 쓰지 않는다.** 발표를 망치는 건 꼬리다. 중앙값·p95·최댓값을 함께 본다.

준비:
    SEARCH_TIMING_LOG=artifacts/timing/dev.jsonl \\
        venv/bin/uvicorn src.backend.main:app

실행:
    venv/bin/python -m src.retrieval.measure_search_latency \\
        --split dev --timing-log artifacts/timing/dev.jsonl \\
        --output-dir experiments/latency/run_v01
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_INPUT = Path("experiments/reranking/eval_queries_v06.csv")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SEARCH_PATH = "/api/v1/search"


# ----------------------------------------------------------------------
# 질의 읽기 — 평가 스크립트와 같은 파일, 같은 규칙
# ----------------------------------------------------------------------


def read_queries(path: Path, split: Optional[str]) -> List[Dict[str, str]]:
    """`query_type=search` 행만, 요청한 split만.

    평가 스크립트의 `read_queries`와 같은 규칙이지만 여기서는 다시 구현한다.
    그쪽 모듈을 부르면 torch·모델 로더까지 딸려 와서, 서버만 띄워 두고 돌리는
    이 클라이언트가 GPU 없는 자리에서 뜨지 않는다.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"query_id", "split", "query_type", "query"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"평가 CSV에 필요한 열이 없습니다: {sorted(missing)}")

    selected = [
        row
        for row in rows
        if row["query_type"].strip().lower() == "search"
        and (not split or row["split"].strip().lower() == split.lower())
    ]
    if not selected:
        raise SystemExit("측정할 search 질의가 없습니다. --split을 확인하세요.")
    return selected


# ----------------------------------------------------------------------
# 요청 한 번
# ----------------------------------------------------------------------


def post_search(
    session: requests.Session,
    base_url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    """한 번 보내고 **클라이언트 시계**로 잰다.

    `perf_counter`는 요청을 만들기 직전부터 본문을 다 읽은 뒤까지다. 서버가
    남기는 시간에는 본문 전송과 네트워크가 빠져 있으므로 둘 다 필요하다.
    """
    started = time.perf_counter()
    error = ""
    status = 0
    body: Dict[str, Any] = {}
    request_id = ""
    try:
        response = session.post(
            base_url.rstrip("/") + SEARCH_PATH, json=payload, timeout=timeout
        )
        status = response.status_code
        request_id = response.headers.get("X-Request-Id", "")
        try:
            body = response.json()
        except ValueError:
            body = {}
    except requests.RequestException as exc:
        error = f"{type(exc).__name__}: {exc}"
    client_ms = (time.perf_counter() - started) * 1000.0

    return {
        "client_ms": client_ms,
        "status": status,
        "request_id": request_id,
        "error": error,
        "body": body,
    }


# ----------------------------------------------------------------------
# 서버 기록과 맞추기
# ----------------------------------------------------------------------


def load_timing_log(path: Path) -> Dict[str, Dict[str, Any]]:
    """요청 id → 구간 기록. 측정 전 줄이 섞여 있어도 id로 고른다."""
    if not path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("request_id"):
                records[record["request_id"]] = record
    return records


def span_run_ms(record: Dict[str, Any], name: str) -> Optional[float]:
    for span in record.get("spans", []):
        if span["name"] == name:
            return float(span["run_ms"])
    return None


def flatten(record: Dict[str, Any]) -> Dict[str, Any]:
    """구간 기록 한 줄을 표의 한 행으로.

    경로 시간은 **더하지 않는다.** 가장 느린 경로 하나와 그 대기만 남긴다 —
    합계는 겹쳐 도는 시간을 중복으로 세므로 실제 경과보다 커진다.
    """
    spans = record.get("spans", [])
    paths = [s for s in spans if s["name"].startswith("path.")]
    attempts = [s for s in spans if s["name"] == "analysis.gemini"]
    failed_attempts = [s for s in attempts if s.get("error")]

    route_ms = span_run_ms(record, "route")
    total_ms = float(record.get("total_ms", 0.0))
    slowest = max(paths, key=lambda s: s["wait_ms"] + s["run_ms"], default=None)

    row: Dict[str, Any] = {
        "server_total_ms": round(total_ms, 2),
        # 미들웨어 전체 − 라우트 = 의존성 해결 + 요청 검증 + 응답 직렬화.
        # 첫 요청에서 모델을 올리면 여기가 크게 튄다.
        "framework_ms": round(total_ms - route_ms, 2) if route_ms is not None else "",
        "route_ms": round(route_ms, 2) if route_ms is not None else "",
        "analysis_ms": _round(span_run_ms(record, "analysis")),
        "analysis_attempts": len(attempts),
        "analysis_failed_attempts": len(failed_attempts),
        "search_ms": _round(span_run_ms(record, "search")),
        "paths_ms": _round(span_run_ms(record, "search.paths")),
        "paths_submitted": _submitted(spans),
        "fuse_ms": _round(span_run_ms(record, "search.fuse")),
        "rerank_ms": _round(span_run_ms(record, "search.rerank")),
        "explain_ms": _round(span_run_ms(record, "explain")),
        "clarify_ms": _round(span_run_ms(record, "clarify")),
        "lyrics_load_ms": _round(span_run_ms(record, "lyrics.load")),
        "lyrics_scan_ms": _round(span_run_ms(record, "lyrics.scan")),
        "text_embed_ms": _round(span_run_ms(record, "text.embed_dense")),
        "text_query_ms": _round(span_run_ms(record, "text.query")),
        "slowest_path": slowest["name"][5:] if slowest else "",
        "slowest_path_wait_ms": round(slowest["wait_ms"], 2) if slowest else "",
        "slowest_path_run_ms": round(slowest["run_ms"], 2) if slowest else "",
        "path_wait_max_ms": (
            round(max(s["wait_ms"] for s in paths), 2) if paths else ""
        ),
        "path_errors": ";".join(
            f"{s['name'][5:]}:{s['error']}" for s in paths if s.get("error")
        ),
    }
    return row


def _round(value: Optional[float]) -> Any:
    return "" if value is None else round(value, 2)


def _submitted(spans: List[Dict[str, Any]]) -> Any:
    for span in spans:
        if span["name"] == "search.paths":
            return span.get("attrs", {}).get("submitted", "")
    return ""


# ----------------------------------------------------------------------
# 분포
# ----------------------------------------------------------------------


def percentile(values: List[float], q: float) -> float:
    """가장 가까운 순위값. 53건에서 보간은 없는 정밀도를 꾸며 낸다."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def summarize(rows: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    out = []
    for column in columns:
        values = [
            float(row[column])
            for row in rows
            if row.get(column) not in (None, "", "nan")
        ]
        if not values:
            continue
        out.append(
            {
                "stage": column,
                "n": len(values),
                "median_ms": round(statistics.median(values), 1),
                "p95_ms": round(percentile(values, 0.95), 1),
                "max_ms": round(max(values), 1),
                "min_ms": round(min(values), 1),
            }
        )
    return out


STAGE_COLUMNS = [
    "client_ms",
    "server_total_ms",
    "framework_ms",
    "route_ms",
    "analysis_ms",
    "search_ms",
    "paths_ms",
    "fuse_ms",
    "rerank_ms",
    "explain_ms",
    "clarify_ms",
    "slowest_path_run_ms",
    "path_wait_max_ms",
    "lyrics_load_ms",
    "lyrics_scan_ms",
    "text_embed_ms",
    "text_query_ms",
]


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------


def measure(args: argparse.Namespace) -> int:
    queries = read_queries(args.input, None if args.split == "all" else args.split)
    if args.query_ids:
        wanted = {q.strip() for q in args.query_ids.split(",") if q.strip()}
        queries = [row for row in queries if row["query_id"].strip() in wanted]
        if not queries:
            raise SystemExit(f"--query-ids와 맞는 질의가 없습니다: {sorted(wanted)}")

    timing_log = Path(args.timing_log)
    # 측정 시작 시점의 줄 수. 이 뒤에 붙은 줄만 이번 측정의 것이다.
    log_lines_before = (
        sum(1 for _ in timing_log.open("r", encoding="utf-8"))
        if timing_log.exists()
        else 0
    )

    session = requests.Session()
    plan: List[Dict[str, Any]] = []

    # 1) 최초 호출과 준비된 상태를 먼저 가른다. 첫 요청은 모델을 올리므로
    #    분포에 섞으면 최댓값이 그것 하나로 결정된다.
    probe = queries[0]
    plan.append({"row": probe, "kind": "cold", "repeat": 0})
    plan.append({"row": probe, "kind": "warm", "repeat": 0})

    # 2) 본 측정
    for row in queries:
        for repeat in range(args.repeat):
            plan.append({"row": row, "kind": "measure", "repeat": repeat})

    results: List[Dict[str, Any]] = []
    for index, item in enumerate(plan, start=1):
        row = item["row"]
        payload = {
            "query": row["query"].strip(),
            "top_k": args.top_k,
            "explain": args.explain,
        }
        print(
            f"[{index}/{len(plan)}] {item['kind']} {row['query_id']}: "
            f"{row['query'].strip()[:40]}",
            flush=True,
        )
        outcome = post_search(session, args.base_url, payload, args.timeout)
        record = {
            "query_id": row["query_id"].strip(),
            "kind": item["kind"],
            "repeat": item["repeat"],
            "turn": 1,
            "query": row["query"].strip(),
            "client_ms": round(outcome["client_ms"], 2),
            "status": outcome["status"],
            "request_id": outcome["request_id"],
            "error": outcome["error"],
        }
        results.append(record)
        print(
            f"    {outcome['client_ms']:.0f}ms status={outcome['status']}"
            + (f" error={outcome['error']}" if outcome["error"] else ""),
            flush=True,
        )

        # 3) 재질문은 분석을 건너뛰는 **다른 경로**다. 1턴 시간에 섞지 않는다.
        if args.clarify and item["kind"] == "measure" and outcome["status"] == 200:
            body = outcome["body"]
            shown = [track["id"] for track in body.get("results", [])]
            follow = {
                "query": row["query"].strip(),
                "top_k": args.top_k,
                "explain": args.explain,
                "prior_analysis": body.get("analysis"),
                "rejected_ids": shown,
                "asked_slots": body.get("asked_slots", []),
                "turn": body.get("turn", 1),
            }
            second = post_search(session, args.base_url, follow, args.timeout)
            results.append(
                {
                    "query_id": row["query_id"].strip(),
                    "kind": "clarify",
                    "repeat": item["repeat"],
                    "turn": 2,
                    "query": row["query"].strip(),
                    "client_ms": round(second["client_ms"], 2),
                    "status": second["status"],
                    "request_id": second["request_id"],
                    "error": second["error"],
                }
            )
            print(f"    재질문 {second['client_ms']:.0f}ms", flush=True)

        if args.sleep:
            time.sleep(args.sleep)

    # 서버 기록과 맞춘다
    log = load_timing_log(timing_log)
    matched = 0
    for record in results:
        server = log.get(record["request_id"])
        if server is None:
            record["server_total_ms"] = ""
            continue
        matched += 1
        record.update(flatten(server))
    if matched == 0:
        print(
            "\n[경고] 서버 구간 기록을 하나도 찾지 못했습니다. 서버를 "
            f"SEARCH_TIMING_LOG={timing_log} 로 띄웠는지 확인하세요.",
            file=sys.stderr,
        )

    # 저장
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "latency_detail.csv"
    fieldnames: List[str] = []
    for record in results:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with detail_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in results:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    measured = [r for r in results if r["kind"] == "measure"]
    summary = summarize(measured, ["client_ms"] + STAGE_COLUMNS[1:])
    summary_path = out_dir / "latency_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["stage", "n", "median_ms", "p95_ms", "max_ms", "min_ms"]
        )
        writer.writeheader()
        writer.writerows(summary)

    cold = next((r for r in results if r["kind"] == "cold"), None)
    warm = next((r for r in results if r["kind"] == "warm"), None)
    clarify_rows = [r for r in results if r["kind"] == "clarify"]

    runinfo = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": {
            "input": str(args.input),
            "split": args.split,
            "base_url": args.base_url,
            "top_k": args.top_k,
            "explain": args.explain,
            "repeat": args.repeat,
            "clarify": args.clarify,
            "query_ids": args.query_ids,
            "timing_log": str(timing_log),
            "output_dir": str(out_dir),
        },
        "requests": {
            "planned": len(plan),
            "recorded": len(results),
            "matched_server_records": matched,
            "failed": sum(1 for r in results if r["status"] != 200),
            "retried_analysis": sum(
                1 for r in results if (r.get("analysis_attempts") or 1) > 1
            ),
        },
        "timing_log_lines_before_run": log_lines_before,
        # 분석 캐시는 서비스 경로에 없다. 이 측정은 실제 경로 그대로이므로
        # 질의 분석 시간이 들어 있다 — 재측정(§5)과 정반대 조건이다.
        "analysis_cache": "off (service path)",
    }
    (out_dir / "latency_runinfo.json").write_text(
        json.dumps(runinfo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 화면 요약
    print("\n== 단계별 분포 (measure 건만, 단위 ms) ==")
    print(f"{'단계':<24}{'n':>4}{'중앙값':>10}{'p95':>10}{'최댓값':>10}")
    for line in summary:
        print(
            f"{line['stage']:<24}{line['n']:>4}{line['median_ms']:>10.0f}"
            f"{line['p95_ms']:>10.0f}{line['max_ms']:>10.0f}"
        )
    if cold and warm:
        print(
            f"\n최초 호출 {cold['client_ms']:.0f}ms · "
            f"두 번째 {warm['client_ms']:.0f}ms "
            f"(같은 질의 {cold['query_id']}) — 차이는 지연 로딩이다"
        )
    if clarify_rows:
        values = [r["client_ms"] for r in clarify_rows]
        print(
            f"재질문 {len(values)}건: 중앙값 {statistics.median(values):.0f}ms "
            f"최댓값 {max(values):.0f}ms (분석을 건너뛴 경로)"
        )
    failed = [r for r in results if r["status"] != 200]
    if failed:
        print(f"\n실패 {len(failed)}건: " + ", ".join(r["query_id"] for r in failed))
    print(f"\n저장: {detail_path} · {summary_path}")
    return 0 if matched else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="실제 API로 검색 지연시간을 재고 서버 구간 기록과 맞춘다",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split", default="dev", help="dev / test / all")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--timing-log",
        default="artifacts/timing/search.jsonl",
        help="서버가 SEARCH_TIMING_LOG로 쓰고 있는 파일",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="화면 기본값이 explain=true다. 끄고 재려면 --no-explain",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="질의당 반복 횟수(일시적 지연 확인)"
    )
    parser.add_argument(
        "--clarify",
        action="store_true",
        help="2턴(prior_analysis) 경로도 따로 잰다",
    )
    parser.add_argument("--query-ids", default="", help="쉼표로 구분한 일부 질의만")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="요청 사이 대기(초). 쿼터가 빡빡할 때만 쓴다",
    )
    return parser


def main() -> int:
    return measure(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
