"""v58 HTTP 검증 드라이버 — `start_server.sh`로 띄운 서버 하나에 시나리오를 보내고 결과를 JSON/CSV로 남긴다.

시나리오(TTL은 서버 환경변수, 기본 20초 · 재시도 15초):
  R1  정상 요청(최초 적재는 예열이 이미 했다) → 결과·인용 판
  gate 모드(자료 v2, 첫 문서 뒤 gate 파일까지 막힘) → TTL+1초 대기
  R2·R2b 동시 요청 → 소요 시간(수정 후: 즉시 · 수정 전: gate가 열릴 때까지) · 인용 판 · 상태(refreshing)
  R3  곧바로 한 번 더 → 자료원 호출이 늘지 않는가
  gate 열기 → 갱신 완료 대기 → R4 새 판인가
  fail 모드 → TTL+1초 대기 → R5 기존 판 유지·실패 기록 → 5초 뒤 R6 재시도 없음(간격 안) → normal(v3) + 간격 경과 → R7(갱신 시작) → R8 새 판
  gate 모드(v4) → TTL+1초 → R9(갱신 시작, 막힘) → SIGTERM → 3초 뒤 살아 있나(갱신을 기다림) → gate 열기 → 종료까지 시간 · 서버 로그에 취소/게시 없음 확인

수정 전 코드(old)에서는 R2가 gate가 열릴 때까지 막히므로 gate를 `--gate-after` 초 뒤 자동으로 연다. 종료 시나리오는 old에서 생략한다.
가사 원문은 기록하지 않는다(판 해시·곡 ID·시간만).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v58_runtime  # noqa: E402

BODY = {"query": v58_runtime.QUERY, "top_k": 15, "candidate_k": 30, "explain": True, "defer_clarify": True, "use_rerank": False}


class Check:
    def __init__(self, name: str, port: int, ttl: float, retry: float, gate_after: float) -> None:
        self.name = name
        self.base = f"http://127.0.0.1:{port}"
        self.ttl, self.retry, self.gate_after = ttl, retry, gate_after
        self.ctl = HERE / "ctl" / f"lyrics_ctl_{name}.json"
        self.gate_file = HERE / "ctl" / f"gate_{name}"
        self.pid = int((HERE / "ctl" / f"server_{name}.pid").read_text().strip())
        self.rows: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.client = httpx.Client(timeout=900.0)

    # --- 보조 ---------------------------------------------------------------
    def log(self, **payload: Any) -> None:
        payload["t"] = round(time.time(), 3)
        self.events.append(payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def set_ctl(self, **ctl: Any) -> None:
        ctl.setdefault("gate_file", str(self.gate_file))
        ctl.setdefault("gate_after", 1)
        self.ctl.write_text(json.dumps(ctl), encoding="utf-8")
        self.log(event="ctl", **ctl)

    def open_gate(self) -> None:
        self.gate_file.write_text("open", encoding="utf-8")
        self.log(event="gate_open")

    def close_gate(self) -> None:
        if self.gate_file.exists():
            self.gate_file.unlink()

    def status(self) -> Dict[str, Any]:
        return self.client.get(f"{self.base}/v58/status").json()

    def health(self) -> Dict[str, Any]:
        return self.client.get(f"{self.base}/health").json()

    def wait_ready(self, timeout: float = 600.0) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                body = self.health()
                if body["warmup"]["state"] in {"ready", "partial_failure"}:
                    self.log(event="ready", warmup=body["warmup"], lyrics_cache=body.get("lyrics_cache"))
                    return body
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError("서버가 준비되지 않았다")

    def request(self, label: str) -> Dict[str, Any]:
        started = time.perf_counter()
        response = self.client.post(f"{self.base}/api/v1/search", json=BODY)
        elapsed = time.perf_counter() - started
        row: Dict[str, Any] = {"arm": self.name, "label": label, "status_code": response.status_code,
                               "elapsed_ms": round(elapsed * 1000, 1), "request_id": response.headers.get("X-Request-Id")}
        if response.status_code == 200:
            body = response.json()
            results = body.get("results") or []
            row["top_ids"] = [r["id"] for r in results][:5]
            row["target_rank"] = next((i + 1 for i, r in enumerate(results) if r["id"] == v58_runtime.TARGET_ID), None)
            target = next((r for r in results if r["id"] == v58_runtime.TARGET_ID), None)
            evidence = ((target or {}).get("explain") or {}).get("evidence") or []
            row["quote_version"] = evidence[0]["data_version"] if evidence else None
            row["quote_span"] = [evidence[0]["char_start"], evidence[0]["char_end"]] if evidence else None
            row["quote_matches_target_text"] = (evidence[0]["quote"] == "창문을 여는 척 웃었고") if evidence else None
        else:
            row["error"] = response.text[:200]
        try:
            st = self.status()
            row["source_calls"] = st["source_calls"]
            lyrics = st["lyrics"]
            row["cache_state"] = lyrics.get("state")
            row["refreshing"] = lyrics.get("refreshing")
            row["refresh_successes"] = lyrics.get("refresh_successes")
            row["refresh_failures"] = lyrics.get("refresh_failures")
        except Exception as exc:  # noqa: BLE001
            row["status_error"] = type(exc).__name__
        self.rows.append(row)
        self.log(event="request", **row)
        return row

    def wait_source_calls(self, at_least: int, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.status()["source_calls"] >= at_least:
                return True
            time.sleep(0.2)
        return False

    def wait_refresh_done(self, timeout: float = 60.0) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            lyrics = self.status()["lyrics"]
            if lyrics.get("refreshing") is False:
                return lyrics
            time.sleep(0.2)
        raise TimeoutError("갱신이 끝나지 않았다")

    def concurrent(self, labels: List[str], stagger: float = 0.3) -> List[Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}

        def worker(label: str, delay: float) -> None:
            time.sleep(delay)
            out[label] = self.request(label)

        threads = [threading.Thread(target=worker, args=(label, i * stagger)) for i, label in enumerate(labels)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [out[label] for label in labels]

    # --- 시나리오 ------------------------------------------------------------
    def run(self, old: bool) -> Dict[str, Any]:
        ready = self.wait_ready()
        initial = self.status()
        self.log(event="initial_status", service_type=initial["service_type"], module=initial["service_module"],
                 guard=initial["guard"], config=initial["config"])
        self.close_gate()

        self.request("R1_normal")
        self.set_ctl(mode="gate", version="v2")
        self.log(event="sleep", seconds=self.ttl + 1, why="TTL 만료 대기")
        time.sleep(self.ttl + 1)

        opener: Optional[threading.Timer] = None
        if old:
            opener = threading.Timer(self.gate_after, self.open_gate)
            opener.start()
        rows = self.concurrent(["R2_after_expiry", "R2b_concurrent"])
        if not old:
            # 갱신이 막힌 상태에서도 상태가 보이고 자료원 호출은 1번만 늘어야 한다
            self.wait_source_calls(2, timeout=10)
            self.request("R3_while_refresh_blocked")
            self.log(event="status_while_blocked", lyrics=self.status()["lyrics"])
            self.open_gate()
        else:
            self.request("R3_after_blocking_reload")
            if opener is not None:
                opener.join()
        if not old:
            self.log(event="refresh_done", lyrics=self.wait_refresh_done())
        self.request("R4_after_refresh")

        # 실패·재시도
        self.close_gate()
        self.set_ctl(mode="fail", version="v2")
        self.log(event="sleep", seconds=self.ttl + 1, why="TTL 만료 대기(fail)")
        time.sleep(self.ttl + 1)
        self.request("R5_after_expiry_source_fails")
        if not old:
            self.log(event="after_failure", lyrics=self.wait_refresh_done())
        time.sleep(5.0)
        self.request("R6_within_retry_interval")
        self.set_ctl(mode="normal", version="v3")
        remaining = max(0.0, self.retry - 5.0 + 1.0)
        self.log(event="sleep", seconds=remaining, why="재시도 간격 경과 대기")
        time.sleep(remaining)
        self.request("R7_after_retry_interval")
        if not old:
            self.log(event="retry_done", lyrics=self.wait_refresh_done())
        self.request("R8_after_recovery")

        shutdown: Dict[str, Any] = {}
        if not old:
            self.close_gate()
            self.set_ctl(mode="gate", version="v4")
            self.log(event="sleep", seconds=self.ttl + 1, why="TTL 만료 대기(종료 시나리오)")
            time.sleep(self.ttl + 1)
            self.request("R9_refresh_blocked_before_shutdown")
            self.wait_source_calls(self.rows[-1]["source_calls"], timeout=10)
            os.kill(self.pid, signal.SIGTERM)
            sigterm_at = time.time()
            self.log(event="sigterm", pid=self.pid)
            time.sleep(3.0)
            alive_after_3s = _alive(self.pid)
            self.log(event="alive_after_3s", alive=alive_after_3s, why="갱신 읽기가 gate에 막혀 있어 종료가 기다려야 한다")
            self.open_gate()
            while _alive(self.pid) and time.time() - sigterm_at < 60:
                time.sleep(0.2)
            shutdown = {"alive_after_3s": alive_after_3s, "exited": not _alive(self.pid),
                        "seconds_from_sigterm_to_exit": round(time.time() - sigterm_at, 2)}
            self.log(event="shutdown", **shutdown)
        else:
            os.kill(self.pid, signal.SIGTERM)
            sigterm_at = time.time()
            while _alive(self.pid) and time.time() - sigterm_at < 60:
                time.sleep(0.2)
            shutdown = {"exited": not _alive(self.pid), "seconds_from_sigterm_to_exit": round(time.time() - sigterm_at, 2)}
            self.log(event="shutdown", **shutdown)

        return {"arm": self.name, "old_code": old, "ttl": self.ttl, "retry": self.retry, "ready": ready,
                "rows": self.rows, "shutdown": shutdown, "events": self.events}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # 좀비(종료했지만 회수 전)도 걸러낸다
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="new")
    parser.add_argument("--port", type=int, default=8058)
    parser.add_argument("--ttl", type=float, default=float(os.getenv("LYRICS_CACHE_TTL_SECONDS", "20")))
    parser.add_argument("--retry", type=float, default=float(os.getenv("LYRICS_REFRESH_RETRY_SECONDS", "15")))
    parser.add_argument("--old", action="store_true")
    parser.add_argument("--gate-after", type=float, default=25.0, help="old 코드에서 막힌 재적재를 풀 때까지(초)")
    args = parser.parse_args()

    check = Check(args.name, args.port, args.ttl, args.retry, args.gate_after)
    report = check.run(old=args.old)
    (HERE / f"http_check_{args.name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (HERE / f"http_check_{args.name}.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["arm", "label", "status_code", "elapsed_ms", "target_rank", "quote_version", "quote_span",
                  "quote_matches_target_text", "source_calls", "cache_state", "refreshing", "refresh_successes",
                  "refresh_failures", "request_id"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])
    print(f"written: http_check_{args.name}.json / .csv")


if __name__ == "__main__":
    main()
