"""v67(v66 server_state.py 이식, 포트·컨테이너만 다름) 서버 상태 지문 — 컨테이너(ID·시작 시각·재시작 수·상태)와 컬렉션(상태·최적화기·점 수·색인된 벡터 수·세그먼트 수)·별칭.

실제 모델 실행 사이에 서버가 재시작되거나 세그먼트가 바뀌면(최적화·HNSW 생성) 수정 전 판의 동점 순서가 바뀔 수 있다.
실행마다 직전·직후에 이 지문을 떠서 기준(`server_state_baseline.json`)과 같은지 본다. 읽기만 한다(쓰기·재시작 없음).

    venv/bin/python server_state.py --out <파일>                 # 지문만 저장
    venv/bin/python server_state.py --out <파일> --expect <기준>  # 기준과 다르면 종료 코드 4
    venv/bin/python server_state.py --wait-green 600 --out <파일> # 모든 컬렉션 green·최적화 끝까지 기다린 뒤 저장(준비 단계 전용)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

URL = "http://127.0.0.1:6370"
CONTAINER = "vague-finder-qdrant-v67"
# 비교에서 빼는 값(상태 지문이 아니라 측정 시각)
VOLATILE = ("captured_at",)


def _get(path: str) -> Any:
    with urllib.request.urlopen(URL + path, timeout=10) as resp:  # 루프백만
        return json.loads(resp.read().decode("utf-8"))["result"]


def capture() -> Dict[str, Any]:
    raw = subprocess.run(["docker", "inspect", CONTAINER], capture_output=True, text=True, check=True).stdout
    info = json.loads(raw)[0]
    container = {
        "name": CONTAINER, "id": info["Id"], "image": info["Config"]["Image"], "status": info["State"]["Status"],
        "started_at": info["State"]["StartedAt"], "restart_count": info["RestartCount"],
        "ports": {k: [b.get("HostIp", "") + ":" + b.get("HostPort", "") for b in (v or [])]
                  for k, v in (info["NetworkSettings"]["Ports"] or {}).items()},
    }
    collections = {}
    for c in sorted(x["name"] for x in _get("/collections")["collections"]):
        d = _get(f"/collections/{c}")
        collections[c] = {
            "status": d.get("status"), "optimizer_status": d.get("optimizer_status"),
            "points_count": d.get("points_count"), "indexed_vectors_count": d.get("indexed_vectors_count"),
            "segments_count": d.get("segments_count"),
        }
    aliases = sorted((a["alias_name"], a["collection_name"]) for a in _get("/aliases")["aliases"])
    version = json.loads(urllib.request.urlopen(URL + "/", timeout=10).read().decode("utf-8")).get("version")
    return {"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "container": container, "server_version": version,
            "collections": collections, "aliases": aliases}


def strip(state: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in state.items() if k not in VOLATILE}


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> list:
    sa, sb = strip(a), strip(b)
    out = []
    for key in sorted(set(sa) | set(sb)):
        if json.dumps(sa.get(key), sort_keys=True) != json.dumps(sb.get(key), sort_keys=True):
            out.append(key)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect", default="")
    parser.add_argument("--wait-green", type=float, default=0.0)
    args = parser.parse_args()
    if args.wait_green:
        end = time.time() + args.wait_green
        stable_since = None
        last = None
        while True:
            state = capture()
            green = all(c["status"] == "green" and c["optimizer_status"] == "ok" for c in state["collections"].values())
            if green and last is not None and not diff(state, last):
                stable_since = stable_since or time.time()
            else:
                stable_since = None
            if stable_since and time.time() - stable_since >= 30:
                break
            if time.time() > end:
                print("서버가 시간 안에 green·안정 상태가 되지 않았다", file=sys.stderr)
                return 5
            last = state
            time.sleep(5)
    state = capture()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.expect:
        expected = json.loads(Path(args.expect).read_text(encoding="utf-8"))
        changed = diff(state, expected)
        if changed:
            print(f"서버 상태가 기준과 다르다: {changed}", file=sys.stderr)
            return 4
        print("server_state: 기준과 같음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
