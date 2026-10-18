"""build_resolution_cache 자체 검사 — 가짜 클라이언트만(네트워크·자격 증명 없음). 합성 질의만 쓴다."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_resolution_cache as brc  # noqa: E402  (ROOT를 sys.path에 넣는다)
from src.retrieval import work_name_resolution as wnr  # noqa: E402

Q = [{"qid": f"s{i}", "query": f"합성 질의 {i} — 의사들이 밴드 하는 드라마 노래"} for i in range(6)]
OK = json.dumps({"decision": "candidates", "works": [
    {"name": "가상병원", "aliases": [], "media_type": "drama", "evidence": "의사들이 밴드 하는 드라마"}]})


def run(script, *, max_calls=100, max_usd=1.0):
    with tempfile.TemporaryDirectory() as d:
        ledger = brc.Ledger(Path(d) / "ledger.jsonl", max_calls=max_calls, max_usd=max_usd)
        meta = brc.build(Q, out=Path(d) / "cache.json", ledger=ledger, client_factory=brc._fake_factory(script),
                         model="gemini-3.1-flash-lite")
        cache = json.loads((Path(d) / "cache.json").read_text(encoding="utf-8"))
        resumed = None
        try:
            brc.Ledger(Path(d) / "ledger.jsonl", max_calls=max_calls, max_usd=max_usd)
        except brc.StopRun as exc:
            resumed = str(exc)
        return meta, cache, ledger.rows, resumed


def main() -> int:
    checks = []
    meta, cache, rows, resumed = run([OK] * 6)
    checks.append(("정상 6건", meta["entries"] == 6 and not meta["stopped"] and resumed is None))
    path = Path(tempfile.mkdtemp()) / "c.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    cached = wnr.CachedWorkNameResolver(path, expect_model="gemini-3.1-flash-lite")
    checks.append(("캐시를 해석기로 읽음", cached.resolve(Q[0]["query"], timeout_seconds=1).works[0].name == "가상병원"))
    checks.append(("질의 원문 미저장", all(q["query"] not in json.dumps(cache, ensure_ascii=False) for q in Q)))
    meta, _, rows, resumed = run([RuntimeError("403 PERMISSION_DENIED"), OK])
    checks.append(("권한 첫 오류 정지", meta["stopped"] == "auth_or_billing_error" and meta["entries"] == 1
                   and resumed is not None))
    meta, _, _, _ = run([RuntimeError("504 DEADLINE_EXCEEDED")] * 3 + [OK])
    checks.append(("504 3회 정지", meta["stopped"] in {"deadline_504x3", "consecutive_failures_3"} and meta["entries"] == 3))
    meta, _, _, _ = run([RuntimeError("500"), RuntimeError("500"), OK, RuntimeError("500"), OK, OK])
    checks.append(("성공이 연속 실패를 끊음", not meta["stopped"] and meta["entries"] == 6))
    meta, _, _, _ = run([OK] * 6, max_calls=2)
    checks.append(("호출 상한", meta["stopped"] == "max_calls" and meta["entries"] == 2))
    one = brc.reserve_usd(wnr.build_resolver_prompt(Q[0]["query"]))
    # 성공은 사용량으로 정산(가짜는 작다), 실패는 예약액 그대로 미확인 — 실패 두 번이면 2.5회분 상한에서 네 번째 전에 멈춘다
    meta, _, rows, _ = run([RuntimeError("500"), OK, RuntimeError("500"), OK, OK, OK], max_usd=one * 2.5)
    unverified = sum(r["reserved_usd"] for r in rows if r.get("kind") == "attempt" and not r.get("ok"))
    checks.append(("실패 예약은 상한에 남음", meta["stopped"] == "max_usd" and meta["entries"] == 3
                   and abs(unverified - 2 * round(one, 6)) < 1e-9))
    for label, ok in checks:
        print("ok  " if ok else "FAIL", label)
    passed = sum(ok for _, ok in checks)
    print(f"{passed}/{len(checks)}  (1회 예약 상한 ${one:.6f})")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
