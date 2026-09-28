"""
Latency recorder -- p50/p95/p99 통계용 타이밍 수집.

사용 예:
    rec = LatencyRecorder()
    with rec.time("total"):
        do_something()
    rec.stats("total")   # {n, p50, p95, p99, mean}

context manager + 직접 record() 두 가지 인터페이스 모두 제공.
검색 라우터 내부에서 stage 별 타이밍을 수집하려면 record() 호출이 유연함.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterator, List


class LatencyRecorder:
    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def time(self, label: str) -> Iterator[None]:
        """with 블록 안의 elapsed time(ms) 을 자동 기록."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[label].append((time.perf_counter() - t0) * 1000.0)

    def record(self, label: str, ms: float) -> None:
        """수동 기록 (외부에서 측정한 값 추가)."""
        self._timings[label].append(ms)

    def percentile(self, label: str, p: float) -> float:
        """선형 보간 없이 가장 가까운 정렬 인덱스 사용 (작은 표본에 적합)."""
        vals = sorted(self._timings.get(label, []))
        if not vals:
            return 0.0
        idx = min(int(len(vals) * p / 100.0), len(vals) - 1)
        return vals[idx]

    def stats(self, label: str) -> Dict[str, float]:
        vals = self._timings.get(label, [])
        if not vals:
            return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        return {
            "n": len(vals),
            "p50": self.percentile(label, 50),
            "p95": self.percentile(label, 95),
            "p99": self.percentile(label, 99),
            "mean": sum(vals) / len(vals),
        }

    def labels(self) -> List[str]:
        return list(self._timings.keys())

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        return {label: self.stats(label) for label in self._timings}

    def reset(self) -> None:
        self._timings.clear()
