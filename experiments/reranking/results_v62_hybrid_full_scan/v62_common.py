"""v62 공용 — 경로 · 외부 호출 차단(v61_guard 재사용) · v61 저장 입력(읽기 전용) · 색인 지문 확인.

v61 도구(`exact_hybrid.py` · `v61_guard.py` · `hybrid_capture.py` · `run_candidates.py` · `compare_candidates.py`)는 이 저장소에
커밋돼 있어 그대로 import한다. v61의 **저장 질의 벡터**(`cand_A/<set>_dev/calls.jsonl` · `calls_vectors.npz`)와 후보 기록
(`candidates.jsonl` · `search_eval_dev_detail.csv`)은 커밋 대상이 아니라 v61 워크트리에만 있다 — `V61_SOURCE_ROOT`에서 **읽기만** 한다.
색인 복사본은 v61 복사본을 다시 복사한 `artifacts/qdrant_v61_audit`(이 워크트리)이고, 내용 지문이 v61 기록(`238299e6c8c17f2a`)과
같아야 한다(다르면 멈춘다).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V61_DIR = ROOT / "experiments/reranking/results_v61_hybrid_scan_audit"
V61_SOURCE_ROOT = Path(os.getenv(
    "V61_SOURCE_ROOT",
    "/private/tmp/vague-finder-hybrid-scan-audit/experiments/reranking/results_v61_hybrid_scan_audit",
))
V61_BASE_COMMIT = "fc20e86"          # PR #76 병합 커밋 — 옛(상한 5000) 구현과 결함 재현 테스트가 있는 판
EXPECTED_FINGERPRINT = "238299e6c8c17f2a"
SETS = ("v06", "v09", "v10")
CACHES = ROOT / "experiments/reranking/results_v49_vertex_baseline"


def add_paths() -> None:
    for p in (str(ROOT), str(V61_DIR), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(ROOT)


def guard(overrides: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """v61_guard로 측정 조건을 고정하고 외부 클라이언트 생성을 막는다. (적용 env, 차단 확인) 반환."""
    add_paths()
    import v61_guard  # noqa: E402

    applied = v61_guard.fixed_env(overrides)
    v61_guard.install()
    return applied, v61_guard.verify()


def verify_guard() -> Dict[str, str]:
    import v61_guard  # noqa: E402

    return v61_guard.verify()


def git_sha() -> str:
    """HEAD 짧은 sha. 운영 코드(`src/`)에 커밋되지 않은 변경이 있으면 `+dirty`를 붙인다 — 커밋 전 검증임을 기록에 남기기 위해."""
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return f"{sha}+dirty" if dirty else sha


def legacy_backend_source() -> str:
    """v61 기준 커밋의 `qdrant_backend.py` 원문(옛 상한 5000 구현). 벤치의 '수정 전' 구성에 쓴다."""
    out = subprocess.run(["git", "show", f"{V61_BASE_COMMIT}:src/vector_db/qdrant_backend.py"], capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError(f"git show 실패: {out.stderr.strip()}")
    return out.stdout


def load_legacy_backend(work_dir: Path):
    """옛 구현을 별도 모듈 이름으로 import한다(현재 모듈은 건드리지 않는다)."""
    import importlib.util

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "qdrant_backend_legacy_fc20e86.py"
    path.write_text(legacy_backend_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("qdrant_backend_legacy_fc20e86", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_calls(set_name: str, arm: str = "A") -> Tuple[List[Dict[str, Any]], Any]:
    """v61 저장 호출 기록과 질의 벡터(읽기 전용)."""
    folder = V61_SOURCE_ROOT / f"cand_{arm}" / f"{set_name}_dev"
    calls_path, vec_path = folder / "calls.jsonl", folder / "calls_vectors.npz"
    if not calls_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f"v61 저장 입력이 없다: {folder} (V61_SOURCE_ROOT를 확인)")
    calls = [json.loads(line) for line in calls_path.open(encoding="utf-8")]
    return calls, np.load(vec_path)


def call_inputs(call: Dict[str, Any], vectors: Any) -> Tuple[List[float], Dict[str, Any], Optional[Dict[str, Any]]]:
    seq = call["seq"]
    dense = vectors[f"call{seq}_dense"].astype(float).tolist()
    sparse = {"indices": vectors[f"call{seq}_sparse_idx"].tolist(), "values": vectors[f"call{seq}_sparse_val"].astype(float).tolist()}
    flt = json.loads(call["filter"]) if call["filter"] else None
    return dense, sparse, flt


def check_fingerprint(fingerprint: Dict[str, Any]) -> Dict[str, Any]:
    if fingerprint.get("content_sha256_16") != EXPECTED_FINGERPRINT:
        raise RuntimeError(f"색인 복사본 지문이 v61 기록과 다르다: {fingerprint} ≠ {EXPECTED_FINGERPRINT}")
    return {**fingerprint, "matches_v61": True}


def analysis_cache_compat() -> Dict[str, Any]:
    """v49 Vertex dev 분석 캐시가 지금 분석기 조건과 맞는지(drift 빈 목록 · 폴백 0)."""
    from src.retrieval.analysis_cache import load_cache  # noqa: E402

    out: Dict[str, Any] = {}
    for set_name in SETS:
        path = CACHES / f"analysis_cache_{set_name}_dev_vertex.json"
        cache = load_cache(path)
        out[set_name] = {
            "path": str(path.relative_to(ROOT)), "entries": len(cache.entries),
            "fingerprint_drift": cache.fingerprint_drift(), "fallback_ids": cache.fallback_ids(), "route": cache.route(),
        }
    return out


def source_inputs_digest() -> Dict[str, Any]:
    """읽기 전용 입력(v61 저장 벡터·기록)의 크기·sha — 어떤 파일을 재생했는지 남긴다."""
    import hashlib

    out: Dict[str, Any] = {"root": str(V61_SOURCE_ROOT), "files": {}}
    for set_name in SETS:
        for name in ("calls.jsonl", "calls_vectors.npz", "candidates.jsonl", "search_eval_dev_detail.csv"):
            path = V61_SOURCE_ROOT / "cand_A" / f"{set_name}_dev" / name
            if path.exists():
                out["files"][f"cand_A/{set_name}_dev/{name}"] = {
                    "bytes": path.stat().st_size, "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                }
    return out
