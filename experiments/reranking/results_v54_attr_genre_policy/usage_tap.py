"""Gemini 호출 기록 — 저장소 안에서 끝나는 사용량 탭(2026-10-15, v50).

v49는 저장소 밖 `sitecustomize`로 `generate_content`를 감쌌다. 그 파일이 없으면 기록이 안 남으므로 이번에는
실험 폴더 안의 이 파일로 대신한다. **프롬프트·응답 본문은 적지 않는다.** 호출마다 한 줄(JSONL):
시각 · 종류(generate/generate_async/count_tokens) · 모델 · 지연 · 토큰(prompt/candidates/thoughts/cached/total) ·
응답 `model_version` · `finish_reason` · `response_id` · 예외 종류(실패 시). 재시도는 호출마다 한 줄씩 남으므로
"호출 수 − 질의 수"가 재시도 수다.

쓰는 법 — 측정 모듈을 이 파일로 감싸서 띄운다(모듈 이름과 인자는 그대로):

    GEMINI_USAGE_TAP=<기록 파일.jsonl> venv/bin/python experiments/reranking/results_v50_vertex_context/usage_tap.py \
        src.retrieval.evaluate_search_accuracy --input ... --split dev ...

`GEMINI_USAGE_TAP`이 비어 있으면 아무것도 기록하지 않고 모듈만 실행한다. 저장소 루트(cwd)를 sys.path 앞에 넣는다.
"""
from __future__ import annotations

import datetime as _dt
import functools
import json
import os
import runpy
import sys
import threading
import time

_LOCK = threading.Lock()


def _usage_row(kind: str, model: str, started: float, response=None, error: BaseException | None = None) -> dict:
    row = {
        "ts": _dt.datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "kind": kind,
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "tag": os.getenv("GEMINI_USAGE_TAG", ""),
        "pid": os.getpid(),
    }
    if error is not None:
        row["error"] = type(error).__name__
        code = getattr(error, "code", None)
        if code is not None:
            row["error_code"] = code
        return row
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        for key, name in (
            ("prompt_token_count", "prompt_tokens"),
            ("candidates_token_count", "candidates_tokens"),
            ("thoughts_token_count", "thoughts_tokens"),
            ("cached_content_token_count", "cached_tokens"),
            ("total_token_count", "total_tokens"),
        ):
            row[name] = getattr(usage, key, None)
    row["model_version"] = getattr(response, "model_version", None)
    row["response_id"] = getattr(response, "response_id", None)
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish = getattr(candidates[0], "finish_reason", None)
        row["finish_reason"] = getattr(finish, "value", finish) if finish is not None else None
    if kind == "count_tokens":
        row["total_tokens"] = getattr(response, "total_tokens", None)
    return row


def _append(path: str, row: dict) -> None:
    line = json.dumps(row, ensure_ascii=False)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def install(path: str) -> None:
    """`google.genai`의 동기·비동기 generate_content와 count_tokens를 감싼다. 한 프로세스에서 한 번만."""
    from google.genai import models as genai_models

    if getattr(genai_models, "_vf_usage_tap_installed", False):
        return

    def wrap_sync(kind: str, original):
        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            model = kwargs.get("model") or (args[0] if args else "")
            started = time.perf_counter()
            try:
                response = original(self, *args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - 기록만 하고 그대로 올린다
                _append(path, _usage_row(kind, str(model), started, error=exc))
                raise
            _append(path, _usage_row(kind, str(model), started, response))
            return response
        return wrapper

    def wrap_async(kind: str, original):
        @functools.wraps(original)
        async def wrapper(self, *args, **kwargs):
            model = kwargs.get("model") or (args[0] if args else "")
            started = time.perf_counter()
            try:
                response = await original(self, *args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                _append(path, _usage_row(kind, str(model), started, error=exc))
                raise
            _append(path, _usage_row(kind, str(model), started, response))
            return response
        return wrapper

    genai_models.Models.generate_content = wrap_sync("generate", genai_models.Models.generate_content)
    genai_models.Models.count_tokens = wrap_sync("count_tokens", genai_models.Models.count_tokens)
    genai_models.AsyncModels.generate_content = wrap_async("generate_async", genai_models.AsyncModels.generate_content)
    genai_models._vf_usage_tap_installed = True


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: usage_tap.py <module> [args...]   (GEMINI_USAGE_TAP=<jsonl>)")
    root = os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    path = os.getenv("GEMINI_USAGE_TAP", "").strip()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        install(path)
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
