"""Effective Context settings, read when the search router is constructed."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import os
import re
from typing import Mapping


ENV_DEFAULTS = {
    "CONTEXT_WEIGHT": "0.5",
    "CONTEXT_NAMED_MEDIA_MULTIPLIER": "2.0",
    "CONTEXT_FACT_K": "100",
    "CONTEXT_SPARSE_K": "100",
    "SEARCH_DEFAULT_CANDIDATE_K": "30",
}
ROUTER_FIELDS = {
    "context_weight": "weight",
    "context_fact_k": "fact_k",
    "context_sparse_k": "sparse_k",
    "context_named_media_multiplier": "named_media_multiplier",
    "default_candidate_k": "candidate_k",
}


@dataclass(frozen=True)
class ContextSearchSettings:
    weight: float = 0.5
    named_media_multiplier: float = 2.0
    fact_k: int = 100
    sparse_k: int = 100
    candidate_k: int = 30

    def router_arguments(self) -> dict[str, float | int]:
        return {name: getattr(self, field) for name, field in ROUTER_FIELDS.items()}


def load_context_settings(environ: Mapping[str, str] | None = None) -> ContextSearchSettings:
    """Missing keys use defaults; invalid explicit values fail with the key name.

    This function does not load .env or cache values. The existing application
    bootstrap/Compose supplies environment variables before calling the factory.
    """
    env = os.environ if environ is None else environ

    def value(key: str) -> str:
        return env.get(key, ENV_DEFAULTS[key]).strip()

    def number(key: str, minimum: float) -> float:
        try:
            result = float(value(key))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a finite number >= {minimum}") from None
        if not math.isfinite(result) or result < minimum:
            raise ValueError(f"{key} must be a finite number >= {minimum}")
        return result

    def integer(key: str, maximum: int | None = None) -> int:
        raw = value(key)
        if not re.fullmatch(r"[0-9]+", raw):
            raise ValueError(f"{key} must be a positive integer")
        result = int(raw)
        if result < 1 or (maximum is not None and result > maximum):
            suffix = f" <= {maximum}" if maximum is not None else ""
            raise ValueError(f"{key} must be a positive integer{suffix}")
        return result

    return ContextSearchSettings(
        weight=number("CONTEXT_WEIGHT", 0.0),
        named_media_multiplier=number("CONTEXT_NAMED_MEDIA_MULTIPLIER", 1.0),
        fact_k=integer("CONTEXT_FACT_K"),
        sparse_k=integer("CONTEXT_SPARSE_K"),
        candidate_k=integer("SEARCH_DEFAULT_CANDIDATE_K", 100),
    )


def assert_fixed_context_settings(router: object) -> dict[str, float | int]:
    """Check effective factory values; never silently replace benchmark settings."""
    expected = ContextSearchSettings().router_arguments()
    observed = {name: getattr(router, "_" + name, None) for name in expected}
    # Router stores default_candidate_k under _default_candidate_k as well.
    if observed != expected:
        raise ValueError(f"Effective Context settings differ from the fixed evaluation: "
                         f"expected={expected}, observed={observed}")
    return observed


def factory_uses_context_settings(factory: ast.FunctionDef | ast.AsyncFunctionDef,
                                  call: ast.Call) -> bool:
    """Recognize explicit validated settings wiring for the legacy activation tool."""
    assignments = [n for n in factory.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == "context_settings"]
    if len(assignments) != 1:
        return False
    load = assignments[0].value
    if (not isinstance(load, ast.Call) or not isinstance(load.func, ast.Name)
            or load.func.id != "load_context_settings" or load.args or load.keywords):
        return False
    if assignments[0].lineno >= call.lineno:
        return False
    keywords = {kw.arg: kw.value for kw in call.keywords}
    return all(isinstance(keywords.get(name), ast.Attribute)
               and isinstance(keywords[name].value, ast.Name)
               and keywords[name].value.id == "context_settings"
               and keywords[name].attr == field
               for name, field in ROUTER_FIELDS.items())
