"""Apply targeted review changes to the user's merged sources, with backups."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from src.retrieval.context_search_settings import ENV_DEFAULTS, ROUTER_FIELDS, factory_uses_context_settings


PROMPT_START = "Context examples (other fields follow the output schema above):"
PROMPT_EXAMPLES = '''Context examples (other fields follow the output schema above):
The following work is fictional and illustrates the schema only; it is not a song fact.
Only use people, works and scene details actually mentioned in the user's query.
Do not add a character, scene, song title or answer from an example to another query.
When the user recalls an unnamed work, retain the supplied scene, era and singer
constraints in its Context search_query. Do not reduce that memory to a broad
category such as "drama OST". Keep independent artwork and waveform clues in
their dedicated fields; do not invent a work name or a missing answer.
- "가상 작품 A에서 주인공이 떠나는 장면에 나온 노래" ->
  context_clues=[{{"target":"가상 작품 A","relation":"삽입곡·배경음악",
  "search_query":"가상 작품 A에서 주인공이 떠나는 장면에 나온 노래","confidence":0.8}}]
- "앨범 표지에 만화 그림이 있고 가사에 주인공이라는 말이 있어" -> context_clues=[]
'''
DOC_START = "<!-- context-review-cleanup:start -->"
DOC_END = "<!-- context-review-cleanup:end -->"
DOC_SECTION = '''<!-- context-review-cleanup:start -->
## 나무위키 Context 검색과 결과 근거

Context 검색은 API의 검색 후보 결합 단계에 연결되어 있다. Dense 사실 검색을
곡별 최고 점수로 집계한 뒤 Sparse 곡 프로필 순위와 내부 RRF로 결합하고,
기존 Text/Image/Audio 검색에는 하나의 보조 경로로 합류한다.

`POST /api/v1/search`의 `results[].context_evidence`에는 질의와 관련된 사실을
선택한 경우에만 `record_id`, `fact_text`, `source_url`, `section`, `category`가
포함된다. 관련성·출처 검사를 통과하지 않으면 근거는 비어 있다. Sparse 일치나
검색 점수 자체를 사실의 근거로 표시하지 않는다. 근거가 없다는 이유만으로
그 곡의 기존 검색 점수를 깎지 않는다. 이 검사는 원문 내용의 진위를 보증하지 않는다.

나무위키 원문, Context artifact, 임베딩, 로컬 Qdrant 데이터, 곡 목록,
평가 질의·라벨·분석 캐시·개인 실행 보고서는 공개 저장소에 올리지 않는다.
기존 `.env`의 Gemini 키와 공용 MongoDB URI도 공개하지 않는다. 공개 가능한 것은
검색 코드, 테스트 코드, 문서와 비밀 값이 없는 `.env.example`이다.

설정은 `load_context_settings()`가 검색 라우터 생성 시 환경변수에서 읽는다.
없으면 `CONTEXT_WEIGHT=0.5`, `CONTEXT_NAMED_MEDIA_MULTIPLIER=2.0`,
`CONTEXT_FACT_K=100`, `CONTEXT_SPARSE_K=100`, `SEARCH_DEFAULT_CANDIDATE_K=30`을
사용한다. `.env`를 바꾼 후에는 컨테이너를 재생성한다. 화면 `top_k=10`은
요청 스키마 기본값이며 이번 재검증에서 최적값을 새로 선택하지 않는다.

리랭킹 OFF인 동일 분석의 Context OFF/ON으로 Candidate@30, Hit@10, MRR@10,
회귀 손실, 실제 이미지·오디오 경로 실행과 표시 근거를 비교한다.
기존 dev 74문항/test 32문항/추가 60문항은 이미 관측한 질의다. 새 분석을
사용해도 새 블라인드 표본으로 간주하지 않는다. 수정 후 성능 주장은 새 실행의
`validation_report.json`과 `api_smoke_report.json`을 확인한 뒤 작성한다.
<!-- context-review-cleanup:end -->
'''
CONFIGURE = '''def _configure(router: Any) -> None:
    # Validate the actual factory; never overwrite settings just to match a report.
    from src.retrieval.context_search_settings import assert_fixed_context_settings
    assert_fixed_context_settings(router)
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode(data: bytes) -> tuple[bytes, str, str]:
    bom = b"\xef\xbb\xbf" if data.startswith(b"\xef\xbb\xbf") else b""
    source = data[len(bom):].decode("utf-8")
    return bom, source, "\r\n" if "\r\n" in source else "\n"


def positions(data: bytes):
    starts, offset = [], 0
    for line in data.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)

    def position(node: ast.AST, end: bool = False) -> int:
        return starts[(node.end_lineno if end else node.lineno) - 1] + (node.end_col_offset if end else node.col_offset)
    return starts, position


def edit_factory(data: bytes) -> bytes:
    bom, source, newline = decode(data)
    content = source.encode("utf-8")
    tree = ast.parse(source)
    factories = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_search_router"]
    if len(factories) != 1:
        raise ValueError("Expected one get_search_router factory")
    factory = factories[0]
    returns = [n for n in factory.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == "SearchRouter"]
    calls = [n for n in ast.walk(factory) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "SearchRouter"]
    if len(returns) != 1 or len(calls) != 1 or any(kw.arg is None for kw in calls[0].keywords):
        raise ValueError("Expected one direct SearchRouter return without **kwargs; merged code preserved")
    call = calls[0]
    keywords = {kw.arg: kw for kw in call.keywords}
    if len(keywords) != len(call.keywords) or "context_search" not in keywords:
        raise ValueError("Explicit Context search wiring is required")
    if isinstance(keywords["context_search"].value, ast.Constant) and keywords["context_search"].value.value is None:
        raise ValueError("Context search must remain enabled")
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
               and n.module == "src.retrieval.context_search_settings"]
    imported = [alias for n in imports for alias in n.names if alias.name == "load_context_settings"]
    if imported and (len(imported) != 1 or imported[0].asname is not None):
        raise ValueError("Ambiguous load_context_settings import")
    if factory_uses_context_settings(factory, call) and imported:
        return data
    if any(isinstance(n, ast.Name) and n.id == "context_settings" for n in ast.walk(factory)):
        raise ValueError("Unrecognized existing context_settings binding; no source overwritten")
    starts, position = positions(content)
    edits = []
    for name, field in ROUTER_FIELDS.items():
        if name not in keywords:
            raise ValueError(f"Merged router factory is missing {name}; inspect the merge first")
        node = keywords[name].value
        edits.append((position(node), position(node, True), f"context_settings.{field}".encode()))
    ret = returns[0]
    start = starts[ret.lineno - 1]
    indent = content[start:position(ret)]
    if indent.strip():
        raise ValueError("Router return must occupy its own line")
    edits.append((start, start, indent + b"context_settings = load_context_settings()" + newline.encode()))
    if not imported:
        top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        if not top_imports:
            raise ValueError("Expected normal application imports before the router factory")
        last = top_imports[-1]
        after = starts[last.end_lineno - 1] + len(content.splitlines(keepends=True)[last.end_lineno - 1])
        addition = b"from src.retrieval.context_search_settings import load_context_settings" + newline.encode()
        edits.append((after, after, addition))
    for start, end, value in sorted(edits, reverse=True):
        content = content[:start] + value + content[end:]
    ast.parse(content.decode("utf-8"))
    return bom + content


def edit_prompt(data: bytes) -> bytes:
    bom, source, newline = decode(data)
    normalized = source.replace("\r\n", "\n")
    if normalized.count(PROMPT_START) != 1:
        raise ValueError("Cannot locate the one Context prompt example block")
    start = normalized.index(PROMPT_START)
    end = normalized.find("\nNow analyze:\n", start)
    if end < 0:
        raise ValueError("Context prompt end marker missing")
    before = ast.parse(source)
    literals = [n for n in before.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_PROMPT_TEMPLATE" for t in n.targets)]
    if len(literals) != 1 or not isinstance(literals[0].value, ast.Constant):
        raise ValueError("Expected one literal _PROMPT_TEMPLATE")
    result = normalized[:start] + PROMPT_EXAMPLES + normalized[end:]
    ast.parse(result)
    return bom + result.replace("\n", newline).encode("utf-8")


def edit_env(data: bytes) -> bytes:
    bom, source, newline = decode(data)
    lines, seen = [], set()
    for line in source.splitlines(keepends=True):
        found = re.match(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=", line)
        key = found.group(1) if found else None
        if key in ENV_DEFAULTS:
            if key not in seen:
                lines.append(f"{key}={ENV_DEFAULTS[key]}{newline}")
                seen.add(key)
        else:
            lines.append(line)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += newline
    missing = [key for key in ENV_DEFAULTS if key not in seen]
    if missing:
        lines.append(newline + "# Context ranking: fixed comparison defaults; restart after changes." + newline)
        lines.extend(f"{key}={ENV_DEFAULTS[key]}{newline}" for key in missing)
    return bom + "".join(lines).encode("utf-8")


def edit_env_example(data: bytes) -> bytes:
    bom, source, newline = decode(edit_env(data))
    start = source.find("# Step 10 Context ranking.")
    if start >= 0:
        end = source.find("CONTEXT_WEIGHT=", start)
        if end < 0 or any(line.strip() and not line.lstrip().startswith("#")
                          for line in source[start:end].splitlines()):
            raise ValueError("Unexpected content in the Context example comments")
        note = ("# Context ranking: read from environment when the search router is created." + newline +
                "# These are the fixed follow-up defaults. Recheck new code with reranking OFF." + newline +
                "# Missing keys use the same defaults. Restart after changing .env." + newline +
                "# candidate_k can be overridden per request; top_k defaults to 10 in its schema." + newline)
        source = source[:start] + note + source[end:]
    return bom + source.encode()


def edit_policy(data: bytes) -> bytes:
    bom, source, newline = decode(data)
    normalized = source.replace("\r\n", "\n")
    lines = []
    for line in normalized.splitlines(keepends=True):
        if ("나무위키" in line or "Context" in line) and "연결 전" in line:
            line = re.sub(r"검색 연결 전(?:\s*프로토타입)?|연결 전(?:\s*프로토타입)?", "API 검색 후보 결합에 연결됨", line)
        lines.append(line)
    text = "".join(lines)
    if DOC_START in text or DOC_END in text:
        if text.count(DOC_START) != 1 or text.count(DOC_END) != 1:
            raise ValueError("Ambiguous Context policy section")
        a, b = text.index(DOC_START), text.index(DOC_END) + len(DOC_END)
        text = text[:a] + DOC_SECTION.rstrip("\n") + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n" + DOC_SECTION
    return bom + text.replace("\n", newline).encode("utf-8")


def edit_evaluator(data: bytes) -> bytes:
    bom, source, newline = decode(data)
    content = source.encode("utf-8")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_configure"]
    if len(functions) != 1:
        raise ValueError("Expected one fixed evaluator _configure function")
    node = functions[0]
    _, position = positions(content)
    current = content[position(node):position(node, True)].decode()
    if "assert_fixed_context_settings" not in current:
        attrs = {n.targets[0].attr for n in node.body if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)}
        expected = {"_context_weight", "_context_named_media_multiplier", "_context_fact_k", "_context_sparse_k"}
        if len(node.body) != 4 or attrs != expected:
            raise ValueError("Unrecognized evaluator configuration; no hidden parameter override allowed")
        content = content[:position(node)] + CONFIGURE.rstrip("\n").replace("\n", newline).encode() + content[position(node, True):]
    text = content.decode()
    text = text.replace('analysis_policy="new cache in this experiment only; same analysis in OFF/ON",',
                        'analysis_policy="new cache in this experiment only; same analysis in OFF/ON",' + newline +
                        '                 question_exposure="post-review reanalysis of previously observed dev/test/60 questions; not a new blind sample",') if 'question_exposure=' not in text else text
    text = text.replace("기존 dev/test는 이미 공개된 회귀 평가이며, 새로운 60문항만 새 독립 평가다.",
                        "dev/test와 추가 60문항 모두 이전 실행에서 관측한 질의다. 이번 실행은 새 분석을 사용하는 회귀 재검증이며 새 블라인드 평가가 아니다.")
    text = text.replace("새 문항은 코퍼스에 근거해 작성한 테스트이며 실제 사용자 질문의 무작위 표본은 아니다.",
                        "추가 60문항은 코퍼스에 근거해 작성한 테스트이며 실제 사용자 질문의 무작위 표본은 아니다. independent는 기존 파일 형식을 유지한 분할명이다.")
    text = text.replace("Registered fixed settings, 106 reused queries, and 60 new reviewed queries.",
                        "Registered fixed settings and reanalysis of 166 previously observed queries; no new blind claim.")
    text = text.replace("new independent search remains unrun.", "the additional disclosed 60-question reanalysis remains unrun.")
    if "새로운 60문항만 새 독립 평가다" in text or "question_exposure=" not in text:
        raise ValueError("Evaluator provenance could not be corrected")
    ast.parse(text)
    return bom + text.encode()


def edit_activation(data: bytes) -> bytes:
    bom, source, newline = decode(data)
    if "# Preserve validated environment wiring." in source:
        return data
    needle = "    lines = content.splitlines(keepends=True)"
    if source.count(needle) != 1:
        raise ValueError("Cannot locate activation rewrite guard insertion point")
    guard = '''    # Preserve validated environment wiring. Do not turn it back into literals.
    from src.retrieval.context_search_settings import factory_uses_context_settings
    if factory_uses_context_settings(factories[0], call):
        return data
'''.replace("\n", newline)
    result = source.replace(needle, guard + needle)
    ast.parse(result)
    return bom + result.encode()


def edit_pipeline_test_stubs(data: bytes) -> bytes:
    """Give existing fake routers real defaults now that _configure validates them."""
    bom, source, newline = decode(data)
    content = source.encode()
    tree = ast.parse(source)
    _, position = positions(content)
    edits = []
    for call in ast.walk(tree):
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
                or call.func.attr != "setattr" or len(call.args) < 3
                or not isinstance(call.args[1], ast.Constant) or call.args[1].value != "_corpus"
                or not isinstance(call.args[0], ast.Attribute) or call.args[0].attr != "old"
                or not isinstance(call.args[0].value, ast.Name) or call.args[0].value.id != "fixed"):
            continue
        factory = call.args[2]
        if not isinstance(factory, ast.Lambda) or not isinstance(factory.body, ast.Tuple) or len(factory.body.elts) != 3:
            continue  # Keep other, nonempty test implementations untouched.
        router = factory.body.elts[1]
        if (isinstance(router, ast.Call) and isinstance(router.func, ast.Name)
                and router.func.id in {"object", "SimpleNamespace"} and not router.args and not router.keywords):
            edits.append((position(router), position(router, True), b"_review_fixed_router_stub()"))
    if not edits:
        return data
    if any(isinstance(n, ast.FunctionDef) and n.name == "_review_fixed_router_stub" for n in tree.body):
        raise ValueError("Existing pipeline fixture helper is ambiguous")
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not imports:
        raise ValueError("Expected pipeline test imports")
    starts, _ = positions(content)
    last = imports[-1]
    after = starts[last.end_lineno - 1] + len(content.splitlines(keepends=True)[last.end_lineno - 1])
    helper = '''

def _review_fixed_router_stub():
    # This test double represents the actual fixed factory, not a setting override.
    from src.retrieval.context_search_settings import ContextSearchSettings
    return SimpleNamespace(**{"_" + name: value
                              for name, value in ContextSearchSettings().router_arguments().items()})
'''.replace("\n", newline).encode()
    edits.append((after, after, helper))
    for start, end, replacement in sorted(edits, reverse=True):
        content = content[:start] + replacement + content[end:]
    ast.parse(content.decode())
    return bom + content


def safe_file(project: Path, name: str) -> Path:
    target = project / name
    if target.is_symlink() or project.resolve() not in target.resolve().parents:
        raise ValueError(f"Project file must stay inside project without links: {name}")
    return target


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".context_review_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install(project: Path, output: Path, *, apply_env: bool = False) -> dict:
    project = project.resolve()
    output = output.resolve()
    if project not in output.parents or output.relative_to(project).parts[0] != "artifacts":
        raise ValueError("Installation reports/backups must stay inside artifacts")
    operations = {
        "src/backend/api/dependencies.py": edit_factory,
        "src/retrieval/query_analyzer.py": edit_prompt,
        "experiments/namuwiki/evaluate_context_fixed.py": edit_evaluator,
        "experiments/namuwiki/apply_validated_context_settings.py": edit_activation,
        ".env.example": edit_env_example,
        "docs/data_policy.md": edit_policy,
    }
    if apply_env:
        operations[".env"] = edit_env
    if (project / "tests/context/test_context_fixed_pipeline.py").is_file():
        operations["tests/context/test_context_fixed_pipeline.py"] = edit_pipeline_test_stubs
    plans = []
    for name, editor in operations.items():
        path = safe_file(project, name)
        before = path.read_bytes()  # Refuse a missing input; do not invent merged sources.
        after = editor(before)
        if name.endswith(".py"):
            compile(after.decode("utf-8-sig"), name, "exec")
        if after != before:
            plans.append((name, path, before, after))
    # Every edit has been validated before any merged source is changed.
    for name, path, before, after in plans:
        if path.read_bytes() != before:
            raise ValueError(f"Project file changed during installation: {name}")
    backups = output / "installation/backups"
    changed = []
    try:
        for name, path, before, after in plans:
            backup = backups / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(before)
            if path.read_bytes() != before:
                raise ValueError(f"Project file changed during installation: {name}")
            atomic_write(path, after)
            changed.append((path, before, after))
    except Exception:
        for path, before, after in reversed(changed):
            if path.read_bytes() == after:
                atomic_write(path, before)
        raise
    result = dict(schema="context_review_cleanup_install_v1", status="applied",
                  recorded_at=datetime.now(timezone.utc).isoformat(),
                  changed_files=[name for name, _, _, _ in plans],
                  source_hashes={name: dict(before_sha256=digest(before), after_sha256=digest(after))
                                 for name, _, before, after in plans if name != ".env"},
                  environment_keys_updated=list(ENV_DEFAULTS) if apply_env else [],
                  protected="Merged router/backend bodies, credentials, inputs, labels, caches and Qdrant not overwritten.")
    destination = output / "installation/install_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        old = destination.read_bytes()
        history = destination.parent / "history"
        history.mkdir(exist_ok=True)
        (history / (digest(old) + ".json")).write_bytes(old)
    atomic_write(destination, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply-env", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    result = install(project, project / args.output_dir, apply_env=args.apply_env)
    print("[PASS] Targeted review cleanup applied: " + ", ".join(result["changed_files"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
