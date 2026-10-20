"""v78 코드 사본·manifest 만들기 — 워크트리 `src`를 `artifacts/src_v78_<트리 지문>`으로 복사하고, 기준 커밋 대비 바뀐/새 `.py`
blob을 `inputs/code_manifest.json`에 적는다(`v78_eval_runtime.code_identity`가 실행마다 대조한다). 모델·DB 호출 없음.

    python make_code_copy.py            # 이미 같은 사본이 있으면 확인만, 다르면 거부(덮어쓰지 않는다)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMIT = "3660083038f666aa2b69892e76db7d07ca6d8024"


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> int:
    tree = subprocess.run(["git", "ls-tree", "-r", COMMIT, "src"], capture_output=True, text=True, cwd=ROOT,
                          check=True).stdout.splitlines()
    base = {line.split("\t")[1]: line.split()[2] for line in tree if line.endswith(".py")}
    have = {str(p.relative_to(ROOT)): blob(p.read_bytes()) for p in (ROOT / "src").rglob("*.py")
            if "__pycache__" not in p.parts}
    removed = sorted(set(base) - set(have))
    if removed:
        print(f"기준 커밋에 있는데 워크트리에 없는 파일: {removed}", file=sys.stderr)
        return 2
    changed = {k: v for k, v in sorted(have.items()) if k in base and base[k] != v}
    added = {k: v for k, v in sorted(have.items()) if k not in base}
    want = sorted([(path, changed.get(path, b)) for path, b in base.items()] + sorted(added.items()))
    tree_sha = hashlib.sha256(json.dumps(want).encode()).hexdigest()[:16]
    manifest = {"base_commit": COMMIT, "src_py_tree_sha16": tree_sha, "src_py_files": len(want),
                "changed_blobs": changed, "added_blobs": added}
    dest = ROOT / f"artifacts/src_v78_{tree_sha}"
    if not dest.exists():
        shutil.copytree(ROOT / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__"))
    copied = sorted((str(p.relative_to(dest)), blob(p.read_bytes())) for p in (dest / "src").rglob("*.py")
                    if "__pycache__" not in p.parts)
    if copied != sorted(want):
        print(f"기존 사본이 워크트리 src와 다르다: {dest}", file=sys.stderr)
        return 2
    out = HERE / "inputs/code_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"tree": tree_sha, "changed": list(changed), "added": list(added)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
