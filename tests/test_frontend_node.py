"""화면(map.js)의 상태 전이 시험을 표준 검증에 묶는다.

`tests/frontend/*.test.cjs`는 Node로 돈다. 따로 돌려야 하면 잊히므로, 문서에 적힌 한 줄
(`venv/bin/python -m pytest tests/ -q`)이 함께 돌리게 한다. Node가 없으면 **건너뛰지 않고 실패한다** —
건너뛰면 화면 회귀가 초록불 뒤에 숨는다. Node.js는 이미 설치 요건이다(yt-dlp의 JS 엔진).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_TESTS = ROOT / "tests" / "frontend"


def test_frontend_node_tests_pass() -> None:
    files = sorted(FRONTEND_TESTS.glob("*.test.cjs"))
    assert files, f"{FRONTEND_TESTS}에 Node 시험 파일이 없다"
    node = shutil.which("node")
    assert node, (
        "Node.js를 찾지 못해 화면 시험을 돌리지 못했다 — Node 18 이상을 설치한다. "
        f"직접 돌리려면: node --test {' '.join(str(f.relative_to(ROOT)) for f in files)}"
    )
    done = subprocess.run(
        [node, "--test", *(str(f) for f in files)],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert done.returncode == 0, f"화면 시험 실패\n{done.stdout[-6000:]}\n{done.stderr[-2000:]}"
    # 파일을 못 찾거나 시험이 0건이어도 0으로 끝나는 일이 없게
    assert "# fail 0" in done.stdout and "# pass 0" not in done.stdout, done.stdout[-2000:]
