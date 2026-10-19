"""(v65 `with_env.py` 그대로 — v63 이식 — 메인 체크아웃 경로는 실행 때 찾는다) 운영 `.env`에서 **지정한 키만** 꺼내 환경으로 넘기고 명령을 실행한다(값은 출력하지 않는다).

워크트리에 `.env`를 복사하지 않기 위한 통로다. 키 목록 밖의 값(Pinecone·HF 토큰 등)은 넘기지 않는다.

    venv/bin/python with_env.py MONGO_URI,MONGO_DB_NAME -- venv/bin/python snapshot_lyrics.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

def _main_checkout() -> Path:
    """메인 체크아웃 = 이 워크트리의 Git 공용 디렉터리(`.git`)의 부모. 개인 경로를 파일에 적지 않으려고 실행 때 찾는다."""
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    return Path(common).parent


MAIN_ENV = Path(os.environ["V73_MAIN_ENV"]) if os.environ.get("V73_MAIN_ENV") else _main_checkout() / ".env"


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("usage: with_env.py KEY1,KEY2 -- <command...>")
    from dotenv import dotenv_values

    keys = [k.strip() for k in sys.argv[1].split(",") if k.strip()]
    values = dotenv_values(MAIN_ENV)
    env = dict(os.environ)
    missing = []
    for key in keys:
        value = values.get(key)
        if value:
            if key == "GOOGLE_APPLICATION_CREDENTIALS" and not os.path.isabs(value):
                # 운영 .env의 상대 경로는 메인 체크아웃 기준이다(워크트리에는 키 파일이 없다)
                value = str(MAIN_ENV.parent / value)
                if not os.path.exists(value):
                    raise SystemExit("GOOGLE_APPLICATION_CREDENTIALS 파일이 없다")
            env[key] = value
        else:
            missing.append(key)
    if missing:
        raise SystemExit(f"운영 .env에 없는 키: {missing}")
    print(f"[with_env] 넘긴 키: {keys}", file=sys.stderr, flush=True)
    os.execvpe(sys.argv[3], sys.argv[3:], env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
