"""운영 `.env`에서 **지정한 키만** 꺼내 환경으로 넘기고 명령을 실행한다(값은 출력하지 않는다).

워크트리에 `.env`를 복사하지 않기 위한 통로다. 키 목록 밖의 값(Pinecone·HF 토큰 등)은 넘기지 않는다.

    venv/bin/python with_env.py MONGO_URI,MONGO_DB_NAME -- venv/bin/python snapshot_lyrics.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MAIN_ENV = Path("/Users/ozzing727/Desktop/Developer/졸작-RISE/vague-finder/.env")


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
