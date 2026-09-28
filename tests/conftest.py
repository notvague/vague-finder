"""테스트 전역 기본값.

**테스트는 모델을 올리지 않는다.** `lifespan`을 타는 테스트가 몇 개 있는데, 거기서
예열이 켜지면 백그라운드 스레드가 KoE5·SigLIP2·CLAP·CE를 내려받아 올리고 로컬
Qdrant 저장 폴더까지 연다. 개발 서버가 떠 있는 채로 테스트를 돌리면 그 폴더를 두
프로세스가 여는 셈이라 한쪽이 죽는다.

`setdefault`이므로 일부러 켜서 돌리는 것은 그대로 된다.
"""

import os

os.environ.setdefault("SEARCH_WARMUP", "0")
