"""공개 `/health`에 싣는 피드백 준비 상태. **상태 코드·건수·시각만** 낸다.

원질의·사유·재사용 문장, 예외 본문, Mongo·Qdrant 연결 정보는 여기로 오지 않는다. 저장소를 새로 부르지도
않는다 — 기동할 때와 색인 작업이 주기마다 확인해 둔 값을 읽을 뿐이라, 꺼져 있으면 연결이 생기지 않는다.
"""
from __future__ import annotations

from typing import Any, Dict

from src.feedback import worker
from src.feedback.settings import index_worker_enabled, retrieval_enabled
from src.feedback.store import collection_enabled, ttl_checked_at, ttl_ready, ttl_report


def feedback_health() -> Dict[str, Any]:
    collecting, working = collection_enabled(), index_worker_enabled()
    checked = collecting or working
    checked_at = ttl_checked_at()
    return {
        "collection": {
            "enabled": collecting,
            # 자동 삭제(TTL)가 준비되지 않았으면 수집을 켰어도 준비 완료가 아니다. 검색은 그대로 된다.
            "ready": collecting and ttl_ready(),
            "ttl": ttl_report() if checked else None,
            # 이 보고를 마지막으로 확인한 시각. 기동할 때, 색인 작업 주기마다, 검색 기록을 남길 때
            # (10분 간격으로 색인 생성 명령을 다시 보내며) 새로 쓴다.
            "ttl_checked_at": checked_at.isoformat() if checked and checked_at else None,
        },
        "index_worker": {"enabled": working, **worker.status()},
        "retrieval": {"enabled": retrieval_enabled()},
    }
