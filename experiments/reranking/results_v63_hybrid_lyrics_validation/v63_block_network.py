"""pytest 플러그인 — 루프백이 아닌 네트워크 연결을 거부한다(테스트 중 외부 호출 차단 확인용, v63).

    PYTHONPATH=experiments/reranking/results_v63_hybrid_lyrics_validation venv/bin/python -m pytest -p v63_block_network tests/ -q

차단된 시도는 끝에 개수와 대상(호스트:포트)을 출력한다. 루프백(127.0.0.1·::1·localhost)과 유닉스 소켓은 허용한다.
"""
from __future__ import annotations

import socket

BLOCKED: list[str] = []
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection
_orig_getaddrinfo = socket.getaddrinfo
LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def _is_local(address) -> bool:
    if isinstance(address, (str, bytes)):
        return True  # 유닉스 소켓
    host = address[0] if address else ""
    return str(host) in LOOPBACK or str(host).startswith("127.")


class ExternalNetworkBlocked(OSError):
    pass


def _guard(address):
    if not _is_local(address):
        BLOCKED.append(f"{address[0]}:{address[1] if len(address) > 1 else ''}")
        raise ExternalNetworkBlocked(f"v63: 외부 네트워크 연결 차단 {address[0]}")


def connect(self, address):
    _guard(address)
    return _orig_connect(self, address)


def connect_ex(self, address):
    _guard(address)
    return _orig_connect_ex(self, address)


def create_connection(address, *args, **kwargs):
    _guard(address)
    return _orig_create_connection(address, *args, **kwargs)


def getaddrinfo(host, *args, **kwargs):
    if host is not None and str(host) not in LOOPBACK and not str(host).startswith("127."):
        BLOCKED.append(f"dns:{host}")
        raise ExternalNetworkBlocked(f"v63: 외부 이름 조회 차단 {host}")
    return _orig_getaddrinfo(host, *args, **kwargs)


def pytest_configure(config):
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"[v63_block_network] 차단된 외부 연결 시도 {len(BLOCKED)}건: {sorted(set(BLOCKED))[:20]}")
