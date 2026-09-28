"""
tests/test_embedding_paths.py

VAGUEFINDER_DATA_DIR의 의미를 크롤러와 임베딩이 같게 보는지 고정한다.

배경 (2026-09-19 점검):

크롤러는 이 값을 **곡 폴더를 직접 담은 폴더**로 본다(.../vague-finder/raw5).
임베딩만 `<그 값>/raw`를 봤다. 팀 드라이브에는 옛 배치 `raw`(341곡)가 실제로 남아 있어서,
같은 환경변수로 크롤러는 raw5에 쌓고 임베딩은 raw를 읽는 상태가 만들어졌다.
한쪽이 조용히 엉뚱한 데이터를 쓰는 종류의 어긋남이라 테스트로 못 박는다.

실행:
    venv/bin/python -m pytest tests/test_embedding_paths.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

from src.crawler.scripts_py import crawl_state
from src.embedding.fixtures.data_songs import (
    DEFAULT_RAW_DIR,
    load_songs_with_report,
    resolve_failed_raw_dir,
    resolve_raw_dir,
)


def make_song(raw_dir: Path, song_id: str) -> Path:
    """검증을 통과하는 최소 곡 폴더."""
    folder = raw_dir / f"가수_곡{song_id}_{song_id}"
    folder.mkdir(parents=True)
    (folder / crawl_state.META_FILE).write_text(
        json.dumps({"song_id": song_id, "metadata": {"title": f"곡{song_id}"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / crawl_state.COVER_FILE).write_bytes(b"\xff\xd8\xff" + b"x" * 200)
    (folder / crawl_state.AUDIO_FILE).write_bytes(b"\x00\x00\x00\x20ftypM4A " + b"y" * 200)
    return folder


def test_data_dir_means_the_song_folder_container(tmp_path, monkeypatch) -> None:
    """VAGUEFINDER_DATA_DIR은 곡 폴더를 **직접** 담은 폴더다 — 크롤러와 같은 뜻.

    회귀 이력: 임베딩만 `<그 값>/raw`를 봤다. 드라이브에 옛 배치 raw(341곡)가 있어서
    크롤러는 raw5에 쌓고 임베딩은 raw를 읽었다.
    """
    songs_dir = tmp_path / "raw5"
    songs_dir.mkdir()
    make_song(songs_dir, "100")
    # 함정: 한 단계 아래 raw에 옛 배치가 있다. 이것을 읽으면 안 된다.
    make_song(songs_dir / "raw", "999")

    monkeypatch.setenv("VAGUEFINDER_DATA_DIR", str(songs_dir))
    assert resolve_raw_dir() == songs_dir

    result = load_songs_with_report(validate_meta=False, move_invalid=False,
                                    require_media_files=True)
    assert [s["id"] for s in result.songs] == ["100"]
    assert result.report.raw_dir == songs_dir


def test_default_is_the_repo_data_raw_when_unset(tmp_path, monkeypatch) -> None:
    """환경변수가 없으면 저장소 안 data/raw를 본다(예전 기본값과 같다)."""
    monkeypatch.delenv("VAGUEFINDER_DATA_DIR", raising=False)
    assert resolve_raw_dir() == DEFAULT_RAW_DIR
    assert DEFAULT_RAW_DIR.name == "raw"
    assert DEFAULT_RAW_DIR.parent.name == "data"


def test_failed_dir_sits_next_to_the_collection(tmp_path, monkeypatch) -> None:
    """검증 실패 폴더는 수집 폴더 옆에 `<이름>_failed`로 생긴다.

    크롤러가 격리 폴더를 `<이름>_incomplete`로 짓는 것과 같은 방식이라, 어느 수집분에서
    나온 것인지 헷갈리지 않는다.
    """
    monkeypatch.delenv("VAGUEFINDER_FAILED_DIR", raising=False)
    assert resolve_failed_raw_dir(Path("/drive/vague-finder/raw5")) == Path(
        "/drive/vague-finder/raw5_failed"
    )
    # 크롤러의 격리 폴더 규칙과 같은 모양인지 같이 확인한다
    data_dir = Path("/drive/vague-finder/raw5")
    assert data_dir.with_name(data_dir.name + "_incomplete").name == "raw5_incomplete"

    monkeypatch.setenv("VAGUEFINDER_FAILED_DIR", str(tmp_path / "따로"))
    assert resolve_failed_raw_dir(Path("/drive/raw5")) == tmp_path / "따로"


def test_crawler_and_embedding_agree_on_the_same_value(tmp_path, monkeypatch) -> None:
    """같은 환경변수로 크롤러와 임베딩이 같은 폴더를 가리킨다."""
    songs_dir = tmp_path / "raw5"
    songs_dir.mkdir()
    make_song(songs_dir, "100")
    monkeypatch.setenv("VAGUEFINDER_DATA_DIR", str(songs_dir))

    # 크롤러 쪽: 완료 판정이 이 폴더를 직접 훑는다
    assert set(crawl_state.scan_complete(songs_dir)) == {"100"}
    # 임베딩 쪽: 같은 폴더를 본다
    assert resolve_raw_dir() == songs_dir


def test_embedding_cli_loads_dotenv_like_the_crawler() -> None:
    """임베딩 CLI도 크롤러처럼 .env를 읽는다.

    회귀 이력: 크롤러(main.py)는 모듈 상단에서 load_dotenv()를 부르는데 임베딩 CLI는 부르지
    않았다. VAGUEFINDER_DATA_DIR을 .env에만 적어 둔 환경에서 임베딩만 기본값(data/raw)을 보고
    "0곡 로드"로 조용히 끝났다. 같은 환경변수인데 한쪽만 읽으면 경로 의미를 맞춰도 소용없다.

    한계: 이 테스트는 호출이 **있는지**만 본다. .env 탐색은 호출한 파일 위치에서 위로
    올라가는 방식이라 임시 디렉터리로 바꿔치울 수 없어서, 실제 로딩까지는 확인하지 못한다.
    (tests/test_qdrant_backend.py의 백엔드 선택 테스트와 같은 방식이다.)
    """
    import ast
    import inspect

    from src.crawler import main as crawler_main
    from src.embedding.cli import run_demo

    def calls_load_dotenv_at_module_level(module) -> bool:
        tree = ast.parse(inspect.getsource(module))
        for node in tree.body:  # 모듈 최상단만 본다
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "load_dotenv":
                    return True
        return False

    assert calls_load_dotenv_at_module_level(crawler_main), "크롤러가 .env를 읽지 않는다"
    assert calls_load_dotenv_at_module_level(run_demo), "임베딩 CLI가 .env를 읽지 않는다"
