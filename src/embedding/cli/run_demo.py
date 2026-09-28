"""
cli/run_demo.py
- cli 데모 실행 스크립트
*docker 컨테이너 안에서 해당 임베딩 파트만 따로 실행하고 싶은 경우*
docker exec -w /app vague-finder-backend sh -lc "python -m src.embedding.cli.run_demo > embedding_debug.txt"
-> 텍스트 파일로 임베딩파트 테스트 (전체)

docker exec -w /app vague-finder-backend sh -lc "export EMBED_IMAGE=0 EMBED_AUDIO=0 && python -m src.embedding.cli.run_demo > text_only.txt"
-> 텍스트만 임베딩

docker exec -w /app vague-finder-backend sh -lc "export EMBED_TEXT=0 EMBED_AUDIO=0 && python -m src.embedding.cli.run_demo > image_only.txt"
-> 이미지만 임베딩

docker exec -w /app vague-finder-backend sh -lc "export EMBED_TEXT=0 EMBED_IMAGE=0 && python -m src.embedding.cli.run_demo > audio_only.txt"
-> 오디오만 임베딩

------------------------------------------------------재임베딩------------------------------------------------------
docker exec -w /app vague-finder-backend sh -lc "python -m src.embedding.cli.run_demo --force > embedding_debug.txt"
-> 텍스트 파일로 임베딩파트 테스트 (전체)

docker exec -w /app vague-finder-backend sh -lc "export EMBED_IMAGE=0 EMBED_AUDIO=0 && python -m src.embedding.cli.run_demo --force > text_only.txt"
-> 텍스트만 임베딩

docker exec -w /app vague-finder-backend sh -lc "export EMBED_TEXT=0 EMBED_AUDIO=0 && python -m src.embedding.cli.run_demo --force > image_only.txt"
-> 이미지만 임베딩

docker exec -w /app vague-finder-backend sh -lc "export EMBED_TEXT=0 EMBED_IMAGE=0 && python -m src.embedding.cli.run_demo --force > audio_only.txt"
-> 오디오만 임베딩
---------------------------------------------------------------------------------------------------------------------------

[변경사항]
- 기존: fixtures/demo_songs.py(하드코딩 5곡)
- 변경: data/raw/*/meta.json을 모두 로드해서 임베딩 (곡 개수 제한 없음)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# 크롤러(main.py)와 같은 자리에서 .env를 읽는다. 이것이 없으면 VAGUEFINDER_DATA_DIR을
# .env에만 적어 둔 환경에서 임베딩만 기본값(data/raw)을 보고 "0곡"으로 조용히 끝난다.
# 이미 설정된 환경변수는 덮어쓰지 않는다(테스트가 monkeypatch로 지정한 값이 이긴다).
load_dotenv()

from src.embedding.fixtures.data_songs import (
    SongLoadReport,
    load_songs_with_report,
    resolve_raw_dir,
)
from src.embedding.text.passage_builder import build_dense_passage, build_sparse_passage
from src.embedding.eval.similarity_report import (
    print_embedding_similarities,
    print_embedding_values,
)
from src.embedding.pipelines.embed_artifacts import (
    embed_text_dense,
    embed_text_sparse_bm25,
    embed_image,
    embed_audio,
)
from src.embedding.embedding_cache import (
    EmbeddingCacheKey,
    load_npy,
    purge_track_artifacts,
)
from src.embedding.text.korean_bm25_tokenizer import normalize_for_bm25
from src.embedding.models.text_koe5 import DEFAULT_KOE5_MODEL
from src.embedding.models.image_siglip2 import DEFAULT_SIGLIP2_CKPT
from src.embedding.models.audio_clap import DEFAULT_CLAP_CKPT

def _debug_print_passages(song: Dict, dense: str, sparse: str) -> None:
    md = song.get("metadata", {})
    artist = md.get("artist", "N/A")
    title = md.get("title", "N/A")

    # BM25에 실제로 전달되는 형태소 정규화 결과
    normalized_sparse = normalize_for_bm25(sparse)

    print("\n" + "=" * 70)
    print(f"[DEBUG] Build *dense* passage for id: {song.get('id', 'N/A')} | artist: {artist} | title: {title}")
    print("\n- passage preview:\n" + dense)

    print("\n" + "=" * 70)
    print(f"[DEBUG] Build *sparse* passage for id: {song.get('id', 'N/A')} | artist: {artist} | title: {title}")
    
    print(
        "\n- raw sparse passage:\n"
        + sparse
    )

    print(
        "\n- normalized sparse passage "
        "(actual BM25 input):\n"
        + normalized_sparse
    )

    print("\n" + "=" * 70)


def _project_root() -> Path:
    # .../Vague-Finder/src/embedding/cli/run_demo.py -> 부모 3번 올라가면 프로젝트 루트
    return Path(__file__).resolve().parents[3]


def _env_flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() not in {"0", "false", "no", "n", "off"}


def _safe_tag(s: str) -> str:
    # 캐시 폴더명으로 쓰기 안전하게 정규화
    return (
        s.replace("/", "__")
         .replace(":", "_")
         .replace("\\", "__")
         .replace(" ", "_")
    )

def _print_validation_report(
    report: SongLoadReport,
    *,
    detail_limit: int,
) -> None:
    """
    embedding_debug.txt에 검증 결과를 출력한다.
    """
    print("\n" + "=" * 70)

    print(
        "[META_VALIDATION_SUMMARY] "
        f"scanned={report.scanned_count} "
        f"passed={report.accepted_count} "
        f"rejected={report.rejected_count} "
        f"moved_to_failed_raw={report.moved_count} "
        f"not_moved={report.move_failed_count}"
    )

    print(
        "[META_VALIDATION_SUMMARY] "
        f"raw_dir={report.raw_dir}"
    )

    print(
        "[META_VALIDATION_SUMMARY] "
        f"failed_raw_dir={report.failed_raw_dir}"
    )

    if report.audit_report_path is not None:
        print(
            "[META_VALIDATION_SUMMARY] "
            f"full_report={report.audit_report_path}"
        )

    limit = max(
        0,
        detail_limit,
    )

    for index, rejected in enumerate(
        report.rejected[:limit],
        start=1,
    ):
        destination = (
            str(rejected.destination_dir)
            if rejected.destination_dir
            else "-"
        )

        print(
            f"[META_VALIDATION_REJECTED "
            f"{index}/{report.rejected_count}] "
            f"id={rejected.song_id or 'N/A'} "
            f"folder={rejected.source_dir.name} "
            f"moved={rejected.moved} "
            f"destination={destination}"
        )

        if rejected.move_error:
            print(
                f"  - <move>: "
                f"{rejected.move_error}"
            )

        for issue in rejected.issues:
            value_suffix = (
                f" | value={issue.value_preview}"
                if issue.value_preview
                else ""
            )

            print(
                f"  - {issue.path}: "
                f"{issue.reason}"
                f"{value_suffix}"
            )

    remaining = (
        report.rejected_count
        - min(
            report.rejected_count,
            limit,
        )
    )

    if remaining > 0:
        print(
            "[META_VALIDATION_REJECTED] "
            f"detail omitted={remaining}; "
            "전체 사유는 "
            "validation_report.jsonl 확인"
        )

    for error in report.audit_write_errors:
        print(
            "[META_VALIDATION_WARN] "
            "validation_report.jsonl "
            f"기록 실패: {error}"
        )

    print("=" * 70)


def _print_embedding_summary(
    songs: List[Dict],
    *,
    do_text: bool,
    do_image: bool,
    do_audio: bool,
) -> None:
    """
    실제 생성된 임베딩 개수를 출력한다.
    """
    total = len(songs)
    enabled_keys: list[str] = []

    print("\n" + "=" * 70)

    print(
        "[EMBEDDING_RESULT_SUMMARY] "
        f"eligible_songs={total}"
    )

    if do_text:
        dense_count = sum(
            song.get("text_dense_values") is not None
            for song in songs
        )

        sparse_count = sum(
            song.get("text_sparse_values") is not None
            for song in songs
        )

        print(
            "[EMBEDDING_RESULT_SUMMARY] "
            f"text_dense_success={dense_count}/{total}"
        )

        print(
            "[EMBEDDING_RESULT_SUMMARY] "
            f"text_sparse_success={sparse_count}/{total}"
        )

        enabled_keys.extend(
            (
                "text_dense_values",
                "text_sparse_values",
            )
        )

    else:
        print(
            "[EMBEDDING_RESULT_SUMMARY] "
            "text=disabled"
        )

    if do_image:
        image_count = sum(
            song.get("image_values") is not None
            for song in songs
        )

        print(
            "[EMBEDDING_RESULT_SUMMARY] "
            f"image_success={image_count}/{total}"
        )

        enabled_keys.append(
            "image_values"
        )

    else:
        print(
            "[EMBEDDING_RESULT_SUMMARY] "
            "image=disabled"
        )

    if do_audio:
        audio_count = sum(
            song.get("audio_values") is not None
            for song in songs
        )

        print(
            "[EMBEDDING_RESULT_SUMMARY] "
            f"audio_success={audio_count}/{total}"
        )

        enabled_keys.append(
            "audio_values"
        )

    else:
        print(
            "[EMBEDDING_RESULT_SUMMARY] "
            "audio=disabled"
        )

    if enabled_keys:
        fully_successful = sum(
            all(
                song.get(key) is not None
                for key in enabled_keys
            )
            for song in songs
        )

        print(
            "[EMBEDDING_RESULT_SUMMARY] "
            "all_enabled_modalities_success="
            f"{fully_successful}/{total}"
        )
    else:
        print(
            "[EMBEDDING_RESULT_SUMMARY] "
            "all_enabled_modalities_success=not_applicable"
        )

    print("=" * 70)


def _default_tags() -> dict:
    # zip의 최신 코드에 박혀있는 기본 모델 ID 기반 자동 태그
    return {
        "text_dense": _safe_tag(DEFAULT_KOE5_MODEL),
        "text_sparse": "bm25",
        "image": _safe_tag(DEFAULT_SIGLIP2_CKPT),
        "audio": _safe_tag(DEFAULT_CLAP_CKPT),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="캐시 있어도 전부 재임베딩")
    return p.parse_args()


def _load_cached_into_songs(
    songs: List[Dict],
    *,
    artifacts_dir: Path,
    td_tag: str | None,
    img_tag: str | None,
    aud_tag: str | None,
) -> None:
    """
    artifacts에 저장된 벡터를 로드해서,
    similarity_report.py가 기대하는 키로 songs에 주입한다.

    매핑:
      - text_dense  -> s["text_dense_values"]
      - image       -> s["image_values"]
      - audio       -> s["audio_values"]
    """
    for s in songs:
        tid = str(s["id"])

        if td_tag:
            s["text_dense_values"] = load_npy(artifacts_dir, EmbeddingCacheKey("text_dense", td_tag, tid))
        if img_tag:
            s["image_values"] = load_npy(artifacts_dir, EmbeddingCacheKey("image", img_tag, tid))
        if aud_tag:
            s["audio_values"] = load_npy(artifacts_dir, EmbeddingCacheKey("audio", aud_tag, tid))


def main() -> None:
    args = _parse_args()

    root = _project_root()
    artifacts_dir = root / "artifacts"
    # VAGUEFINDER_DATA_DIR은 크롤러와 같은 뜻이다 — 곡 폴더를 직접 담은 폴더.
    # 해석은 data_songs에 한곳으로 모아 두었다(resolve_raw_dir 주석 참고).
    data_raw_dir = resolve_raw_dir()

    validation_enabled = _env_flag(
        "META_VALIDATION",
        True,
    )

    move_invalid = _env_flag(
        "META_MOVE_INVALID",
        True,
    )

    require_media_files = _env_flag(
        "META_REQUIRE_MEDIA_FILES",
        True,
    )

    validation_detail_limit = int(
        os.getenv(
            "META_VALIDATION_DETAIL_LIMIT",
            "20",
        )
    )

    load_result = load_songs_with_report(
        data_raw_dir,
        validate_meta=validation_enabled,
        move_invalid=move_invalid,
        require_media_files=require_media_files,
    )

    songs = load_result.songs
    validation_report = load_result.report

    _print_validation_report(
        validation_report,
        detail_limit=validation_detail_limit,
    )

    accepted_ids = {
        str(song["id"])
        for song in songs
    }

    removed_artifacts = []

    # META_MOVE_INVALID=0은 사전 점검 모드로 사용할 수 있으므로
    # 이때는 실제 캐시도 삭제하지 않는다.
    if move_invalid:
        removed_artifacts = purge_track_artifacts(
            artifacts_dir,
            rejected_ids=validation_report.rejected_ids,
            accepted_ids=accepted_ids,
        )

    for artifact_path in removed_artifacts:
        print(
            "[META_VALIDATION_CACHE_CLEAN] "
            f"removed={artifact_path}"
        )

    print(
        "[META_VALIDATION_SUMMARY] "
        "stale_embedding_artifacts_removed="
        f"{len(removed_artifacts)}"
    )

    print(
        f"[INFO] Loaded {len(songs)} "
        f"validated songs from: {data_raw_dir}"
    )

    if not songs:
        print(
            "[WARN] 검증을 통과한 곡이 없어 "
            "임베딩을 시작하지 않습니다."
        )
        return

    # --------------------------------------------------
    # (1) Passage 디버그 출력
    # --------------------------------------------------
    preview_n = int(os.getenv("PASSAGE_PREVIEW_N", "5"))
    for s in songs[:preview_n]:
        dense = build_dense_passage(s)
        sparse = build_sparse_passage(s)
        _debug_print_passages(s, dense, sparse)

    # --------------------------------------------------
    # (2) 임베딩 생성 (artifacts 캐시 기반)
    # --------------------------------------------------
    do_text = _env_flag("EMBED_TEXT", True)
    do_image = _env_flag("EMBED_IMAGE", True)
    do_audio = _env_flag("EMBED_AUDIO", True)

    defaults = _default_tags()
    text_dense_tag = os.getenv("TEXT_DENSE_MODEL_TAG", defaults["text_dense"])
    text_sparse_tag = os.getenv("TEXT_SPARSE_MODEL_TAG", defaults["text_sparse"])
    image_tag = os.getenv("IMAGE_MODEL_TAG", defaults["image"])
    audio_tag = os.getenv("AUDIO_MODEL_TAG", defaults["audio"])

    if do_text:
        embed_text_dense(songs, artifacts_dir=artifacts_dir, model_tag=text_dense_tag, force=args.force)
        embed_text_sparse_bm25(
            songs,
            artifacts_dir=artifacts_dir,
            model_tag=text_sparse_tag,
        )

    if do_image:
        embed_image(songs, artifacts_dir=artifacts_dir, model_tag=image_tag, force=args.force)

    effective_audio_tag = None
    if do_audio:
        mean_center = _env_flag("AUDIO_MEAN_CENTER", True)
        window_seconds = int(os.getenv("AUDIO_WINDOW_SECONDS", "120"))
        segment_seconds = int(os.getenv("AUDIO_SEGMENT_SECONDS", "12"))
        num_segments = int(os.getenv("AUDIO_NUM_SEGMENTS", "8"))

        effective_audio_tag = embed_audio(
            songs,
            data_raw_dir=data_raw_dir,
            artifacts_dir=artifacts_dir,
            model_tag=audio_tag,
            force=args.force,
            window_seconds=window_seconds,
            segment_seconds=segment_seconds,
            num_segments=num_segments,
            mean_center=mean_center,
        )

    # --------------------------------------------------
    # (3) artifacts에서 다시 로드해서 songs에 주입
    # --------------------------------------------------
    _load_cached_into_songs(
        songs,
        artifacts_dir=artifacts_dir,
        td_tag=text_dense_tag if do_text else None,
        img_tag=image_tag if do_image else None,
        aud_tag=effective_audio_tag if do_audio else None,
    )

    _print_embedding_summary(
        songs,
        do_text=do_text,
        do_image=do_image,
        do_audio=do_audio,
    )

    if do_text or do_image or do_audio:
        # (A) 전체 곡 벡터 일부 출력
        print_embedding_values(
            songs,
            max_songs=300,
        )

        # (B) 유사도 출력 (앞쪽 n개만)
        print_embedding_similarities(
            songs[:10]
        )
    else:
        print(
            "[INFO] Validation-only run completed. "
            "Embedding value/similarity output skipped."
        )

if __name__ == "__main__":
    main()