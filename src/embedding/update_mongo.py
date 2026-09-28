from src.common.mongodb import get_collection

songs_col = get_collection("songs")


def mark_audio_embedded(song_id: str, embedding_path: str) -> None:
    """
    song_id 또는 기존 파이프라인이 넘기는 raw 폴더명으로 문서를 찾아 갱신한다.
    """
    key = str(song_id)
    result = songs_col.update_one(
        {
            "$or": [
                {"_id": key},
                {"song_id": key},
                {"folder_name": key},
            ]
        },
        {
            "$set": {
                "paths.audio_embedding": embedding_path,
                "status.audio_embedded": True,
            }
        },
    )

    if result.matched_count == 0:
        print(f"[MONGO][WARN] 업데이트할 곡을 찾지 못함: {key}")
