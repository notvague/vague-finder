"""A changed Context rule must invalidate an otherwise identical analysis cache."""

from pathlib import Path

from src.backend.schemas.query import QueryAnalysis
from src.retrieval.analysis_cache import analyzer_fingerprint, new_cache
from experiments.namuwiki.diagnose_context_step10_after_fix import repaired_analysis


def test_context_safeguards_are_in_analysis_cache_fingerprint(monkeypatch):
    cache = new_cache(source="queries.csv", split="dev")
    original = Path.read_text

    def edited_context(path, *args, **kwargs):
        contents = original(path, *args, **kwargs)
        if path.name == "context_query.py":
            return contents + "\n# changed Context rule\n"
        return contents

    monkeypatch.setattr(Path, "read_text", edited_context)
    assert cache.fingerprint_drift() == ["postprocess_sha"]
    assert analyzer_fingerprint()["postprocess_sha"] != cache.meta["analyzer"]["postprocess_sha"]


def test_repair_keeps_existing_multimodal_fields_and_ignores_cover_art():
    query = "어떤 밴드가 멜론 차트 1위를 했다는 곡이 기억이 안 나"
    old = QueryAnalysis(
        original_query=query, intent_type="mixed", korean_tags=["밴드", "차트"],
        image_english_query="", audio_english_query="", context_clues=[],
    )
    repaired = repaired_analysis(old, query)
    assert repaired.context_clues
    old_fields = old.model_dump(exclude={"context_clues"})
    assert repaired.model_dump(exclude={"context_clues"}) == old_fields

    cover = "멜론 모양의 녹색 그림이 있는 앨범 표지"
    original_cover = old.model_copy(update={"original_query": cover})
    assert repaired_analysis(original_cover, cover).context_clues == []
