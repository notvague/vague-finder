"""Complete source coverage with offset-preserving fragments and bounded overlap."""
import json
import re

from .document_parser import section_path
from .schemas import Analysis, ParsedDocument, digest


def fragments(text: str, limit: int) -> list[tuple[int, int]]:
    result, start = [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            candidates = [match.end() for match in re.finditer(r"(?:[.!?。]\s+|\n+)", text[start:end])]
            if candidates:
                end = start + candidates[-1]
            else:
                space = text.rfind(" ", start + limit // 2, end)
                if space > start:
                    end = space + 1
        result.append((start, end))
        start = end
    return result


def build_chunks(analysis: Analysis, document: ParsedDocument, *, chunk_chars=7000) -> list[dict]:
    if chunk_chars < 1000:
        raise ValueError("chunk_chars must be >= 1000")
    selected = {item.block_id for item in analysis.selection if item.disposition == "selected"}
    units = []
    for block in document.blocks:
        if block.block_id not in selected:
            continue
        ranges = fragments(block.source_text, max(200, chunk_chars // 3))
        for start, end in ranges:
            units.append({"block_id": block.block_id, "start": start, "end": end,
                          "text": block.source_text[start:end],
                          "section_path": section_path(document, block.section_id),
                          "fragmented_block": len(ranges) > 1, "footnote_refs": block.footnote_refs})
    chunks, pending = [], []
    size = 0

    def emit(batch):
        footnote_ids = {ref for unit in batch for ref in unit["footnote_refs"]}
        notes = [{"block_id": "footnote:" + note.footnote_id, "text": note.source_text}
                 for note in document.footnotes if note.footnote_id in footnote_ids]
        chunk = {"units": batch, "footnotes": notes}
        chunk["chunk_id"] = digest([analysis.analysis_id, chunk])
        chunk["budget_method"] = "approximate_serialized_characters_then_API_token_check"
        chunks.append(chunk)

    for unit in units:
        unit_size = len(json.dumps(unit, ensure_ascii=False))
        if pending and size + unit_size > chunk_chars:
            emit(pending)
            # One preceding unit retains local co-reference. Same facts are deduplicated later.
            overlap = pending[-1]
            pending = [overlap] if len(pending) > 1 else []
            size = sum(len(json.dumps(item, ensure_ascii=False)) for item in pending)
        pending.append(unit)
        size += unit_size
    if pending:
        emit(pending)
    return chunks
