"""Exact Unicode code-point offsets, always computed by code, never by an LLM."""
from .schemas import EvidenceSpan, ParsedDocument, Quote


def source_text(document: ParsedDocument, block_id: str, allowed: set[str]) -> str:
    for block in document.blocks:
        if block.block_id == block_id:
            if block_id not in allowed:
                raise ValueError("evidence_outside_target_scope")
            return block.source_text
    if block_id.startswith("footnote:"):
        identity = block_id[len("footnote:"):]
        references = {ref for block in document.blocks if block.block_id in allowed for ref in block.footnote_refs}
        if identity in references:
            for footnote in document.footnotes:
                if footnote.footnote_id == identity:
                    return footnote.source_text
    raise ValueError("unknown_or_unlinked_evidence_block")


def align(document: ParsedDocument, quote: Quote, allowed: set[str]) -> EvidenceSpan:
    text = source_text(document, quote.block_id, allowed)
    start = text.find(quote.quote)
    if start < 0:
        raise ValueError("quote_not_in_source")
    if text.find(quote.quote, start + 1) >= 0:
        raise ValueError("quote_occurs_multiple_times_use_longer_quote")
    return EvidenceSpan(snapshot_id=document.snapshot_id, parser_version=document.parser_version,
                        block_id=quote.block_id, quote=quote.quote, start=start, end=start + len(quote.quote))


def verify_span(document: ParsedDocument, span: EvidenceSpan, allowed: set[str]) -> None:
    if span.snapshot_id != document.snapshot_id or span.parser_version != document.parser_version:
        raise ValueError("stale_evidence_version")
    actual = align(document, Quote(block_id=span.block_id, quote=span.quote), allowed)
    if (actual.start, actual.end) != (span.start, span.end):
        raise ValueError("evidence_offset_mismatch")
