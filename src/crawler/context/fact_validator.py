"""Source alignment is necessary, not sufficient: v1 never auto-accepts facts."""
from .document_parser import compact
from .evidence import align, verify_span
from .schemas import Analysis, Fact, FactCandidate, FactContent, ParsedDocument, digest


def validate_candidate(candidate: FactCandidate, analysis: Analysis, document: ParsedDocument,
                       extractor_key: str, chunk: dict) -> Fact:
    if analysis.binding.status != "verified":
        raise ValueError("unverified_binding")
    allowed = {item.block_id for item in analysis.selection if item.disposition == "selected"}
    allowed &= set(analysis.binding.target_block_ids)
    spans = [align(document, quote, allowed) for quote in candidate.evidence]
    main_spans = [span for span in spans if span.block_id in allowed]
    if not main_spans:
        raise ValueError("fact_requires_main_text_evidence")
    for span in spans:
        if span.block_id.startswith("footnote:"):
            if span.block_id not in {note["block_id"] for note in chunk["footnotes"]}:
                raise ValueError("footnote_not_supplied_to_extractor")
        elif not any(unit["block_id"] == span.block_id and unit["start"] <= span.start and span.end <= unit["end"]
                     for unit in chunk["units"]):
            raise ValueError("quote_outside_supplied_fragment")
    content = {key: getattr(candidate, key) for key in FactContent.model_fields}
    serial = FactContent(**content).model_dump(exclude_none=True)
    content_hash = digest(serial)
    # Stable across repeat chunks and prose-equivalent snapshots, but separate per binding.
    fact_id = digest([analysis.seed.song_id, analysis.binding.binding_id, content_hash])
    reasons = []
    if not any(compact(title) in compact(candidate.statement) for title in [analysis.seed.title, *analysis.seed.title_aliases]):
        reasons.append("statement_missing_target_title")
    if candidate.scope != "target_recording":
        reasons.append("work_or_unresolved_scope")
    if candidate.claim_mode != "asserted":
        reasons.append("reported_or_uncertain_claim")
    if candidate.polarity == "negated":
        reasons.append("negative_relation_not_positive_search_evidence")
    if analysis.binding.recording_variant == "unknown":
        reasons.append("recording_variant_unknown")
    return Fact(**content, fact_id=fact_id, song_id=analysis.seed.song_id,
                binding_id=analysis.binding.binding_id, evidence_spans=spans,
                extractor_version=extractor_key, content_hash=content_hash,
                review_status="needs_review" if reasons else "candidate",
                rejection_reason="; ".join(reasons) if reasons else None)


def revalidate_fact(fact: Fact, analysis: Analysis, document: ParsedDocument) -> None:
    if (fact.song_id != analysis.seed.song_id or fact.binding_id != analysis.binding.binding_id
            or analysis.binding.status != "verified"):
        raise ValueError("fact_binding_mismatch")
    if digest(FactContent.model_validate({key: getattr(fact, key) for key in FactContent.model_fields})
              .model_dump(exclude_none=True)) != fact.content_hash:
        raise ValueError("fact_content_hash_mismatch")
    if fact.fact_id != digest([fact.song_id, fact.binding_id, fact.content_hash]):
        raise ValueError("fact_id_mismatch")
    allowed = {item.block_id for item in analysis.selection if item.disposition == "selected"}
    allowed &= set(analysis.binding.target_block_ids)
    if not any(span.block_id in allowed for span in fact.evidence_spans):
        raise ValueError("missing_main_text_evidence")
    for span in fact.evidence_spans:
        verify_span(document, span, allowed)


def search_eligible(fact: Fact, analysis: Analysis) -> bool:
    return (fact.review_status == "accepted" and bool(fact.reviewer) and bool(fact.review_reason)
            and fact.scope == "target_recording" and fact.polarity == "affirmed"
            and fact.claim_mode == "asserted" and fact.source_verification != "disputed"
            and analysis.binding.recording_variant != "unknown"
            and any(compact(title) in compact(fact.statement) for title in [analysis.seed.title, *analysis.seed.title_aliases]))
