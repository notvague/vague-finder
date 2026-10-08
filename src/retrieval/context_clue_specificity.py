"""Keep a qualified, unnamed media memory out of a broad OST lookup.

Only the source span already selected by the Context boundary rules is used.
This is a specificity safeguard, not entity inference or an extra ranking vote.
"""

from __future__ import annotations

from types import SimpleNamespace


def preserve_context_media_search(
    model_search: str, *, source_query: str, relation: str, target: str = "",
) -> str:
    """Reject a rewrite that turns a detailed memory into category enumeration.

    Named works retain their focused lookup. Truly broad user requests remain
    valid, and informative grounded rewrites remain unchanged. The caller already
    checks grounding and keeps separate lyric, artwork and waveform fields.
    """
    # A local import avoids a module-initialization cycle: context_query invokes
    # this helper only after its normal source/grounding checks have succeeded.
    from src.retrieval.context_query import (
        context_media_description_requires_support, has_specific_media_target,
        is_media_usage_relation,
    )

    source = SimpleNamespace(target=target, relation=relation, search_query=source_query)
    rewrite = SimpleNamespace(target=target, relation=relation, search_query=model_search)
    if not is_media_usage_relation(relation):
        return model_search
    # A grounded work may use a short focused query, but its name must still
    # occur there. Keeping only "OST" would make the primary lookup generic.
    if has_specific_media_target(source) and not has_specific_media_target(rewrite):
        return source_query
    if (context_media_description_requires_support(source)
            and not context_media_description_requires_support(rewrite)):
        return source_query
    return model_search
