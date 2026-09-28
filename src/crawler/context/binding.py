"""Conservative local-identity binding; uncertain pages require a scoped human review."""
import re

from .document_parser import compact, section_path, subtree_blocks
from .evidence import align
from .schemas import Binding, BindingReview, ParsedDocument, Quote, SongSeed, digest


def title_matches(text: str, seed: SongSeed) -> bool:
    value = compact(text)
    aliases = [seed.title, *seed.title_aliases]
    artists = [*seed.artists, *seed.artist_aliases]
    return any(value == compact(title) or any(value == compact(title + artist) for artist in artists)
               for title in aliases)


def local_identity(text: str, seed: SongSeed) -> bool:
    """Only explicit performer→song phrasing in ONE sentence, not page co-occurrence.

    This intentionally abstains on infoboxes, complex prose and collaborative credits.
    Those can be approved through the same exact-evidence review path.
    """
    for sentence in re.split(r"[\n.!?。]", text):
        value = compact(sentence)
        if any(marker in value for marker in ("커버", "리메이크", "라이브", "리믹스", "원곡", "아니다", "아닌")):
            continue
        for title in [seed.title, *seed.title_aliases]:
            for artist in [*seed.artists, *seed.artist_aliases]:
                t, a = re.escape(compact(title)), re.escape(compact(artist))
                patterns = (rf"{a}(?:의곡|의노래|가부른곡|이부른곡){t}",
                            rf"{t}(?:은|는){a}(?:의곡|의노래|가부른곡|이부른곡)")
                if any(re.search(pattern, value) for pattern in patterns):
                    return True
    return False


def bind_song(seed: SongSeed, document: ParsedDocument) -> Binding:
    identity_blocks = [block for block in document.blocks if local_identity(block.source_text, seed)]
    identity_method = "explicit_local_performer_song_sentence"
    if not identity_blocks and seed.page_kind == "song_page" and document.page_title and title_matches(document.page_title, seed):
        # Typical song overview: page title names the song, introduction names its
        # performer and says '...의 ... 타이틀곡'. Do not borrow identity from trivia.
        for block in document.blocks[:5]:
            headings = section_path(document, block.section_id)
            if headings and headings != ["개요"]:
                continue
            value = compact(block.source_text)
            if any(marker in value for marker in ("커버", "리메이크", "라이브", "리믹스", "원곡", "아닌", "아니다")):
                continue
            if any(re.search(re.escape(compact(artist)) + r"의.{0,100}(?:타이틀곡|수록곡|노래)(?:이다|다|[0-9]|$)", value)
                   for artist in [*seed.artists, *seed.artist_aliases]):
                identity_blocks.append(block)
        identity_method = "page_title_and_local_song_overview"
    scope, evidence, reason = [], [], "local_performer_song_identity_not_confirmed"
    status = "ambiguous"
    if document.parse_status == "structure_error":
        status, reason = "pending", "parse_failed"
    elif len(identity_blocks) == 1 and seed.recording_variant in ("original", "unknown"):
        identifying = identity_blocks[0]
        if seed.page_kind == "song_page" and document.page_title and title_matches(document.page_title, seed):
            scope = [block.block_id for block in document.blocks]
        elif seed.page_kind != "song_page":
            matching_sections = [section for section in document.sections if title_matches(section.heading, seed)
                                 and identifying.block_id in subtree_blocks(document, section.section_id)]
            if len(matching_sections) == 1:
                scope = subtree_blocks(document, matching_sections[0].section_id)
            elif seed.page_kind == "artist_song_section":
                # Never inherit the artist's whole overview/biography or adjacent songs.
                scope = [identifying.block_id]
        if scope:
            evidence = [align(document, Quote(block_id=identifying.block_id, quote=identifying.source_text), set(scope))]
            status, reason = "verified", identity_method + "; fact meaning still requires review"
    payload = dict(song_id=seed.song_id, snapshot_id=document.snapshot_id, parser_version=document.parser_version,
                   binding_type=seed.page_kind, target_block_ids=scope,
                   target_section_ids=list(dict.fromkeys(block.section_id for block in document.blocks if block.block_id in scope)),
                   recording_variant=seed.recording_variant, status=status, evidence_spans=evidence, reason=reason)
    return Binding(binding_id=digest({**payload, "evidence_spans": [item.model_dump() for item in evidence]}), **payload)


def reviewed_binding(seed: SongSeed, document: ParsedDocument, review: BindingReview) -> Binding:
    if document.parse_status == "structure_error":
        raise ValueError("cannot_bind_failed_parse")
    ids = {block.block_id for block in document.blocks}
    allowed = set(review.target_block_ids)
    if not allowed <= ids or len(allowed) != len(review.target_block_ids):
        raise ValueError("invalid_or_duplicate_binding_blocks")
    evidence = [align(document, quote, allowed) for quote in review.evidence]
    if not any(span.block_id in allowed for span in evidence):
        raise ValueError("binding_requires_main_text_evidence")
    payload = dict(song_id=seed.song_id, snapshot_id=document.snapshot_id, parser_version=document.parser_version,
                   binding_type=review.binding_type, target_block_ids=review.target_block_ids,
                   target_section_ids=list(dict.fromkeys(block.section_id for block in document.blocks if block.block_id in allowed)),
                   recording_variant=review.recording_variant, status="verified", decision_method="human",
                   reviewer=review.reviewer, evidence_spans=evidence, reason=review.reason)
    return Binding(binding_id=digest({**payload, "evidence_spans": [item.model_dump() for item in evidence]}), **payload)


def direct_song_links(seed: SongSeed, document: ParsedDocument) -> list[str]:
    """Discovery hints only: explicit title anchors in a local artist/song context.

    Returned URLs are NOT fetched recursively or treated as verified identities.
    A reviewer may add them as seed URLs (at most 3 per song in the CLI).
    """
    from .fetcher import article_url
    result = []
    for block in document.blocks:
        context = compact(block.source_text + " ".join(section_path(document, block.section_id)))
        if not any(compact(artist) in context for artist in [*seed.artists, *seed.artist_aliases]):
            continue
        for link in block.links:
            if title_matches(link.text, seed):
                try:
                    result.append(article_url(link.url))
                except ValueError:
                    pass
    return list(dict.fromkeys(result))
