"""Every block gets a disposition; headings prioritize but do not limit recall."""
import re

from .document_parser import compact, section_path
from .schemas import Binding, BlockDecision, ParsedDocument, SongSeed

PRIORITY = ("여담", "제작", "배경", "뮤직비디오", "방송", "미디어", "매체", "역주행", "유행", "활동", "기록")
VERSIONS = {"cover": ("커버", "리메이크"), "live": ("라이브",), "remix": ("리믹스",)}


def select_blocks(seed: SongSeed, document: ParsedDocument, binding: Binding) -> list[BlockDecision]:
    result = []
    scope = set(binding.target_block_ids)
    for block in document.blocks:
        headings = [compact(value) for value in section_path(document, block.section_id)]
        text = block.normalized_text
        disposition, reason, priority = "selected", "scoped_explanatory_text", 1
        if binding.status != "verified":
            disposition, reason = "needs_review", "binding_not_verified"
        elif block.block_id not in scope:
            disposition, reason = "excluded", "outside_reviewed_song_scope"
        elif "lyrics" in block.flags or any(value in ("가사", "가사해석", "가사및해석", "lyrics") for value in headings):
            disposition, reason = "excluded", "lyrics"
        elif any(marker in heading for heading in headings for variant, markers in VERSIONS.items()
                 if variant != binding.recording_variant for marker in markers):
            disposition, reason = "needs_review", "other_or_mixed_recording_version"
        elif block.block_type == "table_row" and (
            len(re.findall(r"\d+", text)) >= 5 or any(value in ("트랙리스트", "수록곡", "차트", "순위") for value in headings)
        ) and not re.search(r"(?:되었|됐다|사용|제작|최초|기록|계기|때문|부른|발매했)", text):
            disposition, reason = "excluded", "raw_track_or_numeric_table"
        elif not any(character.isalnum() for character in text):
            disposition, reason = "excluded", "decorative_text"
        elif any(marker in heading for heading in headings for marker in PRIORITY):
            priority, reason = 3, "background_heading_priority"
        # No whitelist and no minimum character threshold: short/unusual prose survives.
        result.append(BlockDecision(block_id=block.block_id, disposition=disposition, reason=reason, priority=priority))
    return result
