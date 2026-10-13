"""Literal media-name matching; an embedding score cannot supply a work name."""

from __future__ import annotations

import re
import unicodedata


def media_name_in_text(name: str, text: str) -> bool:
    """Match spacing/underscores and Korean particles, not other titles.

    In particular a title followed by another lexical syllable is a different
    entity. Equivalences must be supplied by the caller's explicit alias list.
    """
    name = re.sub(r"[\s_]+", "", unicodedata.normalize("NFKC", name).casefold())
    text = unicodedata.normalize("NFKC", text).casefold()
    if len(name) < 2:
        return False
    # Either side may join or split the same title ("호텔델루나" / "호텔 델루나"),
    # so a separator is optional at every position, not only where the name has one.
    literal = r"[\s_]*".join(re.escape(char) for char in name)
    boundary = r"(?=$|[\W_])"
    # Korean suffixes attach to a proper noun. Do not match e.g. a work name
    # inside another song/work title merely because their prefixes coincide.
    suffix = r"(?:에서|에서는|에선|의|에서의|에|으로|로|를|을|은|는|이|가|도)"
    return bool(re.search(r"(?<![^\W_])" + literal +
                          r"(?:" + boundary + r"|" + suffix + boundary + r")", text))
