"""Deprecated excerpt API; the replacement preserves document structure."""
from src.crawler.context.document_parser import parse_document


def extract_namuwiki_section_context(*args, **kwargs):
    raise RuntimeError(
        "Truncated section excerpts were retired. Use "
        "src.crawler.context.document_parser.parse_document and explicit song binding."
    )


__all__ = ["parse_document"]
