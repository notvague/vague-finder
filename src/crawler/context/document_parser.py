"""Bounded DOM walk. Adapters need validation against real saved Namuwiki HTML.

source_text is the deterministic DOM text of a block (entities decoded, <br> as
newline), not a byte offset into HTML. No global text/node deduplication is used.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from .fetcher import article_url, challenge, needs_rendering
from .schemas import Block, Footnote, Link, ParsedDocument, Section, digest


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def compact(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFKC", value).casefold())


def heading_text(tag: Tag) -> str:
    value = re.sub(r"\s*\[편집\]\s*$", "", tag.get_text(" ", strip=True))
    return re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", value).strip()


def outline_root(soup):
    """Smallest common ancestor of numbered top-level sections, by NODE identity.

    Current rendered pages have deep opaque div wrappers. Never select an opaque
    class or the entire body. Missing/ambiguous outlines are left for review.
    A single section needs a separately verified semantic root (legacy adapters).
    """
    headings = [h for h in soup.find_all("h2")
                if re.match(r"^\s*\d+\.\s*", h.get_text(" ", strip=True))
                and not any(p.name in {"nav", "aside", "header", "footer"}
                            or p.get("role") == "navigation" for p in h.parents if isinstance(p, Tag))]
    if not headings:
        return None
    if len(headings) == 1:
        heading = headings[0]
        # For a very short page, choose the smallest structural ancestor that
        # contains both the numbered heading and actual prose/list content.
        # This remains bounded and never falls back to the whole body.
        for parent in heading.parents:
            if parent.name in {"body", "html", "[document]"}:
                return None
            if parent.name not in {"div", "section", "article", "main"}:
                continue
            content = parent.find(["p", "li", "table", "blockquote"])
            if content and len(parent.get_text(" ", strip=True)) > len(heading.get_text(" ", strip=True)) + 5:
                return parent
        return None
    numbers = [int(re.match(r"^\s*(\d+)", h.get_text()).group(1)) for h in headings]
    if numbers != sorted(set(numbers)):
        return None
    ancestor_sets = [{id(p) for p in h.parents} for h in headings[1:]]
    for parent in headings[0].parents:
        if parent.name in {"body", "html", "[document]"}:
            return None
        if all(id(parent) in ancestors for ancestors in ancestor_sets):
            if parent.name not in {"div", "section", "article", "main"}:
                return None
            return parent
    return None


def parse_document(raw: bytes, snapshot_id: str, url: str) -> ParsedDocument:
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    diagnostics = {"root_selector": None, "removed_ui_nodes": 0, "warnings": []}
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else None
    if title:
        title = re.sub(r"\s*[-–|]\s*나무위키\s*$", "", title)
    canonical = None
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        try:
            canonical = article_url(urljoin(url, link["href"]),strip_query=True)
        except ValueError:
            diagnostics["warnings"].append("invalid_canonical_ignored")

    def failed(reason):
        diagnostics["warnings"].append(reason)
        return ParsedDocument(snapshot_id=snapshot_id, parse_status="structure_error",
                              page_title=title, canonical_url=canonical, article_content_hash=digest([]),
                              sections=[], blocks=[], footnotes=[], diagnostics=diagnostics)

    if challenge(raw):
        return failed("access_challenge")
    if needs_rendering(raw):
        return failed("render_required_client_shell")
    root = None
    # The wiki selectors are legacy adapters, NOT a claim about today's live site.
    for selector in (".wiki-content", ".wiki-inner-content", "article", "main", '[role="main"]'):
        candidates = soup.select(selector)
        identities = {id(node) for node in candidates}
        candidates = [node for node in candidates if not any(id(parent) in identities for parent in node.parents)]
        if len(candidates) == 1:
            root, diagnostics["root_selector"] = candidates[0], selector
            break
        if len(candidates) > 1:
            return failed("ambiguous_article_roots")
    if root is None:
        root = outline_root(soup)
        if root is None:
            return failed("article_root_not_found_no_body_fallback")
        diagnostics["root_selector"] = "numbered_h2_common_ancestor"

    footnotes, footnote_nodes, used = [], [], set()
    for node in root.select('[role="doc-endnote"], .footnotes li, .wiki-macro-footnote [id], li[id^="fn-"]'):
        identity = node.get("id")
        if not identity or identity in used or identity.startswith(("fnref", "rfn")):
            continue
        used.add(identity)
        footnotes.append(Footnote(footnote_id=identity, source_text=node.get_text("", strip=False).strip(),
                                  links=[Link(text=a.get_text("", strip=True), url=urljoin(url, a["href"]))
                                         for a in node.find_all("a", href=True)
                                         if not a["href"].startswith("#")]))
        footnote_nodes.append(node)
    for node in footnote_nodes:
        node.decompose()
    ui_selector = ('script, style, nav, footer, header, aside, form, button, iframe, svg, '
                   '.wiki-toc, .wiki-category, .wiki-edit-section, .footnotes, '
                   '.wiki-macro-footnote, [role="doc-endnotes"], [role="navigation"]')
    for node in root.select(ui_selector):
        if node.parent is not None:
            diagnostics["removed_ui_nodes"] += 1
            node.decompose()
    for node in root.select('a[href*="/edit/"], a[rel="nofollow"][href^="/edit"]'):
        node.decompose()

    sections = [Section(section_id="s0000", parent_id=None, heading="", level=0)]
    section_stack = [sections[0]]
    blocks = []
    heading_names = {f"h{i}" for i in range(1, 7)}
    boundary_names = {"p", "li", "tr", "blockquote", "div", "section", "ul", "ol", "table", "tbody",
                      "thead", "details", "summary", *heading_names}

    def flags_for(node):
        flags = []
        for parent in [node, *list(node.parents)]:
            if not isinstance(parent, Tag):
                continue
            classes = " ".join(parent.get("class", []))
            if re.search(r"(?:^|[-_ ])lyrics?(?:$|[-_ ])", classes, re.I) or parent.get("data-type") == "lyrics":
                flags.append("lyrics")
            if parent is root:
                break
        return sorted(set(flags))

    def emit(nodes, owner, kind):
        texts, links, refs = [], [], []

        def inline(node):
            if isinstance(node, Comment):
                return
            if isinstance(node, NavigableString):
                texts.append(str(node))
                return
            if not isinstance(node, Tag):
                return
            if node.name == "br":
                texts.append("\n")
                return
            if node.name == "a" and node.get("href"):
                href = node["href"]
                if href.startswith("#") and (href[1:] in used or re.match(r"#(?:fn|rfn|note)-", href)):
                    refs.append(href[1:])
                else:
                    links.append(Link(text=node.get_text("", strip=True), url=urljoin(url, href)))
            if node.name in ("td", "th") and texts and not texts[-1].endswith("\n"):
                texts.append("\n")
            for child in node.children:
                inline(child)

        for node in nodes:
            inline(node)
        source = "".join(texts).strip()
        if not source:
            return
        block_id = f"b{len(blocks):05d}"
        section = section_stack[-1]
        blocks.append(Block(block_id=block_id, section_id=section.section_id, order=len(blocks),
                            block_type=kind, source_text=source, normalized_text=normalize(source),
                            links=links, footnote_refs=list(dict.fromkeys(refs)), flags=flags_for(owner)))
        section.own_block_ids.append(block_id)

    def walk(node, inherited="paragraph"):
        if not isinstance(node, Tag):
            return
        if node.name in heading_names:
            if node.name == "h1":
                return
            level = int(node.name[1])
            while section_stack[-1].level >= level:
                section_stack.pop()
            section = Section(section_id=f"s{len(sections):04d}", parent_id=section_stack[-1].section_id,
                              heading=heading_text(node), level=level)
            sections.append(section)
            section_stack.append(section)
            return
        kind = {"li": "list_item", "tr": "table_row", "blockquote": "quote"}.get(node.name, inherited)
        if node.name == "tr":
            # Preserve explanatory table rows; selection handles numerical/track tables.
            emit(list(node.children), node, kind)
            return
        buffer = []
        for child in list(node.children):
            if isinstance(child, Tag) and child.name in boundary_names:
                emit(buffer, node, kind)
                buffer = []
                walk(child, kind)
            else:
                buffer.append(child)
        emit(buffer, node, kind)

    walk(root)
    if not blocks:
        return failed("article_has_no_text_blocks")
    referenced = {ref for block in blocks for ref in block.footnote_refs}
    missing = referenced - {footnote.footnote_id for footnote in footnotes}
    if missing:
        diagnostics["warnings"].append("unresolved_footnotes:" + ",".join(sorted(missing)))
    diagnostics.update(block_count=len(blocks), section_count=len(sections), footnote_count=len(footnotes))
    content = {"sections": [section.model_dump() for section in sections],
               "blocks": [block.model_dump() for block in blocks],
               "footnotes": [note.model_dump() for note in footnotes]}
    return ParsedDocument(snapshot_id=snapshot_id, parse_status="partial" if missing else "ok",
                          page_title=title, canonical_url=canonical, article_content_hash=digest(content),
                          sections=sections, blocks=blocks, footnotes=footnotes, diagnostics=diagnostics)


def section_path(document: ParsedDocument, section_id: str) -> list[str]:
    mapping = {section.section_id: section for section in document.sections}
    result = []
    while section_id:
        section = mapping[section_id]
        if section.heading:
            result.append(section.heading)
        section_id = section.parent_id
    return list(reversed(result))


def subtree_blocks(document: ParsedDocument, section_id: str) -> list[str]:
    allowed = {section_id}
    for section in document.sections:
        if section.parent_id in allowed:
            allowed.add(section.section_id)
    return [block.block_id for block in document.blocks if block.section_id in allowed]
