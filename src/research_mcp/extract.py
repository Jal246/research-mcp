"""Turning downloaded files (web pages, PDFs, plain text) into clean text for the AI.

Web pages become markdown with only the main content: menus, ads, cookie banners and footers are
dropped by trafilatura, which is built for exactly this. Links and tables are kept.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import re
from dataclasses import dataclass

import pypdfium2 as pdfium
import trafilatura

HTML_TYPES = {"text/html", "application/xhtml+xml"}
# Not text/*, but still text worth reading as-is.
TEXT_TYPES = {"application/json", "application/xml", "application/javascript", "application/x-yaml"}


class ExtractError(Exception):
    """A file we can't read, like an image or a page with no text."""


@dataclass(frozen=True)
class Page:
    url: str
    kind: str  # What it was, e.g. "web page", "PDF, 12 pages", "text".
    text: str
    title: str | None = None
    site: str | None = None
    author: str | None = None
    date: str | None = None  # As the page states it, usually "2026-10-25".


def extract(url: str, content_type: str, body: bytes) -> Page:
    """Turn a downloaded file into a Page. content_type is the server's Content-Type header."""
    mime = content_type.split(";")[0].strip().lower()
    if body.startswith(b"%PDF") or mime == "application/pdf":
        return _pdf(url, body)
    if mime in HTML_TYPES or (not mime and _looks_like_html(body)):
        return _html(url, body)
    if mime.startswith("text/") or mime in TEXT_TYPES or mime.endswith(("+json", "+xml")):
        return Page(url, "text", _decode(body, content_type))
    raise ExtractError(f"Can't read {mime or 'this kind of'} files (only web pages, PDFs and text)")


def _looks_like_html(body: bytes) -> bool:
    return body[:1024].lstrip().lower().startswith((b"<!doctype html", b"<html"))


def _decode(body: bytes, content_type: str) -> str:
    charset = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
    try:
        return body.decode(charset.group(1) if charset else "utf-8", errors="replace")
    except LookupError:  # A charset name Python doesn't know.
        return body.decode("utf-8", errors="replace")


def _html(url: str, body: bytes) -> Page:
    # load_html works out the encoding from the page itself. Each call gets a fresh tree, since
    # trafilatura trims the tree it works on.
    if trafilatura.load_html(body) is None:
        raise ExtractError(f"{url} isn't readable HTML")
    # extensive=False: only trust dates the page marks up as its publish date. Otherwise
    # "Copyright 2026" in a footer turns into a publish date of 2026-01-01.
    meta = trafilatura.extract_metadata(trafilatura.load_html(body), default_url=url, extensive=False)
    text = trafilatura.extract(
        trafilatura.load_html(body),
        url=url,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        include_formatting=True,
        include_comments=False,
        favor_recall=True,  # Keep a bit too much rather than lose part of an article.
    )
    if not text:
        # No main content found (e.g. a list of links); fall back to all the page's text.
        text = trafilatura.html2txt(body)
    if not text or not text.strip():
        raise ExtractError(f"{url} has no readable text (it may need JavaScript to show its content)")
    return Page(
        url,
        "web page",
        text.strip(),
        title=meta.title if meta else None,
        site=meta.sitename if meta else None,
        author=meta.author if meta else None,
        date=meta.date if meta else None,
    )


def _pdf(url: str, body: bytes) -> Page:
    try:
        pdf = pdfium.PdfDocument(body)
    except pdfium.PdfiumError:
        raise ExtractError(f"{url} isn't a valid PDF") from None
    try:
        pages = []
        for number, page in enumerate(pdf, start=1):
            text = page.get_textpage().get_text_bounded().replace("\r\n", "\n").strip()
            if text:
                pages.append(f"## Page {number}\n\n{text}")
        count = len(pdf)
        title = pdf.get_metadata_dict().get("Title") or None
    finally:
        pdf.close()
    if not pages:
        raise ExtractError(f"{url} is a PDF with no text in it (probably scanned images)")
    return Page(url, f"PDF, {count} page{'s' * (count != 1)}", "\n\n".join(pages), title=title)
