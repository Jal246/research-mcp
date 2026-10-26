import pytest

from conftest import FIXTURES
from research_mcp.extract import ExtractError, extract

ARTICLE = (FIXTURES / "article.html").read_bytes()


def test_html_keeps_the_article_and_drops_the_clutter():
    page = extract("https://blog.example/rsa", "text/html; charset=utf-8", ARTICLE)
    assert page.kind == "web page"
    assert "# How RSA Works" in page.text
    assert "## Key generation" in page.text
    assert "φ(n)" in page.text  # Non-ASCII survives.
    for clutter in ("Log in", "We use cookies", "All rights reserved"):
        assert clutter not in page.text


def test_html_links_are_absolute_and_tables_are_markdown():
    page = extract("https://blog.example/rsa", "text/html", ARTICLE)
    assert "[Euclid's algorithm](https://blog.example/posts/euclid)" in page.text
    assert "| n | modulus |" in page.text


def test_html_metadata():
    page = extract("https://blog.example/rsa", "text/html", ARTICLE)
    assert page.title == "How RSA Works"
    assert page.site == "Crypto Blog"
    assert page.author == "Ada Example"
    # The real publish date, not a guess from "Copyright 2026" in the footer.
    assert page.date == "2025-03-14"


def test_html_without_content_type_is_detected():
    assert extract("https://blog.example/rsa", "", ARTICLE).kind == "web page"


def test_page_with_no_text():
    with pytest.raises(ExtractError, match="no readable text"):
        extract("https://app.example/", "text/html", b"<html><body><div id='root'></div></body></html>")


def test_pdf_has_numbered_pages_and_title():
    page = extract("https://papers.example/paper.pdf", "application/pdf",
                   (FIXTURES / "sample.pdf").read_bytes())
    assert page.kind == "PDF, 2 pages"
    assert page.title == "Sample Paper"
    assert page.text == (
        "## Page 1\n\nAttention is all you need\n\n## Page 2\n\nSecond page about transformers"
    )


def test_pdf_is_detected_from_its_bytes():
    # Servers often send PDFs as application/octet-stream.
    page = extract("https://x.example/dl", "application/octet-stream", (FIXTURES / "sample.pdf").read_bytes())
    assert page.kind == "PDF, 2 pages"


def test_broken_pdf():
    with pytest.raises(ExtractError, match="isn't a valid PDF"):
        extract("https://x.example/a.pdf", "application/pdf", b"%PDF-1.4 garbage")


@pytest.mark.parametrize("content_type", ["text/plain", "text/markdown", "application/json"])
def test_text_is_returned_as_is(content_type):
    page = extract("https://x.example/f", content_type, b'{"a": 1}\n')
    assert page.kind == "text"
    assert page.text == '{"a": 1}\n'


def test_text_uses_the_declared_charset():
    page = extract("https://x.example/f", "text/plain; charset=latin-1", "café".encode("latin-1"))
    assert page.text == "café"


def test_unsupported_types():
    with pytest.raises(ExtractError, match="Can't read image/png files"):
        extract("https://x.example/cat.png", "image/png", b"\x89PNG")
