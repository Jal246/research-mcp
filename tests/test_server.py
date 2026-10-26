"""Tests that talk to the server through a real MCP client, the same way Claude does."""

import pytest
from mcp import Client

from research_mcp import server
from research_mcp.server import mcp

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def temporary_cache(monkeypatch, tmp_path):
    """Keep fetched pages, searches and the library in a temporary folder, not the real ones."""
    monkeypatch.setattr(server, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")


def text_of(result) -> str:
    return "\n".join(block.text for block in result.content)


async def call(tool: str, args: dict | None = None):
    async with Client(mcp) as client:
        return await client.call_tool(tool, args or {})


async def test_lists_all_tools():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == {
            "fetch_page", "web_search", "search_papers", "save_to_library", "search_library",
            "list_library", "remove_from_library", "find_similar", "crawl_site",
        }


async def test_fetch_page():
    result = await call("fetch_page", {"url": "https://blog.example/rsa"})
    text = text_of(result)
    assert not result.is_error
    assert text.startswith(
        "# How RSA Works\nURL: https://blog.example/rsa\n"
        "Crypto Blog · by Ada Example · published 2025-03-14\nWeb page, all "
    )
    assert "## Key generation" in text
    assert "more characters" not in text


async def test_fetch_page_shows_where_redirects_lead():
    text = text_of(await call("fetch_page", {"url": "https://blog.example/old-rsa"}))
    assert "URL: https://blog.example/rsa\n" in text


async def test_fetch_page_adds_https_and_drops_fragment(fake_internet):
    await call("fetch_page", {"url": "  blog.example/rsa#key-generation "})
    assert fake_internet == ["https://blog.example/rsa"]


async def test_fetch_pdf():
    text = text_of(await call("fetch_page", {"url": "https://papers.example/paper.pdf"}))
    assert text.startswith("# Sample Paper\nURL: https://papers.example/paper.pdf\nPDF, 2 pages, all ")
    assert "## Page 2\n\nSecond page about transformers" in text


async def test_long_pages_come_in_parts():
    url = "https://blog.example/notes.txt"  # "line one\n" x 5000 = 45,000 characters.
    first = text_of(await call("fetch_page", {"url": url, "max_chars": 999}))  # 111 lines.
    assert "Text, characters 0-999 of 45,000" in first
    assert first.endswith("[44,001 more characters. To read on, call fetch_page with start_index=999.]")

    last = text_of(await call("fetch_page", {"url": url, "start_index": 44_500, "max_chars": 1000}))
    assert "characters 44,500-45,000 of 45,000" in last
    assert "more characters" not in last


async def test_parts_end_at_a_line_break():
    url = "https://blog.example/notes.txt"
    text = text_of(await call("fetch_page", {"url": url, "max_chars": 1004}))
    assert "start_index=999." in text  # 111 whole lines of 9 characters, not 1004.


async def test_cache_is_used(fake_internet):
    url = "https://blog.example/notes.txt"
    await call("fetch_page", {"url": url, "max_chars": 1000})
    await call("fetch_page", {"url": url, "start_index": 1000, "max_chars": 1000})
    assert len(fake_internet) == 1
    await call("fetch_page", {"url": url, "refresh": True})
    assert len(fake_internet) == 2


async def test_errors_are_not_cached(fake_internet):
    for _ in range(2):
        assert (await call("fetch_page", {"url": "https://blog.example/missing"})).is_error
    assert len(fake_internet) == 2


@pytest.mark.parametrize(
    "args, message",
    [
        ({"url": "http://localhost:8000"}, "private or local address"),
        ({"url": "https://blog.example/missing"}, "HTTP 404"),
        ({"url": "https://blog.example/cat.png"}, "Can't read image/png files"),
        ({"url": "  "}, "url must not be empty"),
        ({"url": "https://blog.example/rsa", "max_chars": 0}, "max_chars must be between"),
        ({"url": "https://blog.example/rsa", "start_index": -1}, "must not be negative"),
        ({"url": "https://blog.example/rsa", "start_index": 10**6}, "past the end"),
    ],
)
async def test_fetch_page_errors(args, message):
    result = await call("fetch_page", args)
    assert result.is_error
    assert message in text_of(result)


async def test_web_search():
    result = await call("web_search", {"query": "  rsa\n"})
    assert not result.is_error
    assert text_of(result) == """Search results for "rsa"

## DuckDuckGo

1. **RSA cryptosystem - Wikipedia**
   https://en.wikipedia.org/wiki/RSA_cryptosystem
   The RSA (Rivest–Shamir–Adleman) cryptosystem is a family of public-key cryptosystems.
2. **How RSA Works & Why**
   https://blog.example/rsa?ref=ddg
   A friendly walk through key generation.
3. **Prime numbers**
   https://www.example.org/primes/

## Wikipedia

1. **Euler's totient function**
   https://en.wikipedia.org/wiki/Euler%27s_totient_function
   last edited 2026-09-01
   counts the positive integers up to a given integer"""


async def test_web_search_other_sources():
    text = text_of(await call("web_search", {"query": "mcp", "sources": ["github", "hackernews"], "limit": 1}))
    assert "## GitHub\n\n1. **modelcontextprotocol/python-sdk**\n" in text
    assert "   2026-10-25 · ★ 24,666 · Python\n" in text
    assert "## Hacker News\n\n1. **We updated our RSA SSH host key**" in text
    assert "2." not in text


async def test_web_search_reports_failed_sources():
    text = text_of(await call("web_search", {"query": "down", "sources": ["github", "wikipedia"]}))
    assert "## Wikipedia" in text
    assert "GitHub failed: " in text and "HTTP 503" in text


async def test_web_search_with_no_results():
    text = text_of(await call("web_search", {"query": "nothing"}))
    assert "## DuckDuckGo\n\nNo results." in text and "## Wikipedia\n\nNo results." in text


async def test_web_search_fails_when_every_source_fails():
    result = await call("web_search", {"query": "bot check", "sources": ["duckduckgo"]})
    assert result.is_error
    assert "Every source failed:\nDuckDuckGo failed: DuckDuckGo thinks we're a bot" in text_of(result)


async def test_web_search_is_cached(fake_internet):
    for _ in range(2):
        await call("web_search", {"query": "rsa"})
    assert len(fake_internet) == 2  # DuckDuckGo and Wikipedia, once each.


async def test_search_papers():
    text = text_of(await call("search_papers", {"query": "attention"}))
    assert text.startswith(
        'Papers for "attention" from arXiv and Semantic Scholar\n\n'
        "1. **Attention Is All You Need** (2017)\n"
        "   Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit +2 more · "
        "Neural Information Processing Systems · 150,123 citations\n"
        "   arXiv:1706.03762\n"
        "   https://arxiv.org/abs/1706.03762 · PDF: https://arxiv.org/pdf/1706.03762\n"
        "   The dominant sequence transduction models"
    )
    assert "\n\n3. **Attention Mechanisms in Medicine** (2021)\n   Grace Example · 42 citations\n" in text


async def test_search_papers_cuts_long_abstracts(monkeypatch):
    monkeypatch.setattr(server, "ABSTRACT_CHARS", 40)
    text = text_of(await call("search_papers", {"query": "attention", "sources": ["arxiv"], "limit": 1}))
    assert text.endswith("   The dominant sequence transduction …")


async def test_search_papers_options(fake_internet):
    text = text_of(await call("search_papers", {"query": "attention", "year_from": 2020, "sort": "newest"}))
    assert text.startswith('Papers for "attention" from arXiv and Semantic Scholar, 2020 or later, '
                           "sorted by newest\n\n1. **Element-wise")
    assert any("submittedDate" in url for url in fake_internet)
    assert any("year=2020-" in url for url in fake_internet)


async def test_search_papers_when_semantic_scholar_is_busy():
    text = text_of(await call("search_papers", {"query": "always busy", "sort": "most_cited"}))
    assert text.startswith('Papers for "always busy" from arXiv, sorted by most cited (but no '
                           "citation counts were available, so in relevance order)")
    assert text.endswith("Semantic Scholar failed: Semantic Scholar's free tier is busy (it's shared by "
                         "everyone without an API key); try again in a minute, or set "
                         "SEMANTIC_SCHOLAR_API_KEY (free from semanticscholar.org/product/api)")


async def test_search_papers_with_no_results():
    text = text_of(await call("search_papers", {"query": "nothing"}))
    assert text == 'Papers for "nothing" from arXiv and Semantic Scholar\n\nNo papers found.'


@pytest.mark.parametrize(
    "tool, args, message",
    [
        ("web_search", {"query": " "}, "query must not be empty"),
        ("web_search", {"query": "rsa", "limit": 0}, "limit must be between 1 and 20"),
        ("web_search", {"query": "rsa", "limit": 21}, "limit must be between 1 and 20"),
        ("web_search", {"query": "rsa", "sources": ["google"]}, "validation error"),
        ("search_papers", {"query": ""}, "query must not be empty"),
        ("search_papers", {"query": "rag", "year_from": 23}, "year_from must be a year"),
        ("search_papers", {"query": "rag", "sort": "best"}, "validation error"),
        ("search_papers", {"query": "always busy", "sources": ["semantic_scholar"]}, "Every source failed"),
    ],
)
async def test_search_errors(tool, args, message):
    result = await call(tool, args)
    assert result.is_error
    assert message in text_of(result)


# --- Research library ---------------------------------------------------------------------------

async def test_library_starts_empty():
    assert text_of(await call("list_library")) == "The library is empty. Save pages with save_to_library."
    assert text_of(await call("search_library", {"query": "rsa"})).startswith("The library is empty.")


async def test_save_search_list_and_remove():
    text = text_of(await call("save_to_library", {
        "url": "https://blog.example/old-rsa", "note": "Clear RSA intro", "tags": ["Crypto"]}))
    assert text.startswith("Saved to the library (1 page saved):\n\n**How RSA Works**\n"
                           "   https://blog.example/rsa\n   web page · ")
    assert text.endswith(" · tags: crypto\n   Note: Clear RSA intro")
    await call("save_to_library", {"url": "https://papers.example/paper.pdf", "tags": ["ml"]})

    text = text_of(await call("search_library", {"query": "rsa key generation primes"}))
    assert text.startswith('Library results for "rsa key generation primes"\n\n1. **How RSA Works** (similarity 0.')
    assert "\n   > " in text and "RSA is a public-key cryptosystem." in text
    assert "(Passage starts at fetch_page start_index=0.)" in text
    assert "Sample Paper" not in text

    text = text_of(await call("list_library"))
    assert text.startswith("2 pages in the library\nTags: crypto (1), ml (1)\n\n1. **Sample Paper**")
    assert "2. **How RSA Works**" in text
    assert text_of(await call("list_library", {"tag": "crypto"})).startswith(
        "1 page in the library tagged 'crypto'\n\n1. **How RSA Works**")

    text = text_of(await call("remove_from_library", {"url": "blog.example/rsa"}))
    assert text == "Removed How RSA Works (https://blog.example/rsa) from the library."
    assert "How RSA Works" not in text_of(await call("list_library"))


async def test_saving_again_updates():
    await call("save_to_library", {"url": "https://blog.example/rsa", "tags": ["a"]})
    text = text_of(await call("save_to_library", {"url": "https://blog.example/rsa", "tags": ["b"]}))
    assert text.startswith("Updated in the library (1 page saved):")
    assert "tags: a, b" in text


async def test_search_library_finds_nothing_related():
    await call("save_to_library", {"url": "https://blog.example/rsa"})
    text = text_of(await call("search_library", {"query": "banana bread"}))
    assert text == 'Nothing in the library seems related to "banana bread".'
    text = text_of(await call("search_library", {"query": "rsa", "tag": "cooking"}))
    assert text == "No saved pages have the tag 'cooking'."


@pytest.mark.parametrize(
    "tool, args, message",
    [
        ("save_to_library", {"url": "https://blog.example/missing"}, "HTTP 404"),
        ("search_library", {"query": " "}, "query must not be empty"),
        ("search_library", {"query": "rsa", "limit": 50}, "limit must be between 1 and 20"),
        ("remove_from_library", {"url": "https://blog.example/rsa"}, "isn't in the library"),
    ],
)
async def test_library_errors(tool, args, message):
    result = await call(tool, args)
    assert result.is_error
    assert message in text_of(result)


# --- Finding similar pages ----------------------------------------------------------------------

async def test_find_similar(fake_internet):
    text = text_of(await call("find_similar", {"url": "https://blog.example/rsa"}))
    assert text.startswith('Pages similar to "How RSA Works" (https://blog.example/rsa)\n'
                           'Searched DuckDuckGo and Wikipedia for: "How RSA Works"; "')
    assert "1. **RSA cryptosystem - Wikipedia** (similarity 0." in text
    assert "blog.example/rsa?ref=ddg" not in text  # Same site.
    assert "in your library" not in text
    assert sum("duckduckgo" in url for url in fake_internet) == 2  # Title and key words.


async def test_find_similar_same_site_and_library():
    await call("save_to_library", {"url": "https://blog.example/old-rsa"})  # The page itself.
    await call("save_to_library", {"url": "https://papers.example/paper.pdf"})
    text = text_of(await call("find_similar", {"url": "https://blog.example/rsa", "include_same_site": True}))
    assert "**How RSA Works & Why** (similarity" in text
    assert "## Similar pages in your library" not in text  # Only the page itself is about RSA.


async def test_find_similar_when_every_source_fails():
    result = await call("find_similar", {"url": "https://blog.example/rsa", "sources": ["github"],
                                         "limit": 3})
    assert not result.is_error  # GitHub answers (with repos that aren't about RSA).
    result = await call("find_similar", {"url": "https://blog.example/missing"})
    assert result.is_error and "HTTP 404" in text_of(result)


# --- Crawling -----------------------------------------------------------------------------------

async def test_crawl_site(fake_internet):
    text = text_of(await call("crawl_site", {"url": "docs.example/docs/"}))
    assert text.startswith("Crawled https://docs.example/docs/: read 4 pages\n\n"
                           "1. **Example Docs**\n   https://docs.example/docs/\n   web page, ")
    assert "4. **Advanced brewing**" in text
    assert "Couldn't read 1:\n- https://docs.example/docs/missing: " in text
    assert "1 link not followed because the site's robots.txt asks bots not to." in text
    assert "Saved" not in text

    requests = len(fake_internet)
    await call("fetch_page", {"url": "https://docs.example/docs/intro"})
    assert len(fake_internet) == requests  # Cached by the crawl.


async def test_crawl_site_and_save():
    text = text_of(await call("crawl_site", {"url": "https://docs.example/docs/", "max_pages": 2,
                                             "save": True, "tags": ["coffee"]}))
    assert "read 2 pages (2 more found but not read; raise max_pages to read them)" in text
    assert "Saved all 2 to the library with tags coffee." in text
    text = text_of(await call("search_library", {"query": "water temperature degrees", "tag": "coffee"}))
    assert "1. **Introduction**" in text


@pytest.mark.parametrize(
    "args, message",
    [
        ({"url": "https://docs.example/docs/", "max_pages": 0}, "max_pages must be between 1 and 50"),
        ({"url": "https://docs.example/docs/", "max_depth": 6}, "max_depth must be between 0 and 5"),
        ({"url": "https://docs.example/docs/private/"}, "robots.txt asks bots not to read"),
        ({"url": "https://blog.example/missing"}, "Couldn't read any pages:\n- https://blog.example/missing: "),
    ],
)
async def test_crawl_site_errors(args, message):
    result = await call("crawl_site", args)
    assert result.is_error
    assert message in text_of(result)
