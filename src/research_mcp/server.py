"""Research MCP server.

Gives an AI assistant (like Claude) tools to search and read the web: search free web sources
and paper databases, and fetch any page or PDF as clean text.
"""

import logging
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from research_mcp import fetch, papers, search
from research_mcp.cache import PageCache, SearchCache
from research_mcp.extract import ExtractError, Page, extract
from research_mcp.papers import Paper
from research_mcp.search import Result

# Fetched pages are saved here for a while. Override with the RESEARCH_CACHE_DIR environment
# variable. Everything in it can be rebuilt, so it's safe to delete.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
CACHE_DIR = Path(os.environ.get("RESEARCH_CACHE_DIR", DEFAULT_CACHE_DIR)).resolve()
# A cached page is used for this long before it's fetched again; search results for an hour.
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60
SEARCH_CACHE_MAX_AGE_SECONDS = 60 * 60

# fetch_page returns at most this many characters per call, so one huge page can't flood the AI's
# context; it can ask for the rest with start_index.
DEFAULT_MAX_CHARS = 20_000
MAX_MAX_CHARS = 100_000

# Searches return this many results per source by default, and at most MAX_LIMIT.
DEFAULT_LIMIT = 5
MAX_LIMIT = 20
# Paper abstracts are cut to this length; fetch_page on the paper reads the whole thing.
ABSTRACT_CHARS = 600

WebSource = Literal["duckduckgo", "wikipedia", "github", "hackernews"]
PaperSource = Literal["arxiv", "semantic_scholar"]
SOURCE_NAMES = {"duckduckgo": "DuckDuckGo", "wikipedia": "Wikipedia", "github": "GitHub",
                "hackernews": "Hacker News", "arxiv": "arXiv", "semantic_scholar": "Semantic Scholar"}

# httpx logs every request; keep that noise out of the server's logs.
logging.getLogger("httpx").setLevel(logging.WARNING)

mcp = MCPServer(
    name="research",
    instructions=(
        "Tools for searching and reading the web. Use web_search to find pages (DuckDuckGo and "
        "Wikipedia by default; add github for code and hackernews for discussions), and "
        "search_papers for research papers (arXiv and Semantic Scholar). Results only have short "
        "snippets: use fetch_page to read a page, PDF or paper in full as clean markdown. Long "
        "pages come back in parts: if the result says there's more, call fetch_page again with "
        "the start_index it gives, but only if you need the rest. When answering, cite URLs."
    ),
)


def _cache() -> PageCache:
    return PageCache(CACHE_DIR / "pages.db", CACHE_MAX_AGE_SECONDS)


def _search_cache() -> SearchCache:
    return SearchCache(CACHE_DIR / "searches.db", SEARCH_CACHE_MAX_AGE_SECONDS)


def _check_search(query: str, sources: list, limit: int) -> str:
    """The query, tidied, after checking the arguments every search tool takes."""
    if not (query := " ".join(query.split())):
        raise ToolError("query must not be empty")
    if not sources:
        raise ToolError("sources must not be empty")
    if not 1 <= limit <= MAX_LIMIT:
        raise ToolError(f"limit must be between 1 and {MAX_LIMIT}")
    return query


def _failed(errors: dict[str, str]) -> list[str]:
    return [f"{SOURCE_NAMES[name]} failed: {message}" for name, message in errors.items()]


def _normalize(url: str) -> str:
    """Tidy a URL the AI gave us: add https:// if missing and drop the #fragment."""
    url = url.strip()
    if not url:
        raise ToolError("url must not be empty")
    if "://" not in url:
        url = "https://" + url
    return urlunsplit(urlsplit(url)._replace(fragment=""))


async def _get_page(url: str, refresh: bool) -> Page:
    """The page at url, from the cache if we fetched it recently."""
    cache = _cache()
    if not refresh and (page := cache.get(url)):
        return page
    try:
        response = await fetch.fetch(url)
        page = extract(response.url, response.content_type, response.body)
    except (fetch.FetchError, ExtractError) as error:
        raise ToolError(str(error)) from error
    cache.put(url, page)
    return page


def _cut(text: str, start: int, max_chars: int) -> tuple[str, int]:
    """text[start:start + max_chars], ending at a line break if one is near the end.

    Returns the piece and where the next piece starts.
    """
    end = start + max_chars
    if end >= len(text):
        return text[start:], len(text)
    # Don't cut mid-sentence if a line break is in the last fifth of the piece.
    if (newline := text.rfind("\n", start + max_chars * 4 // 5, end)) != -1:
        end = newline + 1
    return text[start:end], end


@mcp.tool()
async def fetch_page(
    url: str, start_index: int = 0, max_chars: int = DEFAULT_MAX_CHARS, refresh: bool = False
) -> str:
    """Read a web page, PDF or text file as clean markdown: just the main content, with links
    and tables kept. Pages are cached for a day.

    Args:
        url: The address, e.g. "https://en.wikipedia.org/wiki/RSA_cryptosystem".
        start_index: Character to start at, for reading the rest of a long page. Use the value
            the previous call said to continue from.
        max_chars: Most characters to return (up to 100000).
        refresh: Fetch the page again even if it's cached (for pages that change often).
    """
    if start_index < 0:
        raise ToolError("start_index must not be negative")
    if not 1 <= max_chars <= MAX_MAX_CHARS:
        raise ToolError(f"max_chars must be between 1 and {MAX_MAX_CHARS}")
    page = await _get_page(_normalize(url), refresh)
    total = len(page.text)
    if start_index >= total and total:
        raise ToolError(f"start_index {start_index} is past the end of the page ({total:,} characters)")
    piece, next_index = _cut(page.text, start_index, max_chars)

    lines = [f"# {page.title}"] if page.title else []
    lines.append(f"URL: {page.url}")
    if details := [d for d in (page.site, page.author and f"by {page.author}",
                               page.date and f"published {page.date}") if d]:
        lines.append(" · ".join(details))
    shown = f"all {total:,} characters" if piece == page.text else (
        f"characters {start_index:,}-{next_index:,} of {total:,}"
    )
    lines += [f"{page.kind[0].upper()}{page.kind[1:]}, {shown}", "", piece.rstrip()]
    if next_index < total:
        lines += ["", f"[{total - next_index:,} more characters. To read on, call fetch_page with "
                      f"start_index={next_index}.]"]
    return "\n".join(lines)


def _format_result(number: int, result: Result) -> str:
    lines = [f"{number}. **{result.title}**", f"   {result.url}"]
    if facts := " · ".join(f for f in (result.date, result.details) if f):
        lines.append(f"   {facts}")
    if result.snippet:
        lines.append(f"   {result.snippet}")
    return "\n".join(lines)


@mcp.tool()
async def web_search(
    query: str, sources: list[WebSource] | None = None, limit: int = DEFAULT_LIMIT
) -> str:
    """Search the web with free sources. Returns titles, URLs and short snippets; use
    fetch_page to read a result in full. Results are cached for an hour.

    Args:
        query: What to search for, e.g. "how does RSA key generation work".
        sources: Where to search (default: duckduckgo and wikipedia).
            duckduckgo: the whole web. wikipedia: encyclopedia articles.
            github: code repositories (with stars and language).
            hackernews: tech news and discussions (with points and comments).
        limit: Most results per source (up to 20; DuckDuckGo gives at most about 10).
    """
    sources = list(dict.fromkeys(sources or ["duckduckgo", "wikipedia"]))  # Drop repeats.
    query = _check_search(query, sources, limit)
    results, errors = await search.web_search(query, sources, limit, _search_cache())
    if not results:
        raise ToolError("Every source failed:\n" + "\n".join(_failed(errors)))
    sections = []
    for name in sources:
        if name in results:
            body = "\n".join(_format_result(i, r) for i, r in enumerate(results[name], 1))
            sections.append(f"## {SOURCE_NAMES[name]}\n\n{body or 'No results.'}")
    return "\n\n".join([f'Search results for "{query}"', *sections, *_failed(errors)])


def _format_paper(number: int, paper: Paper) -> str:
    authors = ", ".join(paper.authors[:4])
    if len(paper.authors) > 4:
        authors += f" +{len(paper.authors) - 4} more"
    facts = [authors, paper.venue, paper.citations is not None and f"{paper.citations:,} citations"]
    ids = [paper.arxiv_id and f"arXiv:{paper.arxiv_id}", paper.doi and f"DOI {paper.doi}"]
    lines = [f"{number}. **{paper.title}**" + (f" ({paper.year})" if paper.year else "")]
    for parts in (facts, ids):
        if parts := [p for p in parts if p]:
            lines.append("   " + " · ".join(parts))
    lines.append(f"   {paper.url}" + (f" · PDF: {paper.pdf_url}" if paper.pdf_url else ""))
    if abstract := paper.abstract:
        if len(abstract) > ABSTRACT_CHARS:
            abstract = abstract[:ABSTRACT_CHARS].rsplit(" ", 1)[0] + " …"
        lines.append(f"   {abstract}")
    return "\n".join(lines)


@mcp.tool()
async def search_papers(
    query: str,
    limit: int = DEFAULT_LIMIT,
    year_from: int | None = None,
    sort: Literal["relevance", "newest", "most_cited"] = "relevance",
    sources: list[PaperSource] | None = None,
) -> str:
    """Search for research papers on arXiv and Semantic Scholar. Returns titles, authors, year,
    citations, links (with a free PDF when there is one) and the start of each abstract. Read
    a paper in full with fetch_page on its PDF link. Results are cached for an hour.

    Args:
        query: Topic or title, e.g. "retrieval augmented generation survey".
        limit: Most papers per source (up to 20). The same paper found twice is shown once.
        year_from: Only papers from this year or later, e.g. 2023.
        sort: relevance (default), newest, or most_cited (citation counts come from Semantic
            Scholar). Sorting reorders the papers found; it doesn't search for different ones.
        sources: arxiv (preprints, mostly physics, maths and computer science, always with a
            free PDF) and/or semantic_scholar (all fields, with citation counts). Default: both.
    """
    sources = list(dict.fromkeys(sources or ["arxiv", "semantic_scholar"]))
    query = _check_search(query, sources, limit)
    if year_from is not None and not 1900 <= year_from <= 2100:
        raise ToolError("year_from must be a year like 2023")
    found, errors = await papers.search_papers(query, sources, limit, year_from, sort, _search_cache())
    if errors.keys() == set(sources):
        raise ToolError("Every source failed:\n" + "\n".join(_failed(errors)))
    searched = " and ".join(SOURCE_NAMES[name] for name in sources if name not in errors)
    heading = f'Papers for "{query}" from {searched}'
    if year_from:
        heading += f", {year_from} or later"
    if sort != "relevance":
        heading += f", sorted by {sort.replace('_', ' ')}"
    if sort == "most_cited" and all(p.citations is None for p in found):
        heading += " (but no citation counts were available, so in relevance order)"
    numbered = [_format_paper(i, p) for i, p in enumerate(found, 1)] or ["No papers found."]
    return "\n\n".join([heading, *numbered, *_failed(errors)])
