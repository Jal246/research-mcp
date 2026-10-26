"""Research MCP server.

Gives an AI assistant (like Claude) tools to search and read the web: search free web sources
and paper databases, fetch any page or PDF as clean text, crawl a site, find similar pages, and
keep a personal library of pages that can be searched by meaning.
"""

import asyncio
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from research_mcp import crawl, fetch, papers, search, similar
from research_mcp.cache import PageCache, SearchCache
from research_mcp.embed import Embedder
from research_mcp.extract import ExtractError, Page, extract
from research_mcp.library import Document, Library, LibraryError
from research_mcp.papers import Paper
from research_mcp.search import Result

# Fetched pages are saved here for a while, and the embedding model is downloaded here. Override
# with the RESEARCH_CACHE_DIR environment variable. Everything in it can be rebuilt, so it's safe
# to delete.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
CACHE_DIR = Path(os.environ.get("RESEARCH_CACHE_DIR", DEFAULT_CACHE_DIR)).resolve()
# The research library. Unlike .cache, this is the user's own work and can't be rebuilt, so it
# gets its own folder. Override with RESEARCH_DATA_DIR.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR = Path(os.environ.get("RESEARCH_DATA_DIR", DEFAULT_DATA_DIR)).resolve()
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
# Passages from the library, and previews of crawled pages, are cut to these lengths.
PASSAGE_CHARS = 500
PREVIEW_CHARS = 200

# crawl_site reads 10 pages by default, and at most MAX_CRAWL_PAGES, up to MAX_DEPTH clicks away.
DEFAULT_CRAWL_PAGES = 10
MAX_CRAWL_PAGES = 50
MAX_DEPTH = 5

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
        "the start_index it gives, but only if you need the rest. find_similar finds pages like "
        "a given one, and crawl_site reads several pages of one site (like a docs site). "
        "The user has a research library: save_to_library keeps a page (with a note and tags) "
        "and search_library searches the saved pages by meaning. Check the library before "
        "searching the web for a topic the user has researched before. When answering, cite URLs."
    ),
)


def _cache() -> PageCache:
    return PageCache(CACHE_DIR / "pages.db", CACHE_MAX_AGE_SECONDS)


def _search_cache() -> SearchCache:
    return SearchCache(CACHE_DIR / "searches.db", SEARCH_CACHE_MAX_AGE_SECONDS)


@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    """The embedding model, loaded once (and downloaded the first time it's ever used)."""
    return Embedder(CACHE_DIR / "models")


def _library() -> Library:
    return Library(DATA_DIR / "library.db", _embedder())


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


def _shorten(text: str, max_chars: int) -> str:
    """text on one line, cut at a word if it's longer than max_chars."""
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"


def _format_document(number: int, document: Document) -> str:
    saved = time.strftime("%Y-%m-%d", time.localtime(document.saved_at))
    facts = [document.kind, f"{document.chars:,} characters", f"saved {saved}",
             document.tags and "tags: " + ", ".join(document.tags)]
    lines = [f"{number}. **{document.title or document.url}**", f"   {document.url}",
             "   " + " · ".join(f for f in facts if f)]
    if document.note:
        lines.append(f"   Note: {document.note}")
    return "\n".join(lines)


async def _library_call(function, *args):
    """Run a library operation in a worker thread (embedding text keeps the CPU busy for a
    while, and the server should stay responsive), turning LibraryErrors into tool errors."""
    try:
        return await asyncio.to_thread(function, *args)
    except LibraryError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def save_to_library(
    url: str, note: str = "", tags: list[str] | None = None, refresh: bool = False
) -> str:
    """Save a page, PDF or paper to the user's research library, to find it again later with
    search_library (which searches by meaning, not just exact words). The full text is kept,
    so it stays searchable even if the page changes or disappears. Saving a page again
    updates it. The very first save takes a while: it downloads a small local model (~70 MB).

    Args:
        url: The page to save, e.g. "https://raft.github.io/raft.pdf".
        note: Why it's worth keeping, e.g. "Clearest explanation of Raft leader election".
        tags: Labels to group pages by, e.g. ["distributed-systems", "thesis"].
        refresh: Fetch the page again even if it's cached.
    """
    page = await _get_page(_normalize(url), refresh)
    library = _library()
    document, existed = await _library_call(library.save, page, note, tags)
    total = len(await _library_call(library.documents))
    action = "Updated in" if existed else "Saved to"
    return (f"{action} the library ({total:,} page{'s' * (total != 1)} saved):\n\n"
            + _format_document(1, document).removeprefix("1. "))


@mcp.tool()
async def search_library(query: str, limit: int = DEFAULT_LIMIT, tag: str | None = None) -> str:
    """Search the user's research library by meaning: finds saved pages about what the query
    means, even if they use different words ("how do nodes pick a leader" finds a passage on
    Raft elections). Each result shows its best-matching passage, a similarity score (higher is
    closer) and where the passage starts, to read on with fetch_page's start_index.

    Args:
        query: A question or topic, e.g. "how do nodes agree on a leader".
        limit: Most pages to return (up to 20).
        tag: Only search pages with this tag.
    """
    query = _check_search(query, ["library"], limit)
    library = _library()
    if not await _library_call(library.documents, tag):
        return (f"No saved pages have the tag {tag!r}." if tag else
                "The library is empty. Save pages with save_to_library.")
    hits = await _library_call(library.search, query, limit, tag)
    if not hits:
        return f'Nothing in the library seems related to "{query}".'
    results = []
    for number, hit in enumerate(hits, 1):
        lines = _format_document(number, hit.document).split("\n")
        lines[0] += f" (similarity {hit.score:.2f})"
        lines += [f"   > {_shorten(hit.passage, PASSAGE_CHARS)}",
                  f"   (Passage starts at fetch_page start_index={hit.start}.)"]
        results.append("\n".join(lines))
    return "\n\n".join([f'Library results for "{query}"', *results])


@mcp.tool()
async def list_library(tag: str | None = None) -> str:
    """List the pages saved in the user's research library, most recently saved first, with
    their notes and tags.

    Args:
        tag: Only list pages with this tag.
    """
    library = _library()
    documents = await _library_call(library.documents, tag)
    if not documents:
        return (f"No saved pages have the tag {tag!r}." if tag else
                "The library is empty. Save pages with save_to_library.")
    heading = f"{len(documents):,} page{'s' * (len(documents) != 1)} in the library"
    if tag:
        heading += f" tagged {tag!r}"
    elif tags := await _library_call(library.tags):
        heading += "\nTags: " + ", ".join(f"{name} ({count})" for name, count in tags.items())
    return "\n\n".join([heading, *(_format_document(i, d) for i, d in enumerate(documents, 1))])


@mcp.tool()
async def remove_from_library(url: str) -> str:
    """Remove a page from the user's research library.

    Args:
        url: The saved page's URL, as list_library shows it.
    """
    document = await _library_call(_library().remove, _normalize(url))
    return f"Removed {document.title or document.url} ({document.url}) from the library."


@mcp.tool()
async def find_similar(
    url: str, limit: int = DEFAULT_LIMIT, include_same_site: bool = False,
    sources: list[WebSource] | None = None,
) -> str:
    """Find pages similar to a given page: other articles, explanations or projects about the
    same thing. Searches the web for the page's title and its key words, then ranks what it
    finds by how close in meaning it is to the page. Also shows similar pages already in the
    user's research library.

    Args:
        url: The page to find others like, e.g. "https://jvns.ca/blog/2017/06/11/log-structured-storage/".
        limit: Most similar pages to return (up to 20).
        include_same_site: Also include pages from the same website (left out by default).
        sources: Where to search (default: duckduckgo and wikipedia; also github, hackernews).
    """
    sources = list(dict.fromkeys(sources or ["duckduckgo", "wikipedia"]))
    _check_search("similar", sources, limit)
    requested = _normalize(url)
    page = await _get_page(requested, False)
    if not (queries := similar.queries(page)):
        raise ToolError(f"{page.url} has no title or words to search for")
    cache = _search_cache()
    searches = await asyncio.gather(*(search.web_search(q, sources, 10, cache) for q in queries))
    found, errors = [], {}
    for results, failed in searches:
        found += [result for name in sources for result in results.get(name, [])]
        errors |= failed
    candidates = similar.candidates(page, found, requested, include_same_site)

    def rank():
        embedder = _embedder()
        vector = embedder.embed([similar.page_text(page)])[0]
        saved = _library().closest(vector, 3, exclude_url=page.url, min_score=embedder.min_similar_score)
        return similar.rank(vector, candidates, embedder, limit), saved

    ranked, saved = await _library_call(rank)
    if not ranked and not saved and len(errors) == len(sources):
        raise ToolError("Every source failed:\n" + "\n".join(_failed(errors)))
    searched = " and ".join(SOURCE_NAMES[name] for name in sources if name not in errors)
    lines = [f'Pages similar to "{page.title or page.url}" ({page.url})',
             f"Searched {searched or 'nothing'} for: " + "; ".join(f'"{q}"' for q in queries)]
    sections = ["\n".join(lines)]
    numbered = []
    for number, (score, result) in enumerate(ranked, 1):
        numbered.append(_format_result(number, result).replace("**\n", f"** (similarity {score:.2f})\n", 1))
    sections += numbered or ["No similar pages found on the web."]
    if saved:
        sections.append("## Similar pages in your library\n\n" + "\n\n".join(
            _format_document(i, hit.document) + f"\n   (similarity {hit.score:.2f})"
            for i, hit in enumerate(saved, 1)))
    return "\n\n".join([*sections, *_failed(errors)])


def _preview(page: Page) -> str:
    """The start of a page's text, without headings."""
    lines = (line for line in page.text.splitlines() if not line.lstrip().startswith("#"))
    return _shorten("\n".join(lines), PREVIEW_CHARS)


@mcp.tool()
async def crawl_site(
    url: str, max_pages: int = DEFAULT_CRAWL_PAGES, max_depth: int = 2, save: bool = False,
    tags: list[str] | None = None,
) -> str:
    """Read several pages of one website, like a docs site, blog or course: starting at url,
    follow links to other pages under the same folder (starting at example.org/docs/ reads
    the docs, not the whole site). Follows robots.txt and waits between requests. Returns each
    page's title, URL, length and opening lines; all pages are cached, so fetch_page reads any
    of them instantly. With save=True they're also saved to the research library, ready for
    search_library.

    Args:
        url: Where to start, e.g. "https://fastapi.tiangolo.com/tutorial/".
        max_pages: Most pages to read (up to 50).
        max_depth: Most clicks away from the start page to go (0 reads only the start page).
        save: Also save every page read to the research library.
        tags: Tags for the saved pages (used with save=True).
    """
    if not 1 <= max_pages <= MAX_CRAWL_PAGES:
        raise ToolError(f"max_pages must be between 1 and {MAX_CRAWL_PAGES}")
    if not 0 <= max_depth <= MAX_DEPTH:
        raise ToolError(f"max_depth must be between 0 and {MAX_DEPTH}")
    start = _normalize(url)
    try:
        found = await crawl.crawl(start, max_pages, max_depth)
    except crawl.CrawlError as error:
        raise ToolError(str(error)) from error
    if not found.pages:
        raise ToolError("Couldn't read any pages:\n" + "\n".join(
            f"- {failed}: {message}" for failed, message in found.failed.items()))
    cache = _cache()
    for requested, page in found.pages:
        cache.put(requested, page)

    count = len(found.pages)
    heading = f"Crawled {start}: read {count} page{'s' * (count != 1)}"
    if found.more:
        heading += f" ({found.more} more found but not read; raise max_pages to read them)"
    sections = [heading]
    for number, (_, page) in enumerate(found.pages, 1):
        sections.append(f"{number}. **{page.title or page.url}**\n   {page.url}\n"
                        f"   {page.kind}, {len(page.text):,} characters · {_preview(page)}")
    if found.failed:
        sections.append(f"Couldn't read {len(found.failed)}:\n" + "\n".join(
            f"- {failed}: {message}" for failed, message in found.failed.items()))
    if found.blocked:
        sections.append(f"{found.blocked} link{'s' * (found.blocked != 1)} not followed because "
                        "the site's robots.txt asks bots not to.")
    if save:
        library = _library()
        for _, page in found.pages:
            await _library_call(library.save, page, "", tags)
        sections.append(f"Saved all {count} to the library"
                        + (f" with tags {', '.join(tags)}." if tags else "."))
    sections.append("Every page read is cached: fetch_page reads any of them in full right away.")
    return "\n\n".join(sections)
