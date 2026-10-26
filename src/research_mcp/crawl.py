"""Reading several pages of one website: a docs site, a blog, a course.

Starting from one page, follow its links to other pages on the same site, breadth first (all
the pages one click away, then two clicks away...), until enough pages have been read. Only
pages under the starting page's folder are followed, so starting at example.org/docs/ reads
the docs, not the whole site.

Crawling is where a client can really bother a website, so this is careful: it follows
robots.txt (the file where sites say what bots may read), waits between requests, and stops
at a fixed number of pages.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import lxml.etree
import lxml.html

from research_mcp import fetch, polite
from research_mcp.extract import ExtractError, Page, extract
from research_mcp.search import same_page_key

# The name robots.txt rules can use for us ("User-agent: research-mcp").
ROBOT_NAME = "research-mcp"
# Seconds between requests to the site, unless its robots.txt asks for longer (up to the max).
INTERVAL = 0.5
MAX_CRAWL_DELAY = 5.0
# Pages fetched at once. polite.get still spaces them out; this just overlaps the waiting.
CONCURRENCY = 3

# Links to these are files we can't read (or don't want to download), so they're not followed.
SKIP_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".css", ".js", ".mjs",
    ".zip", ".gz", ".tgz", ".tar", ".7z", ".rar", ".exe", ".dmg", ".msi", ".whl", ".deb", ".rpm",
    ".mp3", ".mp4", ".webm", ".mov", ".avi", ".woff", ".woff2", ".ttf", ".otf", ".eot",
)


class CrawlError(Exception):
    """A crawl that can't start, like a start page robots.txt doesn't allow."""


@dataclass
class Crawl:
    """What a crawl found."""

    pages: list[tuple[str, Page]] = field(default_factory=list)  # (URL asked for, page), in order read.
    failed: dict[str, str] = field(default_factory=dict)  # URL: what went wrong.
    blocked: int = 0  # Links not followed because robots.txt says no.
    more: int = 0  # Links found but not read because the page limit was reached.


def links(url: str, body: bytes) -> list[str]:
    """The http(s) links in an HTML page, made absolute, without #fragments, each once."""
    try:
        tree = lxml.html.fromstring(body)
    except (lxml.etree.ParserError, ValueError):
        return []
    base = urljoin(url, (tree.xpath("//base/@href") or [""])[0])
    found = []
    for href in tree.xpath("//a/@href"):
        absolute = urljoin(base, href.strip())
        parts = urlsplit(absolute)
        if parts.scheme in ("http", "https"):
            found.append(urlunsplit(parts._replace(fragment="")))
    return list(dict.fromkeys(found))


def folder(url: str) -> str:
    """The folder part of a URL's path: "/docs/guide/intro.html" -> "/docs/guide/"."""
    path = urlsplit(url).path or "/"
    return path[: path.rfind("/") + 1]


def in_scope(url: str, start: str) -> bool:
    """Is url on the same site as start, and under start's folder?"""
    a, b = urlsplit(url), urlsplit(start)
    same_site = (a.hostname or "").removeprefix("www.") == (b.hostname or "").removeprefix("www.")
    return (same_site and (a.path or "/").startswith(folder(start))
            and not a.path.lower().endswith(SKIP_EXTENSIONS))


async def robots(url: str) -> RobotFileParser:
    """The site's robots.txt rules. A site without one (or whose robots.txt is broken) allows everything."""
    parts = urlsplit(url)
    rules = RobotFileParser()
    try:
        response = await polite.get(urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", "")))
        rules.parse(response.body.decode("utf-8", errors="replace").splitlines())
    except fetch.FetchError:
        rules.parse([])
    return rules


async def crawl(start: str, max_pages: int, max_depth: int,
                progress: Callable[[int, int, str], Awaitable[None]] | None = None) -> Crawl:
    """Read up to max_pages pages, starting at start and following links at most max_depth
    clicks away (0 reads only the start page). progress(done, total, message), if given, is
    told about each page read."""
    rules = await robots(start)
    if not rules.can_fetch(ROBOT_NAME, start):
        raise CrawlError(f"The site's robots.txt asks bots not to read {start}")
    interval = min(max(INTERVAL, float(rules.crawl_delay(ROBOT_NAME) or 0)), MAX_CRAWL_DELAY)

    result = Crawl()
    scope = start  # Becomes where the start page really is, if it redirects (e.g. to www.).
    seen = {same_page_key(start)}
    queue = [(start, 0)]

    async def read(url: str) -> tuple[Page, list[str]]:
        response = await polite.get(url, interval)
        page = extract(response.url, response.content_type, response.body)
        return page, links(response.url, response.body) if page.kind == "web page" else []

    while queue and len(result.pages) < max_pages:
        size = min(CONCURRENCY, max_pages - len(result.pages))  # Never read more than max_pages.
        batch, queue = queue[:size], queue[size:]
        outcomes = await asyncio.gather(*(read(url) for url, _ in batch), return_exceptions=True)
        for (url, depth), outcome in zip(batch, outcomes):
            if isinstance(outcome, (fetch.FetchError, ExtractError)):
                result.failed[url] = str(outcome)
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            page, found = outcome
            if not result.pages and url == start:
                scope = page.url
            result.pages.append((url, page))
            if progress:
                await progress(len(result.pages), max_pages, f"Read {page.title or page.url}")
            if depth == max_depth:
                continue
            for link in found:
                if not in_scope(link, scope) or same_page_key(link) in seen:
                    continue
                seen.add(same_page_key(link))
                if rules.can_fetch(ROBOT_NAME, link):
                    queue.append((link, depth + 1))
                else:
                    result.blocked += 1
    result.more = len(queue)
    return result
