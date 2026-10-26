"""Searching the web with free sources: DuckDuckGo, Wikipedia, GitHub and Hacker News.

None of these need an API key (GitHub allows more searches with one, via GITHUB_TOKEN). Each
source is a function that takes a query and returns a list of Results; run_sources runs several
at once, so one slow or broken source doesn't hold up or break the others.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import asyncio
import html
import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, quote, urlencode, urlsplit, urlunsplit

import lxml.html

from research_mcp import fetch, polite
from research_mcp.cache import SearchCache


class SearchError(Exception):
    """A source answered, but not with results (e.g. a CAPTCHA, or a format we don't understand)."""


@dataclass(frozen=True)
class Result:
    title: str
    url: str
    snippet: str  # A sentence or two from the page, or its description.
    source: str  # Which source found it, e.g. "wikipedia".
    date: str | None = None  # YYYY-MM-DD, when the source says.
    details: str | None = None  # Extra facts, e.g. "★ 1,234 · Python" for a GitHub repo.

    @classmethod
    def from_json(cls, data: dict) -> "Result":
        return cls(**data)


def plain(text: str | None) -> str:
    """Text without HTML tags or entities, and with whitespace tidied: "<b>RSA</b>&amp;" -> "RSA&"."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", text or "")).split())


def parse_json(response: fetch.Response) -> dict:
    try:
        return json.loads(response.body)
    except ValueError:
        raise SearchError(f"{response.url} didn't return valid JSON") from None


# --- DuckDuckGo -------------------------------------------------------------------------------
# DuckDuckGo has no free search API, but its no-JavaScript page (html.duckduckgo.com) is simple
# HTML. If it decides we're a bot it shows a CAPTCHA instead, which we report rather than solve.

def parse_duckduckgo(page: bytes) -> list[Result]:
    tree = lxml.html.fromstring(page)
    results = []
    for body in tree.find_class("result__body"):
        if "result--ad" in (body.getparent().get("class") or ""):
            continue  # An ad.
        links = body.find_class("result__a")
        if not links:
            continue
        # Links go through DuckDuckGo's redirector: //duckduckgo.com/l/?uddg=<the real URL>.
        href = links[0].get("href", "")
        url = parse_qs(urlsplit(href).query).get("uddg", [href])[0]
        if not url.startswith(("http://", "https://")) or "duckduckgo.com/y.js" in url:
            continue  # Ads use y.js links.
        snippets = body.find_class("result__snippet")
        results.append(Result(
            title=plain(links[0].text_content()),
            url=url,
            snippet=plain(snippets[0].text_content()) if snippets else "",
            source="duckduckgo",
        ))
    if not results and (b"anomaly" in page or b"challenge" in page):
        raise SearchError("DuckDuckGo thinks we're a bot right now and asked for a CAPTCHA; "
                          "try again in a few minutes, or use other sources")
    return results


async def search_duckduckgo(query: str, limit: int) -> list[Result]:
    response = await polite.get(
        "https://html.duckduckgo.com/html/?" + urlencode({"q": query}), interval=1.0
    )
    return parse_duckduckgo(response.body)[:limit]


# --- Wikipedia --------------------------------------------------------------------------------

async def search_wikipedia(query: str, limit: int) -> list[Result]:
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit,
              "srprop": "snippet|timestamp", "format": "json", "formatversion": 2}
    response = await polite.get("https://en.wikipedia.org/w/api.php?" + urlencode(params))
    data = parse_json(response)
    return [
        Result(
            title=hit["title"],
            url="https://en.wikipedia.org/wiki/" + quote(hit["title"].replace(" ", "_")),
            snippet=plain(hit.get("snippet")),
            source="wikipedia",
            details=f"last edited {hit['timestamp'][:10]}" if hit.get("timestamp") else None,
        )
        for hit in data.get("query", {}).get("search", [])
    ]


# --- GitHub -----------------------------------------------------------------------------------
# Without a token GitHub allows 10 searches a minute; with GITHUB_TOKEN set, 30.

async def search_github(query: str, limit: int) -> list[Result]:
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    url = "https://api.github.com/search/repositories?" + urlencode({"q": query, "per_page": limit})
    try:
        response = await polite.get(url, headers=headers)
    except fetch.FetchError as error:
        if error.status_code in (403, 429):  # GitHub uses 403 for "rate limit exceeded".
            raise SearchError("GitHub's search limit was reached (10 searches a minute without a "
                              "token); wait a minute, or set GITHUB_TOKEN for more") from None
        raise
    results = []
    for repo in parse_json(response).get("items", []):
        details = [f"★ {repo.get('stargazers_count', 0):,}"]
        if repo.get("language"):
            details.append(repo["language"])
        if repo.get("archived"):
            details.append("archived")
        results.append(Result(
            title=repo["full_name"],
            url=repo["html_url"],
            snippet=plain(repo.get("description")),
            source="github",
            date=(repo.get("pushed_at") or "")[:10] or None,
            details=" · ".join(details),
        ))
    return results


# --- Hacker News ------------------------------------------------------------------------------
# Algolia runs the official Hacker News search. Good for opinions and experience reports.

async def search_hackernews(query: str, limit: int) -> list[Result]:
    params = {"query": query, "tags": "story", "hitsPerPage": limit}
    response = await polite.get("https://hn.algolia.com/api/v1/search?" + urlencode(params))
    results = []
    for hit in parse_json(response).get("hits", []):
        discussion = f"https://news.ycombinator.com/item?id={hit['objectID']}"
        results.append(Result(
            title=plain(hit.get("title")),
            url=hit.get("url") or discussion,  # "Ask HN" posts have no link of their own.
            snippet=plain(hit.get("story_text"))[:300],
            source="hackernews",
            date=(hit.get("created_at") or "")[:10] or None,
            details=f"{hit.get('points') or 0:,} points · {hit.get('num_comments') or 0:,} comments"
                    f" · discussion: {discussion}",
        ))
    return results


SOURCES: dict[str, Callable[[str, int], Awaitable[list[Result]]]] = {
    "duckduckgo": search_duckduckgo,
    "wikipedia": search_wikipedia,
    "github": search_github,
    "hackernews": search_hackernews,
}


# --- Running several sources ------------------------------------------------------------------

async def run_sources[T](
    searches: dict[str, Callable[[], Awaitable[list[T]]]],
    limit: int,
    cache: SearchCache | None,
    cache_key: Callable[[str], str],
    from_json: Callable[[dict], T],
) -> tuple[dict[str, list[T]], dict[str, str]]:
    """Run each named search at once, using cached results where we have them.

    Returns (results by source, at most `limit` each; error message by source). Failed searches
    aren't cached.
    """
    async def run(name: str) -> list[T]:
        if cache and (cached := cache.get(cache_key(name))) is not None:
            return [from_json(item) for item in cached]
        found = (await searches[name]())[:limit]  # In case a source ignores the limit we asked for.
        if cache:
            cache.put(cache_key(name), [asdict(item) for item in found])
        return found

    names = list(searches)
    outcomes = await asyncio.gather(*(run(name) for name in names), return_exceptions=True)
    results, errors = {}, {}
    for name, outcome in zip(names, outcomes):
        if isinstance(outcome, (fetch.FetchError, SearchError)):
            errors[name] = str(outcome)
        elif isinstance(outcome, (KeyError, TypeError, ValueError, AttributeError)):
            # The source changed its format; report it instead of crashing the whole search.
            errors[name] = f"couldn't understand its answer ({type(outcome).__name__}: {outcome})"
        elif isinstance(outcome, BaseException):
            raise outcome
        else:
            results[name] = outcome
    return results, errors


def same_page_key(url: str) -> str:
    """A URL with the unimportant differences removed, to spot the same page from two sources."""
    parts = urlsplit(url)
    host = (parts.hostname or "").removeprefix("www.")
    return urlunsplit(("https", host, parts.path.rstrip("/"), parts.query, ""))


async def web_search(
    query: str, sources: list[str], limit: int, cache: SearchCache | None = None
) -> tuple[dict[str, list[Result]], dict[str, str]]:
    """Search each source for up to `limit` results. A page found by an earlier source in
    `sources` isn't repeated under a later one."""
    results, errors = await run_sources(
        {name: (lambda name=name: SOURCES[name](query, limit)) for name in sources}, limit,
        cache, lambda name: f"{name}|{query}|{limit}", Result.from_json,
    )
    seen = set()
    for name in sources:
        if name in results:
            unique = [r for r in results[name] if same_page_key(r.url) not in seen]
            seen.update(same_page_key(r.url) for r in unique)
            results[name] = unique
    return results, errors
