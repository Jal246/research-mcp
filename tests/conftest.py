"""A fake internet for the tests, so they never touch the real network."""

from pathlib import Path

import httpx
import pytest

from research_mcp import fetch, polite

FIXTURES = Path(__file__).parent / "fixtures"

# Fake DNS: what each test host name points to.
FAKE_DNS = {
    "blog.example": ["93.184.215.14"],
    "papers.example": ["93.184.215.15"],
    "intranet.example": ["10.0.0.5"],
    # The real search APIs, answered by fake_api.
    "html.duckduckgo.com": ["52.142.124.215"],
    "en.wikipedia.org": ["208.80.154.224"],
    "api.github.com": ["140.82.112.5"],
    "hn.algolia.com": ["104.18.0.1"],
    "export.arxiv.org": ["128.84.21.199"],
    "api.semanticscholar.org": ["13.32.0.1"],
}

# Which fixture file each search API answers with, and its content type.
API_ANSWERS = {
    "html.duckduckgo.com": ("duckduckgo.html", "text/html"),
    "en.wikipedia.org": ("wikipedia.json", "application/json"),
    "api.github.com": ("github.json", "application/json"),
    "hn.algolia.com": ("hackernews.json", "application/json"),
    "export.arxiv.org": ("arxiv.xml", "application/atom+xml"),
    "api.semanticscholar.org": ("semantic_scholar.json", "application/json"),
}
EMPTY_ANSWERS = {
    "html.duckduckgo.com": "<html><body><div class='no-results'>No results.</div></body></html>",
    "en.wikipedia.org": '{"query": {"search": []}}',
    "api.github.com": '{"items": []}',
    "hn.algolia.com": '{"hits": []}',
    "export.arxiv.org": '<feed xmlns="http://www.w3.org/2005/Atom"></feed>',
    "api.semanticscholar.org": '{"total": 0}',
}


def fake_api(request: httpx.Request, earlier: list[str]) -> httpx.Response:
    """Answer like the real search APIs. Some queries trigger problems:

    "nothing" finds nothing anywhere; "bot check" gets DuckDuckGo's CAPTCHA; "down" breaks
    GitHub; "garbage" makes Wikipedia answer with nonsense; "busy" makes Semantic Scholar answer
    429 once (asking for a 2 second wait); "always busy" makes it answer 429 every time.
    """
    host = request.url.host
    params = request.url.params
    query = params.get("q") or params.get("srsearch") or params.get("query") or params.get("search_query", "")
    tries = sum(url.startswith(f"https://{host}/") for url in earlier)  # Including this one.
    if "nothing" in query:
        return httpx.Response(200, text=EMPTY_ANSWERS[host])
    match host, query:
        case "html.duckduckgo.com", "bot check":
            return httpx.Response(202, content=(FIXTURES / "duckduckgo_captcha.html").read_bytes())
        case "api.github.com", "down":
            return httpx.Response(503)
        case "en.wikipedia.org", "garbage":
            return httpx.Response(200, text="<html>Oops</html>")
        case "api.semanticscholar.org", _ if "always busy" in query or ("busy" in query and tries == 1):
            return httpx.Response(429, headers={"retry-after": "2"}, json={"message": "Too Many Requests"})
    name, content_type = API_ANSWERS[host]
    return httpx.Response(200, headers={"content-type": content_type},
                          content=(FIXTURES / name).read_bytes())


def fake_site(request: httpx.Request, earlier: list[str]) -> httpx.Response:
    """Answer a request the way a few made-up websites would (earlier: the URLs asked for so far)."""
    if request.url.host in API_ANSWERS:
        return fake_api(request, earlier)
    url = str(request.url)
    match url:
        case "https://blog.example/rsa":
            return httpx.Response(
                200, headers={"content-type": "text/html; charset=utf-8"},
                content=(FIXTURES / "article.html").read_bytes(),
            )
        case "https://blog.example/old-rsa":
            return httpx.Response(301, headers={"location": "/rsa"})
        case "https://blog.example/sneaky":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})
        case "https://blog.example/loop":
            return httpx.Response(302, headers={"location": "/loop"})
        case "https://blog.example/notes.txt":
            return httpx.Response(200, headers={"content-type": "text/plain"}, content="line one\n" * 5000)
        case "https://blog.example/huge":
            return httpx.Response(200, headers={"content-type": "text/plain"},
                                  content=b"x" * (fetch.MAX_BYTES + 1))
        case "https://blog.example/cat.png":
            return httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG...")
        case "https://papers.example/paper.pdf":
            return httpx.Response(200, headers={"content-type": "application/pdf"},
                                  content=(FIXTURES / "sample.pdf").read_bytes())
    return httpx.Response(404)


@pytest.fixture(autouse=True)
def fake_internet(monkeypatch):
    """Route every fetch to fake_site, and resolve host names with FAKE_DNS.

    Also counts requests (fake_internet.requests), so tests can check the cache is used.
    """
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return fake_site(request, requests)

    async def resolve(host: str) -> list[str]:
        if host in FAKE_DNS:
            return FAKE_DNS[host]
        if host == "localhost":
            return ["127.0.0.1", "::1"]
        if host[0].isdigit() or ":" in host:  # Already an IP address.
            return [host]
        raise fetch.socket.gaierror(f"unknown host {host}")

    monkeypatch.setattr(fetch, "TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(fetch, "resolve", resolve)
    return requests


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Don't really wait between requests or before retries; record how long we'd have waited."""
    waits = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(polite, "SLEEP", sleep)
    monkeypatch.setattr(polite, "_next_turn", {})
    return waits
