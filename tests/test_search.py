import pytest

from research_mcp import search
from research_mcp.cache import SearchCache
from research_mcp.search import Result, SearchError, plain, same_page_key

from conftest import FIXTURES

pytestmark = pytest.mark.anyio


def test_plain():
    assert plain("The <b>RSA</b> &amp; <span class='x'>DSA</span>\n  systems") == "The RSA & DSA systems"
    assert plain(None) == ""


def test_same_page_key():
    assert same_page_key("http://www.Example.org/primes/") == same_page_key("https://example.org/primes")
    assert same_page_key("https://example.org/a?page=1") != same_page_key("https://example.org/a?page=2")


def test_parse_duckduckgo():
    results = search.parse_duckduckgo((FIXTURES / "duckduckgo.html").read_bytes())
    assert results == [  # No ad; real URLs instead of DuckDuckGo's redirect links.
        Result("RSA cryptosystem - Wikipedia", "https://en.wikipedia.org/wiki/RSA_cryptosystem",
               "The RSA (Rivest–Shamir–Adleman) cryptosystem is a family of public-key cryptosystems.",
               "duckduckgo"),
        Result("How RSA Works & Why", "https://blog.example/rsa?ref=ddg",
               "A friendly walk through key generation.", "duckduckgo"),
        Result("Prime numbers", "https://www.example.org/primes/", "", "duckduckgo"),
    ]


def test_duckduckgo_captcha_is_an_error():
    with pytest.raises(SearchError, match="CAPTCHA"):
        search.parse_duckduckgo((FIXTURES / "duckduckgo_captcha.html").read_bytes())


async def test_duckduckgo_limit():
    assert len(await search.search_duckduckgo("rsa", 2)) == 2


async def test_wikipedia(fake_internet):
    results = await search.search_wikipedia("rsa", 2)
    assert results[0] == Result(
        "RSA cryptosystem", "https://en.wikipedia.org/wiki/RSA_cryptosystem",
        "The RSA (Rivest–Shamir–Adleman) cryptosystem", "wikipedia", details="last edited 2026-10-16",
    )
    assert results[1].url == "https://en.wikipedia.org/wiki/Euler%27s_totient_function"
    assert "srsearch=rsa" in fake_internet[0] and "srlimit=2" in fake_internet[0]


async def test_github():
    results = await search.search_github("mcp", 5)
    assert results[0] == Result(
        "modelcontextprotocol/python-sdk", "https://github.com/modelcontextprotocol/python-sdk",
        "The official Python SDK for Model Context Protocol servers and clients", "github",
        date="2026-10-25", details="★ 24,666 · Python",
    )
    assert results[1].snippet == "" and results[1].details == "★ 3 · archived"


async def test_github_token_is_sent(monkeypatch):
    sent = []
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    original = search.polite.get

    async def spy(url, interval=0.0, headers=None):
        sent.append(headers)
        return await original(url, interval, headers)

    monkeypatch.setattr(search.polite, "get", spy)
    await search.search_github("mcp", 5)
    assert sent[0]["Authorization"] == "Bearer secret-token"


async def test_hackernews():
    results = await search.search_hackernews("rsa", 5)
    assert results[0] == Result(
        "We updated our RSA SSH host key", "https://github.blog/2023-03-23-we-updated-our-rsa-ssh-host-key/",
        "", "hackernews", date="2023-03-24",
        details="712 points · 330 comments · discussion: https://news.ycombinator.com/item?id=35285390",
    )
    # An "Ask HN" post links to its own discussion.
    assert results[1].url == "https://news.ycombinator.com/item?id=46970360"
    assert results[1].snippet == "With quantum computers coming, should I switch & how?"


async def test_web_search_skips_pages_already_found():
    results, errors = await search.web_search("rsa", ["duckduckgo", "wikipedia"], 5)
    assert errors == {}
    assert [r.title for r in results["wikipedia"]] == ["Euler's totient function"]  # RSA was a dup.


async def test_one_broken_source_doesnt_break_the_others():
    results, errors = await search.web_search("down", ["github", "wikipedia"], 5)
    assert list(results) == ["wikipedia"]
    assert "HTTP 503" in errors["github"]


async def test_nonsense_answers_are_reported():
    results, errors = await search.web_search("garbage", ["wikipedia"], 5)
    assert results == {}
    assert "didn't return valid JSON" in errors["wikipedia"]


async def test_unexpected_format_is_reported(monkeypatch):
    async def broken(query, limit):
        return [{}["title"]]

    monkeypatch.setitem(search.SOURCES, "github", broken)
    results, errors = await search.web_search("rsa", ["github"], 5)
    assert errors == {"github": "couldn't understand its answer (KeyError: 'title')"}


async def test_results_are_cached(tmp_path, fake_internet):
    cache = SearchCache(tmp_path / "searches.db", max_age_seconds=100)
    first = await search.web_search("rsa", ["github", "hackernews"], 5, cache)
    second = await search.web_search("rsa", ["github", "hackernews"], 5, cache)
    assert first == second
    assert len(fake_internet) == 2
    await search.web_search("rsa", ["github"], 3, cache)  # A different limit is a new search.
    assert len(fake_internet) == 3


async def test_errors_are_not_cached(tmp_path, fake_internet):
    cache = SearchCache(tmp_path / "searches.db", max_age_seconds=100)
    await search.web_search("garbage", ["wikipedia"], 5, cache)
    await search.web_search("garbage", ["wikipedia"], 5, cache)
    assert len(fake_internet) == 2
