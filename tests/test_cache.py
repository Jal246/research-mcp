import pytest

from research_mcp.cache import PageCache, SearchCache
from research_mcp.extract import Page

PAGE = Page("https://blog.example/rsa", "web page", "# How RSA Works", "How RSA Works", "Crypto Blog",
            "Ada Example", "2025-03-14")


@pytest.fixture
def cache(tmp_path):
    return PageCache(tmp_path / "pages.db", max_age_seconds=100)


def test_empty_cache(cache):
    assert cache.get("https://blog.example/rsa") is None


def test_returns_what_was_stored(cache):
    cache.put("https://blog.example/old-rsa", PAGE, now=1000)
    assert cache.get("https://blog.example/old-rsa", now=1050) == PAGE


def test_old_pages_expire(cache):
    cache.put("https://blog.example/rsa", PAGE, now=1000)
    assert cache.get("https://blog.example/rsa", now=1101) is None


def test_put_replaces(cache):
    cache.put("https://blog.example/rsa", PAGE, now=1000)
    newer = Page(PAGE.url, "web page", "updated")
    cache.put("https://blog.example/rsa", newer, now=1010)
    assert cache.get("https://blog.example/rsa", now=1020) == newer


def test_search_cache(tmp_path):
    cache = SearchCache(tmp_path / "searches.db", max_age_seconds=100)
    assert cache.get("wikipedia|rsa|5") is None
    results = [{"title": "RSA", "authors": ["Ron Rivest"], "year": 1977}]
    cache.put("wikipedia|rsa|5", results, now=1000)
    assert cache.get("wikipedia|rsa|5", now=1050) == results
    assert cache.get("wikipedia|rsa|5", now=1101) is None
    assert cache.get("wikipedia|rsa|10", now=1050) is None
