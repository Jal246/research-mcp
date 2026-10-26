import pytest

from research_mcp import crawl

pytestmark = pytest.mark.anyio

START = "https://docs.example/docs/"


def test_links():
    body = b"""<html><body>
        <a href="intro">Intro</a> <a href="/docs/install#steps">Install</a> <a href="intro">Again</a>
        <a href="mailto:a@b.example">Mail</a> <a href="javascript:void(0)">JS</a> <a>No href</a>
        <a href=" https://other.example/x ">Other</a></body></html>"""
    assert crawl.links("https://docs.example/docs/", body) == [
        "https://docs.example/docs/intro", "https://docs.example/docs/install", "https://other.example/x",
    ]


def test_links_use_the_base_tag():
    body = b'<html><head><base href="/v2/"></head><body><a href="guide">Guide</a></body></html>'
    assert crawl.links("https://docs.example/docs/", body) == ["https://docs.example/v2/guide"]


def test_links_in_garbage():
    assert crawl.links("https://docs.example/", b"") == []


def test_folder():
    assert crawl.folder("https://x.example/docs/guide/intro.html") == "/docs/guide/"
    assert crawl.folder("https://x.example/docs/") == "/docs/"
    assert crawl.folder("https://x.example") == "/"


def test_in_scope():
    assert crawl.in_scope("https://www.docs.example/docs/a", START)
    assert crawl.in_scope("http://docs.example/docs/a/b?page=2", START)
    assert not crawl.in_scope("https://docs.example/about", START)
    assert not crawl.in_scope("https://other.example/docs/a", START)
    assert not crawl.in_scope("https://docs.example/docs/logo.PNG", START)


async def test_crawl(fake_internet):
    found = await crawl.crawl(START, max_pages=10, max_depth=2)
    assert [url for url, _ in found.pages] == [
        START, "https://docs.example/docs/intro", "https://docs.example/docs/install",
        "https://docs.example/docs/advanced",
    ]
    assert [page.title for _, page in found.pages] == [
        "Example Docs", "Introduction", "Installation", "Advanced brewing"]
    assert list(found.failed) == ["https://docs.example/docs/missing"]
    assert "HTTP 404" in found.failed["https://docs.example/docs/missing"]
    assert (found.blocked, found.more) == (1, 0)  # /docs/private/keys; /docs/deeper is too deep.
    assert not any(url.endswith(("/about", "logo.png", "/private/keys")) for url in fake_internet)


async def test_crawl_stops_at_max_pages():
    found = await crawl.crawl(START, max_pages=2, max_depth=5)
    assert len(found.pages) == 2
    assert found.more == 2


async def test_crawl_depth_zero_reads_only_the_start(fake_internet):
    found = await crawl.crawl(START, max_pages=10, max_depth=0)
    assert [url for url, _ in found.pages] == [START]
    assert fake_internet == ["https://docs.example/robots.txt", START]


async def test_crawl_waits_between_requests(sleeps):
    await crawl.crawl(START, max_pages=3, max_depth=1)
    # The start page goes right away; then two pages are fetched at once, and the second books
    # the slot after the first, so they wait half a second and a second.
    assert sleeps == pytest.approx([0.5, 1.0], abs=0.1)


async def test_crawl_respects_robots_txt_for_the_start_page():
    with pytest.raises(crawl.CrawlError, match="robots.txt asks bots not to read"):
        await crawl.crawl("https://docs.example/docs/private/keys", max_pages=5, max_depth=1)


async def test_crawl_without_robots_txt(fake_internet):
    found = await crawl.crawl("https://blog.example/rsa", max_pages=5, max_depth=1)
    assert found.pages[0][1].title == "How RSA Works"
    assert fake_internet[0] == "https://blog.example/robots.txt"  # A 404: everything allowed.


async def test_crawl_follows_a_redirected_start():
    found = await crawl.crawl("https://blog.example/old-rsa", max_pages=5, max_depth=0)
    assert [(url, page.url) for url, page in found.pages] == [
        ("https://blog.example/old-rsa", "https://blog.example/rsa")]
