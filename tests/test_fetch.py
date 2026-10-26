import pytest

from research_mcp.fetch import FetchError, check_url, fetch

pytestmark = pytest.mark.anyio


async def test_fetches_a_page():
    response = await fetch("https://blog.example/rsa")
    assert response.url == "https://blog.example/rsa"
    assert response.content_type.startswith("text/html")
    assert b"How RSA Works" in response.body


async def test_follows_redirects():
    response = await fetch("https://blog.example/old-rsa")
    assert response.url == "https://blog.example/rsa"


@pytest.mark.parametrize(
    "url, message",
    [
        ("file:///etc/passwd", "Only http:// and https://"),
        ("ftp://blog.example/file", "Only http:// and https://"),
        ("https://", "No website name"),
        ("http://localhost:8000/", "private or local address"),
        ("http://127.0.0.1/", "private or local address"),
        ("http://169.254.169.254/latest/meta-data", "private or local address"),  # Cloud metadata.
        ("http://192.168.1.1/", "private or local address"),  # A home router.
        ("http://[::1]/", "private or local address"),
        ("https://intranet.example/", "private or local address"),  # Public name, private IP.
        ("https://no-such-site.example/", "DNS lookup failed"),
    ],
)
async def test_refuses_unsafe_urls(url, message):
    with pytest.raises(FetchError, match=message):
        await check_url(url)


async def test_refuses_redirect_to_a_private_address(fake_internet):
    with pytest.raises(FetchError, match="private or local address"):
        await fetch("https://blog.example/sneaky")
    assert fake_internet == ["https://blog.example/sneaky"]  # Never requested 127.0.0.1.


async def test_gives_up_on_redirect_loops():
    with pytest.raises(FetchError, match="Too many redirects"):
        await fetch("https://blog.example/loop")


async def test_reports_http_errors():
    with pytest.raises(FetchError, match="HTTP 404"):
        await fetch("https://blog.example/missing")


async def test_refuses_huge_files():
    with pytest.raises(FetchError, match="too big"):
        await fetch("https://blog.example/huge")
