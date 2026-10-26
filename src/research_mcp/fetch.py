"""Downloading web pages, safely.

The AI decides which URLs to fetch, and a web page it reads could try to trick it into fetching
something it shouldn't. So this only fetches public http(s) addresses: no file://, no localhost,
no addresses on the user's own network (like a router's admin page or a local dev server).

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

HEADERS = {
    # Say honestly who we are; some sites block unknown bots, and that's their call.
    "User-Agent": "research-mcp/0.1 (+https://github.com/Jal246/research-mcp)",
    "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.8",
}
# Give up on files bigger than this (counted after decompression) or servers slower than this.
MAX_BYTES = 10_000_000
TIMEOUT_SECONDS = 20
MAX_REDIRECTS = 5

# Tests swap in a fake network (httpx.MockTransport) here; None means the real internet.
TRANSPORT: httpx.AsyncBaseTransport | None = None


# Server answers that mean "busy or broken right now", so trying again later might work.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class FetchError(Exception):
    """A problem the user should hear about, like a 404 or a blocked address."""

    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status_code = status_code  # The HTTP status, if the server answered.
        self.retryable = retryable  # Could the same request work if we try again?
        self.retry_after = retry_after  # Seconds the server asked us to wait, if it said.


def _retry_after(headers: httpx.Headers) -> float | None:
    """The server's Retry-After header in seconds (only the number form; dates are rare)."""
    value = headers.get("retry-after", "").strip()
    return float(value) if value.isdigit() else None


@dataclass(frozen=True)
class Response:
    url: str  # Where we ended up, after redirects.
    content_type: str  # E.g. "text/html; charset=utf-8", or "" if the server didn't say.
    body: bytes


async def resolve(host: str) -> list[str]:
    """The IP addresses a host name points to."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


async def check_url(url: str) -> None:
    """Raise FetchError unless url is http(s) and its host only points to public addresses."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError(f"Only http:// and https:// URLs can be fetched, got {url!r}")
    host = parts.hostname
    if not host:
        raise FetchError(f"No website name in {url!r}")
    try:
        addresses = await resolve(host)
    except (socket.gaierror, UnicodeError):
        raise FetchError(f"Couldn't find the website {host!r} (DNS lookup failed)") from None
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])  # Drop an IPv6 zone like "%eth0".
        if not ip.is_global:
            raise FetchError(f"Refusing to fetch {host}: it points to a private or local address ({ip})")


async def fetch(url: str, headers: dict[str, str] | None = None) -> Response:
    """Download url, following redirects (each one re-checked with check_url).

    headers are sent on top of HEADERS, e.g. an API key or a different Accept.
    """
    async with httpx.AsyncClient(
        transport=TRANSPORT, headers=HEADERS | (headers or {}), timeout=TIMEOUT_SECONDS,
        follow_redirects=False,
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await check_url(url)
            try:
                async with client.stream("GET", url) as response:
                    if response.is_redirect and "location" in response.headers:
                        url = urljoin(url, response.headers["location"])
                        continue
                    if response.status_code >= 400:
                        raise FetchError(
                            f"{url} returned HTTP {response.status_code} ({response.reason_phrase})",
                            status_code=response.status_code,
                            retryable=response.status_code in RETRYABLE_STATUSES,
                            retry_after=_retry_after(response.headers),
                        )
                    length = response.headers.get("content-length", "")
                    if length.isdigit() and int(length) > MAX_BYTES:
                        raise FetchError(f"{url} is too big ({int(length):,} bytes; the limit is {MAX_BYTES:,})")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) > MAX_BYTES:
                            raise FetchError(f"{url} is too big (over {MAX_BYTES:,} bytes)")
                    return Response(str(response.url), response.headers.get("content-type", ""), bytes(body))
            except httpx.TimeoutException:
                raise FetchError(
                    f"{url} took longer than {TIMEOUT_SECONDS} seconds to respond", retryable=True
                ) from None
            except httpx.TransportError as error:  # Connection refused, reset, etc.
                raise FetchError(f"Couldn't fetch {url}: {error}", retryable=True) from None
            except httpx.HTTPError as error:
                raise FetchError(f"Couldn't fetch {url}: {error}") from None
    raise FetchError(f"Too many redirects (more than {MAX_REDIRECTS})")
