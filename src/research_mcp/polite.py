"""Being a polite client of free search APIs: rate limits and retries.

Free APIs ask you not to hammer them (arXiv asks for one request every 3 seconds), and when
they're busy they answer "429 Too Many Requests" or "503 Service Unavailable". So every request
waits its turn per website, and a busy answer is retried a couple of times, waiting a little
longer each time.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import asyncio
import random
import time
from urllib.parse import urlsplit

from research_mcp import fetch

# How many times to try a request in total, and how long to wait before the 2nd and 3rd tries
# (doubling each time, plus a bit of randomness so many clients don't all retry at once).
ATTEMPTS = 3
FIRST_RETRY_WAIT = 1.0
# If a server asks us to wait longer than this, give up instead: the AI is waiting for an answer.
MAX_RETRY_WAIT = 10.0

# Tests swap in a fake sleep that only records how long it was asked to wait.
SLEEP = asyncio.sleep

# When each website may next be sent a request (time.monotonic() seconds).
_next_turn: dict[str, float] = {}


async def wait_turn(host: str, interval: float) -> None:
    """Wait until at least `interval` seconds have passed since the last request to host.

    Requests made at the same time line up: each one books the next free slot before sleeping.
    (No lock needed: nothing awaits between reading and booking the slot.)
    """
    now = time.monotonic()
    turn = max(now, _next_turn.get(host, 0.0))
    _next_turn[host] = turn + interval
    if turn > now:
        await SLEEP(turn - now)


async def get(url: str, interval: float = 0.0, headers: dict[str, str] | None = None) -> fetch.Response:
    """fetch.fetch(url), spaced out per website and retried if the server is busy."""
    host = urlsplit(url).hostname or ""
    for attempt in range(ATTEMPTS):
        await wait_turn(host, interval)
        try:
            return await fetch.fetch(url, headers)
        except fetch.FetchError as error:
            if not error.retryable or attempt == ATTEMPTS - 1:
                raise
            wait = error.retry_after
            if wait is None:
                wait = FIRST_RETRY_WAIT * 2**attempt * random.uniform(1.0, 1.5)
            elif wait > MAX_RETRY_WAIT:
                raise fetch.FetchError(
                    f"{host} is busy and asked us to wait {wait:.0f} seconds; try again later",
                    error.status_code,
                ) from None
            await SLEEP(wait)
    raise AssertionError("unreachable")
