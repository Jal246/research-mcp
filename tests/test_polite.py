import pytest

from research_mcp import polite
from research_mcp.fetch import FetchError

pytestmark = pytest.mark.anyio

S2 = "https://api.semanticscholar.org/graph/v1/paper/search?query="


async def test_first_request_doesnt_wait(sleeps):
    await polite.wait_turn("export.arxiv.org", 3.0)
    assert sleeps == []


async def test_requests_to_one_site_are_spaced_out(sleeps):
    for _ in range(3):
        await polite.wait_turn("export.arxiv.org", 3.0)
    # The fake sleep doesn't pass any time, so the 2nd waits ~3s and the 3rd ~6s.
    assert len(sleeps) == 2
    assert 2.9 < sleeps[0] <= 3.0
    assert 5.9 < sleeps[1] <= 6.0


async def test_different_sites_dont_wait_for_each_other(sleeps):
    await polite.wait_turn("export.arxiv.org", 3.0)
    await polite.wait_turn("api.semanticscholar.org", 1.0)
    assert sleeps == []


async def test_retries_when_busy(fake_internet, sleeps):
    response = await polite.get(S2 + "busy")
    assert b"Attention is All you Need" in response.body
    assert len(fake_internet) == 2
    assert sleeps == [2.0]  # The server's Retry-After.


async def test_gives_up_after_three_tries(fake_internet, sleeps):
    with pytest.raises(FetchError, match="HTTP 429") as error:
        await polite.get(S2 + "always+busy")
    assert error.value.status_code == 429
    assert len(fake_internet) == 3
    assert sleeps == [2.0, 2.0]


async def test_backs_off_when_server_doesnt_say_how_long(fake_internet, sleeps):
    with pytest.raises(FetchError, match="HTTP 503"):
        await polite.get("https://api.github.com/search/repositories?q=down")
    assert len(fake_internet) == 3
    assert 1.0 <= sleeps[0] <= 1.5 and 2.0 <= sleeps[1] <= 3.0  # Doubling, with some randomness.


async def test_doesnt_retry_what_cant_work(fake_internet, sleeps):
    with pytest.raises(FetchError, match="HTTP 404"):
        await polite.get("https://blog.example/missing")
    with pytest.raises(FetchError, match="private or local address"):
        await polite.get("http://localhost/")
    assert len(fake_internet) == 1
    assert sleeps == []


async def test_doesnt_wait_too_long(monkeypatch, fake_internet, sleeps):
    monkeypatch.setattr(polite, "MAX_RETRY_WAIT", 1.0)  # The server asks for 2.
    with pytest.raises(FetchError, match="asked us to wait 2 seconds"):
        await polite.get(S2 + "busy")
    assert len(fake_internet) == 1
    assert sleeps == []
