import pytest

from research_mcp import fetch, report
from research_mcp.extract import ExtractError, Page, extract
from research_mcp.library import Library
from research_mcp.papers import Paper
from research_mcp.search import Result

pytestmark = pytest.mark.anyio


def candidate(title, url, snippet=""):
    return report.Candidate(title, url, snippet, "web")


def test_from_result_and_paper():
    result = Result("RSA", "https://a.example", "About RSA", "duckduckgo", "2025-01-02", "★ 5")
    assert report.from_result(result) == report.Candidate(
        "RSA", "https://a.example", "About RSA", "web", "2025-01-02 · ★ 5")
    paper = Paper("Attention", ("A", "B", "C", "D"), 2017, "Abstract", "https://arxiv.org/abs/1",
                  "arxiv", pdf_url="https://arxiv.org/pdf/1", citations=1000)
    assert report.from_paper(paper) == report.Candidate(
        "Attention", "https://arxiv.org/pdf/1", "Abstract", "paper", "2017 · A, B, C · 1,000 citations")


def test_rank(embedder):
    candidates = [
        candidate("Baking bread", "https://bread.example/"),
        candidate("RSA keys", "https://one.example/a", "rsa primes"),
        candidate("RSA again", "https://one.example/a/", "rsa primes"),  # Same page.
        candidate("RSA saved", "https://saved.example/", "rsa primes"),
        candidate("RSA elsewhere", "https://two.example/", "rsa"),
    ]
    query = embedder.embed_query("rsa primes")
    ranked = report.rank(candidates, query, embedder, skip={"https://saved.example"})
    assert [c.title for c in ranked] == ["RSA keys", "RSA elsewhere", "Baking bread"]
    assert report.rank([], query, embedder, skip=set()) == []


def test_take():
    ranked = [candidate("A1", "https://a.example/1"), candidate("A2", "https://a.example/2"),
              candidate("A3", "https://a.example/3"), candidate("B", "https://b.example/"),
              candidate("C", "https://c.example/"), candidate("D", "https://d.example/")]
    per_site = report.Counter()
    taken = report.take(ranked, 3, per_site, refused={"b.example"})
    assert [c.title for c in taken] == ["A1", "A2", "C"]  # A3: two from a.example already.
    assert [c.title for c in ranked] == ["D"]
    assert [c.title for c in report.take(ranked, 3, per_site, set())] == ["D"]
    assert ranked == [] and report.take(ranked, 3, per_site, set()) == []


def test_shortlist():
    pieces = ["Intro about bananas.", "Leader election uses votes.", "Bananas again.",
              "Votes votes votes for the leader in an election.", "Nothing here."]
    assert report.shortlist(pieces, "How does leader election work?", 2) == [0, 1, 3]
    assert report.shortlist(pieces, "leader", 10) == [0, 1, 2, 3, 4]
    assert report.shortlist([], "leader", 2) == []


def test_best_passages(embedder):
    text = "\n\n".join(["Bananas are yellow. " * 40, "Raft elects a leader by votes. " * 25,
                        "Leader election needs votes. " * 30, "More about bananas. " * 40])
    page = Page("https://x.example", "web page", text)
    question = "leader election votes"
    passages = report.best_passages(page, question, embedder.embed_query(question), embedder)
    assert len(passages) == 2  # The banana passages score too low.
    assert passages[0].score >= passages[1].score
    assert all(text[p.start:].startswith(p.text) for p in passages)
    empty = Page("https://x.example", "text", "")
    assert report.best_passages(empty, "x", embedder.embed_query("x"), embedder) == []


def test_best_passages_only_embeds_a_shortlist(embedder, monkeypatch):
    monkeypatch.setattr(report, "EMBED_PER_SOURCE", 2)
    text = "\n\n".join([f"Paragraph {i} about bananas. " * 40 for i in range(10)]
                       + ["Leader election needs votes. " * 30])
    page = Page("https://x.example", "web page", text)
    passages = report.best_passages(page, "leader election", embedder.embed_query("leader election"), embedder)
    # Of 20 chunks: the one using the question's words, and the opening (which, tied with the
    # other banana chunks at no matching words, is also the next best).
    assert len(embedder.embedded) == 2
    assert "Leader election" in passages[0].text


def test_best_passages_keeps_the_best_even_if_it_scores_low(embedder):
    page = Page("https://x.example", "web page", "Bananas are yellow.")
    question = "leader election"
    passages = report.best_passages(page, question, embedder.embed_query(question), embedder)
    assert [p.text for p in passages] == ["Bananas are yellow."]


async def read(url: str) -> Page:
    response = await fetch.fetch(url)
    return extract(response.url, response.content_type, response.body)


async def test_research(tmp_path, embedder):
    library = Library(tmp_path / "library.db", embedder)
    brief, saved = await report.research("rsa primes", 4, True, read, library, None)
    assert saved == []
    assert brief.searched == ["duckduckgo", "wikipedia", "arxiv", "semantic_scholar"]
    urls = [source.page.url for source in brief.sources]
    assert "https://blog.example/rsa?ref=ddg" in urls
    assert all(source.passages for source in brief.sources)
    scores = [source.passages[0].score for source in brief.sources]
    assert scores == sorted(scores, reverse=True)
    assert len(brief.sources) == 4  # www.example.org failed, so another candidate was read.
    assert list(brief.failed) == ["https://www.example.org/primes/"]


async def test_research_uses_the_library_instead_of_reading_again(tmp_path, embedder, fake_internet):
    library = Library(tmp_path / "library.db", embedder)
    library.save(await read("https://blog.example/rsa"))
    fake_internet.clear()
    brief, saved = await report.research("how rsa works", 6, False, read, library, None)
    assert [hit.document.url for hit in saved] == ["https://blog.example/rsa"]
    assert "https://blog.example/rsa?ref=ddg" not in fake_internet
    assert brief.searched == ["duckduckgo", "wikipedia"]
    assert not any("arxiv" in url for url in fake_internet)


async def test_research_reads_backups_when_sources_fail(tmp_path, embedder):
    tried = []

    async def picky(url: str) -> Page:
        tried.append(url)
        if "wikipedia" in url:
            raise fetch.FetchError(f"{url} returned HTTP 403 (Forbidden)", status_code=403)
        return await read(url)

    library = Library(tmp_path / "library.db", embedder)
    brief, _ = await report.research("rsa primes", 2, True, picky, library, None)
    assert len(brief.sources) == 2  # Backups replaced the refused pages.
    assert sum("wikipedia" in url for url in tried) == 1  # Refused once: the rest of it skipped.
    assert sum("wikipedia" in url for url in brief.failed) == 1


async def test_research_reports_failures(tmp_path, embedder):
    async def broken(url: str) -> Page:
        raise ExtractError(f"{url} is unreadable")

    library = Library(tmp_path / "library.db", embedder)
    brief, _ = await report.research("bot check", 3, False, broken, library, None)
    assert brief.sources == []
    assert "DuckDuckGo thinks we're a bot" in brief.failed["duckduckgo"]
    assert brief.searched == ["wikipedia"]
    assert all("is unreadable" in brief.failed[url] for url in brief.failed if url != "duckduckgo")
