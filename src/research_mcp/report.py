"""Researching a question in one go: search, choose sources, read them, keep the best passages.

This is the loop a person (or the AI, one tool call at a time) goes through: search the web and
papers, look at the results, open the most promising few, skim each for the parts that answer
the question. Doing it in one step saves the AI a dozen tool calls and a lot of context: instead
of whole pages it gets a short brief of numbered sources, each with the passages that matter,
ready to write an answer from and cite.

There's no language model here to write the report itself; the AI reading the brief does that.
What this adds is the legwork, and choosing passages by meaning with the embedding model.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import asyncio
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from research_mcp import fetch, papers, search
from research_mcp.cache import SearchCache
from research_mcp.extract import ExtractError, Page
from research_mcp.library import Embedder, Library, chunk
from research_mcp.papers import Paper
from research_mcp.search import Result, same_page_key
from research_mcp.similar import STOP_WORDS, site

# How many search results to consider, per source.
WEB_RESULTS = 8
PAPER_RESULTS = 5
# At most this many sources from one website, so one site can't fill the brief.
PER_SITE = 2
# Only the start of a long page is searched for passages (about 20 chunks): it's where most
# pages answer their question, and embedding a whole book would take minutes.
READ_CHARS = 20_000
# Embedding is the slow part (a tenth of a second or more per chunk on a laptop CPU), so each
# page's chunks are first ranked by the question's words, and only this many (plus the page's
# opening, which often sums it up) are embedded and compared by meaning.
EMBED_PER_SOURCE = 6
# The best passages of each source go in the brief, at most this many.
PASSAGES_PER_SOURCE = 3
# Many sites refuse bots (Stack Overflow and Medium answer 403). When a source can't be read,
# the next best candidate is read instead, for at most this many rounds.
ROUNDS = 3
# Answers that mean the site won't let us read it; the rest of that site is skipped too.
REFUSED = {401, 403, 429}


@dataclass(frozen=True)
class Candidate:
    """Something worth reading: a search result or a paper."""

    title: str
    url: str  # What to read (a paper's free PDF when it has one).
    snippet: str
    kind: str  # "web", "paper" or "library".
    details: str | None = None  # E.g. "2017 · 150,123 citations" for a paper.


@dataclass(frozen=True)
class Passage:
    score: float
    text: str
    start: int  # Where it starts in the page's text (a fetch_page start_index).


@dataclass
class Source:
    candidate: Candidate
    page: Page
    passages: list[Passage]


@dataclass
class Brief:
    question: str
    sources: list[Source] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # Search source or URL: what went wrong.
    searched: list[str] = field(default_factory=list)  # Search sources that worked.


def from_result(result: Result) -> Candidate:
    return Candidate(result.title, result.url, result.snippet, "web",
                     " · ".join(f for f in (result.date, result.details) if f) or None)


def from_paper(paper: Paper) -> Candidate:
    details = [str(paper.year) if paper.year else None, ", ".join(paper.authors[:3]) or None,
               paper.citations is not None and f"{paper.citations:,} citations"]
    return Candidate(paper.title, paper.pdf_url or paper.url, paper.abstract, "paper",
                     " · ".join(d for d in details if d) or None)


def rank(candidates: list[Candidate], query_vector: np.ndarray, embedder: Embedder,
         skip: set[str]) -> list[Candidate]:
    """The candidates, closest in meaning to the question first, each page once, leaving out
    URLs in skip (same_page_keys, e.g. pages already in the library)."""
    fresh = []
    seen = set(skip)
    for candidate in candidates:
        if (key := same_page_key(candidate.url)) not in seen:
            seen.add(key)
            fresh.append(candidate)
    if not fresh:
        return []
    vectors = embedder.embed([f"{c.title}\n\n{c.snippet}" for c in fresh])
    return [fresh[i] for i in np.argsort(-(vectors @ query_vector), kind="stable")]


def take(ranked: list[Candidate], count: int, per_site: Counter, refused: set[str]) -> list[Candidate]:
    """Take up to `count` candidates off the front of ranked (which is changed), skipping sites
    that refused us and sites that already have PER_SITE sources (per_site is updated)."""
    taken = []
    for candidate in list(ranked):
        if len(taken) == count:
            break
        ranked.remove(candidate)
        host = site(candidate.url)
        if host not in refused and per_site[host] < PER_SITE:
            per_site[host] += 1
            taken.append(candidate)
    return taken


def _words(text: str) -> list[str]:
    """The words of text that carry meaning, lowercased, with a plural "s" dropped."""
    return [w.removesuffix("s") for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 2 and w not in STOP_WORDS]


def shortlist(pieces: list[str], question: str, count: int) -> list[int]:
    """Indexes of the pieces most likely to answer the question, judged by how often they use
    its words (cheap, unlike embedding), in order. The first piece is always included."""
    terms = set(_words(question))

    def overlap(piece: str) -> float:
        counts = Counter(_words(piece))
        return sum(math.log1p(counts[term]) for term in terms)  # Repeats count, but less and less.

    ranked = sorted(range(len(pieces)), key=lambda i: overlap(pieces[i]), reverse=True)
    return sorted({0, *ranked[:count]}) if pieces else []


def best_passages(page: Page, question: str, query_vector: np.ndarray,
                  embedder: Embedder) -> list[Passage]:
    """The page's passages that best answer the question, best first. Passages below the
    model's min_score are left out, but the best one is always kept: the page was chosen for
    being on topic, so its closest passage is the most useful thing to show from it."""
    pieces = chunk(page.text[:READ_CHARS])
    pieces = [pieces[i] for i in shortlist([text for _, text in pieces], question, EMBED_PER_SOURCE)]
    if not pieces:
        return []
    title = f"{page.title}\n\n" if page.title else ""
    scores = embedder.embed([title + piece for _, piece in pieces]) @ query_vector
    order = np.argsort(-scores, kind="stable")[:PASSAGES_PER_SOURCE]
    return [Passage(float(scores[i]), pieces[i][1], pieces[i][0]) for n, i in enumerate(order)
            if n == 0 or scores[i] >= embedder.min_score]


async def research(
    question: str, max_sources: int, include_papers: bool, get_page: Callable[[str], Awaitable[Page]],
    library: Library, cache: SearchCache | None,
) -> tuple[Brief, list]:
    """Research question: search, choose up to max_sources sources, read them, keep their best
    passages. get_page(url) reads a page (raising FetchError or ExtractError if it can't).

    Returns the brief, and the library's hits for the question (pages already saved).
    """
    embedder = library.embedder
    brief = Brief(question)
    web_sources = ["duckduckgo", "wikipedia"]
    paper_sources = ["arxiv", "semantic_scholar"] if include_papers else []

    async def no_papers():
        return [], {}

    # First, so the model is loaded once before anything else uses it.
    query_vector = await asyncio.to_thread(embedder.embed_query, question)
    saved, (web, web_errors), (found_papers, paper_errors) = await asyncio.gather(
        asyncio.to_thread(library.closest, query_vector, 3),
        search.web_search(question, web_sources, WEB_RESULTS, cache),
        papers.search_papers(question, paper_sources, PAPER_RESULTS, cache=cache)
        if paper_sources else no_papers(),
    )
    brief.failed |= web_errors | paper_errors
    brief.searched = [s for s in web_sources + paper_sources if s not in brief.failed]

    candidates = [from_result(r) for name in web_sources for r in web.get(name, [])]
    candidates += [from_paper(p) for p in found_papers]
    skip = {same_page_key(hit.document.url) for hit in saved}
    ranked = await asyncio.to_thread(rank, candidates, query_vector, embedder, skip)

    per_site: Counter = Counter()
    refused: set[str] = set()
    for _ in range(ROUNDS):
        chosen = take(ranked, max_sources - len(brief.sources), per_site, refused)
        if not chosen:
            break
        pages = await asyncio.gather(*(get_page(c.url) for c in chosen), return_exceptions=True)
        for candidate, page in zip(chosen, pages):
            if isinstance(page, (fetch.FetchError, ExtractError)):
                brief.failed[candidate.url] = str(page)
                per_site[site(candidate.url)] -= 1  # Another page from the site may work...
                if isinstance(page, fetch.FetchError) and page.status_code in REFUSED:
                    refused.add(site(candidate.url))  # ...unless it refuses bots.
            elif isinstance(page, BaseException):
                raise page
            else:
                passages = await asyncio.to_thread(best_passages, page, question, query_vector, embedder)
                brief.sources.append(Source(candidate, page, passages))
    # Best sources first: by their best passage.
    brief.sources.sort(key=lambda s: s.passages[0].score if s.passages else -1, reverse=True)
    return brief, saved
