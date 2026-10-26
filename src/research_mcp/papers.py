"""Searching for research papers: arXiv and Semantic Scholar.

arXiv has the preprints (free PDFs for almost everything in physics, maths and computer science).
Semantic Scholar covers all fields and knows citation counts, but its free tier is often busy,
which is what polite.get's retries are for (a SEMANTIC_SCHOLAR_API_KEY makes it reliable).
The same paper often comes from both, so results are merged by arXiv ID, DOI or title.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from urllib.parse import urlencode

from research_mcp import fetch, polite
from research_mcp.cache import SearchCache
from research_mcp.search import SearchError, parse_json, run_sources


@dataclass(frozen=True)
class Paper:
    title: str
    authors: tuple[str, ...]
    year: int | None
    abstract: str
    url: str  # The paper's page.
    source: str  # Which source(s) found it, e.g. "arxiv" or "arxiv+semantic_scholar".
    pdf_url: str | None = None  # A free PDF, if there is one.
    venue: str | None = None  # Journal or conference.
    citations: int | None = None
    arxiv_id: str | None = None  # E.g. "1706.03762".
    doi: str | None = None

    @classmethod
    def from_json(cls, data: dict) -> "Paper":
        return cls(**{**data, "authors": tuple(data["authors"])})


def tidy(text: str | None) -> str:
    """Whitespace collapsed: arXiv wraps titles and abstracts over several lines."""
    return " ".join((text or "").split())


# --- arXiv ------------------------------------------------------------------------------------
# arXiv asks for at most one request every 3 seconds. Its search wants every word to match, and
# the words its search engine (Lucene) ignores, like "is", match nothing, so they're left out.

ARXIV_INTERVAL = 3.0
STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
              "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
              "there", "these", "they", "this", "to", "was", "will", "with"}
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def arxiv_query(query: str, year_from: int | None = None) -> str:
    """arXiv's search syntax for a plain query: the exact phrase, or all its important words.

    "attention is all you need" ->
    all:"attention is all you need" OR (all:attention AND all:all AND all:you AND all:need)
    """
    words = re.findall(r"[\w.-]+", query.lower())
    if not words:
        raise SearchError("The query has no words to search arXiv for")
    important = [w for w in words if w not in STOP_WORDS] or words
    search = f'all:"{" ".join(words)}"'
    if len(words) > 1:
        search += " OR (" + " AND ".join(f"all:{w}" for w in important) + ")"
    if year_from:
        search = f"({search}) AND submittedDate:[{year_from}01010000 TO 999912312359]"
    return search


def parse_arxiv(feed: bytes) -> list[Paper]:
    try:
        root = ET.fromstring(feed)
    except ET.ParseError:
        raise SearchError("arXiv didn't return a valid feed") from None
    papers = []
    for entry in root.iterfind("atom:entry", ATOM):
        # <id>http://arxiv.org/abs/1706.03762v7</id>: drop the version to link the latest one.
        abs_url = entry.findtext("atom:id", "", ATOM)
        arxiv_id = re.sub(r"v\d+$", "", abs_url.split("/abs/")[-1])
        if not arxiv_id:
            continue
        published = entry.findtext("atom:published", "", ATOM)
        papers.append(Paper(
            title=tidy(entry.findtext("atom:title", "", ATOM)),
            authors=tuple(tidy(a.findtext("atom:name", "", ATOM))
                          for a in entry.iterfind("atom:author", ATOM)),
            year=int(published[:4]) if published[:4].isdigit() else None,
            abstract=tidy(entry.findtext("atom:summary", "", ATOM)),
            url=f"https://arxiv.org/abs/{arxiv_id}",
            source="arxiv",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            venue=tidy(entry.findtext("arxiv:journal_ref", "", ATOM)) or None,
            arxiv_id=arxiv_id,
            doi=entry.findtext("arxiv:doi", None, ATOM),
        ))
    return papers


async def search_arxiv(query: str, limit: int, year_from: int | None = None) -> list[Paper]:
    params = {"search_query": arxiv_query(query, year_from), "max_results": limit}
    response = await polite.get(
        "https://export.arxiv.org/api/query?" + urlencode(params), interval=ARXIV_INTERVAL
    )
    return parse_arxiv(response.body)


# --- Semantic Scholar -------------------------------------------------------------------------

S2_FIELDS = "title,abstract,year,authors,venue,citationCount,externalIds,openAccessPdf,url"


async def search_semantic_scholar(query: str, limit: int, year_from: int | None = None) -> list[Paper]:
    params = {"query": query, "limit": limit, "fields": S2_FIELDS}
    if year_from:
        params["year"] = f"{year_from}-"
    headers = {"Accept": "application/json"}
    if key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = key
    try:
        response = await polite.get(
            "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode(params),
            interval=1.0, headers=headers,
        )
    except fetch.FetchError as error:
        if error.status_code == 429:
            raise SearchError("Semantic Scholar's free tier is busy (it's shared by everyone without "
                              "an API key); try again in a minute, or set SEMANTIC_SCHOLAR_API_KEY "
                              "(free from semanticscholar.org/product/api)") from None
        raise
    papers = []
    for item in parse_json(response).get("data") or []:
        ids = item.get("externalIds") or {}
        papers.append(Paper(
            title=tidy(item.get("title")),
            authors=tuple(a["name"] for a in item.get("authors") or [] if a.get("name")),
            year=item.get("year"),
            abstract=tidy(item.get("abstract")),
            url=item.get("url") or f"https://www.semanticscholar.org/paper/{item['paperId']}",
            source="semantic_scholar",
            pdf_url=(item.get("openAccessPdf") or {}).get("url") or None,
            venue=item.get("venue") or None,
            citations=item.get("citationCount"),
            arxiv_id=ids.get("ArXiv"),
            doi=ids.get("DOI"),
        ))
    return papers


SOURCES = {"arxiv": search_arxiv, "semantic_scholar": search_semantic_scholar}


# --- Merging and sorting ----------------------------------------------------------------------

def same_paper_keys(paper: Paper) -> set[str]:
    """Ways to recognize this paper from another source: arXiv ID, DOI, or its title's letters."""
    keys = set()
    if letters := re.sub(r"[^a-z0-9]", "", paper.title.lower()):
        keys.add("title:" + letters)
    if paper.arxiv_id:
        keys.add("arxiv:" + paper.arxiv_id.lower())
    if paper.doi:
        keys.add("doi:" + paper.doi.lower())
    return keys


def combine(first: Paper, second: Paper) -> Paper:
    """One paper from two sources' records: first's values, with gaps filled from second."""
    filled = {name: getattr(second, name) for name in first.__dataclass_fields__
              if not getattr(first, name) and getattr(second, name)}
    if second.citations is not None:
        filled["citations"] = second.citations  # Only Semantic Scholar counts citations.
    return replace(first, **filled, source=f"{first.source}+{second.source}")


def merge(lists: list[list[Paper]]) -> list[Paper]:
    """All papers, taking turns between sources (each one's best first), with duplicates combined."""
    merged: list[Paper] = []
    index: dict[str, int] = {}  # same_paper_keys -> position in merged.
    for rank in range(max(map(len, lists), default=0)):
        for papers in lists:
            if rank >= len(papers):
                continue
            paper = papers[rank]
            keys = same_paper_keys(paper)
            if (position := next((index[k] for k in keys if k in index), None)) is not None:
                merged[position] = combine(merged[position], paper)
                keys |= same_paper_keys(merged[position])
            else:
                position = len(merged)
                merged.append(paper)
            index.update(dict.fromkeys(keys, position))
    return merged


SORTS = {
    "relevance": None,
    "newest": lambda p: -(p.year or 0),
    "most_cited": lambda p: -(p.citations or 0),
}


async def search_papers(
    query: str, sources: list[str], limit: int, year_from: int | None = None,
    sort: str = "relevance", cache: SearchCache | None = None,
) -> tuple[list[Paper], dict[str, str]]:
    """Up to `limit` papers from each source, merged and sorted. Also returns errors by source."""
    results, errors = await run_sources(
        {name: (lambda name=name: SOURCES[name](query, limit, year_from)) for name in sources}, limit,
        cache, lambda name: f"{name}|{query}|{limit}|{year_from}", Paper.from_json,
    )
    papers = merge([results[name] for name in sources if name in results])
    if key := SORTS[sort]:
        papers.sort(key=key)  # Stable, so ties keep their relevance order.
    return papers, errors
