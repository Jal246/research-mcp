"""Finding pages similar to a given page, without a paid "find similar" API.

There's no free index of the web to look up neighbours in, so we do what a person would: search
for the page's title and for the words it uses most, then keep the results that are actually
about the same thing. "Actually about" is decided by the embedding model: each result's title
and snippet is compared to the start of the page, and the results are reordered by how close
they are in meaning.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import re
from collections import Counter
from urllib.parse import urlsplit

import numpy as np

from research_mcp.extract import Page
from research_mcp.library import Embedder
from research_mcp.search import Result, same_page_key

# How much of the page is compared with the results: its start says what it's about.
PAGE_CHARS = 2000
KEYWORDS = 6

# Common English words that say nothing about what a page is about.
STOP_WORDS = set("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from
further get gets got had has have having he her here hers him his how however i if in into is
it its itself just let like make makes made many may me might more most much must my no nor
not now of off often on once one only or other our ours out over own per same she should so
some such than that the their theirs them then there these they this those through to too
under until up upon us use used uses using very via was we were what when where which while
who whom why will with would you your yours also new first two three way ways well even back
see still since within without based every another around example examples called given
""".split())

# Separators sites put between a page's title and the site name: "RSA - Wikipedia".
TITLE_SEPARATORS = re.compile(r"\s+[|\-–—·•»]\s+")


def site(url: str) -> str:
    """The website a URL is on, ignoring www.: "https://www.example.org/a" -> "example.org"."""
    return (urlsplit(url).hostname or "").removeprefix("www.")


def title_query(title: str) -> str:
    """The part of a page title that names the page, without the site name: the longest part
    between separators. "How RSA Works | Crypto Blog" -> "How RSA Works"."""
    return max(TITLE_SEPARATORS.split(title.strip()), key=len).strip()


def keywords(text: str, count: int = KEYWORDS) -> list[str]:
    """The words a text uses most, leaving out common words, numbers and URLs."""
    text = re.sub(r"\]\([^)]*\)|https?://\S+", " ", text)  # Markdown link targets and bare URLs.
    words = re.findall(r"[a-z][a-z0-9+#'-]*[a-z0-9+#]", text.lower())
    counts = Counter(w for w in words if len(w) > 2 and w not in STOP_WORDS and "'" not in w)
    return [word for word, _ in counts.most_common(count)]  # Ties stay in order of first use.


def queries(page: Page) -> list[str]:
    """What to search for to find pages like this one: its title, and its most used words."""
    found = []
    if page.title and (title := title_query(page.title)):
        found.append(title)
    if words := keywords(page.text[:20_000]):
        found.append(" ".join(words))
    return list(dict.fromkeys(found))


def page_text(page: Page) -> str:
    """The text that stands for the whole page when comparing it with search results."""
    return f"{page.title}\n\n{page.text[:PAGE_CHARS]}" if page.title else page.text[:PAGE_CHARS]


def candidates(page: Page, results: list[Result], requested_url: str,
               include_same_site: bool) -> list[Result]:
    """Search results worth comparing with the page: not the page itself, each page only once,
    and (unless include_same_site) not from the page's own website."""
    seen = {same_page_key(page.url), same_page_key(requested_url)}
    own_sites = {site(page.url), site(requested_url)}
    kept = []
    for result in results:
        key = same_page_key(result.url)
        if key in seen or (not include_same_site and site(result.url) in own_sites):
            continue
        seen.add(key)
        kept.append(result)
    return kept


def rank(page_vector: np.ndarray, results: list[Result], embedder: Embedder,
         limit: int) -> list[tuple[float, Result]]:
    """The results closest in meaning to the page (page_vector), best first, with their scores.
    Results scoring below the model's min_similar_score are dropped."""
    if not results:
        return []
    vectors = embedder.embed([f"{r.title}\n\n{r.snippet}" for r in results])
    scored = sorted(zip((vectors @ page_vector).tolist(), results), key=lambda pair: pair[0], reverse=True)
    return [(s, result) for s, result in scored if s >= embedder.min_similar_score][:limit]
