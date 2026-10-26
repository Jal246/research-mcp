# research-mcp

A Python [MCP](https://modelcontextprotocol.io) server that lets Claude research the web like [Exa](https://exa.ai), for free: one-call research briefs with citable passages, web and paper search, any page or PDF as clean markdown, structured data from pages, site crawling, "find similar", and a personal research library that's searchable by meaning.

No API keys, no paid search service. Everything runs on your computer, including the embedding model.

## Demo

> **You:** How does Raft handle network partitions?
>
> **Claude** calls `research_report`, which searches DuckDuckGo, Wikipedia and arXiv, reads the six most relevant sources (skipping Stack Overflow and Medium, which refuse bots, for the next best), and returns a brief like this (shortened):
>
> ```
> # Research brief: how does Raft handle network partitions
>
> Searched DuckDuckGo, Wikipedia, arXiv and the library; 6 sources below, best first, each
> with its most relevant passages. Answer from these passages and cite sources by number...
>
> ## [1] Handling Network Partitions in Consensus Algorithms like Raft with Python Simulations
> https://johal.in/handling-network-partitions-in-consensus-algorithms-like-raft-... · published 2026-03-16
> > | Network Partition | Nodes 1,2 isolated from Nodes 3,4,5 | Heartbeat timeouts expire on
>   minority partition | Node 3 starts new election in term 5 | ... | Minority Leader (Node 3) |
>   Cannot commit entries (needs 3/5 acks) | ... | Partition Heals | Node 3 discovers higher term ...
> (similarity 0.82 · start_index=1835)
>
> ## [2] Handling Network Partitions in Distributed Systems - GeeksforGeeks
> ...
> ```
>
> Claude then answers from the passages, citing [1], [2]... and calls `fetch_page` with a `start_index` when it wants more of a source. That's one tool call and under 20,000 characters of context, instead of a dozen calls and six full pages (often over 100,000).

## What it can do

| Tool | What it's for |
|---|---|
| `research_report` | Research a question in one call: searches the web, papers and your library, reads the most relevant sources (at most two per site, with backups for sites that refuse bots), and returns numbered sources with the passages that best answer the question, ready to cite. Can save the sources to the library. |
| `extract_data` | A page's structured data as JSON: schema.org JSON-LD (a product's price, a recipe's ingredients, an event's date), tables as rows, metadata (OpenGraph tags, canonical URL, RSS feeds), headings, code blocks and links. |
| `web_search` | Search the web with DuckDuckGo and Wikipedia, plus GitHub (repositories, with stars) and Hacker News (discussions, with points) when asked. Sources run at the same time; a page two sources both found is shown once. |
| `search_papers` | Search arXiv and Semantic Scholar for papers: authors, year, venue, citation count, free PDF link and abstract. The same paper from both sources is merged into one. Filter by year, sort by relevance, newest or most cited. |
| `fetch_page` | Read a web page, PDF or text file as markdown: just the main content, with links (made absolute) and tables kept. Long pages come in parts. Pages are cached for a day. |
| `crawl_site` | Read several pages of one site (a docs site, blog or course), following links under the starting folder. Follows robots.txt, waits between requests, stops at a page limit. Can save everything it reads to the library. |
| `find_similar` | Find pages like a given one: searches for its title and key words, then ranks the results by how close in meaning they are to the page. Also shows similar pages already in your library. |
| `save_to_library` | Keep a page, PDF or paper in your research library, with a note and tags. The full text is stored, so it stays searchable even if the page disappears. |
| `search_library` | Search the library by meaning: "how do servers pick a leader" finds the passage about Raft elections. Shows the best passage of each page and where to read on. |
| `list_library`, `remove_from_library` | See what's saved (with notes and tags), and tidy up. |

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```sh
git clone https://github.com/Jal246/research-mcp.git
cd research-mcp
uv sync
```

Register the server with Claude Code, using the full path to the repo:

```sh
claude mcp add research -- uv run --directory /path/to/research-mcp research-mcp
```

Then ask Claude things like:

- "Research whether uv is really faster than pip, and save the sources to my library"
- "What are the ingredients and cook time in https://www.bbcgoodfood.com/recipes/easy-pancakes?" (from its JSON-LD)
- "Get the table from https://en.wikipedia.org/wiki/List_of_largest_cities and rank the top 5 by urban area"
- "Read https://en.wikipedia.org/wiki/RSA_cryptosystem and explain key generation"
- "Find the most cited papers on retrieval-augmented generation since 2023 and summarize the top one"
- "What do people on Hacker News think of async Rust runtimes? Which ones are popular on GitHub?"
- "What changed in the MCP Python SDK v2? Read its README"
- "Crawl the uv guides, save them to my library tagged uv, then tell me how to add a dependency"
- "Find articles similar to https://jvns.ca/blog/2017/06/11/log-structured-storage/"
- "Save this paper to my library with the note 'baseline for my thesis'", and weeks later: "What did I save about leader election?"

The first library save downloads a small embedding model (about 70 MB, once).

## Configuration

All optional, set as environment variables:

| Variable | Default | What it is |
|---|---|---|
| `RESEARCH_CACHE_DIR` | `.cache/` | Fetched pages, recent searches and the embedding model (safe to delete; rebuilt automatically) |
| `RESEARCH_DATA_DIR` | `data/` | Your research library (`library.db`). This is your own work: back it up rather than deleting it. |
| `SEMANTIC_SCHOLAR_API_KEY` | none | Makes `search_papers` reliable: without one, Semantic Scholar is shared by everyone and often busy. [Free to request](https://www.semanticscholar.org/product/api#api-key-form). |
| `GITHUB_TOKEN` | none | Raises GitHub's search limit from 10 to 30 a minute. Any token works; it needs no permissions. |

## How it works

```
Claude ──MCP (stdio)──▶ server.py ─┬─▶ report.py      search → choose → read → best passages
                                   ├─▶ structured.py  JSON-LD, tables, metadata
                                   ├─▶ search.py ─┐  DuckDuckGo, Wikipedia, GitHub, Hacker News
                                   ├─▶ papers.py ─┤  arXiv, Semantic Scholar
                                   ├─▶ crawl.py ──┤  links, robots.txt
                                   │              ▼
                                   │          polite.py   rate limits, retries
                                   │              ▼
                                   ├────────▶ fetch.py ──▶ the web (public http/https only)
                                   ├─▶ extract.py          HTML → markdown, PDF → text
                                   ├─▶ similar.py ─┐       title + key words → search → rerank
                                   ├─▶ library.py ─┼─▶ embed.py   local model (fastembed)
                                   │               └─▶ data/library.db (SQLite)
                                   └─▶ cache.py ──▶ .cache/pages.db, searches.db (SQLite)
```

- **`server.py`** defines the tools with the official MCP Python SDK.
- **`search.py`** has one small function per web source, and runs several at once.
- **`papers.py`** does the same for papers, then merges duplicates by arXiv ID, DOI or title.
- **`polite.py`** spaces out requests to each site and retries when a server is busy.
- **`fetch.py`** downloads safely: size and time limits, redirects re-checked at every hop.
- **`extract.py`** turns HTML into markdown with [trafilatura](https://trafilatura.readthedocs.io), and PDFs into text page by page with pypdfium2.
- **`cache.py`** is a small SQLite cache: reading a long page in parts downloads it once, and repeating a search within an hour costs no API calls.
- **`crawl.py`** reads a site breadth first (everything one click away, then two clicks), staying under the starting folder.
- **`embed.py`** turns text into vectors with [fastembed](https://github.com/qdrant/fastembed) and `BAAI/bge-small-en-v1.5`, a small model that runs on the CPU.
- **`library.py`** stores saved pages, cut into chunks of about 1,000 characters with a vector each, in SQLite.
- **`similar.py`** builds searches from a page and reranks what they find by meaning.
- **`report.py`** runs the research loop: search everything at once, rank the results by meaning, read the best few, and pick each one's best passages.
- **`structured.py`** reads what pages mark up for machines (JSON-LD, tables, meta tags) with lxml.

## Design decisions

- **Blocking private addresses.** Claude picks the URLs, and a page it reads could try to trick it into fetching something else. So only public addresses are fetched: `localhost`, `192.168.x.x`, `10.x.x.x`, the cloud metadata address `169.254.169.254` and `file://` are refused. This is checked again after every redirect, because a public page can redirect to `http://127.0.0.1/admin`.
- **trafilatura for extraction.** Telling an article apart from its menus and sidebars is a solved problem, and trafilatura does it well across many sites. Publish dates are only taken from the page's own metadata, never guessed from text, so "Copyright 2026" in a footer doesn't become a publish date.
- **Pagination instead of truncation.** Long pages are returned about 20,000 characters at a time, cut at a line break, and each part says where to continue. Claude reads further only when it needs to.
- **Free sources only, each doing what it's good at.** DuckDuckGo for the open web, Wikipedia for background, GitHub for code, Hacker News for what practitioners think, arXiv for free preprints, Semantic Scholar for citations and other fields. DuckDuckGo has no API, so its plain-HTML page is read instead.
- **One broken source doesn't break a search.** Sources run at the same time, and a failure (busy, down, CAPTCHA, changed format) is reported next to the results from the sources that worked. Only if all of them fail is it an error.
- **Polite by default.** Requests to each site are spaced out (arXiv asks for 3 seconds between requests), and "busy" answers (429, 503) are retried twice, waiting longer each time or as long as the server asks, but never more than 10 seconds. Failed searches aren't cached, so trying again later really tries again.
- **Merged papers.** The same paper often comes from arXiv (with a free PDF) and Semantic Scholar (with a citation count). They're combined into one entry with the best of both.
- **An honest User-Agent.** Some sites block bots; the server says what it is rather than pretending to be a browser.
- **Local embeddings, not a vector database.** A personal library holds hundreds of pages, not millions, so the vectors sit next to the text in SQLite and a search compares the query with every chunk using numpy. It takes milliseconds, and the whole library is one file you can copy. If the embedding model ever changes, pages saved with the old one are re-embedded automatically on the next search, since vectors from two models can't be compared.
- **Links stripped before embedding.** Markdown from Wikipedia is often half link URLs, which mean nothing to the model but cost a lot of its time. Embedding the readable text instead made saving about four times faster (a 50,000 character article went from about 20 seconds to 5).
- **`find_similar` without an index of the web.** Exa answers "find similar" from its own index of the web, and there's no free one. Instead the server searches for the page's title and its most-used words, then keeps the results that are actually close in meaning to the page, by comparing their vectors. Pages from the same site are left out by default: they're usually the page itself, or its neighbours.
- **A careful crawler.** `crawl_site` follows robots.txt (including its crawl delay), waits between requests, stays under the starting folder, skips links to images and downloads, and stops at a page limit (10 by default, at most 50). Everything it reads goes into the page cache, so reading a crawled page in full afterwards costs nothing.
- **A research brief, not a written report.** The server has no language model of its own (that would need a paid API or a big download), and Claude is better at writing the answer anyway. So `research_report` does the legwork: it returns the evidence, numbered for citing, and Claude writes. Passages are chosen by meaning, but only after a cheap word-overlap shortlist, because embedding every chunk of six long pages took over 10 seconds on a laptop CPU.
- **Backups for blocked sources.** Stack Overflow, Medium and others answer bots with 403. When a source can't be read, `research_report` reads the next best candidate instead, and skips the rest of a site that refused once. At most two sources come from one site, so the brief has more than one point of view.
- **Structured data as it is.** `extract_data` doesn't guess: it returns what the page itself declares (JSON-LD, meta tags, table cells), so a price or a date comes out exactly as the site wrote it. CSS and Wikipedia's footnote markers are stripped from cell text, and Wikipedia's navigation boxes aren't treated as tables.
- **Tracking parameters ignored.** `?utm_source=...`, `?ref=...` and similar don't make a page count as a different one, so a result isn't read twice, or read again when it's already in the library.
- **A fake internet in the tests.** The tests use `httpx.MockTransport` and fake DNS, so they're fast, run offline and give the same result every time. They use a fake embedding model too (vectors made by counting words), so the tests don't download anything.

## Known limitations

- DuckDuckGo sometimes decides the server is a bot and shows a CAPTCHA; `web_search` then says so and uses its other sources.
- Without `SEMANTIC_SCHOLAR_API_KEY`, Semantic Scholar is often busy, so `search_papers` falls back to arXiv alone (after a few seconds of retries) and has no citation counts.
- arXiv's own ranking is weak for exact titles: searching "attention is all you need" finds papers with that phrase, but not necessarily the original first. Semantic Scholar ranks better.
- The embedding model is English only, and its similarity scores are rough: a related page might score 0.65 or 0.85. `search_library` drops results under 0.6, so a weakly related page can be missed.
- `find_similar` can only find pages the search sources can find, and it only uses DuckDuckGo and Wikipedia by default.
- `research_report` takes 10-30 seconds: most of it is searching (Semantic Scholar without an API key is often busy and retried) and reading pages. It only searches the start of each page (20,000 characters) for passages.
- `extract_data` reads a table with a two-row header (common on Wikipedia) as one header row plus a row of sub-headings. It fetches the page each time rather than using the cache.
- `crawl_site` only follows links in the HTML, so sites that build their menus with JavaScript only crawl as far as their plain links go.
- Pages that build their content with JavaScript (some web apps) have little or no text to extract.
- Sites behind bot protection (Cloudflare challenges, logins) can't be read.
- Scanned PDFs (images of text) have no text; there's no OCR.
- The private-address check happens before connecting, so a site that changes its DNS answer between the check and the connection (DNS rebinding) could get past it.

## Roadmap

- [x] **Day 1:** `fetch_page`: clean extraction, PDFs, caching, safety checks, tests
- [x] **Day 2:** `web_search` across free sources (DuckDuckGo, Wikipedia, GitHub, Hacker News), `search_papers` (arXiv / Semantic Scholar), rate limiting and retries
- [x] **Day 3:** research library (`save_to_library`, `search_library` with local embeddings, `list_library`, `remove_from_library`), `find_similar`, `crawl_site`
- [x] **Day 4:** `extract_data` (structured data), `research_report`, demo, polish

## Development

```sh
uv run pytest
```
