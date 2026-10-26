"""Pulling structured data out of web pages: the parts a site already marked up for machines.

fetch_page gives the AI a page as prose. But many pages also carry data in a form that needs
no guessing: tables, the page's metadata (OpenGraph tags, feeds, canonical URL), and schema.org
JSON-LD, which sites add for search engines and which often holds exactly what you want (a
product's price, a recipe's ingredients, an event's date, an article's author). This reads
those as they are, so the AI gets facts, not a paraphrase of them.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import json
from urllib.parse import urljoin, urlsplit

import lxml.etree
import lxml.html

# What can be extracted, in the order it's shown.
KINDS = ("metadata", "json_ld", "tables", "headings", "code", "links")
# Long cells and code blocks are cut to this many characters.
MAX_TEXT = 2000
# Text that isn't content: CSS and scripts (which would end up in cell text), and Wikipedia's
# footnote markers ("Population[12]") and "[edit]" links.
NOISE = ('//style | //script[not(@type="application/ld+json")] | //noscript'
         ' | //sup[contains(@class, "reference")] | //span[contains(@class, "mw-editsection")]')


class StructuredError(Exception):
    """A page we can't read structured data from."""


def parse(body: bytes) -> lxml.html.HtmlElement:
    try:
        tree = lxml.html.fromstring(body)
    except (lxml.etree.ParserError, ValueError):
        raise StructuredError("This isn't readable HTML") from None
    for element in tree.xpath(NOISE):
        element.drop_tree()  # Keeps the text after the element.
    return tree


def _text(element: lxml.html.HtmlElement) -> str:
    """An element's text with whitespace tidied, cut to MAX_TEXT."""
    text = " ".join(element.text_content().split())
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + " …"


def metadata(tree: lxml.html.HtmlElement, url: str) -> dict:
    """The page's own description of itself: title, description, language, canonical URL,
    feeds, and every OpenGraph (og:), Twitter card and article: tag."""
    data: dict = {}
    if (title := tree.findtext(".//title")) and title.strip():
        data["title"] = " ".join(title.split())
    if lang := tree.get("lang"):
        data["language"] = lang
    for meta in tree.iter("meta"):
        key = meta.get("property") or meta.get("name") or ""
        content = (meta.get("content") or "").strip()
        if not content:
            continue
        if key.lower() in ("description", "author", "keywords"):
            data[key.lower()] = content
        elif key.startswith(("og:", "twitter:", "article:")):
            data.setdefault(key, content)  # Keep the first when a tag repeats (e.g. og:image).
    for link in tree.iter("link"):
        rel = (link.get("rel") or "").lower().split()
        href = link.get("href")
        if not href:
            continue
        if "canonical" in rel:
            data["canonical"] = urljoin(url, href)
        elif "alternate" in rel and link.get("type") in ("application/rss+xml", "application/atom+xml"):
            data.setdefault("feeds", []).append({"title": link.get("title"), "url": urljoin(url, href)})
    return data


def json_ld(tree: lxml.html.HtmlElement) -> list:
    """Every schema.org JSON-LD item on the page. Items wrapped in an @graph are unwrapped, so
    the list holds the things themselves (an Article, a Product, a BreadcrumbList...)."""
    items = []
    for script in tree.xpath('//script[@type="application/ld+json"]'):
        try:
            data = json.loads(script.text or "", strict=False)  # strict=False: allow raw newlines.
        except ValueError:
            continue  # Broken JSON-LD is common; skip it rather than fail.
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and isinstance(item.get("@graph"), list):
                items += [thing for thing in item["@graph"] if isinstance(thing, dict)]
            elif isinstance(item, dict):
                items.append(item)
    return items


def _cells(row: lxml.html.HtmlElement) -> list[str]:
    cells = []
    for cell in row.xpath("./th|./td"):
        span = cell.get("colspan", "1")
        cells += [_text(cell)] * (min(int(span), 50) if span.isdigit() and int(span) > 0 else 1)
    return cells


def tables(tree: lxml.html.HtmlElement, max_rows: int) -> list[dict]:
    """Each table as {"caption", "columns", "rows"}, with rows as lists of cell texts.

    The header is the <thead>, or else a first row made only of <th> cells. Tables used for page
    layout (a single column) and Wikipedia's navigation boxes are skipped.
    """
    found = []
    for table in tree.iter("table"):
        if "navbox" in (table.get("class") or ""):
            continue
        rows = table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
        header_rows = table.xpath("./thead/tr")
        if not header_rows and rows and rows[0].xpath("./th") and not rows[0].xpath("./td"):
            header_rows = rows[:1]
        columns = _cells(header_rows[-1]) if header_rows else []
        body = [cells for row in rows if row not in header_rows and (cells := _cells(row))]
        if max(len(columns), *(len(r) for r in body), 0) < 2:
            continue
        entry = {"caption": _text(caption[0]) if (caption := table.xpath("./caption")) else None,
                 "columns": columns, "rows": body[:max_rows]}
        if len(body) > max_rows:
            entry["more_rows"] = len(body) - max_rows
        found.append(entry)
    return found


def headings(tree: lxml.html.HtmlElement) -> list[dict]:
    """The page's outline: every h1-h6, in order, as {"level", "text"}."""
    return [{"level": int(h.tag[1]), "text": text}
            for h in tree.iter("h1", "h2", "h3", "h4", "h5", "h6") if (text := _text(h))]


def code(tree: lxml.html.HtmlElement) -> list[dict]:
    """Code blocks (<pre>), with their language when the page says (class="language-python")."""
    blocks = []
    for pre in tree.iter("pre"):
        classes = " ".join(e.get("class") or "" for e in [pre, *pre.iter("code")]).split()
        language = next((c.split("-", 1)[1] for c in classes if c.startswith(("language-", "lang-"))), None)
        text = pre.text_content().strip("\n")
        if text.strip():
            blocks.append({"language": language,
                           "code": text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "\n…"})
    return blocks


def links(tree: lxml.html.HtmlElement, url: str) -> list[dict]:
    """Every http(s) link as {"text", "url", "internal"} (internal: on the same site), each URL once."""
    site = (urlsplit(url).hostname or "").removeprefix("www.")
    found: dict[str, dict] = {}
    for a in tree.iter("a"):
        target = urljoin(url, (a.get("href") or "").strip())
        parts = urlsplit(target)
        if parts.scheme not in ("http", "https"):
            continue
        target = parts._replace(fragment="").geturl()
        text = _text(a) or a.get("title") or a.get("aria-label") or ""
        if target not in found:
            found[target] = {"text": text, "url": target,
                             "internal": (parts.hostname or "").removeprefix("www.") == site}
        elif not found[target]["text"]:
            found[target]["text"] = text
    return list(found.values())


def extract(url: str, body: bytes, kinds: list[str], max_items: int) -> dict:
    """The structured data of the kinds asked for, by kind. Lists are cut to max_items (and
    tables to max_items rows); a "more" entry says how much was left out."""
    tree = parse(body)
    result: dict = {}
    more: dict[str, int] = {}
    for kind in (k for k in KINDS if k in kinds):
        match kind:
            case "metadata":
                result[kind] = metadata(tree, url)
                continue
            case "json_ld":
                items = json_ld(tree)
            case "tables":
                items = tables(tree, max_items)
            case "headings":
                items = headings(tree)
            case "code":
                items = code(tree)
            case "links":
                items = links(tree, url)
        result[kind] = items[:max_items]
        if len(items) > max_items:
            more[kind] = len(items) - max_items
    if more:
        result["more"] = more
    return result
