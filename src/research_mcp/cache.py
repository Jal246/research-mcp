"""Remembering pages we've already fetched, so reading a long page in parts (or asking about it
again) doesn't download and re-extract it every time. And remembering search results for a
while, because free search APIs only allow a few searches a minute. And remembering pages as
downloaded, for extract_data, which reads the HTML itself rather than its extracted text.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from research_mcp.extract import Page
from research_mcp.fetch import Response

PAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    requested_url TEXT PRIMARY KEY,     -- What was asked for (may redirect to url).
    url           TEXT NOT NULL,
    kind          TEXT NOT NULL,
    text          TEXT NOT NULL,
    title         TEXT,
    site          TEXT,
    author        TEXT,
    date          TEXT,
    fetched_at    REAL NOT NULL         -- Unix time.
);
"""


SEARCHES_SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    key         TEXT PRIMARY KEY,       -- Source, query and options, e.g. "arxiv|rsa|5".
    results     TEXT NOT NULL,          -- JSON list of results.
    searched_at REAL NOT NULL           -- Unix time.
);
"""


RESPONSES_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    requested_url TEXT PRIMARY KEY,
    url           TEXT NOT NULL,        -- Where it ended up, after redirects.
    content_type  TEXT NOT NULL,
    body          BLOB NOT NULL,        -- As downloaded (up to fetch.MAX_BYTES).
    fetched_at    REAL NOT NULL         -- Unix time.
);
"""


class _SQLiteCache:
    """One SQLite file (created on first use) whose entries expire after max_age_seconds."""

    SCHEMA = ""

    def __init__(self, path: Path, max_age_seconds: float):
        self.path = path
        self.max_age_seconds = max_age_seconds

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Open the database for one operation: commit if it succeeds, roll back if it fails."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.executescript(self.SCHEMA)
                yield conn
        finally:
            conn.close()


class PageCache(_SQLiteCache):
    """Extracted pages, by the URL that was asked for."""

    SCHEMA = PAGES_SCHEMA

    def get(self, requested_url: str, now: float | None = None) -> Page | None:
        """The cached page for this URL, or None if there isn't one or it's too old."""
        now = time.time() if now is None else now
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM pages WHERE requested_url = ? AND fetched_at > ?",
                (requested_url, now - self.max_age_seconds),
            ).fetchone()
        if row is None:
            return None
        return Page(
            row["url"], row["kind"], row["text"], row["title"], row["site"], row["author"], row["date"]
        )

    def put(self, requested_url: str, page: Page, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (requested_url, page.url, page.kind, page.text, page.title, page.site, page.author,
                 page.date, now),
            )


class ResponseCache(_SQLiteCache):
    """Downloaded files as they came, by the URL that was asked for. Bodies can be megabytes,
    so expired ones are deleted whenever a new one is saved, rather than kept forever."""

    SCHEMA = RESPONSES_SCHEMA

    def get(self, requested_url: str, now: float | None = None) -> Response | None:
        now = time.time() if now is None else now
        with self._db() as conn:
            row = conn.execute(
                "SELECT url, content_type, body FROM responses WHERE requested_url = ? AND fetched_at > ?",
                (requested_url, now - self.max_age_seconds),
            ).fetchone()
        return None if row is None else Response(row["url"], row["content_type"], bytes(row["body"]))

    def put(self, requested_url: str, response: Response, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._db() as conn:
            conn.execute("DELETE FROM responses WHERE fetched_at <= ?", (now - self.max_age_seconds,))
            conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
                (requested_url, response.url, response.content_type, response.body, now),
            )


class SearchCache(_SQLiteCache):
    """Search results (anything JSON can store), by a key naming the search."""

    SCHEMA = SEARCHES_SCHEMA

    def get(self, key: str, now: float | None = None) -> list | None:
        """The cached results for this search, or None if there aren't any or they're too old."""
        now = time.time() if now is None else now
        with self._db() as conn:
            row = conn.execute(
                "SELECT results FROM searches WHERE key = ? AND searched_at > ?",
                (key, now - self.max_age_seconds),
            ).fetchone()
        return None if row is None else json.loads(row["results"])

    def put(self, key: str, results: list, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO searches VALUES (?, ?, ?)", (key, json.dumps(results), now)
            )
