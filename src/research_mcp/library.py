"""A personal research library: pages worth keeping, searchable by meaning.

Saving a page stores its full text (so it can still be searched if the site goes away) and cuts
it into chunks of about a thousand characters. Each chunk gets a vector from the embedding model,
so searching "how do nodes agree on a leader" finds the passage about Raft elections even if it
never uses those words.

Everything is in one SQLite file. Unlike the cache, this is the user's own work and can't be
rebuilt, so it lives in the data folder, not .cache.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from research_mcp.extract import Page
from research_mcp.search import same_page_key

# Pages are cut into chunks of about this many characters; each chunk gets its own vector.
CHUNK_CHARS = 1000

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id       INTEGER PRIMARY KEY,
    key      TEXT NOT NULL UNIQUE,  -- same_page_key(url), so http/https or www. don't save twice.
    url      TEXT NOT NULL,
    kind     TEXT NOT NULL,
    title    TEXT,
    site     TEXT,
    author   TEXT,
    date     TEXT,
    text     TEXT NOT NULL,
    note     TEXT NOT NULL DEFAULT '',  -- Why it was saved, in the user's (or AI's) words.
    model    TEXT NOT NULL,             -- The embedding model its chunk vectors came from.
    saved_at REAL NOT NULL              -- Unix time.
);
CREATE TABLE IF NOT EXISTS tags (
    document_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tag         TEXT NOT NULL,
    PRIMARY KEY (document_id, tag)
);
CREATE TABLE IF NOT EXISTS chunks (
    document_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    start       INTEGER NOT NULL,  -- Where the chunk starts in the text (a fetch_page start_index).
    text        TEXT NOT NULL,
    vector      BLOB NOT NULL,     -- float32 numbers, length 1.
    PRIMARY KEY (document_id, start)
);
"""


class Embedder(Protocol):
    """What the library needs from an embedding model (see embed.Embedder)."""

    name: str
    min_score: float  # For a query and a passage.
    min_similar_score: float  # For two passages.

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...


class LibraryError(Exception):
    """A problem the user should hear about, like removing a page that isn't saved."""


@dataclass(frozen=True)
class Document:
    url: str
    kind: str
    title: str | None
    site: str | None
    author: str | None
    date: str | None
    note: str
    tags: tuple[str, ...]
    saved_at: float
    chars: int  # Length of the saved text.


@dataclass(frozen=True)
class Hit:
    """A document found by a search, with its best-matching passage."""

    document: Document
    score: float  # Cosine similarity of the passage to the query, -1 to 1 (higher is closer).
    passage: str
    start: int  # Where the passage starts in the document's text.


def chunk(text: str, max_chars: int = CHUNK_CHARS) -> list[tuple[int, str]]:
    """Cut text into pieces of at most max_chars, between paragraphs where possible.

    Returns (where the piece starts in text, piece) pairs.
    """
    spans = []  # (start, end) of each paragraph, with huge paragraphs cut at spaces.
    for match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, re.DOTALL):
        start, end = match.span()
        while end - start > max_chars:
            cut = text.rfind(" ", start + max_chars // 2, start + max_chars)
            if cut == -1:  # No space (e.g. a long URL): just cut.
                cut = start + max_chars
            spans.append((start, cut))
            start = cut
            while start < end and text[start].isspace():
                start += 1
        spans.append((start, end))
    # Join neighbouring paragraphs while they fit, so chunks aren't single short lines.
    pieces: list[tuple[int, int]] = []
    for start, end in spans:
        if pieces and end - pieces[-1][0] <= max_chars:
            pieces[-1] = (pieces[-1][0], end)
        else:
            pieces.append((start, end))
    return [(start, text[start:end]) for start, end in pieces]


def clean_tags(tags: list[str]) -> list[str]:
    """Tags lowercased, with spaces turned into dashes and repeats dropped: ["ML Papers"] -> ["ml-papers"]."""
    cleaned = ("-".join(tag.lower().split()) for tag in tags)
    return list(dict.fromkeys(tag for tag in cleaned if tag))


class Library:
    def __init__(self, path: Path, embedder: Embedder):
        self.path = path
        self.embedder = embedder

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Open the database for one operation: commit if it succeeds, roll back if it fails."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # So removing a document removes its chunks.
        try:
            with conn:
                conn.executescript(SCHEMA)
                yield conn
        finally:
            conn.close()

    def _store_chunks(self, conn: sqlite3.Connection, document_id: int, title: str | None,
                      text: str) -> None:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        pieces = chunk(text)
        # The title goes in front of every chunk, so a passage that never names its topic
        # ("It then picks two primes...") still matches searches about it.
        vectors = self.embedder.embed([f"{title}\n\n{piece}" if title else piece for _, piece in pieces])
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
            [(document_id, start, piece, vector.astype(np.float32).tobytes())
             for (start, piece), vector in zip(pieces, vectors)],
        )

    def save(self, page: Page, note: str = "", tags: list[str] | None = None,
             now: float | None = None) -> tuple[Document, bool]:
        """Save a page, or update it if it's already saved (its text is replaced, tags are
        added to the ones it has, and the note is replaced if a new one is given).

        Returns the saved document and whether it was already in the library.
        """
        now = time.time() if now is None else now
        key = same_page_key(page.url)
        with self._db() as conn:
            row = conn.execute("SELECT id, note FROM documents WHERE key = ?", (key,)).fetchone()
            fields = (page.url, page.kind, page.title, page.site, page.author, page.date, page.text,
                      note.strip() or (row["note"] if row else ""), self.embedder.name, now)
            if row:
                document_id = row["id"]
                conn.execute(
                    "UPDATE documents SET url = ?, kind = ?, title = ?, site = ?, author = ?, date = ?,"
                    " text = ?, note = ?, model = ?, saved_at = ? WHERE id = ?", (*fields, document_id),
                )
            else:
                document_id = conn.execute(
                    "INSERT INTO documents (url, kind, title, site, author, date, text, note, model,"
                    " saved_at, key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (*fields, key),
                ).lastrowid
            conn.executemany("INSERT OR IGNORE INTO tags VALUES (?, ?)",
                             [(document_id, tag) for tag in clean_tags(tags or [])])
            self._store_chunks(conn, document_id, page.title, page.text)
            return self._document(conn, document_id), row is not None

    def _document(self, conn: sqlite3.Connection, document_id: int) -> Document:
        row = conn.execute(
            "SELECT url, kind, title, site, author, date, note, saved_at, length(text) AS chars"
            " FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        tags = conn.execute("SELECT tag FROM tags WHERE document_id = ? ORDER BY tag", (document_id,))
        return Document(row["url"], row["kind"], row["title"], row["site"], row["author"],
                        row["date"], row["note"], tuple(t["tag"] for t in tags), row["saved_at"],
                        row["chars"])

    def _ids(self, conn: sqlite3.Connection, tag: str | None) -> list[int]:
        """Ids of all documents, or of those with this tag, newest first."""
        if tag is None:
            rows = conn.execute("SELECT id FROM documents ORDER BY saved_at DESC, id DESC")
        else:
            rows = conn.execute(
                "SELECT id FROM documents JOIN tags ON tags.document_id = documents.id"
                " WHERE tag = ? ORDER BY saved_at DESC, id DESC", ((clean_tags([tag]) or [""])[0],),
            )
        return [row["id"] for row in rows]

    def documents(self, tag: str | None = None) -> list[Document]:
        """Every saved document (or those with this tag), most recently saved first."""
        with self._db() as conn:
            return [self._document(conn, document_id) for document_id in self._ids(conn, tag)]

    def tags(self) -> dict[str, int]:
        """Each tag and how many documents have it, most used first."""
        with self._db() as conn:
            rows = conn.execute("SELECT tag, count(*) AS n FROM tags GROUP BY tag ORDER BY n DESC, tag")
            return {row["tag"]: row["n"] for row in rows}

    def get(self, url: str) -> Document | None:
        with self._db() as conn:
            row = conn.execute("SELECT id FROM documents WHERE key = ?", (same_page_key(url),)).fetchone()
            return self._document(conn, row["id"]) if row else None

    def remove(self, url: str) -> Document:
        """Remove a saved page (its chunks and tags go with it). Returns what was removed."""
        with self._db() as conn:
            row = conn.execute("SELECT id FROM documents WHERE key = ?", (same_page_key(url),)).fetchone()
            if row is None:
                raise LibraryError(f"{url} isn't in the library")
            document = self._document(conn, row["id"])
            conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            return document

    def _reembed_stale(self, conn: sqlite3.Connection) -> None:
        """Re-embed documents saved with a different model: vectors from two models can't be compared."""
        stale = conn.execute("SELECT id, title, text FROM documents WHERE model != ?", (self.embedder.name,))
        for row in stale.fetchall():
            self._store_chunks(conn, row["id"], row["title"], row["text"])
            conn.execute("UPDATE documents SET model = ? WHERE id = ?", (self.embedder.name, row["id"]))

    def closest(self, vector: np.ndarray, limit: int, tag: str | None = None,
                exclude_url: str | None = None, min_score: float | None = None) -> list[Hit]:
        """The documents with a passage closest to vector (a unit vector), best first, each with
        its best passage. Documents scoring below min_score (by default the model's min_score,
        right for a query vector) are left out."""
        min_score = self.embedder.min_score if min_score is None else min_score
        exclude = same_page_key(exclude_url) if exclude_url else None
        with self._db() as conn:
            self._reembed_stale(conn)
            best: dict[int, tuple[float, sqlite3.Row]] = {}
            for document_id in self._ids(conn, tag):
                rows = conn.execute("SELECT start, text, vector FROM chunks WHERE document_id = ?",
                                    (document_id,)).fetchall()
                if not rows:
                    continue
                matrix = np.vstack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
                scores = matrix @ vector
                top = int(scores.argmax())
                best[document_id] = (float(scores[top]), rows[top])
            ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
            hits = []
            for document_id, (score, row) in ranked:
                if score < min_score or len(hits) == limit:
                    break
                document = self._document(conn, document_id)
                if exclude and same_page_key(document.url) == exclude:
                    continue
                hits.append(Hit(document, score, row["text"], row["start"]))
            return hits

    def search(self, query: str, limit: int, tag: str | None = None) -> list[Hit]:
        """The saved documents most related in meaning to query, best first."""
        return self.closest(self.embedder.embed_query(query), limit, tag)
