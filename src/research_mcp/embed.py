"""Turning text into vectors that capture its meaning, with a small model that runs locally.

No API key, and it works offline after the first ~70 MB download. Each text becomes a list of
384 numbers (a "vector"); texts that mean similar things get vectors pointing the same way, so
"how do computers agree on a value" lands close to a page about Raft consensus.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import re
from functools import cached_property
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
# Texts embedded together. Small batches are faster here: every text in a batch is padded to
# the length of the longest one.
BATCH_SIZE = 4


def readable(text: str) -> str:
    """Markdown as a person reads it: link text without the link, no images or HTML tags.

    URLs mean nothing to the model but cost a lot: the model's time grows with the length of the
    text, and a Wikipedia paragraph is often half link targets. Dropping them makes embedding
    about four times faster, and the vectors are about the words, not the URLs.
    """
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # Images.
    text = re.sub(r"\]\([^)]*\)", "]", text)  # [text](url) -> [text]
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split())


def unit(vectors: np.ndarray) -> np.ndarray:
    """Scale vectors to length 1, so a dot product between two of them is their cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class Embedder:
    """The embedding model, loaded the first time it's used (so the server starts instantly)."""

    # Similarity scores go from -1 to 1, but with this model even unrelated text scores about
    # 0.5, while real matches for a search query score about 0.65 and up. Results below this
    # are dropped.
    min_score = 0.6
    # Two passages (rather than a query and a passage) score higher: two pages on unrelated
    # computer topics score about 0.7, while pages on the same subject score 0.8 and up.
    min_similar_score = 0.75
    name = MODEL

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

    @cached_property
    def _model(self):
        from fastembed import TextEmbedding  # Imported here: it takes a second or two to load.

        return TextEmbedding(MODEL, cache_dir=str(self.model_dir))

    def embed(self, texts: list[str]) -> np.ndarray:
        """One unit vector per text (a matrix with a row per text), for passages to search."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = self._model.embed([readable(t) for t in texts], batch_size=BATCH_SIZE)
        return unit(np.array(list(vectors), dtype=np.float32))

    def embed_query(self, query: str) -> np.ndarray:
        """A unit vector for a search query. The model expects queries to be marked as such
        (query_embed adds an instruction prefix), which makes matching passages score higher."""
        return unit(np.array(list(self._model.query_embed([query]))[0], dtype=np.float32))
