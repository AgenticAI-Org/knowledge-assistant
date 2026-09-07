"""Keyword retrieval over the persisted BM25 corpus.

The on-disk artifact is plain JSON holding the chunk list, not a pickled
BM25Okapi. The BM25 model is rebuilt from that corpus on load, which keeps the
index format inspectable, portable across library versions, and -- most
importantly -- avoids `pickle.load` on a file that may have been produced
elsewhere. Tokenization therefore lives in exactly one place, so the query and
the corpus can never drift apart.
"""
from __future__ import annotations

import json
import threading

from rank_bm25 import BM25Okapi

from src.config import settings
from src.schemas import Chunk, RetrievedChunk

_bm25: BM25Okapi | None = None
_chunks: list[Chunk] | None = None
_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def write_corpus(chunks: list[Chunk]) -> None:
    """Persists the chunk corpus that `_load_index` rebuilds the BM25 model from."""
    settings.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunks": [c.model_dump() for c in chunks]}
    settings.bm25_index_path.write_text(json.dumps(payload), encoding="utf-8")


def reset_cache() -> None:
    """Drops the in-process BM25 cache so the next search re-reads from disk.

    Called after an index rebuild -- without it a long-lived process (the
    Streamlit app) keeps answering from the corpus it loaded at startup.
    """
    global _bm25, _chunks
    with _lock:
        _bm25 = None
        _chunks = None


def _load_index() -> tuple[BM25Okapi, list[Chunk]]:
    global _bm25, _chunks
    with _lock:
        if _bm25 is None or _chunks is None:
            if not settings.bm25_index_path.exists():
                raise FileNotFoundError(
                    f"BM25 corpus not found at {settings.bm25_index_path}. "
                    "Run `uv run python -m scripts.build_index` first."
                )
            raw = json.loads(settings.bm25_index_path.read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in raw["chunks"]]
            _bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
            _chunks = chunks
        return _bm25, _chunks


def bm25_search(query: str, k: int | None = None) -> list[RetrievedChunk]:
    k = settings.top_k_bm25 if k is None else k
    bm25, chunks = _load_index()
    scores = bm25.get_scores(tokenize(query))

    # Drop zero-score chunks before slicing, so `bm25_rank` is a dense 0..n-1
    # ranking over actual matches rather than over padded non-matches.
    matches = [i for i in range(len(scores)) if scores[i] > 0]
    matches.sort(key=lambda i: scores[i], reverse=True)

    return [
        RetrievedChunk(
            chunk_id=chunks[i].metadata.chunk_id,
            text=chunks[i].text,
            metadata=chunks[i].metadata,
            bm25_rank=rank,
        )
        for rank, i in enumerate(matches[:k])
    ]
