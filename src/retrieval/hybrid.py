"""Reciprocal Rank Fusion (RRF) of vector and BM25 result lists.

RRF is used instead of blending raw scores because cosine similarity
(vector search) and BM25 scores live on incomparable scales -- fusing by
rank position avoids that normalization problem entirely.
"""
from __future__ import annotations

from src.config import settings
from src.schemas import RetrievedChunk


def fuse(
    vector_hits: list[RetrievedChunk],
    bm25_hits: list[RetrievedChunk],
    rrf_k: int = settings.rrf_k,
) -> list[RetrievedChunk]:
    by_id: dict[str, RetrievedChunk] = {}

    for hit in vector_hits:
        by_id[hit.chunk_id] = hit.model_copy()

    for hit in bm25_hits:
        if hit.chunk_id in by_id:
            by_id[hit.chunk_id].bm25_rank = hit.bm25_rank
        else:
            by_id[hit.chunk_id] = hit.model_copy()

    for chunk in by_id.values():
        score = 0.0
        if chunk.vector_rank is not None:
            score += 1.0 / (rrf_k + chunk.vector_rank + 1)
        if chunk.bm25_rank is not None:
            score += 1.0 / (rrf_k + chunk.bm25_rank + 1)
        chunk.fused_score = score

    return sorted(by_id.values(), key=lambda c: c.fused_score, reverse=True)
