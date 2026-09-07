"""Shared Pydantic data models used across ingestion, retrieval, and generation."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    source: str
    chunk_id: str
    chunk_index: int
    page: int | None = None


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    vector_rank: int | None = None
    bm25_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


class Citation(BaseModel):
    source: str
    chunk_ids: list[str]


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    latency_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class IndexStats(BaseModel):
    num_source_documents: int
    num_chunks: int
    sources: list[str]
