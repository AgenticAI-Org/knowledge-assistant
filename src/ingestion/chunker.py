"""Splits loaded documents into retrieval-sized chunks with deterministic IDs."""
from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.schemas import Chunk, ChunkMetadata


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    per_source_counter: dict[str, int] = defaultdict(int)
    chunks: list[Chunk] = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        for piece in splitter.split_text(doc.page_content):
            if not piece.strip():
                continue
            index = per_source_counter[source]
            per_source_counter[source] += 1
            chunks.append(
                Chunk(
                    text=piece,
                    metadata=ChunkMetadata(
                        source=source,
                        chunk_id=f"{source}::{index}",
                        chunk_index=index,
                        page=doc.metadata.get("page"),
                    ),
                )
            )

    return chunks
