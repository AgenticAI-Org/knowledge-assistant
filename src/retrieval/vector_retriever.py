"""Vector/semantic retrieval over the persisted Chroma collection."""
from __future__ import annotations

import threading

from langchain_chroma import Chroma

from src.config import settings
from src.ingestion.embedder import get_embeddings_client
from src.schemas import ChunkMetadata, RetrievedChunk

_vectorstore: Chroma | None = None
_lock = threading.Lock()


def reset_cache() -> None:
    """Drops the cached Chroma handle so the next search reopens the collection.

    Required after an index rebuild: the rebuild replaces the persist directory
    underneath us, and a cached handle would keep pointing at the old one.
    """
    global _vectorstore
    with _lock:
        _vectorstore = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    with _lock:
        if _vectorstore is None:
            _vectorstore = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=get_embeddings_client(),
                persist_directory=str(settings.chroma_persist_dir),
            )
        return _vectorstore


def vector_search(query: str, k: int | None = None) -> list[RetrievedChunk]:
    k = settings.top_k_vector if k is None else k
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)

    hits: list[RetrievedChunk] = []
    for rank, (doc, _distance) in enumerate(results):
        meta = doc.metadata
        hits.append(
            RetrievedChunk(
                chunk_id=meta["chunk_id"],
                text=doc.page_content,
                # Chroma drops None-valued metadata, so `page` is simply absent
                # for DOCX/TXT sources; ChunkMetadata.page defaults to None.
                metadata=ChunkMetadata(**meta),
                vector_rank=rank,
            )
        )
    return hits
