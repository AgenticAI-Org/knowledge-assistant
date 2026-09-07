"""Builds the Chroma vector index and the BM25 keyword corpus from data/.

Rebuilds both indexes together every run so chunk IDs never drift apart
between the two retrieval backends -- see docs/LLD.md §3.3.

The rebuild is staged: the new vector index is written to a temporary directory
and only swapped into place once every batch has embedded successfully. A failure
partway through (API outage, exhausted quota, unreadable document) therefore
leaves the previous working index untouched rather than destroying it.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import RETRYABLE_ERRORS, get_embeddings_client
from src.ingestion.loaders import load_documents
from src.logging_config import get_stage_logger
from src.retrieval import bm25_retriever, vector_retriever
from src.schemas import Chunk, IndexStats


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=RETRYABLE_ERRORS,
)
def _add_batch(vectorstore: Chroma, batch: list[Chunk]) -> None:
    """Embeds and writes one batch, retrying only on transient API failures."""
    vectorstore.add_texts(
        texts=[c.text for c in batch],
        metadatas=[c.metadata.model_dump() for c in batch],
        ids=[c.metadata.chunk_id for c in batch],
    )


def _build_vector_index(chunks: list[Chunk], target_dir: Path) -> None:
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings_client(),
        persist_directory=str(target_dir),
    )
    # Batched so a large corpus stays inside Chroma's max batch size and the
    # embedding API's per-request token limit.
    size = settings.embed_batch_size
    for start in range(0, len(chunks), size):
        _add_batch(vectorstore, chunks[start : start + size])


def build_index() -> IndexStats:
    logger = get_stage_logger(__name__, session_id="ingestion", query_id="build_index")

    documents, failed = load_documents(settings.data_dir)
    if failed:
        # Refuse to replace a working index with a silently partial corpus.
        raise RuntimeError(
            "Could not read these documents, so the index was not rebuilt: "
            + ", ".join(sorted(failed))
        )
    if not documents:
        raise RuntimeError(f"No documents found in {settings.data_dir}")

    chunks: list[Chunk] = chunk_documents(documents)
    if not chunks:
        raise RuntimeError("Documents loaded but produced no chunks")
    logger.info(
        "chunked_documents",
        extra={"stage": "chunking", "num_documents": len(documents), "num_chunks": len(chunks)},
    )

    if settings.top_k_vector >= len(chunks) or settings.top_k_bm25 >= len(chunks):
        # Not fatal, but it makes retrieval a no-op and the retrieval eval meaningless:
        # every query would return the entire corpus. See docs/EVALUATION.md.
        logger.warning(
            "retrieval_k_exceeds_corpus",
            extra={
                "stage": "chunking",
                "num_chunks": len(chunks),
                "top_k_vector": settings.top_k_vector,
                "top_k_bm25": settings.top_k_bm25,
            },
        )

    # --- Vector index (Chroma), staged then swapped ---
    settings.chroma_persist_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="chroma-build-", dir=settings.chroma_persist_dir.parent)
    )
    previous = settings.chroma_persist_dir.with_name(settings.chroma_persist_dir.name + ".old")
    try:
        _build_vector_index(chunks, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        logger.exception("vector_index_build_failed", extra={"stage": "vector_index"})
        raise

    shutil.rmtree(previous, ignore_errors=True)
    if settings.chroma_persist_dir.exists():
        settings.chroma_persist_dir.rename(previous)
    staging.rename(settings.chroma_persist_dir)
    shutil.rmtree(previous, ignore_errors=True)
    logger.info("vector_index_built", extra={"stage": "vector_index", "num_chunks": len(chunks)})

    # --- Keyword corpus (BM25 model is rebuilt from this at query time) ---
    bm25_retriever.write_corpus(chunks)
    logger.info("bm25_index_built", extra={"stage": "bm25_index", "num_chunks": len(chunks)})

    # Long-lived processes (the Streamlit app) cache both retrievers in module
    # globals; without this they would keep serving the pre-rebuild corpus.
    bm25_retriever.reset_cache()
    vector_retriever.reset_cache()

    sources = sorted({c.metadata.source for c in chunks})
    return IndexStats(num_source_documents=len(sources), num_chunks=len(chunks), sources=sources)
