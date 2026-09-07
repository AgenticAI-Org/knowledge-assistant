# High-Level Design — Employee Knowledge Assistant

## 1. Purpose

An internal RAG assistant that answers employee questions (leave, IT, travel, benefits,
code of conduct) from a local corpus of company policy documents, grounding every answer
in retrieved text and citing its sources. Built for the IIT Patna USDC GenAI Development
Program capstone (Project 2 — Enterprise Knowledge Assistant with Advanced RAG).

## 2. Architecture

![Architecture Diagram](architecture_diagram.png)

The system has two independent paths:

- **Ingestion path** (offline, run via `scripts/build_index.py` whenever `data/` changes):
  loads documents → splits into chunks → embeds each chunk with OpenAI embeddings →
  persists into a local ChromaDB collection, and in parallel builds a BM25 keyword index
  over the same chunk set.
- **Query path** (online, once per user turn): a follow-up question is first condensed
  into a standalone query using recent chat history, then retrieved via both vector
  search and BM25, fused with Reciprocal Rank Fusion (RRF), reranked with a local
  cross-encoder, and finally answered by the LLM using only the top reranked context —
  or, if that context is too weak, answered with an explicit "not found" fallback instead
  of guessing.

## 3. Components

| Component | Responsibility | Key module |
|---|---|---|
| Ingestion Service | Load, chunk, embed, and index documents | `src/ingestion/` |
| Vector Retriever | Semantic search over Chroma | `src/retrieval/vector_retriever.py` |
| BM25 Retriever | Keyword search over the local BM25 index | `src/retrieval/bm25_retriever.py` |
| Hybrid Fusion | Combines the two ranked lists via RRF | `src/retrieval/hybrid.py` |
| Reranker | Cross-encoder scoring of fused candidates | `src/retrieval/reranker.py` |
| Memory Manager | Query condensation + chat history formatting | `src/memory/conversation.py` |
| Generation Service | Builds the grounded prompt and calls the LLM | `src/generation/` |
| Hallucination Guardrail | Decides if context is sufficient / detects refusal | `src/guardrails/hallucination.py` |
| Logging Subsystem | Structured, per-query traceable logging | `src/logging_config.py` |
| Streamlit UI | Chat, history, sources, reset, error display | `app/streamlit_app.py` |

## 4. Key Design Decisions

- **RRF over score-blending** — cosine similarity (vector) and BM25 scores live on
  incomparable scales; fusing by rank position sidesteps normalization entirely and is
  the standard technique for hybrid retrieval.
- **Local cross-encoder reranker** — keeps the project free of a second paid API,
  satisfying the assignment's "no expensive infra" constraint, at the cost of being
  CPU-bound and therefore the slowest single stage in the query path.
- **Two-stage hallucination guardrail** — a cheap pre-generation check (is the top
  reranked score above a floor?) avoids paying for an LLM call when context is clearly
  insufficient; a cheap post-generation check catches the case where the model itself
  declines despite passable context. Both are heuristics, not a second LLM-judge call,
  to keep runtime latency/cost low — a stricter LLM-judge check is reserved for the
  offline eval harness (`eval/run_generation_eval.py`).
- **Citations derived from retrieved chunks, not parsed from the LLM's output** — the
  sources cited are exactly the sources of the chunks fed to the LLM as context, which is
  more reliable than trusting the model to self-report which sources it used.
- **ChromaDB over FAISS** — native per-chunk metadata storage (source filename, page,
  chunk index) is what citations are built from; Chroma's persistence API is simpler to
  wire into a Streamlit app than raw FAISS + a manual metadata store.

## 5. Non-Goals / Limitations

- No multi-user auth or access control — this is a single-tenant local demo.
- No incremental re-indexing — `build_index()` rebuilds the full index every run so the
  vector and BM25 indexes never drift apart; fine at this corpus size, would need a diffing
  strategy at real enterprise scale.
- Query condensation adds one extra LLM round-trip per follow-up turn — a latency/cost
  trade-off made in exchange for correctness on pronoun/topic-carry-over questions.

See `docs/LLD.md` for module-level detail, `docs/EVALUATION.md` for how this design is
tested, and `docs/LOGGING.md` for the logging schema.
