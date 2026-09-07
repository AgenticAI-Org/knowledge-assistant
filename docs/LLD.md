# Low-Level Design — Employee Knowledge Assistant

## 1. Project Structure

```
knowledge-assistant/
├── README.md
├── docs/                   HLD.md, LLD.md, EVALUATION.md, LOGGING.md, architecture diagram
├── data/                   sample policy documents (PDF/DOCX/TXT)
├── storage/                generated at runtime: chroma/ + bm25_index.pkl (gitignored)
├── eval/                   eval_questions.jsonl, retrieval/generation eval scripts, report.md
├── src/
│   ├── config.py           pydantic Settings, reads .env
│   ├── logging_config.py   structured JSON logging + StageLogger/stage_timer
│   ├── schemas.py          Pydantic models: Chunk, RetrievedChunk, Answer, Citation, IndexStats
│   ├── ingestion/          loaders.py, chunker.py, embedder.py, indexer.py
│   ├── retrieval/          vector_retriever.py, bm25_retriever.py, hybrid.py, reranker.py
│   ├── memory/             conversation.py (ChatTurn, condense_query, format_history)
│   ├── generation/         prompts.py, llm.py, answer_service.py
│   └── guardrails/         hallucination.py
├── app/streamlit_app.py    UI entrypoint
├── scripts/build_index.py  CLI: uv run python -m scripts.build_index
├── logs/                   gitignored, created at runtime
└── pyproject.toml          uv-managed deps
```

## 2. Core Data Models (`src/schemas.py`)

```python
class ChunkMetadata(BaseModel):
    source: str          # filename, e.g. "Leave_Policy.pdf"
    chunk_id: str          # f"{source}::{chunk_index}", deterministic
    chunk_index: int
    page: int | None

class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    vector_rank: int | None
    bm25_rank: int | None
    fused_score: float
    rerank_score: float | None

class Citation(BaseModel):
    source: str
    chunk_ids: list[str]

class Answer(BaseModel):
    text: str
    citations: list[Citation]
    grounded: bool                 # False -> hallucination guardrail triggered
    latency_ms: int
    token_usage: dict[str, int]
    retrieved_chunks: list[RetrievedChunk]

class IndexStats(BaseModel):
    num_source_documents: int
    num_chunks: int
    sources: list[str]
```

Structured outputs are used throughout (not raw strings) so both the UI and the eval
harness can consume the same objects deterministically.

## 3. Module Contracts

- **`ingestion/loaders.py::load_documents(data_dir) -> list[Document]`** — dispatches by
  extension (`.pdf`, `.docx`, `.txt`) to a per-format loader; a per-file failure is logged
  and skipped rather than aborting the whole batch.
- **`ingestion/chunker.py::chunk_documents(documents) -> list[Chunk]`** — `RecursiveCharacterTextSplitter`
  (`chunk_size`/`chunk_overlap` from config); assigns a deterministic `chunk_id` per
  `(source, index)` pair so the same chunk always gets the same ID across rebuilds.
- **`ingestion/embedder.py::embed_texts(client, texts)`** — thin retry wrapper (via
  `tenacity`) around `OpenAIEmbeddings.embed_documents`.
- **`ingestion/indexer.py::build_index() -> IndexStats`** — idempotent: clears and
  rebuilds *both* the Chroma collection and the BM25 pickle on every call so the two
  indexes can never drift out of sync (same `chunks` list backs both).
- **`retrieval/vector_retriever.py::vector_search(query, k) -> list[RetrievedChunk]`** —
  semantic search via the persisted Chroma collection; sets `vector_rank`.
- **`retrieval/bm25_retriever.py::bm25_search(query, k) -> list[RetrievedChunk]`** —
  loads the pickled `(BM25Okapi, chunks)` pair once (module-level cache), scores by
  tokenized overlap; sets `bm25_rank`.
- **`retrieval/hybrid.py::fuse(vector_hits, bm25_hits, rrf_k) -> list[RetrievedChunk]`** —
  Reciprocal Rank Fusion: `score = Σ 1/(rrf_k + rank + 1)` across whichever lists a chunk
  appears in; deduplicates by `chunk_id`.
- **`retrieval/reranker.py::rerank(query, candidates, top_n) -> list[RetrievedChunk]`** —
  batches `(query, chunk.text)` pairs through a local `sentence-transformers` CrossEncoder;
  sets `rerank_score`, returns the top `top_n`.
- **`memory/conversation.py`**:
  - `ChatTurn(question, answer)` — the unit of stored history (owned by Streamlit
    `session_state`, not this module).
  - `format_history(history, max_turns) -> str` — renders the last N turns for prompt
    injection.
  - `condense_query(client, history, question) -> str` — rewrites a follow-up into a
    standalone query; **skips the LLM call entirely when `history` is empty** (the first
    turn is already standalone), which is both a cost optimization and the reason a fresh
    session never pays the condensation tax.
- **`generation/answer_service.py::answer(question, history, session_id) -> Answer`** —
  the single orchestration entrypoint the UI calls. Pipeline: condense → hybrid retrieve →
  classify intent → rerank → guardrail pre-check → generate → guardrail post-check → build
  citations. Every
  stage is wrapped in `stage_timer` (see `docs/LOGGING.md`) and tagged with a per-question
  `query_id`.
- **`retrieval/query_intent.py::classify(query) -> QueryIntent`** — labels the condensed
  query `FACTOID` or `BROAD` with a small keyword heuristic. Two decisions depend on it:

  | | `FACTOID` | `BROAD` |
  |---|---|---|
  | Chunks kept | `top_k_rerank` (4) | `top_k_rerank_broad` (12) |
  | Rerank score floor | applied | **not applied** |

  The cross-encoder scores "is this passage the answer to this *question*". An instruction
  like "summarise the leave policy in bullet points" is not a question, so every passage
  scores low however relevant it is — that query tops out around −7 while "How many days of
  annual leave?" scores +7 against the same document. A single absolute threshold across
  both is meaningless, so the floor is applied only where it means something. Grounding for
  broad queries rests on the answer prompt and the post-generation refusal check, which is
  what actually catches absent topics: the "sabbatical leave" eval case scores −2.4 and
  clears the floor anyway, and is refused post-generation regardless.

  The wider window exists because answering "summarise the leave policy" from 4 chunks of a
  10-chunk policy yields a summary that silently omits most of it.

  The heuristic is deliberately conservative — an unmatched query falls back to `FACTOID`,
  preserving the previous behaviour. It is a keyword match rather than an LLM call so it
  stays deterministic, free, and testable (`tests/test_query_intent.py`).
- **`guardrails/hallucination.py`**:
  - `has_sufficient_context(retrieved, *, apply_score_floor=True) -> bool` — top
    `rerank_score >= settings.rerank_score_floor`. Fails closed on an unscored candidate.
    `apply_score_floor=False` (broad queries) checks only that something was retrieved.
  - `build_fallback_answer(latency_ms) -> Answer` — the deterministic "not found" response,
    used when context is insufficient (pre-generation) so the LLM is never called.
  - `is_self_reported_refusal(answer_text) -> bool` — catches the case where context passed
    the floor check but the model still declined; in that case citations are dropped.
  - `build_citations(retrieved) -> list[Citation]` — groups the final context chunks by
    source; this is what "grounds" the citation list in what the LLM actually saw.

## 4. Configuration (`src/config.py`)

A single `Settings(BaseSettings)` reads from `.env` (see `.env.example`): API keys, model
names (`llm_model`, `embedding_model`, `rerank_model_name`), retrieval knobs
(`top_k_vector`, `top_k_bm25`, `rrf_k`, `top_k_rerank`, `top_k_rerank_broad`,
`rerank_score_floor`), chunking
knobs (`chunk_size`, `chunk_overlap`), memory window (`memory_window_turns`), and logging
(`log_level`, `log_json`). No hard-coded values live in the pipeline modules themselves.

## 5. Query Sequence

```
User question
  -> condense_query()                (skipped if first turn)
  -> vector_search() + bm25_search()  (parallelizable; run sequentially for simplicity)
  -> fuse()                           (RRF)
  -> rerank()                         (cross-encoder, top_k_rerank)
  -> has_sufficient_context()?
       no  -> build_fallback_answer() -> return
       yes -> build prompt (context + history) -> invoke_llm()
              -> is_self_reported_refusal()?
                   yes -> citations = [] , grounded = False
                   no  -> build_citations() , grounded = True
  -> Answer(...) -> Streamlit renders text + sources expander
```
