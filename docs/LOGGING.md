# Logging Mechanism

Logging is treated as a first-class module (`src/logging_config.py`), not `print()`
statements — it's an explicit grading line item ("Engineering Practices: ... logging") and
is also what powers the operational half of the evaluation report (`docs/EVALUATION.md` §4).

## 1. Design

- **Structured JSON**, one object per line, written by `JsonFormatter` — easy to `grep`/
  parse programmatically (the eval report parser reads this format directly).
- **Two handlers**: a rotating file handler (`logs/app.log`, 5 MB × 5 backups) and a
  plain-text console handler at `INFO` for live visibility during the demo.
- **Per-query context**: every pipeline call is logged through a `StageLogger`
  (`logging.LoggerAdapter`) stamped with `session_id` (Streamlit session) and `query_id`
  (a fresh `uuid4` per question). Filtering `logs/app.log` on one `query_id` reconstructs
  that question's full trace end to end: condense → retrieve → rerank → generate.
- **Timing**: the `stage_timer` context manager wraps each pipeline stage, logging one
  `INFO` record on success (`stage_complete`) or `ERROR` on failure (`stage_failed`), both
  carrying `duration_ms`.

## 2. Log Levels

| Level | Used for |
|---|---|
| `DEBUG` | Full prompt/context payloads — **never logged at INFO**, to avoid dumping entire policy documents into the log file by default. |
| `INFO` | Stage boundaries, timings, and summary results (query text, chunk counts, sources cited, latency). |
| `WARNING` | Guardrail triggers — pre-generation ("context too weak, skipping LLM call") and post-generation ("model self-reported a refusal"). |
| `ERROR` | Tool/API failures, with stack trace (`stage_failed`). |

## 3. Field Schema (INFO records)

Every `stage_complete` record carries at minimum:

| Field | Meaning |
|---|---|
| `timestamp`, `level`, `logger`, `message` | Standard `logging` fields |
| `session_id` | Streamlit session UUID |
| `query_id` | Per-question UUID |
| `stage` | One of `condense_query`, `retrieval`, `rerank`, `generation` |
| `duration_ms` | Stage wall-clock time |

Plus stage-specific fields:

| Stage | Extra fields |
|---|---|
| `condense_query` | `condensed` (bool — was the query actually rewritten) |
| `retrieval` | `vector_hits`, `bm25_hits`, `fused_candidates` |
| `rerank` | `reranked_count`, `top_score` |
| `generation` | `input_tokens`, `output_tokens` |
| `answer_service` (final summary) | `latency_ms`, `grounded`, `num_citations` |

`ingestion.indexer` logs its own stages (`chunking`, `vector_index`, `bm25_index`) with
`num_documents`/`num_chunks` under `session_id="ingestion"`.

## 4. What Is Never Logged

- `OPENAI_API_KEY` or any secret — excluded by construction (never placed into a logged
  `extra` dict).
- Full prompt/context text at `INFO` level or above (see DEBUG note above).

## 5. Operational Use

- **During the demo**: `tail -f logs/app.log | python -m json.tool` (or just eyeball the
  console handler output) to show a question's trace live.
- **For evaluation**: `eval/build_report.py` reads `logs/app.log`, groups `duration_ms` by
  `stage`, and reports p50/p95 latency plus aggregate token usage as the "Operational
  Summary" section of `eval/report.md`.
