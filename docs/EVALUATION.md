# Evaluation Strategy

Three layers: **retrieval quality**, **end-to-end answer quality**, and a lightweight
**operational eval** derived from the structured logs. Together they let the demo say
"here's how I know it works" rather than eyeballing a handful of chat turns — this is the
"Test" step of the assignment's `Understand → Design → Implement → Test → Explain` rubric.

## 1. Test Set — `eval/eval_questions.jsonl`

18 hand-written cases against the sample policy corpus:

- **13 single-turn, answerable** questions (`expects_answer: true`), each tagged with
  `expected_sources` — the document(s) that should be cited.
- **3 single-turn, unanswerable** questions (`expects_answer: false`) — topics genuinely
  absent from the corpus (sabbatical leave, pet insurance, pantry snacks), used to test
  the hallucination guardrail.
- **2 multi-turn** conversations, each a 2-turn sequence where the second turn only makes
  sense if conversational memory correctly resolves a pronoun/implicit reference (e.g.
  "What about carry-forward?" after asking about the leave policy).

## 2. Retrieval Evaluation — `eval/run_retrieval_eval.py`

For every answerable single-turn question, retrieval is run **three ways** — vector-only,
BM25-only, and hybrid+reranked — and scored against `expected_sources`:

Scored at **two granularities**:

- **Document-level** — did a chunk from an `expected_sources` document reach the top *k*?
- **Passage-level** — did the specific chunk that actually *contains the answer* reach the
  top *k*, out of the whole corpus?

Document-level is the weaker metric and saturates almost immediately: with only 7 documents,
retrieving 4 chunks makes hitting the right document nearly free, and all three arms score
1.00. Passage-level is the one with headroom, and it is what the comparison actually rests on.

The passage ground truth comes from `answer_keywords` on each answerable question — the
distinctive fact tokens ("24 days", "INR 2,500"). Gold chunk IDs are resolved against the
live corpus at eval time rather than stored in the file, so the annotation survives
re-chunking; chunk IDs shift whenever `CHUNK_SIZE` changes. If a question's keywords stop
matching any chunk, the report flags it as a broken annotation instead of silently counting
it as a retrieval miss.

For each granularity:

- **Hit Rate@k** — fraction of questions where a correct chunk appears in the top *k*.
- **MRR@k (Mean Reciprocal Rank)** — average of `1 / rank` of the first correct chunk
  within the top *k* (0 if none found).

Reporting all three side by side is the point: it's the concrete evidence that hybrid
search + reranking actually improve retrieval over either method alone, not just an
architectural claim. Two conditions have to hold for that evidence to mean anything, and
the script enforces both:

**All three arms are scored at the same k** (`EVAL_TOP_K`, default 4). The hybrid arm draws
a wider candidate pool (`TOP_K_VECTOR` + `TOP_K_BM25`) and narrows it via RRF and the
cross-encoder, but it is judged on the same number of final results as the baselines.
Scoring vector-only over 10 candidates against hybrid over 4 would be comparing Hit Rate@10
with Hit Rate@4 — not a comparison at all, and one biased against the method under test.

**The corpus must be substantially larger than k.** If `TOP_K_VECTOR`/`TOP_K_BM25` are
greater than or equal to the total chunk count, every query trivially retrieves the entire
corpus, every method scores Hit Rate 1.00, and the eval demonstrates nothing. The retrieval
eval refuses to emit a table in that case and says so instead; `build_index` also logs a
`retrieval_k_exceeds_corpus` warning at ingest time.

Run: `uv run python -m eval.run_retrieval_eval`

## 3. Generation Evaluation — `eval/run_generation_eval.py`

Runs the full `answer_service.answer()` pipeline per question and scores:

- **Faithfulness** and **Answer Relevancy** (0.0–1.0 each) via a **hand-rolled LLM-judge**
  (`JudgeScore`, structured output from `gpt-4o-mini`) — a full framework like RAGAS was
  evaluated but pulled in a fragile transitive dependency (`langchain_community.chat_models.vertexai`,
  broken against this project's pinned `langchain-community` version) for a scope this
  small, so a ~40-line judge prompt was used instead, per the "hand-rolled equivalent"
  fallback in the original design.
- **Citation Coverage** — did the answer's citations include at least one expected source?
- **Refusal Accuracy** — for the unanswerable questions, did the system correctly return
  the "not found" fallback instead of fabricating an answer?
- **Multi-turn Memory Accuracy** — for each multi-turn case, does the *final* turn's answer
  still cite the correct source after depending on conversational memory to resolve the
  follow-up?

Run: `uv run python -m eval.run_generation_eval`

## 4. Operational Evaluation (log-derived)

Not a separate harness — `eval/build_report.py` parses `logs/app.log` (JSON lines, see
`docs/LOGGING.md`) and reports p50/p95 latency per pipeline stage (`condense_query`,
`retrieval`, `rerank`, `generation`) plus total input/output token counts observed across
whatever's been logged so far (app usage + eval runs both contribute).

## 5. Combined Report

`uv run python -m eval.build_report` runs all three layers and writes `eval/report.md`
(gitignored — regenerate it rather than committing a stale copy) with one table per layer.
This is the artifact to include in the submission as evidence of testing.

## 6. Known Limitations of This Eval

- The LLM-judge is itself an LLM call and therefore not perfectly reliable — scores should
  be read as directional signal, not ground truth. A larger/held-out human-labeled set
  would be the natural next step at real scale.
- The sample corpus is small (7 short documents, ~57 chunks). That is large enough for k=10
  retrieval to be genuinely selective, but Hit Rate/MRR will still read better here than on a
  real enterprise policy corpus with thousands of chunks and many near-duplicate passages.
- Document-level scores are reported for continuity but are saturated at this corpus size
  and should not be read as evidence of anything. Read the passage-level columns.
- The eval question set (18 cases) is small enough that a single question moves Hit Rate by
  roughly 0.08, so small differences between arms are not statistically meaningful.
