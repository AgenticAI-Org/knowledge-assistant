"""Retrieval-only eval: compares vector-only, BM25-only, and hybrid+reranked retrieval.

Reports Hit Rate@k and MRR against `expected_sources` in eval_questions.jsonl. This is
what demonstrates hybrid search + reranking actually improve retrieval, rather than just
asserting it (see docs/EVALUATION.md).

Two things make the comparison valid:
  * All three arms are truncated to the SAME k (`settings.eval_top_k`). Scoring
    vector-only over 10 candidates against hybrid over 4 would compare Hit Rate@10
    with Hit Rate@4, which is not a comparison at all.
  * The corpus must contain meaningfully more chunks than k. If k >= the corpus size,
    every arm retrieves everything and every score is trivially 1.00. The run aborts
    with an explanation rather than emitting a meaningless table.

Run: uv run python -m eval.run_retrieval_eval
"""
from __future__ import annotations

from eval.common import (
    first_correct_rank,
    gold_chunk_ids,
    hit_rate_and_mrr,
    single_turn_questions,
)
from src.config import settings
from src.retrieval.bm25_retriever import bm25_search
from src.retrieval.hybrid import fuse
from src.retrieval.reranker import rerank
from src.retrieval.vector_retriever import vector_search


def _corpus_chunks():
    from src.retrieval.bm25_retriever import _load_index

    _, chunks = _load_index()
    return chunks


def run() -> str:
    questions = [q for q in single_turn_questions() if q.get("expects_answer")]
    k = settings.eval_top_k
    chunks = _corpus_chunks()
    corpus_size = len(chunks)

    if corpus_size <= max(settings.top_k_vector, settings.top_k_bm25):
        return (
            "## Retrieval Evaluation\n\n"
            f"**Not run.** The corpus holds {corpus_size} chunks but retrieval requests up to "
            f"{max(settings.top_k_vector, settings.top_k_bm25)} candidates, so every query returns "
            "the whole corpus and all three methods would score 1.00 by construction. "
            "Add documents or lower `CHUNK_SIZE` so the corpus is comfortably larger than k, "
            "then rebuild the index and re-run.\n"
        )

    doc_ranks: dict[str, list[int | None]] = {"Vector-only": [], "BM25-only": [], "Hybrid + Reranked": []}
    passage_ranks: dict[str, list[int | None]] = {k_: [] for k_ in doc_ranks}
    unresolved: list[str] = []

    for q in questions:
        question, expected = q["question"], q["expected_sources"]
        gold = gold_chunk_ids(q, chunks)
        if not gold:
            unresolved.append(q["id"])

        vector_hits = vector_search(question, k=settings.top_k_vector)
        bm25_hits = bm25_search(question, k=settings.top_k_bm25)
        fused = fuse(vector_hits, bm25_hits)
        reranked = rerank(question, fused, top_n=k)

        # Every arm is judged on its own top-k, so the arms are directly comparable.
        for label, hits in [
            ("Vector-only", vector_hits[:k]),
            ("BM25-only", bm25_hits[:k]),
            ("Hybrid + Reranked", reranked[:k]),
        ]:
            doc_ranks[label].append(first_correct_rank([c.metadata.source for c in hits], expected))
            if gold:
                passage_ranks[label].append(
                    first_correct_rank([c.chunk_id for c in hits], gold)
                )

    lines = [
        "## Retrieval Evaluation",
        "",
        f"Evaluated on {len(questions)} single-turn questions with known expected sources, "
        f"over a corpus of {corpus_size} chunks.",
        "",
        f"All three methods are scored at the same cut-off (k={k}); the hybrid arm draws its "
        f"candidates from {settings.top_k_vector} vector + {settings.top_k_bm25} BM25 hits "
        "fused by RRF and reranked.",
        "",
        "Two granularities are reported. **Document-level** asks whether the right *document* "
        "was retrieved; with only a handful of documents this saturates easily. **Passage-level** "
        "asks whether the specific chunk that actually contains the answer was retrieved, out of "
        f"{corpus_size} candidates -- that is the metric with room to separate the methods.",
        "",
        f"| Method | Doc Hit Rate@{k} | Doc MRR@{k} | Passage Hit Rate@{k} | Passage MRR@{k} |",
        "|---|---|---|---|---|",
    ]
    for label in ("Vector-only", "BM25-only", "Hybrid + Reranked"):
        d_hit, d_mrr = hit_rate_and_mrr(doc_ranks[label])
        p_hit, p_mrr = hit_rate_and_mrr(passage_ranks[label])
        lines.append(f"| {label} | {d_hit:.2f} | {d_mrr:.2f} | {p_hit:.2f} | {p_mrr:.2f} |")
    lines.append("")
    lines.append(
        f"Passage-level scored on {len(passage_ranks['Vector-only'])} of {len(questions)} questions "
        "(those annotated with `answer_keywords`)."
    )
    if unresolved:
        lines.append("")
        lines.append(
            "> **Warning:** no gold passage resolved for question(s) "
            + ", ".join(unresolved)
            + ". Their `answer_keywords` no longer match any chunk -- fix the annotation "
            "rather than reading this as a retrieval failure."
        )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
