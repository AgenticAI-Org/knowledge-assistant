"""Shared helpers for the eval scripts: loading the question set and retrieval metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVAL_QUESTIONS_PATH = Path(__file__).resolve().parent / "eval_questions.jsonl"


def load_questions() -> list[dict[str, Any]]:
    questions = []
    with open(EVAL_QUESTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def single_turn_questions() -> list[dict[str, Any]]:
    return [q for q in load_questions() if q.get("type") == "single_turn"]


def multi_turn_questions() -> list[dict[str, Any]]:
    return [q for q in load_questions() if q.get("type") == "multi_turn"]


def first_correct_rank(retrieved_sources: list[str], expected_sources: list[str]) -> int | None:
    """1-indexed rank of the first retrieved chunk whose source is in expected_sources, else None."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return rank
    return None


def hit_rate_and_mrr(all_ranks: list[int | None]) -> tuple[float, float]:
    n = len(all_ranks)
    if n == 0:
        return 0.0, 0.0
    hits = sum(1 for r in all_ranks if r is not None)
    mrr = sum(1.0 / r for r in all_ranks if r is not None) / n
    return hits / n, mrr


def gold_chunk_ids(question: dict[str, Any], chunks: list[Any]) -> list[str]:
    """Chunk IDs of the passages that actually contain the answer to `question`.

    Resolved from `answer_keywords` against the live corpus rather than stored as
    fixed chunk IDs, so the ground truth survives re-chunking (IDs shift whenever
    CHUNK_SIZE changes). A question with no keywords has no passage-level ground
    truth and is skipped by the passage metrics.
    """
    keywords = question.get("answer_keywords") or []
    if not keywords:
        return []
    return [
        c.metadata.chunk_id
        for c in chunks
        if all(k.lower() in c.text.lower() for k in keywords)
    ]
