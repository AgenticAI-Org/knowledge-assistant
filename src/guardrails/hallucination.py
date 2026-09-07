"""Hallucination guardrail: decides whether retrieved context is strong enough to answer from.

Two checks, cheap-first (see docs/LLD.md §3.3):
1. Pre-generation: is the top reranked chunk's score above a floor? If not, skip the LLM
   call entirely and return the fallback -- saves cost/latency and is fully deterministic.
2. Post-generation: did the model produce a refusal itself despite decent context
   (e.g. the retrieved chunks were topically close but didn't actually contain the answer)?
"""
from __future__ import annotations

import re

from src.config import settings
from src.schemas import Answer, Citation, RetrievedChunk

NOT_FOUND_MESSAGE = "I could not find this information in the available documents."

# The model is instructed to return NOT_FOUND_MESSAGE verbatim, but it paraphrases
# often enough that a bare substring test misses real refusals -- and a refusal that
# slips through is recorded as a grounded answer and then feeds conversation memory.
_REFUSAL_PATTERNS = (
    r"could not find (?:this|that|the) information",
    r"couldn'?t find (?:this|that|the) information",
    r"(?:can(?:no|')?t|unable to|not able to) find (?:this|that|it|the answer|any information)",
    r"(?:do(?:es)? not|don'?t|doesn'?t) (?:contain|include|have|provide|mention|specify)"
    r"(?:\s+\w+){0,4}\s+(?:information|details|answer)",
    r"(?:no|not any) (?:relevant )?information (?:is )?(?:available|found|provided)",
    r"(?:is )?not (?:covered|addressed|specified|mentioned) in the (?:available |provided )?"
    r"(?:documents?|policies|policy|context)",
)
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def has_sufficient_context(
    retrieved: list[RetrievedChunk], *, apply_score_floor: bool = True
) -> bool:
    """True only when the top candidate was actually scored and cleared the floor.

    Fails closed: an unscored candidate means the rerank stage did not do its job,
    which is exactly when the guardrail should engage rather than stand down.

    `apply_score_floor=False` checks only that something was retrieved. Callers use
    it for broad/instruction-style queries, where the cross-encoder scores every
    passage low regardless of relevance and the floor would reject valid context.
    Grounding for those queries still rests on the answer prompt and the
    post-generation refusal check, which is what catches genuinely absent topics.
    """
    if not retrieved:
        return False
    if not apply_score_floor:
        return True
    top_score = retrieved[0].rerank_score
    if top_score is None:
        return False
    return top_score >= settings.rerank_score_floor


def build_fallback_answer(latency_ms: int = 0) -> Answer:
    return Answer(text=NOT_FOUND_MESSAGE, citations=[], grounded=False, latency_ms=latency_ms)


def is_self_reported_refusal(answer_text: str) -> bool:
    """Whether the model's own answer is a refusal rather than a grounded answer."""
    normalized = " ".join(answer_text.split())
    if not normalized:
        return True
    return bool(_REFUSAL_RE.search(normalized))


def build_citations(retrieved: list[RetrievedChunk]) -> list[Citation]:
    by_source: dict[str, list[str]] = {}
    for chunk in retrieved:
        by_source.setdefault(chunk.metadata.source, []).append(chunk.chunk_id)
    return [Citation(source=source, chunk_ids=ids) for source, ids in by_source.items()]
