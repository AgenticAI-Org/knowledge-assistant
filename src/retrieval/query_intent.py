"""Classifies a query as a factoid lookup or a broad "tell me about X" request.

Two downstream decisions depend on this, and both go wrong when every query is
treated as a factoid question:

1. **The rerank score floor.** The cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores
   "is this passage the answer to this question". A summarization instruction is not
   a question, so every passage scores low however relevant it is -- "Summarise the
   leave policy in bullet points" tops out around -5.5 while "How many days of annual
   leave?" scores +7.0 against the very same document. An absolute threshold across
   both is meaningless, so the floor is only applied to factoid queries.

2. **How many chunks to keep.** `top_k_rerank` is sized for pinpoint lookup. Answering
   "summarise the leave policy" from 4 chunks of a 10-chunk policy produces a summary
   that silently omits most of it, so broad queries get a wider window.

Deliberately a small keyword heuristic rather than an LLM call: it runs in
microseconds on the hot path, is fully deterministic, and is trivial to test. It is
tuned to be conservative -- an unmatched query falls back to FACTOID, preserving the
existing behaviour.
"""
from __future__ import annotations

import enum
import re


class QueryIntent(enum.Enum):
    FACTOID = "factoid"
    BROAD = "broad"


# Explicit summarize/enumerate/overview markers only. A plain "What is the leave
# policy?" stays FACTOID: retrieval handles it well already, and widening the window
# for every vaguely open question would inflate cost on the common path.
_BROAD_PATTERNS = (
    r"\bsummar(?:y|ies|ise|ize|ised|ized|ising|izing|isation|ization)\b",
    r"\brecap\b",
    r"\boutlines?\b",
    r"\boverview\b",
    r"\bgist\b",
    r"\bhighlights\b",
    r"\bbullet[- ]?points?\b",
    r"\bkey (?:points|takeaways|highlights|details)\b",
    r"\bmain points\b",
    r"\blist (?:the|all|out|every|any)\b",
    r"\btell me (?:about|everything)\b",
    r"\bwalk me through\b",
    r"\bbrief me\b",
    r"\beverything (?:about|in|on)\b",
    r"\bwhat (?:does|do) the .{0,40}?(?:policy|handbook|document|faqs?)\b"
    r".{0,20}?\b(?:cover|say|include|contain)\b",
)
_BROAD_RE = re.compile("|".join(_BROAD_PATTERNS), re.IGNORECASE)


def classify(query: str) -> QueryIntent:
    """Best-effort intent for a (already condensed) standalone query."""
    return QueryIntent.BROAD if _BROAD_RE.search(query) else QueryIntent.FACTOID
