"""Unit tests for the hallucination guardrail.

These cover the two failure modes the guardrail previously had: a paraphrased
refusal being recorded as a grounded answer, and an unscored candidate silently
disabling the pre-generation check.
"""
from __future__ import annotations

import pytest

from src.config import settings
from src.guardrails.hallucination import (
    NOT_FOUND_MESSAGE,
    build_citations,
    build_fallback_answer,
    has_sufficient_context,
    is_self_reported_refusal,
)
from src.schemas import ChunkMetadata, RetrievedChunk


def scored(score, *, chunk_id="a", source="s.pdf"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="text",
        metadata=ChunkMetadata(source=source, chunk_id=chunk_id, chunk_index=0),
        rerank_score=score,
    )


class TestHasSufficientContext:
    def test_empty_is_insufficient(self):
        assert has_sufficient_context([]) is False

    def test_score_above_floor_is_sufficient(self):
        assert has_sufficient_context([scored(settings.rerank_score_floor + 1)]) is True

    def test_score_below_floor_is_insufficient(self):
        assert has_sufficient_context([scored(settings.rerank_score_floor - 1)]) is False

    def test_score_exactly_at_floor_is_sufficient(self):
        assert has_sufficient_context([scored(settings.rerank_score_floor)]) is True

    def test_unscored_candidate_fails_closed(self):
        # An unscored top candidate means reranking did not run. That is when the
        # guardrail should engage, not stand down.
        assert has_sufficient_context([scored(None)]) is False

    def test_only_the_top_candidate_decides(self):
        assert has_sufficient_context([scored(-100.0), scored(100.0)]) is False


class TestIsSelfReportedRefusal:
    def test_exact_fallback_phrase(self):
        assert is_self_reported_refusal(NOT_FOUND_MESSAGE) is True

    def test_empty_answer(self):
        assert is_self_reported_refusal("   ") is True

    @pytest.mark.parametrize(
        "text",
        [
            "I could not find this information in the available documents.",
            "Unfortunately, I could not find this information in the available documents.",
            "I'm not able to find that in the provided policies.",
            "I couldn't find this information in the documents provided.",
            "The available documents do not contain information about that.",
            "That is not covered in the available documents.",
            "No relevant information is available on this topic.",
        ],
    )
    def test_paraphrased_refusals_are_detected(self, text):
        assert is_self_reported_refusal(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "You are entitled to 24 days of paid annual leave per calendar year.",
            "Reset your VPN password through the IT Service Portal under 'Password Reset'.",
            "You can carry forward a maximum of 8 unused annual leave days.",
            "Employees receive 12 days of paid sick leave per year.",
            "The per diem is INR 2,500 per day in metro cities.",
        ],
    )
    def test_real_answers_are_not_flagged(self, text):
        assert is_self_reported_refusal(text) is False

    def test_case_and_whitespace_insensitive(self):
        assert is_self_reported_refusal("I  COULD NOT FIND THIS\n INFORMATION anywhere.") is True


class TestCitations:
    def test_chunks_are_grouped_by_source(self):
        citations = build_citations(
            [
                scored(1.0, chunk_id="a::0", source="Leave_Policy.pdf"),
                scored(1.0, chunk_id="a::1", source="Leave_Policy.pdf"),
                scored(1.0, chunk_id="b::0", source="IT_Policy.docx"),
            ]
        )
        by_source = {c.source: c.chunk_ids for c in citations}
        assert by_source == {
            "Leave_Policy.pdf": ["a::0", "a::1"],
            "IT_Policy.docx": ["b::0"],
        }

    def test_fallback_answer_is_ungrounded_and_uncited(self):
        fallback = build_fallback_answer(latency_ms=12)
        assert fallback.grounded is False
        assert fallback.citations == []
        assert fallback.latency_ms == 12
