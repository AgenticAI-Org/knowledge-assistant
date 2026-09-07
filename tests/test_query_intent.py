"""Tests for query intent classification and the guardrail behaviour it drives."""
from __future__ import annotations

import pytest

from src.config import settings
from src.guardrails.hallucination import has_sufficient_context
from src.retrieval.query_intent import QueryIntent, classify
from src.schemas import ChunkMetadata, RetrievedChunk


def scored(score):
    return RetrievedChunk(
        chunk_id="a",
        text="text",
        metadata=ChunkMetadata(source="s.pdf", chunk_id="a", chunk_index=0),
        rerank_score=score,
    )


class TestClassify:
    @pytest.mark.parametrize(
        "query",
        [
            "summaries leave policy in bullet points",  # the reported typo'd query
            "Summarise the leave policy in bullet points",
            "Summarize the travel policy",
            "Give me a summary of the IT policy",
            "Give me an overview of the leave policy",
            "List the key points of the travel policy",
            "List all the benefits available to me",
            "What are the main points of the remote work policy?",
            "Tell me about the travel policy",
            "Walk me through the expense process",
            "What does the leave policy cover?",
            "Outline the code of conduct",
            "bullet points on sick leave",
        ],
    )
    def test_broad_queries(self, query):
        assert classify(query) is QueryIntent.BROAD

    @pytest.mark.parametrize(
        "query",
        [
            "How many days of paid annual leave am I entitled to?",
            "How many leave days can I carry forward?",
            "How do I reset my VPN password?",
            "What is the annual leave policy?",
            "What about carry-forward?",
            "Who should I contact for an urgent IT system outage?",
            "What is the per diem rate in metro cities?",
        ],
    )
    def test_factoid_queries(self, query):
        assert classify(query) is QueryIntent.FACTOID

    def test_unmatched_query_defaults_to_factoid(self):
        # Conservative fallback: unknown phrasing keeps the pre-existing behaviour.
        assert classify("qwerty asdf") is QueryIntent.FACTOID

    def test_classification_is_case_insensitive(self):
        assert classify("SUMMARISE THE LEAVE POLICY") is QueryIntent.BROAD


class TestFloorIsIntentAware:
    def test_broad_query_ignores_a_low_score(self):
        # The exact failure reported: a valid summarization request scored -7.07 and
        # was refused before the LLM was ever called.
        low = scored(-7.07)
        assert has_sufficient_context([low], apply_score_floor=True) is False
        assert has_sufficient_context([low], apply_score_floor=True) != has_sufficient_context(
            [low], apply_score_floor=False
        )
        assert has_sufficient_context([low], apply_score_floor=False) is True

    def test_broad_query_still_requires_some_context(self):
        assert has_sufficient_context([], apply_score_floor=False) is False

    def test_factoid_query_keeps_the_floor(self):
        assert has_sufficient_context([scored(settings.rerank_score_floor - 1)]) is False
        assert has_sufficient_context([scored(settings.rerank_score_floor + 1)]) is True

    def test_floor_is_applied_by_default(self):
        assert has_sufficient_context([scored(-100.0)]) is False


class TestBroadWindowIsWider:
    def test_broad_window_exceeds_factoid_window(self):
        assert settings.top_k_rerank_broad > settings.top_k_rerank

    def test_broad_window_fits_the_candidate_pool(self):
        # Candidates come from the fused vector+BM25 lists; a window larger than the
        # pool would silently just return the pool.
        assert settings.top_k_rerank_broad <= settings.top_k_vector + settings.top_k_bm25
