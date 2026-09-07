"""Unit tests for the pure retrieval logic: RRF fusion, ranking, and eval metrics."""
from __future__ import annotations

from eval.common import first_correct_rank, hit_rate_and_mrr
from src.retrieval.hybrid import fuse
from src.schemas import ChunkMetadata, RetrievedChunk


def make_chunk(chunk_id: str, *, source: str = "s.pdf", vector_rank=None, bm25_rank=None):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=chunk_id,
        metadata=ChunkMetadata(source=source, chunk_id=chunk_id, chunk_index=0),
        vector_rank=vector_rank,
        bm25_rank=bm25_rank,
    )


class TestFuse:
    def test_chunk_in_both_lists_outranks_chunks_in_only_one(self):
        vector = [make_chunk("a", vector_rank=0), make_chunk("b", vector_rank=1)]
        bm25 = [make_chunk("b", bm25_rank=0), make_chunk("c", bm25_rank=1)]

        fused = fuse(vector, bm25)

        # "b" is ranked lower than "a" by vector search alone, but appearing in
        # both lists is what RRF is for.
        assert [c.chunk_id for c in fused][0] == "b"

    def test_ranks_from_both_backends_are_preserved(self):
        fused = {c.chunk_id: c for c in fuse([make_chunk("b", vector_rank=1)], [make_chunk("b", bm25_rank=0)])}
        assert fused["b"].vector_rank == 1
        assert fused["b"].bm25_rank == 0

    def test_bm25_only_hit_survives_fusion(self):
        fused = fuse([make_chunk("a", vector_rank=0)], [make_chunk("c", bm25_rank=0)])
        assert {c.chunk_id for c in fused} == {"a", "c"}

    def test_scores_are_descending(self):
        fused = fuse(
            [make_chunk("a", vector_rank=0), make_chunk("b", vector_rank=1)],
            [make_chunk("b", bm25_rank=0)],
        )
        scores = [c.fused_score for c in fused]
        assert scores == sorted(scores, reverse=True)

    def test_empty_inputs(self):
        assert fuse([], []) == []


class TestEvalMetrics:
    def test_first_correct_rank_is_one_indexed(self):
        assert first_correct_rank(["a.pdf", "b.pdf"], ["b.pdf"]) == 2

    def test_first_correct_rank_returns_none_when_absent(self):
        assert first_correct_rank(["a.pdf"], ["b.pdf"]) is None

    def test_hit_rate_and_mrr(self):
        hit_rate, mrr = hit_rate_and_mrr([1, 2, None, 4])
        assert hit_rate == 0.75
        assert mrr == (1 / 1 + 1 / 2 + 1 / 4) / 4

    def test_empty_is_not_a_division_by_zero(self):
        assert hit_rate_and_mrr([]) == (0.0, 0.0)


class TestGoldChunkResolution:
    """Passage-level ground truth is resolved from keywords against the live corpus."""

    class FakeChunk:
        def __init__(self, chunk_id, text):
            self.text = text
            self.metadata = ChunkMetadata(source="s.pdf", chunk_id=chunk_id, chunk_index=0)

    def corpus(self):
        return [
            self.FakeChunk("a::0", "Entitled to 24 days of paid annual leave."),
            self.FakeChunk("a::1", "Carry forward a maximum of 8 unused days."),
            self.FakeChunk("a::2", "Unrelated text about parking."),
        ]

    def test_single_keyword_resolves_one_chunk(self):
        from eval.common import gold_chunk_ids

        assert gold_chunk_ids({"answer_keywords": ["24 days"]}, self.corpus()) == ["a::0"]

    def test_all_keywords_must_match_the_same_chunk(self):
        from eval.common import gold_chunk_ids

        # "24 days" and "8 unused" each match, but never the same chunk.
        assert gold_chunk_ids({"answer_keywords": ["24 days", "8 unused"]}, self.corpus()) == []

    def test_matching_is_case_insensitive(self):
        from eval.common import gold_chunk_ids

        assert gold_chunk_ids({"answer_keywords": ["ANNUAL LEAVE"]}, self.corpus()) == ["a::0"]

    def test_question_without_keywords_has_no_ground_truth(self):
        from eval.common import gold_chunk_ids

        assert gold_chunk_ids({}, self.corpus()) == []


class TestEvalQuestionSet:
    def test_every_answerable_question_has_gold_keywords(self):
        from eval.common import single_turn_questions

        answerable = [q for q in single_turn_questions() if q.get("expects_answer")]
        missing = [q["id"] for q in answerable if not q.get("answer_keywords")]
        assert missing == [], f"answerable questions without passage ground truth: {missing}"

    def test_unanswerable_questions_have_no_expected_sources(self):
        from eval.common import single_turn_questions

        for q in single_turn_questions():
            if not q.get("expects_answer"):
                assert q["expected_sources"] == []
