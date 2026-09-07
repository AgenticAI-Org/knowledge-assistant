"""Single orchestration entrypoint the UI calls: condense -> retrieve -> rerank -> generate."""
from __future__ import annotations

import time
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.generation.llm import get_chat_client, invoke_llm
from src.generation.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE, format_context
from src.guardrails.hallucination import (
    build_citations,
    build_fallback_answer,
    has_sufficient_context,
    is_self_reported_refusal,
)
from src.logging_config import get_stage_logger, stage_timer
from src.memory.conversation import ChatTurn, condense_query, format_history
from src.retrieval.bm25_retriever import bm25_search
from src.retrieval.hybrid import fuse
from src.retrieval.query_intent import QueryIntent, classify
from src.retrieval.reranker import rerank
from src.retrieval.vector_retriever import vector_search
from src.schemas import Answer


def answer(question: str, history: list[ChatTurn], session_id: str) -> Answer:
    query_id = str(uuid.uuid4())
    logger = get_stage_logger(__name__, session_id=session_id, query_id=query_id)
    start = time.perf_counter()

    client = get_chat_client()

    with stage_timer(logger, "condense_query") as result:
        standalone_question = condense_query(client, history, question)
        result["condensed"] = standalone_question != question

    with stage_timer(logger, "retrieval") as result:
        vector_hits = vector_search(standalone_question, k=settings.top_k_vector)
        bm25_hits = bm25_search(standalone_question, k=settings.top_k_bm25)
        fused = fuse(vector_hits, bm25_hits)
        result["vector_hits"] = len(vector_hits)
        result["bm25_hits"] = len(bm25_hits)
        result["fused_candidates"] = len(fused)

    # A broad "summarise/list/overview" request needs a wider context window than a
    # pinpoint lookup, and the rerank score floor does not apply to it at all.
    intent = classify(standalone_question)
    is_broad = intent is QueryIntent.BROAD
    top_n = settings.top_k_rerank_broad if is_broad else settings.top_k_rerank

    with stage_timer(logger, "rerank", intent=intent.value) as result:
        top_chunks = rerank(standalone_question, fused, top_n=top_n)
        result["reranked_count"] = len(top_chunks)
        result["top_n"] = top_n
        result["top_score"] = top_chunks[0].rerank_score if top_chunks else None

    if not has_sufficient_context(top_chunks, apply_score_floor=not is_broad):
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.warning(
            "guardrail_triggered_pre_generation",
            extra={
                "stage": "guardrail",
                "reason": "no_context" if is_broad else "low_context_relevance",
                "intent": intent.value,
                "latency_ms": latency_ms,
            },
        )
        fallback = build_fallback_answer(latency_ms=latency_ms)
        fallback.retrieved_chunks = top_chunks
        return fallback

    with stage_timer(logger, "generation") as result:
        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(
                content=ANSWER_USER_TEMPLATE.format(
                    context=format_context([c.text for c in top_chunks]),
                    history=format_history(history),
                    question=standalone_question,
                )
            ),
        ]
        answer_text, token_usage = invoke_llm(client, messages)
        result["output_tokens"] = token_usage.get("output_tokens", 0)
        result["input_tokens"] = token_usage.get("input_tokens", 0)

    grounded = not is_self_reported_refusal(answer_text)
    citations = build_citations(top_chunks) if grounded else []

    if not grounded:
        logger.warning("model_self_reported_refusal", extra={"stage": "guardrail"})

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "answer_complete",
        extra={
            "stage": "answer_service",
            "intent": intent.value,
            "latency_ms": latency_ms,
            "grounded": grounded,
            "num_citations": len(citations),
        },
    )

    return Answer(
        text=answer_text,
        citations=citations,
        grounded=grounded,
        latency_ms=latency_ms,
        token_usage=token_usage,
        retrieved_chunks=top_chunks,
    )
