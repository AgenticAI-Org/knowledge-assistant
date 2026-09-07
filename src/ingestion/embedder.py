"""OpenAI embeddings client plus the shared retry predicate for API calls."""
from __future__ import annotations

import logging

import openai
from langchain_openai import OpenAIEmbeddings
from tenacity import retry_if_exception_type

from src.config import settings

logger = logging.getLogger(__name__)

# Retry only failures that a retry can actually fix. Retrying an auth error or a
# malformed request just burns ~20s of backoff before surfacing the same error.
RETRYABLE_ERRORS = retry_if_exception_type(
    (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    )
)


def get_embeddings_client() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
