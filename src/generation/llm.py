"""OpenAI chat client wrapper with retry and token usage accounting."""
from __future__ import annotations

import threading

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.ingestion.embedder import RETRYABLE_ERRORS

_client: ChatOpenAI | None = None
_lock = threading.Lock()


def get_chat_client() -> ChatOpenAI:
    global _client
    with _lock:
        if _client is None:
            _client = ChatOpenAI(
                model=settings.llm_model, api_key=settings.openai_api_key, temperature=0.2
            )
        return _client


def _as_text(content: object) -> str:
    """Normalizes a message content payload to plain text.

    `BaseMessage.content` is `str | list[str | dict]`; the list form appears with
    multi-part/multimodal responses. Everything downstream (Answer.text, the
    refusal check) assumes a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=RETRYABLE_ERRORS,
)
def invoke_llm(client: ChatOpenAI, messages: list[BaseMessage]) -> tuple[str, dict[str, int]]:
    """Calls the chat model and returns (text, token_usage)."""
    response = client.invoke(messages)
    usage_meta = response.usage_metadata or {}
    usage = {
        "input_tokens": usage_meta.get("input_tokens", 0),
        "output_tokens": usage_meta.get("output_tokens", 0),
        "total_tokens": usage_meta.get("total_tokens", 0),
    }
    return _as_text(response.content), usage
