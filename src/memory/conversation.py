"""Conversation memory: turn storage, standalone-query condensation, and history formatting.

Storage itself lives in Streamlit `session_state` (see app/streamlit_app.py) as a plain
list of ChatTurn -- this module only knows how to *use* that history, keeping the two
responsibilities (condensing a follow-up into a standalone query, vs. formatting recent
turns for the final answer prompt) separate per docs/LLD.md.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.config import settings
from src.generation.llm import invoke_llm
from src.generation.prompts import CONDENSE_SYSTEM_PROMPT, CONDENSE_USER_TEMPLATE


class ChatTurn(BaseModel):
    question: str
    answer: str


def format_history(history: list[ChatTurn], max_turns: int = settings.memory_window_turns) -> str:
    if not history:
        return "(no previous conversation)"
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer}")
    return "\n".join(lines)


def condense_query(client, history: list[ChatTurn], question: str) -> str:
    """Rewrites a follow-up question into a standalone query using recent history.

    Skips the extra LLM call entirely when there's no history yet -- the first
    question in a session is already standalone.
    """
    if not history:
        return question

    messages = [
        SystemMessage(content=CONDENSE_SYSTEM_PROMPT),
        HumanMessage(
            content=CONDENSE_USER_TEMPLATE.format(
                history=format_history(history), question=question
            )
        ),
    ]
    condensed, _usage = invoke_llm(client, messages)
    return condensed.strip() or question
