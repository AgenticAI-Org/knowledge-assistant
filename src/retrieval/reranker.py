"""Local cross-encoder reranking of hybrid-fused candidates."""
from __future__ import annotations

import math
import threading

from sentence_transformers import CrossEncoder

from src.config import settings
from src.schemas import RetrievedChunk

_model: CrossEncoder | None = None
_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model
    with _lock:
        if _model is None:
            _model = CrossEncoder(settings.rerank_model_name)
        return _model


def rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int | None = None
) -> list[RetrievedChunk]:
    top_n = settings.top_k_rerank if top_n is None else top_n
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)

    # rerank_score is Optional on the model; every candidate was just scored, but
    # sort on an explicit float so an unscored one sinks instead of raising.
    ranked = sorted(
        candidates, key=lambda c: c.rerank_score if c.rerank_score is not None else -math.inf,
        reverse=True,
    )
    return ranked[:top_n]
