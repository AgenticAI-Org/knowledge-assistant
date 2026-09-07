"""Structured JSON logging setup + per-query context helpers.

Every pipeline stage logs through a `StageLogger` (a `logging.LoggerAdapter`)
carrying `session_id` and `query_id`, so a single question's trace can be
reconstructed by filtering the log file on `query_id`. See docs/LOGGING.md
for the field schema.
"""
from __future__ import annotations

import contextlib
import json
import logging
import logging.handlers
import time
from typing import Any, Iterator

from src.config import settings

_CONFIGURED = False

# Fields present on every LogRecord by default -- anything else on the record
# is "extra" context we want to surface in the JSON output.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key != "message":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(settings.log_level)

    formatter = JsonFormatter() if settings.log_json else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    # Tagged so a re-run replaces only our own handlers rather than clearing the
    # root logger wholesale, which would also discard Streamlit's logging setup.
    for existing in list(root.handlers):
        if getattr(existing, "_knowledge_assistant", False):
            root.removeHandler(existing)

    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    console_handler.setLevel("INFO")

    file_handler._knowledge_assistant = True  # type: ignore[attr-defined]
    console_handler._knowledge_assistant = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Keep noisy third-party libraries at WARNING so app-level signal isn't buried.
    for noisy in ("httpx", "openai", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


class StageLogger(logging.LoggerAdapter):
    """LoggerAdapter that stamps every record with session_id/query_id."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        extra.update(self.extra)
        return msg, kwargs


def get_stage_logger(name: str, session_id: str, query_id: str) -> StageLogger:
    configure_logging()
    base = logging.getLogger(name)
    return StageLogger(base, {"session_id": session_id, "query_id": query_id})


@contextlib.contextmanager
def stage_timer(logger: StageLogger, stage: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Times a pipeline stage and logs one INFO record on exit with duration_ms.

    Usage:
        with stage_timer(logger, "retrieval", top_k=10) as result:
            hits = do_retrieval()
            result["retrieved_count"] = len(hits)
    """
    start = time.perf_counter()
    result: dict[str, Any] = {}
    try:
        yield result
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.error("stage_failed", extra={"stage": stage, "duration_ms": duration_ms, **fields})
        raise
    else:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info("stage_complete", extra={"stage": stage, "duration_ms": duration_ms, **fields, **result})
