"""CLI entrypoint to (re)build the vector + BM25 indexes from data/.

Usage: uv run python -m scripts.build_index
"""
from __future__ import annotations

import sys

from src.ingestion.indexer import build_index
from src.logging_config import configure_logging


def main() -> None:
    configure_logging()
    try:
        stats = build_index()
    except Exception as exc:  # noqa: BLE001 -- top-level CLI error boundary
        print(f"Index build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Indexed {stats.num_chunks} chunks from {stats.num_source_documents} documents:")
    for source in stats.sources:
        print(f" - {source}")


if __name__ == "__main__":
    main()
