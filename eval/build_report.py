"""Runs retrieval eval + generation eval + an operational summary from logs/, and writes
the combined eval/report.md submitted as evidence of the Test step (see docs/EVALUATION.md).

Run: uv run python -m eval.build_report
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from src.config import settings
from src.logging_config import configure_logging

REPORT_PATH = Path(__file__).resolve().parent / "report.md"


def operational_summary() -> str:
    log_path = settings.log_dir / "app.log"
    if not log_path.exists():
        return "## Operational Summary\n\nNo logs found yet -- run the app or eval scripts first.\n"

    durations_by_stage: dict[str, list[int]] = {}
    tokens_in = tokens_out = 0

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage = record.get("stage")
            duration_ms = record.get("duration_ms")
            if stage and isinstance(duration_ms, int):
                durations_by_stage.setdefault(stage, []).append(duration_ms)
            tokens_in += record.get("input_tokens", 0) or 0
            tokens_out += record.get("output_tokens", 0) or 0

    lines = ["## Operational Summary (from logs/app.log)", "", "| Stage | p50 (ms) | p95 (ms) | count |", "|---|---|---|---|"]
    for stage, durations in sorted(durations_by_stage.items()):
        durations_sorted = sorted(durations)
        p50 = durations_sorted[len(durations_sorted) // 2]
        p95 = durations_sorted[min(len(durations_sorted) - 1, int(len(durations_sorted) * 0.95))]
        lines.append(f"| {stage} | {p50} | {p95} | {len(durations)} |")
    lines.append("")
    lines.append(f"Total tokens observed in logs -- input: {tokens_in}, output: {tokens_out}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    configure_logging()

    from eval.run_generation_eval import run as run_generation_eval
    from eval.run_retrieval_eval import run as run_retrieval_eval

    sections = [
        f"# Evaluation Report\n\nGenerated: {dt.datetime.now().isoformat(timespec='seconds')}\n",
        run_retrieval_eval(),
        run_generation_eval(),
        operational_summary(),
    ]

    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
