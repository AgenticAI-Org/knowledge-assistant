"""End-to-end generation eval: faithfulness/relevancy (LLM-as-judge), citation coverage,
hallucination refusal accuracy, and multi-turn memory accuracy.

A hand-rolled LLM-judge is used instead of a third-party eval framework (e.g. RAGAS) to
avoid an extra heavy/fragile dependency for a scope this small -- see docs/EVALUATION.md
for the metric definitions and the rationale.

Run: uv run python -m eval.run_generation_eval
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from eval.common import multi_turn_questions, single_turn_questions
from src.generation.answer_service import answer as run_answer
from src.generation.llm import get_chat_client
from src.memory.conversation import ChatTurn

JUDGE_SYSTEM_PROMPT = """You are an evaluator scoring a RAG assistant's answer against the \
retrieved context it was given. Score two things on a 0.0-1.0 scale:
- faithfulness: is every claim in the answer actually supported by the context? (1.0 = fully \
supported, 0.0 = fabricated / not supported)
- answer_relevancy: does the answer actually address the user's question? (1.0 = fully \
relevant and on-topic, 0.0 = off-topic or non-answer)"""

JUDGE_USER_TEMPLATE = """Question: {question}

Context given to the assistant:
{context}

Assistant's answer: {answer}

Score faithfulness and answer_relevancy."""


class JudgeScore(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)


def judge_answer(client, question: str, answer_text: str, context_texts: list[str]) -> JudgeScore:
    structured_client = client.with_structured_output(JudgeScore)
    context = "\n\n".join(context_texts) if context_texts else "(no context)"
    prompt = JUDGE_USER_TEMPLATE.format(question=question, context=context, answer=answer_text)
    return structured_client.invoke([{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}])


def _citation_sources(answer_obj) -> set[str]:
    return {c.source for c in answer_obj.citations}


def run() -> str:
    client = get_chat_client()
    lines = ["## Generation Evaluation", ""]

    # --- Positive + negative single-turn cases ---
    positive = [q for q in single_turn_questions() if q.get("expects_answer")]
    negative = [q for q in single_turn_questions() if not q.get("expects_answer")]

    faithfulness_scores, relevancy_scores, citation_hits = [], [], []
    for q in positive:
        result = run_answer(question=q["question"], history=[], session_id="eval-generation")
        score = judge_answer(
            client, q["question"], result.text, [c.text for c in result.retrieved_chunks]
        )
        faithfulness_scores.append(score.faithfulness)
        relevancy_scores.append(score.answer_relevancy)
        citation_hits.append(bool(set(q["expected_sources"]) & _citation_sources(result)))

    refusal_correct = 0
    for q in negative:
        result = run_answer(question=q["question"], history=[], session_id="eval-generation")
        if not result.grounded:
            refusal_correct += 1

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else float("nan")
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else float("nan")
    citation_coverage = sum(citation_hits) / len(citation_hits) if citation_hits else float("nan")
    refusal_accuracy = refusal_correct / len(negative) if negative else float("nan")

    lines += [
        f"Evaluated {len(positive)} answerable questions and {len(negative)} unanswerable questions.",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Faithfulness (avg, LLM-judge) | {avg_faithfulness:.2f} |",
        f"| Answer Relevancy (avg, LLM-judge) | {avg_relevancy:.2f} |",
        f"| Citation Coverage (expected source cited) | {citation_coverage:.2f} |",
        f"| Refusal Accuracy (correctly said 'not found') | {refusal_accuracy:.2f} |",
        "",
    ]

    # --- Multi-turn memory cases ---
    memory_hits = 0
    memory_total = 0
    for q in multi_turn_questions():
        history: list[ChatTurn] = []
        last_result = None
        for turn in q["turns"]:
            last_result = run_answer(question=turn["question"], history=history, session_id="eval-generation-mt")
            history.append(ChatTurn(question=turn["question"], answer=last_result.text))

        final_turn = q["turns"][-1]
        memory_total += 1
        if last_result and set(final_turn["expected_sources"]) & _citation_sources(last_result):
            memory_hits += 1

    memory_accuracy = memory_hits / memory_total if memory_total else float("nan")
    lines += [
        f"Evaluated {memory_total} multi-turn conversations "
        "(does the follow-up turn correctly resolve via memory and cite the right source?).",
        "",
        f"**Multi-turn Memory Accuracy: {memory_accuracy:.2f}**",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
