"""LLM-as-judge scoring of answer quality against a rubric."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

JUDGE_PROMPT = """You are grading an assistant's answer to a user question.

Question:
{question}

Rubric (what a good answer must contain):
{rubric}

Assistant answer:
{answer}

Grade the answer from 1 (wrong or empty) to 5 (fully correct and complete) based \
only on the rubric. Respond with a single JSON object and nothing else:
{{"score": <1-5>, "reason": "<one short sentence>"}}"""


def judge_answer(chat_model, question: str, answer: str, rubric: str) -> tuple[int, str]:
    """Return a 1-5 quality score and a short justification."""
    prompt = JUDGE_PROMPT.format(question=question, rubric=rubric, answer=answer)
    response = chat_model.invoke([HumanMessage(content=prompt)])
    content = response.content
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        return 1, "could not parse judge output"
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return 1, "could not parse judge output"
    score = int(data.get("score", 1))
    return max(1, min(5, score)), str(data.get("reason", ""))
