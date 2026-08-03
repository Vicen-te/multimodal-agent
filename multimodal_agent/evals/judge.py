"""LLM-as-judge scoring of answer quality against a rubric."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage

from .._content import content_to_text

JUDGE_PROMPT = """You are grading an assistant's answer to a user question.

Question:
{question}

Rubric (what a good answer must contain):
{rubric}

Assistant answer:
{answer}

Grade the answer from 1 (wrong or empty) to 5 (fully correct and complete) based \
only on the rubric. Respond with a single raw JSON object and nothing else -- no \
markdown, no code fences, no extra text:
{{"score": <1-5>, "reason": "<one short sentence>"}}"""

_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*"?([1-5])')


def judge_answer(chat_model, question: str, answer: str, rubric: str) -> tuple[int, str]:
    """Return a 1-5 quality score and a short justification.

    Small local judge models often wrap the JSON in markdown or add prose, so a
    strict ``json.loads`` can fail; fall back to pulling the score out with a
    regex before giving up, otherwise a correct answer gets a spurious 1.
    """
    prompt = JUDGE_PROMPT.format(question=question, rubric=rubric, answer=answer)
    content = content_to_text(chat_model.invoke([HumanMessage(content=prompt)]).content)

    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(content[start : end + 1])
            return max(1, min(5, int(data["score"]))), str(data.get("reason", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    match = _SCORE_RE.search(content)
    if match:
        return int(match.group(1)), "score parsed from non-JSON judge output"
    return 1, "could not parse judge output"
