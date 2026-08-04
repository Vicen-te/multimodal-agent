"""LLM-as-judge scoring of answer quality against a rubric."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage

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

_FORMAT_REMINDER = (
    'Reply again with only the JSON object {"score": <1-5>, "reason": "<one short '
    'sentence>"} and nothing else.'
)


def _parse_score(content: str) -> tuple[int, str] | None:
    """Pull the verdict out of the judge's reply, or None when nothing parses."""
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
    return None


def judge_answer(chat_model, question: str, answer: str, rubric: str) -> tuple[int, str]:
    """Return a 1-5 quality score and a short justification.

    Small local judge models sometimes wrap the JSON in markdown or drop it
    entirely. Parsing falls back to a regex over the score, and when that also
    fails the judge is asked once more with a format-only reminder before giving
    up, so a correct answer does not take a spurious 1 from a formatting slip.
    """
    prompt = JUDGE_PROMPT.format(question=question, rubric=rubric, answer=answer)
    messages = [HumanMessage(content=prompt)]
    content = content_to_text(chat_model.invoke(messages).content)
    parsed = _parse_score(content)
    if parsed is None:
        retry = messages + [AIMessage(content=content), HumanMessage(content=_FORMAT_REMINDER)]
        content = content_to_text(chat_model.invoke(retry).content)
        parsed = _parse_score(content)
    return parsed if parsed is not None else (1, "could not parse judge output")
