"""Run the agent over the eval set and report routing and quality metrics."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from ..agent.runner import _initial_state, build_default_agent, encode_image
from ..config import get_settings
from ..providers.chat import build_chat_model
from .dataset import EvalCase, load_cases
from .judge import judge_answer

TOOL_NAMES = {"AnalyzeImage", "SearchDocs"}
RESULTS_PATH = Path(__file__).resolve().parents[2] / "eval_results.md"


def collect_tool_calls(messages: list) -> set[str]:
    used: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                used.add(call["name"])
    return used & TOOL_NAMES


def route_pass(expected: str, used: set[str]) -> bool:
    if expected == "none":
        return not used
    if expected == "both":
        return TOOL_NAMES.issubset(used)
    return expected in used


def render_table(rows: list[dict]) -> str:
    header = "| id | expected | used | route | score | reflections |\n"
    header += "|----|----------|------|:-----:|:-----:|:-----------:|\n"
    lines = [
        "| {id} | {expected} | {used} | {route} | {score} | {reflections} |".format(
            id=row["id"],
            expected=row["expected"],
            used=", ".join(sorted(row["used"])) or "-",
            route="pass" if row["route_ok"] else "FAIL",
            score=row["score"],
            reflections=row["reflections"],
        )
        for row in rows
    ]
    return header + "\n".join(lines)


def summarise(rows: list[dict]) -> str:
    total = len(rows)
    routing_acc = sum(r["route_ok"] for r in rows) / total
    avg_score = sum(r["score"] for r in rows) / total
    reflection_rate = sum(1 for r in rows if r["reflections"] > 0) / total
    return (
        f"\n**Cases:** {total}  \n"
        f"**Routing accuracy:** {routing_acc:.0%}  \n"
        f"**Average answer score:** {avg_score:.2f} / 5  \n"
        f"**Reflection rate:** {reflection_rate:.0%}\n"
    )


def evaluate_case(graph, judge_model, case: EvalCase) -> dict:
    state = _initial_state(case.question, encode_image(case.image), None)
    result = graph.invoke(state)
    messages = result["messages"]
    used = collect_tool_calls(messages)
    answer = messages[-1].content
    score, _ = judge_answer(judge_model, case.question, answer, case.rubric)
    return {
        "id": case.id,
        "expected": case.expected_tool,
        "used": used,
        "route_ok": route_pass(case.expected_tool, used),
        "score": score,
        "reflections": result.get("reflections", 0),
    }


def main() -> None:
    settings = get_settings()
    graph = build_default_agent(settings)
    judge_model = build_chat_model(settings)

    rows = []
    for case in load_cases():
        print(f"running {case.id} ...")
        rows.append(evaluate_case(graph, judge_model, case))

    report = "# Eval results\n\n" + render_table(rows) + "\n" + summarise(rows)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
