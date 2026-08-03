"""Run the agent over the eval set and report routing and quality metrics."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from .._content import content_to_text
from ..agent.graph import REVISION_PREFIX, deterministic_defects
from ..agent.runner import _initial_state, build_default_agent, encode_image
from ..config import Settings, get_settings
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


def expected_tools(expected: str) -> set[str]:
    """The exact tool set a case should have called."""
    if expected == "none":
        return set()
    if expected == "both":
        return set(TOOL_NAMES)
    return {expected}


def route_pass(expected: str, used: set[str]) -> bool:
    """Whether the expected tools were called, tolerating extra ones."""
    if expected == "none":
        return not used
    if expected == "both":
        return TOOL_NAMES.issubset(used)
    return expected in used


def revision_requests(messages: list) -> list[str]:
    """The critiques the reflection node sent back asking for a revision."""
    requests = []
    for message in messages:
        if isinstance(message, HumanMessage):
            text = content_to_text(message.content)
            if text.startswith(REVISION_PREFIX):
                requests.append(text[len(REVISION_PREFIX) :].strip())
    return requests


def first_draft(messages: list) -> str:
    """The answer the agent wrote before any reflection pass.

    Reflection only ever runs after the agent has produced an answer, so this
    first draft is what the run would have returned with the node disabled --
    the history already holds the counterfactual, no second run needed.
    """
    for message in messages:
        if isinstance(message, AIMessage):
            text = content_to_text(message.content)
            if text.strip():
                return text
    return ""


def render_table(rows: list[dict]) -> str:
    header = (
        "| id | expected | used | route | extra tools | citations | score "
        "| review passes | rewrites |\n"
    )
    header += (
        "|----|----------|------|:-----:|:-----------:|:---------:|:-----:"
        "|:-------------:|:--------:|\n"
    )
    lines = [
        (
            "| {id} | {expected} | {used} | {route} | {extra} | {citations} | {score} "
            "| {passes} | {rewrites} |"
        ).format(
            id=row["id"],
            expected=row["expected"],
            used=", ".join(sorted(row["used"])) or "-",
            route="pass" if row["route_ok"] else "FAIL",
            extra=", ".join(sorted(row["extra"])) or "-",
            citations=citation_state(row["citation_defects"]),
            score=row["score"],
            passes=row["reflections"],
            rewrites=len(row["revisions"]),
        )
        for row in rows
    ]
    return header + "\n".join(lines)


def render_details(rows: list[dict]) -> str:
    """Every answer in full, with its sources, critiques, and judge verdict."""
    blocks = []
    for row in rows:
        parts = [
            f"### `{row['id']}`" + (" -- negative control" if row.get("control") else "") + "\n",
            f"**Question:** {row['question']}",
            f"**Image:** {row['image'] or 'none'}",
            f"**Rubric:** {row['rubric']}\n",
            f"- tools used: `{', '.join(sorted(row['used'])) or 'none'}`"
            f" (expected `{row['expected']}`)",
            f"- sources retrieved: `{', '.join(sorted(set(row['retrieved']))) or 'none'}`",
            f"- score: **{row['score']}/5** -- {row['reason']}",
            f"- citations: {citation_state(row['citation_defects'])}",
            f"- review passes: {row['reflections']}",
        ]
        for defect in row["citation_defects"]:
            parts.append(f"- citation defect: _{defect}_")
        for critique in row["revisions"]:
            parts.append(f"- revision requested: _{critique}_")
        if row.get("revised"):
            parts.append(
                f"\n<details><summary>draft before reflection "
                f"(scored {row['draft_score']}/5 -- {row['draft_reason']})</summary>\n\n"
                f"```\n{row['draft'].strip()}\n```\n\n</details>"
            )
        parts.append(f"\n**Answer:**\n\n```\n{row['answer'].strip()}\n```")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def summarise(rows: list[dict]) -> str:
    total = len(rows)
    routing_acc = sum(r["route_ok"] for r in rows) / total
    exact_acc = sum(r["route_exact"] for r in rows) / total
    avg_score = sum(r["score"] for r in rows) / total
    clean = sum(1 for r in rows if not r["citation_defects"])
    reviewed = sum(1 for r in rows if r["reflections"] > 0) / total
    rewritten = sum(1 for r in rows if r["revisions"]) / total
    return (
        f"\n**Cases:** {total}  \n"
        f"**Routing accuracy:** {routing_acc:.0%}  \n"
        f"**Exact routing (no extra tools):** {exact_acc:.0%}  \n"
        f"**Average answer score:** {avg_score:.2f} / 5  \n"
        f"**Citation integrity:** {clean}/{total} answers clean  \n"
        f"**Answers reviewed:** {reviewed:.0%}  \n"
        f"**Answers rewritten:** {rewritten:.0%}\n"
    )


def citation_state(defects: list[str]) -> str:
    """Render a citation check as a short cell."""
    return "ok" if not defects else f"{len(defects)} bad"


def control_failed(row: dict) -> bool:
    """Whether a negative control failed, which is exactly what a control must do."""
    return row["score"] <= 2 or not row["route_ok"]


def summarise_controls(rows: list[dict]) -> str:
    failed = sum(1 for r in rows if control_failed(r))
    return f"\n**Controls failing as designed:** {failed}/{len(rows)}\n"


def render_comparison(rows: list[dict]) -> str:
    """Draft versus final per case, on both the rubric score and citation integrity."""
    header = (
        "| id | score before | score after | delta | citations before | citations after "
        "| answer rewritten |\n"
    )
    header += (
        "|----|:------------:|:-----------:|:-----:|:----------------:|:---------------:"
        "|:----------------:|\n"
    )
    lines = []
    for row in rows:
        delta = row["score"] - row["draft_score"]
        lines.append(
            f"| {row['id']} | {row['draft_score']} | {row['score']} | {delta:+d} "
            f"| {citation_state(row['draft_defects'])} "
            f"| {citation_state(row['citation_defects'])} "
            f"| {'yes' if row['revised'] else 'no'} |"
        )
    return header + "\n".join(lines)


def summarise_reflection(rows: list[dict]) -> str:
    total = len(rows)
    rewritten = [r for r in rows if r["revised"]]
    before = sum(r["draft_score"] for r in rows) / total
    after = sum(r["score"] for r in rows) / total
    improved = sum(1 for r in rewritten if r["score"] > r["draft_score"])
    regressed = sum(1 for r in rewritten if r["score"] < r["draft_score"])
    clean_before = sum(1 for r in rows if not r["draft_defects"])
    clean_after = sum(1 for r in rows if not r["citation_defects"])
    return (
        f"\n**Answers rewritten:** {len(rewritten)} / {total}  \n"
        f"**Average score before reflection:** {before:.2f} / 5  \n"
        f"**Average score after reflection:** {after:.2f} / 5  \n"
        f"**Rewrites that improved the score:** {improved}  \n"
        f"**Rewrites that lowered the score:** {regressed}  \n"
        f"**Citation integrity before reflection:** {clean_before}/{total} clean  \n"
        f"**Citation integrity after reflection:** {clean_after}/{total} clean\n"
    )


def evaluate_case(graph, judge_model, case: EvalCase, *, judge_draft: bool = False) -> dict:
    state = _initial_state(case.question, encode_image(case.image), None)
    result = graph.invoke(state)
    messages = result["messages"]
    used = collect_tool_calls(messages)
    answer = content_to_text(messages[-1].content)
    draft = first_draft(messages)
    score, reason = judge_answer(judge_model, case.question, answer, case.rubric)

    revised = draft.strip() != answer.strip()
    # Scoring the draft only pays for a judge call when reflection actually
    # rewrote the answer; otherwise the draft is the answer.
    if judge_draft and revised:
        draft_score, draft_reason = judge_answer(judge_model, case.question, draft, case.rubric)
    else:
        draft_score, draft_reason = score, reason

    # Citation integrity is what the reflection node actually enforces, and the
    # rubrics do not grade it, so measure it directly on both sides of a rewrite.
    retrieved = result.get("retrieved_ids", [])
    wanted = expected_tools(case.expected_tool)

    return {
        "id": case.id,
        "question": case.question,
        "image": Path(case.image).name if case.image else None,
        "rubric": case.rubric,
        "expected": case.expected_tool,
        "used": used,
        "extra": used - wanted,
        "route_ok": route_pass(case.expected_tool, used),
        "route_exact": used == wanted,
        "score": score,
        "reason": reason,
        "answer": answer,
        "draft": draft,
        "revised": revised,
        "draft_score": draft_score,
        "draft_reason": draft_reason,
        "citation_defects": deterministic_defects(answer, retrieved),
        "draft_defects": deterministic_defects(draft, retrieved),
        "retrieved": retrieved,
        "revisions": revision_requests(messages),
        "reflections": result.get("reflections", 0),
        "control": case.control,
    }


def run_suite(settings: Settings, cases: list[EvalCase], *, judge_draft: bool = False) -> list[dict]:
    graph = build_default_agent(settings)
    judge_model = build_chat_model(settings)
    rows = []
    for case in cases:
        print(f"running {case.id} ...")
        rows.append(evaluate_case(graph, judge_model, case, judge_draft=judge_draft))
    return rows


def build_report(
    settings: Settings, cases: list[EvalCase], detailed: bool, compare: bool
) -> str:
    # Comparing needs the node on: the run then carries both the draft it wrote
    # first and the answer reflection settled on.
    if compare:
        settings = replace(settings, enable_reflection=True)

    rows = run_suite(settings, cases, judge_draft=compare)
    graded = [row for row in rows if not row["control"]]
    controls = [row for row in rows if row["control"]]

    sections = ["# Eval results\n", render_table(graded), summarise(graded)]
    if controls:
        sections.append("\n## Negative controls\n")
        sections.append(
            "These cases are impossible by design -- a withheld image, a premise "
            "the corpus cannot support -- so failing them is the correct outcome; "
            "a control that passes flags a rubric or a judge gone soft.\n"
        )
        sections.append(render_table(controls))
        sections.append(summarise_controls(controls))
    if compare:
        sections.append("\n## Before and after reflection\n")
        sections.append(render_comparison(graded))
        sections.append(summarise_reflection(graded))
    if detailed:
        sections.append("\n## Case details\n")
        sections.append(render_details(rows))
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="include every answer in full with its sources, critiques, and judge verdict",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also score the draft written before reflection, to show what the node changed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help=f"where to write the report (default: {RESULTS_PATH.name})",
    )
    args = parser.parse_args()

    report = build_report(get_settings(), load_cases(), args.detailed, args.compare)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
