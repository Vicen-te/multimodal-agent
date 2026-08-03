"""Evaluation cases: (image, question, expected tool, rubric) triples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"
CASES_PATH = DATA_DIR / "cases.jsonl"
IMAGES_DIR = DATA_DIR / "images"


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    image: Optional[str]
    expected_tool: str  # AnalyzeImage | SearchDocs | both | none
    rubric: str
    control: bool = False  # designed-to-fail case, reported outside the headline metrics


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        image_name = raw.get("image")
        image_path = str(IMAGES_DIR / image_name) if image_name else None
        cases.append(
            EvalCase(
                id=raw["id"],
                question=raw["question"],
                image=image_path,
                expected_tool=raw["expected_tool"],
                rubric=raw["rubric"],
                control=bool(raw.get("control", False)),
            )
        )
    return cases
