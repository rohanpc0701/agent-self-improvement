"""Rubric LLM judge for FinancePro-Bench (stub interface until Task 2 fills body).

Task 1 only needs the import surface for the finance adapter. Full grading,
parser, and JUDGE_MODEL != TEACHER_MODEL assert land in Task 2.
"""
from __future__ import annotations

from typing import Any


def grade(
    question: str,
    rubric: str,
    answer: str,
    model: str | None = None,
    passes: int = 1,
) -> dict[str, Any]:
    """Grade an answer against a rubric. Implemented in Task 2 (G0.2)."""
    raise NotImplementedError(
        "correction.judge.grade is implemented in Task 2 (G0.2). "
        "Mock this in hermetic tests."
    )
