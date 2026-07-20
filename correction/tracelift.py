"""Executor-grounded uplift gating for few-shot memory writes.

Inspired by TraceLift's uplift idea (arXiv:2605.03862): credit an artifact only
when it improves a frozen student/executor — not merely when it is correct.

This module selects FewShotExample items by measured student uplift on a
LEARN-side validation slice (never held-out recovery).
"""
from __future__ import annotations

import os
from typing import Protocol

from contracts.schemas import AgentConfig, FewShotExample
from harness.feed import FeedItem


class _RunAdapter(Protocol):
    def run_item(
        self, item: FeedItem, config: AgentConfig, use_rules: bool = True
    ): ...


def estimate_uplift(
    adapter: _RunAdapter,
    student_config: AgentConfig,
    candidate: FewShotExample,
    val_items: list[FeedItem],
    k: int = 3,
) -> float:
    """u = mean(pass | candidate) - mean(pass | empty) over val_items × k runs."""
    if not val_items or k < 1:
        return 0.0

    bare = student_config.model_copy(
        update={"few_shot_examples": [], "config_id": "uplift-bare"}
    )
    with_ex = student_config.model_copy(
        update={
            "few_shot_examples": [candidate],
            "config_id": "uplift-with",
        }
    )

    bare_scores: list[float] = []
    with_scores: list[float] = []
    for item in val_items:
        for _ in range(k):
            rec_b = adapter.run_item(item, bare, use_rules=False)
            rec_w = adapter.run_item(item, with_ex, use_rules=False)
            if rec_b is not None:
                bare_scores.append(rec_b.execution_accuracy)
            if rec_w is not None:
                with_scores.append(rec_w.execution_accuracy)

    if not bare_scores or not with_scores:
        return 0.0
    return (sum(with_scores) / len(with_scores)) - (sum(bare_scores) / len(bare_scores))


def select_uplift_memory(
    adapter: _RunAdapter,
    student_config: AgentConfig,
    candidates: list[FewShotExample],
    val_items: list[FeedItem],
    max_keep: int = 5,
    k: int = 3,
    min_u: float = 0.0,
) -> tuple[list[FewShotExample], list[tuple[FewShotExample, float]]]:
    """Keep candidates with u > min_u, ranked by u, capped at max_keep.

    Returns (kept, all_scored) where all_scored is (example, u) sorted by u desc.
    Kept examples are re-tagged source='uplift'.
    """
    scored: list[tuple[FewShotExample, float]] = []
    for cand in candidates:
        u = estimate_uplift(adapter, student_config, cand, val_items, k=k)
        scored.append((cand, u))
    scored.sort(key=lambda t: t[1], reverse=True)

    kept: list[FewShotExample] = []
    for cand, u in scored:
        if u <= min_u:
            continue
        if len(kept) >= max_keep:
            break
        kept.append(
            cand.model_copy(update={"source": "uplift"})
        )
    return kept, scored


def build_val_slice(
    learn_items: list[FeedItem],
    candidate_questions: set[str],
    n: int = 12,
) -> list[FeedItem]:
    """Unique hard LEARN items whose questions are not in the candidate set."""
    seen: set[str] = set()
    out: list[FeedItem] = []
    for it in learn_items:
        if it.difficulty not in ("hard", "extra"):
            continue
        if it.question in candidate_questions:
            continue
        if it.question_id in seen:
            continue
        seen.add(it.question_id)
        out.append(it)
        if len(out) >= n:
            break
    return out


def uplift_enabled() -> bool:
    return os.environ.get("UPLIFT_GATE", "1") != "0"


def memory_max_total(default: int = 5) -> int:
    raw = os.environ.get("MEMORY_MAX_TOTAL", "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default
