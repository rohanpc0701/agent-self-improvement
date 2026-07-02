"""TaskAdapter protocol — swap benchmark domain without touching detector/correction contracts."""
from __future__ import annotations

from typing import Protocol

from contracts.schemas import AgentConfig, FewShotExample, TelemetryRecord
from correction.learner import FailingCase
from harness.feed import FeedItem


class TaskAdapter(Protocol):
    """Minimal surface for orchestrator-driven runs across domains."""

    name: str

    def load_questions(self) -> list[dict]: ...

    def build_feed(self, n: int, full: bool, seed: int) -> list[FeedItem]: ...

    def build_continuous_feed(
        self, n_cycles: int, full: bool, seed: int
    ) -> list[FeedItem]: ...

    def run_item(
        self, item: FeedItem, config: AgentConfig, use_rules: bool = True
    ) -> TelemetryRecord | None: ...

    def make_examples(
        self,
        failing_cases: list[FailingCase],
        anchor_cases: list[FailingCase],
    ) -> list[FewShotExample]: ...
