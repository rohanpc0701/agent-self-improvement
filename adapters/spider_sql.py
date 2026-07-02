"""Spider text-to-SQL adapter — wraps existing harness paths."""
from __future__ import annotations

from contracts.schemas import AgentConfig, FewShotExample
from correction.correction import handle as correction_handle
from correction.learner import FailingCase
from harness.feed import FeedItem, build_continuous_stream, build_stream
from harness.runner import run_item
from harness.spider import load_questions


class SpiderSQLAdapter:
    name = "spider"

    def load_questions(self) -> list[dict]:
        return load_questions()

    def build_feed(self, n: int, full: bool, seed: int) -> list[FeedItem]:
        per_phase = 80 if full else n
        return build_stream(
            self.load_questions(),
            n_baseline=per_phase,
            n_degraded=per_phase,
            n_recovery=per_phase,
            seed=seed,
            same_db_split=True,
            baseline_easy_only=True,
        )

    def build_continuous_feed(
        self, n_cycles: int, full: bool, seed: int
    ) -> list[FeedItem]:
        per_phase = 50 if full else 30
        n_baseline = 80 if full else 40
        return build_continuous_stream(
            self.load_questions(),
            n_baseline=n_baseline,
            n_degraded=per_phase,
            n_recovery=per_phase,
            n_cycles=n_cycles,
            seed=seed,
            same_db_split=True,
            baseline_easy_only=True,
        )

    def run_item(
        self, item: FeedItem, config: AgentConfig, use_rules: bool = True
    ):
        return run_item(item, config, use_rules=use_rules)

    def make_examples(
        self,
        failing_cases: list[FailingCase],
        anchor_cases: list[FailingCase],
    ) -> list[FewShotExample]:
        from contracts.schemas import DriftEvent, FailureMode

        # Reuse correction.handle severity gate + teacher verification path.
        stub = DriftEvent(
            detected_at=0.0,
            channel="execution_accuracy",
            severity=1.0,
            window_mean=0.0,
            baseline_mean=1.0,
            failure_mode=FailureMode.NONE,
        )
        return correction_handle(stub, failing_cases, anchor_cases).new_few_shot_examples
