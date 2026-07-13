"""
Shared contracts for all stages.

Domain-agnostic field names (output / domain_id) are canonical. Old SQL-shaped
keys (generated_sql, correct_sql, db_id, invalid_sql) remain accepted on input
via validation aliases so historical events.jsonl still loads.

    from contracts.schemas import TelemetryRecord, DriftEvent, CorrectionAction, AgentConfig
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, AliasChoices, field_validator


class Difficulty(str, Enum):
    """Difficulty buckets for stratified accuracy (contamination-free improvement signal)."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTRA = "extra"


class FailureMode(str, Enum):
    """How a failing run failed — set by the detector, consumed by correction."""

    VALID_BUT_WRONG = "valid_but_wrong"  # ran/parsed, wrong answer
    INVALID_OUTPUT = "invalid_output"  # did not parse / execute / validate
    NONE = "none"

    @classmethod
    def _missing_(cls, value: object):
        # Historical events used invalid_sql
        if value == "invalid_sql":
            return cls.INVALID_OUTPUT
        return None


class FewShotExample(BaseModel):
    """A (question, correct_output) pair the agent learns from."""

    model_config = ConfigDict(populate_by_name=True)

    question: str
    correct_output: str = Field(
        validation_alias=AliasChoices("correct_output", "correct_sql"),
    )
    domain_id: str = Field(
        default="",
        validation_alias=AliasChoices("domain_id", "db_id"),
    )
    source: str = "teacher"  # "teacher" | "gold" | "anchor" | ...


class AgentConfig(BaseModel):
    """Agent state. few_shot_examples starts empty and grows via correction."""

    config_id: str
    model: str
    prompt_version: str = "v1"
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)


class TelemetryRecord(BaseModel):
    """One agent run. Harness emits → detector consumes."""

    model_config = ConfigDict(populate_by_name=True)

    run_id: str
    timestamp: float
    difficulty: Difficulty

    execution_accuracy: float = Field(..., ge=0.0, le=1.0)

    # True if the output was well-formed enough to score (valid SQL / runnable code / …)
    query_valid: bool

    generated_complexity: int = 0
    required_complexity: int = 0

    latency_ms: float = 0.0
    tokens: int = 0

    question: str = ""
    generated_output: str = Field(
        default="",
        validation_alias=AliasChoices("generated_output", "generated_sql"),
    )
    domain_id: str = Field(
        default="",
        validation_alias=AliasChoices("domain_id", "db_id"),
    )
    config_id: str = ""
    reasoning: str = ""

    @property
    def complexity_gap(self) -> int:
        """Positive = agent under-reached required complexity."""
        return self.required_complexity - self.generated_complexity


class DriftEvent(BaseModel):
    """Detector emits → correction consumes (windowed threshold breach)."""

    detected_at: float
    channel: str
    severity: float
    window_mean: float
    baseline_mean: float
    failure_mode: FailureMode = FailureMode.NONE
    failing_run_ids: list[str] = Field(default_factory=list)

    @field_validator("failure_mode", mode="before")
    @classmethod
    def _coerce_failure_mode(cls, v: object) -> object:
        if v == "invalid_sql":
            return FailureMode.INVALID_OUTPUT
        return v


class CorrectionAction(BaseModel):
    """Correction emits → harness consumes (appended to few_shot_examples)."""

    triggered_by: str
    new_few_shot_examples: list[FewShotExample] = Field(default_factory=list)
    rationale: str = ""
