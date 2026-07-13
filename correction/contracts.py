"""Local contracts for the correction module.

FailedRun, CorrectionRule, CorrectionContext are internal to this stage.
The global DriftEvent and CorrectionAction live in contracts/schemas.py.
"""
from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class FailedRun(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    run_id: str
    domain_id: str = Field(validation_alias=AliasChoices("domain_id", "db_id"))
    question: str
    broken_output: str = Field(
        validation_alias=AliasChoices("broken_output", "broken_sql"),
    )
    execution_error: str | None = None
    expected_result: list | None = None
    observed_result: list | None = None
    # Domain context: Spider {table: [cols]}, coding {topic: [cues]}, etc.
    schema: dict = Field(default_factory=dict)


class CorrectionRule(BaseModel):
    id: str
    scope: Literal["db", "global"]
    db_id: str | None = None  # domain scope key (Spider db / coding topic)
    trap: str
    fix: str
    trigger: str
    applies_to: list[str] = Field(default_factory=list)
    source: Literal["react_repair", "seed"]
    hits: int = 0
    seen_dbs: list[str] = Field(default_factory=list)


class CorrectionContext(BaseModel):
    db_id: str  # domain_id at retrieval time
    question: str
    injected_rules: list[str]
    rule_ids: list[str]
