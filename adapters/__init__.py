"""Benchmark adapters for the self-improvement orchestrator."""
from __future__ import annotations

from core.adapter import TaskAdapter


def get_adapter(name: str) -> TaskAdapter:
    if name == "finance":
        from adapters.finance import FinanceAdapter

        return FinanceAdapter()
    raise ValueError(
        f"Unknown adapter {name!r} — choose finance"
    )


__all__ = ["get_adapter", "TaskAdapter"]
