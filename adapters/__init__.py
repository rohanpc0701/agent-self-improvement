"""Benchmark adapters for the self-improvement orchestrator."""
from __future__ import annotations

from core.adapter import TaskAdapter


def get_adapter(name: str) -> TaskAdapter:
    if name == "spider":
        from adapters.spider_sql import SpiderSQLAdapter

        return SpiderSQLAdapter()
    if name == "gsm8k":
        from adapters.gsm8k_math import GSM8KMathAdapter

        return GSM8KMathAdapter()
    if name == "coding":
        from adapters.coding import CodingAdapter

        return CodingAdapter()
    if name == "finance":
        from adapters.finance import FinanceAdapter

        return FinanceAdapter()
    raise ValueError(
        f"Unknown adapter {name!r} — choose spider, gsm8k, coding, or finance"
    )


__all__ = ["get_adapter", "TaskAdapter"]
