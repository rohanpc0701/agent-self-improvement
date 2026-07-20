"""Example memory: cap and merge few-shot lists across correction cycles."""
from __future__ import annotations

import os

from contracts.schemas import FewShotExample


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def merge_examples(
    existing: list[FewShotExample],
    new: list[FewShotExample],
    max_total: int | None = None,
    max_per_db: int | None = None,
) -> list[FewShotExample]:
    """Merge new examples into existing, dedupe by (db_id, question), cap per db and total.

    FIFO eviction within each db_id when over cap — oldest examples drop first.
    Env overrides: MEMORY_MAX_TOTAL, MEMORY_MAX_PER_DB (used when args are None).
    """
    if max_total is None:
        max_total = _env_int("MEMORY_MAX_TOTAL", 32)
    if max_per_db is None:
        max_per_db = _env_int("MEMORY_MAX_PER_DB", 8)

    by_key: dict[tuple[str, str], FewShotExample] = {}
    order: list[tuple[str, str]] = []

    for ex in existing + new:
        key = (ex.domain_id or "", ex.question)
        if key not in by_key:
            order.append(key)
        by_key[key] = ex

    # Per-db cap (FIFO)
    per_db: dict[str, list[tuple[str, str]]] = {}
    for key in order:
        db = key[0]
        per_db.setdefault(db, []).append(key)
    kept_keys: list[tuple[str, str]] = []
    for db, keys in per_db.items():
        kept_keys.extend(keys[-max_per_db:])

    kept_keys.sort(key=lambda k: order.index(k))
    merged = [by_key[k] for k in kept_keys]
    return merged[:max_total]
