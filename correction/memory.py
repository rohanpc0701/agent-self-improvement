"""Example memory: cap and merge few-shot lists across correction cycles."""
from __future__ import annotations

from contracts.schemas import FewShotExample


def merge_examples(
    existing: list[FewShotExample],
    new: list[FewShotExample],
    max_total: int = 32,
    max_per_db: int = 8,
) -> list[FewShotExample]:
    """Merge new examples into existing, dedupe by (db_id, question), cap per db and total.

    FIFO eviction within each db_id when over cap — oldest examples drop first.
    """
    by_key: dict[tuple[str, str], FewShotExample] = {}
    order: list[tuple[str, str]] = []

    for ex in existing + new:
        key = (ex.db_id or "", ex.question)
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
