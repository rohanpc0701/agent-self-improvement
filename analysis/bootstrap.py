"""Paired bootstrap for held-out accuracy deltas (RSI-Mem G0.2)."""
from __future__ import annotations

import random
from typing import Sequence


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Resample problem indices; two-sided p for mean(b)-mean(a) ≠ 0.

    ``a`` and ``b`` are per-problem scores of equal length (paired).
    Returns delta = mean(b)-mean(a), percentile CI, and two-sided p-value.
    """
    if len(a) != len(b):
        raise ValueError(f"paired lengths differ: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("empty paired vectors")
    n = len(a)
    aa = list(a)
    bb = list(b)
    delta = sum(bb[i] - aa[i] for i in range(n)) / n
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        d = sum(bb[i] - aa[i] for i in idxs) / n
        boots.append(d)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    if delta >= 0:
        p = 2.0 * sum(1 for d in boots if d <= 0) / n_boot
    else:
        p = 2.0 * sum(1 for d in boots if d >= 0) / n_boot
    p = min(1.0, p)
    return {
        "delta": delta,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
    }
