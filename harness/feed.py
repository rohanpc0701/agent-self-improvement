"""The change-point stratified feed (rules/02-tech-decisions.md).

Phase 1 baseline: sample easy/medium.   --change-point-->
Phase 2 degraded: sample hard/extra LEARN split (failures here become few-shot examples).
Phase 3 recovery: sample hard/extra HELD-OUT split (disjoint — never used as few-shot source).

The LEARN/HELD-OUT split is the key benchmark-credibility guarantee: the agent cannot
regurgitate examples it was given — it must generalize to questions it has never seen.

Fast REPLAY mode: pre-compute the full stream once, replay instantly on stage.

When HELDOUT_MANIFEST points at a frozen manifest JSON, LEARN/HELD-OUT pools come
from that split (RSI-Mem honesty rule) instead of re-splitting.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class FeedItem:
    question_id: str
    question: str
    gold_output: str  # gold SQL / code / answer (domain-specific)
    domain_id: str  # Spider db_id, coding topic, etc.
    difficulty: str
    phase: str   # "baseline" | "degraded" | "recovery"


def _load_manifest_pools(
    hard_extra: list[dict],
) -> tuple[list[dict], list[dict]] | None:
    """If HELDOUT_MANIFEST is set, return (learn_pool, heldout_pool) from it."""
    path = os.environ.get("HELDOUT_MANIFEST", "").strip()
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HELDOUT_MANIFEST not found: {path}")
    m = json.loads(p.read_text())
    by_id = {q["id"]: q for q in hard_extra}
    missing_h = [i for i in m["heldout_ids"] if i not in by_id]
    missing_l = [i for i in m["learn_ids"] if i not in by_id]
    if missing_h or missing_l:
        raise ValueError(
            f"Manifest ids missing from fixture: "
            f"heldout={missing_h[:5]} learn={missing_l[:5]}"
        )
    # LEARN stream = learn_ids ∪ validation_ids (validation is LEARN-side only)
    learn_ids = list(m["learn_ids"]) + list(m.get("validation_ids") or [])
    learn_pool = [by_id[i] for i in learn_ids if i in by_id]
    heldout_pool = [by_id[i] for i in m["heldout_ids"]]
    return learn_pool, heldout_pool


def _split_hard(
    hard_extra: list[dict], learn_frac: float, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split hard/extra into disjoint LEARN and HELD-OUT sets."""
    shuffled = hard_extra.copy()
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * learn_frac))
    return shuffled[:cut], shuffled[cut:]


def _split_hard_by_db(
    hard_extra: list[dict], rng: random.Random, heldout_frac: float = 0.4
) -> tuple[list[dict], list[dict]]:
    """DB-aware fractional split.

    For each database with ≥2 hard/extra questions: `heldout_frac` of its questions
    (min 1) go to HELD-OUT, the rest go to LEARN. This guarantees every HELD-OUT question
    has at least one same-DB question in LEARN for schema-relevant few-shot examples.

    At heldout_frac=0.4 with 8 concentrated DBs: ~50 held-out, ~80 LEARN — large enough
    for statistically meaningful accuracy comparisons.

    Single-question databases go entirely to LEARN — they cannot contribute a same-DB
    example to HELD-OUT, so testing them would not demonstrate the relevance mechanism.
    """
    by_db: dict[str, list[dict]] = defaultdict(list)
    for q in hard_extra:
        by_db[q["db_id"]].append(q)

    learn: list[dict] = []
    heldout: list[dict] = []
    for qs in by_db.values():
        if len(qs) == 1:
            learn.extend(qs)
        else:
            shuffled = qs.copy()
            rng.shuffle(shuffled)
            n_held = max(1, int(len(shuffled) * heldout_frac))
            heldout.extend(shuffled[:n_held])
            learn.extend(shuffled[n_held:])
    return learn, heldout


def build_stream(
    questions: list[dict],
    n_baseline: int = 80,
    n_degraded: int = 80,
    n_recovery: int = 80,
    seed: int = 42,
    learn_frac: float = 0.5,
    same_db_split: bool = False,
    db_heldout_frac: float = 0.4,
    baseline_easy_only: bool = False,
) -> list[FeedItem]:
    """Pre-compute the full demo stream. Call once; replay fast.

    hard/extra questions are split into disjoint LEARN (degraded phase, few-shot source)
    and HELD-OUT (recovery phase, benchmark eval) sets. Recovery accuracy is therefore an
    out-of-sample generalization claim, not memorisation of injected examples.

    same_db_split=True uses _split_hard_by_db (leave-one-out per database) instead of a
    random fraction split. This guarantees every HELD-OUT question has at least one same-DB
    question in LEARN, making injected few-shot examples schema-relevant rather than noise.

    baseline_easy_only=True draws the baseline phase from EASY questions only. On complex
    schemas a weak base model fails ~50% of "medium" questions, so an easy+medium baseline
    is noisy (~0.60) and dips far enough to false-trigger the drift detector BEFORE the
    change-point. Easy-only gives the stable-high baseline the change-point story requires.

    If HELDOUT_MANIFEST is set, LEARN/HELD-OUT pools are taken from that frozen split.
    """
    rng = random.Random(seed)
    baseline_difficulties = ("easy",) if baseline_easy_only else ("easy", "medium")
    easy_med = [q for q in questions if q["difficulty"] in baseline_difficulties]
    hard_extra = [q for q in questions if q["difficulty"] in ("hard", "extra")]

    if not easy_med:
        raise ValueError(
            f"No baseline questions ({'/'.join(baseline_difficulties)}) — "
            "run fixtures/prepare_spider.py first"
        )
    if not hard_extra:
        raise ValueError("No hard/extra questions — run fixtures/prepare_spider.py first")

    manifest_pools = _load_manifest_pools(hard_extra)
    if manifest_pools is not None:
        learn_pool, heldout_pool = manifest_pools
    elif same_db_split:
        learn_pool, heldout_pool = _split_hard_by_db(
            hard_extra, rng, heldout_frac=db_heldout_frac
        )
    else:
        learn_pool, heldout_pool = _split_hard(hard_extra, learn_frac, rng)

    if not heldout_pool:
        raise ValueError(
            f"Held-out pool is empty (learn_frac={learn_frac}, {len(hard_extra)} hard questions). "
            "Lower learn_frac or add more hard questions."
        )

    def _pick(pool: list[dict], n: int, phase: str) -> list[FeedItem]:
        # Prefer without-replacement so n_heldout≈unique held-out Qs for honest eval.
        if n <= len(pool):
            chosen = rng.sample(pool, n)
        else:
            chosen = pool.copy()
            rng.shuffle(chosen)
            chosen.extend(rng.choices(pool, k=n - len(pool)))
        return [
            FeedItem(
                question_id=q["id"],
                question=q["question"],
                gold_output=q["expected_sql"],
                domain_id=q["db_id"],
                difficulty=q["difficulty"],
                phase=phase,
            )
            for q in chosen
        ]

    return (
        _pick(easy_med, n_baseline, "baseline")
        + _pick(learn_pool, n_degraded, "degraded")    # few-shot source; failures extracted here
        + _pick(heldout_pool, n_recovery, "recovery")  # benchmark eval; never a few-shot source
    )


def stream(items: list[FeedItem]) -> Iterator[FeedItem]:
    yield from items


def build_hard_curriculum_stream(
    questions: list[dict],
    n_baseline: int = 40,
    n_learn: int = 100,
    n_heldout: int = 40,
    seed: int = 42,
    db_heldout_frac: float = 0.35,
) -> list[FeedItem]:
    """Eval feed: minimal easy warmup → long hard LEARN → held-out hard eval.

    Easy baseline exists only so the detector can freeze a high baseline
    (baseline_len≈40). Teaching signal and KG come from hard LEARN instances.
    Held-out hard questions are disjoint from LEARN (same-domain split).
    """
    return build_stream(
        questions,
        n_baseline=n_baseline,
        n_degraded=n_learn,
        n_recovery=n_heldout,
        seed=seed,
        same_db_split=True,
        db_heldout_frac=db_heldout_frac,
        baseline_easy_only=True,
    )


def build_continuous_stream(
    questions: list[dict],
    n_baseline: int = 40,
    n_degraded: int = 50,
    n_recovery: int = 50,
    n_cycles: int = 3,
    seed: int = 42,
    same_db_split: bool = True,
    db_heldout_frac: float = 0.4,
    baseline_easy_only: bool = True,
) -> list[FeedItem]:
    """Multi-cycle stream: baseline warmup, then repeated degraded→recovery ramps.

    Each cycle can trigger a new drift event and correction in --continuous mode.
    Uses the same LEARN/HELD-OUT split as build_stream for all cycles.
    """
    rng = random.Random(seed)
    baseline_difficulties = ("easy",) if baseline_easy_only else ("easy", "medium")
    easy_med = [q for q in questions if q["difficulty"] in baseline_difficulties]
    hard_extra = [q for q in questions if q["difficulty"] in ("hard", "extra")]

    if not easy_med or not hard_extra:
        raise ValueError("Need easy/medium and hard/extra questions — run prepare_spider.py")

    manifest_pools = _load_manifest_pools(hard_extra)
    if manifest_pools is not None:
        learn_pool, heldout_pool = manifest_pools
    elif same_db_split:
        learn_pool, heldout_pool = _split_hard_by_db(hard_extra, rng, heldout_frac=db_heldout_frac)
    else:
        learn_pool, heldout_pool = _split_hard(hard_extra, 0.5, rng)

    def _pick(pool: list[dict], n: int, phase: str) -> list[FeedItem]:
        return [
            FeedItem(
                question_id=q["id"],
                question=q["question"],
                gold_output=q["expected_sql"],
                domain_id=q["db_id"],
                difficulty=q["difficulty"],
                phase=phase,
            )
            for q in rng.choices(pool, k=n)
        ]

    items = _pick(easy_med, n_baseline, "baseline")
    for _ in range(n_cycles):
        items += _pick(learn_pool, n_degraded, "degraded")
        items += _pick(heldout_pool, n_recovery, "recovery")
    return items
