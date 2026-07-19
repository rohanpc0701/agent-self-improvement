"""Wires the full live loop. BUILT TOGETHER AT THE INTEGRATION CHECKPOINT (hr 5-6).

Flow:
    Pass 1 (baseline + degraded): run agent item-by-item, feed records to the
      detector; on DriftEvent, call correction to build few-shot examples and
      append CorrectionAction to events.jsonl.
    Pass 2 (recovery): run the held-out items; _active_config in the harness
      re-reads the CorrectionAction each item so the agent has the learned examples.
    Comparison: print held-out hard-bucket accuracy with vs without examples,
      side-by-side — the unambiguous self-improvement signal.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from contracts.eventlog import DEFAULT_LOG, append_event
from contracts.schemas import AgentConfig, DriftEvent, FewShotExample
from correction.learner import FailingCase
from detector.config import DetectorConfig
from detector.detector import Detector
from harness.feed import FeedItem
from harness.runner import run_item, run_stream

# Student model. Override via env for a local open-source student, e.g.:
#   export AGENT_BASE_URL=http://localhost:11434/v1
#   export AGENT_MODEL=qwen2.5:1.5b-instruct
_BASE_MODEL = os.environ.get("AGENT_MODEL", "MiniMax-M2.7-highspeed")
_SEED = 42  # fixed so dry-run-heldout and the live run measure the exact same pools

# Number of easy baseline successes to inject as anti-forgetting anchors.
_N_ANCHORS = 2


# ---------------------------------------------------------------------------
# Feed construction
# ---------------------------------------------------------------------------

def ceiling_run(items: list[FeedItem]) -> dict[str, float]:
    """Run the teacher model on unique held-out questions to measure the accuracy ceiling.

    Same questions, same eval harness, same difficulty slice as the base/recovery runs —
    the only variable is the model (teacher tier vs. base tier). This gives a same-slice,
    same-harness ceiling: an apples-to-apples upper bound rather than a blended leaderboard
    number.

    No events.jsonl writes (contamination-free; does not affect _active_config).
    """
    from correction.teacher import generate_sql as teacher_generate
    from harness.evaluator import execution_accuracy as eval_acc
    from harness.spider import get_db_path, schema_text

    held_out = [it for it in items if it.phase == "recovery"]
    seen_ids: set[str] = set()
    unique: list[FeedItem] = []
    for it in held_out:
        if it.question_id not in seen_ids:
            unique.append(it)
            seen_ids.add(it.question_id)

    total = len(unique)
    pairs_by_diff: dict[str, list[tuple[str, float]]] = {}
    n_skipped = 0

    print(
        f"[ceiling] {total} unique held-out questions, teacher model (no events.jsonl writes).",
        flush=True,
    )

    for i, item in enumerate(unique, 1):
        db_path = get_db_path(item.domain_id)
        schema = schema_text(db_path)
        try:
            sql = teacher_generate(item.question, schema)
        except Exception as e:
            n_skipped += 1
            print(f"  [{i:>3}/{total}] [{item.difficulty:<6}] ERROR {e}", flush=True)
            continue
        acc = eval_acc(sql, item.gold_output, db_path)
        if acc is None:
            n_skipped += 1
            print(f"  [{i:>3}/{total}] [{item.difficulty:<6}] SKIP (gold failed)", flush=True)
            continue
        mark = "✓" if acc == 1.0 else "✗"
        print(f"  [{i:>3}/{total}] [{item.difficulty:<6}] {mark}  {item.question[:55]}", flush=True)
        pairs_by_diff.setdefault(item.difficulty, []).append((item.question, acc))

    all_pairs = [p for ps in pairs_by_diff.values() for p in ps]
    overall, n_unique = _unique_acc(all_pairs)
    result: dict[str, float] = {"overall": overall}
    for diff in ("easy", "medium", "hard", "extra"):
        if diff in pairs_by_diff:
            acc_val, _ = _unique_acc(pairs_by_diff[diff])
            result[diff] = acc_val

    print(f"\n  teacher ceiling overall: {overall:.3f}  ({n_unique} unique q, {n_skipped} skipped)")
    for diff in ("easy", "medium", "hard", "extra"):
        if diff in result and diff in pairs_by_diff:
            acc_val, n_u = _unique_acc(pairs_by_diff[diff])
            print(f"  {diff:<8}: {acc_val:.3f}  ({n_u} unique q)")

    return result


def _mcnemar_report(label: str, pairs: list[tuple[float, float]]) -> None:
    """Print a paired 2×2 table + McNemar exact two-sided p + 95% CI for one bucket.

    pairs: list of (without_acc, with_acc), each binary (1.0 correct / 0.0 wrong).
    """
    from math import comb

    n = len(pairs)
    if n == 0:
        print(f"\n  [{label}] no questions in this bucket.")
        return

    a = sum(1 for wo, w in pairs if wo == 1.0 and w == 1.0)  # right→right
    b = sum(1 for wo, w in pairs if wo == 0.0 and w == 1.0)  # wrong→right (improved)
    c = sum(1 for wo, w in pairs if wo == 1.0 and w == 0.0)  # right→wrong (regressed)
    d = sum(1 for wo, w in pairs if wo == 0.0 and w == 0.0)  # wrong→wrong

    disc = b + c
    if disc == 0:
        p_val = 1.0
    else:
        tail = min(b, c)
        p_val = min(1.0, 2.0 * sum(comb(disc, k) * (0.5 ** disc) for k in range(tail + 1)))

    deltas = [w - wo for wo, w in pairs]
    mean_delta = sum(deltas) / n
    var = sum((di - mean_delta) ** 2 for di in deltas) / (n - 1) if n > 1 else 0.0
    se = (var / n) ** 0.5
    ci_lo, ci_hi = mean_delta - 1.96 * se, mean_delta + 1.96 * se

    wo_acc = sum(wo for wo, _ in pairs) / n
    w_acc = sum(w for _, w in pairs) / n

    print(f"\n{'=' * 60}")
    print(f"  McNemar paired test — {label}  (n={n})")
    print(f"{'=' * 60}")
    print(f"  WITHOUT examples : {int(round(wo_acc * n))}/{n} correct  ({wo_acc:.3f})")
    print(f"  WITH    examples : {int(round(w_acc * n))}/{n} correct  ({w_acc:.3f})")
    print(f"  Paired 2×2: right→right={a}  wrong→right={b}(improved)  "
          f"right→wrong={c}(regressed)  wrong→wrong={d}")
    print(f"  Discordant b+c={disc}  |  McNemar exact p = {p_val:.4f}  (two-sided)")
    print(f"  Paired Δ = {mean_delta:+.3f}  95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    if p_val < 0.05:
        print(f"  ✓ p<0.05 — not attributable to chance at this sample size.")
    elif p_val < 0.10:
        print(f"  ~ p<0.10 — trend, not conclusive at n={n}.")
    else:
        print(f"  ✗ p≥0.10 — cannot rule out chance at this sample size.")
    print("=" * 60, flush=True)


def compare_teacher_run(items: list[FeedItem], adapter_name: str = "coding") -> None:
    """After learning: student+memory vs unaided teacher on the same held-out questions.

    Requires a prior --full run (CorrectionAction in events.jsonl). Measures whether
    accumulated few-shots + KG close the gap to the teacher on completely new problems.
    """
    from contracts.eventlog import read_events
    from contracts.schemas import CorrectionAction

    corrections = read_events(only="correction")
    if not corrections:
        print(
            "[compare-teacher] No correction in events.jsonl — run --full first.",
            file=sys.stderr,
        )
        sys.exit(1)
    latest: CorrectionAction = corrections[-1]
    student_cfg = _make_base_config("compare-student").model_copy(
        update={
            "config_id": "compare-student",
            "few_shot_examples": latest.new_few_shot_examples,
        }
    )

    held_out = [it for it in items if it.phase == "recovery"]
    seen_ids: set[str] = set()
    unique: list[FeedItem] = []
    for it in held_out:
        if it.question_id not in seen_ids:
            unique.append(it)
            seen_ids.add(it.question_id)

    # Headline = hard bucket; still score extra for overall.
    hard_items = [it for it in unique if it.difficulty == "hard"]
    pool = hard_items or unique
    total = len(pool)
    print(
        f"\n[compare-teacher] {total} unique held-out questions "
        f"(adapter={adapter_name}, student+memory vs unaided teacher)",
        flush=True,
    )
    print(
        f"  student examples loaded: {len(latest.new_few_shot_examples)}",
        flush=True,
    )

    student_scores: list[float] = []
    teacher_scores: list[float] = []
    # per-question: student win / teacher win / tie both right / tie both wrong
    n_s_only = n_t_only = n_both = n_neither = 0

    for i, item in enumerate(pool, 1):
        rec_s = _run_item(item, student_cfg, adapter_name, use_rules=True)
        if rec_s is None:
            print(f"  [{i:>2}/{total}] SKIP student  {item.question[:45]}", flush=True)
            continue
        s_acc = rec_s.execution_accuracy

        t_acc = _teacher_score_item(item, adapter_name)
        if t_acc is None:
            print(
                f"  [{i:>2}/{total}] [{item.difficulty:<6}] "
                f"S={'✓' if s_acc == 1.0 else '✗'} T=ERR  {item.question[:40]}",
                flush=True,
            )
            continue

        student_scores.append(s_acc)
        teacher_scores.append(t_acc)
        if s_acc == 1.0 and t_acc == 1.0:
            n_both += 1
            tag = "both✓"
        elif s_acc == 1.0 and t_acc == 0.0:
            n_s_only += 1
            tag = "S>T"
        elif s_acc == 0.0 and t_acc == 1.0:
            n_t_only += 1
            tag = "T>S"
        else:
            n_neither += 1
            tag = "both✗"

        print(
            f"  [{i:>2}/{total}] [{item.difficulty:<6}] "
            f"S={'✓' if s_acc == 1.0 else '✗'} T={'✓' if t_acc == 1.0 else '✗'} "
            f"{tag:<6}  {item.question[:40]}",
            flush=True,
        )

    if not student_scores:
        print("[compare-teacher] No scorable questions.", file=sys.stderr)
        return

    n = len(student_scores)
    s_mean = sum(student_scores) / n
    t_mean = sum(teacher_scores) / n
    gap = s_mean - t_mean
    print(f"\n{'=' * 60}")
    print(f"  Student+memory vs teacher (held-out, n={n} unique Qs)")
    print(f"    Student + few-shots/KG : {s_mean:.3f}")
    print(f"    Teacher (unaided)      : {t_mean:.3f}")
    print(f"    Gap (student - teacher): {gap:+.3f}")
    print(
        f"    Per-Q: both✓={n_both}  S>T={n_s_only}  T>S={n_t_only}  both✗={n_neither}"
    )
    if gap >= 0:
        print("  ✓ Learned student matched or beat unaided teacher on this slice.")
    elif gap > -0.15:
        print("  ~ Student close to teacher after learning — gap still open.")
    else:
        print("  ✗ Teacher still substantially ahead on held-out.")
    print("=" * 60, flush=True)


def _teacher_score_item(item: FeedItem, adapter_name: str) -> float | None:
    """Unaided teacher accuracy on one feed item. None if unscorable."""
    if adapter_name == "coding":
        from adapters.coding import _index, teacher_solve, verify_solution

        problem = _index().get(item.question_id)
        if problem is None:
            return None
        code = teacher_solve(problem)
        if not code:
            return 0.0
        acc, _, _ = verify_solution(f"```python\n{code}\n```", problem)
        return acc

    # Spider (and similar): teacher SQL vs gold execution
    from correction.teacher import generate_sql as teacher_generate
    from harness.evaluator import execution_accuracy as eval_acc
    from harness.spider import get_db_path, schema_text

    try:
        db_path = get_db_path(item.domain_id)
        schema = schema_text(db_path)
        sql = teacher_generate(item.question, schema)
        acc = eval_acc(sql, item.gold_output, db_path)
        return acc
    except Exception as exc:
        print(f"  [teacher] spider score failed: {exc}", flush=True)
        return None


def significance_run(items: list[FeedItem], adapter_name: str = "spider") -> None:
    """McNemar paired significance test: same held-out questions WITHOUT vs WITH examples.

    Reads the CorrectionAction from events.jsonl (run --full first).
    Runs each unique held-out question exactly twice — base config, then with-examples
    config — so each question is its own control. No events.jsonl writes.

    McNemar exact (two-sided) on the discordant pairs tests H₀: examples have no net
    effect. Normal-approximation 95% CI on the paired per-question deltas gives the
    effect size with uncertainty.
    """
    from contracts.eventlog import read_events
    from contracts.schemas import CorrectionAction

    corrections = read_events(only="correction")
    if not corrections:
        print(
            "[significance] No correction event in events.jsonl — run --full first.",
            file=sys.stderr,
        )
        sys.exit(1)
    latest: CorrectionAction = corrections[-1]
    print(
        f"[significance] Loaded CorrectionAction: {len(latest.new_few_shot_examples)} examples.",
        flush=True,
    )

    base_config = _make_base_config("sig-base")
    with_config = base_config.model_copy(update={
        "config_id": "sig-with",
        "few_shot_examples": latest.new_few_shot_examples,
    })

    held_out = [it for it in items if it.phase == "recovery"]
    seen_ids: set[str] = set()
    unique: list[FeedItem] = []
    for it in held_out:
        if it.question_id not in seen_ids:
            unique.append(it)
            seen_ids.add(it.question_id)

    total = len(unique)
    print(
        f"[significance] {total} unique held-out questions × 2 passes = {total * 2} agent calls.",
        flush=True,
    )

    results: list[tuple[str, str, float, float]] = []  # (question_id, difficulty, wo, w)
    n_skipped = 0

    for i, item in enumerate(unique, 1):
        # WITHOUT pass: no examples AND no knowledge-graph rules (clean control)
        rec_wo = _run_item(item, base_config, adapter_name, use_rules=False)
        rec_w = _run_item(item, with_config, adapter_name)

        if rec_wo is None or rec_w is None:
            n_skipped += 1
            print(f"  [{i:>2}/{total}] SKIP (gold failed)", flush=True)
            continue

        wo = rec_wo.execution_accuracy
        w = rec_w.execution_accuracy
        results.append((item.question_id, item.difficulty, wo, w))

        if wo == 0.0 and w == 1.0:
            verdict = "✗→✓ IMPROVED"
        elif wo == 1.0 and w == 0.0:
            verdict = "✓→✗ REGRESSED"
        elif wo == 1.0 and w == 1.0:
            verdict = "✓→✓"
        else:
            verdict = "✗→✗"
        print(
            f"  [{i:>2}/{total}] [{item.difficulty:<6}] {verdict}  {item.question[:50]}",
            flush=True,
        )

    if not results:
        print("[significance] No scorable questions.", file=sys.stderr)
        return

    # The headline claim is HARD-bucket, so test that bucket first; overall second.
    hard_pairs = [(wo, w) for _, diff, wo, w in results if diff == "hard"]
    all_pairs = [(wo, w) for _, _, wo, w in results]
    _mcnemar_report("HARD bucket (the headline claim)", hard_pairs)
    _mcnemar_report("Overall (hard + extra)", all_pairs)


def _get_adapter(name: str = "spider"):
    from adapters import get_adapter
    return get_adapter(name)


def _build_feed(n: int, full: bool, adapter_name: str = "spider") -> list[FeedItem]:
    """Load questions and build the change-point stream via the task adapter."""
    adapter = _get_adapter(adapter_name)
    return adapter.build_feed(n, full, _SEED)


def _build_continuous_feed(
    n_cycles: int = 2,
    full: bool = False,
    adapter_name: str = "spider",
) -> list[FeedItem]:
    adapter = _get_adapter(adapter_name)
    return adapter.build_continuous_feed(n_cycles, full, _SEED)


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _unique_acc(
    question_acc_pairs: list[tuple[str, float]],
) -> tuple[float, int]:
    """Mean accuracy over unique questions.

    When the same question appears multiple times (stream samples with replacement),
    take the first occurrence — at temperature=0.0 the model is deterministic so all
    occurrences have the same accuracy. Returns (mean, n_unique).
    """
    by_q: dict[str, float] = {}
    for q, acc in question_acc_pairs:
        if q not in by_q:
            by_q[q] = acc
    if not by_q:
        return 0.0, 0
    return sum(by_q.values()) / len(by_q), len(by_q)


# ---------------------------------------------------------------------------
# Step 0.5: headroom gate (plan 006)
# ---------------------------------------------------------------------------

def dry_run_heldout(
    items: list[FeedItem],
    config: Optional[AgentConfig] = None,
    adapter_name: str = "spider",
) -> dict[str, float]:
    """Run held-out (recovery) items at base config, no corrections injected.

    Contamination-free by construction:
    - Fresh AgentConfig with empty few_shot_examples (no _active_config call).
    - Adapter run_item is called directly — it does not touch events.jsonl.

    Returns a dict with "overall" and per-difficulty unique-question accuracy.
    Unique-question accuracy deduplicates repeated samples (stream uses rng.choices with
    replacement, so the same question can appear multiple times; at temperature=0.0 the
    model is deterministic, so the first occurrence is canonical).
    """
    if config is None:
        config = AgentConfig(
            config_id="v0-base-dryrun",
            model=_BASE_MODEL,
            few_shot_examples=[],
        )

    held_out = [it for it in items if it.phase == "recovery"]
    total = len(held_out)
    # (question, accuracy) pairs per difficulty, for unique-question dedup
    pairs_by_diff: dict[str, list[tuple[str, float]]] = {}
    n_scored = 0
    n_skipped = 0

    print(
        f"[dry-run-heldout] {total} held-out items, base config (no corrections). "
        f"adapter={adapter_name} — expect a few seconds each.",
        flush=True,
    )

    for i, item in enumerate(held_out, 1):
        rec = _run_item(item, config, adapter_name, use_rules=False)
        if rec is None:
            n_skipped += 1
            print(f"  [{i:>3}/{total}] [{item.difficulty:<6}] SKIP (gold output failed)  "
                  f"{item.question[:55]}", flush=True)
            continue
        n_scored += 1
        pairs_by_diff.setdefault(item.difficulty, []).append(
            (item.question, rec.execution_accuracy)
        )
        mark = "✓" if rec.execution_accuracy == 1.0 else "✗"
        print(f"  [{i:>3}/{total}] [{item.difficulty:<6}] {mark}  "
              f"{item.question[:55]}", flush=True)

    all_pairs = [p for ps in pairs_by_diff.values() for p in ps]
    overall, n_unique = _unique_acc(all_pairs)
    result: dict[str, float] = {"overall": overall}
    for diff in ("easy", "medium", "hard", "extra"):
        if diff in pairs_by_diff:
            acc, n_u = _unique_acc(pairs_by_diff[diff])
            result[diff] = acc

    print(
        f"\n  overall : {overall:.3f}  "
        f"({n_unique} unique q, {n_scored} runs, {n_skipped} skipped — gold output failed)"
    )
    for diff in ("easy", "medium", "hard", "extra"):
        if diff in result and diff in pairs_by_diff:
            acc_u, n_u = _unique_acc(pairs_by_diff[diff])
            print(f"  {diff:<8}: {acc_u:.3f}  ({n_u} unique q, {len(pairs_by_diff[diff])} runs)")

    if overall >= 0.7:
        print(
            "\n  WARNING: base accuracy on held-out is high (>=0.7). "
            "The recovery 'V' may reflect pool difficulty, not learning. "
            "See plan 006 Step 0.5 — consider re-seeding the split."
        )
    else:
        print(
            f"\n  Headroom confirmed: base struggles on held-out ({overall:.3f}). "
            "Recovery improvement will be attributable to learned examples."
        )

    return result


# ---------------------------------------------------------------------------
# Step 5: two-pass live loop
# ---------------------------------------------------------------------------

def _run_item(
    item: FeedItem,
    config: AgentConfig,
    adapter_name: str = "spider",
    use_rules: bool = True,
):
    return _get_adapter(adapter_name).run_item(item, config, use_rules=use_rules)


def _apply_correction(
    adapter_name: str,
    event: DriftEvent,
    failing_cases: list[FailingCase],
    anchor_cases: list[FailingCase],
):
    from contracts.schemas import CorrectionAction

    adapter = _get_adapter(adapter_name)
    examples = adapter.make_examples(failing_cases, anchor_cases)
    n_teacher = sum(1 for e in examples if e.source == "teacher")
    n_gold = sum(1 for e in examples if e.source == "gold")
    n_anchor = sum(1 for e in examples if e.source == "anchor")
    rationale = (
        f"Drift on {event.channel}: window={event.window_mean:.3f}, "
        f"baseline={event.baseline_mean:.3f}, severity={event.severity:.3f}. "
        f"failure_mode={event.failure_mode.value}. "
        f"Injecting {len(examples)} examples "
        f"({n_teacher} teacher-verified, {n_gold} gold-fallback, {n_anchor} anchor)."
    )
    return CorrectionAction(
        triggered_by=event.channel,
        new_few_shot_examples=examples,
        rationale=rationale,
    )


def _make_base_config(run_suffix: str = "v0") -> AgentConfig:
    return AgentConfig(
        config_id=f"v0-base-{run_suffix}",
        model=_BASE_MODEL,
        few_shot_examples=[],
    )


def _build_failing_cases(
    event: DriftEvent,
    run_id_to_record_and_item: dict,
) -> list[FailingCase]:
    """Map event.failing_run_ids back to FailingCase bundles.

    The orchestrator holds the emitted TelemetryRecord AND the FeedItem (which
    carries gold_sql), so this requires no events.jsonl re-reading.
    """
    cases = []
    for run_id in event.failing_run_ids:
        entry = run_id_to_record_and_item.get(run_id)
        if entry is None:
            continue
        rec, item = entry
        cases.append(FailingCase(
            run_id=run_id,
            question=rec.question or item.question,
            domain_id=rec.domain_id or item.domain_id,
            broken_output=rec.generated_output,
            gold_output=item.gold_output,
            difficulty=item.difficulty,
        ))
    return cases


def _harvest_failing_cases(
    run_id_map: dict,
    max_per_db: int = 6,
    max_total: int = 24,
) -> list[FailingCase]:
    """Harvest ALL degraded-phase failures, not just the drift event's capped list.

    The detector caps failing_run_ids at 8, and the stream samples with replacement,
    so drift-captured cases alone give most schemas 0-1 examples — too dilute a dose
    for a small student (validated: ~6 same-DB examples per question recovered +27pts
    in the probe; 1-2 recovered nothing). Dedupe by question, round-robin across
    db_ids so every degraded schema gets coverage, cap per-DB and total to bound
    teacher API calls.
    """
    by_db: dict[str, list[FailingCase]] = {}
    seen_questions: set[str] = set()
    for rec, item in run_id_map.values():
        if item.phase != "degraded" or rec.execution_accuracy != 0.0:
            continue
        if item.question_id in seen_questions:
            continue
        seen_questions.add(item.question_id)
        by_db.setdefault(item.domain_id, []).append(FailingCase(
            run_id=rec.run_id,
            question=rec.question or item.question,
            domain_id=rec.domain_id or item.domain_id,
            broken_output=rec.generated_output,
            gold_output=item.gold_output,
            difficulty=item.difficulty,
        ))

    # Round-robin across DBs so no schema is starved before caps hit.
    cases: list[FailingCase] = []
    for depth in range(max_per_db):
        for db in sorted(by_db):
            if depth < len(by_db[db]) and len(cases) < max_total:
                cases.append(by_db[db][depth])
    return cases


def _write_rules_to_graph(
    event: DriftEvent,
    failing_cases: list[FailingCase],
    max_cases: int = 3,
) -> int:
    """Write correction rules to the knowledge graph (persistent memory layer).

    For each failing case: teacher repairs the broken SQL (ReAct), distill diffs
    broken-vs-fixed into a (trap, fix) rule, and the rule is attached to schema
    nodes in the graph. The agent's prompt hook (harness/agent._correction_rules_block)
    then surfaces matching rules on future runs against the same schema.

    Capped at max_cases to bound teacher API calls; failures are non-fatal —
    the few-shot CorrectionAction path is the primary channel, the graph is
    persistent cross-run memory. Returns the number of rules written.
    """
    from correction.contracts import FailedRun
    from correction.on_drift import on_drift_event
    from harness.evaluator import execute
    from harness.spider import get_db_path

    import sqlite3

    def _schema_dict(db_path: str) -> dict:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]
        out: dict = {}
        for t in tables:
            cur.execute(f"PRAGMA table_info('{t}')")
            out[t] = [c[1] for c in cur.fetchall()]
        con.close()
        return out

    n_written = 0
    for case in failing_cases[:max_cases]:
        try:
            db_path = get_db_path(case.domain_id)
            expected = execute(case.gold_output, db_path)
            observed = execute(case.broken_output, db_path)
            failed = FailedRun(
                run_id=case.run_id,
                domain_id=case.domain_id,
                question=case.question,
                broken_output=case.broken_output,
                expected_result=expected[:5] if expected else None,
                observed_result=observed[:5] if observed else None,
                schema=_schema_dict(db_path),
            )
            rule = on_drift_event(event, failed, db_path=Path(db_path))
            if rule is not None:
                n_written += 1
                print(f"  [graph] rule {rule.id}: {rule.trap[:60]}", flush=True)
        except Exception as exc:
            print(f"  [graph] skipped {case.run_id}: {exc}", flush=True)
    return n_written


def _pick_anchors(
    baseline_items: list[FeedItem],
    baseline_records: list,
    n: int = _N_ANCHORS,
) -> list[FailingCase]:
    """Pick n easy baseline successes as anti-forgetting anchors."""
    anchors = []
    for item, rec in zip(baseline_items, baseline_records):
        if len(anchors) >= n:
            break
        if item.difficulty in ("easy", "medium") and rec.execution_accuracy == 1.0:
            anchors.append(FailingCase(
                run_id=rec.run_id,
                question=rec.question or item.question,
                domain_id=rec.domain_id or item.domain_id,
                broken_output="",
                gold_output=item.gold_output,
                difficulty=item.difficulty,
            ))
    return anchors


def _pass1(
    items: list[FeedItem],
    base_config: AgentConfig,
    detector: Detector,
    run_id_map: dict,
    baseline_items_out: list,
    baseline_recs_out: list,
    adapter_name: str = "spider",
) -> DriftEvent | None:
    """Run baseline + degraded items, feed detector, return DriftEvent when fired."""
    drift_event: DriftEvent | None = None
    pass1_items = [it for it in items if it.phase in ("baseline", "degraded")]
    total = len(pass1_items)

    print(f"\n[pass 1] {total} items (baseline + degraded) ...", flush=True)

    for i, item in enumerate(pass1_items, 1):
        rec = _run_item(item, base_config, adapter_name)
        if rec is None:
            print(f"  [{i:>3}/{total}] [{item.phase:<9}] [{item.difficulty:<6}] SKIP", flush=True)
            continue

        append_event(rec)
        run_id_map[rec.run_id] = (rec, item)

        if item.phase == "baseline":
            baseline_items_out.append(item)
            baseline_recs_out.append(rec)

        ev = detector.update(rec)
        mark = "✓" if rec.execution_accuracy == 1.0 else "✗"
        detector_tag = " 🔔DRIFT" if ev else ""
        print(
            f"  [{i:>3}/{total}] [{item.phase:<9}] [{item.difficulty:<6}] {mark}"
            f"  {item.question[:50]}{detector_tag}",
            flush=True,
        )

        if ev and drift_event is None:
            drift_event = ev
            append_event(ev)
            print(
                f"\n  [detector] Drift detected! channel={ev.channel}, "
                f"severity={ev.severity:.3f}, "
                f"failing_run_ids={ev.failing_run_ids}\n",
                flush=True,
            )

    return drift_event


def _pass2(
    items: list[FeedItem],
    base_config: AgentConfig,
    adapter_name: str = "spider",
) -> list:
    """Run recovery items. _active_config re-reads CorrectionAction each item.

    AGENT_USE_RULES=0 disables knowledge-graph rule injection during recovery —
    useful to isolate the few-shot-example effect (small students can be confused
    by abstract rule text; examples are the validated channel).
    """
    from harness.runner import _active_config

    use_rules = os.environ.get("AGENT_USE_RULES", "1") != "0"
    recovery_items = [it for it in items if it.phase == "recovery"]
    total = len(recovery_items)
    rules_note = "rules ON" if use_rules else "rules OFF (AGENT_USE_RULES=0)"
    print(
        f"\n[pass 2] {total} held-out items (recovery, learned examples, {rules_note}, "
        f"adapter={adapter_name}) ...",
        flush=True,
    )
    records = []
    for i, item in enumerate(recovery_items, 1):
        config = _active_config(base_config)
        rec = _run_item(item, config, adapter_name, use_rules=use_rules)
        if rec is None:
            print(f"  [{i:>3}/{total}] SKIP", flush=True)
            continue
        append_event(rec)
        records.append(rec)
        mark = "✓" if rec.execution_accuracy == 1.0 else "✗"
        print(
            f"  [{i:>3}/{total}] [{item.difficulty:<6}] {mark}  {item.question[:55]}",
            flush=True,
        )
    return records


def _print_comparison(
    recovery_records: list,
    base_accs: dict[str, float],
    diff: str = "hard",
) -> None:
    """Print side-by-side hard-bucket accuracy with vs without examples (unique-question)."""
    diff_recs = [r for r in recovery_records if r.difficulty.value == diff]
    if not diff_recs:
        print(f"\n[result] No {diff} recovery records to compare.")
        return
    with_acc, n_unique = _unique_acc([(r.question, r.execution_accuracy) for r in diff_recs])
    without_acc = base_accs.get(diff, base_accs.get("overall", 0.0))
    print(f"\n{'='*60}")
    print(f"  Self-improvement result ({diff} bucket, {n_unique} unique held-out questions):")
    print(f"    WITHOUT examples (base)  : {without_acc:.3f}")
    print(f"    WITH examples (recovered): {with_acc:.3f}")
    delta = with_acc - without_acc
    sign = "+" if delta >= 0 else ""
    print(f"    Delta                    : {sign}{delta:.3f}")
    if delta > 0:
        print(f"  ✓ Agent improved on {diff} queries after learning from its own failures.")
    else:
        print(f"  ✗ No improvement detected on {diff} queries.")
    print("=" * 60, flush=True)


def probe_relevance(items: list[FeedItem], adapter_name: str = "spider") -> None:
    """Cheap with/without probe: runs unique held-out questions twice (no events.jsonl writes).

    Uses gold SQL/code from same-DB LEARN questions as examples (zero teacher API calls).
    This isolates "does schema-relevant injection help?" from teacher quality.
    Typically ~22 agent calls (2 × 11 unique held-out Qs) vs 240 for a full run.
    """
    base_config = _make_base_config("probe-base")

    # Unique held-out questions (deduplicate by question_id since stream uses choices())
    seen_ids: set[str] = set()
    unique_heldout: list[FeedItem] = []
    for it in items:
        if it.phase == "recovery" and it.question_id not in seen_ids:
            unique_heldout.append(it)
            seen_ids.add(it.question_id)

    # Build same-DB LEARN examples (gold) per db_id — unique questions only
    heldout_ids = {it.question_id for it in unique_heldout}
    learn_by_db: dict[str, list[FeedItem]] = {}
    seen_learn: dict[str, set[str]] = {}
    for it in items:
        if it.phase == "degraded" and it.question_id not in heldout_ids:
            db = it.domain_id
            if db not in seen_learn:
                seen_learn[db] = set()
            if it.question_id not in seen_learn[db]:
                learn_by_db.setdefault(db, []).append(it)
                seen_learn[db].add(it.question_id)

    total = len(unique_heldout)
    print(
        f"\n[probe] {total} unique held-out questions × 2 passes "
        f"({total * 2} agent calls, adapter={adapter_name}, no events.jsonl writes)",
        flush=True,
    )

    without_accs: dict[str, float] = {}
    with_accs: dict[str, float] = {}

    for i, it in enumerate(unique_heldout, 1):
        # WITHOUT — base config, no examples, no knowledge-graph rules
        rec_wo = _run_item(it, base_config, adapter_name, use_rules=False)
        if rec_wo is not None:
            without_accs[it.question_id] = rec_wo.execution_accuracy
            mark_wo = "✓" if rec_wo.execution_accuracy == 1.0 else "✗"
        else:
            mark_wo = "SKIP"

        # WITH — same-DB gold examples only (zero teacher calls)
        same_db = learn_by_db.get(it.domain_id, [])
        examples = [
            FewShotExample(
                question=l.question, correct_output=l.gold_output, domain_id=l.domain_id, source="gold"
            )
            for l in same_db[:4]
        ]
        cfg_with = AgentConfig(
            config_id="probe-with", model=_BASE_MODEL, few_shot_examples=examples
        )
        rec_w = _run_item(it, cfg_with, adapter_name)
        if rec_w is not None:
            with_accs[it.question_id] = rec_w.execution_accuracy
            mark_w = "✓" if rec_w.execution_accuracy == 1.0 else "✗"
        else:
            mark_w = "SKIP"

        n_ex = len(examples)
        print(
            f"  [{i:>2}/{total}] [{it.domain_id:<32}] [{it.difficulty:<6}] "
            f"NO-EX:{mark_wo}  +{n_ex}ex:{mark_w}  {it.question[:40]}",
            flush=True,
        )

    # Comparison
    common = [q for q in without_accs if q in with_accs]
    if not common:
        print("[probe] No scorable questions.")
        return
    wo_mean = sum(without_accs[q] for q in common) / len(common)
    w_mean = sum(with_accs[q] for q in common) / len(common)
    delta = w_mean - wo_mean
    print(f"\n{'='*60}")
    print(f"  Probe: same-DB gold examples, {len(common)} unique held-out Qs")
    print(f"    WITHOUT examples : {wo_mean:.3f}")
    print(f"    WITH    examples : {w_mean:.3f}")
    print(f"    Delta            : {delta:+.3f}")
    if delta >= 0.05:
        print("  ✓ Schema-relevant examples help — proceed to --full run.")
    elif delta <= -0.02:
        print("  ✗ Examples hurting accuracy — check db_id filter or example quality.")
    else:
        print("  ~ Marginal delta — base may be near ceiling; consider weaker model.")
    print("=" * 60, flush=True)


_ABLATION_ARMS = {
    # arm: (AGENT_USE_EXAMPLES value, use_rules)
    "none": ("0", False),
    "examples": ("1", False),
    "rules": ("0", True),
    "both": ("1", True),
}


def _load_latest_examples():
    from contracts.eventlog import read_events

    corrections = read_events(only="correction")
    if not corrections:
        print(
            "[ablation] No correction in events.jsonl — run a learn phase first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return corrections[-1].new_few_shot_examples


def run_ablation_eval(
    items: list, adapter_name: str, arms: list[str]
) -> dict[str, dict]:
    """Frozen-memory ablation: each arm replays the same unique held-out
    hard questions with a different memory channel enabled."""
    examples = _load_latest_examples()
    config = _make_base_config("ablation").model_copy(
        update={"few_shot_examples": examples}
    )

    held = [it for it in items if it.phase == "recovery"]
    seen: set[str] = set()
    pool = []
    for it in held:
        if it.difficulty == "hard" and it.question_id not in seen:
            pool.append(it)
            seen.add(it.question_id)

    print(
        f"\n[ablation] arms={arms} on {len(pool)} unique held-out hard Qs "
        f"({len(examples)} examples in frozen memory)",
        flush=True,
    )

    results: dict[str, dict] = {}
    prev_flag = os.environ.get("AGENT_USE_EXAMPLES")
    try:
        for arm in arms:
            flag, use_rules = _ABLATION_ARMS[arm]
            os.environ["AGENT_USE_EXAMPLES"] = flag
            per_q: dict[str, float] = {}
            zero_inj = 0
            inj_counts: list[int] = []
            for i, item in enumerate(pool, 1):
                rec = _run_item(item, config, adapter_name, use_rules=use_rules)
                if rec is None:
                    continue
                append_event(rec)
                per_q[item.question_id] = rec.execution_accuracy
                stats = rec.injection_stats or {}
                n_inj = stats.get("examples_injected", 0)
                inj_counts.append(n_inj)
                if flag == "1" and n_inj == 0:
                    zero_inj += 1
                print(
                    f"  [{arm:<8}] [{i:>2}/{len(pool)}] "
                    f"{'✓' if rec.execution_accuracy == 1.0 else '✗'} "
                    f"inj={n_inj} {item.question[:40]}",
                    flush=True,
                )
            n = len(per_q)
            acc = sum(per_q.values()) / n if n else 0.0
            results[arm] = {
                "acc": acc,
                "n": n,
                "per_q": per_q,
                "mean_examples": sum(inj_counts) / n if n else 0.0,
                "zero_injection_pct": 100.0
                * (
                    (zero_inj / n)
                    if flag == "1" and n
                    else (1.0 if n else 0.0)
                ),
            }
    finally:
        if prev_flag is None:
            os.environ.pop("AGENT_USE_EXAMPLES", None)
        else:
            os.environ["AGENT_USE_EXAMPLES"] = prev_flag

    print(f"\n{'=' * 60}")
    print("  ABLATION (frozen memory, same held-out hard Qs)")
    for arm, r in results.items():
        print(
            f"    {arm:<9}: acc={r['acc']:.3f}  n={r['n']}  "
            f"mean_inj={r['mean_examples']:.2f}  "
            f"zero_inj={r['zero_injection_pct']:.0f}%"
        )
    print("=" * 60, flush=True)

    if "none" in results:
        base = results["none"]["per_q"]
        for arm in ("examples", "rules", "both"):
            if arm in results:
                pairs = [
                    (base[q], results[arm]["per_q"][q])
                    for q in base
                    if q in results[arm]["per_q"]
                ]
                _mcnemar_report(f"{arm} vs none", pairs)
    return results


def run_hard_curriculum_eval(
    log_path: Path,
    adapter_name: str = "coding",
    n_baseline: int = 40,
    n_learn: int = 100,
    n_heldout: int = 40,
    db_heldout_frac: float = 0.5,
) -> None:
    """Hard-curriculum eval pipeline.

    1. Minimal easy baseline (detector warmup only — not the teaching diet)
    2. Long hard LEARN phase (~n_learn instances) → drift → teacher few-shots + KG
    3. Freeze memory
    4. Compare student+memory vs unaided teacher on held-out hard (never in LEARN)

    Designed as an evaluation story, not a balanced demo stream.
    """
    if adapter_name != "coding":
        print(
            "[hard-curriculum] Currently implemented for --adapter coding "
            "(hard coding + unit-test teacher compare).",
            file=sys.stderr,
        )
        sys.exit(1)

    adapter = _get_adapter(adapter_name)
    items = adapter.build_hard_curriculum_feed(
        seed=_SEED,
        n_baseline=n_baseline,
        n_learn=n_learn,
        n_heldout=n_heldout,
        db_heldout_frac=db_heldout_frac,
    )
    n_base = sum(1 for it in items if it.phase == "baseline")
    n_deg = sum(1 for it in items if it.phase == "degraded")
    n_rec = sum(1 for it in items if it.phase == "recovery")
    n_held_unique = len({it.question_id for it in items if it.phase == "recovery"})

    print(f"\n{'=' * 60}", flush=True)
    print("  HARD-CURRICULUM EVAL PIPELINE", flush=True)
    print(f"  adapter={adapter_name}", flush=True)
    print(f"  easy warmup (detector) : {n_base}", flush=True)
    print(f"  hard LEARN instances   : {n_deg}", flush=True)
    print(f"  held-out samples       : {n_rec} ({n_held_unique} unique Qs)", flush=True)
    print("  then: freeze memory → student+KG vs unaided teacher", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    base_config = _make_base_config("hard-curriculum")
    detector = Detector(DetectorConfig())
    run_id_map: dict = {}
    baseline_items: list = []
    baseline_recs: list = []

    drift_event = _pass1(
        items, base_config, detector, run_id_map, baseline_items, baseline_recs, adapter_name
    )

    if drift_event is None:
        print(
            "\n[hard-curriculum] WARNING: no drift after hard LEARN. "
            "KG/examples may be empty — compare will be weak.",
            file=sys.stderr,
        )
    else:
        print("[correction] Building few-shots + KG from hard LEARN failures ...", flush=True)
        failing_cases = _harvest_failing_cases(run_id_map)
        if not failing_cases:
            failing_cases = _build_failing_cases(drift_event, run_id_map)
        anchor_cases = _pick_anchors(baseline_items, baseline_recs)
        n_domains = len({c.domain_id for c in failing_cases})
        print(
            f"  {len(failing_cases)} failing cases across {n_domains} domains, "
            f"{len(anchor_cases)} anchors",
            flush=True,
        )
        action = _apply_correction(adapter_name, drift_event, failing_cases, anchor_cases)
        append_event(action)
        print(
            f"  CorrectionAction: {len(action.new_few_shot_examples)} examples — "
            f"{sum(1 for e in action.new_few_shot_examples if e.source == 'teacher')} teacher, "
            f"{sum(1 for e in action.new_few_shot_examples if e.source == 'gold')} gold, "
            f"{sum(1 for e in action.new_few_shot_examples if e.source == 'anchor')} anchor",
            flush=True,
        )
        from adapters.coding import write_graph_rules

        print("[graph] Writing KG rules from hard failures ...", flush=True)
        n_rules = write_graph_rules(drift_event, failing_cases)
        print(f"  {n_rules} rule(s) in correction/graph_store.json", flush=True)

    # Contaminaton-free: student WITHOUT on held-out (optional baseline for the claim)
    print(
        "\n[hard-curriculum] Held-out WITHOUT (student, no memory) ...",
        flush=True,
    )
    base_accs = dry_run_heldout(items, adapter_name=adapter_name)

    # Student WITH memory on held-out stream (writes recovery telemetry)
    recovery_records = _pass2(items, base_config, adapter_name=adapter_name)
    _print_comparison(recovery_records, base_accs=base_accs, diff="hard")

    # Headline eval: student+memory vs unaided teacher on new hard problems
    print(
        "\n[hard-curriculum] Freeze memory — compare student+KG vs teacher "
        "on held-out hard ...",
        flush=True,
    )
    compare_teacher_run(items, adapter_name=adapter_name)

    print(f"\n[hard-curriculum] events.jsonl written to {log_path}", flush=True)


def run_full_loop(
    items: list[FeedItem],
    log_path: Path,
    adapter_name: str = "spider",
) -> None:
    """Execute the full two-pass self-improvement loop."""
    base_config = _make_base_config()
    detector = Detector(DetectorConfig())

    # Map run_id -> (TelemetryRecord, FeedItem) so correction can build FailingCase bundles.
    run_id_map: dict = {}
    baseline_items: list = []
    baseline_recs: list = []

    # -------------------------------------------------------------------------
    # Pass 1: baseline + degraded — detect drift, fire correction
    # -------------------------------------------------------------------------
    drift_event = _pass1(
        items, base_config, detector, run_id_map, baseline_items, baseline_recs, adapter_name
    )

    if drift_event is None:
        print(
            "\n[orchestrator] WARNING: no drift detected after pass 1. "
            "Try --full to run more records (detector needs baseline_len=40 + window=25).",
            file=sys.stderr,
        )
        # Still run pass 2 so the log has recovery telemetry; just no correction.
    else:
        # -----------------------------------------------------------------
        # Correction: build examples from failing cases + easy anchors
        # -----------------------------------------------------------------
        print("[correction] Building few-shot examples from failing cases ...", flush=True)
        # Harvest the full degraded window (deduped, per-DB balanced) — the
        # drift event's failing_run_ids are a capped subset, too dilute to teach
        # a small student. Falls back to the event's list if harvesting is empty.
        failing_cases = _harvest_failing_cases(run_id_map)
        if not failing_cases:
            failing_cases = _build_failing_cases(drift_event, run_id_map)
        anchor_cases = _pick_anchors(baseline_items, baseline_recs)
        n_dbs = len({c.domain_id for c in failing_cases})
        print(
            f"  {len(failing_cases)} failing cases across {n_dbs} domains, "
            f"{len(anchor_cases)} anchors",
            flush=True,
        )
        action = _apply_correction(adapter_name, drift_event, failing_cases, anchor_cases)
        append_event(action)
        print(
            f"  CorrectionAction: {len(action.new_few_shot_examples)} examples — "
            f"{sum(1 for e in action.new_few_shot_examples if e.source == 'teacher')} teacher, "
            f"{sum(1 for e in action.new_few_shot_examples if e.source == 'gold')} gold, "
            f"{sum(1 for e in action.new_few_shot_examples if e.source == 'anchor')} anchor",
            flush=True,
        )
        print(f"  rationale: {action.rationale}", flush=True)

        if adapter_name in ("spider", "coding"):
            # -----------------------------------------------------------------
            # Knowledge graph: persist (trap, fix) rules from the same failures
            # -----------------------------------------------------------------
            print("[graph] Distilling failures into knowledge-graph rules ...", flush=True)
            if adapter_name == "coding":
                from adapters.coding import write_graph_rules

                n_rules = write_graph_rules(drift_event, failing_cases)
            else:
                n_rules = _write_rules_to_graph(drift_event, failing_cases)
            print(f"  {n_rules} rule(s) written to correction/graph_store.json", flush=True)

    # -------------------------------------------------------------------------
    # Measure base accuracy on held-out WITHOUT examples (contamination-free)
    # -------------------------------------------------------------------------
    print(
        "\n[orchestrator] Measuring base (no-correction) accuracy on held-out pool ...",
        flush=True,
    )
    base_accs = dry_run_heldout(items, adapter_name=adapter_name)

    # -------------------------------------------------------------------------
    # Pass 2: recovery — agent reads CorrectionAction via _active_config
    # -------------------------------------------------------------------------
    recovery_records = _pass2(items, base_config, adapter_name=adapter_name)

    # -------------------------------------------------------------------------
    # Print the improvement claim (hard bucket is the benchmark — see plan 006)
    # -------------------------------------------------------------------------
    _print_comparison(recovery_records, base_accs=base_accs, diff="hard")

    # Optional: student+memory vs unaided teacher on the same held-out slice
    if os.environ.get("COMPARE_TEACHER", "").strip() in ("1", "true", "yes"):
        compare_teacher_run(items, adapter_name=adapter_name)

    print(f"\n[orchestrator] events.jsonl written to {log_path}", flush=True)


def run_continuous_loop(
    items: list[FeedItem],
    log_path: Path,
    adapter_name: str = "spider",
    max_corrections: int = 3,
    cooldown: int = 15,
) -> None:
    """Multi-cycle stream: drift → correct → recover can repeat; examples accumulate."""
    from harness.runner import _active_config

    base_config = _make_base_config("continuous")
    detector = Detector(DetectorConfig())
    run_id_map: dict = {}
    baseline_items: list = []
    baseline_recs: list = []
    correction_count = 0
    drift_count = 0
    total = len(items)

    print(
        f"\n[continuous] {total} items, adapter={adapter_name}, "
        f"max_corrections={max_corrections}, cooldown={cooldown}",
        flush=True,
    )

    for i, item in enumerate(items, 1):
        config = _active_config(base_config)
        rec = _run_item(item, config, adapter_name)
        if rec is None:
            print(f"  [{i:>3}/{total}] [{item.phase:<9}] SKIP", flush=True)
            continue

        append_event(rec)
        run_id_map[rec.run_id] = (rec, item)
        if item.phase == "baseline":
            baseline_items.append(item)
            baseline_recs.append(rec)

        ev = detector.update(rec)
        mark = "✓" if rec.execution_accuracy == 1.0 else "✗"
        drift_tag = f" 🔔DRIFT#{drift_count + 1}" if ev else ""
        print(
            f"  [{i:>3}/{total}] [{item.phase:<9}] [{item.difficulty:<6}] {mark}"
            f"  {item.question[:45]}{drift_tag}",
            flush=True,
        )

        if ev is None:
            continue

        drift_count += 1
        append_event(ev)
        print(
            f"\n  [detector] Drift #{drift_count}: severity={ev.severity:.3f}, "
            f"window={ev.window_mean:.3f} vs baseline={ev.baseline_mean:.3f}\n",
            flush=True,
        )

        if correction_count >= max_corrections:
            print(
                f"  [continuous] max_corrections={max_corrections} reached — skipping correction.",
                flush=True,
            )
            detector.resume_after_correction(cooldown=cooldown)
            continue

        failing_cases = _harvest_failing_cases(run_id_map)
        if not failing_cases:
            failing_cases = _build_failing_cases(ev, run_id_map)
        anchor_cases = _pick_anchors(baseline_items, baseline_recs)
        action = _apply_correction(adapter_name, ev, failing_cases, anchor_cases)
        append_event(action)
        correction_count += 1
        print(
            f"  [correction #{correction_count}] {len(action.new_few_shot_examples)} examples",
            flush=True,
        )
        if adapter_name == "spider":
            _write_rules_to_graph(ev, failing_cases)
        elif adapter_name == "coding":
            from adapters.coding import write_graph_rules

            write_graph_rules(ev, failing_cases)
        detector.resume_after_correction(cooldown=cooldown)

    print(
        f"\n[continuous] complete: {drift_count} drift(s), {correction_count} correction(s). "
        f"Log: {log_path}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python orchestrator.py",
        description="Agent self-improvement orchestrator (integration checkpoint).",
    )
    p.add_argument(
        "--n", type=int, default=40,
        help="Questions per phase (default 40; must be >=40 for the detector to warm up).",
    )
    p.add_argument(
        "--full", action="store_true",
        help="80 questions per phase — full demo stream.",
    )
    p.add_argument(
        "--dry-run-heldout", action="store_true",
        help=(
            "Run only the held-out pool at base config. No detector, no correction, "
            "no event-log writes. Validates that recovery has headroom (plan 006 Step 0.5)."
        ),
    )
    p.add_argument(
        "--dry-run-degraded", action="store_true",
        help=(
            "Run only the degraded (LEARN) pool at base config; print accuracy. "
            "Confirms the detector will fire before committing to a full --full run."
        ),
    )
    p.add_argument(
        "--probe", action="store_true",
        help=(
            "Cheap with/without test: run each unique held-out question twice — "
            "once WITHOUT examples, once WITH same-DB gold examples. "
            "Zero teacher API calls. ~22 agent calls instead of 240. "
            "Use this to validate that schema-relevant examples help before --full."
        ),
    )
    p.add_argument(
        "--significance", action="store_true",
        help=(
            "McNemar paired significance test: run each unique held-out question twice "
            "(WITHOUT then WITH learned examples from events.jsonl). Requires --full to "
            "have been run first. No events.jsonl writes."
        ),
    )
    p.add_argument(
        "--compare-teacher", action="store_true",
        help=(
            "After learning: score student+memory vs unaided teacher on the same held-out "
            "hard questions. Requires --full first (reads CorrectionAction from events.jsonl). "
            "Coding uses unit-test verify; spider uses SQL EX."
        ),
    )
    p.add_argument(
        "--hard-curriculum", action="store_true",
        help=(
            "Eval pipeline (coding): easy warmup → ~100 hard LEARN → teacher/KG → "
            "student+memory vs unaided teacher on held-out hard. Implies --fresh."
        ),
    )
    p.add_argument(
        "--n-learn", type=int, default=100,
        help="Hard LEARN instances for --hard-curriculum (default 100).",
    )
    p.add_argument(
        "--n-heldout", type=int, default=40,
        help="Held-out samples for --hard-curriculum (default 40; unique Qs fewer).",
    )
    p.add_argument(
        "--heldout-frac",
        type=float,
        default=0.5,
        help=(
            "Fraction of hard questions held out per topic for curriculum/ablation "
            "(default 0.5 → ~30/30 split once hard pool ≥ 60)."
        ),
    )
    p.add_argument(
        "--ablation",
        type=str,
        default=None,
        help=(
            "Comma list of arms (none,examples,rules,both) or 'all'. "
            "Requires a prior learn phase in events.jsonl."
        ),
    )
    p.add_argument(
        "--ceiling", action="store_true",
        help=(
            "Run the teacher model on the unique held-out pool. "
            "Produces an apples-to-apples ceiling: same questions, same eval, stronger model. "
            "No events.jsonl writes. Use the hard-bucket result to replace the SOTA line in "
            "viewer/static/app.js (TEACHER_CEILING constant)."
        ),
    )
    p.add_argument(
        "--adapter",
        choices=("spider", "gsm8k", "coding"),
        default="spider",
        help="Task domain: spider (SQL), gsm8k (math), or coding (unit-tested Python).",
    )
    p.add_argument(
        "--continuous",
        action="store_true",
        help="Multi-cycle stream: repeated drift→correct→recover; examples accumulate.",
    )
    p.add_argument(
        "--max-corrections",
        type=int,
        default=3,
        help="Cap correction cycles in --continuous mode (default 3).",
    )
    p.add_argument(
        "--n-cycles",
        type=int,
        default=2,
        help="Degraded/recovery cycles in --continuous feed (default 2).",
    )
    p.add_argument(
        "--fresh", action="store_true",
        help=(
            "Truncate events.jsonl before a real run so stale correction events "
            "cannot contaminate _active_config."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    from harness.agent import require_api_key
    require_api_key()

    items = _build_feed(args.n, args.full, adapter_name=args.adapter)

    # The validation modes exist to test the --full headline, and the correction
    # examples in events.jsonl were generated under --full. Force the 80/phase feed
    # so the held-out sample (and RNG trajectory) matches, regardless of --n.
    if (args.significance or args.ceiling or args.compare_teacher) and not args.full:
        print("[orchestrator] forcing --full feed for validation mode "
              "(matches the headline's held-out sample).", flush=True)
        items = _build_feed(args.n, full=True, adapter_name=args.adapter)

    if args.significance:
        significance_run(items, adapter_name=args.adapter)
        sys.exit(0)

    if args.compare_teacher:
        compare_teacher_run(items, adapter_name=args.adapter)
        sys.exit(0)

    if args.ablation:
        arms = (
            ["none", "examples", "rules", "both"]
            if args.ablation == "all"
            else [a.strip() for a in args.ablation.split(",")]
        )
        unknown = set(arms) - set(_ABLATION_ARMS)
        if unknown:
            sys.exit(f"unknown ablation arms: {sorted(unknown)}")
        adapter = _get_adapter(args.adapter)
        ablate_items = adapter.build_hard_curriculum_feed(
            seed=_SEED,
            n_heldout=args.n_heldout,
            db_heldout_frac=args.heldout_frac,
        )
        run_ablation_eval(ablate_items, args.adapter, arms)
        sys.exit(0)

    if args.ceiling:
        ceiling_run(items)
        sys.exit(0)

    if args.dry_run_heldout:
        dry_run_heldout(items, adapter_name=args.adapter)
        sys.exit(0)

    if args.probe:
        probe_relevance(items, adapter_name=args.adapter)
        sys.exit(0)

    if args.dry_run_degraded:
        base_cfg = _make_base_config("dry-degraded")
        degraded = [it for it in items if it.phase == "degraded"]
        total = len(degraded)
        accs: list[float] = []
        print(f"[dry-run-degraded] {total} degraded items, base config, "
              f"adapter={args.adapter} ...", flush=True)
        for i, item in enumerate(degraded, 1):
            rec = _run_item(item, base_cfg, args.adapter, use_rules=False)
            if rec is None:
                print(f"  [{i:>3}/{total}] SKIP (gold failed)", flush=True)
                continue
            accs.append(rec.execution_accuracy)
            mark = "✓" if rec.execution_accuracy == 1.0 else "✗"
            print(f"  [{i:>3}/{total}] [{item.difficulty:<6}] {mark}  {item.question[:55]}", flush=True)
        if accs:
            avg = sum(accs) / len(accs)
            print(f"\n  degraded accuracy: {avg:.3f} ({len(accs)} runs)")
            if avg <= 0.70:
                print("  ✓ Degraded accuracy low — detector should fire during a full run.")
            else:
                print("  ✗ Degraded accuracy high — drift may not fire. Check feed configuration.")
        sys.exit(0)

    log_path = Path(DEFAULT_LOG)
    if args.fresh or args.hard_curriculum:
        if log_path.exists():
            log_path.unlink()
            print(f"[orchestrator] {DEFAULT_LOG} cleared (--fresh).")
        graph_store = Path("correction/graph_store.json")
        if graph_store.exists():
            graph_store.unlink()
            print("[orchestrator] correction/graph_store.json cleared (--fresh).")

    if args.hard_curriculum:
        run_hard_curriculum_eval(
            log_path,
            adapter_name=args.adapter,
            n_learn=args.n_learn,
            n_heldout=args.n_heldout,
            db_heldout_frac=args.heldout_frac,
        )
        sys.exit(0)

    if args.continuous:
        cont_items = _build_continuous_feed(
            n_cycles=args.n_cycles,
            full=args.full,
            adapter_name=args.adapter,
        )
        run_continuous_loop(
            cont_items,
            log_path,
            adapter_name=args.adapter,
            max_corrections=args.max_corrections,
        )
        sys.exit(0)

    run_full_loop(items, log_path, adapter_name=args.adapter)
