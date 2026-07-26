# LOCKED RULE: Research integrity (non-negotiable)

This project's only product is a **believable number**. Every rule here exists because a specific
mistake was made and caught.

## Publish nulls
A trustworthy null beats a noisy false positive. Four tracks have nulled (Spider drift-era coding,
GSM8K, FinancePro at k=3) and each null is recorded with its numbers in `docs/FINDINGS_*.md`. Do not
quietly drop a failed arm, re-run until a delta looks good, or report the best of several seeds.

If a headline changes after more data, say so in the same document (`+8.6` at 6 tasks → `+5.3` at 12
is in the memo for exactly this reason).

## Never report a number you cannot recompute
`runs/` is gitignored, so **the artifact is not the record — the memo is**. That obliges you to:
- paste the numbers into the memo/findings doc verbatim, with `n`, `k`, and the arm definitions;
- state the split, seed, dataset SHA, model slugs, and gated/ungated status;
- keep the per-cell JSONL until the result is written up, and say plainly if it is lost.

The PRBench `+5.3` cells were deleted; the CLAUDE.md "Known gaps" section says so. That is the
standard: an unrecomputable number is labelled unrecomputable, not quietly reused.

## Held-out means held-out
Touched once per arm. No selection, no tuning, no gating, no prompt iteration against it. Validation
exists for gating decisions. If you looked at a held-out score and then changed anything, the split
is burned — freeze a new one and say why.

## The rubric firewall is a correctness property, not a formality
The student never sees a rubric; the teacher sees train rubrics only; memory items are entity-scrubbed
and checked against rubric stems. A leak turns the result into "the model was told the answer." The
tests that assert this must never be allowed to go vacuous — a passing test that cannot fail is worse
than no test (see CLAUDE.md "Known gaps" for a live instance).

## Do not pretend the running method is the designed method
When a bug changes what actually ran, the finding is what ran. The PRBench lift came from lessons
built with an **empty** teacher answer (a token-budget bug); the memo says so, in the section that
reports the mechanism. Same discipline for injection caps, skipped items, and failed reps.

## Report the operational caveats with the result
Every result states: `n`, `k`, gated/ungated, `provider_fallback_count()`, tasks dropped for judge or
rubric parse failures, and whether a CI was computed. "Not significance-tested" is a required phrase
when no bootstrap was run — never "significant" without one.

## Data rights
PRBench (Scale AI, CC-BY) and FinancePro-Bench (CC-BY-4.0) are cited with their licenses; license
files live beside the cached fixtures. Rubrics are the benchmark authors' work — cache them, don't
republish them as ours. Don't pull in assets we don't have rights to.
