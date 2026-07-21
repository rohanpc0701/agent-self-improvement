# TraceLift on FinancePro-Bench — Result Summary

**To:** Kartheek (CTO)
**From:** Rohan Chavan
**Date:** 2026-07-21
**Test you requested:** one implementation at a time — FinancePro-Bench, GLM 5.2 teacher, Qwen 3.6 27B student, does TraceLift improve the student? No fine-tuning.

## Bottom line

**TraceLift did not reliably improve the student.** A first single-pass run looked positive (+5.6 normalized points, which would have cleared our +4 success bar), but that result was **measurement noise** — under 3 averaged repeats the gap collapsed to **−2.4** (null-to-negative). We are reporting the null, not the +5.6.

This is now the **third domain** — after coding and GSM8K — where teacher-built in-context memory fails to transfer once run-to-run noise is averaged out.

## What we ran

- **Student:** Qwen 3.6 27B (reasoning disabled for speed/stability), on OpenRouter.
- **Teacher:** GLM 5.2 — repairs the student's failed answers and distills the fix into reusable memory.
- **Judge:** GPT-5.2 (different model family from the teacher, to avoid self-preference bias), grading against the official point rubrics.
- **Method:** TraceLift = teacher-built memory (category playbooks + trap rules) injected into the student's prompt. No weights touched.
- **Measured:** student-alone (A1) vs student+memory (A4) on the held-out finance questions where the memory applies (Credit/Trading), normalized rubric score.

## The numbers

| Memory quality | Single pass | 3-repeat average |
|---|---:|---:|
| Boilerplate (first distillation) | GAP −6.1 (hurt) | — |
| Good (fixed distillation) | GAP **+5.6** | GAP **−2.4** |

The single-pass +5.6 was driven mostly by one question that scored +17.9; on identical (temperature-0) re-runs that same question came back **−13.2**. Per-question noise on this stack is **±10–20 points**, so any single-pass delta smaller than ~15 points is not trustworthy.

## Two things that did hold up

1. **The pipeline works end to end** — student fails → GLM repairs → memory is distilled → injected → judged — with no manual steps. And a real fix landed along the way: the first distillation produced generic boilerplate that actively *hurt* the student (−6.1); rewriting it so the teacher writes genuine, transferable reasoning (grounded in ASC/IFRS standards, with no leakage of question specifics) removed the harm.
2. **Our methodology caught a false positive.** Without the repeat-and-average discipline we established earlier, we would have reported "TraceLift works, +5.6." It doesn't. That noise discipline is the durable asset from this work.

## Honest caveats (on the null too)

- Small sample (5 scored held-out questions; 2 lost to judge/rubric parsing issues), Credit/Trading categories only, no confidence interval. This is "no reliable lift," not "proven exactly zero."
- We bypassed TraceLift's uplift-gate (the harness stage that filters memory to only items that measurably help) because it was unreliable in this run. In principle a working gate could isolate a helping subset — but nothing in the current evidence suggests a meaningful effect to isolate.

## Recommendation

On this evidence, in-context (prompt-based) memory is **marginal at best** for a ~27B student on expert rubric reasoning — consistent with two prior domains. The lever that our background research points to for closing this kind of gap is **delivering the teacher's knowledge through weights (LoRA)** rather than the prompt — but that is fine-tuning, which you ruled out for this test. So within the no-fine-tuning constraint, the honest conclusion is: **the mechanism runs, but it does not produce a reliable improvement here.**

If we want to push the in-context path further before accepting the null: larger held-out set across more categories, ≥3 repeats per arm with paired statistics, and a working uplift-gate. Happy to scope that, or to move to a different bet — your call.

Full data and methodology: `docs/FINDINGS_FINANCE.md` §G–H (verbatim numbers, all runs).
