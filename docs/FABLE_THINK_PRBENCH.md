# Fable, think about this: does teacher-built frozen memory *work*?

**Date:** 2026-07-23 · **Branch:** `feat/finance-tracelift` · **Author:** Rohan
**Your job:** Read the result honestly, then reason about **what evidence would make a skeptic
agree the technique works** — and what's the cheapest path to that evidence. Do not invent
numbers; everything below is from real runs. Prefer honest bounds over optimism.

---

## 1. The claim we are trying to establish

> A **stronger teacher model** can build **frozen, prompt-appended memory** (no fine-tuning, no
> RL, no weight updates) that lifts a **cheaper student model** on **held-out** hard
> domain-reasoning tasks.

Not: retry-until-correct, model swap at inference, or fine-tuning. The memory is plain text
prepended to the student's prompt. The student's weights never change. "Learning" = the memory
list grows from the teacher reading the student's graded mistakes.

**Goal bar (decided with Rohan today):** we are standing behind the **existence** claim — "this
*can* work" — but the evidence must be **robust enough to convince a skeptic**, not a single
lucky number.

---

## 2. Setup (what actually ran)

| Piece | Value |
|---|---|
| Benchmark | PRBench (Scale AI, arXiv 2511.11562), **Corporate Finance** subset |
| Scoring | Weighted-criteria rubric per task (REQ criteria +weight, AVOID traps −weight) → LLM judge yes/no per criterion → normalized 0–100 |
| Student | `deepseek/deepseek-v4-pro`, Fireworks-pinned on OpenRouter (provider pinned, `allow_fallbacks=false`) |
| Teacher | `anthropic/claude-fable-5` (you) |
| Judge | `openai/gpt-5.2` (≠ teacher, ≠ student) |
| Split | 50 train / 15 val / 28 held-out, seed 42 |
| Eval | 12 held-out tasks × **k=3** reps, per-task averaged (temp-0 judge noise is ±10–20 pts, so k=3) |
| Memory | **ungated** (uplift gate bypassed for this run) |
| Firewall | student never sees the rubric; teacher sees rubric train-only; judge-only on held-out |

**How memory is built (the "gaps-only" design that won):**
1. Student answers a **train** task.
2. Judge grades it against the rubric → list of **missed criteria**.
3. Teacher (you) reads `(question + student answer + graded missed-criteria)` and distills **one
   transferable, specifics-free lesson** (playbook / trap / skeleton). Entity-scrubbed, leak-safe.
4. Lesson frozen into `runs/prbench_memory.json`.
5. At **held-out** eval, lessons for the matching category are prepended to the student prompt.
   Student re-answers. Judge re-scores. That delta is the signal.

---

## 3. The result (the full ladder)

12 held-out × k=3, per-task k-avg → mean over tasks:

| Arm | Score | Note |
|---|--:|---|
| **PLAIN** — student alone (floor) | **71.9** | zero-shot baseline |
| **MEM** — student + frozen memory | **77.2** | **Δ +5.3** · 8 wins / 2 ties / 2 losses · per-task sd 6.4 |
| **TEACHER** — Fable alone (ceiling) | **90.5** | beats student on **11/12**; student never beats it |

- Student→teacher gap = **18.6 pts**. Gaps-only memory closes **28%** of it (5.3 / 18.6) with
  prompt text alone.
- The ceiling proves the gap is real headroom, not a weak-teacher artifact. ~13 pts remain that
  in-context lessons don't reach.

**Statistical honesty (this is the whole problem):**
- n = 12. **No bootstrap CI computed yet.** Not significance-tested.
- Single split, single seed (42).
- +5.3 is a point estimate; per-task sd 6.4 means the CI could be wide.

---

## 4. Two findings that shape the method

**(a) Teacher-empty token bug (now fixed).** In the *original* build, your full teacher answer
was silently EMPTY. `TEACHER_MAX_TOKENS` defaulted to 4000; you're a reasoning model, and
`max_tokens` caps reasoning+content together, so a hard answer burned the whole 4000 on reasoning
→ `finish_reason=length` → empty content, returned with no error. So the +5.3 lessons were
distilled from `(question + student answer + missed-criteria)` — **without** a teacher gold
answer. Fixed: default 4000→12000 + a loud warning on empty content (`adapters/prbench.py`).

**(b) Gaps-only beats three-way (the surprising part).** After the fix we re-ran the build *with*
your real worked answer added as a third input (true three-way contrast). Δ **dropped to +2.5**
(regressions doubled, 2→5). Adding your full solution did **not** help. Conclusion: **the teacher
needs only to see *where the student failed*, not to solve the task.** Simpler + cheaper + better.
The +5.3 vs +2.5 gap is within n=12 noise, so the defensible phrasing is "gaps-only is *not
worse*," not "significantly better."

---

## 5. The decision we need you to reason about

We agreed the bar is: **enough evidence that the technique works**, skeptic-proof. Candidate levers
to raise confidence, with rough cost:

| Lever | What it buys | Cost |
|---|---|---|
| **(a) Paired bootstrap CI on current 12** | Does +5.3's CI clear 0? | ~free (resample existing k=3 scores, no API) |
| **(b) Expand to full 28 held-out** | Doubles n, tightens CI, kills "cherry-picked 12" | real API cost, ~2.3× runs |
| **(c) Second split / second seed** | Guards against "seed 42 was lucky" | medium API cost |
| **(d) Per-task win/loss sign test** | 8W/2T/2L → distribution-free signal | ~free |
| **(e) Turn uplift gate on** | Kills the 2 regressions, chases the 13-pt headroom | build change + rerun |

Rohan's leaning (my recommendation to him): bar = **"+Δ with bootstrap CI clearing 0 on the full
28 held-out (k=3), plus win/loss sign test."** = (a)+(b)+(d). Hold (c) as a stretch only if 28
lands ambiguous. Uplift gate (e) and LoRA are *after* "works" is banked, not part of proving it.

**Questions for you, Fable:**
1. Is CI-clears-0 on n=28 the right bar, or is single-seed still a fatal hole a reviewer stabs?
2. With per-task sd 6.4 and n=28, is a bootstrap CI even likely to clear 0 — or do we need k>3
   / more held-out / paired variance reduction to get there? Rough power intuition?
3. Anything cheaper than (b) that a skeptic would still accept?
4. Is "gaps-only not worse than three-way" strong enough to *drop* the three-way arm entirely, or
   should we keep it as a documented control?
5. Any confound in the setup (firewall, provider pin, judge = gpt-5.2) that undermines the
   existence claim before we spend API budget scaling it?

Reason step by step. Give a recommended bar and the cheapest credible path to it.
