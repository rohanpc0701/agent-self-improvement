# Teacher-Built Memory on PRBench Corporate Finance — Result Memo

**For:** Kartheek (CTO) · **Date:** 2026-07-21 · **Status:** EVAL COMPLETE

A stronger teacher (Fable) reads the student's graded mistakes and writes transferable
lessons. Those lessons, frozen into a small prompt-appended memory, raise a cheaper
student **+5.3 points** on held-out PRBench Corporate Finance tasks. The student's
weights never change.

> **PLAIN 71.9 → MEM 77.2 · Δ = +5.3** normalized rubric points (MEM − PLAIN, held-out).
> 8 wins / 2 ties / 2 losses · per-task Δ sd 6.4 · 12 held-out × k=3 · ungated memory.

This memo states exactly what was run, what the number means, and — with equal weight —
what it does **not** yet prove. The pipeline is clean; the effect is real and positive;
it is not yet significance-tested.

---

## 1. The claim under test

Can a **stronger teacher model build memory that improves a cheaper student model** on
hard domain-reasoning tasks — **without any fine-tuning or RL**? The only thing that
changes about the student is text prepended to its prompt. If it works, it's a fast,
cheap alternative to the training pipeline: no gradient updates, no weight access,
memory built purely from the student's own graded failures.

Headline comparison: **student-alone (A1 / "PLAIN")** vs **student + frozen
teacher-built memory (A4 / "MEM")** on questions the memory was never built from. Same
model, same weights, same questions — the only variable is whether the memory block is
present.

## 2. Setup — three models, one benchmark

| Role | Model | Job |
|---|---|---|
| **Student** | `deepseek/deepseek-v4-pro` | The cheaper model we're improving. Weights frozen. Pinned to the **Fireworks** provider on OpenRouter for reproducibility (Together as loud fallback). |
| **Teacher** | `anthropic/claude-fable-5` | Stronger model. Reads the student's graded mistakes and writes a transferable lesson. Train-time only. |
| **Judge** | `openai/gpt-5.2` | Grades every answer against PRBench's official weighted-criteria rubric → 0–100 normalized. Different family from teacher *and* student — no self-preference. Hard assert in code. |

### Benchmark & data split

**PRBench** (Professional Reasoning Bench, Scale AI; arXiv 2511.11562), **Corporate
Finance** subset. Free-text expert answers graded by weighted rubrics (10–30 criteria,
weights −10…+10, including trap penalties). Frozen once, seed 42, dataset SHA-256 pinned.

| Split | Tasks | Use |
|---|--:|---|
| Train | 50 | Memory is built here (12 used this run). |
| Validation | 15 | Reserved for the uplift gate (not exercised this run — see §6). |
| Held-out | 28 | Final comparison. First **12** evaluated, each at **k=3**. |

**Rubric firewall (enforced in code):** the student *never* sees a rubric. The teacher
sees rubrics for train tasks only. Held-out rubrics are judge-only. Memory items are
entity-scrubbed so no held-out question text can leak through a lesson.

## 3. Method — contrastive memory

The design distinction from the earlier (null) attempt: memory is not a distilled repair
of the teacher's own answer. It is built **contrastively** — Fable is anchored to the
*specific rubric criteria the student missed* and writes a general lesson from that gap.
(Note: the teacher's own full answer was intended as a third input but was empty in this
run due to a token-budget bug — see §5.5. What actually drove the lift is the teacher
reflecting on the student's **graded** gaps.)

### Build phase — on 12 train tasks

```
for each train task:
  A1  teacher (Fable)  answers the task      # EMPTY in this run (token bug, §5.5)
  A2  student (DeepSeek) answers the task
      judge grades the student → list of MISSED rubric criteria
      Fable( task, teacher answer, student answer, missed criteria )
        → ONE transferable lesson  (≤450 tokens, leak-scrubbed,
           no company names / tickers / dollar amounts)
freeze the 10 lessons from the LOWEST-scoring tasks → prbench_memory.json
```

The student scored 32–87 on the 10 tasks that seeded memory (most to learn where it
struggled most). Two tasks it already scored 100 on were auto-dropped — nothing to teach.
Lessons are ~870–1300 chars each, playbook-style (e.g. *"Stress the structure, not just
the price — a playbook for stock-consideration deals"*).

### Eval phase — on 12 held-out tasks

- **PLAIN (A1):** student answers the held-out task with no memory → judge → score.
- **MEM (A4):** the frozen Corporate-Finance lessons are prepended to the *same* task →
  student answers → judge → score.
- **Δ = MEM − PLAIN**, per task, averaged over **k=3** reps (temp-0 judge noise is ±10–20
  pts single-pass; k=3 smooths it).

## 4. Results — 12 held-out tasks, k=3

| Arm | Score | vs floor | |
|---|--:|--:|---|
| PLAIN — student alone | 71.9 | — | floor |
| MEM — student + memory (gaps-only) | 77.2 | **+5.3** | the claim · 8W/2T/2L |
| TEACHER — Fable alone | 90.5 | +18.6 | ceiling |

**The ladder: floor 71.9 → +5.3 with memory → ceiling 90.5.** The student→teacher gap is
**18.6 pts**; the gaps-only memory closes **28% of it** (5.3 / 18.6), no fine-tuning. Fable
is a genuine ceiling — it beats the student on **11/12** tasks and the student never beats
it, so this is not a weak-teacher artifact. Memory earns a real but **bounded** slice;
~13 pts of headroom remain that in-context lessons alone don't reach (that headroom is the
argument for the uplift gate and, eventually, the weights lever — §8).

*(Two memory variants were tested — see §4a. The `MEM` row above is the better one.)*

### Per-task delta — memory helps most where the student was weakest

| Task | PLAIN | MEM | Δ |
|---|--:|--:|--:|
| 91ea5c… | 46.4 | 55.5 | **+9.0** |
| 643796… | 77.1 | 90.0 | **+12.9** |
| 60d7d7… | 54.7 | 64.0 | **+9.3** |
| bbde88… | 88.0 | 85.6 | −2.3 |
| 4924c9… | 89.1 | 97.5 | **+8.5** |
| 91e69a… | 73.6 | 88.1 | **+14.4** |
| 5d6d37… | 75.4 | 80.5 | **+5.2** |
| ba6263… | 49.2 | 60.2 | **+11.0** |
| 48e764… | 61.5 | 61.5 | 0.0 |
| d946a5… | 86.1 | 88.8 | **+2.6** |
| 8d127f… | 82.9 | 76.0 | −6.9 |
| b008a3… | 79.2 | 79.2 | 0.0 |

The three biggest lifts (+12.9, +14.4, +11.0) land on tasks the student was
middling-to-weak on. The two regressions are the story's honest edge (§6). A full worked example — one task through student, judge, teacher, and the distilled lesson — is in §5.

### 4a. Which memory design — gaps-only vs three-way contrast (resolved)

The token bug in §5.5 meant the first build distilled lessons from *(student answer + graded
missed-criteria)* with an **empty** teacher answer. After fixing the budget (4k→12k) we
re-ran the build so the teacher's real answer was included — a true three-way contrast — and
re-scored held-out:

| Memory design | Δ vs PLAIN | W/T/L | per-task sd |
|---|--:|---|--:|
| **v1 — gaps-only** (student answer + missed criteria) | **+5.3** | 8 / 2 / 2 | 6.4 |
| v2 — three-way (+ teacher's full answer) | +2.5 | 7 / 0 / 5 | 7.4 |

Adding the teacher's worked solution **did not help** — the point estimate dropped and
regressions doubled (2→5). With n=12 and sd ~6–7 the 2.8-pt difference is within noise, so
v2 is not *significantly* worse — but it is clearly **not better**, and simpler wins ties.
**Decision: keep the gaps-only design.** The finding: the teacher doesn't need to *solve*
the problem, only to *see where the student failed* — a cheaper, cleaner mechanism.

## 5. Worked example — one train task through all four roles

To make the pipeline concrete, here is one real train task (`f69d3244…`, a merger
renegotiation under an MAE trigger) traced through student → judge → teacher → lesson.
Text is verbatim from the run, trimmed for length.

### 5.1 The question (excerpt)

> Two complications have emerged. First, the acquirer just released Q2 earnings showing a 12% revenue miss versus guidance, triggering the Material Adverse Effect clause in our merger agreement. They're demanding renegotiation and have proposed three alternatives: (a) reduce the enterprise value from $420m to $378 but keep the existing collar structure, (b) keep the $420m value but widen the collar asymmetrically to a $41.60 floor (20% down) and $54.60 cap (5% up), or (c) switch to a fixed 7.5m shares with no collar protection at all.   Second, their stock has dropped from $52.00 to $48.00 on the earnings news, and our FX exposure has shifted - the $/£ 1.25 to 1.22, and our treasury team estimates it could range between 1.18 and 1.28 by the (now delayed) close in 8 months.  I need a full risk analysis of these three alternatives. For each option, please calculate our net GBP proceeds, assu
> … *[full question ~1,540 chars: asks for net GBP proceeds per alternative, an FX
> Monte-Carlo over 1.18–1.28 with 5th/50th/95th percentiles, and a risk-adjusted pick]*

### 5.2 The rubric (judge-only — 12 weighted criteria, max 87 pts)

PRBench grades against explicit weighted criteria. `REQ` = points for satisfying it;
`AVOID` = a negative-weight trap. The student never sees this.

| # | Kind | Weight | Judge decision | Criterion |
|---|---|--:|---|---|
| C1 | REQ | +6 | ❌ no | Floor breach probability tests collar mechanics understanding |
| C2 | REQ | +10 | ✅ yes | Fee application sequence tests transaction cost treatment |
| C3 | REQ | +5 | ✅ yes | Enterprise value comparison tests Alternative A weakness |
| C4 | REQ | +5 | ❌ no | Timing critique tests Alternative C structural weakness |
| C5 | REQ | +7 | ✅ yes | Recommendation synthesis tests integrated decision-making |
| C6 | REQ | +7 | ❌ no | VWAP specification tests price measurement governance |
| C7 | REQ | +7 | ❌ no | Anti-dilution clause tests equity protection mechanics |
| C8 | REQ | +10 | ❌ no | Re-opener rights test ongoing deal protection |
| C9 | REQ | +10 | ✅ yes | FX hedging recommendation tests currency risk mitigation |
| C10 | REQ | +4 | ✅ yes | Independence assumption tests correlation modelling awareness |
| C11 | REQ | +10 | ❌ no | SSE assumption tests UK tax regime application |
| C12 | REQ | +6 | ✅ yes | This considers enumerating key assumptions beyond independence for auditability |

### 5.3 Student answer (DeepSeek V4, no memory — excerpt)

```
# Comprehensive Risk Analysis of Renegotiation Alternatives

## Key Assumptions First

Before diving into calculations, let me state my explicit assumptions:

1. **Base case**: Original deal was £336m at 1.25 $/£ (enterprise value of $420m)
2. **Advisory fees**: 1.5% of gross proceeds
3. **Tax rate**: Not specified, so I'll assume 20% UK capital gains tax on net gain (after deducting tax basis). However, since you haven't provided your tax basis in the business, I'll calculate **net proceeds after advisory fees but before tax**, then note the tax impact separately. If you need after-tax figures, please provide your tax basis.
4. **Probability distribution**: 60% chance of stock recovery to $50-52 range; 40% chance of remaining at $46-48. I'll model this as a two-point distribution for simplicity.
5. **FX range**: 1.18-1.28 $/£, with current spot at 1.22
6. **Closing**: Now delayed to 8 months
7. **No walk-away**: Analysis assumes you must choose one of the three alternatives (though I'll flag if walking away is superior)

---

## Alternative A: Reduced EV ($378m) with Original Collar

**Structure**: $378m enterprise value, reference price $52.00, floor $46.80, cap $57.20

**Share calculations**:
- At reference: $378m ÷ $52.00 = 7.269m shares
- Floor shares: $378m ÷ $46.80 = 8.077m shares
- Cap shares: $378m ÷ $57.20 = 6.608m shares

**Outcomes by stock price scenario**:

| Stock Price | Mechanism | Shares (m) | Gross $ Value (m) |
|------------|-----------|------------|------
```

The student is well-structured but **hedges instead of deciding** and misses hard
mechanics. It punts on tax ("assume 20%… please provide your basis"), invents its own
60/40 probability split, and never challenges the premise.

### 5.4 Judge grade vs rubric

The judge (GPT-5.2) decides yes/no per criterion, then sums weights:

- **Satisfied (6):** C2 fee sequence, C3 EV comparison, C5 recommendation synthesis,
  C9 FX hedging, C10 independence assumption, C12 assumption enumeration.
- **Missed (6):**
  - Floor breach probability tests collar mechanics understanding.
  - Timing critique tests Alternative C structural weakness.
  - VWAP specification tests price measurement governance.
  - Anti-dilution clause tests equity protection mechanics.
  - Re-opener rights test ongoing deal protection.
  - SSE assumption tests UK tax regime application.
- **Score:** raw 42 / 87 max = **48.3 / 100**.

### 5.5 Teacher answer (Fable — excerpt)

```
# Risk Analysis: Renegotiation Alternatives

## 0. A Threshold Point Before the Numbers: Challenge the MAE Claim

Before evaluating their alternatives, note that **their MAE trigger is legally questionable**:

- MAE clauses in merger agreements almost universally protect the **buyer against deterioration of the target** — not the reverse. Unless you have an unusual bilateral/reciprocal MAE, an acquirer's own revenue miss does not give *them* a renegotiation right.
- Even where applicable, case law (e.g., *Akorn v. Fresenius*, Del. Ch. 2018 — the only Delaware case ever finding an MAE) sets a high bar: the adverse effect must be **durationally significant** ("measured in years, not months") and company-specific. A **single-quarter 12% revenue miss almost certainly does not qualify**.
- Their stock falling from $52 to $48 is precisely the risk the **collar was designed to absorb** — at $48 you'd still receive $420m in value under the existing deal.

**Implication: you are negotiating from strength.** Treat their three alternatives as an opening position, not a menu you must choose from. That said, here is the full analysis, because commercially you may still choose to trade.

---

## 1. Mechanics of Each Alternative

| Parameter | **Option A** | **Option B** | **Option C** |
|---|---|---|---|
| Fixed value inside collar | $378m | $420m | n/a |
| Floor / Cap | $46.80 / $57.20 | $41.60 / $54.60 | none |
| Shares at floor | 378/46.80 = **8.077m** | 420/41.60 = **10.096m** | — |
|
```

The teacher immediately does what the student didn't: **challenges the MAE claim itself**
(cites *Akorn v. Fresenius*, notes MAE protects the buyer, a one-quarter miss won't
qualify), then lays out the payoff mechanics in a table. This is the reasoning the rubric
rewards.

> **Honest note on what actually ran.** In the *build* run, `TEACHER_MAX_TOKENS` was unset
> → default 4000. Fable is a reasoning model; on a full finance answer it spends 4000 on
> reasoning and returns with `finish_reason: length` and **empty content**. It only
> produces the answer above at ~12k tokens (shown here from a re-capture). So during the
> actual build the teacher's full-answer leg was **empty**, and each frozen lesson was
> distilled by Fable from *(problem + student answer + the graded missed-criteria)* — not
> from a teacher gold answer. The mechanism that produced +5.3 is **"teacher reflects on
> the student's graded gaps,"** not "teacher contrasts its own answer." Fixing the token
> budget so the teacher-answer leg is non-empty was run as a follow-up — and the gaps-only
> lesson still won (§4a). So the empty leg was, if anything, a lucky simplification.
>
> **Code fix applied (2026-07-22).** The default `TEACHER_MAX_TOKENS` in
> `adapters/prbench.py` was raised 4000 → **12000** (both `answer_teacher_alone` and
> `teacher_repair`), and both now route through `_teacher_text()`, which prints a loud
> `WARNING: teacher returned EMPTY content …` when content is empty — so this failure can
> never again pass silently. Root cause in one line: a reasoning model emits hidden
> reasoning tokens before visible content and `max_tokens` caps the total, so a hard
> answer burned the whole 4000 budget on reasoning and hit `finish_reason: length` with
> nothing written; the short distill call (cap 450) survived because compressing supplied
> material needs almost no reasoning.

### 5.6 The distilled lesson (what gets frozen into memory)

Fable turns the gap into a general, entity-scrubbed playbook — no company, ticker, or
dollar figure survives, so it transfers to other Corporate-Finance questions:

```
**LESSON: Stress the Structure, Not Just the Price — a Playbook for Stock-Consideration Renegotiations**

The recurring trap: modeling only the headline value under midpoint scenarios, while ignoring the *mechanics* that determine what you actually receive. Fix it with this sequence:

1. **Map each payoff structure first.** For every alternative, write the payoff function across the full price range — identify where floors/caps bind and explicitly compute the probability the price breaches each threshold. A collar's value lives in its tails.

2. **Sequence the cash waterfall correctly.** Gross proceeds → transaction fees → taxable gain → tax → net. Fees typically apply to gross; tax applies to gain over basis. Check jurisdiction-specific exemptions (e.g., substantial shareholding regimes) before assuming a headline tax rate.

3. **Model risk factors jointly, not independently.** Equity price and FX are often correlated (especially post-shock). State your correlation assumption; test sensitivity to it. Report median, 5th, and 95th percentiles — VaR comparisons drive risk-adjusted choice, not expected value alone.

4. **Interrogate structural terms be
```

This single item is one of the 10 frozen into `prbench_memory.json`. At eval time it (and
the other 9) are prepended to a held-out question's prompt — that is the entire
intervention.

## 6. What this does NOT prove — read before quoting the number

- **Not significance-tested.** +5.3 is a point estimate over **n=12** tasks with per-task
  Δ sd 6.4. No bootstrap CI or p-value computed yet. Directionally strong (8 of 12
  positive), but do not present it as "significant" until the paired bootstrap runs.
- **Ungated.** The uplift gate — the mechanism that keeps only lessons proven to help on
  validation — was **bypassed**. We injected all 10 lessons unconditionally. So this tests
  *"dump every teacher lesson,"* not the fully gated method. Getting the gate live is the
  single biggest lever left.
- **Two genuine regressions** (−2.3, −6.9). Memory *hurt* those tasks. Likely cause:
  retrieval key is category only, and every task here shares the "Corporate Finance"
  category, so the same 10 lessons hit every question with no within-category selectivity.
  A mismatched lesson can mislead. This is precisely what the uplift gate + finer retrieval
  would catch.
- **Regression to the mean.** Interim Δ was +8.6 at 6 tasks, settling to +5.3 at 12. The
  early tasks were favorable; +5.3 is the honest full-set figure.
- **Teacher-answer leg was empty (build bug).** `TEACHER_MAX_TOKENS` defaulted to 4000;
  Fable truncated in reasoning and returned empty content, so lessons were built from the
  student answer + graded gaps only, not a teacher gold answer (§5.5). Not necessarily
  worse — but it means the running method ≠ the method as originally framed.
- **No compute-matched baseline.** We compare against student-alone, not against the
  student given the same extra tokens as self-critique retries. "Memory beats nothing" is
  shown; "memory beats equivalent test-time compute" is not (that arm was dropped by design
  for this run).

## 7. Reproducibility

**Provider discipline.** Student pinned to Fireworks via OpenRouter `provider.order` +
`allow_fallbacks:false`, re-applied every retry. **1 fallback** to Together fired during
the scoring run (logged, loud stderr warning). Pin held for all other calls.

**Determinism knobs.** Split seed 42, dataset SHA-256 pinned. k=3 reps per arm/task,
cached to `prbench_memory_scores.jsonl` (resumable). Judge ≠ teacher ≠ student asserted.
Judge temp 0.

**Artifacts.**

```
fixtures/prbench_corpfin_manifest.json   split 50/15/28, seed 42, sha pinned
runs/prbench_memory.json                 10 frozen contrastive lessons (ungated)
runs/prbench_memory_build.jsonl          per-train-task build log
runs/prbench_memory_scores.jsonl         every A1/A4 cell, k=3 (72 cells)
scripts/prbench_build_memory.py          BUILD phase
scripts/prbench_memory_eval.py           EVAL phase (PLAIN vs MEM)
adapters/prbench.py                      contrastive build_memory_item + firewall
correction/prbench_judge.py              weighted-criteria judge + missed-criteria signal
```

## 8. Bottom line & next

**The mechanism works, within bounds.** A stronger teacher, reading only the student's
graded mistakes, wrote frozen prompt-memory that lifted a cheaper student +5.3 points on
unseen expert finance tasks — no weight updates — closing **28% of the 18.6-pt gap** to the
Fable ceiling (90.5). The lift concentrates where the student was weakest. Simpler memory
(gaps-only) beat the richer three-way contrast (§4a). Two things are already settled: the
token bug is fixed, and gaps-only is the design.

To move from "promising signal" to "defensible claim," in priority order:

1. **Paired bootstrap** over the 12 tasks → CI + p-value on the +5.3. (Fast; can run now.)
2. **Turn the uplift gate on** — the two regressions are the argument for it, and ~13 pts of
   ceiling headroom remain. Gated memory should keep the wins and drop the misleaders.
3. **Add the compute-matched retry arm** so the claim is "beats equivalent inference
   spend," not just "beats nothing."
4. **Extend to the full 28 held-out** for power.
5. **The weights lever (LoRA)** if in-context memory plateaus well short of 90.5 — the
   remaining headroom may need knowledge the prompt can't inject.

*Resolved this cycle: teacher token budget fixed (4k→12k); three-way contrast tested and
rejected in favor of gaps-only (§4a); Fable ceiling measured at 90.5.*

---

*Student deepseek/deepseek-v4-pro (Fireworks-pinned) · Teacher anthropic/claude-fable-5 ·
Judge openai/gpt-5.2 · all OpenRouter. PRBench Corporate Finance · 12 held-out × k=3 ·
ungated · 1 provider fallback · generated 2026-07-21.*
