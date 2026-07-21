# The Method, Step by Step — How Memory Is Made and Used

Plain walkthrough of exactly what "TraceLift on FinancePro" does. Three models,
one loop. No fine-tuning — the student's weights never change; the only thing
that changes is text added to its prompt.

---

## The three models (roles)

| Role | Model | Job |
|---|---|---|
| **Student** | Qwen 3.6 27B | Answers finance questions. This is the model we're trying to improve. Weights frozen. |
| **Teacher** | GLM 5.2 | Stronger model. Fixes the student's failed answers and writes reusable lessons from them. |
| **Judge** | GPT-5.2 | Grades any answer against the official point rubric → a 0–100 score. Different model family from the teacher so it doesn't favor the teacher's style. |

## The data (frozen before anything runs)

400 finance questions split, seeded, once:
- **TRAIN (200):** where memory is BUILT. Student fails here, teacher repairs, lessons are extracted.
- **VALIDATION (80):** where memory is TESTED during building (the uplift gate).
- **HELD-OUT (120):** where the final A1-vs-A4 comparison happens. Touched once, never used to build or tune memory.

Rule enforced in code: the student **never** sees a grading rubric. The teacher sees rubrics only for TRAIN questions.

---

## PART 1 — HOW MEMORY IS CREATED (on the TRAIN set)

### Step 1: Student attempts a TRAIN question (no memory yet)
The bare student answers a training question. Example prompt it gets:

```
[system] You are an expert finance professional. Answer with clear, structured
reasoning. Cite standards and show steps. Produce the substantive answer only.
[user]  <the finance question, 3–10k chars>
```

It produces an answer. The judge grades it against that question's rubric → a score.

### Step 2: Detect a failure
If the score is below a threshold, this question is a **failure** — a case worth
learning from. (On finance the student fails most hard questions, so failures are plentiful.)

### Step 3: Teacher repairs the failed answer
The teacher (GLM 5.2) is given the question + the student's broken answer + **the
TRAIN rubric** (it's allowed to see train rubrics), and writes a corrected analysis.
Because it has the rubric, "teacher + rubric" produces a genuinely strong repair.

### Step 4: Teacher DISTILLS the repair into a reusable memory item
This is the key step — and the one we had to fix. We do **not** store the raw
repaired answer (too long, too specific to that one question). Instead, a second
teacher call turns the repair into a **transferable lesson**. Three kinds:

- **playbook** — a reusable step-by-step analytical checklist for that category
- **trap** — one named error to avoid ("TRAP: <trigger> → <what to do instead>")
- **skeleton** — a compact worked-reasoning template (Issue → Framework → steps → Conclusion)

The distillation prompt forbids any specifics — no company names, tickers,
dollar amounts, or details unique to the source question. So the lesson is
**leak-safe by construction** and **general** (transfers to OTHER questions in
the category).

**Real example of a distilled playbook** (category = Credit, `source=tracelift`):

```
Category playbook (Credit):
1. Identify the reporting entity and consolidation scope first. Determine which
   entity holds the reporting obligation, then apply the applicable consolidation
   standard (ASC 810 / IFRS 10). Assess whether the entity is a VIE...
2. Map the legal structure against the accounting substance. Separate legal
   isolation (bankruptcy-remote SPV, true-sale opinions) from consolidation
   outcomes — distinct analyses run in parallel, not conflated.
3. Trace the cash flow waterfall...
```

Contrast: the **old broken distillation** produced generic boilerplate like
"map facts → governing framework → ordered gates" (150 chars, no real content).
That version actively *hurt* the student. Fixing it (teacher writes the lesson
directly instead of regex-extracting + entity-scrubbing) is what made the memory
carry real reasoning.

Each distilled item is stored as a small record:
```json
{ "question": "[FINANCE_PLAYBOOK] Credit",   // a category-keyed tag, NOT the train question
  "correct_output": "Category playbook (Credit): 1. Identify the reporting entity...",
  "domain_id": "Credit",                      // category = retrieval key
  "source": "tracelift" }
```

### Step 5: The uplift gate (the TraceLift core — see note)
For each candidate item, measure whether it actually helps:
```
u = mean student score WITH the item  −  mean student score WITHOUT the item
    (on VALIDATION questions, same frozen student)
```
Keep the item only if `u > +threshold`. Drop the rest. This is a *measurement*,
not training — same frozen student, just inference + a comparison.

> **IMPORTANT honesty note:** in the finance runs this gate never completed
> (harness bugs). We **bypassed it** and kept ALL distilled items ("ungated
> memory"). So the results test "dump all teacher lessons," not the real
> gated method. Getting the gate working is the biggest open lever.

### Step 6: Freeze the memory
The surviving items become a frozen store (`runs/finance_memory_good.json`, 10
items for Credit + Trading). Nothing about the student changed — we just built a
small library of category lessons.

---

## PART 2 — HOW MEMORY IS APPLIED (on the HELD-OUT set)

### Step 7: Retrieve items for the current question
The held-out question has a category (e.g. "Credit"). We pull memory items whose
`domain_id` matches that category — capped at ~4 (1 playbook + ≤2 traps + ≤1
skeleton). **Category is the only retrieval key** (a known limitation — coarse).
If the question's category has no memory, nothing is injected and the answer is
identical to bare.

### Step 8: Build the student prompt WITH memory
The retrieved lessons are prepended to the question. The student sees:

```
[system] You are an expert finance professional. Answer with clear, structured
reasoning. Cite standards and show steps. Produce the substantive answer only.
[user]
Category memory (TraceLift):

Memory 1 [playbook]:
[FINANCE_PLAYBOOK] Credit
Category playbook (Credit): 1. Identify the reporting entity and consolidation
scope first... 2. Map legal structure against accounting substance... 3. ...

Memory 2 [trap]:
[FINANCE_TRAP] Credit
TRAP: applying safe-harbor without aggregating the decision-maker's other
interests → always list related-party interests first.

<the held-out finance question>
```

That's the entire mechanism — **the memory is just text prepended to the
prompt.** Same model, same weights, same question. The only variable is whether
those "Category memory" lines are present.

### Step 9: Grade and compare
- **A1 (student alone):** answer the held-out question with NO memory → judge → score.
- **A4 (student + memory):** answer the SAME question WITH the memory block → judge → score.
- **GAP = A4 − A1.** Positive = memory helped. This is measured per question and averaged.

Because temp-0 scores are noisy (±10–20 pts run-to-run), we run each arm **3
times and average** before trusting any GAP.

---

## The whole loop in one picture

```
TRAIN set:
  student answers ─▶ judge scores ─▶ [low score = failure]
        │
        ▼
  teacher repairs the failure (has train rubric)
        │
        ▼
  teacher distills repair ─▶ playbook / trap / skeleton  (general, no specifics)
        │
        ▼
  uplift gate: keep item only if it raises student score on VALIDATION   ← (bypassed in practice)
        │
        ▼
  FROZEN MEMORY (small library of category lessons)

HELD-OUT set:
  question ─▶ retrieve same-category memory ─▶ prepend to prompt ─▶ student answers
                                                                        │
   bare student answers the same question (A1) ──────────────┐        (A4)
                                                              ▼          ▼
                                                     judge scores both → GAP = A4 − A1
```

## What we found (see docs/FINDINGS_FINANCE.md)
- Bad memory (boilerplate) HURT: GAP −6.1.
- Good memory (real reasoning), single pass: GAP +5.6 — but that was noise.
- Good memory, 3-repeat averaged: **GAP +0.0** — a clean null.
- Conclusion: on this student, in-context memory doesn't reliably help — BUT the
  actual uplift gate (Step 5) was never tested, so the real gated method is still
  an open question.

## Where each step lives in the code
- Steps 1, 7, 8 (student gen + retrieval + injection): `adapters/finance.py`
  (`generate_answer`, `select_category_memory`, `build_student_prompt`).
- Steps 3, 4 (teacher repair + distill): `adapters/finance.py`
  (`teacher_repair`, `_teacher_distill`, `distill_memory_item`).
- Step 5 (uplift gate): `correction/tracelift.py` (`estimate_uplift`, `select_uplift_memory`).
- Step 9 (grading): `correction/judge.py` (`grade`).
- Orchestration: `scripts/finance_tracelift.py` (build), `scripts/finance_eval.py` (A1/A4).
