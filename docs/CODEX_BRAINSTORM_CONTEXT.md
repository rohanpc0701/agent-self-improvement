# Codex brainstorm context — full project state (2026-07-29)

One self-contained brief. Everything below is from real runs and committed docs; no invented
numbers. Two repos: `agent-self-improvement` (this one, history + finance TraceLift phase) and
`~/Desktop/Projects/reasoning-rsi` (clean-room rebuild where the question was **settled**,
github `rohanpc0701/reasoning-rsi`).

---

## 1. The research question

> Can a **stronger teacher model** build **frozen, prompt-appended memory** (no fine-tuning, no
> RL, no weight updates) that lifts a **cheaper student model** on **held-out** hard
> domain-reasoning tasks?

"Learning" = a text lesson store grows from the teacher reading the student's graded mistakes;
lessons are prepended to the student's prompt at inference. Constraint throughout: **no
fine-tuning** (CTO-imposed) until in-context options are exhausted.

## 2. Project arc (how we got here)

1. **Hackathon origin (this repo):** drift detection + self-improvement loop for a text-to-SQL
   agent on Spider (few-shot examples learned from failures). Scaffolding only; long superseded.
2. **RSI-Mem v1 (coding):** teacher-built memory for coding students. **Null.** Findings:
   injection mechanics work; ~3B capacity floor; memory bundle 0.000 on nemo (4 repeats),
   −0.265 on Qwen3.5-4B (3 repeats); KG rules inert. Established the **variance protocol**:
   temp-0 nondeterminism is ±10–20 normalized pts per question; deltas need ≥3 repeats averaged.
3. **GSM8K (reasoning):** null.
4. **FinancePro-Bench (RSI-Mem v2):** single-pass +5.6 headline **reversed to +0.0 at k=3** —
   caught a false positive. Null #3.
5. **PRBench Corporate Finance / TraceLift (this repo, `feat/finance-tracelift`):** early n=12
   run showed +5.3 (gaps-only distillation) — later **superseded/unrecomputable** (score cells
   deleted; pipeline had 6 measured defects, see reasoning-rsi `docs/AUDIT.md`).
6. **Clean-room rebuild (`reasoning-rsi`, settled 2026-07-27):** defect-free rig, 144 hermetic
   tests, pre-registered bar. The definitive results below.

## 3. Settled results (2026-07-27, pre-registered, paired bootstrap)

**Setup:** PRBench Corporate Finance (Scale AI, arXiv 2511.11562), weighted-criteria rubric →
LLM judge. Student `deepseek/deepseek-v4-pro`, teacher `anthropic/claude-fable-5`, judge
`openai/gpt-5.2` (judge ≠ teacher ≠ student). Split 50 train / 15 val / 28 held-out, seed 42.
28 held-out × k=3.

### Run 1 — Fireworks-pinned (280 clean cells)

| Arm | Score | Gap closed |
|---|--:|--:|
| PLAIN (student alone) | 73.1 | — |
| + gated teacher memory (13/17 admitted) | 74.5 | 9% |
| + 3 self-critique passes (REFINE) | 79.7 | **41%** |
| Teacher alone (ceiling) | 89.1 | 100% |

- **MEM − PLAIN = +1.39 [−2.17, +4.91], p=0.43 → memory NULL.** Kill criteria K1 + K3 fired.
  The gate's +7.1 validation uplift shrank to +1.5 held-out (selection on noisy k=1 statistic,
  flagged in advance).
- **REFINE − PLAIN = +6.64 [+1.35, +12.10], p=0.014 → the control won.**
- Mechanism as pre-registered: memory recovered −0.1 pts of Financial Accuracy (text can't fix
  arithmetic; also rules out judge-gaming). REFINE recovered +2.0 there.
- Per-task memory deltas are structured, not random: r(Δ, omission-loss)=+0.55,
  r(Δ, PLAIN score)=−0.44. Broadcast injection helps weak-omission drafts, hurts strong ones,
  nets zero. Retrieval had zero selectivity (single-category benchmark).
- Verbosity ruled out: refined answers 1.53× longer but within-arm r(len, score)≈0.
- Power: detecting a true +1.4 at 80% needs ~330 tasks. "Run more" is not a rescue.

### Run 2 — teacher-trace grid, Prime Intellect (504/504 cells, 0 errors)

| Arm | Score | vs REFINE |
|---|--:|--:|
| PLAIN | 77.1 | −0.7 |
| REFINE (generic self-critique) | 77.8 | — (**does NOT replicate on Prime**) |
| CMEM (gated lessons at revise step) | 78.8 | +1.0 n.s. (memory null, 4th delivery form) |
| TCRIT (census-aimed critique) | 80.1 | +2.3 n.s. |
| **HINT (teacher ≤250-word per-task plan)** | **83.8** | **+6.0, p=.001, 22W/6L** |
| **TRACE (full teacher walkthrough)** | **83.9** | **+6.1, p<.001, 20W/7L** |

- TRACE − HINT = +0.07 → **HINT (short per-task plan) is the ship-form**; full walkthrough adds
  nothing.
- Biggest teacher-arm recovery is Financial Accuracy (+3.5/+2.4): a per-task plan names which
  quantities to compute — no frozen general lesson could.
- **Platform flip (first-class finding):** on Fireworks self-critique wins (+6.6 sig) and on
  Prime only teacher guidance does (+6 sig); REFINE = +0.7 n.s. on Prime. Same model slugs,
  different serving stack → the winning intervention is **stack-conditional**. Deployment
  recipes must be validated per platform.
- Prime TEACHER ceiling arm not yet run (~$8).

### The bottom line

- **Frozen/broadcast teacher memory is dead: null in four delivery forms across two platforms
  and four domains** (coding, GSM8K, FinancePro, PRBench).
- **What works: per-task guidance.** Product shape = **planner–executor** — frontier model
  writes a short plan per task; cheap model executes and self-revises.
- The remaining ~9-pt gap (refined student → teacher) is arithmetic + knowledge that no prompt
  text reached → **weights lever (LoRA/SFT) is the CTO conversation**, now backed by data.
- Total cost of knowing: ~$20 (run 1) + ~$75–85 (grid). The defect-free rig is the durable asset.

## 4. Open questions / queued experiments (each behind a pre-flight)

From `reasoning-rsi/docs/ideation/2026-07-27-post-null-next-steps.md` (26 ideas → 7 survivors):

1. **Targeted self-critique from the failure census** (~$8) — census: 94% of lost points are
   *omitted required criteria* (risk disclosures, committing to a recommendation, quantifying
   uncertainty); traps only 2.6%. Can aimed critique beat 41% of the gap? (TCRIT +2.3 n.s. on
   Prime was directional.)
2. ~~Decision memo~~ — shipped (`reasoning-rsi/docs/DECISION_MEMO.md`, PR #1).
3. **REFINE dose curve** (1/2/3/5 passes + best-of-3 control, ~$12) — the cost knob; separates
   revision from sampling diversity.
4. **MEM × REFINE factorial with draft-conditional injection** (~$8) — the one memory variant
   still alive: decide *after a first draft* whether the task needs the lesson.
5. **Harden the positive claim** — judge test–retest noise floor (~$1), length-score correlation
   ($0), style-only revision control (~$3).
6. **Per-task variance mining → failure-conditioned injection** ($0 to mine) — do the ±20-swing
   winners share a signature? If not, memory closes for good.
7. **Replicate on a second PRBench topic** (~$8) — generality.
8. (new, post-grid) **Prime TEACHER ceiling** (~$8) — completes the ladder.

Strategy ladder if everything nulls (`docs/STRATEGY_LADDER.md`): better retrieval →
failure-conditioned memory → compute-matched retry baseline → procedure-vs-knowledge diagnostic
→ bigger student → **rung 7: LoRA/SFT on teacher repairs (needs CTO to lift the no-fine-tuning
constraint)**. Rungs 0–1 are now done (nulled); the evidence points at guidance + weights.

## 5. Measurement discipline (non-negotiable, hard-won)

- ≥3 repeats averaged before believing any delta; temp-0 noise is ±10–20 pts per question.
  (This caught two false positives: FinancePro +5.6→0.0, and would have caught +5.3.)
- Held-out touched once per arm; validation only for gating. Pre-register bars + mechanism
  predictions before runs; a null is a decision, not a debate. Publish nulls.
- Judge ≠ teacher ≠ student, always asserted. Student provider-pinned
  (`allow_fallbacks=false`); report fallback count each run.
- Reasoning models: `max_tokens` caps reasoning+content together — low caps silently return
  EMPTY content (`finish_reason=length`). Cost one real bug (teacher answers silently empty).
  Budget ~12k tokens; warn loudly on empty.
- Gaps-only distillation (teacher sees question + student answer + missed criteria, NOT its own
  solution) is not-worse than three-way contrast — design lesson only.

## 6. Working protocols (Rohan's standing rules)

- **Pre-flight approval:** before ANY run spending API $ or mutating state, write a 3-section
  plan (plain-English top line / exact commands + cost / outcomes-decision table) and WAIT for
  explicit approval. High-stakes, stakeholder-scrutinized project.
- **Budgets are fallbacks:** warn + continue at cap; hard stop only at 4×. Never let a budget
  interrupt planned spend.
- **Model split:** Fable 5 plans/decides/brainstorms; Opus 5 (coding tier) builds.
- **No RALPH/Zenith citations** in docs unless directly relevant. No AI co-author trailers on
  commits.
- Long runs: DigitalOcean VM in persistent tmux (`bash -lc`, not bare `source`); verify by
  output file growth, not pgrep. VM key was pasted in chat — rotate/delete droplet when done.

## 7. Infrastructure

- **reasoning-rsi** (settled work): `results/cells.jsonl` + `results/prime/cells.jsonl`
  committed — every number regenerates via `python scripts/run_eval.py --summarize`, no API.
  Key docs: `DECISION_MEMO.md`, `RESULTS_PRIME_GRID.md`, `AUDIT.md` (6 predecessor defects),
  `ROADMAP.md` (pre-registration record), `METHOD.md`, `preflights/`.
- **agent-self-improvement** (this repo): history + the pre-rebuild TraceLift pipeline.
  Key docs: `docs/STRATEGY_LADDER.md`, `docs/FABLE_THINK_PRBENCH.md`, `docs/FINDINGS_FINANCE.md`
  (§H = the k=3 reversal), `docs/FINDINGS_CODING.md`, `docs/RSI_MEM_V2_FINANCE.md`.
- Models used across phases: students qwen3-8b / qwen3.6-27b / deepseek-v4-pro; teachers
  minimax-m3 / glm-5.2 / claude-fable-5; judge gpt-5.2. Platforms: OpenRouter
  (Fireworks-pinned), Prime Intellect.

## 8. What Codex should brainstorm on

The live decision space:

- **Exploit the guidance win:** planner–executor productization — when/how to route which tasks
  to a HINT call; can hints be cached/clustered so the frontier model isn't in every hot path?
- **Close the memory line honestly:** draft-conditional injection (survivor 4/6) — the only
  variant the data leaves alive.
- **The platform flip:** why does serving stack change which intervention works on identical
  slugs? Diagnostic worth designing?
- **The weights conversation:** what evidence package justifies lifting the no-fine-tuning
  constraint; LoRA on teacher repairs vs on HINT-guided trajectories.
- **Generality:** second topic, second student tier, cross-domain hint transfer.
- Anything new must respect: pre-flight protocol, ≥k=3, pre-registered bars, held-out hygiene,
  no fine-tuning without CTO sign-off.
