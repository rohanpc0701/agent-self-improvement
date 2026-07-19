# Findings — Coding Domain (Short-Horizon)

**Updated:** 2026-07-19
**Scope:** everything measured on the coding adapter to date. Numbers are from
real runs; nulls are reported as nulls. Sources: `docs/FABLE_HANDOFF.md` §6,
`docs/superpowers/specs/2026-07-18-diagnostics-ablation-design.md` (results
section), `runs/*.log`.

## TL;DR

1. The self-improvement loop (drift → teacher few-shots + KG → recovery)
   **works end-to-end mechanically**: detection fires, teacher repairs verify,
   memory injects (audit-confirmed).
2. On a **3B student, memory does not transfer** to held-out hard problems —
   the binding constraint is student capacity, proven by ablation + McNemar
   (p ≈ 0.0000 vs a strong model on identical questions).
3. **Teacher-verified ≠ useful-to-student.** Examples reached prompts
   (≈2.9/prompt, 0% zero-injection) and still produced no lift — slight
   (non-significant) harm. Memory quality must be measured by consumer gain,
   not producer correctness (TraceLift-style executor-grounded utility,
   [arXiv:2605.03862](https://arxiv.org/abs/2605.03862)).
4. Student choice is a **band problem**: unaided accuracy must sit ~0.3–0.6 on
   the target pool or the experiment cannot show learning (too weak = floor,
   too strong = ceiling). Measured sweep below.

## Timeline of measurements

### A. Classic `--full` loop, 3B student (earlier)

| | Acc |
|---|---|
| WITHOUT | 0.273 |
| WITH (examples + KG) | 0.455 |
| Δ | **+0.182** |

Single seed; same-distribution recovery stream. This is the loop working when
the held-out questions have same-topic LEARN neighbors and the pool is small.

### B. Hard-curriculum eval, 3B (the null that started diagnostics)

Student WITHOUT 0.111 → WITH 0.111 (**Δ +0.000**); student+KG vs unaided
teacher 0.111 vs 1.000 (gap −0.889). Infrastructure fine, no transfer.

### C. Frozen-memory ablation + injection audit, 3B (2026-07-19)

Hard pool expanded to 75 (35 original + 40 imported from MBPP+/HumanEval+ via
EvalPlus, difficulty-probed at k=2 temp 0.7). n=29 unique held-out hard.

| arm | acc | mean_inj | zero_inj% |
|---|---:|---:|---:|
| none | 0.103 | 0.00 | 100 |
| examples | 0.034 | 2.93 | 0 |
| rules | 0.103 | 0.00 | 100 |
| both | 0.034 | 2.93 | 0 |

- **Injection audit:** examples DO reach prompts. Retrieval plumbing is not
  the failure.
- **McNemar (vs none):** examples −0.069 (2 discordant, p=0.50 — noise-
  compatible, cannot claim harm); rules ±0 (p=1.0).
- **Capacity pair:** `qwen/qwen3-coder` bare 0.931 vs 3B+memory 0.034 —
  26/29 discordant, **p ≈ 0.0000**. Decision-tree row fired: capacity floor.

### D. Student band sweep (2026-07-19, bare `none` arm, n=29)

| model | unaided acc | verdict |
|---|---:|---|
| meta-llama/Llama-3.2-3B-Instruct | 0.103 | too weak (floor) |
| **mistralai/mistral-nemo (12B)** | **0.414** | **in band** |
| **Qwen/Qwen3.5-4B** | **0.552** | **in band** |
| openai/gpt-oss-20b | 0.828 | too strong |
| qwen/qwen3-coder | 0.931 | too strong (ceiling ref) |
| qwen/qwen3-8b | 0.966 | too strong |

Dead Prime IDs (404 despite listing): `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-9B`,
`Qwen/Qwen2.5-Coder-7B-Instruct`.

### E. In-band replication runs — mistral-nemo + Qwen3.5-4B

Full ablate pipeline (own learn phase per student, frozen memory, 4 arms +
capacity leg) launched 2026-07-19; artifacts under `runs/nemo_artifacts/` and
`runs/qwen35-4b_artifacts/`.

**Results: PENDING — append tables here when the chain completes.**

## Operational lessons (cost us real runs)

- Prime rate-limits (429) killed a full pipeline; one throttled call had no
  retry. Fixed: exponential backoff on 429/5xx in student + teacher calls
  (`adapters/coding.py::_chat_with_retry`, commit `63b1ea5`).
- Prime's model listing includes IDs that 404 on chat. Smoke-test a model ID
  with one cheap call before committing a run to it.
- MiniMax direct API ran out of balance silently earlier — teacher failures
  must be loud (already fixed via Prime routing, `correction/provider.py`).
- Deterministic (temp-0) eval + failure-filtered problem selection would make
  the WITHOUT baseline 0.000 by construction — difficulty probes must run at
  temp > 0 (probe used k=2 @ 0.7, keep pass-rate ≤ 0.5).

## What this means for the thesis

The claim "a weak student + teacher memory recovers on new hard work" needs
three preconditions the 3B runs lacked:

1. **Student in the learnable band** (0.3–0.6 unaided) — otherwise floor/ceiling.
2. **Memory gated by measured student gain**, not teacher verification alone.
3. **Enough discordant pairs** for paired stats — pool size and band both feed
   this.

Next milestones (in order): (1) in-band replication (running), (2)
memory-as-model with executor-grounded utility gating, (3) failure-conditioned
ICL if utility-gated memory still shows no lift on an in-band student.
