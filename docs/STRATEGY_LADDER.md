# Strategy Ladder — what to try if the current bet nulls

**Current bet:** TraceLift (uplift-gated in-context memory) — Fable teacher, DeepSeek-V4
student (Fireworks-pinned), on PRBench Corporate Finance. If this shows no reliable
held-out lift (like coding, GSM8K, and FinancePro before it), escalate in this order.
Each rung is pre-registered so a null triggers a decision, not a debate.

Constraint carried through: **no fine-tuning** (CTO) until/unless the evidence forces it.

| # | Lever | What it fixes | Cost | Notes |
|---|-------|---------------|------|-------|
| **0** | *(current)* TraceLift uplift-gated memory | baseline: does gated teacher memory transfer? | running | Ungated already nulled on FinancePro; the GATE itself is barely tested — finish it first |
| **1** | **Finish + trust the gate** | we've never completed a gate run + gated held-out eval | low | Interim: 5/5 items positive on validation. The real question (validation→held-out transfer) is UNANSWERED. This is rung 0's completion, not a new idea |
| **2** | **Better retrieval (RA-RFT)** | memory injected by topic only (coarse) | med | Retrieve by *predicted trap / reasoning strategy*, not category. PRBench's weighted criteria give explicit failure modes to key on |
| **3** | **Failure-conditioned memory** | generic playbooks don't target the student's actual error | med | Give the student the specific trap it's about to hit (from its own failure pattern), not a general checklist |
| **4** | **Compute-matched retry baseline (RALPH)** | is memory even beating raw test-time compute? | med | A4 must beat best-of-k self-critique retries at equal token budget — else "memory" is just spending inference. Never measured. Honest bar |
| **5** | **Procedure-vs-knowledge diagnostic (OPHSD)** | is the teacher's edge even promptable? | low | Compare uplift on tasks that quote the governing standard vs not. If it's a KNOWLEDGE gap, in-context memory fundamentally can't fix it → forces rung 7 |
| **6** | **Bigger / different student** | capacity floor | med | DeepSeek-V4 is already strong; if it floors on PRBench-Hard, that's capacity, not method |
| **7** | **Weights lever: LoRA / SFT** on teacher repairs | in-context delivery is the bottleneck, not the content | high | The research-backed answer (OPHSD/SIA) if ICL is dead. **Requires lifting the no-fine-tuning constraint** — a CTO decision, justified only after rungs 1-5 null |

## Measurement discipline (applies to every rung)
- ≥3 repeats + averaging before believing any GAP (temp-0 noise is ±10-20 pts).
- Held-out touched once per arm; validation for gating only.
- Report provider-fallback count every run (reproducibility).
- Publish nulls. A trustworthy null beats a noisy false positive (we already caught a
  fake +5.6 on FinancePro that averaged to +0.0).

## The honest meta-read (as of 2026-07-21)
Three domains nulled on ungated in-context memory (coding, GSM8K, FinancePro). The gate
— TraceLift's actual mechanism — has never been fully tested. So the live question is
**rung 1**: does a *completed, gated* run transfer on a real hard benchmark (PRBench)?
If rung 1 nulls, rungs 2-5 are cheap in-context refinements; if those null, rung 7
(weights) is where the evidence points, and that's the CTO conversation.
