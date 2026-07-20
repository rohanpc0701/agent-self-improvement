# Findings — Reasoning Domain (GSM8K)

**Updated:** 2026-07-19  
**Scope:** uplift-gated few-shot memory on GSM8K (TraceLift-*inspired*, not a TraceLift training replica).

## Thesis under test

Write frozen memory only when an item has **positive causal uplift** for the
student on a LEARN validation slice (`u = pass_with − pass_without`), keep ≤5
items, then measure held-out hard `none` vs `uplift-memory`.

## Implementation (landed)

- Fixture expanded to **250** problems (`scripts/prepare_gsm8k.py`)
- Hard-curriculum + ablation wired for `--adapter gsm8k`
- [`correction/tracelift.py`](../correction/tracelift.py): uplift estimate + selection
- GSM8K injection audit (`build_user_prompt`, `AGENT_USE_EXAMPLES`, `injection_stats`)
- Entrypoint: `bash scripts/use_openrouter_gsm8k.sh {prepare|band|tracelift|ablate}`

## Results

*(Append after live band-check + tracelift run.)*

### Band check

| model | unaided hard (held-out) | verdict |
|---|---:|---|
| TBD | TBD | need ~0.3–0.6 |

### Uplift-gated curriculum

| | |
|---|---|
| candidates scored / kept | TBD |
| WITHOUT (held-out hard) | TBD |
| WITH uplift-memory | TBD |
| Δ | TBD |
| memory size | ≤5 |

## Notes

- Do not claim TraceLift paper replication — this is executor-grounded **memory write** gating.
- Deltas &lt; ~0.06 need ≥3 repeats (see coding FINDINGS variance protocol).
