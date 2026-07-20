# Findings — Reasoning Domain (GSM8K)

**Updated:** 2026-07-19  
**Scope:** uplift-gated few-shot memory on GSM8K (TraceLift-*inspired*, not a
TraceLift training replica). Numbers from live OpenRouter runs; nulls reported
as nulls.

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
- Hard-curriculum synthesizes a DriftEvent for GSM8K when the detector does not
  fire (weak easy baseline) so uplift gating still runs on LEARN failures

## Results — 2026-07-19

**Student:** `meta-llama/llama-3.2-3b-instruct` @ OpenRouter  
**Teacher (compare only):** `qwen/qwen3-coder-plus`  
**Logs:** `runs/gsm8k_band_*.log`, `runs/gsm8k_tracelift_*.log`, `runs/gsm8k_ablate_*.log`

### Band check (bare held-out hard, n=30)

| model | unaided hard | verdict |
|---|---:|---|
| qwen/qwen-2.5-7b-instruct | 0.933 | too strong (ceiling) |
| google/gemma-3-4b-it | ~0.80 (n=15) | too strong |
| mistralai/mistral-nemo | ~0.87 (n=15) | too strong |
| **meta-llama/llama-3.2-3b-instruct** | **0.467** | **in band** |
| meta-llama/llama-3.2-1b-instruct | 0.367 | in band (weaker) |

GSM8K is largely saturated by ≥7B instruct models — band requires ~3B class.

### Uplift gate (LEARN val)

Scored 8/24 gold+anchor candidates on 6 LEARN val items (k=1, max_keep=5):

| u | kept? |
|---:|---|
| +0.333 | yes |
| +0.333 | yes |
| +0.167 | yes |
| +0.000 | no |
| −0.333 | no |

**Kept 3 uplift examples** (all `source=uplift`). Gate worked: dropped zero/negative-u items.

### Curriculum headline (single pass; not frozen ablation)

| | acc (n=24 unique hard) |
|---|---:|
| WITHOUT | 0.333 |
| WITH (recovery stream) | 0.417 |
| Δ | **+0.083** |

Student+memory vs unaided teacher (same pass): 0.333 vs 0.958 (gap −0.625).

### Frozen-memory ablation (decisive; same 28 held-out Qs)

| arm | acc | n | mean_inj | zero_inj% |
|---|---:|---:|---:|---:|
| none | 0.321 | 28 | 0.00 | 100 |
| examples (uplift memory) | 0.179 | 28 | 0.82 | **43** |

McNemar examples vs none: Δ = −0.143, b+c=10 (3 improved / 7 regressed), p=0.3438 — not significant, direction is **harm**.

## Interpretation

1. **Uplift gating is mechanically real** — positive-u items selected, negatives dropped, memory size = 3 ≤ 5.
2. **Held-out transfer failed** under frozen ablation. Curriculum +0.083 did not replicate; ablation shows harm.
3. **High zero-injection (43%)** — topic-filtered retrieval often injects 0 of the 3 uplift examples on held-out questions. Val uplift ≠ held-out relevance.
4. **Same lesson as coding, on reasoning:** executor-grounded write policy is necessary but not sufficient; need better cross-problem retrieval (RA-RFT / strategy labels) or a domain where procedural memory transfers (ALFWorld).
5. Deltas near the noise floor still want ≥3 repeats before claiming lift (coding variance protocol).

## Next

1. Fix retrieval so uplift items can inject on dissimilar held-out (or store procedural rules, not only same-topic few-shots).
2. Or move the same uplift+cap write policy to ALFWorld where Memp-style procedural memory is supposed to help.
