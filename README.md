# Agent Self-Improvement — Reasoning RSI

[![CI](https://github.com/rohanpc0701/agent-self-improvement/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanpc0701/agent-self-improvement/actions/workflows/ci.yml)

**Can a stronger teacher model make a cheaper student model better at expert reasoning — with no
fine-tuning, no RL, and no weight access?**

The teacher never sees the answer key for the questions we score. It reads the student's *graded
failures* on a training split, writes short transferable lessons, and those lessons are frozen and
prepended to the student's prompt at eval time. The student's weights never change. The only
intervention is text in the context window.

Tasks are expert free-text problems graded by weighted rubrics (an LLM judge decides each criterion
yes/no → normalized 0–100), so "better" is a continuous, rubric-grounded score rather than a vibe.

---

## Result of record

**PRBench Corporate Finance** (Scale AI, arXiv 2511.11562) — 12 held-out tasks × k=3 reps,
per-task averaged. Student `deepseek/deepseek-v4-pro`, teacher `anthropic/claude-fable-5`, judge
`openai/gpt-5.2`, all on OpenRouter.

| Arm | Score | |
|---|--:|---|
| PLAIN — student alone | 71.9 | floor |
| **MEM — student + frozen teacher-built memory** | **77.2** | **Δ +5.3** · 8 W / 2 T / 2 L · per-task sd 6.4 |
| TEACHER — teacher alone | 90.5 | ceiling |

Memory closes **28%** of the 18.6-point student→teacher gap with prompt text alone. The teacher beats
the student on 11/12 tasks and the student never beats it, so the ceiling is real headroom rather than
a weak-teacher artifact.

**What this does not prove** — read before quoting the number:

- **Not significance-tested.** n=12, per-task Δ sd 6.4, no bootstrap CI or p-value computed.
- **Ungated.** The uplift gate (keep only lessons proven to help on validation) was bypassed.
- **Two genuine regressions** (−2.3, −6.9): a mismatched lesson can mislead.
- **No compute-matched baseline** in that run: "memory beats nothing" is shown, "memory beats the
  same tokens spent on self-critique retries" is not.
- **The per-cell scores are gone** (`runs/` is gitignored and the file was deleted), so the headline
  is not currently recomputable. Treat it as a documented prior, not a verified figure.
- **Fewer lessons reach the prompt than the memo implies** — injection is capped at one playbook.

Full memo: [`docs/updates/2026-07-21-prbench-corpfin-cto.md`](docs/updates/2026-07-21-prbench-corpfin-cto.md)
(with a dated Corrections section). Ceiling + design resolution:
[`docs/updates/2026-07-22-prbench-ceiling-update.md`](docs/updates/2026-07-22-prbench-ceiling-update.md).

### The honest counterweight

On **FinancePro-Bench** (400 expert finance questions) the same mechanism produced a single-pass
`+5.6` that **averaged to `+0.0` at k=3** — a false positive caught by the repeat protocol
([`docs/FINDINGS_FINANCE.md`](docs/FINDINGS_FINANCE.md) §H). Earlier coding and GSM8K tracks also
nulled. That is why every delta here carries `n`, `k`, and its caveats, and why the escalation path is
pre-registered in [`docs/STRATEGY_LADDER.md`](docs/STRATEGY_LADDER.md).

---

## Method

```
BUILD (train split)                              EVAL (held-out split)
  student answers task                             PLAIN   student alone
  judge grades vs rubric → missed criteria         MEM     student + frozen memory
  teacher(task, student answer, missed criteria)   REFINE  self-critique → revise ×3
      → ONE transferable lesson, ≤450 tokens,      TEACHER teacher alone (ceiling)
        entity-scrubbed, no numbers or entities
  freeze lessons from lowest-scoring tasks       every arm → judge → normalized 0–100
      → runs/prbench_memory.json                 Δ = MEM − PLAIN per task, averaged over k
```

**Gaps-only wins.** Adding the teacher's own worked answer as a third input dropped the delta to
`+2.5` and doubled the regressions. The teacher does not need to solve the task — only to see where
the student failed. Cheaper, simpler, and not worse.

**Rubric firewall** (enforced in code, covered by tests): the student never sees a rubric; the
teacher sees train-split rubrics only; held-out rubrics are judge-only; memory items are scrubbed of
named entities and checked against rubric stems.

**Reproducibility knobs:** split manifests frozen with seed 42 and a pinned dataset SHA-256; the
student slug pinned to a single OpenRouter provider (`allow_fallbacks=false`, re-applied on retry,
fallbacks counted and reported); judge at temp 0 and asserted to differ from the teacher; every
`(arm, task, rep)` cell cached so runs resume and summaries recompute without API calls.

---

## Quickstart

```bash
make install          # pip3 install -r requirements.txt   (Python ≥ 3.10)
make test             # 195 hermetic tests, no API keys, no network
```

Live runs need `OPENROUTER_API_KEY` in `.env` (gitignored).

```bash
# PRBench Corporate Finance — the active track
python scripts/prbench_freeze_splits.py --check          # verify split + dataset SHA
python scripts/prbench_build_memory.py --n-train 12 --max-items 10
python scripts/prbench_memory_eval.py --k 3 --arms PLAIN,MEM,TEACHER
python scripts/prbench_memory_eval.py --summarize        # re-aggregate from cache, no API

# planner–executor arms: A1 alone, A2 self-refine, A4 teacher hints + refine, A5 teacher
python scripts/prbench_planner_eval.py --dry-run
python scripts/prbench_planner_eval.py --k 3
```

FinancePro-Bench track: `scripts/finance_freeze_splits.py`, `scripts/finance_tracelift.py`
(uplift-gated build), `scripts/finance_baselines.py`, `scripts/finance_eval.py`.

Rebuild a dataset cache (needs `datasets`): `python scripts/prepare_prbench.py`.

---

## Layout

```
adapters/     prbench.py, finance.py — prompts, rubric firewall, teacher calls, memory items
correction/   prbench_judge.py, judge.py (judges) · provider.py (teacher client) · tracelift.py (uplift gate)
harness/      agent.py — student client, retry/backoff, OpenRouter provider pin
scripts/      the runnable experiments (freeze splits → build memory → eval arms)
analysis/     bootstrap.py — paired bootstrap CI + p-value
contracts/    shared Pydantic records
fixtures/     cached datasets + frozen split manifests (committed on purpose)
docs/         findings, dated result memos, plan of record, strategy ladder
docs/archive/ retired Spider-drift / hackathon / v1-coding docs — history only
runs/         gitignored: logs, per-cell scores, frozen memory stores
```

Legacy modules from the retired drift-detection era still live in `correction/` and `harness/`;
[`STRUCTURE.md`](STRUCTURE.md) marks live vs legacy. Start with
[`CLAUDE.md`](CLAUDE.md) — it lists the known gaps so you don't rediscover them.

## Data

- **PRBench** — `ScaleAI/PRBench` finance split, topic *Corporate Finance* (CC-BY). 93 tasks cached.
- **FinancePro-Bench** — [`Sanscritic/finance-pro-bench`](https://huggingface.co/datasets/Sanscritic/finance-pro-bench), 400 questions (CC-BY-4.0).

Rubrics are the benchmark authors' work, cached for evaluation only.

## Author

[Rohan Chavan](https://github.com/rohanpc0701)
