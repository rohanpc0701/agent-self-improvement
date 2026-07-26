# CLAUDE.md — correction/

## What this dir is now
The **teacher and judge side** of the loop: grade an answer against a rubric, hand the teacher the
graded gaps, and decide which resulting lesson is allowed into frozen memory.

Live modules: `prbench_judge.py`, `judge.py`, `provider.py`, `tracelift.py`. Everything else is
retired drift-detection machinery (see `STRUCTURE.md`).

## `prbench_judge.py` — the scoring contract
```
positive criterion satisfied     → + weight
detrimental criterion committed  → + weight (negative)
raw = Σ applied weights ;  max = Σ positive weights
score = clamp(raw, 0..max) / max × 100
```
- Judge prompt asks for exactly one `C<n>: yes|no` line per criterion, no prose, temp 0. The answer
  is wrapped in `<answer>` tags and declared untrusted (prompt-injection guard) — keep that.
- `grade()` asserts the judge slug ≠ `TEACHER_MODEL`. Never remove that assert.
- `score_from_decisions` also returns **`missed`** = required-but-unsatisfied + detrimental-committed.
  That list is the *only* thing the teacher gets to reason about when writing a lesson (gaps-only).
- **Known sharp edge:** a *partial* parse is accepted (repair-retry fires only when zero decisions
  parse), and unparsed criteria count as unsatisfied → a deflated score. `n_decided` is returned but
  not checked. Fixing this means deciding what to do with a partially graded cell — don't paper over it.

`judge.py` is the FinancePro equivalent (`Item R*(max N)` rubric format, `JUDGE_PASSES` averaging,
repair-retry on a missing `TOTAL`).

## `provider.py` — teacher client
Resolution order: explicit `TEACHER_BASE_URL` + `TEACHER_API_KEY` → OpenRouter
(`TEACHER_USE_OPENROUTER=1`) → Prime (`TEACHER_USE_PRIME=1`) → MiniMax. Raises if nothing resolves.
The live tracks use OpenRouter for all three roles.

**Reasoning-teacher trap:** hidden reasoning tokens count against `max_tokens`, so a small cap
returns `finish_reason=length` with **empty content and no error**. `adapters/prbench.py` defaults
`TEACHER_MAX_TOKENS` to 12000 and routes teacher calls through `_teacher_text()`, which warns loudly
on empty content. Any new teacher call must do the same.

## `tracelift.py` — the uplift gate
A candidate lesson is admitted only if it measurably helps the **frozen** student on a validation
slice: `u = mean(score | candidate) − mean(score | no memory)` over `val_items × k`, keep `u > min_u`,
rank by `u`, cap the store. This is the mechanism the whole method rests on — "ungated" results are a
weaker claim and must be labelled as such.

Current state: the gate takes an adapter with `run_item(item, config)` returning a record with
`execution_accuracy`. `adapters/finance.py` implements that; **`adapters/prbench.py` does not**, and
`scripts/finance_tracelift.py` carries its own judge-scored gate. So "turn the gate on for PRBench"
is new plumbing, not a flag.

## Legacy in this dir
`graph.py` + `store.py` (knowledge-graph `(trap, fix)` rules), `inject.py` (still imported by
`harness/agent.py`), `learner.py` + `teacher.py` (SQL-era few-shots and anchoring), `correction.py`,
`on_drift.py`, `memory.py`, `repair.py`, `distill.py`, `contracts.py`. Tests in `correction/tests/`
pin them. Don't build new work on these; don't delete them as a drive-by.

## Build/run
```bash
make test                                    # includes correction/tests
python scripts/prbench_build_memory.py --n-train 12    # judge + teacher, live
```
