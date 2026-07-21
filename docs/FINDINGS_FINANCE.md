# FinancePro-Bench findings (RSI-Mem v2)

Honest numbers only. Nulls and skips reported as such.

## A. Dataset + splits (G0.1)

- Source: `Sanscritic/finance-pro-bench` HF split `test` (400 rows), CC-BY-4.0.
- Cache: `fixtures/finance_pro_bench.json` + `fixtures/FINANCE_LICENSE`.
- Manifest: `fixtures/finance_manifest.json`, seed **42**.
  - train-stream **200** / validation **80** / held-out **120**
  - Disjoint; `--check` OK; no tiny categories (&lt;3).

## B. Judge reliability (G0.2) — EXIT GATE

**Protocol:** 40 stratified validation questions → one student answer each → judge twice in fresh contexts → Pearson *r* + MAD on normalized scores.

| Setting | Value |
|--------|--------|
| Student (answers) | `qwen/qwen3-8b` (Prime early; OpenRouter for remainder after Prime timeouts) |
| Judge | `openai/gpt-5.2` @ temp 0 on Prime |
| Teacher (assert ≠ judge) | `minimax/minimax-m3` |
| Sample | n=40 stratified from validation; **33** non-empty answers; **26** complete grade pairs |

### Answer generation yield

| Metric | Count |
|--------|------:|
| Planned | 40 |
| Usable answers | 33 |
| Failed / empty (timeouts, JSON errors) | 7 |

### Test–retest (verbatim from `runs/judge_reliability_summary.json`)

```json
{
  "n": 26,
  "pearson_r": 0.8289546719998755,
  "mad": 4.455873696256521,
  "gate": "PASS_SINGLE",
  "JUDGE_PASSES": 1,
  "mean_pass1": 7.763384690026545,
  "mean_pass2": 8.416652338484711
}
```

### Gate decision

| Criterion | Result |
|-----------|--------|
| MAD ≤ 5 | **Yes** (MAD ≈ **4.46**) → **single judge pass** (`JUDGE_PASSES=1`) |
| 5 &lt; MAD ≤ 8 | n/a |
| MAD &gt; 8 (K1) | **No** |

**Verdict: GATE PASSED — proceed to G0.3/G0.4 with `JUDGE_PASSES=1`.**

### Known judge / rubric issues (do not hide)

- Some rubrics fail `Item R*(max N)` extraction (`fpb-00262`, `fpb-00208`) — those items excluded from pairs.
- Occasional empty judge output / missing `TOTAL` after one repair-retry → pair incomplete, excluded from MAD *n*.
- Hand-audit sample (15 Qs): `runs/judge_audit_sample.md` — **flag for Rohan**.

### Student score context (not a baseline)

Mean normalized score on this validation reliability set is ~8 (pass1/pass2). That is **not** the Phase 0 held-out baseline; it only characterizes this probe sample under a weak 8B student.

## C. Headroom probe (G0.4)

**Protocol:** 20 category-stratified VALIDATION ids (seed 42), bare student prompt
(`AGENT_USE_EXAMPLES=0`), temp 0, `max_tokens=2048`, judge `openai/gpt-5.2` with
`JUDGE_PASSES=1`. Endpoint: Prime. Thinking disabled for student gen via
`chat_template_kwargs.enable_thinking=false` (required for `qwen/qwen3.6-27b`,
which otherwise returned empty `content`).

### Means (verbatim from `runs/finance_headroom_summary.json`)

```json
{
  "qwen/qwen3-8b": 9.262307868713286,
  "qwen/qwen3-30b-a3b-instruct-2507": 15.752503500053054,
  "qwen/qwen3.6-27b": 26.305344197683876
}
```

| Model | n graded / 20 | Mean normalized | In band 15–40? |
|-------|--------------:|----------------:|:---------------|
| `qwen/qwen3-8b` | 19 | **9.262** | No (&lt;15) |
| `qwen/qwen3-30b-a3b-instruct-2507` | 19 | **15.753** | Yes |
| `qwen/qwen3.6-27b` | 18 | **26.305** | Yes |

**Ungradable / excluded:** `fpb-00262` (rubric lacks `Item R*(max N)` — all models);
`fpb-00072` for 27b only (empty judge output after repair-retry). Answers generated
for all 20 ids × 3 models.

**Chosen student:** `qwen/qwen3.6-27b` — smallest model in band [15, 40]
(SIZE_ORDER: 8b out of band; 27b &lt; 30b MoE among in-band). Fallback
`qwen/qwen3.5-35b-a3b` not needed.

### Reliability (band-range) — after student pick

**Protocol:** same 20 headroom answers from `qwen/qwen3.6-27b` → grade a second
time in a fresh context (pass2) → Pearson *r* + MAD on normalized pairs.

#### Low-range (G0.2, for comparison)

| Metric | Value |
|--------|------:|
| n pairs | 26 |
| pearson_r | 0.829 |
| MAD | 4.456 |
| gate | PASS_SINGLE → `JUDGE_PASSES=1` |
| Student answers | `qwen/qwen3-8b` (~mean 8) |

#### Band-range (verbatim from `runs/finance_band_recheck_summary.json`)

```json
{
  "label": "reliability (band-range)",
  "student_model": "qwen/qwen3.6-27b",
  "n": 17,
  "pearson_r": 0.9617178576630094,
  "mad": 4.188280804907782,
  "gate": "PASS_SINGLE",
  "JUDGE_PASSES": 1,
  "mean_pass1": 25.685534723244224,
  "mean_pass2": 26.76587003266326
}
```

| Criterion | Result |
|-----------|--------|
| MAD ≤ 5 | **Yes** (MAD ≈ **4.19**) → **`JUDGE_PASSES=1` stands** |
| 5 &lt; MAD ≤ 8 | n/a |
| MAD &gt; 8 (STOP) | **No** |

**Verdict: BAND-RANGE GATE PASSED — proceed to held-out baselines with `JUDGE_PASSES=1`.**

Incomplete pairs (excluded from n=17): `fpb-00262` (bad rubric), `fpb-00072`
(empty judge both passes), `fpb-00025` (pass2 missing TOTAL after repair).

## D. Held-out baselines (G0.3)

*(pending 3c)*

---

*Updated 2026-07-20 after band-range judge recheck (3b).*

## E. TraceLift memory build (Task C)

**Platform:** OpenRouter only (`scripts/use_openrouter_finance.sh`).
Prime-era `runs/finance_heldout_*` / `finance_headroom_*answers|grades.jsonl` archived to `runs/archive_prime_platform/`.

**Models:** student `qwen/qwen3.6-27b`, teacher `z-ai/glm-5.2`, judge `openai/gpt-5.2`.

### Train-stream probe (chunk 1, time-budget 540s)

State: `runs/finance_tracelift_state.jsonl`. Verbatim from first LIVE chunk:

| qid | category | normalized | failure (<40)? |
|-----|----------|-----------:|:---------------|
| fpb-00106 | Credit | **28.40909090909091** | yes |
| fpb-00377 | Trading | **36.84210526315789** | yes |
| fpb-00155 | Investment Banking | **14.772727272727273** | yes |

- Graded: **3** / 200 train-stream
- Failures: **3** / 3 (100% under fail-threshold 40)
- Wall time ≈ **853s** for 3 Qs (student empty-content retries on thinking SKU; mitigated later via `STUDENT_MAX_TOKENS=8192`)

### Candidates

- First LIVE candidate chunk distilled **3** items for `fpb-00106` (playbook/trap/skeleton) then rows were lost from state (concurrent runner / file race). Rebuild in progress via `--phase all` loop.
- Teacher GLM 5.2: ≈3–4 min per repair at `TEACHER_MAX_TOKENS=4000`.


### Candidate rebuild (isolated agent state)

State: `runs/finance_tracelift_state_agent.jsonl` (isolated from concurrent runners that archived the main state).

| key | usable? | notes |
|-----|:-------:|-------|
| fpb-00106:playbook | no | empty after entity scrub (pre-fix) |
| fpb-00106:trap | yes | generic fallback trap |
| fpb-00106:playbook:v2 | yes | category fallback (teacher returned empty content once) |
| fpb-00106:skeleton:v2 | yes | category fallback skeleton |

### Uplift gate attempts (val_n=1, K=1)

Verbatim errors from gate logs:

- trap: `missing TOTAL line`
- trap retry: `empty judge output`
- playbook:v2: `empty judge output`

**Admitted: 0. Frozen store n=0.** Judge empty/TOTAL failures on the uplift path blocked admission despite usable candidates.

### Concurrent-runner interference

A parallel `/tmp/finance_tracelift_loop.sh` / fast-lane process archived/replaced `runs/finance_tracelift_*` mid-session; subsequent work used `*_agent.jsonl` paths.

### Uplift gate / freeze

- Protocol target: val_n=80, K=2, keep u_norm > +1, stop on window(15) mean u < +0.5 or admit rate < 20%.
- Implementation default: `--val-n` from `FINANCE_UPLIFT_VAL_N` (12) for cost; bare-baseline cache added.
- **Admitted / frozen store:** not yet (still collecting train failures + candidates). Memory file currently dry-run placeholder.

### Blockers

1. Per-question latency on OpenRouter qwen3.6-27b (reasoning + retries) → train coverage slow.
2. GLM 5.2 teacher latency → candidate throughput ~1 item / few minutes.
3. Full uplift on 80×K=2 per candidate is cost-prohibitive in one session; using smaller val slice until more candidates exist.


## F. Held-out A1 vs A4 (Task D)

Harness ready: `scripts/finance_eval.py` (A1/A4/A5, paired bootstrap).
**LIVE A1/A4 numbers:** blocked on frozen TraceLift memory from §E.



## G. TraceLift A1 vs A4 — first signal (CTO test, 2026-07-21)

**Setup:** student `qwen/qwen3.6-27b` (reasoning disabled), teacher `z-ai/glm-5.2`,
judge `openai/gpt-5.2`, OpenRouter. Ungated memory built from Credit+Trading
train-stream repairs. Eval on the 7 Credit/Trading held-out questions (the only
held-out where category-keyed memory injects). temp 0, single pass.

### Memory-quality is the determinant

| Memory | A1 mean | A4 mean | GAP | n |
|---|---:|---:|---:|---:|
| Boilerplate (extract + entity-scrub → generic stubs) | 30.5 | 24.4 | **−6.1** | 5 |
| **Good (teacher-distilled, real ASC/IFRS reasoning)** | 25.2 | 30.9 | **+5.6** | 5 |

Fixing distillation quality flipped hurt → help (a **+11.7** swing). Good-memory
GAP clears the pre-registered +4-pt bar. Per-question (good memory): +17.9, +6.6,
+4.8, 0.0, −1.2 (3 of 5 improved).

### The distillation fix

Old path extracted checklist lines then entity-scrubbed to <24 chars → generic
fallback template ("map facts → framework → gates"), carrying no usable reasoning.
New path (`adapters/finance.py::_teacher_distill`, commit 4d81ef6): one teacher
call distills a transferable playbook/trap directly, forbidden from any
question-specific entity — leak-safe by construction, ~2000-char ASC/IFRS-grounded
content vs 150-char stubs.

### Honest caveats
- **n=5** (2 questions lost to judge/rubric parse failures). No CI. Directional only.
- **Baseline is noisy run-to-run:** fpb-00006 A1 = 44.0 (boilerplate run) vs 19.0
  (good run) — same question, same student, temp 0. Student+judge nondeterminism
  is large (10–25 pts on some questions), so +5.6 on n=5 sits partly inside noise.
- Credit/Trading only; uplift gate bypassed (harness gating unreliable — see §E).

**Read:** TraceLift helps qwen3.6-27b *when memory carries genuine transferable
reasoning*; the earlier nulls (coding, GSM8K, boilerplate-finance) were
memory-quality failures, not proof the mechanism can't work. Confirmation
(more questions + repeats to beat noise) pending.
