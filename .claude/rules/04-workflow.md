# LOCKED RULE: Workflow & git

## Experiments are scripts, runs are resumable
One experiment = one script in `scripts/`, run explicitly with flags. Every `(arm, task, rep)` cell
is cached to a JSONL under `runs/` keyed `"{arm}|{tid}|{rep}"` **before** the next cell starts, so:
- a killed or rate-limited run resumes where it stopped,
- `--summarize` recomputes the table from cache with **zero** API calls,
- a rep that errors is skipped and retried, never silently scored as 0.

Long runs go in a persistent `tmux` session (watchers die, the host survives). Log stdout to
`runs/<name>.log`.

## Never run two writers against one state file
Concurrent runners have overwritten and archived each other's state mid-session
(`docs/FINDINGS_FINANCE.md` §E). One writer per JSONL. If you need a parallel run, give it its own
suffixed path.

## Cost discipline before a live run
1. `--dry-run` (1 task, k=1, printed outputs) to see the shape.
2. Confirm the model slugs, the split, and the arm list.
3. Then the real run with `--k 3`.

A build call on a reasoning teacher needs a large `max_tokens` (~12k): reasoning tokens count
against the same budget, so a small cap returns `finish_reason=length` with **empty content** and no
error. Empty teacher output must warn loudly, never pass silently.

## Tests
`make test` — hermetic, no API keys, no network. Judge/teacher calls are mocked. Run it before and
after any change; a red suite blocks a PR. New behaviour on the live path needs a test that would
fail without it — the rubric firewall and score arithmetic especially.

## Git
- Branch per line of work: `feat/<track>` for experiments, `docs/<topic>` for documentation.
- The current active branch is `feat/finance-tracelift`. `main` is far behind and still contains the
  retired Spider/drift-detection tree — do not treat `main` as the source of truth for method docs.
- Commit small, push often. Never force-push, never commit to `main` directly.
- `contracts/schemas.py` changes get announced before pushing.
- `runs/` and `.env` are gitignored. Never commit an API key or a 50 MB log.

## Docs
- `docs/updates/<date>-<topic>.md` — a dated result memo. **Append corrections, never rewrite
  history**: a memo is what was reported on that date.
- `docs/FINDINGS_*.md` — the running evidence log per track, nulls included.
- `docs/archive/` — retired eras. Read-only history; don't cite as current.
- When code contradicts a doc, fix the doc in the same PR.
