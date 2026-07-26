# CLAUDE.md — harness/

## What this dir is now
The **student's transport layer**. `agent.py` owns every call the student model makes: client
construction, retry/backoff, and the OpenRouter provider pin. Nothing here decides *what* to ask —
prompt assembly lives in `adapters/`.

Everything else in this directory is retired Spider-era machinery (see `STRUCTURE.md`).

## `agent.py` — what to know before touching it

- **`_chat_with_retry(client, **kwargs)`** is the single chokepoint for student, teacher, and judge
  calls (`correction/prbench_judge.py` and `adapters/*` all route through it). Retries only
  `408/429/5xx` + connection/timeout errors, exponential backoff with jitter, max 5 retries.
- **Provider pin.** For the pinned slug (`OPENROUTER_PIN_MODEL`, default `deepseek/deepseek-v4-pro`)
  it injects `provider: {order, allow_fallbacks: false, require_parameters: true}` into `extra_body`
  on **every attempt** — so a caller that rebuilt `extra_body` (e.g. an empty-content retry) cannot
  silently fall off the pin. Then `_assert_provider`:
  - served by `order[0]` → silent OK,
  - served by another provider in the allow-list → loud stderr warning + `provider_fallback_count()`,
  - anything else, or no `provider` field on the response → `ProviderPinError`, never retried.
  Report `provider_fallback_count()` with every result (`.claude/rules/02-tech-decisions.md`).
- **`ProviderPinError` is not retryable.** Provider drift means the data is from a different serving
  config; abort rather than continue.
- **Fail fast on missing credentials** (`MissingCredentialsError`). Never emit placeholder/error
  answers into a scored run — a fake answer becomes a real 0 in the mean.

### Environment
| Var | Effect |
|---|---|
| `OPENROUTER_API_KEY` (or `PRIME_API_KEY` / `MINIMAX_API_KEY`) | student credentials; picked by base URL |
| `AGENT_BASE_URL` | defaults to MiniMax; set to `https://openrouter.ai/api/v1` for the live tracks |
| `OPENROUTER_PIN_MODEL`, `OPENROUTER_PROVIDER_ORDER`, `OPENROUTER_PROVIDER_QUANT` | pin target, ordered allow-list (primary first), optional precision lock |
| `STUDENT_MAX_TOKENS` | student generation cap (adapters default 6000) |
| `AGENT_ENABLE_THINKING` | unset/`0` sends `reasoning: {enabled: false}` — some reasoning SKUs otherwise return empty `content` |
| `AGENT_TIMEOUT_S` | per-request timeout |

`agent._client` is a module-level cache; scripts set `agent._client = None` between arms so an env
change (base URL, key) actually takes effect. Keep doing that.

## Legacy in this dir
- `feed.py` — `FeedItem` + the change-point stream sampler. The dataclass is still the input type for
  `correction/tracelift.py` and `adapters/finance.run_item`; the sampler is unused.
- `evaluator.py` — SQL execution-accuracy comparison. Imported only by `correction/learner.py`.
- `tests/test_feed_and_eval.py` — pins the above.

Don't extend these. A new benchmark implements `core.adapter.TaskAdapter` in `adapters/`.

## Build/run
No entrypoint here. Experiments live in `scripts/`; see the repo-root `CLAUDE.md`.
