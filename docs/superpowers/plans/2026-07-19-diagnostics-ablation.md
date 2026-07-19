# Diagnostics Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument and run a frozen-memory ablation (examples / rules / both / none + 7B capacity probe) with an injection audit, on a hard pool expanded to ≥60 problems via MBPP+/HumanEval+ import.

**Architecture:** Extend the existing coding adapter + orchestrator. A pure prompt-assembly helper returns injection stats alongside the prompt; `TelemetryRecord` gains an optional `injection_stats` field (additive contract bump). A new import script fetches EvalPlus problems, validates golds in the sandbox, probes difficulty with the 3B student, and appends `h2_*` hard problems to the fixture. A new `--ablation` orchestrator mode replays held-out questions under each arm against the same frozen memory.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, `evalplus` (new dev-time dependency, Apache-2.0), OpenAI-compatible client on Prime Inference.

**Spec:** `docs/superpowers/specs/2026-07-18-diagnostics-ablation-design.md`

## Global Constraints

- Run everything from repo root; imports assume repo root on path.
- `contracts/schemas.py` is the single shared contract: the ONLY change allowed is the additive `injection_stats: dict | None = None` field (Task 2). Announce before pushing.
- All tests hermetic — no live API calls in pytest. Existing ~235 tests must stay green: `python3 -m pytest -q`.
- New fixture problem IDs are prefixed `h2_`. Existing IDs never change.
- Eval arms run at `temperature=0.0`; the difficulty probe runs at `temperature=0.7`, k=2.
- Student default `meta-llama/Llama-3.2-3B-Instruct`; capacity probe `Qwen/Qwen2.5-Coder-7B-Instruct` (fallback `meta-llama/Llama-3.1-8B-Instruct`), overridden via `PRIME_AGENT_MODEL`.
- Commit after every task. No `Co-authored-by: Cursor` trailers. Author: Rohan Chavan `<rohanpc@vt.edu>`.

---

### Task 1: Prompt assembly with `AGENT_USE_EXAMPLES` flag + injection stats

Extract prompt building from `generate_code` into a pure, testable helper that also reports what was injected. Add the `AGENT_USE_EXAMPLES` env flag and a `temperature` parameter.

**Files:**
- Modify: `adapters/coding.py` (functions `_examples_block`, `generate_code`)
- Test: `tests/test_coding_adapter.py`

**Interfaces:**
- Produces: `build_user_prompt(question: str, config: AgentConfig, topic: str, use_rules: bool) -> tuple[str, dict]` — returns `(user_prompt, injection_stats)` where `injection_stats = {"examples_available": int, "examples_injected": int, "example_ids": list[str], "rules_injected": int}`.
- Produces: `generate_code(question, config, topic="arrays", use_rules=True, temperature=0.0) -> tuple[str, int, float, str, dict]` — 5-tuple; the new last element is `injection_stats`. (Task 2 updates the caller `CodingAdapter.run_item`.)
- `AGENT_USE_EXAMPLES` env var (default `"1"`): when `"0"`, no examples are injected and `examples_injected == 0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_coding_adapter.py`:

```python
class TestPromptAssembly:
    def _config(self):
        from contracts.schemas import AgentConfig, FewShotExample
        return AgentConfig(
            config_id="t",
            model="m",
            few_shot_examples=[
                FewShotExample(question="q1", correct_output="def a(): pass", domain_id="dp"),
                FewShotExample(question="q2", correct_output="def b(): pass", domain_id="graphs"),
            ],
        )

    def test_injects_same_topic_examples_with_stats(self, monkeypatch):
        from adapters.coding import build_user_prompt
        monkeypatch.delenv("AGENT_USE_EXAMPLES", raising=False)
        prompt, stats = build_user_prompt("solve it", self._config(), topic="dp", use_rules=False)
        assert "q1" in prompt and "q2" not in prompt
        assert stats["examples_available"] == 2
        assert stats["examples_injected"] == 1
        assert stats["example_ids"] == ["q1"]
        assert stats["rules_injected"] == 0

    def test_examples_flag_off(self, monkeypatch):
        from adapters.coding import build_user_prompt
        monkeypatch.setenv("AGENT_USE_EXAMPLES", "0")
        prompt, stats = build_user_prompt("solve it", self._config(), topic="dp", use_rules=False)
        assert "q1" not in prompt
        assert stats["examples_injected"] == 0
        assert stats["examples_available"] == 2

    def test_rules_counted(self, tmp_path, monkeypatch):
        import correction.graph as g
        from correction.graph import add_rule
        from adapters.coding import build_user_prompt
        monkeypatch.setattr(g, "_STORE_PATH", tmp_path / "graph_store.json")
        add_rule("dp", "off-by-one in base case", "start dp table at index 0")
        monkeypatch.delenv("AGENT_USE_EXAMPLES", raising=False)
        _, stats = build_user_prompt("solve it", self._config(), topic="dp", use_rules=True)
        assert stats["rules_injected"] >= 1
```

Note: check `correction/graph.py` for the real rule-adding function name and store-path attribute (`_STORE_PATH` is used by existing tests at `tests/test_coding_adapter.py:55`); mirror the existing `test_write_graph_rules_fallback` setup if `add_rule` has a different name.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_coding_adapter.py::TestPromptAssembly -v`
Expected: FAIL — `ImportError: cannot import name 'build_user_prompt'`

- [ ] **Step 3: Implement `build_user_prompt` and rewire `generate_code`**

In `adapters/coding.py`, replace `_examples_block` usage with:

```python
def _use_examples() -> bool:
    return os.environ.get("AGENT_USE_EXAMPLES", "1") != "0"


def build_user_prompt(
    question: str,
    config: AgentConfig,
    topic: str,
    use_rules: bool,
) -> tuple[str, dict]:
    """Assemble the student prompt; report exactly what memory entered it."""
    stats = {
        "examples_available": len(config.few_shot_examples),
        "examples_injected": 0,
        "example_ids": [],
        "rules_injected": 0,
    }
    parts: list[str] = []

    if _use_examples():
        same = [
            e for e in config.few_shot_examples
            if not e.domain_id or e.domain_id == topic
        ][:3]
        if same:
            lines = ["Similar solved problems:"]
            for ex in same:
                lines.append(
                    f"Problem: {ex.question}\n```python\n{ex.correct_output}\n```"
                )
            parts.append("\n\n".join(lines) + "\n\n")
            stats["examples_injected"] = len(same)
            stats["example_ids"] = [ex.question for ex in same]

    if use_rules:
        rules = _rules_block(topic, question)
        if rules:
            parts.append(rules + "\n\n")
            stats["rules_injected"] = rules.count("\n- ") or 1

    parts.append(f"Problem:\n{question}\n")
    return "".join(parts), stats
```

(`example_ids` uses the question text — `FewShotExample` has no id field; do not add one.)

For `rules_injected`: open `correction/inject.py` and count rules from `build_context(topic, question)` directly (e.g. `len(ctx.rules)` or equivalent) instead of string-counting, if the context object exposes a list. Prefer the structured count; the `count("\n- ")` fallback above is only if the block is opaque text.

Then rewire `generate_code` (keep the same client/messages logic, currently at `adapters/coding.py:94-121`):

```python
def generate_code(
    question: str,
    config: AgentConfig,
    topic: str = "arrays",
    use_rules: bool = True,
    temperature: float = 0.0,
) -> tuple[str, int, float, str, dict]:
    client = agent._get_client()
    user, stats = build_user_prompt(question, config, topic, use_rules)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = (resp.choices[0].message.content or "").strip()
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens, latency_ms, "", stats
```

Delete `_examples_block` and update any other caller (`grep -n "_examples_block\|generate_code(" adapters/ harness/ orchestrator.py tests/`). `CodingAdapter.run_item` unpacks the old 4-tuple at `adapters/coding.py:279` — update it minimally now (`text, tokens, latency_ms, reasoning, _stats = generate_code(...)`); Task 2 wires the stats into telemetry.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_coding_adapter.py -v`
Expected: PASS (new + existing)

- [ ] **Step 5: Full suite**

Run: `python3 -m pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add adapters/coding.py tests/test_coding_adapter.py
git commit -m "Add AGENT_USE_EXAMPLES flag and injection stats to prompt assembly"
```

---

### Task 2: `TelemetryRecord.injection_stats` (additive contract bump)

**Files:**
- Modify: `contracts/schemas.py` (class `TelemetryRecord`, ends near line 96)
- Modify: `adapters/coding.py` (`CodingAdapter.run_item`, near line 272)
- Test: `tests/test_contracts.py` (or the existing schema test file — find with `grep -rln "TelemetryRecord" tests/`)

**Interfaces:**
- Consumes: `generate_code(...) -> (text, tokens, latency_ms, reasoning, stats)` from Task 1.
- Produces: `TelemetryRecord.injection_stats: dict | None` (default `None`). Later tasks read it from recovery-phase records.

- [ ] **Step 1: Write the failing tests**

```python
def test_injection_stats_roundtrip(tmp_path):
    from contracts.schemas import TelemetryRecord
    rec = TelemetryRecord(
        run_id="r1", timestamp=1.0, difficulty="hard",
        execution_accuracy=1.0, query_valid=True,
        injection_stats={"examples_available": 3, "examples_injected": 2,
                         "example_ids": ["a", "b"], "rules_injected": 1},
    )
    assert TelemetryRecord.model_validate(rec.model_dump()).injection_stats[
        "examples_injected"] == 2


def test_legacy_record_without_injection_stats_parses():
    from contracts.schemas import TelemetryRecord
    rec = TelemetryRecord(
        run_id="r2", timestamp=1.0, difficulty="easy",
        execution_accuracy=0.0, query_valid=False,
    )
    assert rec.injection_stats is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest -k injection_stats -v`
Expected: FAIL — unexpected keyword / attribute error

- [ ] **Step 3: Implement**

In `contracts/schemas.py`, add to `TelemetryRecord` after `reasoning: str = ""`:

```python
    # Diagnostics: what memory actually entered this run's prompt (None = not recorded)
    injection_stats: dict | None = None
```

In `adapters/coding.py` `run_item`, unpack and pass through:

```python
        text, tokens, latency_ms, reasoning, stats = generate_code(
            item.question, config, topic=item.domain_id, use_rules=use_rules
        )
```

and add `injection_stats=stats,` to the `TelemetryRecord(...)` construction (near line 284).

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -q`
Expected: all pass (legacy events load because the field defaults to None)

- [ ] **Step 5: Commit**

```bash
git add contracts/schemas.py adapters/coding.py tests/
git commit -m "Add optional injection_stats to TelemetryRecord (additive contract bump)"
```

**Announce in team chat before pushing:** "TelemetryRecord gained optional `injection_stats` (default None) — additive, old events.jsonl unaffected, re-pull contracts."

---

### Task 3: Import script — fetch + validate stage

`scripts/import_problems.py fetch` pulls MBPP+ and HumanEval+ via the `evalplus` package, converts to fixture schema, validates every gold in the sandbox, and writes candidates to `fixtures/imported_candidates.json`.

**Files:**
- Create: `scripts/import_problems.py`
- Test: `tests/test_import_problems.py`
- Modify: `requirements.txt` (add `evalplus`)

**Interfaces:**
- Produces: `convert_evalplus_item(item: dict, source: str) -> dict | None` — returns a fixture-schema problem dict (`id, question, function_name, tests, topic, difficulty, gold_solution`) or `None` if unusable.
- Produces: `label_topic(question: str, code: str) -> str` — keyword heuristic returning one of `dp|arrays|graphs|strings|greedy|arithmetic`.
- Produces: `fixtures/imported_candidates.json` — list of fixture-schema dicts with extra key `"source": "mbpp+"|"humaneval+"`, `difficulty` provisionally `"hard"` (finalized by Task 4's probe).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_import_problems.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.import_problems import convert_evalplus_item, label_topic


FAKE_HE_ITEM = {
    "task_id": "HumanEval/0",
    "entry_point": "has_close_elements",
    "prompt": (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ Check if in given list of numbers, are any two numbers closer '
        'to each other than given threshold. """\n'
    ),
    "canonical_solution": (
        "    for i, a in enumerate(numbers):\n"
        "        for j, b in enumerate(numbers):\n"
        "            if i != j and abs(a - b) < threshold:\n"
        "                return True\n"
        "    return False\n"
    ),
    "base_input": [[[1.0, 2.0, 3.0], 0.5], [[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3]],
}


class TestConvert:
    def test_maps_schema(self):
        p = convert_evalplus_item(FAKE_HE_ITEM, source="humaneval+")
        assert p["id"] == "h2_humaneval_0"
        assert p["function_name"] == "has_close_elements"
        assert p["source"] == "humaneval+"
        assert p["difficulty"] == "hard"
        assert len(p["tests"]) == 2
        # expected computed by executing the gold on each input
        assert p["tests"][0] == {"args": [[1.0, 2.0, 3.0], 0.5], "expected": False}
        assert p["tests"][1]["expected"] is True
        # question is the natural-language docstring, not raw code
        assert "closer to each other" in p["question"]
        # gold is a complete runnable definition
        assert p["gold_solution"].startswith("from typing import List")

    def test_rejects_non_json_io(self):
        bad = dict(FAKE_HE_ITEM, base_input=[[{1, 2, 3}]])  # set: not JSON-safe
        assert convert_evalplus_item(bad, source="humaneval+") is None


class TestLabelTopic:
    def test_keywords(self):
        assert label_topic("find the longest common subsequence", "") == "dp"
        assert label_topic("reverse the given string", "") == "strings"
        assert label_topic("shortest path between nodes in a graph", "") == "graphs"
        assert label_topic("compute the sum of two numbers", "") == "arithmetic"
        # default bucket
        assert label_topic("do the thing", "") == "arrays"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_import_problems.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.import_problems'`

- [ ] **Step 3: Implement the script**

Create `scripts/import_problems.py` (also create an empty `scripts/__init__.py` if imports fail):

```python
#!/usr/bin/env python3
"""Import MBPP+/HumanEval+ problems into the coding fixture.

Stages:
  fetch  — pull via evalplus, convert, sandbox-validate golds
           → fixtures/imported_candidates.json
  probe  — 3B student k=2 @ temp 0.7 per candidate; pass-rate <= 0.5 → hard;
           append survivors to fixtures/coding_subset.json   (Task 4)

Licenses: MBPP+ (Apache-2.0), HumanEval+ (MIT) via the evalplus package.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.sandbox import execution_accuracy  # noqa: E402

CANDIDATES_PATH = ROOT / "fixtures" / "imported_candidates.json"
FIXTURE_PATH = ROOT / "fixtures" / "coding_subset.json"

_TOPIC_KEYWORDS = [
    ("dp", ["subsequence", "knapsack", "dynamic", "climb", "coin", "partition",
            "edit distance", "longest increasing", "minimum cost", "ways to"]),
    ("graphs", ["graph", "node", "edge", "tree", "path", "bfs", "dfs",
                "island", "connected"]),
    ("strings", ["string", "palindrome", "substring", "anagram", "character",
                 "vowel", "letter", "word"]),
    ("greedy", ["greedy", "interval", "minimum number of", "maximum profit",
                "jump", "schedule"]),
    ("arithmetic", ["sum of two", "digit", "prime", "factorial", "gcd", "lcm",
                    "power", "arithmetic", "multiply", "divide"]),
]


def label_topic(question: str, code: str) -> str:
    text = f"{question} {code}".lower()
    for topic, kws in _TOPIC_KEYWORDS:
        if any(kw in text for kw in kws):
            return topic
    return "arrays"


def _json_roundtrip_safe(value) -> bool:
    try:
        return json.loads(json.dumps(value)) == value
    except (TypeError, ValueError):
        return False


def _docstring_question(prompt: str, entry_point: str) -> str:
    """Extract the natural-language docstring of entry_point as the question."""
    try:
        tree = ast.parse(prompt + "    pass\n")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                doc = ast.get_docstring(node)
                if doc:
                    # strip doctest-style examples; keep prose
                    return re.split(r"\n\s*>>>", doc)[0].strip()
    except SyntaxError:
        pass
    return prompt.strip()


def convert_evalplus_item(item: dict, source: str) -> dict | None:
    entry = item["entry_point"]
    gold = item["prompt"] + item["canonical_solution"]
    inputs = item.get("base_input") or []
    if not inputs:
        return None

    tests = []
    for args in inputs[:8]:  # cap tests per problem
        if not _json_roundtrip_safe(args):
            return None
        acc_ns: dict = {}
        try:
            exec(gold, acc_ns)  # trusted benchmark gold, local machine only
            expected = acc_ns[entry](*[json.loads(json.dumps(a)) for a in args])
        except Exception:
            return None
        if not _json_roundtrip_safe(expected):
            return None
        tests.append({"args": args, "expected": expected})
    if len(tests) < 3:
        return None

    slug = re.sub(r"\W+", "_", item["task_id"].lower()).strip("_")
    question = _docstring_question(item["prompt"], entry)
    return {
        "id": f"h2_{slug}",
        "question": question,
        "function_name": entry,
        "tests": tests,
        "topic": label_topic(question, gold),
        "difficulty": "hard",  # provisional; probe stage finalizes
        "gold_solution": gold.rstrip() + "\n",
        "source": source,
    }


def fetch() -> None:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    existing_names = set()
    for p in json.loads(FIXTURE_PATH.read_text()):
        existing_names.add(p["function_name"])

    candidates: list[dict] = []
    for source, loader in (("humaneval+", get_human_eval_plus),
                           ("mbpp+", get_mbpp_plus)):
        for item in loader().values():
            p = convert_evalplus_item(item, source=source)
            if p is None or p["function_name"] in existing_names:
                continue
            # sandbox-validate the gold against the converted tests
            acc, _, err = execution_accuracy(
                p["gold_solution"], p["function_name"], p["tests"]
            )
            if acc != 1.0:
                print(f"  drop {p['id']}: gold fails converted tests ({err!r})")
                continue
            candidates.append(p)

    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=1))
    from collections import Counter
    print(f"{len(candidates)} candidates → {CANDIDATES_PATH}")
    print("topics:", dict(Counter(c["topic"] for c in candidates)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["fetch", "probe"])
    ap.add_argument("--k", type=int, default=2, help="probe samples per problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-keep", type=int, default=40,
                    help="cap on new hard problems appended to the fixture")
    args = ap.parse_args()
    if args.stage == "fetch":
        fetch()
    else:
        from scripts.import_problems_probe import probe  # Task 4
        probe(k=args.k, temperature=args.temperature, max_keep=args.max_keep)


if __name__ == "__main__":
    main()
```

Add `evalplus` to `requirements.txt`.

Note on `exec` of gold solutions: these are trusted, widely-used benchmark solutions executed locally at import time only; runtime scoring still goes through `harness/sandbox.py`. Keep the `exec` in the import script only — never move it into the harness.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_import_problems.py -v`
Expected: PASS (tests exercise conversion only — no network, no evalplus import)

- [ ] **Step 5: Run the real fetch once**

Run: `pip3 install evalplus && python3 scripts/import_problems.py fetch`
Expected: prints candidate count (hundreds) + topic histogram; creates `fixtures/imported_candidates.json`. If the count is < 100, inspect drop reasons before continuing.

Topic-label sanity check: if the default `arrays` bucket holds > 50% of candidates, the keyword heuristic is under-labeling — topics drive the retrieval filter, so mislabels corrupt the diagnosis. In that case extend `_TOPIC_KEYWORDS`, or label the `arrays` leftovers with one cheap teacher batch call (spec §1) before proceeding.

- [ ] **Step 6: Commit**

```bash
git add scripts/import_problems.py tests/test_import_problems.py requirements.txt scripts/__init__.py
git commit -m "Add EvalPlus import script: fetch, convert, sandbox-validate golds"
```

(`fixtures/imported_candidates.json` is an intermediate — add it to `.gitignore` instead of committing.)

---

### Task 4: Import script — difficulty probe + fixture merge

Probe each candidate with the live 3B student (k=2 @ temp 0.7, bare prompt), keep pass-rate ≤ 0.5 as hard, append to `fixtures/coding_subset.json`.

**Files:**
- Create: `scripts/import_problems_probe.py`
- Test: `tests/test_import_problems.py` (extend)

**Interfaces:**
- Consumes: `fixtures/imported_candidates.json` from Task 3; `generate_code(question, config, topic, use_rules, temperature)` from Task 1; `verify_solution(output, problem)` from `adapters/coding.py`.
- Produces: `select_hard(results: list[tuple[dict, float]], max_keep: int) -> list[dict]` — pure filter, pass-rate ≤ 0.5, topic-balanced up to `max_keep`.
- Produces: updated `fixtures/coding_subset.json` with appended `h2_*` problems (`difficulty: "hard"`, `source` key retained), plus `fixtures/coding_subset.backup.json` copy of the pre-merge file.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_import_problems.py`:

```python
from scripts.import_problems_probe import select_hard


def _cand(i, topic):
    return {"id": f"h2_c{i}", "question": "q", "function_name": "f",
            "tests": [], "topic": topic, "difficulty": "hard",
            "gold_solution": "def f(): pass", "source": "mbpp+"}


class TestSelectHard:
    def test_filters_by_pass_rate(self):
        results = [(_cand(1, "dp"), 0.0), (_cand(2, "dp"), 0.5),
                   (_cand(3, "dp"), 1.0)]
        kept = select_hard(results, max_keep=10)
        assert [c["id"] for c in kept] == ["h2_c1", "h2_c2"]

    def test_topic_balance_under_cap(self):
        results = [(_cand(i, "dp"), 0.0) for i in range(10)]
        results += [(_cand(100 + i, "graphs"), 0.0) for i in range(2)]
        kept = select_hard(results, max_keep=6)
        topics = [c["topic"] for c in kept]
        assert len(kept) == 6
        assert topics.count("graphs") == 2  # scarce topic fully kept
        assert topics.count("dp") == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_import_problems.py -v`
Expected: FAIL — no module `import_problems_probe`

- [ ] **Step 3: Implement**

Create `scripts/import_problems_probe.py`:

```python
#!/usr/bin/env python3
"""Difficulty probe: live 3B student, k samples @ temp>0; keep pass-rate <= 0.5.

Probing at temperature 0.7 (not 0.0) keeps the temp-0 eval baseline honest:
a problem filtered on a deterministic temp-0 failure would make the WITHOUT
arm 0.000 by construction.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.import_problems import CANDIDATES_PATH, FIXTURE_PATH  # noqa: E402


def select_hard(results: list[tuple[dict, float]], max_keep: int) -> list[dict]:
    """Keep candidates with probe pass-rate <= 0.5, topic-balanced up to max_keep."""
    hard = [(c, r) for c, r in results if r <= 0.5]
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for c, _ in hard:
        by_topic[c["topic"]].append(c)

    kept: list[dict] = []
    # round-robin across topics so scarce topics survive the cap
    pools = sorted(by_topic.items(), key=lambda kv: len(kv[1]))
    while len(kept) < max_keep and any(pool for _, pool in pools):
        for _, pool in pools:
            if pool and len(kept) < max_keep:
                kept.append(pool.pop(0))
    return kept


def probe(k: int = 2, temperature: float = 0.7, max_keep: int = 40) -> None:
    from adapters.coding import generate_code, verify_solution
    from orchestrator import _make_base_config

    candidates = json.loads(CANDIDATES_PATH.read_text())
    config = _make_base_config("difficulty-probe")  # empty few-shots
    results: list[tuple[dict, float]] = []

    for i, p in enumerate(candidates, 1):
        passes = 0
        for _ in range(k):
            text, *_rest = generate_code(
                p["question"], config, topic=p["topic"],
                use_rules=False, temperature=temperature,
            )
            acc, _, _ = verify_solution(text, p)
            passes += int(acc == 1.0)
        rate = passes / k
        results.append((p, rate))
        print(f"  [{i}/{len(candidates)}] {p['id']} pass-rate={rate:.1f}", flush=True)

    kept = select_hard(results, max_keep=max_keep)

    shutil.copy(FIXTURE_PATH, FIXTURE_PATH.with_suffix(".backup.json"))
    fixture = json.loads(FIXTURE_PATH.read_text())
    existing = {p["id"] for p in fixture}
    new = [c for c in kept if c["id"] not in existing]
    fixture.extend(new)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=1))

    from collections import Counter
    n_hard = sum(1 for p in fixture if p["difficulty"] in ("hard", "extra"))
    print(f"\nappended {len(new)} hard problems → {FIXTURE_PATH}")
    print(f"hard pool now: {n_hard}")
    print("new topics:", dict(Counter(c["topic"] for c in new)))
```

Set `AGENT_USE_EXAMPLES=1` irrelevant here — config has no examples; `use_rules=False` keeps the probe bare.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_import_problems.py -v`
Expected: PASS

- [ ] **Step 5: Run the live probe (needs PRIME_API_KEY, costs ~2×candidates cheap 3B calls)**

```bash
set -a; source .env; set +a
export AGENT_BASE_URL="https://api.pinference.ai/api/v1"
export AGENT_MODEL="meta-llama/Llama-3.2-3B-Instruct"
python3 scripts/import_problems.py probe --max-keep 40
```

Expected: fixture hard pool ≥ 60 (`python3 -c "import json;print(sum(1 for p in json.load(open('fixtures/coding_subset.json')) if p['difficulty'] in ('hard','extra')))"`). If < 60, raise `--max-keep` or note Stage 2 (LCB) trigger in the run log.

Then confirm the fixture-wide gold validation still passes:
`python3 -m pytest tests/test_coding_adapter.py -k gold -v` → PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/import_problems_probe.py tests/test_import_problems.py fixtures/coding_subset.json .gitignore
git commit -m "Probe imported problems with 3B student; append hard pool to fixture"
```

(Do NOT commit `coding_subset.backup.json` or `imported_candidates.json` — gitignore both.)

---

### Task 5: 30/30 topic-stratified held-out split

Pass `db_heldout_frac` through the coding adapter and size the split so ~30 unique hard questions land in held-out.

**Files:**
- Modify: `adapters/coding.py` (`build_hard_curriculum_feed`, line ~256)
- Modify: `orchestrator.py` (`run_hard_curriculum_eval` call site, ~line 953; `_build_parser` gains `--heldout-frac`)
- Test: `tests/test_coding_adapter.py` (extend `TestFeed`-style class near `test_phase_counts_and_heldout_disjoint`, line 131)

**Interfaces:**
- Consumes: `build_hard_curriculum_stream(questions, n_baseline, n_learn, n_heldout, seed, db_heldout_frac)` — already accepts `db_heldout_frac` (`harness/feed.py:147`).
- Produces: `CodingAdapter.build_hard_curriculum_feed(seed, n_baseline=40, n_learn=100, n_heldout=40, db_heldout_frac=0.5)`.
- Produces: orchestrator flag `--heldout-frac` (default 0.5) forwarded to the adapter.

- [ ] **Step 1: Write the failing test**

```python
    def test_heldout_frac_controls_unique_split(self):
        from adapters.registry import get_adapter
        a = get_adapter("coding")
        items = a.build_hard_curriculum_feed(seed=42, db_heldout_frac=0.5)
        learn_ids = {i.question_id for i in items if i.phase == "degraded"}
        held_ids = {i.question_id for i in items if i.phase == "recovery"}
        assert learn_ids.isdisjoint(held_ids)
        # with a ~60-problem hard pool, 0.5 frac → ~half unique Qs held out
        assert len(held_ids) >= 20
        # deterministic given seed
        items2 = a.build_hard_curriculum_feed(seed=42, db_heldout_frac=0.5)
        assert {i.question_id for i in items2 if i.phase == "recovery"} == held_ids
```

(Adjust the import to the real adapter accessor — see `orchestrator._get_adapter` at line 392 for the canonical way.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_coding_adapter.py -k heldout_frac -v`
Expected: FAIL — unexpected keyword `db_heldout_frac`

- [ ] **Step 3: Implement**

`adapters/coding.py`:

```python
    def build_hard_curriculum_feed(
        self,
        seed: int,
        n_baseline: int = 40,
        n_learn: int = 100,
        n_heldout: int = 40,
        db_heldout_frac: float = 0.5,
    ) -> list[FeedItem]:
        """Easy warmup (detector only) → hard LEARN → held-out hard eval."""
        return build_hard_curriculum_stream(
            self.load_questions(),
            n_baseline=n_baseline,
            n_learn=n_learn,
            n_heldout=n_heldout,
            seed=seed,
            db_heldout_frac=db_heldout_frac,
        )
```

`orchestrator.py`: add `--heldout-frac` (type float, default 0.5) in `_build_parser`, thread it into `run_hard_curriculum_eval(...)` and the `adapter.build_hard_curriculum_feed(...)` call. Also raise the held-out replay: with ~30 unique held-out Qs, keep `n_heldout` (stream length) ≥ 30 — pass `--n-heldout 30` at run time; no code change needed for that.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_coding_adapter.py tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add adapters/coding.py orchestrator.py tests/test_coding_adapter.py
git commit -m "Thread db_heldout_frac through coding curriculum feed for 30/30 split"
```

---

### Task 6: Orchestrator `--ablation` mode

Replay held-out questions under each arm against the frozen memory in `events.jsonl`, print per-arm accuracy, McNemar pairs, and the injection audit.

**Files:**
- Modify: `orchestrator.py` (new function `run_ablation_eval` after `compare_teacher_run` ~line 271; `_build_parser` gains `--ablation`; `main` dispatch)
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: latest `CorrectionAction` from `read_events(only="correction")` (same guard as `compare_teacher_run`, orchestrator.py:160-183); `_run_item(item, config, adapter_name, use_rules)`; `_mcnemar_report(label, pairs)`; `TelemetryRecord.injection_stats`.
- Produces: `run_ablation_eval(items: list[FeedItem], adapter_name: str, arms: list[str]) -> dict[str, dict]` — returns `{arm: {"acc": float, "n": int, "zero_injection_pct": float, "mean_examples": float, "per_q": {question_id: float}}}` (returned for tests; printed for humans).
- Produces: CLI `--ablation none,examples,rules,both` (comma list; `all` = those four).
- Arm semantics: `none` = examples off + rules off · `examples` = examples on, rules off · `rules` = examples off, rules on · `both` = both on. Examples toggled via `AGENT_USE_EXAMPLES` env; rules via the existing `use_rules` parameter.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py` (mirror the file's existing mocking style for `_run_item` / adapter — check its imports first):

```python
class TestAblationArms:
    def test_arm_env_matrix(self, monkeypatch):
        import orchestrator as orch

        calls = []

        def fake_run_item(item, config, adapter_name="coding", use_rules=True):
            import os
            from contracts.schemas import TelemetryRecord, Difficulty
            calls.append((os.environ.get("AGENT_USE_EXAMPLES"), use_rules))
            return TelemetryRecord(
                run_id=f"{item.question_id}_x", timestamp=0.0,
                difficulty=Difficulty(item.difficulty),
                execution_accuracy=1.0, query_valid=True,
                injection_stats={"examples_available": 2, "examples_injected": 0,
                                 "example_ids": [], "rules_injected": 0},
            )

        monkeypatch.setattr(orch, "_run_item", fake_run_item)
        monkeypatch.setattr(
            orch, "_load_latest_examples", lambda: [], raising=False
        )

        from harness.feed import FeedItem
        items = [FeedItem(question_id="q1", question="?", difficulty="hard",
                          domain_id="dp", phase="recovery")]

        result = orch.run_ablation_eval(items, "coding", ["none", "both"])

        assert ("0", False) in calls   # none arm
        assert ("1", True) in calls    # both arm
        assert result["none"]["n"] == 1
        assert result["none"]["zero_injection_pct"] == 100.0
```

(Adapt `FeedItem` construction to its real signature — check `harness/feed.py` imports/fields; if a helper builds items in existing orchestrator tests, reuse it.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_orchestrator.py -k ablation -v`
Expected: FAIL — `run_ablation_eval` missing

- [ ] **Step 3: Implement `run_ablation_eval`**

Add to `orchestrator.py` after `compare_teacher_run`:

```python
_ABLATION_ARMS = {
    # arm: (AGENT_USE_EXAMPLES value, use_rules)
    "none": ("0", False),
    "examples": ("1", False),
    "rules": ("0", True),
    "both": ("1", True),
}


def _load_latest_examples():
    from contracts.eventlog import read_events

    corrections = read_events(only="correction")
    if not corrections:
        print(
            "[ablation] No correction in events.jsonl — run a learn phase first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return corrections[-1].new_few_shot_examples


def run_ablation_eval(
    items: list, adapter_name: str, arms: list[str]
) -> dict[str, dict]:
    """Frozen-memory ablation: each arm replays the same unique held-out
    hard questions with a different memory channel enabled."""
    import os

    examples = _load_latest_examples()
    config = _make_base_config("ablation").model_copy(
        update={"few_shot_examples": examples}
    )

    held = [it for it in items if it.phase == "recovery"]
    seen: set[str] = set()
    pool = []
    for it in held:
        if it.difficulty == "hard" and it.question_id not in seen:
            pool.append(it)
            seen.add(it.question_id)

    print(f"\n[ablation] arms={arms} on {len(pool)} unique held-out hard Qs "
          f"({len(examples)} examples in frozen memory)", flush=True)

    results: dict[str, dict] = {}
    prev_flag = os.environ.get("AGENT_USE_EXAMPLES")
    try:
        for arm in arms:
            flag, use_rules = _ABLATION_ARMS[arm]
            os.environ["AGENT_USE_EXAMPLES"] = flag
            per_q: dict[str, float] = {}
            zero_inj = 0
            inj_counts: list[int] = []
            for i, item in enumerate(pool, 1):
                rec = _run_item(item, config, adapter_name, use_rules=use_rules)
                if rec is None:
                    continue
                append_event(rec)
                per_q[item.question_id] = rec.execution_accuracy
                stats = rec.injection_stats or {}
                n_inj = stats.get("examples_injected", 0)
                inj_counts.append(n_inj)
                if flag == "1" and n_inj == 0:
                    zero_inj += 1
                print(f"  [{arm:<8}] [{i:>2}/{len(pool)}] "
                      f"{'✓' if rec.execution_accuracy == 1.0 else '✗'} "
                      f"inj={n_inj} {item.question[:40]}", flush=True)
            n = len(per_q)
            acc = sum(per_q.values()) / n if n else 0.0
            results[arm] = {
                "acc": acc,
                "n": n,
                "per_q": per_q,
                "mean_examples": sum(inj_counts) / n if n else 0.0,
                "zero_injection_pct": 100.0 * (
                    (zero_inj / n) if flag == "1" and n else (1.0 if n else 0.0)
                ),
            }
    finally:
        if prev_flag is None:
            os.environ.pop("AGENT_USE_EXAMPLES", None)
        else:
            os.environ["AGENT_USE_EXAMPLES"] = prev_flag

    # ---- report ----
    print(f"\n{'=' * 60}")
    print("  ABLATION (frozen memory, same held-out hard Qs)")
    for arm, r in results.items():
        print(f"    {arm:<9}: acc={r['acc']:.3f}  n={r['n']}  "
              f"mean_inj={r['mean_examples']:.2f}  "
              f"zero_inj={r['zero_injection_pct']:.0f}%")
    print("=" * 60, flush=True)

    if "none" in results:
        base = results["none"]["per_q"]
        for arm in ("examples", "rules", "both"):
            if arm in results:
                pairs = [
                    (base[q], results[arm]["per_q"][q])
                    for q in base
                    if q in results[arm]["per_q"]
                ]
                _mcnemar_report(f"{arm} vs none", pairs)
    return results
```

Wire the CLI in `_build_parser` / `main`:

```python
    p.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Comma list of arms (none,examples,rules,both) or 'all'. "
        "Requires a prior learn phase in events.jsonl.",
    )
```

and in `main`, before the other modes:

```python
    if args.ablation:
        arms = (
            ["none", "examples", "rules", "both"]
            if args.ablation == "all"
            else [a.strip() for a in args.ablation.split(",")]
        )
        unknown = set(arms) - set(_ABLATION_ARMS)
        if unknown:
            sys.exit(f"unknown ablation arms: {sorted(unknown)}")
        items = adapter.build_hard_curriculum_feed(
            seed=_SEED, n_heldout=args.n_heldout,
            db_heldout_frac=args.heldout_frac,
        )
        run_ablation_eval(items, args.adapter, arms)
        return
```

(Match the real `main` structure — find where `--compare-teacher` dispatches and mirror it, including how `adapter`/`args.n_heldout` are already resolved there.)

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Full suite + commit**

```bash
python3 -m pytest -q
git add orchestrator.py tests/test_orchestrator.py
git commit -m "Add frozen-memory ablation mode with injection audit and McNemar pairs"
```

---

### Task 7: `ablate` entrypoint in the Prime script

One command: learn phase → 3B ablation (4 arms) → 7B capacity probe (none + both).

**Files:**
- Modify: `scripts/use_prime_student.sh` (add `ablate` case; extend usage line)

**Interfaces:**
- Consumes: `--hard-curriculum`, `--ablation`, `--heldout-frac`, `--n-heldout` from Tasks 5–6; env contract (`AGENT_MODEL`, `PRIME_CAPACITY_MODEL`).

- [ ] **Step 1: Add the case**

Insert before the `*)` case in `scripts/use_prime_student.sh`:

```bash
  ablate)
    # Diagnostics: learn once, then frozen-memory ablation + capacity probe.
    #   Arms (3B): none / examples / rules / both  — same held-out Qs
    #   Capacity (7B): none / both
    # Override: PRIME_CAPACITY_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    #   N_LEARN=100 N_HELDOUT=30 bash scripts/use_prime_student.sh ablate
    capacity_model="${PRIME_CAPACITY_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

    echo "== 1/3 learn phase (fresh) =="
    python3 orchestrator.py --adapter coding --hard-curriculum --fresh \
      --n-learn "${N_LEARN:-100}" --n-heldout "${N_HELDOUT:-30}" \
      --heldout-frac "${HELDOUT_FRAC:-0.5}"

    echo "== 2/3 ablation arms (student: $AGENT_MODEL) =="
    python3 orchestrator.py --adapter coding --ablation all \
      --n-heldout "${N_HELDOUT:-30}" --heldout-frac "${HELDOUT_FRAC:-0.5}"

    echo "== 3/3 capacity probe (student: $capacity_model) =="
    AGENT_MODEL="$capacity_model" \
      python3 orchestrator.py --adapter coding --ablation none,both \
      --n-heldout "${N_HELDOUT:-30}" --heldout-frac "${HELDOUT_FRAC:-0.5}"
    ;;
```

Update the usage line: `echo "Usage: $0 {list|smoke|probe|degraded|full|compare|full-compare|curriculum|ablate}"`.

Check how `orchestrator.py` reads the student model (`_make_base_config` / `_BASE_MODEL`, orchestrator.py:561): if it reads `AGENT_MODEL` at import time, the inline `AGENT_MODEL=...` override works; if it caches elsewhere, adjust the override accordingly (verify with `grep -n "_BASE_MODEL" orchestrator.py harness/*.py`).

- [ ] **Step 2: Syntax-check**

Run: `bash -n scripts/use_prime_student.sh`
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add scripts/use_prime_student.sh
git commit -m "Add ablate entrypoint: learn once, run ablation arms + capacity probe"
```

---

### Task 8: Live run + results record

Not a code task — the milestone's deliverable is measured numbers.

- [ ] **Step 1: Preflight**

```bash
bash scripts/use_prime_student.sh smoke     # student + teacher endpoints alive
bash scripts/use_prime_student.sh list      # confirm capacity model id exists on Prime
```

If `Qwen/Qwen2.5-Coder-7B-Instruct` is absent from the list output, set `PRIME_CAPACITY_MODEL=meta-llama/Llama-3.1-8B-Instruct`.

- [ ] **Step 2: Run**

```bash
bash scripts/use_prime_student.sh ablate 2>&1 | tee runs/ablate_$(date +%Y%m%d_%H%M).log
```

(`mkdir -p runs` first; gitignore `runs/`.)

- [ ] **Step 3: Record results**

Append a `## Results — <date>` section to `docs/superpowers/specs/2026-07-18-diagnostics-ablation-design.md` with:
- the per-arm table (acc, n, mean_inj, zero_inj%) copied from the report block
- McNemar lines for the three within-run pairs (examples/rules/both vs none)
- the fourth pre-registered pair — cap-none (7B) vs 3B-both — computed by hand from the per-question ✓/✗ lines in the log (the two arms run in separate orchestrator invocations, so the code doesn't pair them automatically; the per-Q lines share `question_id` order)
- which decision-tree row fired (§7 of the spec) and therefore what the next milestone is

Numbers verbatim from the log — nulls included, no rounding up (handoff doc rule).

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-18-diagnostics-ablation-design.md .gitignore
git commit -m "Record ablation results and decision-tree outcome"
```
