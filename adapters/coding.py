"""Hard coding problems adapter — unit-test verified Python solutions."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from contracts.schemas import (
    AgentConfig,
    Difficulty,
    DriftEvent,
    FewShotExample,
    TelemetryRecord,
)
from correction.learner import FailingCase
from harness import agent
from harness.feed import (
    FeedItem,
    build_continuous_stream,
    build_hard_curriculum_stream,
    build_stream,
)
from harness.sandbox import extract_python_code, execution_accuracy

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "coding_subset.json"
_log = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert Python programmer. Solve the problem by writing a single "
    "correct Python function. Return ONLY a python markdown code block with the "
    "function — no explanation, no tests, no extra prose."
)

_BY_ID: dict[str, dict] | None = None


def _index() -> dict[str, dict]:
    global _BY_ID
    if _BY_ID is None:
        raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        _BY_ID = {p["id"]: p for p in raw}
    return _BY_ID


def load_coding_questions() -> list[dict]:
    """Map coding problems into the feed's shared question shape."""
    out = []
    for p in _index().values():
        out.append({
            "id": p["id"],
            "question": (
                f"{p['question']}\n\n"
                f"Implement exactly: `def {p['function_name']}(...):`\n"
                "Return a ```python``` code block containing only that function."
            ),
            "expected_sql": p["gold_solution"],
            "db_id": p["topic"],
            "difficulty": p["difficulty"],
        })
    return out


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def _use_examples() -> bool:
    return os.environ.get("AGENT_USE_EXAMPLES", "1") != "0"


def _rules_block(topic: str, question: str) -> tuple[str, int]:
    """Runtime KG injection — same store as Spider, keyed by topic (db_id).

    Returns (prompt_block, n_rules).
    """
    if not topic:
        return "", 0
    try:
        from correction.inject import build_context, format_prompt_block

        ctx = build_context(topic, question)
        block = format_prompt_block(ctx)
        return block, len(ctx.injected_rules)
    except Exception:
        return "", 0


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
        rules, n_rules = _rules_block(topic, question)
        if rules:
            parts.append(rules + "\n\n")
            stats["rules_injected"] = n_rules

    parts.append(f"Problem:\n{question}\n")
    return "".join(parts), stats


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


def verify_solution(output: str, problem: dict) -> tuple[float, bool, str]:
    code = extract_python_code(output)
    if not code.strip():
        return 0.0, False, "empty code"
    return execution_accuracy(code, problem["function_name"], problem["tests"])


def _teacher_chat(system: str, user: str) -> str | None:
    """Shared teacher call; returns extracted Python code or None."""
    try:
        from correction.provider import teacher_client_and_model

        client, model = teacher_client_and_model()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = _strip_think(resp.choices[0].message.content or "")
        return extract_python_code(raw) or None
    except Exception as exc:
        _log.warning("coding teacher call failed: %s", exc)
        print(f"  [teacher] call failed: {exc}", flush=True)
        return None


def _teacher_repair(problem: dict, broken: str) -> str | None:
    """Ask the teacher model for a repaired function; return code or None."""
    prompt = (
        f"{problem['question']}\n\n"
        f"Implement exactly: `def {problem['function_name']}(...):`\n"
        "Broken attempt:\n"
        f"```python\n{broken}\n```\n"
        "Return a corrected ```python``` code block only."
    )
    return _teacher_chat(
        "You are an expert Python programmer. Fix the broken function. "
        "Return ONLY a python markdown code block.",
        prompt,
    )


def teacher_solve(problem: dict) -> str | None:
    """Teacher solves a problem from scratch (no student attempt, no few-shots)."""
    prompt = (
        f"{problem['question']}\n\n"
        f"Implement exactly: `def {problem['function_name']}(...):`\n"
        "Return a ```python``` code block containing only that function."
    )
    return _teacher_chat(_SYSTEM, prompt)


def write_graph_rules(
    event: DriftEvent,
    failing_cases: list[FailingCase],
    max_cases: int = 3,
) -> int:
    """Distill (trap, fix) rules from coding failures into the shared KG store."""
    del event  # severity already gated by detector
    from correction.contracts import FailedRun
    from correction.distill import distill_code
    from correction.graph import add_rule, maybe_promote

    idx = _index()
    n_written = 0
    for case in failing_cases[:max_cases]:
        try:
            qid = case.run_id.rsplit("_", 1)[0]
            problem = idx.get(qid)
            fn = problem["function_name"] if problem else case.domain_id
            # Gold is always unit-test correct; teacher already ran in make_examples.
            fixed = case.gold_output

            failed = FailedRun(
                run_id=case.run_id,
                domain_id=case.domain_id,
                question=case.question,
                broken_output=case.broken_output,
                schema={case.domain_id: [fn, "algorithm", "edge_cases"]},
            )
            rule = distill_code(failed, fixed)
            if fn and fn.lower() not in rule.trigger.lower():
                rule.trigger = fn
            rule.applies_to = list({*rule.applies_to, f"schema:{case.domain_id}:{fn}"})
            rule.seen_dbs = [case.domain_id]
            add_rule(rule)
            maybe_promote(rule)
            n_written += 1
            print(f"  [graph] rule {rule.id}: {rule.trap[:60]}", flush=True)
        except Exception as exc:
            print(f"  [graph] skipped {case.run_id}: {exc}", flush=True)
    return n_written


class CodingAdapter:
    name = "coding"

    def load_questions(self) -> list[dict]:
        return load_coding_questions()

    def build_feed(self, n: int, full: bool, seed: int) -> list[FeedItem]:
        per_phase = 80 if full else n
        return build_stream(
            self.load_questions(),
            n_baseline=per_phase,
            n_degraded=per_phase,
            n_recovery=per_phase,
            seed=seed,
            same_db_split=True,
            baseline_easy_only=True,
        )

    def build_continuous_feed(
        self, n_cycles: int, full: bool, seed: int
    ) -> list[FeedItem]:
        per_phase = 50 if full else 30
        n_baseline = 80 if full else 40
        return build_continuous_stream(
            self.load_questions(),
            n_baseline=n_baseline,
            n_degraded=per_phase,
            n_recovery=per_phase,
            n_cycles=n_cycles,
            seed=seed,
            same_db_split=True,
            baseline_easy_only=True,
        )

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

    def run_item(
        self, item: FeedItem, config: AgentConfig, use_rules: bool = True
    ) -> TelemetryRecord | None:
        problem = _index().get(item.question_id)
        if problem is None:
            return None

        text, tokens, latency_ms, reasoning, stats = generate_code(
            item.question, config, topic=item.domain_id, use_rules=use_rules
        )
        code = extract_python_code(text)
        acc, valid, _err = verify_solution(text, problem)
        return TelemetryRecord(
            run_id=f"{item.question_id}_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            difficulty=Difficulty(item.difficulty),
            execution_accuracy=acc,
            query_valid=valid,
            generated_complexity=code.count("for ") + code.count("while "),
            required_complexity=0,
            latency_ms=latency_ms,
            tokens=tokens,
            question=item.question,
            generated_output=code or text,
            db_id=item.domain_id,
            config_id=config.config_id,
            reasoning=reasoning,
            injection_stats=stats,
        )

    def make_examples(
        self,
        failing_cases: list[FailingCase],
        anchor_cases: list[FailingCase],
    ) -> list[FewShotExample]:
        """Teacher-repair when possible; always fall back to gold solution."""
        examples: list[FewShotExample] = []
        idx = _index()

        for case in failing_cases:
            qid = case.run_id.rsplit("_", 1)[0]
            problem_rec = idx.get(qid)
            gold = case.gold_output
            source = "gold"

            if problem_rec is not None:
                teacher_code = _teacher_repair(problem_rec, case.broken_output)
                if teacher_code:
                    acc, _, _ = execution_accuracy(
                        teacher_code,
                        problem_rec["function_name"],
                        problem_rec["tests"],
                    )
                    if acc == 1.0:
                        gold = teacher_code
                        source = "teacher"

            examples.append(FewShotExample(
                question=case.question,
                correct_output=gold,
                domain_id=case.domain_id,
                source=source,
            ))

        for case in anchor_cases:
            examples.append(FewShotExample(
                question=case.question,
                correct_output=case.gold_output,
                domain_id=case.domain_id,
                source="anchor",
            ))
        return examples
