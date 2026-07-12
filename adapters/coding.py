"""Hard coding problems adapter — unit-test verified Python solutions."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from openai import OpenAI

from contracts.schemas import AgentConfig, Difficulty, FewShotExample, TelemetryRecord
from correction.learner import FailingCase
from harness import agent
from harness.feed import FeedItem, build_continuous_stream, build_stream
from harness.sandbox import extract_python_code, execution_accuracy

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "coding_subset.json"
_MINIMAX_BASE_URL = "https://api.minimax.io/v1"

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


def _examples_block(examples: list[FewShotExample], topic: str) -> str:
    same = [e for e in examples if not e.db_id or e.db_id == topic][:3]
    if not same:
        return ""
    lines = ["Similar solved problems:"]
    for ex in same:
        lines.append(f"Problem: {ex.question}\n```python\n{ex.correct_sql}\n```")
    return "\n\n".join(lines) + "\n\n"


def generate_code(
    question: str, config: AgentConfig, topic: str = "arrays"
) -> tuple[str, int, float, str]:
    client = agent._get_client()
    user = _examples_block(config.few_shot_examples, topic) + f"Problem:\n{question}\n"
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = (resp.choices[0].message.content or "").strip()
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens, latency_ms, ""


def verify_solution(output: str, problem: dict) -> tuple[float, bool, str]:
    code = extract_python_code(output)
    if not code.strip():
        return 0.0, False, "empty code"
    return execution_accuracy(code, problem["function_name"], problem["tests"])


def _teacher_repair(problem: dict, broken: str) -> str | None:
    """Ask the teacher model for a repaired function; return code or None."""
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        return None
    try:
        model = os.environ.get("TEACHER_MODEL", "MiniMax-M3")
        client = OpenAI(api_key=key, base_url=_MINIMAX_BASE_URL)
        prompt = (
            f"{problem['question']}\n\n"
            f"Implement exactly: `def {problem['function_name']}(...):`\n"
            "Broken attempt:\n"
            f"```python\n{broken}\n```\n"
            "Return a corrected ```python``` code block only."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python programmer. Fix the broken function. "
                        "Return ONLY a python markdown code block."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return extract_python_code(resp.choices[0].message.content or "")
    except Exception:
        return None


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

    def run_item(
        self, item: FeedItem, config: AgentConfig, use_rules: bool = True
    ) -> TelemetryRecord | None:
        del use_rules  # KG rules are SQL-specific
        problem = _index().get(item.question_id)
        if problem is None:
            return None

        text, tokens, latency_ms, reasoning = generate_code(
            item.question, config, topic=item.db_id
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
            generated_sql=code or text,
            db_id=item.db_id,
            config_id=config.config_id,
            reasoning=reasoning,
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
            gold = case.gold_sql
            source = "gold"

            if problem_rec is not None:
                teacher_code = _teacher_repair(problem_rec, case.broken_sql)
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
                correct_sql=gold,
                db_id=case.db_id,
                source=source,
            ))

        for case in anchor_cases:
            examples.append(FewShotExample(
                question=case.question,
                correct_sql=case.gold_sql,
                db_id=case.db_id,
                source="anchor",
            ))
        return examples
