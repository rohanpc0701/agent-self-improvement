"""GSM8K grade-school math adapter."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

from contracts.schemas import AgentConfig, Difficulty, FewShotExample, TelemetryRecord
from correction.learner import FailingCase
from harness import agent
from harness.feed import (
    FeedItem,
    build_continuous_stream,
    build_hard_curriculum_stream,
    build_stream,
)

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "gsm8k_subset.json"

_SYSTEM = (
    "You are a careful math tutor. Solve the word problem step by step, "
    "then give the final numeric answer on its own line as: #### <number>"
)

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_PROMPT_CAP = 5


def load_gsm8k_questions() -> list[dict]:
    raw = json.loads(_FIXTURE.read_text())
    out = []
    for q in raw:
        out.append({
            "id": q["id"],
            "question": q["question"],
            "expected_sql": str(q["answer"]).replace(",", ""),
            "db_id": q.get("topic", "math"),
            "difficulty": q["difficulty"],
        })
    return out


def extract_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def _normalize_num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def verify_answer(output: str, gold: str) -> float | None:
    pred = extract_answer(output)
    if pred is None:
        return 0.0
    g = _normalize_num(gold)
    p = _normalize_num(pred)
    if g is None or p is None:
        return 0.0
    return 1.0 if abs(g - p) < 1e-6 else 0.0


def _use_examples() -> bool:
    return os.environ.get("AGENT_USE_EXAMPLES", "1") != "0"


def build_user_prompt(
    question: str,
    config: AgentConfig,
    topic: str,
) -> tuple[str, dict]:
    """Assemble student prompt; report what memory entered it."""
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
        ][:_PROMPT_CAP]
        if same:
            lines = ["Similar solved problems:"]
            for ex in same:
                lines.append(f"Q: {ex.question}\nA: #### {ex.correct_output}")
            parts.append("\n".join(lines) + "\n\n")
            stats["examples_injected"] = len(same)
            stats["example_ids"] = [ex.question for ex in same]
    parts.append(f"Q: {question}\nA:")
    return "".join(parts), stats


def generate_math(
    question: str,
    config: AgentConfig,
    topic: str = "math",
    temperature: float = 0.0,
) -> tuple[str, int, float, str, dict]:
    """Call the shared OpenAI-compatible client."""
    client = agent._get_client()
    user, stats = build_user_prompt(question, config, topic)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=512,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = (resp.choices[0].message.content or "").strip()
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens, latency_ms, "", stats


def teacher_solve(question: str) -> str | None:
    """Unaided teacher numeric answer (#### format preferred)."""
    from correction.provider import teacher_client_and_model

    client, model = teacher_client_and_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Q: {question}\nA:"},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    text = (resp.choices[0].message.content or "").strip()
    return extract_answer(text)


class GSM8KMathAdapter:
    name = "gsm8k"

    def load_questions(self) -> list[dict]:
        return load_gsm8k_questions()

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
        del use_rules  # KG rules are SQL/coding-specific
        text, tokens, latency_ms, reasoning, stats = generate_math(
            item.question, config, topic=item.domain_id
        )
        acc = verify_answer(text, item.gold_output)
        if acc is None:
            return None
        valid = extract_answer(text) is not None
        return TelemetryRecord(
            run_id=f"{item.question_id}_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            difficulty=Difficulty(item.difficulty),
            execution_accuracy=acc,
            query_valid=valid,
            generated_complexity=0,
            required_complexity=0,
            latency_ms=latency_ms,
            tokens=tokens,
            question=item.question,
            generated_output=text,
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
        examples: list[FewShotExample] = []
        for case in failing_cases:
            examples.append(FewShotExample(
                question=case.question,
                correct_output=case.gold_output,
                domain_id=case.domain_id,
                source="gold",
            ))
        for case in anchor_cases:
            examples.append(FewShotExample(
                question=case.question,
                correct_output=case.gold_output,
                domain_id=case.domain_id,
                source="anchor",
            ))
        return examples
