"""GSM8K grade-school math adapter."""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from contracts.schemas import AgentConfig, Difficulty, FewShotExample, TelemetryRecord
from correction.learner import FailingCase
from harness import agent
from harness.feed import FeedItem, build_continuous_stream, build_stream

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "gsm8k_subset.json"

_SYSTEM = (
    "You are a careful math tutor. Solve the word problem step by step, "
    "then give the final numeric answer on its own line as: #### <number>"
)

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


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


def _examples_block(examples: list[FewShotExample], topic: str) -> str:
    same = [e for e in examples if not e.db_id or e.db_id == topic][:8]
    if not same:
        return ""
    lines = ["Similar solved problems:"]
    for ex in same:
        lines.append(f"Q: {ex.question}\nA: #### {ex.correct_sql}")
    return "\n".join(lines) + "\n\n"


def generate_math(
    question: str, config: AgentConfig, topic: str = "math"
) -> tuple[str, int, float, str]:
    """Call the shared OpenAI-compatible client (Ollama or MiniMax)."""
    client = agent._get_client()
    user = _examples_block(config.few_shot_examples, topic) + f"Q: {question}\nA:"
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = (resp.choices[0].message.content or "").strip()
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens, latency_ms, ""


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

    def run_item(
        self, item: FeedItem, config: AgentConfig, use_rules: bool = True
    ) -> TelemetryRecord | None:
        del use_rules  # KG rules are SQL-specific
        text, tokens, latency_ms, reasoning = generate_math(
            item.question, config, topic=item.db_id
        )
        acc = verify_answer(text, item.gold_sql)
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
            generated_sql=text,
            db_id=item.db_id,
            config_id=config.config_id,
            reasoning=reasoning,
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
                correct_sql=case.gold_sql,
                db_id=case.db_id,
                source="gold",
            ))
        for case in anchor_cases:
            examples.append(FewShotExample(
                question=case.question,
                correct_sql=case.gold_sql,
                db_id=case.db_id,
                source="anchor",
            ))
        return examples
