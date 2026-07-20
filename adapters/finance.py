"""FinancePro-Bench adapter — rubric-graded free-text reasoning (RSI-Mem v2).

Rubric access policy (docs/RSI_MEM_V2_FINANCE.md §1):
- Student never sees rubrics.
- Teacher may see rubrics for TRAIN-STREAM ids only.
- Validation / held-out rubrics: judge-only.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

from contracts.schemas import AgentConfig, Difficulty, FewShotExample, TelemetryRecord
from correction.learner import FailingCase
from harness import agent
from harness.feed import FeedItem

_ROOT = Path(__file__).resolve().parent.parent
_DATASET = _ROOT / "fixtures" / "finance_pro_bench.json"
_MANIFEST = _ROOT / "fixtures" / "finance_manifest.json"
_log = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert finance professional. Answer the question with clear, "
    "structured reasoning. Cite standards and show steps where relevant. "
    "Do not invent rubric scores — produce the substantive answer only."
)

_BY_ID: dict[str, dict] | None = None
_MANIFEST_CACHE: dict | None = None


def _load_raw() -> list[dict]:
    raw = json.loads(_DATASET.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items" in raw:
        return list(raw["items"])
    if isinstance(raw, list):
        return raw
    raise ValueError(f"unexpected dataset shape in {_DATASET}")


def _index() -> dict[str, dict]:
    global _BY_ID
    if _BY_ID is None:
        _BY_ID = {p["id"]: p for p in _load_raw()}
    return _BY_ID


def load_manifest(path: Path | None = None) -> dict:
    global _MANIFEST_CACHE
    p = path or Path(os.environ.get("FINANCE_MANIFEST", str(_MANIFEST)))
    if _MANIFEST_CACHE is None or path is not None:
        data = json.loads(p.read_text(encoding="utf-8"))
        if path is None:
            _MANIFEST_CACHE = data
        return data
    return _MANIFEST_CACHE


def get_problem(qid: str) -> dict:
    return _index()[qid]


def rubric_for(qid: str) -> str:
    return get_problem(qid)["rubric"]


def split_of(qid: str, manifest: dict | None = None) -> str:
    """Return 'train' | 'validation' | 'heldout'."""
    m = manifest or load_manifest()
    if qid in m["train_ids"]:
        return "train"
    if qid in m["validation_ids"]:
        return "validation"
    if qid in m["heldout_ids"]:
        return "heldout"
    raise KeyError(f"{qid} not in finance manifest")


def assert_rubric_allowed_for_teacher(qid: str, manifest: dict | None = None) -> None:
    """Raise if teacher is not allowed to see this question's rubric."""
    if split_of(qid, manifest) != "train":
        raise PermissionError(
            f"rubric firewall: teacher may not see rubric for {qid} "
            f"(split={split_of(qid, manifest)})"
        )


def build_student_prompt(
    question: str,
    config: AgentConfig,
    category: str,
) -> tuple[str, dict]:
    """Assemble student prompt from question + memory only — never rubric."""
    stats = {
        "examples_available": len(config.few_shot_examples),
        "examples_injected": 0,
        "example_ids": [],
        "rules_injected": 0,
    }
    parts: list[str] = []
    if os.environ.get("AGENT_USE_EXAMPLES", "1") != "0":
        same = [
            e
            for e in config.few_shot_examples
            if not e.domain_id or e.domain_id == category
        ][:3]
        if same:
            lines = ["Worked exemplars:"]
            for i, ex in enumerate(same):
                lines.append(f"Example {i + 1}:\nQ: {ex.question}\nA: {ex.correct_output}")
                stats["example_ids"].append(getattr(ex, "source", f"ex{i}"))
            stats["examples_injected"] = len(same)
            parts.append("\n\n".join(lines) + "\n\n")
    parts.append(question)
    return "".join(parts), stats


def build_teacher_prompt(
    question: str,
    *,
    qid: str,
    rubric: str | None = None,
    broken: str | None = None,
    manifest: dict | None = None,
) -> str:
    """Teacher prompt. Rubric only when qid is train-stream."""
    parts = [question]
    if rubric is not None:
        assert_rubric_allowed_for_teacher(qid, manifest)
        parts.append("\n\n--- OFFICIAL RUBRIC (train-stream only) ---\n")
        parts.append(rubric)
    if broken:
        parts.append("\n\n--- STUDENT ATTEMPT ---\n")
        parts.append(broken)
        parts.append("\n\nProvide a corrected expert answer.")
    return "".join(parts)


def generate_answer(
    question: str,
    config: AgentConfig,
    category: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> tuple[str, dict]:
    prompt, stats = build_student_prompt(question, config, category)
    # Belt-and-suspenders: never allow "Rubric" section markers from dataset.
    client = agent._get_client()
    from adapters.coding import _chat_with_retry

    resp = _chat_with_retry(
        client,
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text, stats


def load_finance_questions(split: str | None = None) -> list[dict]:
    """Map finance items into the shared feed question shape.

    split=None → all; or 'train'/'validation'/'heldout' from manifest.
    """
    m = load_manifest()
    allow: set[str] | None = None
    if split == "train":
        allow = set(m["train_ids"])
    elif split == "validation":
        allow = set(m["validation_ids"])
    elif split == "heldout":
        allow = set(m["heldout_ids"])
    elif split is not None:
        raise ValueError(f"unknown split {split!r}")

    out = []
    for p in _load_raw():
        if allow is not None and p["id"] not in allow:
            continue
        out.append(
            {
                "id": p["id"],
                "question": p["question"],
                # Feed expects expected_sql / db_id keys historically.
                "expected_sql": "",  # no gold free-text in this bench
                "db_id": p["category"],
                "difficulty": "hard",
                "category": p["category"],
                # Rubric stored separately — never copy into feed prompts.
                "_rubric_id": p["id"],
            }
        )
    return out


class FinanceAdapter:
    name = "finance"

    def load_questions(self) -> list[dict]:
        return load_finance_questions()

    def build_feed(self, n: int, full: bool, seed: int) -> list[FeedItem]:
        """Train-stream feed (seeded). Validation/held-out are separate eval paths."""
        import random

        train = load_finance_questions("train")
        rng = random.Random(seed)
        rng.shuffle(train)
        k = len(train) if full else min(n, len(train))
        chosen = train[:k]
        return [
            FeedItem(
                question_id=q["id"],
                question=q["question"],
                gold_output="",
                domain_id=q["db_id"],
                difficulty=q["difficulty"],
                phase="degraded",
            )
            for q in chosen
        ]

    def build_continuous_feed(
        self, n_cycles: int, full: bool, seed: int
    ) -> list[FeedItem]:
        items = self.build_feed(n=80, full=full, seed=seed)
        return items * max(1, n_cycles)

    def run_item(
        self, item: FeedItem, config: AgentConfig, use_rules: bool = True
    ) -> TelemetryRecord | None:
        del use_rules  # KG not wired for finance Phase 0
        from correction.judge import grade

        problem = get_problem(item.question_id)
        answer, stats = generate_answer(
            item.question, config, category=item.domain_id
        )
        # Firewall: student prompt path must not have received rubric.
        student_prompt, _ = build_student_prompt(
            item.question, config, item.domain_id
        )
        if problem["rubric"] and problem["rubric"][:80] in student_prompt:
            raise RuntimeError("rubric firewall violated: rubric leaked into student prompt")

        result = grade(
            question=item.question,
            rubric=problem["rubric"],
            answer=answer,
        )
        normalized = float(result["normalized"])
        return TelemetryRecord(
            run_id=f"{item.question_id}_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            difficulty=Difficulty.HARD,
            execution_accuracy=max(0.0, min(1.0, normalized / 100.0)),
            query_valid=True,
            generated_complexity=0,
            required_complexity=0,
            generated_output=answer,
            gold_output="",
            domain_id=item.domain_id,
            injection_stats=stats,
        )

    def make_examples(
        self,
        failing_cases: list[FailingCase],
        anchor_cases: list[FailingCase],
    ) -> list[FewShotExample]:
        """Phase 0 stub — memory write path arrives in Phase 2."""
        del failing_cases, anchor_cases
        return []
