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
    p = Path(path) if path is not None else Path(
        os.environ.get("FINANCE_MANIFEST", str(_MANIFEST))
    )
    p = p.resolve()
    if _MANIFEST_CACHE is not None and _MANIFEST_CACHE.get("_path") == str(p):
        return _MANIFEST_CACHE["data"]
    data = json.loads(p.read_text(encoding="utf-8"))
    _MANIFEST_CACHE = {"_path": str(p), "data": data}
    return data


def get_problem(qid: str) -> dict:
    return _index()[qid]


def rubric_for(
    qid: str, *, role: str = "judge", manifest: dict | None = None
) -> str:
    """Return rubric text with role ACL: student never; teacher train-only; judge any."""
    if role == "student":
        raise PermissionError("rubric firewall: student may never read rubrics")
    if role == "teacher":
        assert_rubric_allowed_for_teacher(qid, manifest)
    elif role != "judge":
        raise ValueError(f"unknown rubric role {role!r}")
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
    *,
    forbidden_rubric_stems: list[str] | None = None,
) -> tuple[str, dict]:
    """Assemble student prompt from question + memory only — never rubric."""
    stats = {
        "examples_available": len(config.few_shot_examples),
        "examples_injected": 0,
        "example_ids": [],
        "rules_injected": 0,
    }
    parts: list[str] = []
    stems = forbidden_rubric_stems or []
    if os.environ.get("AGENT_USE_EXAMPLES", "1") != "0":
        same = [
            e
            for e in config.few_shot_examples
            if not e.domain_id or e.domain_id == category
        ][:3]
        if same:
            lines = ["Worked exemplars:"]
            for i, ex in enumerate(same):
                blob = f"{ex.question}\n{ex.correct_output}"
                _assert_no_rubric_smuggle(blob, stems, where="few-shot example")
                lines.append(f"Example {i + 1}:\nQ: {ex.question}\nA: {ex.correct_output}")
                stats["example_ids"].append(getattr(ex, "source", f"ex{i}"))
            stats["examples_injected"] = len(same)
            parts.append("\n\n".join(lines) + "\n\n")
    parts.append(question)
    prompt = "".join(parts)
    _assert_no_rubric_smuggle(prompt, stems, where="student prompt")
    return prompt, stats


def _assert_no_rubric_smuggle(
    text: str, stems: list[str], *, where: str
) -> None:
    """Reject text that carries known non-train rubric content or rubric markers."""
    if "OFFICIAL RUBRIC" in text:
        raise PermissionError(f"rubric firewall: OFFICIAL RUBRIC marker in {where}")
    for stem in stems:
        if stem and stem in text:
            raise PermissionError(
                f"rubric firewall: non-train rubric stem leaked into {where}"
            )


def all_rubric_stems(n: int = 120) -> list[str]:
    """Stable prefixes of ALL rubrics — student must never see any of them."""
    stems: list[str] = []
    for p in _load_raw():
        stem = p["rubric"].strip()[:n]
        if len(stem) >= 40:
            stems.append(stem)
    return stems


def non_train_rubric_stems(manifest: dict | None = None, n: int = 120) -> list[str]:
    """Deprecated alias — student firewall uses all stems."""
    del manifest
    return all_rubric_stems(n=n)


def build_teacher_prompt(
    question: str,
    *,
    qid: str,
    rubric: str | None = None,
    broken: str | None = None,
    manifest: dict | None = None,
) -> str:
    """Teacher prompt. Rubric only when qid is train-stream.

    Rubric text is always resolved via ``rubric_for(qid, role='teacher')``.
    If ``rubric`` is passed, it must exactly match the fixture text.
    """
    parts = [question]
    if rubric is not None:
        # ACL + identity check (rejects held-out text under a train qid).
        expected = rubric_for(qid, role="teacher", manifest=manifest)
        if rubric != expected:
            raise PermissionError(
                f"rubric firewall: rubric text does not match fixture for {qid}"
            )
        parts.append("\n\n--- OFFICIAL RUBRIC (train-stream only) ---\n")
        parts.append(expected)
    if broken:
        # Refuse pasted rubric markers even when rubric=None.
        if "OFFICIAL RUBRIC" in broken:
            raise PermissionError(
                "rubric firewall: student attempt appears to contain rubric text"
            )
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
    forbidden_rubric_stems: list[str] | None = None,
) -> tuple[str, dict]:
    prompt, stats = build_student_prompt(
        question,
        config,
        category,
        forbidden_rubric_stems=forbidden_rubric_stems,
    )
    client = agent._get_client()
    from adapters.coding import _chat_with_retry

    # Thinking SKUs (e.g. qwen3.6-27b) can return content=None and spend the
    # entire max_tokens budget on reasoning. Default: disable thinking so bare
    # student answers are comparable across candidates. Opt-in via env.
    kwargs: dict = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": int(os.environ.get("AGENT_MAX_RETRIES", "5")),
    }
    if os.environ.get("AGENT_ENABLE_THINKING", "").strip() not in ("1", "true", "yes"):
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _call(kw: dict):
        return _chat_with_retry(client, **kw)

    resp = _call(kwargs)
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        reasoning = getattr(msg, "reasoning", None) or (
            (msg.model_extra or {}).get("reasoning") if hasattr(msg, "model_extra") else None
        )
        if reasoning:
            _log.warning(
                "empty content from %s (reasoning_len=%d); retrying with max_tokens=8192",
                config.model,
                len(str(reasoning)),
            )
            # Thinking models often exhaust a 2048 budget on reasoning alone.
            for bump in (8192, 16384):
                kw2 = dict(kwargs)
                kw2["max_tokens"] = bump
                # Prefer bare request — chat_template_kwargs can leave some
                # providers in reasoning-only mode with empty content.
                kw2.pop("extra_body", None)
                _log.warning(
                    "empty content from %s; retry max_tokens=%d no extra_body",
                    config.model,
                    bump,
                )
                resp2 = _call(kw2)
                text = (resp2.choices[0].message.content or "").strip()
                if text:
                    break
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
        # Default train-only so undifferentiated feeds never mix held-out.
        return load_finance_questions("train")

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

        stems = all_rubric_stems()
        # Firewall before any LLM call.
        build_student_prompt(
            item.question,
            config,
            item.domain_id,
            forbidden_rubric_stems=stems,
        )
        answer, stats = generate_answer(
            item.question,
            config,
            category=item.domain_id,
            forbidden_rubric_stems=stems,
        )

        result = grade(
            question=item.question,
            rubric=rubric_for(item.question_id, role="judge"),
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
