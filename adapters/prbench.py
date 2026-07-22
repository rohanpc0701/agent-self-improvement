"""PRBench Corporate-Finance adapter (single-turn) for the teacher/student loop.

Reuses the generic finance student-generation + teacher-distillation; scores with
the PRBench weighted-criteria judge. Rubric firewall: student never sees rubrics;
teacher sees rubrics only for TRAIN-stream tasks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from contracts.schemas import FewShotExample

_ROOT = Path(__file__).resolve().parent.parent
_DATASET = _ROOT / "fixtures" / "prbench_corpfin.json"
_MANIFEST = _ROOT / "fixtures" / "prbench_corpfin_manifest.json"

_TASKS: dict[str, dict] | None = None
_MANIFEST_CACHE: dict | None = None

_TOPIC = "Corporate Finance"


def _load_raw() -> list[dict]:
    return json.loads(_DATASET.read_text(encoding="utf-8"))["items"]


def _index() -> dict[str, dict]:
    global _TASKS
    if _TASKS is None:
        _TASKS = {t["id"]: t for t in _load_raw()}
    return _TASKS


def get_task(tid: str) -> dict:
    return _index()[tid]


def load_manifest(path: Path | None = None) -> dict:
    global _MANIFEST_CACHE
    p = Path(path) if path else Path(os.environ.get("PRBENCH_MANIFEST", str(_MANIFEST)))
    if _MANIFEST_CACHE is not None and _MANIFEST_CACHE.get("_path") == str(p):
        return _MANIFEST_CACHE["data"]
    data = json.loads(p.read_text(encoding="utf-8"))
    _MANIFEST_CACHE = {"_path": str(p), "data": data}
    return data


def split_of(tid: str, manifest: dict | None = None) -> str:
    m = manifest or load_manifest()
    for split in ("train_ids", "validation_ids", "heldout_ids"):
        if tid in m[split]:
            return split.replace("_ids", "")
    raise KeyError(f"{tid} not in PRBench manifest")


def rubric_for(tid: str, *, role: str, manifest: dict | None = None) -> list[dict]:
    """Rubric ACL: student never; teacher train-only; judge any."""
    if role == "student":
        raise PermissionError("rubric firewall: student may never read rubrics")
    if role == "teacher" and split_of(tid, manifest) != "train":
        raise PermissionError(
            f"rubric firewall: teacher may not see rubric for {tid} "
            f"(split={split_of(tid, manifest)})"
        )
    if role not in ("teacher", "judge"):
        raise ValueError(f"unknown rubric role {role!r}")
    return get_task(tid)["rubric"]


def score_answer(tid: str, answer: str, model: str | None = None) -> dict:
    """Judge the answer against this task's PRBench rubric → normalized 0-100."""
    from correction.prbench_judge import grade

    t = get_task(tid)
    return grade(t["question"], t["rubric"], answer, model=model)


_SYSTEM = (
    "You are an expert corporate finance professional. Answer the final user turn "
    "with clear, rigorous, structured reasoning. Cite standards and show steps where "
    "relevant. Produce the substantive answer only."
)


def _turns_of(task_or_question) -> list[dict]:
    if isinstance(task_or_question, dict):
        return task_or_question.get("turns") or [
            {"role": "user", "content": task_or_question["question"]}
        ]
    return [{"role": "user", "content": str(task_or_question)}]


def _student_messages(turns: list[dict], memory: list[FewShotExample], hints: str = "") -> list[dict]:
    """system (+ optional memory, + optional teacher hints) then the conversation."""
    from adapters.finance import select_category_memory, memory_kind_of

    system = _SYSTEM
    if memory:
        sel = select_category_memory(memory, _TOPIC)
        if sel:
            blocks = ["Category memory (TraceLift):"]
            for i, ex in enumerate(sel, 1):
                blocks.append(f"Memory {i} [{memory_kind_of(ex) or 'other'}]:\n"
                              f"{ex.question}\n{ex.correct_output}")
            system = system + "\n\n" + "\n\n".join(blocks)
    if hints:
        system = (system + "\n\nApproach guidance from a senior expert (follow it; "
                  "it is guidance only — you must do the actual reasoning and math):\n" + hints)
    return [{"role": "system", "content": system}, *turns]


def _student_call(messages: list[dict], model: str, max_tokens: int | None = None):
    from harness import agent
    from harness.agent import _chat_with_retry

    client = agent._get_client()
    mt = max_tokens or int(os.environ.get("STUDENT_MAX_TOKENS", "6000"))
    extra = {}
    if os.environ.get("AGENT_ENABLE_THINKING", "").strip() not in ("1", "true", "yes"):
        extra = {"reasoning": {"enabled": False}}
    resp = _chat_with_retry(client, model=model, messages=messages,
                            temperature=0.0, max_tokens=mt,
                            **({"extra_body": extra} if extra else {}))
    return (resp.choices[0].message.content or "").strip()


def generate_answer(task_or_question, config, memory: list[FewShotExample] | None = None,
                    hints: str = ""):
    """Student answers the final turn (A1 with no memory/hints; A4 with hints)."""
    turns = _turns_of(task_or_question)
    mem = memory if memory is not None else list(config.few_shot_examples)
    messages = _student_messages(turns, mem, hints=hints)
    stats = {"examples_injected": len([m for m in mem if not m.domain_id or m.domain_id == _TOPIC][:4]),
             "n_turns": sum(1 for t in turns if t["role"] == "user"),
             "hint_chars": len(hints)}
    return _student_call(messages, config.model), stats


# ── Planner–executor arms ────────────────────────────────────────────────────

_HINT_SYSTEM = (
    "You are a senior corporate-finance expert coaching a junior analyst. Reason "
    "through the problem yourself, then give ONLY short approach guidance the junior "
    "should follow: the framework to apply, the ordered steps, and the traps to avoid. "
    "HARD RULES: <=300 tokens. Do NOT state the final answer, specific numeric results, "
    "or conclusions — guidance on HOW to approach only, so the junior does the actual work."
)


def teacher_hints(task_or_question, *, client=None, model: str | None = None,
                  manifest: dict | None = None) -> str:
    """A4 teacher step: Fable reasons, then emits <=300-token guidance (no answer).

    Rubric-blind: the teacher never sees the rubric here (held-out firewall holds
    even for train, since hints are approach-only).
    """
    from correction.provider import teacher_client_and_model
    from harness.agent import _chat_with_retry

    turns = _turns_of(task_or_question)
    convo = "\n\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
    if client is None:
        client, resolved = teacher_client_and_model()
    else:
        resolved = model or os.environ.get("TEACHER_MODEL") or "anthropic/claude-fable-5"
    resp = _chat_with_retry(
        client, model=resolved,
        messages=[{"role": "system", "content": _HINT_SYSTEM},
                  {"role": "user", "content": f"Problem (answer the final turn):\n{convo}"}],
        temperature=0.0, max_tokens=700,
    )
    return (resp.choices[0].message.content or "").strip()


def answer_teacher_alone(task_or_question, *, client=None, model: str | None = None) -> str:
    """A5 ceiling: the teacher solves the task directly (full answer)."""
    from correction.provider import teacher_client_and_model
    from harness.agent import _chat_with_retry

    turns = _turns_of(task_or_question)
    if client is None:
        client, resolved = teacher_client_and_model()
    else:
        resolved = model or os.environ.get("TEACHER_MODEL") or "anthropic/claude-fable-5"
    resp = _chat_with_retry(
        client, model=resolved,
        messages=[{"role": "system", "content": _SYSTEM}, *turns],
        temperature=0.0, max_tokens=int(os.environ.get("TEACHER_MAX_TOKENS", "4000")),
    )
    return (resp.choices[0].message.content or "").strip()


_CRITIQUE = (
    "Critique your own answer above against what an expert grader would want: what's "
    "missing, wrong, or shallow? List the gaps, then STOP (do not rewrite yet)."
)
_REVISE = "Now produce an improved final answer addressing every gap you listed."


def answer_with_retries(task_or_question, config) -> tuple[str, dict]:
    """A2 compute-matched control: student answers, self-critiques, revises. No teacher.

    Spends extra compute the 'dumb' way (its own reasoning) to match A4's hint overhead —
    isolates 'teacher content' from 'more inference compute'.
    """
    turns = _turns_of(task_or_question)
    base_msgs = _student_messages(turns, [])
    a0 = _student_call(base_msgs, config.model)
    crit_msgs = base_msgs + [{"role": "assistant", "content": a0},
                             {"role": "user", "content": _CRITIQUE}]
    crit = _student_call(crit_msgs, config.model, max_tokens=1500)
    rev_msgs = crit_msgs + [{"role": "assistant", "content": crit},
                            {"role": "user", "content": _REVISE}]
    final = _student_call(rev_msgs, config.model)
    return final, {"passes": 3, "critique_chars": len(crit)}


def teacher_repair(tid: str, student_answer: str, *, client=None, model: str | None = None,
                   manifest: dict | None = None) -> str:
    """Teacher rewrites the answer to satisfy the (train-only) rubric criteria."""
    from correction.provider import teacher_client_and_model
    from harness.agent import _chat_with_retry

    t = get_task(tid)
    rubric = rubric_for(tid, role="teacher", manifest=manifest)  # ACL: train only
    crit_lines = "\n".join(
        f"- {'AVOID: ' if c['weight'] < 0 else ''}{c['description']}" for c in rubric
    )
    if client is None:
        client, resolved = teacher_client_and_model()
    else:
        resolved = model or os.environ.get("TEACHER_MODEL") or ""
    prompt = (
        f"Corporate finance question:\n{t['question']}\n\n"
        f"A weaker model answered:\n{student_answer[:2000]}\n\n"
        "Rewrite a correct, complete expert answer that satisfies ALL of these "
        f"graded criteria (and avoids the AVOID ones):\n{crit_lines}\n\n"
        "Return only the improved answer."
    )
    resp = _chat_with_retry(
        client, model=resolved,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=int(os.environ.get("TEACHER_MAX_TOKENS", "4000")),
    )
    return (resp.choices[0].message.content or "").strip()


def distill_memory_item(tid: str, repaired: str, *, kind: str = "playbook",
                        client=None, model: str | None = None):
    """Distill the repair into a transferable Corporate-Finance memory item.

    Reuses the finance teacher-distillation (category-generic, leak-safe).
    """
    from adapters.finance import _teacher_distill

    if client is None:
        from correction.provider import teacher_client_and_model
        client, _ = teacher_client_and_model()
    body = _teacher_distill(_TOPIC, repaired, kind, client=client, model=model)
    prefix = {"playbook": "[FINANCE_PLAYBOOK]", "trap": "[FINANCE_TRAP]",
              "skeleton": "[FINANCE_SKELETON]"}[kind]
    return FewShotExample(
        question=f"{prefix} {_TOPIC}",
        correct_output=body or "",
        domain_id=_TOPIC,
        source="tracelift",
    )
